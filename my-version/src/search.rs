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
use crate::expr::{self, crossover, eval, mutate, random_tree, Node, Op};
use crate::metrics::{score_weighted, Metric};
use crate::pareto::{Archive, Sol};
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
}

/// A per-generation progress report.
#[derive(Clone, Debug)]
pub struct Progress {
    pub generation: usize,
    pub total_generations: usize,
    pub best_error: f64,
    pub pareto_members: usize,
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
    fn evaluated(tree: Node, train: &Sample, cfg: &Config) -> Self {
        let tree = expr::simplify(&tree);
        let complexity = cfg.complexity_model.complexity(&tree);
        // #25: a dimensionally inconsistent candidate is rejected with an
        // infinite error, not merely penalised. The predicate is a no-op when
        // no units are declared or they are not required.
        let error = if cfg.accepts_dimensions(&tree) {
            eval_error(&tree, train, cfg.metric)
        } else {
            f64::INFINITY
        };
        Ind {
            tree,
            error,
            complexity,
        }
    }

    fn reeval(&mut self, train: &Sample, cfg: &Config) {
        self.tree = expr::simplify(&self.tree);
        self.complexity = cfg.complexity_model.complexity(&self.tree);
        // #25: re-evaluation must not resurrect a rejected candidate.
        self.error = if cfg.accepts_dimensions(&self.tree) {
            eval_error(&self.tree, train, cfg.metric)
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

pub fn eval_error(tree: &Node, s: &Sample, metric: Metric) -> f64 {
    if s.is_empty() {
        return f64::INFINITY;
    }
    let pred: Vec<f64> = s.x.iter().map(|row| eval(tree, row)).collect();
    // The parameter count lets the information criteria (AIC/AICc/BIC) work as
    // search metrics; it is ignored by every other metric. The weight vector is
    // empty for an unweighted sample, which takes the identical unweighted path.
    score_weighted(&pred, &s.y, &s.w, metric, expr::count_constants(tree))
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
    })
}

/// Rebuild an archive from its members and the final populations, scoring every
/// expression on the **full** training sample. Used only when the generations
/// ran on subsets, so the returned front and every error derived from it are
/// full-data even though fitness was estimated.
fn full_data_archive(
    archive: &Archive,
    islands: &[Vec<Ind>],
    train: &Sample,
    cfg: &Config, // #25: the rebuild must apply the same dimensional filter
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
        // #25: a rejected candidate must not sneak back in on the full-data
        // rebuild; infinite error keeps it out of the archive.
        let error = if cfg.accepts_dimensions(&tree) {
            eval_error(&tree, train, cfg.metric)
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

/// Evolve one island by a single generation. Pure with respect to the rest of
/// the world, so it can run on its own thread.
fn evolve_island(isl: &mut Vec<Ind>, rng: &mut Mt, cfg: &Config, train: &Sample, nvars: usize) {
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
    next.iter().for_each(|i| {
        seen.insert(expr::structural_hash(&i.tree));
    });
    let mut attempts = 0usize;
    let max_attempts = pop * 50 + 50;

    while next.len() < pop && attempts < max_attempts {
        attempts += 1;
        let p1 = tournament(isl, rng, 4);
        let mut child = isl[p1].tree.clone();
        if rng.bool(0.5) {
            let p2 = tournament(isl, rng, 4);
            let parent2 = isl[p2].tree.clone();
            crossover(&mut child, &parent2, rng);
        }
        mutate(&mut child, rng, nvars, &cfg.ops);
        if rng.bool(0.5) {
            mutate(&mut child, rng, nvars, &cfg.ops);
        }

        let mut tree = expr::simplify(&child);
        if cfg.complexity_model.complexity(&tree) > cfg.max_complexity as f64 {
            continue;
        }
        if cfg.dedup && !seen.insert(expr::structural_hash(&tree)) {
            continue;
        }

        if cfg.fit_iters > 0 {
            crate::fit::fit_weighted(&mut tree, &train.x, &train.y, &train.w, cfg.fit_iters);
            tree = expr::simplify(&tree);
        }
        let ind = Ind::evaluated(tree, train, cfg); // #25
        if ind.complexity > cfg.max_complexity as f64 {
            continue;
        }
        next.push(ind);
    }

    // Fill any remainder with fresh random trees.
    while next.len() < pop {
        let mut tree = expr::simplify(&random_tree(rng, nvars, &cfg.ops, cfg.init_depth));
        if cfg.fit_iters > 0 {
            crate::fit::fit_weighted(&mut tree, &train.x, &train.y, &train.w, cfg.fit_iters);
            tree = expr::simplify(&tree);
        }
        next.push(Ind::evaluated(tree, train, cfg)); // #25
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
fn prepare_seeds(cfg: &Config, train: &Sample, seeds: &[Node]) -> Vec<Ind> {
    let mut out: Vec<Ind> = Vec::new();
    // Two sets: one for the simplified input form (so a repeated line is
    // dropped even before fitting) and one for the fitted tree (so two
    // different starts that converge to the same expression count once).
    let mut seen_input: HashSet<u64> = HashSet::new();
    let mut seen_fitted: HashSet<u64> = HashSet::new();
    for seed in seeds {
        let mut tree = expr::simplify(seed);
        if !seen_input.insert(expr::structural_hash(&tree)) {
            continue;
        }
        if cfg.fit_iters > 0 {
            crate::fit::fit_weighted(&mut tree, &train.x, &train.y, &train.w, cfg.fit_iters);
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
    let t0 = std::time::Instant::now();

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
        for isl in 0..cfg.islands.max(1) {
            let depth = cfg.init_depth.min(1 + isl % 3);
            let mut pop = Vec::with_capacity(cfg.pop);
            for _ in 0..cfg.pop {
                pop.push(Ind::evaluated(
                    random_tree(&mut rng, nvars, &cfg.ops, depth),
                    init_train,
                    cfg, // #25
                ));
            }
            islands.push(pop);
        }

        gens_done = 0;
        converged_early = false;
        best_prev = f64::INFINITY;
        no_improve = 0;
    }

    // ---- seeded solutions (issue #16) ---------------------------------
    // Process every seed once: simplify, fit constants on the training
    // sample, evaluate its error, then insert it into the archive and into
    // every island's initial population. Seeds are elites: they are inserted
    // even above `max_complexity`, which only limits generated offspring.
    let mut seed_results: Vec<SeedResult> = Vec::new();
    if !seeds.is_empty() {
        // The population is being inserted before generation 0, whose subset
        // (if any) already scores the initial individuals; use the same sample
        // so every individual competes on identical rows.
        let init_sample = training_for_generation(cfg, train, 0);
        let init_train = init_sample.as_ref().unwrap_or(train);
        for (k, ind) in prepare_seeds(cfg, train, seeds).into_iter().enumerate() {
            archive.try_insert(Sol {
                complexity: expr::complexity(&ind.tree),
                cost: ind.complexity,
                tree: ind.tree.clone(),
                error: ind.error,
            });
            if k < cfg.pop {
                let in_pop = if init_sample.is_some() {
                    Ind::evaluated(ind.tree.clone(), init_train, cfg) // #25
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

    if !converged_early {
        for gen in gens_done..cfg.generations {
            gens_done = gen + 1;

            // One fixed subset for the whole generation; a fresh draw next time.
            // `None` means the full training sample, exactly as before.
            let gen_sample = training_for_generation(cfg, train, gen);
            let gen_train = gen_sample.as_ref().unwrap_or(train);

            // With subsampling active, re-score the carried-over population on
            // this generation's subset so every individual competes on the same
            // rows. (Inactive runs keep the original errors untouched.)
            if gen > 0 && gen_sample.is_some() {
                for pop in islands.iter_mut() {
                    for ind in pop.iter_mut() {
                        ind.reeval(gen_train, cfg); // #25
                    }
                }
            }

            // ---- parallel generation across islands ---------------------
            std::thread::scope(|s| {
                let handles: Vec<_> = islands
                    .iter_mut()
                    .zip(rngs.iter_mut())
                    .map(|(isl, r)| s.spawn(move || evolve_island(isl, r, cfg, gen_train, nvars)))
                    .collect();
                for h in handles {
                    let _ = h.join();
                }
            });

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

            if let Some(report) = &cfg.progress {
                report(Progress {
                    generation: gens_done,
                    total_generations: cfg.generations,
                    best_error: gen_best,
                    pareto_members: archive.items.len(),
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
                            *worst = Ind::evaluated(t.clone(), gen_train, cfg); // #25
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
                            let mut t = random_tree(&mut r, nvars, &cfg.ops, cfg.init_depth);
                            if cfg.fit_iters > 0 {
                                crate::fit::fit_weighted(
                                    &mut t,
                                    &gen_train.x,
                                    &gen_train.y,
                                    &gen_train.w,
                                    cfg.fit_iters,
                                );
                            }
                            *ind = Ind::evaluated(t, gen_train, cfg); // #25
                        }
                        rngs[ii] = r;
                        stagnant[ii] = 0;
                    }
                }
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

    if subsampling_active(cfg, train) {
        // Generation fitness was estimated on subsets, so the front is rebuilt
        // from the surviving trees and scored on every training row. From here
        // on the archive and every number derived from it are full-data.
        archive = full_data_archive(&archive, &islands, train, cfg); // #25
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
    let test_errors: Vec<f64> = sorted
        .iter()
        .map(|s| eval_error(&s.tree, test, cfg.metric))
        .collect();
    let mut final_archive = Archive::new(archive.cap);
    final_archive.items = sorted;
    Ok(Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early,
        seeds: seed_results,
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
        Sample { x, y, w: Vec::new() }
    }

    #[test]
    fn prepare_seeds_simplifies_fits_and_dedupes() {
        let s = sample();
        let mut cfg = Config::default();
        cfg.fit_iters = 8;
        // A deliberately wrong constant on the right shape.
        let seed = node(Op::Mul, vec![constant(3.0), input(0)]);
        let accepted = prepare_seeds(&cfg, &s, &[seed.clone(), seed]);
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
        let accepted = prepare_seeds(&cfg, &s, &[seed]);
        assert!(accepted[0].error > 1e-3, "an unfitted 3*x is not exact");
    }
}
