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

use crate::data::Sample;
use crate::expr::{self, complexity, crossover, eval, mutate, random_tree, Node, Op};
use crate::metrics::{score, Metric};
use crate::pareto::{Archive, Sol};
use crate::rng::Mt;
use std::collections::HashSet;

#[derive(Clone)]
pub struct Config {
    pub generations: usize,
    pub islands: usize,
    pub pop: usize,
    pub max_complexity: usize,
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
            progress: None,
        }
    }
}

#[derive(Clone)]
struct Ind {
    tree: Node,
    error: f64,
    complexity: usize,
}

impl Ind {
    fn evaluated(tree: Node, train: &Sample, metric: Metric) -> Self {
        let tree = expr::simplify(&tree);
        let complexity = complexity(&tree);
        let error = eval_error(&tree, train, metric);
        Ind {
            tree,
            error,
            complexity,
        }
    }

    fn reeval(&mut self, train: &Sample, metric: Metric) {
        self.tree = expr::simplify(&self.tree);
        self.complexity = complexity(&self.tree);
        self.error = eval_error(&self.tree, train, metric);
    }
}

pub fn eval_error(tree: &Node, s: &Sample, metric: Metric) -> f64 {
    if s.is_empty() {
        return f64::INFINITY;
    }
    let pred: Vec<f64> = s.x.iter().map(|row| eval(tree, row)).collect();
    score(&pred, &s.y, metric)
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
            ind.reeval(train, cfg.metric);
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
        if complexity(&tree) > cfg.max_complexity {
            continue;
        }
        if cfg.dedup && !seen.insert(expr::structural_hash(&tree)) {
            continue;
        }

        if cfg.fit_iters > 0 {
            crate::fit::fit(&mut tree, &train.x, &train.y, cfg.fit_iters);
            tree = expr::simplify(&tree);
        }
        let ind = Ind::evaluated(tree, train, cfg.metric);
        if ind.complexity > cfg.max_complexity {
            continue;
        }
        next.push(ind);
    }

    // Fill any remainder with fresh random trees.
    while next.len() < pop {
        let mut tree = expr::simplify(&random_tree(rng, nvars, &cfg.ops, cfg.init_depth));
        if cfg.fit_iters > 0 {
            crate::fit::fit(&mut tree, &train.x, &train.y, cfg.fit_iters);
            tree = expr::simplify(&tree);
        }
        next.push(Ind::evaluated(tree, train, cfg.metric));
    }

    *isl = next;
}

pub struct Outcome {
    pub archive: Archive,
    pub test_errors: Vec<f64>,
    pub generations: usize,
    pub converged_early: bool,
}

pub fn run(cfg: &Config, train: &Sample, test: &Sample) -> Outcome {
    let nvars = train.x.first().map(|r| r.len()).unwrap_or(0);
    let mut rng = Mt::new(cfg.seed);
    let mut archive = Archive::new(4000);
    let t0 = std::time::Instant::now();

    // Distinct, reproducible streams per island.
    let mut rngs: Vec<Mt> = (0..cfg.islands.max(1))
        .map(|i| Mt::new(cfg.seed ^ (0x9E37_79B9u32.wrapping_mul(i as u32 + 1))))
        .collect();

    let mut islands: Vec<Vec<Ind>> = Vec::with_capacity(cfg.islands.max(1));
    let mut stagnant: Vec<usize> = vec![0; cfg.islands.max(1)];
    for isl in 0..cfg.islands.max(1) {
        let depth = cfg.init_depth.min(1 + isl % 3);
        let mut pop = Vec::with_capacity(cfg.pop);
        for _ in 0..cfg.pop {
            pop.push(Ind::evaluated(
                random_tree(&mut rng, nvars, &cfg.ops, depth),
                train,
                cfg.metric,
            ));
        }
        islands.push(pop);
    }

    let mut gens_done = 0usize;
    let mut converged_early = false;
    let mut best_prev = f64::INFINITY;
    let mut no_improve = 0usize;

    for gen in 0..cfg.generations {
        gens_done = gen + 1;

        // ---- parallel generation across islands -------------------------
        std::thread::scope(|s| {
            let handles: Vec<_> = islands
                .iter_mut()
                .zip(rngs.iter_mut())
                .map(|(isl, r)| s.spawn(move || evolve_island(isl, r, cfg, train, nvars)))
                .collect();
            for h in handles {
                let _ = h.join();
            }
        });

        // ---- archive + convergence bookkeeping --------------------------
        let mut gen_best = f64::INFINITY;
        for pop in &islands {
            for ind in pop {
                if ind.error < gen_best {
                    gen_best = ind.error;
                }
                archive.try_insert(Sol {
                    tree: ind.tree.clone(),
                    error: ind.error,
                    complexity: ind.complexity,
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

        // ---- migration between islands ----------------------------------
        if cfg.islands > 1 && cfg.migration_interval > 0 && (gen + 1) % cfg.migration_interval == 0
        {
            let bests: Vec<Option<Node>> = islands
                .iter()
                .map(|pop| {
                    pop.iter()
                        .min_by(|a, b| {
                            a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|i| i.tree.clone())
                })
                .collect();
            for ii in 0..islands.len() {
                let src = (ii + 1) % islands.len();
                if let Some(t) = &bests[src] {
                    if let Some(worst) = islands[ii].iter_mut().max_by(|a, b| {
                        a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal)
                    }) {
                        *worst = Ind::evaluated(t.clone(), train, cfg.metric);
                    }
                }
            }
        }

        // ---- restart stagnant islands -----------------------------------
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
                            crate::fit::fit(&mut t, &train.x, &train.y, cfg.fit_iters);
                        }
                        *ind = Ind::evaluated(t, train, cfg.metric);
                    }
                    rngs[ii] = r;
                    stagnant[ii] = 0;
                }
            }
        }
    }

    for pop in &islands {
        for ind in pop {
            archive.try_insert(Sol {
                tree: ind.tree.clone(),
                error: ind.error,
                complexity: ind.complexity,
            });
        }
    }
    let sorted = archive.sorted();
    let test_errors: Vec<f64> = sorted
        .iter()
        .map(|s| eval_error(&s.tree, test, cfg.metric))
        .collect();
    let mut final_archive = Archive::new(archive.cap);
    final_archive.items = sorted;
    Outcome {
        archive: final_archive,
        test_errors,
        generations: gens_done,
        converged_early,
    }
}
