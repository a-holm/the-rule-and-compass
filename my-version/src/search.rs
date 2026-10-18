//! Island-model, multi-objective evolutionary search.
//!
//! A population of expression individuals evolves on separate islands, tracked
//! against a global Pareto front over (error, complexity), with generations,
//! migration and restarts.

use crate::data::Sample;
use crate::expr::{self, complexity, crossover, eval, mutate, random_tree, Node, Op};
use crate::metrics::{score, Metric};
use crate::pareto::{Archive, Sol};
use crate::rng::Mt;

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
}

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
    fn new(tree: Node) -> Self {
        let c = complexity(&tree);
        Ind {
            tree,
            error: f64::INFINITY,
            complexity: c,
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

fn dominates(ae: f64, ac: usize, be: f64, bc: usize) -> bool {
    ae <= be && ac <= bc && (ae < be || ac < bc)
}

fn tournament(island: &[Ind], rng: &mut Mt, k: usize) -> usize {
    let n = island.len();
    if n == 0 {
        return 0;
    }
    let mut best = rng.usize_below(n);
    for _ in 1..k {
        let c = rng.usize_below(n);
        let better = if dominates(island[c].error, island[c].complexity, island[best].error, island[best].complexity) {
            true
        } else if dominates(island[best].error, island[best].complexity, island[c].error, island[c].complexity) {
            false
        } else {
            rng.bool(0.5)
        };
        if better {
            best = c;
        }
    }
    best
}

/// Coordinate-descent fitting of constant leaves. [I]
fn fit_constants(tree: &mut Node, rng: &mut Mt, train: &Sample, metric: Metric, passes: usize) {
    if !eval_error(tree, train, metric).is_finite() {
        return;
    }
    for _ in 0..passes {
        let n = complexity(tree);
        for i in 0..n {
            let old = match expr::nth(tree, i) {
                Some(t) if t.op == Op::Const => t.val,
                _ => continue,
            };
            let mut best_v = old;
            let mut best_e = eval_error(tree, train, metric);
            let cands = [
                old * 0.5,
                old * 0.8,
                old * 1.25,
                old * 2.0,
                old + 0.1,
                old - 0.1,
                old + 0.5,
                old - 0.5,
                old + 1.0,
                old - 1.0,
                rng.range(-3.0, 3.0),
            ];
            for c in cands {
                if let Some(t) = expr::nth_mut(tree, i) {
                    if t.op == Op::Const {
                        t.val = c;
                    }
                }
                let e = eval_error(tree, train, metric);
                if e < best_e {
                    best_e = e;
                    best_v = c;
                }
            }
            if let Some(t) = expr::nth_mut(tree, i) {
                if t.op == Op::Const {
                    t.val = best_v;
                }
            }
        }
    }
}

pub struct Outcome {
    pub archive: Archive,
    pub test_errors: Vec<f64>,
    pub generations: usize,
}

pub fn run(cfg: &Config, train: &Sample, test: &Sample) -> Outcome {
    let nvars = train.x.first().map(|r| r.len()).unwrap_or(0);
    let mut rng = Mt::new(cfg.seed);
    let mut archive = Archive::new(4000);

    let mut islands: Vec<Vec<Ind>> = Vec::with_capacity(cfg.islands);
    let mut stagnant: Vec<usize> = vec![0; cfg.islands];
    for isl in 0..cfg.islands {
        let depth = cfg.init_depth.min(1 + isl % 3);
        let mut pop = Vec::with_capacity(cfg.pop);
        for _ in 0..cfg.pop {
            pop.push(Ind::new(random_tree(&mut rng, nvars, &cfg.ops, depth)));
        }
        islands.push(pop);
    }

    for gen in 0..cfg.generations {
        for ii in 0..islands.len() {
            for ind in islands[ii].iter_mut() {
                if !ind.error.is_finite() {
                    ind.reeval(train, cfg.metric);
                }
            }
            let snapshot: Vec<(Node, f64, usize)> = islands[ii]
                .iter()
                .map(|i| (i.tree.clone(), i.error, i.complexity))
                .collect();
            for (tree, error, complexity) in snapshot {
                archive.try_insert(Sol {
                    tree,
                    error,
                    complexity,
                });
            }

            let mut next: Vec<Ind> = Vec::with_capacity(cfg.pop);
            if let Some(best) = islands[ii]
                .iter()
                .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
            {
                next.push(best.clone());
            }
            while next.len() < cfg.pop {
                let p1 = tournament(&islands[ii], &mut rng, 4);
                let mut child = islands[ii][p1].tree.clone();
                if rng.bool(0.5) {
                    let p2 = tournament(&islands[ii], &mut rng, 4);
                    let parent2 = islands[ii][p2].tree.clone();
                    crossover(&mut child, &parent2, &mut rng);
                }
                mutate(&mut child, &mut rng, nvars, &cfg.ops);
                if rng.bool(0.5) {
                    mutate(&mut child, &mut rng, nvars, &cfg.ops);
                }
                let mut ind = Ind::new(expr::simplify(&child));
                if ind.complexity > cfg.max_complexity {
                    continue;
                }
                ind.reeval(train, cfg.metric);
                fit_constants(&mut ind.tree, &mut rng, train, cfg.metric, 1);
                ind.reeval(train, cfg.metric);
                if ind.complexity <= cfg.max_complexity {
                    next.push(ind);
                }
            }
            while next.len() < cfg.pop {
                let t = random_tree(&mut rng, nvars, &cfg.ops, cfg.init_depth);
                let mut ind = Ind::new(t);
                ind.reeval(train, cfg.metric);
                next.push(ind);
            }
            islands[ii] = next;
        }

        if cfg.islands > 1 && cfg.migration_interval > 0 && (gen + 1) % cfg.migration_interval == 0 {
            let bests: Vec<Option<Node>> = islands
                .iter()
                .map(|pop| {
                    pop.iter()
                        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
                        .map(|i| i.tree.clone())
                })
                .collect();
            for ii in 0..islands.len() {
                let src = (ii + 1) % islands.len();
                if let Some(t) = &bests[src] {
                    if let Some(worst) = islands[ii]
                        .iter_mut()
                        .max_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
                    {
                        *worst = Ind::new(t.clone());
                        worst.reeval(train, cfg.metric);
                    }
                }
            }
        }

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
                    for ind in islands[ii].iter_mut() {
                        *ind = Ind::new(random_tree(&mut rng, nvars, &cfg.ops, cfg.init_depth));
                        ind.reeval(train, cfg.metric);
                    }
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
        generations: cfg.generations,
    }
}
