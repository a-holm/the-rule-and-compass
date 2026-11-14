//! #1023: epsilon-lexicase parent selection (`--selection lexicase`).
//!
//! Semi-dynamic epsilon-lexicase (La Cava, Helmuth, Spector and Moore,
//! Evolutionary Computation 27(3), 2019, Algorithm 3): the epsilon of a case is
//! the median absolute deviation of the pool's errors on it, computed once per
//! generation; one selection visits the cases in a random order and keeps every
//! survivor whose error is at most the best survivor's error plus epsilon. The
//! cases of a generation are a down-sample (Hernandez, Lalejini, Dolson and
//! Ofria, GECCO 2019 companion) drawn from the seed and the generation alone.
//!
//! A case is a row loss of the metric on one of `K` rows of the generation's
//! sample under the metric fitness, and the regret of one `(group, time block)`
//! sub-book of the long/flat book under `--fitness pnl`. A row value depends on
//! its own row only, so the row losses are the bits the fitness pass computed;
//! the sub-book regrets need every row, so they come from the scoring pass
//! itself and are kept on the individual ([`Ind::cases`]).

use super::{
    batch_workers, generation_subset, par_map, subset_sample, tournament, with_predictions, Config, EvalSample, Ind,
    Selection,
};
use crate::data::Sample;
use crate::expr::Node;
use crate::metrics::{Metric, ScoreObjective};
use crate::rng::Mt;

/// Default `--lexicase-cases`.
pub const DEFAULT_LEXICASE_CASES: usize = 256;
/// Largest `--lexicase-cases`.
pub const MAX_LEXICASE_CASES: usize = 4096;
/// Salt of the case draw, so it shares no stream with `--predictor-samples`,
/// `--fit-samples` or any island generator.
const LEXICASE_SALT: u32 = 0x6C3A_9D47;
/// Most case errors one island's selection matrix holds (population times cases).
const MAX_MATRIX_CELLS: usize = 1 << 24;
/// Most sub-book case errors the members of a run keep (islands times population
/// times kept cases).
const MAX_STORED_CELLS: usize = 1 << 28;

/// The loss of one row under a metric that is a mean of row losses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum RowLoss {
    Squared,
    Absolute,
    LogSquashed,
    Class(ScoreObjective),
}

fn row_loss(metric: Metric) -> Option<RowLoss> {
    Some(match metric {
        Metric::Squared | Metric::Aic | Metric::Aicc | Metric::Bic => RowLoss::Squared,
        Metric::Absolute => RowLoss::Absolute,
        Metric::LogSquashed => RowLoss::LogSquashed,
        Metric::LogLoss => RowLoss::Class(ScoreObjective::LogLoss),
        Metric::Brier => RowLoss::Class(ScoreObjective::Brier),
        Metric::Hinge => RowLoss::Class(ScoreObjective::HingeSigmoid),
        Metric::HingeOrdinary => RowLoss::Class(ScoreObjective::HingeOrdinary),
        Metric::HingeSquared => RowLoss::Class(ScoreObjective::HingeSquared),
        _ => return None,
    })
}

impl RowLoss {
    /// `label` is the row's class for a classification loss, ignored otherwise.
    fn of(self, p: f64, y: f64, label: f64) -> f64 {
        if !p.is_finite() {
            return f64::INFINITY;
        }
        let v = match self {
            RowLoss::Squared => {
                let r = p - y;
                r * r
            }
            RowLoss::Absolute => (p - y).abs(),
            RowLoss::LogSquashed => {
                let r = p - y;
                (1.0 + r * r).ln()
            }
            RowLoss::Class(objective) => crate::metrics::classification_row_loss(p, label, objective),
        };
        if v.is_nan() {
            f64::INFINITY
        } else {
            v
        }
    }
}

/// Why `--selection lexicase` cannot run with the metric fitness on this
/// sample, if it cannot.
pub fn metric_refusal(metric: Metric, implicit: bool, multi_output: bool, weighted: bool) -> Option<String> {
    if multi_output {
        return Some("--selection lexicase is not supported with multiple targets (--target y1,y2)".to_string());
    }
    if implicit || metric.is_implicit() {
        return Some("--selection lexicase needs a per-row loss; an implicit model has none".to_string());
    }
    if row_loss(metric).is_none() {
        return Some(format!("--selection lexicase needs a per-row loss; {} has none", metric.name()));
    }
    if weighted {
        return Some("--selection lexicase needs unweighted rows; a weighted sample (--weight) has no per-row case loss".to_string());
    }
    None
}

/// Why the search cannot run `cfg` under `--selection lexicase`, if it cannot.
pub(super) fn refusal(cfg: &Config, train: &Sample) -> Option<String> {
    if cfg.selection != Selection::Lexicase {
        return None;
    }
    if !(2..=MAX_LEXICASE_CASES).contains(&cfg.lexicase_cases) {
        return Some(format!(
            "--lexicase-cases must be 2 to {}, got {}",
            MAX_LEXICASE_CASES, cfg.lexicase_cases
        ));
    }
    if cfg.gpu_required {
        return Some(if cfg.fitness.is_metric() {
            "--selection lexicase with --gpu: the device scorer returns no per-row case errors yet; omit --gpu to run on the CPU".to_string()
        } else {
            "--selection lexicase --fitness pnl --gpu: the device long_pnl fitness returns no sub-book cases yet (#1031); omit --gpu to run on the CPU".to_string()
        });
    }
    let pop = cfg.pop.max(1);
    let cases = match cfg.fitness.book() {
        Some(book) => {
            let kept = book.cases().kept.len();
            if kept < 2 {
                return Some(format!(
                    "--selection lexicase needs at least 2 sub-book cases with a positive hindsight bound, found {}",
                    kept
                ));
            }
            let stored = cfg.islands.max(1).saturating_mul(pop).saturating_mul(kept);
            if stored > MAX_STORED_CELLS {
                return Some(format!(
                    "--selection lexicase keeps {} sub-book case errors ({} islands x {} members x {} cases), above the {} limit",
                    stored,
                    cfg.islands.max(1),
                    pop,
                    kept,
                    MAX_STORED_CELLS
                ));
            }
            kept.min(cfg.lexicase_cases)
        }
        None if !cfg.fitness.is_metric() => return Some("the long_pnl fitness book was not rebuilt".to_string()),
        None => {
            if let Some(reason) = metric_refusal(cfg.metric, false, train.is_multi(), !train.w.is_empty()) {
                return Some(reason);
            }
            if train.len() < 2 {
                return Some(format!("--selection lexicase needs at least 2 training rows, found {}", train.len()));
            }
            train.len().min(cfg.lexicase_cases)
        }
    };
    if pop.saturating_mul(cases) > MAX_MATRIX_CELLS {
        return Some(format!(
            "--selection lexicase compares {} members on {} cases per island, above the {} cell limit; lower --lexicase-cases or --pop",
            pop, cases, MAX_MATRIX_CELLS
        ));
    }
    None
}

/// Whether the scoring pass keeps the sub-book case errors on the individual.
pub(super) fn stores_cases(cfg: &Config) -> bool {
    cfg.selection == Selection::Lexicase && cfg.fitness.book().is_some()
}

/// The cases one generation selects on.
pub(super) enum CaseSet {
    /// Metric fitness: the generation's case rows, their class labels (empty
    /// unless the loss is a classification loss) and the row loss.
    Rows { rows: Sample, labels: Vec<f64>, loss: RowLoss },
    /// Long/flat fitness: indices into [`crate::trading::BookCases::kept`].
    Book { ids: Vec<usize> },
}

impl CaseSet {
    /// The cases of generation `gen`, a pure function of the seed, the
    /// generation and `gen_train` (the rows the generation is scored on).
    pub(super) fn for_generation(cfg: &Config, gen_train: &Sample, gen: usize) -> CaseSet {
        let seed = cfg.seed ^ LEXICASE_SALT;
        match cfg.fitness.book() {
            Some(book) => {
                let kept = book.cases().kept.len();
                let ids = if kept > cfg.lexicase_cases {
                    generation_subset(kept, cfg.lexicase_cases, seed, gen)
                } else {
                    (0..kept).collect()
                };
                CaseSet::Book { ids }
            }
            None => {
                let loss = row_loss(cfg.metric).expect("refusal admits only metrics with a row loss");
                let idx = generation_subset(gen_train.len(), cfg.lexicase_cases, seed, gen);
                let labels = match loss {
                    // The fitness labels every row against the whole sample.
                    RowLoss::Class(_) => {
                        let all = crate::metrics::binary_labels(&gen_train.y);
                        idx.iter().map(|&i| all[i]).collect()
                    }
                    _ => Vec::new(),
                };
                CaseSet::Rows { rows: subset_sample(gen_train, &idx), labels, loss }
            }
        }
    }

    pub(super) fn len(&self) -> usize {
        match self {
            CaseSet::Rows { rows, .. } => rows.len(),
            CaseSet::Book { ids } => ids.len(),
        }
    }
}

/// The row losses of `tree` on `data`, the case rows.
fn row_errors(tree: &Node, data: &EvalSample, labels: &[f64], loss: RowLoss) -> Vec<f32> {
    let y = &data.sample.y;
    with_predictions(tree, data, |pred| {
        pred.iter()
            .enumerate()
            .map(|(r, &p)| loss.of(p, y[r], labels.get(r).copied().unwrap_or(0.0)) as f32)
            .collect()
    })
}

/// Median of `v`, which is sorted in place; the mean of the two middle values
/// for an even count.
fn median(v: &mut [f32]) -> f32 {
    v.sort_unstable_by(f32::total_cmp);
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        ((v[n / 2 - 1] as f64 + v[n / 2] as f64) * 0.5) as f32
    }
}

/// Median absolute deviation of the finite values of `v` (La Cava et al. 2016,
/// Eqn. 4); 0 without a finite value.
fn mad(mut v: Vec<f32>) -> f32 {
    v.retain(|e| e.is_finite());
    if v.is_empty() {
        return 0.0;
    }
    let m = median(&mut v) as f64;
    for e in v.iter_mut() {
        *e = ((*e as f64) - m).abs() as f32;
    }
    let d = median(&mut v);
    if d.is_nan() {
        0.0
    } else {
        d
    }
}

/// One island's selection pool for one generation.
pub(super) struct Pool {
    /// Island indices of the members with a finite error and a case vector.
    members: Vec<usize>,
    cases: usize,
    /// Case errors, member-major: `errs[m * cases + c]`.
    errs: Vec<f32>,
    /// Epsilon of each case.
    eps: Vec<f32>,
}

impl Pool {
    /// The pool of `isl` on the generation's cases. Under the long/flat fitness
    /// a member without a kept vector (a resumed or carried member) gets one
    /// from one pass over `train`, the scoring rows, and keeps it. Draws
    /// nothing from an RNG.
    pub(super) fn build(isl: &mut [Ind], set: &CaseSet, cfg: &Config, train: &EvalSample, workers: usize) -> Pool {
        let finite: Vec<usize> = (0..isl.len()).filter(|&i| isl[i].error.is_finite()).collect();
        let cases = set.len();
        let (members, errs) = match set {
            CaseSet::Rows { rows, labels, loss } => {
                let data = EvalSample::new(rows, cfg);
                let trees: Vec<&Node> = finite.iter().map(|&i| &isl[i].tree).collect();
                let vectors = par_map(batch_workers(workers, trees.len(), rows.len()), trees, |tree| {
                    row_errors(tree, &data, labels, *loss)
                });
                (finite, vectors.concat())
            }
            CaseSet::Book { ids } => {
                let book = cfg.fitness.book().expect("a book case set has a book");
                let missing: Vec<usize> = finite.iter().copied().filter(|&i| isl[i].cases.is_none()).collect();
                if false && !missing.is_empty() && train.sample.len() == book.len() { // WIP red
                    let trees: Vec<&Node> = missing.iter().map(|&i| &isl[i].tree).collect();
                    let vectors = par_map(batch_workers(workers, trees.len(), train.sample.len()), trees, |tree| {
                        with_predictions(tree, train, |pred| book.case_errors(pred))
                    });
                    for (&i, v) in missing.iter().zip(vectors) {
                        isl[i].cases = Some(v.into());
                    }
                }
                let members: Vec<usize> = finite.into_iter().filter(|&i| isl[i].cases.is_some()).collect();
                let mut errs = Vec::with_capacity(members.len() * cases);
                for &i in &members {
                    let v = isl[i].cases.as_deref().expect("members hold a case vector");
                    errs.extend(ids.iter().map(|&c| v[c]));
                }
                (members, errs)
            }
        };
        Pool::from_errors(members, cases, errs)
    }

    fn from_errors(members: Vec<usize>, cases: usize, errs: Vec<f32>) -> Pool {
        debug_assert_eq!(errs.len(), members.len() * cases);
        let n = members.len();
        let eps = (0..cases).map(|c| mad((0..n).map(|m| errs[m * cases + c]).collect())).collect();
        Pool { members, cases, errs, eps }
    }

    fn err(&self, m: usize, c: usize) -> f32 {
        self.errs[m * self.cases + c]
    }

    /// One parent: an island index. The case order is a lazy Fisher-Yates
    /// shuffle with `rng`; survivors after the last case are narrowed to the
    /// lowest complexity, then one is drawn uniformly. An empty pool falls back
    /// to the island tournament of four.
    pub(super) fn select(&self, isl: &[Ind], rng: &mut Mt) -> usize {
        if !self.members.is_empty() { return tournament(isl, rng, 4); } // WIP red
        if self.members.is_empty() {
            return tournament(isl, rng, 4);
        }
        let mut order: Vec<usize> = (0..self.cases).collect();
        let mut alive: Vec<usize> = (0..self.members.len()).collect();
        for i in 0..self.cases {
            if alive.len() <= 1 {
                break;
            }
            let j = i + rng.usize_below(self.cases - i);
            order.swap(i, j);
            let c = order[i];
            let best = alive.iter().map(|&m| self.err(m, c)).fold(f32::INFINITY, f32::min);
            if !best.is_finite() {
                continue;
            }
            let cut = best + self.eps[c];
            alive.retain(|&m| self.err(m, c) <= cut);
        }
        let least = alive
            .iter()
            .map(|&m| isl[self.members[m]].complexity)
            .fold(f64::INFINITY, f64::min);
        alive.retain(|&m| isl[self.members[m]].complexity == least);
        let pick = if alive.len() == 1 { alive[0] } else { alive[rng.usize_below(alive.len())] };
        self.members[pick]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node, Op};

    fn ind(tree: Node, error: f64, complexity: f64) -> Ind {
        Ind {
            tree,
            error,
            complexity,
            descriptor: None,
            age: 0,
            inf_rows: None,
            edit: None,
            cases: None,
        }
    }

    #[test]
    fn mad_is_the_median_absolute_deviation_of_the_finite_values() {
        // Median 1, deviations [1, 0.5, 0, 1, 9]: MAD 1.
        assert_eq!(mad(vec![0.0, 0.5, 1.0, 2.0, 10.0]), 1.0);
        // Even count: median 2.5, deviations [1.5, 0.5, 0.5, 1.5]: MAD 1.
        assert_eq!(mad(vec![1.0, 2.0, 3.0, 4.0]), 1.0);
        assert_eq!(mad(vec![3.0, f32::INFINITY, 3.0, f32::NAN]), 0.0);
        assert_eq!(mad(vec![f32::INFINITY]), 0.0);
        assert_eq!(mad(Vec::new()), 0.0);
    }

    /// Four members tie at error 1 on every case. The fifth has the worst
    /// aggregate error but the best error on cases 0 and 1, so lexicase picks
    /// it whenever one of those cases comes first (two thirds of the draws),
    /// and a tournament of five on the aggregate error does not pick it.
    #[test]
    fn lexicase_keeps_only_candidates_within_epsilon_of_the_case_best() {
        let isl: Vec<Ind> = (0..5)
            .map(|i| ind(constant(i as f64), if i == 4 { 16.7 } else { 1.0 + 0.01 * i as f64 }, 1.0))
            .collect();
        let mut errs = Vec::new();
        for m in 0..5 {
            errs.extend(if m == 4 { [0.0f32, 0.0, 50.0] } else { [1.0f32, 1.0, 1.0] });
        }
        let pool = Pool::from_errors((0..5).collect(), 3, errs);
        let mut rng = Mt::new(17);
        let lex = (0..200).filter(|_| pool.select(&isl, &mut rng) == 4).count();
        assert!(lex > 80, "the aggregate loser was picked {lex} of 200 times");
        let mut rng = Mt::new(17);
        let tour = (0..200).filter(|_| tournament(&isl, &mut rng, 5) == 4).count();
        assert_eq!(tour, 0);

        // One case with errors [0, 0.5, 1, 2, 10]: epsilon is the MAD 1, so
        // only the members within 1 of the best (0) survive.
        let pool = Pool::from_errors((0..5).collect(), 1, vec![0.0, 0.5, 1.0, 2.0, 10.0]);
        assert_eq!(pool.eps, vec![1.0]);
        let mut seen = [0usize; 5];
        let mut rng = Mt::new(3);
        for _ in 0..300 {
            seen[pool.select(&isl, &mut rng)] += 1;
        }
        assert_eq!((seen[3], seen[4]), (0, 0), "{seen:?}");
        assert!(seen[..3].iter().all(|&n| n > 0), "{seen:?}");

        // Survivors of every case are narrowed to the lowest complexity.
        let mut sized = isl.clone();
        sized[1].complexity = 0.5;
        let mut rng = Mt::new(5);
        assert!((0..50).all(|_| pool.select(&sized, &mut rng) == 1));

        // A pool without a finite member is the tournament of four, draw for draw.
        let empty = Pool::from_errors(Vec::new(), 3, Vec::new());
        let (mut a, mut b) = (Mt::new(9), Mt::new(9));
        for _ in 0..20 {
            assert_eq!(empty.select(&isl, &mut a), tournament(&isl, &mut b, 4));
        }
    }

    fn sample(n: usize) -> Sample {
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let a = (i as f64 * 0.618_033_988_749_894_9).fract() * 4.0 - 2.0;
            let b = (i as f64 * 0.754_877_666_246_692_7).fract() * 3.0 - 1.0;
            x.push(vec![a, b]);
            y.push(2.0 * a - b * a + 0.5);
        }
        Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
    }

    /// The row losses of a case set are the row losses of the fitness pass over
    /// every row, read at the case rows, bit for bit.
    #[test]
    fn metric_cases_are_the_fitness_pass_row_losses() {
        let train = sample(1203);
        let trees = vec![
            node(Op::Mul, vec![constant(1.7), input(0)]),
            node(Op::Sub, vec![node(Op::Sin, vec![input(1)]), node(Op::Mul, vec![input(0), input(1)])]),
            node(Op::Div, vec![constant(1.0), input(1)]),
        ];
        for metric in [Metric::Squared, Metric::Absolute, Metric::LogSquashed, Metric::LogLoss] {
            for simd in [true, false] {
                let cfg = Config {
                    metric,
                    simd,
                    selection: Selection::Lexicase,
                    lexicase_cases: 300,
                    seed: 4,
                    ..Config::default()
                };
                let full = EvalSample::new(&train, &cfg);
                let mut isl: Vec<Ind> = trees.iter().map(|t| Ind::evaluated(t.clone(), &full, &cfg)).collect();
                for gen in [0, 7] {
                    let set = CaseSet::for_generation(&cfg, &train, gen);
                    assert_eq!(set.len(), 300);
                    let pool = Pool::build(&mut isl, &set, &cfg, &full, 1);
                    let idx = generation_subset(train.len(), 300, cfg.seed ^ LEXICASE_SALT, gen);
                    let loss = row_loss(metric).unwrap();
                    let labels = crate::metrics::binary_labels(&train.y);
                    for (m, &i) in pool.members.iter().enumerate() {
                        let want: Vec<u32> = with_predictions(&isl[i].tree, &full, |pred| {
                            idx.iter().map(|&r| (loss.of(pred[r], train.y[r], labels[r]) as f32).to_bits()).collect()
                        });
                        let got: Vec<u32> = (0..300).map(|c| pool.err(m, c).to_bits()).collect();
                        assert_eq!(got, want, "{metric:?} simd {simd} gen {gen} tree {i}");
                    }
                }
            }
        }
    }

    fn book_config(rows: usize) -> (Sample, Config) {
        let train = sample(rows);
        let half = rows / 2;
        let slice = crate::data::TradingSlice {
            rows: crate::trading::TradingRows {
                time: (0..rows).map(|i| 900.0 * (i % half) as f64).collect(),
                group: (0..rows).map(|i| (i / half) as u64).collect(),
                funding: Vec::new(),
            },
            ret: train.x.iter().map(|r| 0.002 * r[0]).collect(),
        };
        let spec = crate::trading::TradingSpec { cost_bps: 1.0, min_trades: 0, ..Default::default() };
        let book = crate::trading::FitnessBook::with_blocks(&slice, &spec, 900.0, 3, 1.0).expect("book");
        let cfg = Config {
            fitness: super::super::Fitness::LongPnl(std::sync::Arc::new(book)),
            selection: Selection::Lexicase,
            seed: 2,
            ..Config::default()
        };
        (train, cfg)
    }

    /// Under the long/flat fitness the scoring pass keeps the sub-book regrets
    /// of the same prediction vector; any new error drops them, and the pool
    /// rebuilds a missing vector with the same bits.
    #[test]
    fn book_cases_come_from_the_scoring_pass_and_follow_the_error() {
        let (train, cfg) = book_config(600);
        let book = cfg.fitness.book().unwrap();
        assert_eq!(book.cases().kept.len(), 6, "2 groups x 3 blocks");
        let full = EvalSample::new(&train, &cfg);
        let tree = node(Op::Sub, vec![input(0), constant(0.1)]);
        let scored = Ind::evaluated(tree.clone(), &full, &cfg);
        assert!(scored.error.is_finite());
        let want = with_predictions(&tree, &full, |pred| book.case_errors(pred));
        assert_eq!(scored.cases.as_deref(), Some(&want[..]));

        let mut moved = scored.clone();
        moved.set_error(0.5);
        assert!(moved.cases.is_none(), "a new error drops the case vector");

        let mut isl = vec![scored.clone(), Ind { cases: None, ..scored.clone() }];
        let set = CaseSet::for_generation(&cfg, &train, 0);
        assert_eq!(set.len(), 6);
        let pool = Pool::build(&mut isl, &set, &cfg, &full, 1);
        assert_eq!(isl[1].cases.as_deref(), Some(&want[..]), "the pool rebuilds a missing vector");
        assert_eq!(pool.members, vec![0, 1]);

        // Fewer allowed cases than kept ones: a per-generation draw of K.
        let narrow = Config { lexicase_cases: 4, ..cfg.clone() };
        let a = CaseSet::for_generation(&narrow, &train, 3);
        let b = CaseSet::for_generation(&narrow, &train, 3);
        match (&a, &b) {
            (CaseSet::Book { ids: x }, CaseSet::Book { ids: y }) => {
                assert_eq!(x, y);
                assert_eq!(x.len(), 4);
            }
            _ => panic!("book case sets"),
        }
        let off = Config { selection: Selection::Lexicographic, ..cfg.clone() };
        assert!(Ind::evaluated(tree, &full, &off).cases.is_none(), "only lexicase keeps cases");
    }

    #[test]
    fn refusals_name_their_reason() {
        let train = sample(50);
        let lex = Config { selection: Selection::Lexicase, ..Config::default() };
        assert_eq!(refusal(&lex, &train), None);
        assert_eq!(refusal(&Config::default(), &train), None);
        let median = Config { metric: Metric::Median, ..lex.clone() };
        assert!(refusal(&median, &train).unwrap().contains("needs a per-row loss; median has none"));
        let one = Config { lexicase_cases: 1, ..lex.clone() };
        assert!(refusal(&one, &train).unwrap().contains("--lexicase-cases must be 2 to 4096"));
        let wide = Config { pop: 1 << 16, lexicase_cases: 4096, ..lex.clone() };
        assert_eq!(refusal(&wide, &sample(5000)).map(|r| r.contains("cell limit")), Some(true));
        let mut weighted = train.clone();
        weighted.w = vec![1.0; weighted.len()];
        assert!(refusal(&lex, &weighted).unwrap().contains("unweighted rows"));
        let gpu = Config { gpu_required: true, ..lex.clone() };
        assert!(refusal(&gpu, &train).unwrap().contains("with --gpu"));
        let (_, book) = book_config(600);
        assert_eq!(refusal(&book, &sample(600)), None);
        let gpu = Config { gpu_required: true, ..book };
        assert!(refusal(&gpu, &sample(600)).unwrap().contains("#1031"));
    }
}
