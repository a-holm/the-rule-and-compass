//! #1183: `rcsr enumerate` (type stubs for the red commit).
#![allow(dead_code, unused_variables, unused_imports)]

use crate::data::{Sample, TradingSlice, TradingSplit};
use crate::devlog::{num, text, Json};
use crate::eval_soa;
use crate::expr::{self, Node, Op};
use crate::trading::{BlockStat, FitnessBook, FitnessTotals, Objective, TradingRows, TradingSpec};
use std::collections::{BTreeSet, HashSet};

pub const SCHEMA: &str = "rcsr-enumerate/1";
pub const DEFAULT_MAX_SIZE: usize = 3;
pub const PLAIN_MAX_SIZE: usize = 4;
pub const LARGE_MAX_SIZE: usize = 6;
pub const DEFAULT_CUTS: usize = 16;
pub const MAX_CUTS: usize = 64;
pub const DEFAULT_TOP: usize = 50;
pub const MIN_NULL_SHIFTS: usize = 19;

pub fn op_name(op: Op) -> String {
    op.symbol().to_string()
}

pub fn class_ops(ops: &[Op]) -> Result<Vec<Op>, String> {
    Ok(ops.to_vec())
}

pub fn root_skipped(op: Op, ops: &[Op]) -> bool {
    false
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SizeCount {
    pub size: usize,
    pub raw: u64,
    pub pool: u64,
    pub canonical: u64,
}

pub fn counts(nvars: usize, ops: &[Op], max_size: usize) -> Vec<SizeCount> {
    vec![SizeCount::default(); max_size]
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Entry {
    op: Op,
    a: u32,
    b: u32,
}

pub struct Class {
    ops: Vec<Op>,
    max_size: usize,
    entries: Vec<Entry>,
    bounds: Vec<usize>,
    scored: Vec<u32>,
}

impl Class {
    pub fn build(nvars: usize, ops: &[Op], max_size: usize, canonical: bool) -> Result<Class, String> {
        Err("not implemented".to_string())
    }
    pub fn ops(&self) -> &[Op] {
        &self.ops
    }
    pub fn max_size(&self) -> usize {
        self.max_size
    }
    pub fn scored(&self) -> &[u32] {
        &self.scored
    }
    pub fn pool_len(&self, s: usize) -> usize {
        0
    }
    pub fn size_of(&self, entry: usize) -> usize {
        0
    }
    pub fn node(&self, i: usize) -> Node {
        expr::input(0)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    Above,
    Below,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cut {
    pub side: Side,
    pub value: f64,
    pub share: Option<f64>,
}

pub fn shares(q: usize) -> Vec<f64> {
    Vec::new()
}

pub fn grid(values: &[f64], q: usize, scratch: &mut Vec<f64>) -> Vec<Cut> {
    Vec::new()
}

pub struct Scan<'a> {
    pub class: &'a Class,
    pub cols: &'a [Vec<f64>],
    pub rows: usize,
    pub cuts: usize,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Outcome {
    NonFinite,
    Constant,
    Scored { fitness: f64, cut: Cut, cuts: usize },
}

pub struct Pass {
    pub best: Vec<f64>,
    pub outcomes: Vec<Outcome>,
    pub books: HashSet<u64>,
}

pub fn pass(scan: &Scan, books: &[&FitnessBook], detail: bool) -> Pass {
    Pass { best: vec![f64::INFINITY; books.len()], outcomes: Vec::new(), books: HashSet::new() }
}

pub fn shifted_slice(slice: &TradingSlice, num: u64, den: u64) -> Result<TradingSlice, String> {
    Err("not implemented".to_string())
}

#[derive(Clone, Debug, PartialEq)]
pub struct NullPlan {
    pub periods: usize,
    pub block: usize,
    pub lo: u64,
    pub count: u64,
    pub nums: Vec<u64>,
}

pub fn null_plan(rows: &TradingRows, horizon_seconds: f64, shifts: usize, seed: u32) -> Result<NullPlan, String> {
    Err("not implemented".to_string())
}

pub struct BookParams<'a> {
    pub spec: &'a TradingSpec,
    pub horizon_seconds: f64,
    pub blocks: usize,
    pub lambda: f64,
    pub stat: BlockStat,
    pub segments: &'a [f64],
}

pub fn book_of(slice: &TradingSlice, p: &BookParams) -> Result<FitnessBook, String> {
    Err("not implemented".to_string())
}

pub fn p_family(fitness: f64, null: &[f64]) -> f64 {
    1.0
}

#[derive(Clone, Debug)]
pub struct Member {
    pub position: usize,
    pub size: usize,
    pub cut: Cut,
    pub constant: f64,
    pub fitness: f64,
    pub net: f64,
    pub long_rows: u64,
    pub flips: u64,
    pub exits: u64,
    pub entry_times: u64,
    pub rows: usize,
    pub tree: String,
    pub expression: String,
    pub node: Node,
}

pub struct MemberCtx<'a> {
    pub class: &'a Class,
    pub cols: &'a [Vec<f64>],
    pub rows: usize,
    pub time: &'a [f64],
    pub book: &'a FitnessBook,
    pub means: &'a [f64],
    pub stds: &'a [f64],
    pub vars: &'a [String],
}

pub fn members(ctx: &MemberCtx, outcomes: &[Outcome], top: usize) -> (Vec<Option<Member>>, Vec<Member>) {
    (Vec::new(), Vec::new())
}

#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub max_size: usize,
}

static REQUEST: std::sync::OnceLock<Request> = std::sync::OnceLock::new();

pub fn request() -> Option<&'static Request> {
    REQUEST.get()
}

pub fn claim(args: &[String]) -> Vec<String> {
    eprintln!("error: rcsr enumerate: not implemented");
    std::process::exit(2);
}

pub struct Inputs<'a> {
    pub target: &'a str,
    pub vars: &'a [String],
    pub ops: &'a [Op],
    pub train: &'a Sample,
    pub validation: Option<&'a Sample>,
    pub trading: Option<&'a TradingSplit>,
    pub means: &'a [f64],
    pub stds: &'a [f64],
    pub book: Option<&'a FitnessBook>,
    pub spec: Option<&'a TradingSpec>,
    pub objective: Objective,
    pub horizon_seconds: f64,
    pub blocks: usize,
    pub lambda: f64,
    pub stat: BlockStat,
    pub train_segments: Vec<f64>,
    pub validation_segments: Vec<f64>,
    pub split_json: String,
    pub seed: u32,
    pub threads: usize,
}

pub fn start(req: &Request, inputs: &Inputs) -> ! {
    eprintln!("error: rcsr enumerate: not implemented");
    std::process::exit(2);
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        fn unit(&mut self) -> f64 {
            ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
        }

        fn gauss(&mut self) -> f64 {
            let (u, v) = (self.unit(), self.unit());
            (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
        }
    }

    fn ops(names: &str) -> Vec<Op> {
        Op::parse_list_checked(names).unwrap()
    }

    const OPS10: &str = "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss";

    /// A panel of `groups x steps` rows in canonical order: raw columns with
    /// means and scales away from 0 and 1, their normalised copies, and returns.
    struct Panel {
        raw: Vec<Vec<f64>>,
        cols: Vec<Vec<f64>>,
        means: Vec<f64>,
        stds: Vec<f64>,
        slice: TradingSlice,
        vars: Vec<String>,
    }

    fn panel(seed: u64, groups: usize, steps: usize, nvars: usize, ret: impl Fn(&[f64], &mut Rng) -> f64) -> Panel {
        let mut rng = Rng(seed);
        let n = groups * steps;
        let mut raw = vec![vec![0.0f64; n]; nvars];
        let (mut time, mut group, mut r) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for g in 0..groups {
            for k in 0..steps {
                let i = g * steps + k;
                let row: Vec<f64> = (0..nvars).map(|j| 2.0 * rng.unit() - 1.0).collect();
                for j in 0..nvars {
                    raw[j][i] = 3.0 + j as f64 + (1.5 + 0.5 * j as f64) * row[j];
                }
                time.push(3600.0 * k as f64);
                group.push(g as u64);
                r.push(ret(&row, &mut rng));
            }
        }
        let means: Vec<f64> = raw.iter().map(|c| c.iter().sum::<f64>() / n as f64).collect();
        let stds: Vec<f64> = raw.iter().zip(&means).map(|(c, m)| (c.iter().map(|v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt()).collect();
        let cols = raw.iter().enumerate().map(|(j, c)| c.iter().map(|v| (v - means[j]) / stds[j]).collect()).collect();
        let vars = (0..nvars).map(|j| format!("x{}", j + 1)).collect();
        Panel { raw, cols, means, stds, slice: TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret: r }, vars }
    }

    fn spec() -> TradingSpec {
        TradingSpec { cost_bps: 5.0, min_trades: 1, ..TradingSpec::default() }
    }

    fn book(p: &Panel, blocks: usize) -> FitnessBook {
        FitnessBook::with_blocks(&p.slice, &spec(), 0.0, blocks, 1.0).unwrap()
    }

    fn noise(_: &[f64], rng: &mut Rng) -> f64 {
        0.004 * rng.gauss()
    }

    #[test]
    fn class_counts_match_the_closed_form_and_a_brute_force_count() {
        for (nvars, names, max) in [(3usize, "add,sub,mul,abs", 3usize), (2, OPS10, 4), (2, "sub,div,neg,mul", 4), (32, OPS10, 3)] {
            let o = ops(names);
            let closed = counts(nvars, &o, max);
            let canonical = Class::build(nvars, &o, max, true).unwrap();
            let raw = Class::build(nvars, &o, max, false).unwrap();
            assert_eq!(raw.scored().len() as u64, closed.iter().map(|c| c.raw).sum::<u64>(), "{names}: raw");
            for c in &closed {
                assert_eq!(canonical.pool_len(c.size) as u64, c.pool, "{names} size {}: pool", c.size);
                let scored = canonical.scored().iter().filter(|&&e| canonical.size_of(e as usize) == c.size).count() as u64;
                assert_eq!(scored, c.canonical, "{names} size {}: canonical", c.size);
            }
            if nvars > 3 {
                continue;
            }
            // Brute force: the raw trees without a sub(a, a) or div(a, a) node, up to
            // the order of the children of add and mul.
            fn form(n: &Node, vars: &[String]) -> Option<String> {
                let kids: Option<Vec<String>> = n.kids.iter().map(|k| form(k, vars)).collect();
                let mut kids = kids?;
                if matches!(n.op, Op::Sub | Op::Div) && kids[0] == kids[1] {
                    return None;
                }
                if matches!(n.op, Op::Add | Op::Mul) {
                    kids.sort();
                }
                Some(match n.op {
                    Op::Input => vars[n.val as usize].clone(),
                    _ => format!("{}({})", op_name(n.op), kids.join(",")),
                })
            }
            let vars: Vec<String> = (0..nvars).map(|j| format!("v{j}")).collect();
            for c in &closed {
                let forms: HashSet<String> = (0..raw.entries.len()).filter(|&e| raw.size_of(e) == c.size).filter_map(|e| form(&raw.node(e), &vars)).collect();
                assert_eq!(forms.len() as u64, c.pool, "{names} size {}: brute-force pool", c.size);
            }
        }
        // 32 variables and the ten campaign operators: raw counts of the issue table;
        // the pool leaves out sub(a, a) and div(a, a); canonical leaves out five roots.
        let c = counts(32, &ops(OPS10), 4);
        let got: Vec<(u64, u64, u64)> = c.iter().map(|s| (s.raw, s.pool, s.canonical)).collect();
        assert_eq!(got, vec![(32, 32, 32), (192, 192, 32), (5_248, 4_192, 3_232), (80_640, 62_016, 41_056)]);
    }

    #[test]
    fn class_order_is_fixed() {
        let class = Class::build(2, &ops("abs,add,sub"), 3, true).unwrap();
        let vars = vec!["x1".to_string(), "x2".to_string()];
        let order: Vec<String> = class.scored().iter().map(|&e| expr::to_string(&class.node(e as usize), &vars)).collect();
        assert_eq!(
            order,
            [
                "x1", "x2", "abs(x1)", "abs(x2)", "abs(abs(x1))", "abs(abs(x2))", "(x1 + x1)", "(x1 + x2)", "(x2 + x2)", "(x1 - x2)", "(x2 - x1)"
            ]
        );
        let raw = Class::build(2, &ops("tanh,mul"), 3, false).unwrap();
        let order: Vec<String> = raw.scored().iter().map(|&e| expr::to_string(&raw.node(e as usize), &vars)).collect();
        assert_eq!(order, ["x1", "x2", "tanh(x1)", "tanh(x2)", "tanh(tanh(x1))", "tanh(tanh(x2))", "(x1 * x1)", "(x1 * x2)", "(x2 * x1)", "(x2 * x2)"]);
        let skipped = Class::build(2, &ops("tanh,mul"), 3, true).unwrap();
        let order: Vec<String> = skipped.scored().iter().map(|&e| expr::to_string(&skipped.node(e as usize), &vars)).collect();
        assert_eq!(order, ["x1", "x2", "(x1 * x1)", "(x1 * x2)", "(x2 * x2)"]);
        assert!(class_ops(&ops("add,if")).unwrap_err().contains("'if'"));
    }

    #[test]
    fn cut_grid_holds_the_shares() {
        let mut rng = Rng(5);
        let n = 1000usize;
        let mut values: Vec<f64> = (0..n).map(|i| i as f64 * 0.37 - 50.0).collect();
        for i in (1..n).rev() {
            values.swap(i, (rng.next() % (i as u64 + 1)) as usize);
        }
        let mut scratch = Vec::new();
        let cuts = grid(&values, 16, &mut scratch);
        assert_eq!(cuts.len(), 16);
        let sh = shares(16);
        assert_eq!(sh.len(), 8);
        assert_eq!(sh[0], 0.5);
        assert_eq!(sh[2], 0.5 * 0.7 * 0.7);
        for (k, c) in cuts.iter().enumerate() {
            let s = sh[k % 8];
            assert_eq!(c.share, Some(s));
            assert_eq!(c.side, if k < 8 { Side::Above } else { Side::Below });
            let long = values.iter().filter(|&&v| if c.side == Side::Above { v > c.value } else { v < c.value }).count();
            assert_eq!(long, (s * n as f64).round() as usize, "cut {k}");
        }
        // Fewer distinct values than cuts: every midpoint, on both sides.
        let few: Vec<f64> = (0..300).map(|i| [2.0, -1.0, 0.5][i % 3]).collect();
        let cuts = grid(&few, 16, &mut scratch);
        let got: Vec<(Side, f64, Option<f64>)> = cuts.iter().map(|c| (c.side, c.value, c.share)).collect();
        assert_eq!(got, vec![(Side::Above, -0.25, None), (Side::Above, 1.25, None), (Side::Below, -0.25, None), (Side::Below, 1.25, None)]);
        assert!(grid(&[1.5; 40], 16, &mut scratch).is_empty());
    }

    #[test]
    fn shifted_returns_follow_the_shifted_positions_rule() {
        let groups: Vec<u64> = [0u64; 7].iter().chain([1u64; 12].iter()).chain([2u64; 5].iter()).copied().collect();
        let n = groups.len();
        let ones: Vec<f64> = (0..n).map(|i| if (i * 5 + i / 3) % 4 == 0 { 1.0 } else { 0.0 }).collect();
        let funding: Vec<f64> = (0..n).map(|i| i as f64 * 1e-5).collect();
        let slice = TradingSlice { rows: TradingRows { time: (0..n).map(|i| i as f64).collect(), group: groups.clone(), funding: funding.clone() }, ret: ones.clone() };
        for (num, den) in [(0u64, 12u64), (3, 12), (7, 12), (11, 12), (25, 12), (5, 7)] {
            let shifted = shifted_slice(&slice, num, den).unwrap();
            assert_eq!(shifted.ret, crate::baselines::shifted_positions(&ones, &groups, num, den).unwrap(), "{num}/{den}");
            let index: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let moved = shifted_slice(&TradingSlice { ret: index, ..slice.clone() }, num, den).unwrap().ret;
            for i in 0..n {
                assert_eq!(shifted.rows.funding[i], funding[moved[i] as usize], "funding moves with its return");
            }
        }
    }

    #[test]
    fn shifted_returns_equal_inversely_shifted_positions() {
        // Balanced: every group has `steps` rows (one joint period shift), the
        // positions are flat on both rows of the rotation seam, and the returns
        // there are far below cost, so no path is long across a seam.
        let (groups, steps, num) = (3usize, 120usize, 37u64);
        let mut rng = Rng(77);
        let n = groups * steps;
        let mut ret = vec![0.0f64; n];
        let mut pos = vec![0.0f64; n];
        let seams = [0usize, steps - 1, (steps - num as usize) % steps, (steps - num as usize - 1) % steps, num as usize, num as usize - 1];
        for g in 0..groups {
            for k in 0..steps {
                let i = g * steps + k;
                ret[i] = if seams.contains(&k) { -0.05 } else { 0.004 * rng.gauss() + 0.001 };
                pos[i] = if seams.contains(&k) || rng.unit() < 0.4 { 0.0 } else { 1.0 };
            }
        }
        let rows = TradingRows { time: (0..n).map(|i| 3600.0 * (i % steps) as f64).collect(), group: (0..n).map(|i| (i / steps) as u64).collect(), funding: Vec::new() };
        let slice = TradingSlice { rows, ret };
        let spec = spec();
        let fixed = FitnessBook::new(&slice, &spec, 0.0).unwrap();
        let shifted = FitnessBook::new(&shifted_slice(&slice, num, steps as u64).unwrap(), &spec, 0.0).unwrap();
        let back = crate::baselines::shifted_positions(&pos, &slice.rows.group, steps as u64 - num, steps as u64).unwrap();
        let a = shifted.totals(&pos);
        let b = fixed.totals(&back);
        assert_eq!((a.flips, a.exits, a.long_rows), (b.flips, b.exits, b.long_rows));
        assert!((a.gross - b.gross).abs() <= 1e-12, "{} vs {}", a.gross, b.gross);
        assert!((shifted.bound() - fixed.bound()).abs() <= 1e-12 * fixed.bound());
        let (fa, fb) = (shifted.regret(&pos), fixed.regret(&back));
        assert!((fa - fb).abs() <= 1e-9, "{fa} vs {fb}");
    }

    fn ctx<'a>(class: &'a Class, p: &'a Panel, book: &'a FitnessBook) -> MemberCtx<'a> {
        MemberCtx { class, cols: &p.cols, rows: p.slice.ret.len(), time: &p.slice.rows.time, book, means: &p.means, stds: &p.stds, vars: &p.vars }
    }

    #[test]
    fn enumerated_fitness_is_the_book_of_the_printed_expression() {
        // 4 groups x 500 steps = 2,000 rows; returns follow a rule on x1 and x2.
        let p = panel(1183, 4, 500, 3, |x, rng| if x[0] - x[1] < -0.3 { 0.003 } else { -0.002 } + 0.004 * rng.gauss());
        let class = Class::build(3, &ops(OPS10), 3, true).unwrap();
        let rows: Vec<Vec<f64>> = (0..2000).map(|i| p.raw.iter().map(|c| c[i]).collect()).collect();
        for blocks in [1usize, 4] {
            let b = book(&p, blocks);
            let scan = Scan { class: &class, cols: &p.cols, rows: 2000, cuts: 16, threads: 2 };
            let pass = pass(&scan, &[&b], true);
            let (best, top) = members(&ctx(&class, &p, &b), &pass.outcomes, 50);
            assert_eq!(top.len(), 50);
            let reported: Vec<&Member> = best.iter().flatten().chain(top.iter()).collect();
            assert_eq!(reported.len(), 53);
            for m in reported {
                let tree = expr::parse(&m.expression, &p.vars).unwrap();
                let pred: Vec<f64> = rows.iter().map(|r| expr::eval(&tree, r)).collect();
                assert_eq!(b.regret(&pred).to_bits(), m.fitness.to_bits(), "{}: {}", m.expression, m.fitness);
                let Outcome::Scored { fitness, cut, .. } = pass.outcomes[m.position] else { panic!("{} was not scored", m.expression) };
                assert_eq!((fitness.to_bits(), cut), (m.fitness.to_bits(), m.cut));
            }
            assert_eq!(top[0].fitness.to_bits(), pass.best[0].to_bits());
        }
    }

    #[test]
    fn canonical_forms_keep_every_book() {
        let p = panel(3, 4, 500, 3, noise);
        let b = book(&p, 2);
        let mut sets = Vec::new();
        for canonical in [true, false] {
            let class = Class::build(3, &ops(OPS10), 3, canonical).unwrap();
            let scan = Scan { class: &class, cols: &p.cols, rows: 2000, cuts: 16, threads: 4 };
            let pass = pass(&scan, &[&b], true);
            sets.push((class.scored().len(), pass.books));
        }
        assert!(sets[0].0 < sets[1].0, "the skips leave out trees: {} vs {}", sets[0].0, sets[1].0);
        assert!(sets[0].1.len() > 300, "{} books", sets[0].1.len());
        assert_eq!(sets[0].1, sets[1].1, "the canonical class keeps every book of the raw class");
    }

    #[test]
    fn thread_count_does_not_change_the_pass() {
        let p = panel(9, 3, 400, 3, noise);
        let b = book(&p, 4);
        let shifted = book_of(&shifted_slice(&p.slice, 101, 400).unwrap(), &BookParams { spec: &spec(), horizon_seconds: 0.0, blocks: 4, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] }).unwrap();
        let class = Class::build(3, &ops(OPS10), 3, true).unwrap();
        let run = |threads: usize| pass(&Scan { class: &class, cols: &p.cols, rows: 1200, cuts: 16, threads }, &[&b, &shifted], true);
        let (one, four) = (run(1), run(4));
        assert_eq!(one.outcomes, four.outcomes);
        assert_eq!(one.books, four.books);
        assert_eq!(one.best.iter().map(|f| f.to_bits()).collect::<Vec<_>>(), four.best.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
    }

    #[test]
    fn null_plan_draws_distinct_shifts_from_the_walk_forward_range() {
        let p = panel(4, 2, 300, 1, noise);
        let plan = null_plan(&p.slice.rows, 3600.0, 19, 7).unwrap();
        assert_eq!(plan.periods, 300);
        assert_eq!(plan.block, 7);
        assert_eq!((plan.lo, plan.count), (7, 287));
        assert_eq!(plan.nums.len(), 19);
        assert!(plan.nums.windows(2).all(|w| w[0] < w[1]));
        assert!(plan.nums.iter().all(|&k| k >= plan.lo && k < plan.lo + plan.count));
        assert_eq!(plan, null_plan(&p.slice.rows, 3600.0, 19, 7).unwrap());
        assert_ne!(plan.nums, null_plan(&p.slice.rows, 3600.0, 19, 8).unwrap().nums);
        assert_eq!(null_plan(&p.slice.rows, 3600.0, 287, 1).unwrap().nums.len(), 287);
        let e = null_plan(&p.slice.rows, 3600.0, 288, 1).unwrap_err();
        assert!(e.contains("287 distinct shifts"), "{e}");
        assert_eq!(p_family(0.5, &[0.6, 0.7, 0.5, 0.4]), 3.0 / 5.0);
    }

    /// The null holds its level: on signal-free panels with 19 shared shifts,
    /// `p_family <= 0.05` in at most 18 of 200 panels (the 99 % point of
    /// Binomial(200, 0.05)). Run by hand: it prints the count.
    #[test]
    #[ignore]
    fn family_null_level_on_signal_free_panels() {
        let class = Class::build(3, &ops("add,sub,mul,abs"), 3, true).unwrap();
        let mut hits = 0;
        for k in 0..200u64 {
            let p = panel(10_000 + k, 4, 300, 3, noise);
            let b = book(&p, 1);
            let params = BookParams { spec: &spec(), horizon_seconds: 3600.0, blocks: 1, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
            let plan = null_plan(&p.slice.rows, 3600.0, 19, k as u32).unwrap();
            let shifted: Vec<FitnessBook> = plan.nums.iter().map(|&s| book_of(&shifted_slice(&p.slice, s, plan.periods as u64).unwrap(), &params).unwrap()).collect();
            let mut books: Vec<&FitnessBook> = vec![&b];
            books.extend(shifted.iter());
            let pass = pass(&Scan { class: &class, cols: &p.cols, rows: 1200, cuts: 16, threads: 4 }, &books, false);
            if p_family(pass.best[0], &pass.best[1..]) <= 0.05 {
                hits += 1;
            }
        }
        println!("family null level: p_family <= 0.05 in {hits} of 200 signal-free panels (19 shifts)");
        assert!(hits <= 18, "{hits} of 200");
    }

    /// CPU seconds at the campaign training shape (306 groups x 7,403 hours, 32
    /// variables, 4 blocks, the ten campaign operators) on 4 threads. Run by hand:
    /// it prints the seconds and asserts no time.
    #[test]
    #[ignore]
    fn enumerate_cost_at_the_campaign_shape() {
        let (groups, steps, nvars) = (306usize, 7403usize, 32usize);
        let n = groups * steps;
        let mut rng = Rng(268);
        let cols: Vec<Vec<f64>> = (0..nvars).map(|_| (0..n).map(|_| rng.gauss()).collect()).collect();
        let (mut time, mut group, mut ret) = (Vec::with_capacity(n), Vec::with_capacity(n), Vec::with_capacity(n));
        for g in 0..groups {
            for k in 0..steps {
                time.push(3600.0 * k as f64);
                group.push(g as u64);
                ret.push(0.004 * rng.gauss() + 0.0005 * cols[0][g * steps + k]);
            }
        }
        let slice = TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret };
        let params = BookParams { spec: &spec(), horizon_seconds: 3600.0, blocks: 4, lambda: 1.0, stat: BlockStat::MeanSd, segments: &[] };
        let b = book_of(&slice, &params).unwrap();
        let o = ops(OPS10);
        for max in [3usize, 4] {
            let class = Class::build(nvars, &o, max, true).unwrap();
            let start = std::time::Instant::now();
            let p = pass(&Scan { class: &class, cols: &cols, rows: n, cuts: 16, threads: 4 }, &[&b], true);
            let secs = start.elapsed().as_secs_f64();
            println!(
                "enumerate cost: max size {max}: {} trees, {} distinct books, {n} rows, 4 blocks, 16 cuts, 1 book, 4 threads: {secs:.1} s wall",
                class.scored().len(),
                p.books.len()
            );
        }
        let plan = null_plan(&slice.rows, 3600.0, 19, 1).unwrap();
        let shifted: Vec<FitnessBook> = plan.nums.iter().map(|&s| book_of(&shifted_slice(&slice, s, plan.periods as u64).unwrap(), &params).unwrap()).collect();
        let mut books: Vec<&FitnessBook> = vec![&b];
        books.extend(shifted.iter());
        let class = Class::build(nvars, &o, 2, true).unwrap();
        for k in [1usize, 20] {
            let start = std::time::Instant::now();
            pass(&Scan { class: &class, cols: &cols, rows: n, cuts: 16, threads: 4 }, &books[..k], false);
            println!("enumerate cost: max size 2: {} trees, {k} book(s), 4 threads: {:.1} s wall", class.scored().len(), start.elapsed().as_secs_f64());
        }
    }
}
