//! #471: structural edits, arity-balanced trees, internal crossover cuts and
//! the adaptive edit mix. Every new setting is opt-in; the default archive is
//! the pre-change archive bit for bit.

use rcsr::data::Sample;
use rcsr::expr::{
    self, complexity, constant, input, mutate_with_opts, node, random_tree_opts, EditKind, Node,
    Op, VariationOpts, CLASSIC_MIX, EDIT_KINDS,
};
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::rng::Mt;
use rcsr::search::{self, Config, MutationMix, Outcome, TreeArity};

fn only(kind: EditKind) -> VariationOpts {
    let mut mix = [0.0; EDIT_KINDS];
    mix[kind.index()] = 1.0;
    VariationOpts { mix, ..VariationOpts::default() }
}

fn ops() -> Vec<Op> {
    vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Pow,
        Op::Sin,
        Op::Exp,
        Op::Neg,
        Op::Square,
        Op::If,
    ]
}

/// Leaves in pre-order, as printable tokens.
fn leaves(n: &Node, out: &mut Vec<String>) {
    if n.kids.is_empty() {
        out.push(format!("{:?}:{}", n.op, n.val.to_bits()));
    }
    for k in &n.kids {
        leaves(k, out);
    }
}

fn leaf_list(n: &Node) -> Vec<String> {
    let mut v = Vec::new();
    leaves(n, &mut v);
    v
}

fn is_subsequence(small: &[String], big: &[String]) -> bool {
    let mut it = big.iter();
    small.iter().all(|s| it.any(|b| b == s))
}

fn op_count(n: &Node, op: Op) -> usize {
    usize::from(n.op == op) + n.kids.iter().map(|k| op_count(k, op)).sum::<usize>()
}

fn valid(n: &Node) -> bool {
    n.kids.len() == n.op.arity() && n.kids.iter().all(valid)
}

/// The first pre-order node where `a` and `b` differ: in operator or arity, with
/// its operands exchanged, or deeper.
fn first_diff<'a>(a: &'a Node, b: &'a Node) -> (&'a Node, &'a Node) {
    if a.op != b.op || a.val.to_bits() != b.val.to_bits() || a.kids.len() != b.kids.len() {
        return (a, b);
    }
    if a.kids.len() == 2 && a.kids[0] == b.kids[1] && a.kids[1] == b.kids[0] && a.kids[0] != a.kids[1] {
        return (a, b);
    }
    for (x, y) in a.kids.iter().zip(&b.kids) {
        if x != y {
            return first_diff(x, y);
        }
    }
    (a, b)
}

fn parents() -> Vec<Node> {
    let x1 = || input(0);
    let x2 = || input(1);
    vec![
        // sub(div(x1, x2), pow(x2, 2)): swap, rotate, insert and delete sites
        node(
            Op::Sub,
            vec![
                node(Op::Div, vec![x1(), x2()]),
                node(Op::Pow, vec![x2(), constant(2.0)]),
            ],
        ),
        node(
            Op::Add,
            vec![x1(), node(Op::Mul, vec![x2(), node(Op::Sin, vec![x1()])])],
        ),
        node(Op::Div, vec![node(Op::Sin, vec![x1()]), constant(3.0)]),
    ]
}

#[test]
fn mutate_insert_wraps_site() {
    let ops = ops();
    for seed in 0..1000u32 {
        let p = parents()[seed as usize % 3].clone();
        let mut rng = Mt::new(seed + 1);
        let mut c = p.clone();
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Insert));
        assert_eq!(kind, Some(EditKind::Insert), "seed {seed}");
        assert!(valid(&c), "seed {seed}: {c:?}");
        assert!(c != p, "seed {seed}: unchanged");
        assert!(complexity(&c) > complexity(&p), "seed {seed}");
        assert!(
            is_subsequence(&leaf_list(&p), &leaf_list(&c)),
            "seed {seed}: the parent's leaves must survive in order"
        );
    }
}

#[test]
fn insert_never_wraps_neg_in_neg() {
    let only_neg = [Op::Neg];
    let p = node(Op::Neg, vec![input(0)]);
    let want = node(Op::Neg, vec![node(Op::Neg, vec![input(0)])]);
    for seed in 0..200u32 {
        let mut c = p.clone();
        let mut rng = Mt::new(seed + 1);
        let kind = mutate_with_opts(&mut c, &mut rng, 1, &only_neg, None, &only(EditKind::Insert));
        assert_eq!(kind, Some(EditKind::Insert), "seed {seed}");
        assert_eq!(c, want, "seed {seed}: only the leaf is a site");
    }
}

#[test]
fn mutate_delete_hoists_child() {
    let ops = ops();
    for seed in 0..1000u32 {
        let p = parents()[seed as usize % 3].clone();
        let mut rng = Mt::new(seed + 1);
        let mut c = p.clone();
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Delete));
        assert_eq!(kind, Some(EditKind::Delete), "seed {seed}");
        assert!(valid(&c), "seed {seed}");
        assert!(c != p, "seed {seed}");
        assert!(complexity(&c) < complexity(&p), "seed {seed}");
        assert!(
            is_subsequence(&leaf_list(&c), &leaf_list(&p)),
            "seed {seed}: the child's leaves come from the parent"
        );
    }
}

#[test]
fn delete_on_a_single_node_is_not_a_site() {
    let ops = ops();
    let p = input(0);
    for seed in 0..100u32 {
        let mut c = p.clone();
        let mut rng = Mt::new(seed + 1);
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Delete));
        assert_ne!(kind, Some(EditKind::Delete));
        assert!(c != p, "the contract: a mutation never returns the parent");
    }
}

#[test]
fn mutate_swap_only_noncommutative() {
    let ops = ops();
    // add(x1, mul(x2, x1)) has only commutative binary nodes.
    let commutative = parents()[1].clone();
    for seed in 0..1000u32 {
        let mut c = commutative.clone();
        let mut rng = Mt::new(seed + 1);
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Swap));
        assert_ne!(kind, Some(EditKind::Swap), "seed {seed}: no site exists");
        assert!(c != commutative);
    }
    for seed in 0..1000u32 {
        let p = parents()[if seed % 2 == 0 { 0 } else { 2 }].clone();
        let mut c = p.clone();
        let mut rng = Mt::new(seed + 1);
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Swap));
        assert_eq!(kind, Some(EditKind::Swap), "seed {seed}");
        assert!(valid(&c));
        assert!(c != p);
        assert_eq!(complexity(&c), complexity(&p));
        let (a, b) = first_diff(&p, &c);
        assert!(matches!(a.op, Op::Sub | Op::Div | Op::Pow), "swapped a {:?}", a.op);
        assert_eq!(a.op, b.op);
        assert_eq!(a.kids[0], b.kids[1]);
        assert_eq!(a.kids[1], b.kids[0]);
    }
}

#[test]
fn mutate_rotate_preserves_leaves() {
    let ops = ops();
    let sites = [
        parents()[0].clone(),
        parents()[1].clone(),
        node(Op::Sub, vec![input(0), node(Op::Div, vec![input(1), constant(2.0)])]),
    ];
    for seed in 0..1000u32 {
        let p = sites[seed as usize % 3].clone();
        let mut c = p.clone();
        let mut rng = Mt::new(seed + 1);
        let kind = mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &only(EditKind::Rotate));
        assert_eq!(kind, Some(EditKind::Rotate), "seed {seed}");
        assert!(valid(&c));
        assert!(c != p);
        assert_eq!(complexity(&c), complexity(&p));
        assert_eq!(leaf_list(&c), leaf_list(&p), "seed {seed}: leaf order must not change");
        for op in [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Pow] {
            assert_eq!(op_count(&c, op), op_count(&p, op), "seed {seed}: operator multiset");
        }
    }
}

#[test]
fn extended_mix_always_changes_the_tree() {
    let ops = ops();
    let opts = VariationOpts {
        mix: MutationMix::Extended.weights(),
        ..VariationOpts::default()
    };
    for seed in 0..1000u32 {
        let p = parents()[seed as usize % 3].clone();
        let mut c = p.clone();
        let mut rng = Mt::new(seed + 1);
        mutate_with_opts(&mut c, &mut rng, 2, &ops, None, &opts);
        assert!(c != p && valid(&c), "seed {seed}");
    }
}

#[test]
fn balanced_arity_frequencies() {
    let ops = rcsr::search::Config::default().ops;
    let mut rng = Mt::new(17);
    let (mut binary, mut internal) = (0usize, 0usize);
    fn count(n: &Node, binary: &mut usize, internal: &mut usize) {
        if !n.kids.is_empty() {
            *internal += 1;
            if n.kids.len() == 2 {
                *binary += 1;
            }
        }
        for k in &n.kids {
            count(k, binary, internal);
        }
    }
    for _ in 0..100_000 {
        let t = random_tree_opts(&mut rng, 3, &ops, 1, None, true);
        count(&t, &mut binary, &mut internal);
    }
    let share = binary as f64 / internal as f64;
    assert!((share - 0.50).abs() <= 0.01, "binary share {share} over {internal} internal nodes");

    let mut rng = Mt::new(17);
    let (mut b2, mut i2) = (0usize, 0usize);
    for _ in 0..100_000 {
        let t = random_tree_opts(&mut rng, 3, &ops, 1, None, false);
        count(&t, &mut b2, &mut i2);
    }
    let uniform = b2 as f64 / i2 as f64;
    assert!(uniform < 0.40, "the uniform draw is far from balanced: {uniform}");
}

#[test]
fn balanced_without_unary_operators_still_builds() {
    let ops = [Op::Add, Op::Mul];
    let mut rng = Mt::new(3);
    for _ in 0..200 {
        let t = random_tree_opts(&mut rng, 2, &ops, 3, None, true);
        assert!(valid(&t));
    }
}

#[test]
fn internal_crossover_cuts_internal_nodes() {
    let b = node(Op::Add, vec![input(0), node(Op::Sin, vec![input(1)])]);
    // p = 1: every cut point is internal, so the donor is never a bare leaf
    // and the receiver's graft site is never a leaf of `a`.
    let a = node(Op::Mul, vec![node(Op::Exp, vec![input(0)]), constant(2.0)]);
    let mut rng = Mt::new(9);
    let mut uniform_leaf = false;
    for _ in 0..500 {
        let mut c = a.clone();
        expr::crossover_with_opts(&mut c, &b, &mut rng, None, 1.0);
        assert!(valid(&c));
        assert!(complexity(&c) >= 2, "an internal graft cannot leave a bare leaf");
        let mut u = a.clone();
        expr::crossover_with_opts(&mut u, &b, &mut rng, None, 0.0);
        uniform_leaf |= complexity(&u) == 1;
    }
    assert!(uniform_leaf, "the uniform cut reaches bare leaves");
}

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2);
    }
    Sample::new(x, y)
}

fn archive_hash(out: &Outcome) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for s in &out.archive.items {
        eat(&s.error.to_bits().to_le_bytes());
        eat(&(s.complexity as u64).to_le_bytes());
        eat(expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]).as_bytes());
    }
    h
}

fn config(strategy: FitStrategy) -> Config {
    Config {
        generations: 12,
        islands: 2,
        pop: 60,
        max_complexity: 14,
        fit_iters: 8,
        patience: 0,
        seed: 5,
        fit_strategy: strategy,
        metric: Metric::Squared,
        gpu: false,
        gpu_notes: false,
        ..Default::default()
    }
}

/// Archive hash of the pre-change CPU path for `config` on `sample(300, 5)`.
const GOLDEN: &[(FitStrategy, u64)] = &[
    (FitStrategy::Exact, 0xbd59_e38d_0c07_2a90),
    (FitStrategy::Auto, 0xdb2e_af98_9cca_2d6b),
];

#[test]
fn classic_mix_is_bit_identical() {
    assert_eq!(CLASSIC_MIX, [0.25, 0.35, 0.40, 0.0, 0.0, 0.0, 0.0]);
    let train = sample(300, 5);
    let test = sample(100, 6);
    for (strategy, want) in GOLDEN {
        let cfg = config(*strategy);
        assert_eq!(cfg.mutation_mix, MutationMix::Classic);
        assert_eq!(cfg.tree_arity, TreeArity::Uniform);
        assert_eq!(cfg.crossover_internal, 0.0);
        assert!(!cfg.adaptive_mix);
        let h = archive_hash(&search::run(&cfg, &train, &test));
        assert_eq!(h, *want, "{strategy:?}: archive differs from the pre-change path ({h:#018x})");
    }
}

fn adaptive_config() -> Config {
    Config {
        mutation_mix: MutationMix::Extended,
        tree_arity: TreeArity::Balanced,
        crossover_internal: 0.9,
        adaptive_mix: true,
        generations: 20,
        ..config(FitStrategy::Exact)
    }
}

fn last_checkpoint(cfg: &Config, train: &Sample, test: &Sample) -> (u64, Vec<Vec<u64>>, String) {
    let mut last = None;
    let out = search::run_resumable(cfg, train, test, None, 20, |cp| {
        last = Some(cp.to_text());
    })
    .expect("run");
    let text = last.expect("a checkpoint was taken");
    let cp = rcsr::checkpoint::Checkpoint::from_text(&text).expect("parse");
    let mixes = cp.islands.iter().map(|i| i.mix.encode()).collect();
    (archive_hash(&out), mixes, text)
}

#[test]
fn adaptive_mix_is_deterministic() {
    let train = sample(200, 5);
    let test = sample(80, 6);
    let cfg = adaptive_config();
    let (h1, m1, _) = last_checkpoint(&cfg, &train, &test);
    let (h2, m2, _) = last_checkpoint(&cfg, &train, &test);
    assert_eq!(h1, h2, "archive");
    assert_eq!(m1, m2, "counts after 20 generations");
    for (i, m) in m1.iter().enumerate() {
        assert_eq!(m.len(), 10 * 2 * EDIT_KINDS, "island {i}: a full window of 10 generations");
        assert!(m.iter().any(|&v| v > 0), "island {i}: counts were recorded");
    }
    let mut threaded = cfg.clone();
    threaded.threads = 3;
    let (h3, m3, _) = last_checkpoint(&threaded, &train, &test);
    assert_eq!(h1, h3, "archive differs with --threads 3");
    assert_eq!(m1, m3, "counts differ with --threads 3");
}

#[test]
fn adaptive_mix_survives_a_checkpoint_resume() {
    let train = sample(200, 5);
    let test = sample(80, 6);
    let full = adaptive_config();
    let straight = search::run(&full, &train, &test);

    let mut half = full.clone();
    half.generations = 10;
    let mut text = None;
    search::run_resumable(&half, &train, &test, None, 10, |cp| text = Some(cp.to_text()))
        .expect("first half");
    let cp = rcsr::checkpoint::Checkpoint::from_text(&text.expect("checkpoint")).expect("parse");
    assert!(cp.config.adaptive_mix && cp.config.mutation_mix == MutationMix::Extended);
    assert_eq!(cp.config.crossover_internal, 0.9);
    let resumed = search::run_resumable(&full, &train, &test, Some(cp), 20, |_| {}).expect("resume");
    assert_eq!(archive_hash(&straight), archive_hash(&resumed));
}

#[test]
fn a_resume_refuses_a_changed_edit_setting() {
    let train = sample(100, 5);
    let test = sample(40, 6);
    let mut cfg = config(FitStrategy::Exact);
    cfg.generations = 2;
    cfg.mutation_mix = MutationMix::Extended;
    let mut text = None;
    search::run_resumable(&cfg, &train, &test, None, 2, |cp| text = Some(cp.to_text())).unwrap();
    let cp = rcsr::checkpoint::Checkpoint::from_text(&text.unwrap()).unwrap();
    let mut other = cfg.clone();
    other.mutation_mix = MutationMix::Classic;
    other.generations = 3;
    let err = search::run_resumable(&other, &train, &test, Some(cp), 1, |_| {}).err().expect("a changed edit setting is refused");
    assert!(err.contains("mutation_mix"), "{err}");
}
