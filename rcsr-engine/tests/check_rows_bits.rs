//! #1051: the exact row decisions of the device book check, on the host mirror.
//!
//! A slot the row decisions settle gives bit planes (enter, stay) that equal the
//! planes of the CPU prediction vector. A book fed from the planes, through the
//! synthetic prediction that has exactly those planes, gives the CPU totals,
//! value, book shape and signature bit for bit.

use rcsr::data::TradingSlice;
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::flat_bound::rows::{compile_check_program, settle_slot, SlotState, Thresholds};
use rcsr::trading::{BandSpec, BandUnits, FitnessBook, TradingRows, TradingSpec};

const VARS: usize = 4;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

struct Fixture {
    n: usize,
    cols: Vec<Vec<f64>>,
    slice: TradingSlice,
}

/// 90,000 rows in 8 groups. Column 0 is the signal, column 1 and 2 are wide
/// uniform, column 3 reaches the `exp` guard on some rows.
fn fixture(n: usize, groups: usize, seed: u64) -> Fixture {
    let mut rng = Lcg(seed);
    let per = n / groups;
    let mut rows = TradingRows::default();
    let mut ret = Vec::with_capacity(n);
    let mut cols = vec![Vec::with_capacity(n); VARS];
    for i in 0..n {
        let g = (i / per).min(groups - 1);
        rows.group.push(g as u64);
        rows.time.push(1.6e9 + 900.0 * (i - g * per) as f64);
        let signal = rng.unit() - 0.5;
        cols[0].push(signal);
        cols[1].push(rng.unit() * 4.0 - 2.0);
        cols[2].push(rng.unit() * 4.0 - 2.0);
        cols[3].push(if rng.below(50) == 0 { (rng.unit() - 0.2) * 900.0 } else { rng.unit() * 6.0 - 3.0 });
        ret.push(0.004 * signal + 0.003 * (rng.unit() - 0.5));
    }
    Fixture { n, cols, slice: TradingSlice { rows, ret } }
}

fn book(f: &Fixture, band: f64, hold: usize, min_trades: usize, blocks: usize) -> FitnessBook {
    let spec = TradingSpec {
        cost_bps: 2.0,
        min_trades,
        band: BandSpec { units: BandUnits::Score, resolved: band, ..BandSpec::score(band) },
        min_holding: hold,
        ..Default::default()
    };
    FitnessBook::with_blocks(&f.slice, &spec, 900.0, blocks, 1.0).expect("a profitable fixture book")
}

fn un(op: Op, a: Node) -> Node {
    node(op, vec![a])
}

fn bin(op: Op, a: Node, b: Node) -> Node {
    node(op, vec![a, b])
}

fn random_tree(r: &mut Lcg, depth: usize) -> Node {
    if depth == 0 || r.below(4) == 0 {
        return if r.below(3) == 0 {
            constant(((r.unit() * 8.0).floor() - 4.0) / 2.0 + 0.25)
        } else {
            input(r.below(VARS))
        };
    }
    let k = |r: &mut Lcg| random_tree(r, depth - 1);
    match r.below(15) {
        0 => un(Op::Neg, k(r)),
        1 => un(Op::Abs, k(r)),
        2 => un(Op::Square, k(r)),
        3 => un(Op::Exp, k(r)),
        4 => un(Op::Tanh, k(r)),
        5 => un(Op::Logistic, k(r)),
        6 => un(Op::Gauss, k(r)),
        7 => bin(Op::Add, k(r), k(r)),
        8 => bin(Op::Sub, k(r), k(r)),
        9 => bin(Op::Mul, k(r), k(r)),
        10 => bin(Op::Div, k(r), k(r)),
        11 => bin(Op::Min, k(r), k(r)),
        12 => bin(Op::Max, k(r), k(r)),
        13 => bin(Op::Greater, k(r), k(r)),
        _ => node(Op::If, vec![k(r), k(r), k(r)]),
    }
}

fn named_trees() -> Vec<Node> {
    let e = |x: Node| un(Op::Exp, x);
    vec![
        input(0),
        bin(Op::Mul, constant(0.75), input(0)),
        bin(Op::Sub, un(Op::Tanh, input(1)), constant(0.05)),
        bin(Op::Mul, constant(0.3), bin(Op::Add, input(0), input(2))),
        un(Op::Gauss, bin(Op::Mul, constant(1e6), input(0))),
        un(Op::Logistic, bin(Op::Sub, input(3), constant(3423.03))),
        bin(Op::Sub, input(1), input(1)),
        e(e(input(0))),
        bin(Op::Sub, un(Op::Logistic, bin(Op::Sub, e(input(3)), e(input(2)))), constant(2.0)),
        bin(Op::Sub, constant(0.0), bin(Op::Mul, e(input(3)), e(input(3)))),
        un(Op::Tanh, bin(Op::Div, input(0), bin(Op::Sub, input(1), input(1)))),
        node(Op::If, vec![bin(Op::Greater, input(0), constant(0.1)), input(0), constant(-1.0)]),
        bin(Op::Min, input(0), bin(Op::Max, input(1), constant(0.2))),
        bin(Op::Sub, un(Op::Logistic, bin(Op::Mul, constant(8.0), input(0))), constant(0.5)),
        constant(-1.0),
        constant(1.0),
    ]
}

/// The synthetic prediction with the planes of `enter` and `stay` and no
/// non-finite row: every pricing function reads a prediction through the same
/// two comparisons, so it prices this vector as the plane-fed device book does.
fn synthetic(planes_enter: &[u32], planes_stay: &[u32], n: usize, t: Thresholds) -> Vec<f64> {
    (0..n)
        .map(|row| {
            let (e, s) = ((planes_enter[row >> 5] >> (row & 31)) & 1, (planes_stay[row >> 5] >> (row & 31)) & 1);
            assert!(e <= s, "enter implies stay");
            match (e, s) {
                (1, _) => f64::MAX,
                (0, 1) => t.enter,
                _ => f64::MIN,
            }
        })
        .collect()
}

#[derive(Default, Debug)]
struct Counts {
    settled: usize,
    inf: usize,
    unsettled: usize,
    undecided_rows: usize,
    compared: usize,
}

fn run(f: &Fixture, trees: &[Node], pairs: &[(f64, usize)], floors: &[usize], blocks: &[usize]) -> Counts {
    let mut counts = Counts::default();
    for tree in trees {
        let Some(words) = compile_check_program(tree, VARS) else {
            panic!("not encodable: {tree:?}");
        };
        let pred: Vec<f64> = (0..f.n)
            .map(|row| {
                let x: Vec<f64> = f.cols.iter().map(|c| c[row]).collect();
                rcsr::expr::eval(tree, &x)
            })
            .collect();
        for &(band, hold) in pairs {
            let probe = book(f, band, hold, 0, 1);
            let t = Thresholds::new(probe.threshold(), band).unwrap();
            let planes = settle_slot(&words, &f.cols, f.n, t, |row| pred[row]);
            counts.undecided_rows += planes.undecided.len();
            match planes.state {
                SlotState::Unsettled => counts.unsettled += 1,
                SlotState::Inf => {
                    counts.inf += 1;
                    assert!(pred.iter().any(|p| !p.is_finite()), "{tree:?}: Inf slot with an all-finite prediction");
                }
                SlotState::Settled => {
                    counts.settled += 1;
                    assert!(pred.iter().all(|p| p.is_finite()), "{tree:?}: settled slot with a non-finite row");
                    let synth = synthetic(&planes.enter, &planes.stay, f.n, t);
                    for &floor in floors {
                        for &nb in blocks {
                            let b = book(f, band, hold, floor, nb);
                            let (want, shape, sig) = b.regret_shape_signed(&pred);
                            let (got, gshape, gsig) = b.regret_shape_signed(&synth);
                            let what = format!("{tree:?} band {band} hold {hold} floor {floor} blocks {nb}");
                            assert_eq!(got.to_bits(), want.to_bits(), "value {what}");
                            assert_eq!(gshape, shape, "shape {what}");
                            assert_eq!(gsig, sig, "signature {what}");
                            let (w, g) = (b.band_pass(&pred, band, nb > 1), b.band_pass(&synth, band, nb > 1));
                            for (k, (a, c)) in w.iter().zip(&g).enumerate() {
                                assert_eq!(c.gross.to_bits(), a.gross.to_bits(), "gross block {k} {what}");
                                assert_eq!((c.flips, c.exits, c.long_rows), (a.flips, a.exits, a.long_rows), "counts block {k} {what}");
                                assert!(!c.nonfinite && !a.nonfinite);
                            }
                            counts.compared += 1;
                        }
                    }
                }
            }
        }
    }
    counts
}

#[test]
fn settled_planes_price_to_the_cpu_totals() {
    let f = fixture(90_000, 8, 17);
    let mut r = Lcg(0xdead_beef);
    let mut trees = named_trees();
    while trees.len() < 56 {
        let t = random_tree(&mut r, 4);
        if compile_check_program(&t, VARS).is_some() {
            trees.push(t);
        }
    }
    let pairs = [(0.0, 12), (0.0, 24), (0.0, 2), (1e-3, 8), (0.0, 1)];
    let c = run(&f, &trees, &pairs, &[0, 1, 64], &[1, 8]);
    println!("{c:?}");
    assert!(c.compared >= 200, "{c:?}");
    assert!(c.inf > 0 && c.settled > 0, "{c:?}");
}

fn size(n: &Node) -> usize {
    1 + n.kids.iter().map(size).sum::<usize>()
}

/// Settle share on check-like slots (CPU score `+inf`: a non-finite row or fewer
/// than 500 flips) of random trees up to 12 nodes, band 0, hold 12.
#[test]
#[ignore]
fn settle_share_band_hold_12_floor_500() {
    let f = fixture(90_000, 8, 17);
    let b = book(&f, 0.0, 12, 500, 8);
    let t = Thresholds::new(b.threshold(), 0.0).unwrap();
    let mut r = Lcg(0x5eed);
    let (mut total, mut inf, mut settled, mut unsettled) = (0usize, 0usize, 0usize, 0usize);
    let mut undecided: Vec<usize> = Vec::new();
    while total < 600 {
        let tree = random_tree(&mut r, 4);
        if size(&tree) > 12 || compile_check_program(&tree, VARS).is_none() {
            continue;
        }
        let pred: Vec<f64> = (0..f.n)
            .map(|row| rcsr::expr::eval(&tree, &f.cols.iter().map(|c| c[row]).collect::<Vec<_>>()))
            .collect();
        if b.regret(&pred).is_finite() {
            continue;
        }
        total += 1;
        let words = compile_check_program(&tree, VARS).unwrap();
        let p = settle_slot(&words, &f.cols, f.n, t, |row| pred[row]);
        undecided.push(p.undecided.len());
        match p.state {
            SlotState::Inf => inf += 1,
            SlotState::Settled => settled += 1,
            SlotState::Unsettled => unsettled += 1,
        }
    }
    undecided.sort_unstable();
    let q = |p: usize| undecided[(undecided.len() - 1) * p / 100];
    println!(
        "slots {total} inf {inf} settled {settled} unsettled {unsettled} share {:.3}; undecided rows p50 {} p90 {} p99 {} max {}",
        (inf + settled) as f64 / total as f64,
        q(50),
        q(90),
        q(99),
        undecided[undecided.len() - 1]
    );
}
