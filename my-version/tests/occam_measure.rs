//! #1184: measurement tests of the Occam rule on synthetic panels (CPU, ignored).
//!
//! Run by hand: `cargo test --release --test occam_measure -- --ignored --nocapture`.
//! Panel generator: the 8-feature, 192-rule design of the Fable gate experiment, at 5 symbols and
//! 6,000 periods.

use rcsr::occam::{select_occam, Bits, OccamParams, Scored};
use rcsr::selection::RobustMeta;
use rcsr::trading::{blocked_alpha, TradingRows, TradingSpec};

const S: usize = 5;
const T: usize = 6000;
const NF: usize = 8;
const NCUT: usize = 12;
const SINGLE_BITS: f64 = 8.58;
const COMPOSITE_BITS: f64 = 22.18;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    fn uni(&mut self) -> f64 {
        ((self.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    }
    fn norm(&mut self) -> f64 {
        let (u, v) = (self.uni(), self.uni());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    }
}

fn ar1(rng: &mut Rng, n: usize, rho: f64) -> Vec<f64> {
    let s = (1.0 - rho * rho).sqrt();
    let mut x = rng.norm();
    (0..n)
        .map(|_| {
            x = rho * x + s * rng.norm();
            x
        })
        .collect()
}

struct Panel {
    ret: Vec<f64>,
    feat: Vec<Vec<f64>>,
    gate: Vec<f64>,
}

/// `signal` is the loading of the return on feature 0.
fn panel(rng: &mut Rng, signal: f64) -> Panel {
    let gate = ar1(rng, T, 0.9);
    let common = ar1(rng, T, 0.9);
    let mut feat: Vec<Vec<f64>> = (0..NF).map(|_| vec![0.0; S * T]).collect();
    let mut ret = vec![0.0; S * T];
    for s in 0..S {
        let x1 = ar1(rng, T, 0.97);
        for f in 0..NF {
            let v = if f == 0 { x1.clone() } else if f == 1 { common.clone() } else { ar1(rng, T, 0.9) };
            feat[f][s * T..(s + 1) * T].copy_from_slice(&v);
        }
        for t in 0..T {
            ret[s * T + t] = 0.008 * rng.norm() + signal * x1[t];
        }
    }
    Panel { ret, feat, gate }
}

fn rows() -> TradingRows {
    TradingRows {
        time: (0..S * T).map(|i| (i % T) as f64).collect(),
        group: (0..S * T).map(|i| (i / T) as u64).collect(),
        funding: Vec::new(),
    }
}

struct Rule {
    f: usize,
    up: bool,
    cut: f64,
}

fn rules(p: &Panel) -> Vec<Rule> {
    let mut out = Vec::new();
    for f in 0..NF {
        let mut s = p.feat[f].clone();
        s.sort_by(|a, b| a.total_cmp(b));
        let q = |share: f64| s[((s.len() - 1) as f64 * share).round() as usize];
        for j in 0..NCUT {
            let share = 0.5 * 0.7f64.powi(j as i32);
            out.push(Rule { f, up: true, cut: q(1.0 - share) });
            out.push(Rule { f, up: false, cut: q(share) });
        }
    }
    out
}

fn pred(r: &Rule, p: &Panel) -> Vec<f64> {
    p.feat[r.f].iter().map(|&x| if (x > r.cut) == r.up { 1.0 } else { -1.0 }).collect()
}

fn word(pred: &[f64], p: &Panel, rows: &TradingRows, bits: f64) -> Option<Scored> {
    let a = blocked_alpha(pred, &p.ret, rows, &TradingSpec::default(), 1).ok()?;
    let hold = if a.entries == 0 { 1 } else { a.long_rows.div_ceil(a.entries) };
    Some(Scored {
        tier: 0,
        expression_exact: String::new(),
        bits: Bits { structure: bits, constants: 0.0, total: bits, nodes: 1, variables: 1, constant_count: 0 },
        series: a.series,
        entries: a.entries,
        hold,
    })
}

fn meta(i: usize) -> RobustMeta {
    RobustMeta { archive_rank: i + 1, complexity: 1.0, usable: true, objective: 0.0 }
}

fn code() -> rcsr::occam::CodeSpec {
    rcsr::occam::CodeSpec::v1(&[rcsr::expr::Op::Add], NF)
}

#[test]
#[ignore = "measurement: 300 signal-free panels, minutes of CPU"]
fn level_on_signal_free_panels() {
    let (mut rng, rows) = (Rng(11), rows());
    let mut certified = 0;
    for _ in 0..300 {
        let p = panel(&mut rng, 0.0);
        let books: Vec<Option<Scored>> = rules(&p).iter().map(|r| word(&pred(r, &p), &p, &rows, SINGLE_BITS)).collect();
        let metas: Vec<RobustMeta> = (0..books.len()).map(meta).collect();
        let sel = select_occam(&metas, OccamParams::default(), &code(), &mut |i| Ok(books[i].iter().cloned().collect())).unwrap();
        certified += usize::from(sel.chosen.is_some());
    }
    println!("certified {certified} of 300");
    assert!(certified <= 24, "certified {certified} of 300");
}

#[test]
#[ignore = "measurement: 100 one-rule panels, minutes of CPU"]
fn parsimony_on_the_one_rule_panel() {
    let (mut rng, rows) = (Rng(23), rows());
    let mut single = 0;
    let mut certified = 0;
    for _ in 0..100 {
        let p = panel(&mut rng, 0.0006);
        let rs = rules(&p);
        let preds: Vec<Vec<f64>> = rs.iter().map(|r| pred(r, &p)).collect();
        let mut books: Vec<Option<Scored>> = preds.iter().map(|q| word(q, &p, &rows, SINGLE_BITS)).collect();
        let singles = books.len();
        let mut order: Vec<usize> = (0..singles).collect();
        let z = |i: usize| books[i].as_ref().map_or(f64::NEG_INFINITY, |w| rcsr::occam::evidence(&w.series, w.hold).ok().and_then(|e| e.z).unwrap_or(f64::NEG_INFINITY));
        order.sort_by(|&a, &b| z(b).total_cmp(&z(a)));
        let top: Vec<usize> = order[..6].to_vec();
        let med = {
            let mut g = p.gate.clone();
            g.sort_by(|a, b| a.total_cmp(b));
            g[T / 2]
        };
        for &a in &top {
            for &b in &top {
                if a == b {
                    continue;
                }
                let q: Vec<f64> = (0..S * T).map(|i| if p.gate[i % T] > med { preds[a][i] } else { preds[b][i] }).collect();
                books.push(word(&q, &p, &rows, COMPOSITE_BITS));
            }
        }
        let metas: Vec<RobustMeta> = (0..books.len()).map(meta).collect();
        let sel = select_occam(&metas, OccamParams::default(), &code(), &mut |i| Ok(books[i].iter().cloned().collect())).unwrap();
        if let Some(c) = sel.chosen {
            certified += 1;
            single += usize::from(c < singles);
        }
    }
    println!("certified {certified} of 100, single-feature finalists {single}");
    assert!(single >= 95, "single-feature finalists {single} of 100 (certified {certified})");
}
