//! #116: fit-arm timing harness (ignored by default; run with `--ignored`).
//!
//! ```sh
//! cargo test --release --test fit_portfolio_bench -- --ignored --nocapture
//! ```
//!
//! It measures the cells the issue asks about that are cheap enough for a
//! shared host: affine QR versus exact LM for p = 2..16, VarPro versus LM for
//! a separable 3-constant tree, and streaming versus resident LM for a large
//! generated row source. Times are wall-clock seconds, median of three.

use rcsr::expr::{constant, input, node, Op};
use rcsr::fit::{
    fit_stream_with, fit_weighted, fit_weighted_strategy, FitStrategy, RowSource,
};
use rcsr::expr::Node;
use std::time::Instant;

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn affine_tree(p: usize) -> Node {
    let mut sum = constant(0.1);
    let mut power = input(0);
    for j in 0..p {
        if j > 0 {
            power = node(Op::Mul, vec![power, input(0)]);
        }
        sum = node(Op::Add, vec![sum, node(Op::Mul, vec![constant(0.1 * (j + 1) as f64), power.clone()])]);
    }
    sum
}

fn affine_rows(n: usize) -> (Vec<Vec<f64>>, Vec<f64>) {
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![-0.8 + 1.6 * i as f64 / (n - 1) as f64]).collect();
    let y: Vec<f64> = x
        .iter()
        .map(|r| {
            let t = r[0];
            let mut s = 0.1;
            let mut power = 1.0;
            for j in 0..16 {
                power *= t;
                s += 0.1 * (j + 1) as f64 * power;
            }
            s
        })
        .collect();
    (x, y)
}

#[test]
#[ignore]
fn bench_affine_qr_vs_lm() {
    for n in [1_000usize, 10_000, 100_000] {
        let (x, y) = affine_rows(n);
        for p in [2usize, 4, 8, 16] {
            let lm = affine_tree(p);
            let qr = affine_tree(p);
            let a = median((0..3).map(|_| {
                let mut t = lm.clone();
                let s = Instant::now();
                fit_weighted(&mut t, &x, &y, &[], 8);
                s.elapsed().as_secs_f64()
            }).collect());
            let b = median((0..3).map(|_| {
                let mut t = qr.clone();
                let s = Instant::now();
                fit_weighted_strategy(&mut t, &x, &y, &[], 8, FitStrategy::Auto);
                s.elapsed().as_secs_f64()
            }).collect());
            println!("affine N={n:>7} p={p:>2} lm={a:.6}s auto={b:.6}s speedup={:.1}x", a / b);
        }
    }
}

fn separable_tree() -> Node {
    node(
        Op::Add,
        vec![
            node(
                Op::Mul,
                vec![
                    constant(0.4),
                    node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                ],
            ),
            constant(0.0),
        ],
    )
}

#[test]
#[ignore]
fn bench_varpro_vs_lm() {
    for n in [1_000usize, 10_000, 100_000] {
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![-3.0 + 6.0 * i as f64 / (n - 1) as f64]).collect();
        let y: Vec<f64> = x.iter().map(|r| 2.5 * (1.3 * r[0]).sin() + 0.7).collect();
        let base = separable_tree();
        let a = median((0..3).map(|_| {
            let mut t = base.clone();
            let s = Instant::now();
            fit_weighted(&mut t, &x, &y, &[], 8);
            s.elapsed().as_secs_f64()
        }).collect());
        let b = median((0..3).map(|_| {
            let mut t = base.clone();
            let s = Instant::now();
            fit_weighted_strategy(&mut t, &x, &y, &[], 8, FitStrategy::VarPro);
            s.elapsed().as_secs_f64()
        }).collect());
        let mut t8 = base.clone();
        let sse8 = fit_weighted_strategy(&mut t8, &x, &y, &[], 8, FitStrategy::VarPro);
        let mut t40 = base.clone();
        let sse40 = fit_weighted_strategy(&mut t40, &x, &y, &[], 40, FitStrategy::VarPro);
        println!("varpro N={n:>7} lm={a:.6}s varpro={b:.6}s speedup={:.1}x sse8={sse8:.3e} sse40={sse40:.3e}", a / b);
    }
}

struct Generated {
    n: usize,
}

impl RowSource for Generated {
    fn rows(&self) -> usize {
        self.n
    }
    fn for_each_row(&self, f: &mut dyn FnMut(&[f64], f64, f64) -> bool) -> bool {
        for i in 0..self.n {
            let x = -3.0 + 6.0 * i as f64 / (self.n - 1) as f64;
            if !f(&[x], 2.0 * (0.9 * x).sin() + 0.3, 1.0) {
                return false;
            }
        }
        true
    }
}

#[test]
#[ignore]
fn bench_stream_vs_resident() {
    for n in [100_000usize, 1_000_000] {
        let source = Generated { n };
        let mut stream_tree = separable_tree();
        let a = median((0..3).map(|_| {
            let mut t = stream_tree.clone();
            let s = Instant::now();
            fit_stream_with(&mut t, &source, 8);
            s.elapsed().as_secs_f64()
        }).collect());
        stream_tree = separable_tree();
        fit_stream_with(&mut stream_tree, &source, 8);

        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        source.for_each_row(&mut |row, t, _| {
            x.push(row.to_vec());
            y.push(t);
            true
        });
        let base = separable_tree();
        let b = median((0..3).map(|_| {
            let mut t = base.clone();
            let s = Instant::now();
            fit_weighted(&mut t, &x, &y, &[], 8);
            s.elapsed().as_secs_f64()
        }).collect());
        println!("stream N={n:>7} resident={b:.6}s stream={a:.6}s ratio={:.2}x", b / a);
    }
}