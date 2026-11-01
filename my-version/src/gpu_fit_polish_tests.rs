//! #496: the polish arithmetic switch of the batched fit kernel.
//!
//! `f64` is the historical kernel; `f32` runs the Gauss-Newton steps on the f32
//! rows; `off` returns after the LM phase. Device tests skip cleanly on a
//! machine without CUDA.

use super::*;
use crate::expr::{constant, input, node, Node, Op};
use crate::fit::{self, DeviceLoss, FitPolish};

const MODES: [FitPolish; 3] = [FitPolish::F64, FitPolish::F32, FitPolish::Off];

/// `Σ coeffs[k] · x_k` over `p` orthogonal sinusoid columns.
fn fixture(p: usize, n: usize) -> (Vec<Vec<f64>>, Vec<f64>, Node) {
    let mut x = vec![vec![0.0; p]; n];
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = std::f64::consts::TAU * i as f64 / (n - 1) as f64;
        let mut v = 0.0;
        for k in 0..p {
            let a = ((k + 1) as f64 * t).sin();
            x[i][k] = a;
            v += (1.0 + 0.31 * k as f64) * a;
        }
        y.push(v + (((i * 37) % 11) as f64 - 5.0) * 2e-4);
    }
    let mut terms: Vec<Node> = (0..p)
        .map(|k| node(Op::Mul, vec![constant(0.25 + 0.07 * k as f64), input(k)]))
        .collect();
    let mut tree = terms.pop().expect("one term");
    while let Some(t) = terms.pop() {
        tree = node(Op::Add, vec![t, tree]);
    }
    (x, y, tree)
}

fn evaluator(x: &[Vec<f64>], y: &[f64]) -> Option<GpuEvaluator> {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let (n, p) = (x.len(), x[0].len());
    let mut cols64 = vec![0.0f64; n * p];
    for (i, row) in x.iter().enumerate() {
        for (k, &v) in row.iter().enumerate() {
            cols64[k * n + i] = v;
        }
    }
    let cols32: Vec<f32> = cols64.iter().map(|&v| v as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut ev = GpuEvaluator::new(&cols32, n, p, &y32).expect("device evaluator");
    assert!(ev.set_fit_data(&cols64, y, &[]), "f64 fit rows uploaded");
    Some(ev)
}

fn sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let mut t = tree.clone();
    fit::fit_weighted(&mut t, x, y, &[], 0)
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|c| c.to_bits()).collect()
}

#[test]
fn fit_polish_names_round_trip() {
    for m in MODES {
        assert_eq!(FitPolish::parse(m.name()), Some(m));
    }
    assert_eq!(FitPolish::default(), FitPolish::F64);
    assert_eq!(FitPolish::parse("f16"), None);
    assert_eq!(FitPolish::F64.code(), 0);
}

/// An unset evaluator and an explicit `f64` give the same bits, and switching
/// through the other modes and back leaves no state behind.
#[test]
fn fit_polish_f64_is_the_default_kernel() {
    let (x, y, tree) = fixture(5, 600);
    let Some(mut ev) = evaluator(&x, &y) else {
        return;
    };
    let plan = fit::gpu_fit_plan(&tree).expect("plan");
    let (default, ms) = ev.fit_batch(&[&plan], 8, DeviceLoss::Squared);
    assert!(ms >= 0.0 && default.len() == 1);
    ev.set_fit_polish(FitPolish::F64);
    let (explicit, _) = ev.fit_batch(&[&plan], 8, DeviceLoss::Squared);
    assert_eq!(bits(&default[0].c), bits(&explicit[0].c));
    assert_eq!(default[0].sse.to_bits(), explicit[0].sse.to_bits());
    assert_eq!(default[0].status, explicit[0].status);
    ev.set_fit_polish(FitPolish::Off);
    let (off, _) = ev.fit_batch(&[&plan], 8, DeviceLoss::Squared);
    assert_ne!(bits(&default[0].c), bits(&off[0].c), "the LM phase alone stops short of the polish");
    ev.set_fit_polish(FitPolish::F64);
    let (again, _) = ev.fit_batch(&[&plan], 8, DeviceLoss::Squared);
    assert_eq!(bits(&default[0].c), bits(&again[0].c));
}

/// `f32` and `off` return finite constants whose SSE is no worse than the
/// start constants, with a status from the documented set (never
/// `recovered_skipped`, which only the f64 assembly sets).
#[test]
fn fit_polish_f32_and_off_never_lose_to_the_start_constants() {
    let n = 800;
    for p in 1..=20usize {
        let (x, y, tree) = fixture(p, n);
        let Some(mut ev) = evaluator(&x, &y) else {
            return;
        };
        let plan = fit::gpu_fit_plan(&tree).expect("plan");
        let start = sse(&tree, &x, &y);
        for loss in [DeviceLoss::Squared, DeviceLoss::Absolute] {
            for mode in [FitPolish::F32, FitPolish::Off] {
                ev.set_fit_polish(mode);
                let (out, ms) = ev.fit_batch(&[&plan], 8, loss);
                assert!(ms >= 0.0 && out.len() == 1, "p={p} {loss:?} {mode:?}");
                let res = &out[0];
                assert!(res.c.iter().all(|v| v.is_finite()), "p={p} {loss:?} {mode:?}");
                assert!(
                    [0, 4, 5, 6].contains(&res.status),
                    "p={p} {loss:?} {mode:?}: status {}",
                    res.status
                );
                let mut fitted = tree.clone();
                plan.apply(&mut fitted, &res.c);
                let after = sse(&fitted, &x, &y);
                if loss == DeviceLoss::Squared {
                    assert!(
                        after <= start,
                        "p={p} {mode:?}: SSE {after:e} is worse than the start {start:e}"
                    );
                }
            }
        }
    }
}

/// The cycle split is recorded per launch, resets on read, and the polish
/// share is smaller without the f64 polish.
#[test]
fn fit_polish_cycle_split_is_recorded_and_resets() {
    let (x, y, tree) = fixture(8, 4000);
    let Some(mut ev) = evaluator(&x, &y) else {
        return;
    };
    let plan = fit::gpu_fit_plan(&tree).expect("plan");
    let refs: Vec<&fit::GpuFitPlan> = (0..64).map(|_| &plan).collect();
    let mut share = Vec::new();
    for mode in MODES {
        ev.set_fit_polish(mode);
        let (_, ms) = ev.fit_batch(&refs, 8, DeviceLoss::Squared);
        assert!(ms > 0.0, "{mode:?}");
        let polish = ev.take_polish_ms(ms);
        assert!(polish >= 0.0 && polish < ms as f64, "{mode:?}: {polish} of {ms}");
        assert_eq!(ev.take_polish_ms(ms), 0.0, "the read resets the counters");
        share.push(polish / ms as f64);
    }
    assert!(share[0] > 0.0, "the f64 polish takes measurable cycles");
    assert!(share[2] < share[0], "off ({}) spends less of the kernel on polish than f64 ({})", share[2], share[0]);
    assert!(share[1] < share[0], "f32 polish ({}) is cheaper than f64 ({})", share[1], share[0]);
}
