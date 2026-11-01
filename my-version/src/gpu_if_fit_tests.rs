//! #403: the `if` operator in the batched fit kernel.
//!
//! The condition selects a branch with `> 0`, the adjoint flows to the taken
//! branch only, and rows exactly at the switch point take the else branch like
//! the CPU. Device tests skip cleanly on a machine without CUDA.

use super::*;
use crate::data::Sample;
use crate::expr::{constant, input, node, Node, Op};
use crate::fit::{self, FitStrategy};
use crate::search::{self, Config};

const ROWS: usize = 640;

/// Inputs on an exact f32 grid: `x0` hits `0` on every fifth row, `x1`
/// equals `x0` on every seventh row, and `x2` is negative exactly where
/// `x0 > 0`, so a `sqrt` of it is `NaN` only on rows that never take it.
fn rows() -> (Vec<Vec<f64>>, Vec<f64>) {
    let mut x = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let x0 = if i % 5 == 0 { 0.0 } else { ((i * 37) % 33) as f64 / 8.0 - 2.0 };
        let x1 = if i % 7 == 0 { x0 } else { ((i * 53) % 29) as f64 / 8.0 - 1.5 };
        let x2 = if x0 > 0.0 { -0.25 - (i % 3) as f64 / 4.0 } else { 0.25 + (i % 4) as f64 / 4.0 };
        x.push(vec![x0, x1, x2]);
    }
    let y = x
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let noise = (((i * 31) % 11) as f64 - 5.0) * 1e-3;
            let v = if r[0] > 0.0 { 2.0 * r[1] + 1.0 } else { 3.0 - r[1] };
            v + noise
        })
        .collect();
    (x, y)
}

fn affine(a: f64, b: f64, v: usize) -> Node {
    node(Op::Add, vec![node(Op::Mul, vec![constant(a), input(v)]), constant(b)])
}

fn constants(tree: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

fn sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let mut t = tree.clone();
    fit::fit_weighted(&mut t, x, y, &[], 0)
}

fn trees() -> Vec<(&'static str, Node)> {
    vec![
        ("if(x0, a x1 + b, c x1 + d)", node(Op::If, vec![input(0), affine(0.5, 0.0, 1), affine(0.5, 0.0, 1)])),
        (
            "if(x0 - x1, a x1 + b, c x1 + d)",
            node(
                Op::If,
                vec![
                    node(Op::Sub, vec![input(0), input(1)]),
                    affine(1.0, 0.5, 1),
                    affine(-0.5, 2.0, 1),
                ],
            ),
        ),
        (
            "if(x0, a x1 + b, c sqrt(x2) + d)",
            node(
                Op::If,
                vec![
                    input(0),
                    affine(0.5, 0.0, 1),
                    node(Op::Add, vec![node(Op::Mul, vec![constant(1.0), node(Op::Sqrt, vec![input(2)])]), constant(0.0)]),
                ],
            ),
        ),
    ]
}

#[test]
fn if_plan_carries_the_third_child_in_the_slot_word() {
    let t = node(Op::If, vec![input(0), constant(1.5), constant(2.5)]);
    let plan = fit::gpu_fit_plan(&t).expect("`if` has a fit opcode");
    let rec: Vec<&[u32]> = plan.words.chunks(4).collect();
    assert_eq!(rec.len(), 4);
    assert_eq!(rec[3][0], 32, "the root is the `if` opcode");
    assert_eq!(&rec[3][1..], &[0, 1, 2], "condition, then branch, else branch");
    assert!(fit::gpu_fit_op_coverage(&Op::default_set(), false));
}

#[test]
fn if_device_fit_matches_the_cpu_fit_including_the_switch_point() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (x, y) = rows();
    let n = ROWS;
    let v = 3;
    let mut cols64 = vec![0.0f64; n * v];
    for (r, row) in x.iter().enumerate() {
        for (k, &val) in row.iter().enumerate() {
            cols64[k * n + r] = val;
        }
    }
    let cols32: Vec<f32> = cols64.iter().map(|&a| a as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&a| a as f32).collect();
    let mut ev = GpuEvaluator::new(&cols32, n, v, &y32).expect("device evaluator");
    assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
    assert!(x.iter().filter(|r| r[0] == 0.0).count() >= n / 5, "switch-point rows present");
    for (k, (name, start)) in trees().into_iter().enumerate() {
        let begin = sse(&start, &x, &y);
        let mut cpu = start.clone();
        fit::fit_weighted(&mut cpu, &x, &y, &[], 8);
        let cpu_sse = fit::fit_weighted(&mut cpu, &x, &y, &[], 8);
        let plan = fit::gpu_fit_plan(&start).expect("plan");
        let (out, ms) = ev.fit_batch(&[&plan], 8, fit::DeviceLoss::Squared);
        assert!(ms >= 0.0, "{name}: the launch was refused");
        assert!(fit::fit_status_usable(out[0].status), "{name}: status {}", out[0].status);
        let mut dev = start.clone();
        plan.apply(&mut dev, &out[0].c);
        let dev_sse = sse(&dev, &x, &y);
        eprintln!("{name}: start {begin:.6e} cpu {cpu_sse:.6e} device {dev_sse:.6e} c {:?}", constants(&dev));
        let want = if k == 0 { 1e-2 } else { 0.99 };
        assert!(dev_sse < begin * want, "{name}: the fit did not move: {dev_sse} vs {begin}");
        assert!(
            dev_sse <= cpu_sse * (1.0 + 1e-8) + 1e-12,
            "{name}: device {dev_sse:.12e} vs cpu {cpu_sse:.12e}"
        );
    }
}

fn sample(x: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

/// A device-dispatched search over operators including `if` fits on the
/// device and names no `unsupported_plan` fallback.
#[test]
fn a_search_with_if_fits_on_the_device() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let (x, y) = rows();
    let train = sample(x.clone(), y.clone());
    let test = sample(x, y);
    let cfg = Config {
        generations: 4,
        islands: 1,
        pop: 24,
        max_complexity: 12,
        fit_iters: 6,
        patience: 0,
        seed: 5,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Square, Op::Neg, Op::If],
        dispatch: crate::dispatch::DispatchMode::Device,
        gpu: true,
        fit_strategy: FitStrategy::Gpu,
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    let g = out.gpu.as_ref().expect("a fit ledger");
    assert!(g.fit_gpu > 0, "nothing was fitted on the device: {g:?}");
    assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback);
    assert_eq!(
        g.fit_fallback_by_cause.get("unsupported_plan").copied().unwrap_or(0),
        0,
        "{:?}",
        g.fit_fallback_by_cause
    );
}
