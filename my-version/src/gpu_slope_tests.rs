//! #236: the slope (first-difference MAE) scorer `RC_SCORE_SLOPE` and the
//! slope fit arm `RC_LOSS_SLOPE` of the batched kernel.
//!
//! The device differences row r against row r - 1 of the resident sample, the
//! pair set the search admits only when the sample's slope pairs are the
//! consecutive rows. Every reported number is the exact CPU `f64` value.
//! Device tests skip cleanly on a machine without CUDA.

use super::*;
use crate::data::Sample;
use crate::dispatch::{DispatchMode, Route, Stage};
use crate::expr::{self, constant, input, node, Node, Op};
use crate::fit::{self, DeviceLoss, FitStrategy};
use crate::metrics::{self, Metric, ScoreObjective, ScoreStatus};
use crate::search::{self, Config};

/// `x0` on `[-3, 3]` and exactly 0 on the middle row, `x1` uniform noise, and
/// a smooth target with a trend and noise: magnitudes like the returns and
/// moving averages of the project's time-series cells.
fn series(n: usize, seed: u64) -> Sample {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64 - (n / 2) as f64) * 6.0 / n as f64;
        let u = unit() * 2.0 - 1.0;
        x.push(vec![t, u]);
        y.push(3.0 * t.sin() + 0.5 * t + 0.3 * u.tanh() + 0.05 * (unit() - 0.5));
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

/// `a·x0 + b`; plan slot 0 is `a`, slot 1 is `b`.
fn affine(a: f64, b: f64) -> Node {
    node(Op::Add, vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)])
}

/// `a·x0 + b + c·tanh(x1)`.
fn shaped(a: f64, b: f64, c: f64) -> Node {
    node(
        Op::Add,
        vec![affine(a, b), node(Op::Mul, vec![constant(c), node(Op::Tanh, vec![input(1)])])],
    )
}

/// `a·sin(b·x0) + c·x0 + d`.
fn wave(a: f64, b: f64, c: f64, d: f64) -> Node {
    node(
        Op::Add,
        vec![
            node(
                Op::Mul,
                vec![constant(a), node(Op::Sin, vec![node(Op::Mul, vec![constant(b), input(0)])])],
            ),
            affine(c, d),
        ],
    )
}

/// Non-finite on the row where `x0 == 0` only.
fn hole() -> Node {
    node(Op::Div, vec![input(1), input(0)])
}

fn constants(tree: &Node) -> Vec<f64> {
    let mut out = Vec::new();
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
    }
    walk(tree, &mut out);
    out
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// The exact CPU slope of a tree on the sample.
fn slope(tree: &Node, s: &Sample) -> f64 {
    let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(tree, row)).collect();
    metrics::score(&pred, &s.y, Metric::Slope)
}

fn evaluator(s: &Sample) -> Option<GpuEvaluator> {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let n = s.len();
    let v = s.x.first().map(|r| r.len()).unwrap_or(0);
    let mut cols = vec![0.0f64; n * v];
    for (r, row) in s.x.iter().enumerate() {
        for (k, &x) in row.iter().enumerate().take(v) {
            cols[k * n + r] = x;
        }
    }
    let cols32: Vec<f32> = cols.iter().map(|&x| x as f32).collect();
    let y32: Vec<f32> = s.y.iter().map(|&y| y as f32).collect();
    let mut ev = GpuEvaluator::new(&cols32, n, v, &y32).expect("device evaluator");
    assert!(ev.set_fit_data(&cols, &s.y, &s.w), "f64 fit rows uploaded");
    Some(ev)
}

fn score_trees() -> Vec<Node> {
    vec![
        affine(0.5, 0.0),
        affine(3.4, -1.0),
        shaped(2.9, 0.2, 0.25),
        wave(2.5, 1.1, 0.4, 0.3),
        hole(),
    ]
}

/// The device slope equals the CPU reducer on the same f32 rows to 1e-12 and
/// the exact `f64` metric to 1e-6; one row has no pair, and a single
/// non-finite row (the predecessor or the successor of a pair) is `+inf`.
#[test]
fn slope_device_score_matches_cpu() {
    for n in [1usize, 2, 3, 300, 4_099] {
        let s = series(n, 7);
        let Some(mut ev) = evaluator(&s) else {
            return;
        };
        let trees = score_trees();
        let pop = Population::from_trees(trees.iter());
        let (prog, offsets) = pop.programs();
        let pcount = vec![0i32; trees.len()];
        let batch = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
        assert!(batch.ms >= 0.0, "n={n}: launch");
        let preds = ev.predictions(prog, offsets).expect("prediction readback");
        let y32: Vec<f64> = s.y.iter().map(|&v| v as f32 as f64).collect();
        for (k, tree) in trees.iter().enumerate() {
            let got = batch.values[k];
            let exact = slope(tree, &s);
            if n < 2 || k == trees.len() - 1 {
                assert_eq!(batch.status[k], ScoreStatus::NonFinite, "n={n} tree {k}");
                assert_eq!(got, f64::INFINITY, "n={n} tree {k}");
                assert_eq!(exact, f64::INFINITY, "n={n} tree {k}: the CPU agrees");
                continue;
            }
            assert_eq!(batch.status[k], ScoreStatus::Ok, "n={n} tree {k}");
            let p32: Vec<f64> = preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
            let reduced = metrics::score_native(&p32, &y32, ScoreObjective::Slope, 0);
            assert_eq!(reduced.status, ScoreStatus::Ok);
            assert!(
                (got - reduced.value).abs() <= 1e-12 * reduced.value.abs().max(1e-300),
                "n={n} tree {k}: device {got} vs reducer {}",
                reduced.value
            );
            if n >= 300 {
                let rel = (got - exact).abs() / exact.abs();
                assert!(rel <= 1e-6, "n={n} tree {k}: device {got} vs exact {exact} ({rel:e})");
            }
            let one = ev.score_batch(
                &prog[offsets[k] as usize..offsets[k + 1] as usize],
                &[0, offsets[k + 1] - offsets[k]],
                ScoreObjective::Slope,
                &[0],
            );
            assert_eq!(one.values[0].to_bits(), got.to_bits(), "n={n} tree {k}: alone vs batch");
        }
    }
}

/// #293 pattern: row ranges with cuts at arbitrary rows, including one-row
/// ranges, give the bits of the resident launch. A pair whose rows straddle a
/// cut reads its predecessor from the resident columns.
#[test]
fn slope_score_is_partition_invariant() {
    let n = 2_053;
    let s = series(n, 9);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let trees = score_trees();
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let pcount = vec![0i32; trees.len()];
    let want = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    assert!(want.ms >= 0.0);
    let uniform = |step: usize| -> Vec<RowRange> {
        (0..n)
            .step_by(step)
            .map(|start| RowRange {
                start,
                len: step.min(n - start),
            })
            .collect()
    };
    let cuts = [1usize, 2, 3, 255, 256, 257, 700, 701, 1_999, 2_052];
    let mut uneven = Vec::new();
    let mut start = 0usize;
    for &c in &cuts {
        uneven.push(RowRange { start, len: c - start });
        start = c;
    }
    uneven.push(RowRange { start, len: n - start });
    let parts = [
        ("whole", vec![RowRange { start: 0, len: n }]),
        ("uniform 256", uniform(256)),
        ("uniform 1000", uniform(1_000)),
        ("uneven", uneven),
    ];
    for (name, ranges) in &parts {
        let got = ev.score_rows(prog, offsets, ScoreObjective::Slope, &pcount, ranges);
        assert!(got.ms >= 0.0, "{name}: launch");
        for k in 0..trees.len() {
            assert_eq!(got.status[k], want.status[k], "{name} slot {k}");
            assert_eq!(
                got.values[k].to_bits(),
                want.values[k].to_bits(),
                "{name} slot {k}: {} vs {}",
                got.values[k],
                want.values[k]
            );
        }
    }
}

/// A tiled context and a weighted context refuse a slope launch, and a search
/// whose plan is tiled keeps the CPU scorer with the named reason.
#[test]
fn slope_scoring_is_refused_in_tiled_residency() {
    let n = 4_000;
    let s = series(n, 11);
    let Some(mut resident) = evaluator(&s) else {
        return;
    };
    let mut cols = vec![0.0f32; n * 2];
    for (r, row) in s.x.iter().enumerate() {
        for (k, &x) in row.iter().enumerate() {
            cols[k * n + r] = x as f32;
        }
    }
    let y32: Vec<f32> = s.y.iter().map(|&y| y as f32).collect();
    let mut tiled =
        GpuEvaluator::new_tiled(cols, n, 2, &y32, 512, None, None).expect("tiled evaluator");
    assert_eq!(tiled.residency(), crate::memory::Residency::Tiled);
    let trees = score_trees();
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let pcount = vec![0i32; trees.len()];
    let refused = tiled.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    assert!(refused.ms < 0.0, "a tiled slope launch must be refused");
    assert!(refused.status.iter().all(|&st| st == ScoreStatus::DeviceError));
    let squared = tiled.score_batch(prog, offsets, ScoreObjective::Squared, &pcount);
    assert!(squared.ms >= 0.0, "the tiled context still scores other objectives");

    assert!(resident.set_score_weights(&vec![1.0; n]));
    let weighted = resident.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    assert!(weighted.ms < 0.0, "a weighted slope launch must be refused");

    let train = series(60_000, 12);
    let test = series(2_000, 13);
    let mut cfg = Config {
        generations: 2,
        islands: 1,
        pop: 16,
        max_complexity: 8,
        fit_iters: 0,
        patience: 0,
        seed: 3,
        metric: Metric::Slope,
        gpu: true,
        gpu_notes: false,
        dispatch: DispatchMode::Device,
        ..Default::default()
    };
    let plan = search::device_plan_for(&cfg, &train).expect("a resident plan");
    assert_eq!(plan.mode, crate::memory::Residency::Resident);
    cfg.gpu_memory = Some(plan.required - 1);
    let out = search::run(&cfg, &train, &test);
    assert_eq!(out.backend, crate::backend::Kind::Cpu);
    let reason = out.backend_reason.clone().unwrap_or_default();
    assert!(
        reason.contains("tiled residency refuses slope scoring"),
        "the refusal must be named: {reason:?}"
    );
}

/// D8: on a population, the device slope has relative error at most 1e-6
/// against the exact CPU `f64` slope, and orders the candidates as the CPU
/// does.
#[test]
fn slope_device_scores_rank_like_the_cpu() {
    let s = series(20_000, 15);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let trees: Vec<Node> = (0..48)
        .map(|k| match k % 3 {
            0 => affine(0.1 * k as f64, 0.3),
            1 => shaped(2.0 + 0.05 * k as f64, -0.2, 0.01 * k as f64),
            _ => wave(1.0 + 0.04 * k as f64, 1.0, 0.5, 0.0),
        })
        .collect();
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let batch = ev.score_batch(prog, offsets, ScoreObjective::Slope, &vec![0i32; trees.len()]);
    assert!(batch.all_ok(), "{:?}", batch.status);
    let exact: Vec<f64> = trees.iter().map(|t| slope(t, &s)).collect();
    let mut worst = 0.0f64;
    for k in 0..trees.len() {
        worst = worst.max((batch.values[k] - exact[k]).abs() / exact[k].abs());
    }
    eprintln!("slope D8: max relative error {worst:.3e} over {} candidates", trees.len());
    assert!(worst <= 1e-6, "max relative error {worst:e}");
    let mut dev_order: Vec<usize> = (0..trees.len()).collect();
    dev_order.sort_by(|&a, &b| batch.values[a].partial_cmp(&batch.values[b]).unwrap());
    let mut cpu_order: Vec<usize> = (0..trees.len()).collect();
    cpu_order.sort_by(|&a, &b| exact[a].partial_cmp(&exact[b]).unwrap());
    assert_eq!(dev_order, cpu_order, "no rank inversion");
}

/// The f64 assembly is the differenced normal equations of one IRLS round:
/// rows `√u·(∇f_r − ∇f_{r−1})`, response `√u·g`, loss `Σ|g|`. Sizes cross the
/// f32 and f64 chunk boundaries, so the carried row is used. The additive
/// constant's differenced column is exactly zero.
#[test]
fn slope_pair_assembly_matches_cpu_differenced_normal_equations() {
    for n in [2usize, 3, 129, 257, 600] {
        let s = series(n, 17);
        let Some(mut ev) = evaluator(&s) else {
            return;
        };
        let trees = [shaped(2.5, 0.4, 0.3), shaped(-1.0, 2.0, 1.5)];
        let plans: Vec<fit::GpuFitPlan> =
            trees.iter().map(|t| fit::gpu_fit_plan(t).expect("plan")).collect();
        for (k, t) in trees.iter().enumerate() {
            assert_eq!(plans[k].c0, constants(t), "slot order is (a, b, c)");
        }
        let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
        let out = ev.fit_probe(&refs, DeviceLoss::Slope, 0).expect("probe ran");
        for (k, tree) in trees.iter().enumerate() {
            let c = constants(tree);
            let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(tree, r)).collect();
            let grad = |i: usize| [s.x[i][0], 1.0, s.x[i][1].tanh()];
            let mut loss = 0.0;
            let mut jtj = [[0.0f64; 3]; 3];
            let mut jtr = [0.0f64; 3];
            for i in 1..n {
                let g = (pred[i] - pred[i - 1]) - (s.y[i] - s.y[i - 1]);
                loss += g.abs();
                let w = (1.0 / g.abs().max(1e-6)).sqrt();
                let (gi, gp) = (grad(i), grad(i - 1));
                let d: Vec<f64> = (0..3).map(|j| w * (gi[j] - gp[j])).collect();
                for a in 0..3 {
                    for b in 0..3 {
                        jtj[a][b] += d[a] * d[b];
                    }
                    jtr[a] += d[a] * w * g;
                }
            }
            let ([a32, l32, a64, l64], got_jtr, got_jtj) = &out[k];
            let tag = format!("n={n} tree {k} {c:?}");
            let tol = |want: f64| 1e-10 * want.abs().max(1.0);
            assert!((a64 - loss).abs() <= tol(loss), "{tag}: f64 assembly {a64} vs {loss}");
            assert!((l64 - loss).abs() <= tol(loss), "{tag}: f64 trial {l64} vs {loss}");
            assert!((a32 - l32).abs() <= 1e-12 * loss.max(1.0), "{tag}: f32 {a32} vs {l32}");
            assert!((a32 - loss).abs() <= 1e-5 * loss.max(1.0), "{tag}: f32 {a32} vs {loss}");
            for a in 0..3 {
                assert!((got_jtr[a] - jtr[a]).abs() <= tol(jtr[a]), "{tag}: jtr[{a}] {} vs {}", got_jtr[a], jtr[a]);
                for b in 0..=a {
                    let got = got_jtj[a * (a + 1) / 2 + b];
                    assert!((got - jtj[a][b]).abs() <= tol(jtj[a][b]), "{tag}: jtj[{a}][{b}] {got} vs {}", jtj[a][b]);
                }
            }
            assert_eq!(got_jtj[2], 0.0, "{tag}: the additive column is not exactly zero");
            assert_eq!(got_jtj[1], 0.0, "{tag}");
            assert_eq!(got_jtj[4], 0.0, "{tag}");
            assert_eq!(got_jtr[1], 0.0, "{tag}");
        }
    }
}

/// D6: the additive constant is frozen by the zero-diagonal rule, so its value
/// is the squared warm start's on the device and on the CPU arm.
#[test]
fn slope_additive_constant_is_frozen_and_keeps_the_squared_level() {
    let s = series(600, 19);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let start = shaped(0.0, 0.0, 0.0);
    let plan = fit::gpu_fit_plan(&start).expect("plan");
    let (squared, ms) = ev.fit_batch(&[&plan], 8, DeviceLoss::Squared);
    assert!(ms >= 0.0);
    let (out, ms) = ev.slope_fit_batch(&[&plan], 8);
    assert!(ms >= 0.0);
    assert!(fit::fit_status_usable(out[0].status), "status {}", out[0].status);
    assert_eq!(
        out[0].c[1].to_bits(),
        squared[0].c[1].to_bits(),
        "the level moved: device {:?} vs squared {:?}",
        out[0].c,
        squared[0].c
    );
    let mut fitted = start.clone();
    plan.apply(&mut fitted, &out[0].c);
    assert!(slope(&fitted, &s) < slope(&start, &s), "the free constants were not fitted");

    let mut cpu = start.clone();
    fit::fit_weighted(&mut cpu, &s.x, &s.y, &[], 8);
    let level = constants(&cpu)[1];
    let pairs = fit::consecutive_pairs(s.len());
    fit::slope_irls(&mut cpu, &s.x, &s.y, &pairs, 8);
    assert_eq!(constants(&cpu)[1].to_bits(), level.to_bits(), "the CPU level moved");
}

/// The arm never returns constants with a higher exact slope than the incoming
/// ones, on the device and on the CPU arm.
#[test]
fn slope_irls_never_worsens_the_metric() {
    let s = series(1_500, 21);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let starts = [
        affine(0.0, 0.0),
        affine(3.4, 1.0),
        shaped(2.9, 0.2, 0.3),
        shaped(-2.0, 5.0, -1.0),
        wave(2.8, 1.0, 0.5, 0.0),
        wave(0.1, 3.0, -1.0, 2.0),
    ];
    let plans: Vec<fit::GpuFitPlan> = starts.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    for iters in [1usize, 2, 8] {
        let (out, ms) = ev.slope_fit_batch(&refs, iters);
        assert!(ms >= 0.0);
        for (k, start) in starts.iter().enumerate() {
            let begin = slope(start, &s);
            let mut dev = start.clone();
            plans[k].apply(&mut dev, &out[k].c);
            let got = slope(&dev, &s);
            assert!(got <= begin * (1.0 + 1e-12), "iters {iters} tree {k}: device {got} > start {begin}");
            assert!((out[k].sse - got).abs() <= 1e-9 * got, "iters {iters} tree {k}: sse {} vs {got}", out[k].sse);
            let mut cpu = start.clone();
            let before = slope(&cpu, &s);
            fit::slope_irls(&mut cpu, &s.x, &s.y, &fit::consecutive_pairs(s.len()), iters);
            assert!(slope(&cpu, &s) <= before, "iters {iters} tree {k}: the CPU arm worsened");
        }
    }
}

/// When the incoming constants are already at least as good as anything the
/// arm reaches, they come back bit for bit with `no_improvement`.
#[test]
fn slope_fit_keeps_incoming_constants_when_better() {
    let n = 600;
    let s = Sample {
        x: (0..n).map(|i| vec![i as f64, 0.0]).collect(),
        y: (0..n).map(|i| 3.0 * i as f64 + if i >= n / 2 { 100.0 } else { 0.0 }).collect(),
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let incoming = affine(3.0, 7.0);
    assert_eq!(slope(&incoming, &s), 100.0 / (n - 1) as f64);
    let plan = fit::gpu_fit_plan(&incoming).unwrap();
    let (out, ms) = ev.slope_fit_batch(&[&plan], 8);
    assert!(ms >= 0.0);
    assert_eq!(bits(&out[0].c), bits(&[3.0, 7.0]), "incoming constants not kept");
    assert_eq!(out[0].status, 6, "no_improvement");
    let mut cpu = incoming.clone();
    fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &[], 8, Metric::Slope);
    assert_eq!(bits(&constants(&cpu)), bits(&[3.0, 7.0]), "the CPU arm kept them too");
}

/// D7 algorithm parity: from the same start and with the iterations to
/// converge, the device slope is at most the CPU arm's (squared fit then
/// differenced IRLS) times `1 + 1e-5`.
#[test]
fn slope_device_fit_matches_cpu() {
    let s = series(4_000, 23);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let starts = [affine(0.0, 0.0), shaped(0.3, 0.1, 0.2), wave(1.0, 1.0, 0.0, 0.0)];
    let plans: Vec<fit::GpuFitPlan> = starts.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let iters = 32;
    let (out, ms) = ev.slope_fit_batch(&refs, iters);
    assert!(ms >= 0.0);
    for (k, start) in starts.iter().enumerate() {
        let mut dev = start.clone();
        plans[k].apply(&mut dev, &out[k].c);
        let dev_score = slope(&dev, &s);
        let mut cpu = start.clone();
        fit::fit_weighted(&mut cpu, &s.x, &s.y, &[], iters);
        fit::slope_irls(&mut cpu, &s.x, &s.y, &fit::consecutive_pairs(s.len()), iters);
        let cpu_score = slope(&cpu, &s);
        let mut full = start.clone();
        let full_score = fit::fit_weighted_metric(&mut full, &s.x, &s.y, &[], iters, Metric::Slope);
        eprintln!(
            "slope tree {k}: start {:.9} device {dev_score:.9} cpu irls {cpu_score:.9} cpu arm {full_score:.9} status {}",
            slope(start, &s),
            out[k].status
        );
        assert!(fit::fit_status_usable(out[k].status), "tree {k}: status {}", out[k].status);
        assert!(
            dev_score <= cpu_score * (1.0 + 1e-5) + 1e-12,
            "tree {k}: device {dev_score:.12e} vs cpu {cpu_score:.12e}"
        );
    }
}

/// A candidate's constants do not depend on the batch it is fitted in, and a
/// repeated launch gives the same bits.
#[test]
fn slope_gpu_fit_is_deterministic_and_batch_independent() {
    let s = series(1_000, 25);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let batch: Vec<Node> = vec![
        shaped(0.3, 0.1, 0.2),
        affine(1.0, 0.0),
        wave(1.0, 1.0, 0.0, 0.0),
        shaped(2.0, -1.0, 0.5),
        affine(-1.0, 3.0),
        wave(2.0, 0.8, 0.1, 0.2),
        shaped(0.0, 0.0, 0.0),
    ];
    let plans: Vec<fit::GpuFitPlan> = batch.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let (all, ms) = ev.slope_fit_batch(&refs, 8);
    assert!(ms >= 0.0);
    let (again, _) = ev.slope_fit_batch(&refs, 8);
    for k in 0..batch.len() {
        assert_eq!(bits(&all[k].c), bits(&again[k].c), "tree {k}: repeat");
        assert_eq!(all[k].status, again[k].status);
        let (one, _) = ev.slope_fit_batch(&[refs[k]], 8);
        assert_eq!(bits(&all[k].c), bits(&one[0].c), "tree {k}: alone vs batch");
        assert_eq!(all[k].sse.to_bits(), one[0].sse.to_bits(), "tree {k}");
    }
}

/// `--fit-iters 0` moves no constant on the device arm, the CPU arm or a whole
/// search on either backend.
#[test]
fn slope_no_fit_is_a_no_op() {
    let s = series(400, 27);
    let tree = shaped(0.3, 0.1, 0.2);
    let mut cpu = tree.clone();
    fit::fit_slope_pairs(&mut cpu, &s.x, &s.y, &fit::consecutive_pairs(s.len()), 0);
    assert_eq!(bits(&constants(&cpu)), bits(&constants(&tree)));
    fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &[], 0, Metric::Slope);
    assert_eq!(bits(&constants(&cpu)), bits(&constants(&tree)));
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let plan = fit::gpu_fit_plan(&tree).unwrap();
    let (out, ms) = ev.slope_fit_batch(&[&plan], 0);
    assert_eq!(ms, 0.0, "nothing is launched");
    assert_eq!(bits(&out[0].c), bits(&constants(&tree)));

    let train = series(600, 28);
    let test = series(200, 29);
    let run = search::run(&slope_config(0, 5), &train, &test);
    assert_eq!(run.backend, crate::backend::Kind::Gpu, "{:?}", run.backend_reason);
    let g = run.gpu.as_ref().expect("a gpu ledger");
    assert_eq!((g.fit_attempts, g.fit_gpu, g.fit_refits), (0, 0, 0), "{g:?}");
    for item in &run.archive.items {
        assert_eq!(
            item.error.to_bits(),
            search::eval_error(&item.tree, &train, Metric::Slope).to_bits()
        );
    }
}

/// The pairs of a panel laid out as consecutive groups of `sizes` rows: row `r`
/// pairs with row `r - 1` unless `r` opens a group.
fn panel_pairs(sizes: &[usize]) -> Vec<[u32; 2]> {
    let mut pairs = Vec::new();
    let mut start = 0usize;
    for &size in sizes {
        for r in start + 1..start + size {
            pairs.push([(r - 1) as u32, r as u32]);
        }
        start += size;
    }
    pairs
}

/// The mask the search uploads for `pairs`, through the search's own builder.
fn mask_of(s: &Sample) -> search::SlopePairMask {
    search::slope_pair_mask(s).expect("a pair list the mask represents")
}

fn with_pairs(mut s: Sample, pairs: Vec<[u32; 2]>) -> Sample {
    s.pairs = Some(pairs);
    s
}

/// A panel of groups of 1, 2, 700, 1, 300 and 1 rows.
const PANEL: [usize; 6] = [1, 2, 700, 1, 300, 1];

/// The device slope under a pair mask equals the CPU slope over the same
/// pairs: 1e-12 against the reducer on the f32 rows, 1e-6 against the exact
/// `f64` metric. Pairs never cross a group, and a group of one row has none.
#[test]
fn slope_device_matches_the_cpu_on_a_grouped_panel() {
    let n: usize = PANEL.iter().sum();
    let pairs = panel_pairs(&PANEL);
    assert_eq!(pairs.len(), n - PANEL.len());
    let mut s = with_pairs(series(n, 41), pairs.clone());
    // `hole()` is then non-finite on row 703 only, a group of one row.
    s.x[n / 2][0] = 0.1;
    s.x[703][0] = 0.0;
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let mask = mask_of(&s);
    assert!(ev.set_slope_pairs(&mask.bits, mask.pairs));
    let trees = score_trees();
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let pcount = vec![0i32; trees.len()];
    let batch = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    assert!(batch.ms >= 0.0, "launch");
    let preds = ev.predictions(prog, offsets).expect("prediction readback");
    let y32: Vec<f64> = s.y.iter().map(|&v| v as f32 as f64).collect();
    for (k, tree) in trees.iter().enumerate() {
        let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(tree, row)).collect();
        let exact = metrics::score_slope_pairs(&pred, &s.y, &pairs);
        if k == trees.len() - 1 {
            assert_eq!(batch.status[k], ScoreStatus::NonFinite, "tree {k}");
            assert_eq!(batch.values[k], f64::INFINITY, "tree {k}");
            assert_eq!(exact, f64::INFINITY, "tree {k}: the CPU agrees");
            continue;
        }
        assert_eq!(batch.status[k], ScoreStatus::Ok, "tree {k}");
        let p32: Vec<f64> = preds[k * n..(k + 1) * n].iter().map(|&v| v as f64).collect();
        let reduced = metrics::score_slope_pairs(&p32, &y32, &pairs);
        let got = batch.values[k];
        assert!(
            (got - reduced).abs() <= 1e-12 * reduced.abs().max(1e-300),
            "tree {k}: device {got} vs reducer {reduced}"
        );
        let rel = (got - exact).abs() / exact.abs();
        assert!(rel <= 1e-6, "tree {k}: device {got} vs exact {exact} ({rel:e})");
        let unmasked = slope(tree, &s);
        assert!((unmasked - exact).abs() > 1e-9, "tree {k}: the boundaries change the metric");
    }
}

/// A pair list with no pair scores `+inf` with the non-finite status, as the
/// CPU does; one pair is its single difference; a mask can be cleared.
#[test]
fn slope_pair_mask_edge_cases() {
    let n = 70;
    let s = series(n, 43);
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let trees = vec![affine(0.5, 0.0), shaped(2.9, 0.2, 0.25)];
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let pcount = vec![0i32; trees.len()];
    let consecutive = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);

    let none = with_pairs(s.clone(), Vec::new());
    let mask = mask_of(&none);
    assert_eq!((mask.pairs, mask.bits.len()), (0, 3));
    assert!(ev.set_slope_pairs(&mask.bits, mask.pairs));
    let got = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    for k in 0..trees.len() {
        assert_eq!(got.status[k], ScoreStatus::NonFinite, "no pair, tree {k}");
        assert_eq!(got.values[k], f64::INFINITY, "no pair, tree {k}");
    }

    let one = with_pairs(s.clone(), vec![[40, 41]]);
    let mask = mask_of(&one);
    assert!(ev.set_slope_pairs(&mask.bits, mask.pairs));
    let got = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    for (k, tree) in trees.iter().enumerate() {
        let want = metrics::score_slope_pairs(
            &s.x.iter().map(|row| expr::eval(tree, row)).collect::<Vec<_>>(),
            &s.y,
            &[[40, 41]],
        );
        assert!((got.values[k] - want).abs() <= 1e-5, "one pair, tree {k}: {} vs {want}", got.values[k]);
    }

    assert!(ev.set_slope_pairs(&[], 0), "an empty mask clears");
    let cleared = ev.score_batch(prog, offsets, ScoreObjective::Slope, &pcount);
    for k in 0..trees.len() {
        assert_eq!(cleared.values[k].to_bits(), consecutive.values[k].to_bits(), "cleared, tree {k}");
    }
    assert!(!ev.set_slope_pairs(&[0u32; 2], 0), "a mask of the wrong length is refused");
}

/// The device slope fit under a pair mask is at most the CPU fit over the same
/// pairs (squared fit, differenced IRLS) times `1 + 1e-5`, and it reports the
/// slope over the masked pair count.
#[test]
fn slope_device_fit_matches_the_cpu_on_a_grouped_panel() {
    let n: usize = PANEL.iter().sum();
    let pairs = panel_pairs(&PANEL);
    let s = with_pairs(series(n, 45), pairs.clone());
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let mask = mask_of(&s);
    assert!(ev.set_slope_pairs(&mask.bits, mask.pairs));
    let starts = [affine(0.0, 0.0), shaped(0.3, 0.1, 0.2), wave(1.0, 1.0, 0.0, 0.0)];
    let plans: Vec<fit::GpuFitPlan> = starts.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let iters = 32;
    let (out, ms) = ev.slope_fit_batch(&refs, iters);
    assert!(ms >= 0.0);
    let pair_slope = |t: &Node| {
        let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(t, row)).collect();
        metrics::score_slope_pairs(&pred, &s.y, &pairs)
    };
    for (k, start) in starts.iter().enumerate() {
        let mut dev = start.clone();
        plans[k].apply(&mut dev, &out[k].c);
        let dev_score = pair_slope(&dev);
        let mut cpu = start.clone();
        fit::fit_weighted(&mut cpu, &s.x, &s.y, &[], iters);
        fit::slope_irls(&mut cpu, &s.x, &s.y, &pairs, iters);
        let cpu_score = pair_slope(&cpu);
        eprintln!(
            "panel slope tree {k}: start {:.9} device {dev_score:.9} cpu irls {cpu_score:.9} status {}",
            pair_slope(start),
            out[k].status
        );
        assert!(fit::fit_status_usable(out[k].status), "tree {k}: status {}", out[k].status);
        assert!(
            dev_score <= cpu_score * (1.0 + 1e-5) + 1e-12,
            "tree {k}: device {dev_score:.12e} vs cpu {cpu_score:.12e}"
        );
        assert!(dev_score <= pair_slope(start) + 1e-12, "tree {k}: never worse than the start");
        assert!(
            (out[k].sse - dev_score).abs() <= 1e-6 * dev_score,
            "tree {k}: reported {} vs pair slope {dev_score}",
            out[k].sse
        );
    }
}

/// A `--group-by` style panel run keeps the slope scorer and fit on the device:
/// the fit ledger counts device fits and every archive error is the exact CPU
/// slope over the panel's pairs.
#[test]
fn slope_panel_run_fits_slope_on_the_device() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let sizes = [1usize, 2, 900, 1, 700, 1, 395];
    let n: usize = sizes.iter().sum();
    let train = with_pairs(level_and_slope(n, 51), panel_pairs(&sizes));
    let test = level_and_slope(600, 52);
    let cfg = slope_config(6, 9);
    let run = search::run(&cfg, &train, &test);
    assert_eq!(run.backend, crate::backend::Kind::Gpu, "{:?}", run.backend_reason);
    let eval = run.dispatch.ledger.entry(Stage::Eval).expect("an eval ledger entry");
    assert_eq!(eval.route, Route::Device, "the slope scorer did not run on the device");
    let g = run.gpu.as_ref().expect("a fit ledger");
    assert!(g.fit_gpu > 0, "nothing was fitted on the device: {g:?}");
    assert!(!run.archive.items.is_empty());
    for item in &run.archive.items {
        assert_eq!(
            item.error.to_bits(),
            search::eval_error(&item.tree, &train, Metric::Slope).to_bits(),
            "an archive row is not the exact CPU slope over the panel pairs"
        );
    }
}

fn slope_config(fit_iters: usize, seed: u32) -> Config {
    Config {
        generations: 4,
        islands: 1,
        pop: 24,
        max_complexity: 10,
        fit_iters,
        patience: 0,
        seed,
        metric: Metric::Slope,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Sin, Op::Tanh, Op::Neg],
        gpu: true,
        gpu_notes: false,
        fit_strategy: FitStrategy::Gpu,
        // Under compute-sanitizer the cost model would keep every stage on the
        // CPU and no kernel would run.
        dispatch: DispatchMode::Device,
        ..Default::default()
    }
}

/// A slow level the inputs cannot explain plus a fast term they can: `3·x1`
/// is the slope optimum, while a level statistic prefers models of the level.
fn level_and_slope(n: usize, seed: u64) -> Sample {
    let mut s = series(n, seed);
    for (i, row) in s.x.iter_mut().enumerate() {
        row[0] = i as f64 / n as f64;
    }
    for (i, y) in s.y.iter_mut().enumerate() {
        *y = 10.0 * (std::f64::consts::TAU * i as f64 / n as f64).sin() + 3.0 * s.x[i][1];
    }
    s
}

/// A slope search ranks on slope on the device: the evaluator and the fit
/// arm run there, every archived error is the exact CPU slope of its tree, and
/// the best front member is as good as the `--no-gpu` run's on a cell where a
/// level statistic would rank a different law first.
#[test]
fn slope_gpu_front_is_ranked_on_slope() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = level_and_slope(3_000, 31);
    let test = level_and_slope(600, 32);
    let cfg = slope_config(6, 9);
    let gpu = search::run(&cfg, &train, &test);
    assert_eq!(gpu.backend, crate::backend::Kind::Gpu, "{:?}", gpu.backend_reason);
    let eval = gpu.dispatch.ledger.entry(Stage::Eval).expect("an eval ledger entry");
    assert_eq!(eval.route, Route::Device, "the slope scorer did not run on the device");
    let g = gpu.gpu.as_ref().expect("a fit ledger");
    assert!(g.fit_gpu > 0, "nothing was fitted on the device: {g:?}");
    assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback);
    assert!(!gpu.archive.items.is_empty());
    for item in &gpu.archive.items {
        assert_eq!(
            item.error.to_bits(),
            search::eval_error(&item.tree, &train, Metric::Slope).to_bits(),
            "an archive row is not the exact CPU slope"
        );
    }
    let cpu = search::run(&Config { gpu: false, ..cfg }, &train, &test);
    let best = |out: &search::Outcome| {
        out.archive.items.iter().map(|s| s.error).fold(f64::INFINITY, f64::min)
    };
    eprintln!("slope front: gpu best {:.9} cpu best {:.9}", best(&gpu), best(&cpu));
    assert!(best(&gpu) <= best(&cpu) * 1.05, "gpu {} vs cpu {}", best(&gpu), best(&cpu));
}

/// Ship-gate measurement: the CPU slope arm per candidate against one device
/// batch on the same candidates and iterations. Rows from `RCSR_BENCH_ROWS`
/// (default 154 000).
#[test]
#[ignore]
fn bench_slope_fit_stage() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let rows: usize = std::env::var("RCSR_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(154_000);
    let s = series(rows, 41);
    let mut ev = evaluator(&s).expect("device evaluator");
    let starts: Vec<Node> = (0..32)
        .map(|k| {
            let a = 0.2 + 0.1 * k as f64;
            match k % 3 {
                0 => affine(a, -0.1 * (k % 5) as f64),
                1 => shaped(a, 0.05 * (k % 3) as f64, 0.1 + 0.02 * k as f64),
                _ => wave(a, 0.9, 0.3, 0.1),
            }
        })
        .collect();
    let plans: Vec<fit::GpuFitPlan> = starts.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let t0 = std::time::Instant::now();
    let cpu: Vec<f64> = starts
        .iter()
        .map(|t| {
            let mut tree = t.clone();
            fit::fit_weighted_metric(&mut tree, &s.x, &s.y, &[], 8, Metric::Slope)
        })
        .collect();
    let cpu_s = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let (out, kernel_ms) = ev.slope_fit_batch(&refs, 8);
    let dev_s = t1.elapsed().as_secs_f64();
    // The squared LM on both sides, the unit of the dispatcher's fit rates.
    let t2 = std::time::Instant::now();
    for t in &starts {
        let mut tree = t.clone();
        fit::fit_weighted(&mut tree, &s.x, &s.y, &[], 8);
    }
    let cpu_sq_s = t2.elapsed().as_secs_f64();
    let t3 = std::time::Instant::now();
    let (_, _) = ev.fit_batch(&refs, 8, DeviceLoss::Squared);
    let dev_sq_s = t3.elapsed().as_secs_f64();
    eprintln!(
        "bench slope: squared LM cpu={cpu_sq_s:.3}s device={dev_sq_s:.4}s; slope/squared cpu={:.2} device={:.2}",
        cpu_s / cpu_sq_s.max(1e-9),
        dev_s / dev_sq_s.max(1e-9)
    );
    let mut fallbacks = 0;
    let mut wins = 0;
    let mut worst: f64 = f64::NEG_INFINITY;
    for (k, res) in out.iter().enumerate() {
        if !fit::fit_status_usable(res.status) {
            fallbacks += 1;
            continue;
        }
        let mut tree = starts[k].clone();
        plans[k].apply(&mut tree, &res.c);
        let d = slope(&tree, &s);
        let rel = (d - cpu[k]) / cpu[k].abs().max(1e-12);
        worst = worst.max(rel);
        if d <= cpu[k] * (1.0 + 1e-5) + 1e-12 {
            wins += 1;
        }
    }
    eprintln!(
        "bench slope: N={rows} P={} iters=8 cpu={cpu_s:.3}s ({:.2} ms/cand) device={dev_s:.4}s kernel={kernel_ms:.1}ms speedup={:.1}x fallbacks={fallbacks} device<=cpu*(1+1e-5)={wins}/{} worst_rel={worst:.3e}",
        starts.len(),
        1e3 * cpu_s / starts.len() as f64,
        cpu_s / dev_s.max(1e-9),
        starts.len()
    );
}

/// D8 on a real cell: a CSV from `RCSR_BENCH_CSV`, target `RCSR_BENCH_TARGET`
/// (default `y1`), inputs `RCSR_BENCH_VARS` (default `r1,r3,r7,ma7`), rows in
/// the stable order of `RCSR_BENCH_TIME` (default `ts`), the order of
/// `--split temporal`. Prints the maximum relative error of the device slope
/// against the exact CPU slope over a population and whether the ranking
/// matches.
#[test]
#[ignore]
fn bench_slope_score_precision_on_csv() {
    let Ok(path) = std::env::var("RCSR_BENCH_CSV") else {
        eprintln!("set RCSR_BENCH_CSV");
        return;
    };
    let target = std::env::var("RCSR_BENCH_TARGET").unwrap_or_else(|_| "y1".to_string());
    let vars = std::env::var("RCSR_BENCH_VARS").unwrap_or_else(|_| "r1,r3,r7,ma7".to_string());
    let time = std::env::var("RCSR_BENCH_TIME").unwrap_or_else(|_| "ts".to_string());
    let text = std::fs::read_to_string(&path).expect("read the CSV");
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().expect("a header").split(',').collect();
    let col = |name: &str| header.iter().position(|h| *h == name).unwrap_or_else(|| panic!("no column {name}"));
    let (ti, yi) = (col(&time), col(&target));
    let vi: Vec<usize> = vars.split(',').map(col).collect();
    let mut rows: Vec<(f64, Vec<f64>, f64)> = Vec::new();
    for line in lines {
        let f: Vec<&str> = line.split(',').collect();
        let num = |k: usize| f[k].parse::<f64>().unwrap_or(f64::NAN);
        let x: Vec<f64> = vi.iter().map(|&k| num(k)).collect();
        let y = num(yi);
        if y.is_finite() && x.iter().all(|v| v.is_finite()) {
            rows.push((num(ti), x, y));
        }
    }
    rows.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let s = Sample {
        x: rows.iter().map(|r| r.1.clone()).collect(),
        y: rows.iter().map(|r| r.2).collect(),
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let nv = vi.len();
    let trees: Vec<Node> = (0..64)
        .map(|k| {
            let a = 0.05 * (k as f64 + 1.0);
            let lin = node(Op::Add, vec![node(Op::Mul, vec![constant(a), input(k % nv)]), constant(0.01 * k as f64)]);
            match k % 4 {
                0 => lin,
                1 => node(Op::Add, vec![lin, node(Op::Mul, vec![constant(-0.3), input((k + 1) % nv)])]),
                2 => node(Op::Mul, vec![constant(a), node(Op::Tanh, vec![input(k % nv)])]),
                _ => node(Op::Sub, vec![lin, node(Op::Mul, vec![input((k + 2) % nv), input((k + 3) % nv)])]),
            }
        })
        .collect();
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    let batch = ev.score_batch(prog, offsets, ScoreObjective::Slope, &vec![0i32; trees.len()]);
    assert!(batch.all_ok(), "{:?}", batch.status);
    let exact: Vec<f64> = trees.iter().map(|t| slope(t, &s)).collect();
    let mut worst = 0.0f64;
    for k in 0..trees.len() {
        worst = worst.max((batch.values[k] - exact[k]).abs() / exact[k].abs());
    }
    let mut dev_order: Vec<usize> = (0..trees.len()).collect();
    dev_order.sort_by(|&a, &b| batch.values[a].partial_cmp(&batch.values[b]).unwrap());
    let mut cpu_order: Vec<usize> = (0..trees.len()).collect();
    cpu_order.sort_by(|&a, &b| exact[a].partial_cmp(&exact[b]).unwrap());
    let inversions = (0..trees.len()).filter(|&i| dev_order[i] != cpu_order[i]).count();
    eprintln!(
        "slope D8 on {path}: rows={} candidates={} max_rel_err={worst:.3e} rank_positions_differing={inversions}",
        s.len(),
        trees.len()
    );
}
