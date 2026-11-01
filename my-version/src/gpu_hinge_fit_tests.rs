//! SM-1: the hinge fit codes of the batched kernel (6 `hinge_squared`, 7
//! `hinge_ordinary`, 8 `hinge`).
//!
//! The kernel statistic is the configured hinge form for both precisions and
//! both `hinge_ordinary` phases, the device fit matches the CPU arm's schedule
//! on the objective, and every number a search reports is the exact CPU `f64`
//! value. Device tests skip cleanly on a machine without CUDA.

use super::*;
use crate::data::Sample;
use crate::expr::{self, constant, input, node, Node, Op};
use crate::fit::{self, BatchStatus, BatchedGpu, DeviceLoss, FitStrategy};
use crate::metrics::{self, Metric, ScoreObjective};
use crate::search::{self, Config};

const FORMS: [(Metric, DeviceLoss, ScoreObjective); 3] = [
    (Metric::HingeSquared, DeviceLoss::HingeSquared, ScoreObjective::HingeSquared),
    (Metric::HingeOrdinary, DeviceLoss::HingeOrdinary, ScoreObjective::HingeOrdinary),
    (Metric::Hinge, DeviceLoss::HingeSigmoid, ScoreObjective::HingeSigmoid),
];

/// Positives at `+sep/2`, negatives at `−sep/2`, each spread by `spread·u`
/// with `u` a centred sum of three uniforms; every `flip`-th label flipped.
fn cell(pos: usize, neg: usize, sep: f64, spread: f64, flip: usize, seed: u64) -> Sample {
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut unit = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(pos + neg);
    let mut y = Vec::with_capacity(pos + neg);
    for i in 0..pos + neg {
        let positive = i < pos;
        let u = unit() + unit() + unit() - 1.5;
        let centre = if positive { 0.5 * sep } else { -0.5 * sep };
        x.push(vec![centre + spread * u, unit() * 2.0 - 1.0]);
        let flipped = flip > 0 && i % flip == flip - 1;
        y.push(if positive != flipped { 1.0 } else { 0.0 });
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn overlap() -> Sample {
    cell(300, 500, 1.0, 1.0, 0, 11)
}

fn separable() -> Sample {
    cell(300, 500, 4.0, 0.5, 0, 12)
}

fn imbalanced() -> Sample {
    cell(40, 760, 1.5, 1.0, 0, 13)
}

fn noisy() -> Sample {
    cell(300, 500, 2.0, 0.5, 7, 14)
}

/// `a·x0 + b`; plan slot 0 is `a`, slot 1 is `b`.
fn affine(a: f64, b: f64) -> Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
    )
}

/// `a·x0 + b + c·tanh(x1)`.
fn shaped(a: f64, b: f64, c: f64) -> Node {
    node(
        Op::Add,
        vec![
            affine(a, b),
            node(Op::Mul, vec![constant(c), node(Op::Tanh, vec![input(1)])]),
        ],
    )
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

fn bits(tree: &Node) -> Vec<u64> {
    constants(tree).iter().map(|v| v.to_bits()).collect()
}

fn score(tree: &Node, s: &Sample, m: Metric) -> f64 {
    fit::exact_metric_score(tree, &s.x, &s.y, &s.w, m)
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
    assert!(ev.set_fit_sign(&fit::classification_signs_weighted(&s.y, &s.w)));
    Some(ev)
}

/// The raw kernel answer for one tree: constants applied, status.
fn device_fit(ev: &mut GpuEvaluator, tree: &Node, iters: usize, loss: DeviceLoss) -> (Node, i32) {
    let plan = fit::gpu_fit_plan(tree).expect("plan");
    let (out, ms) = ev.fit_batch(&[&plan], iters, loss);
    assert!(ms >= 0.0, "{loss:?}: the launch was refused");
    let mut fitted = tree.clone();
    plan.apply(&mut fitted, &out[0].c);
    (fitted, out[0].status)
}

/// SM-1: the kernel's loss statistic is the configured hinge form, row for row,
/// in the f32 assembly, the f32 trial sweep, the f64 assembly and the f64
/// trial sweep, for both `hinge_ordinary` phases; and the f64 `Jᵀr` is the
/// gradient of that form (Gauss-Newton, logistic) or of its majoriser (MM).
#[test]
fn hinge_kernel_loss_equals_classification_row_loss() {
    let mut s = overlap();
    s.w = (0..s.len())
        .map(|i| if i % 9 == 4 { 0.0 } else { 0.5 + (i % 5) as f64 * 0.25 })
        .collect();
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let signs = fit::classification_signs_weighted(&s.y, &s.w);
    let trees = [affine(0.7, -0.2), affine(-1.3, 0.4), affine(2.5, 0.1), affine(0.0, 0.0)];
    let plans: Vec<fit::GpuFitPlan> = trees.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    for (k, t) in trees.iter().enumerate() {
        assert_eq!(plans[k].c0, constants(t), "slot order is (a, b)");
    }
    for (metric, loss, objective) in FORMS {
        let phases: &[i32] = if loss == DeviceLoss::HingeOrdinary { &[0, 1] } else { &[0] };
        for &phase in phases {
            let out = ev.fit_probe(&refs, loss, phase).expect("probe ran");
            for (k, tree) in trees.iter().enumerate() {
                let [a32, l32, a64, l64] = out[k].0;
                let jtr = &out[k].1;
                let mut want = 0.0;
                let mut grad = [0.0f64; 2];
                for i in 0..s.len() {
                    let w = s.w[i];
                    if w <= 0.0 {
                        continue;
                    }
                    let p = expr::eval(tree, &s.x[i]);
                    let t = signs[i];
                    let l = 0.5 * (t + 1.0);
                    want += w * metrics::classification_row_loss(p, l, objective);
                    let u = t * p;
                    let (sw, resp) = match (loss, phase) {
                        (DeviceLoss::HingeSquared, _) => (if u < 1.0 { w } else { 0.0 }, p - t),
                        (DeviceLoss::HingeOrdinary, 1) => {
                            let a = (1.0 - u).abs().max(1e-6);
                            (w / (2.0 * a), p - t * (1.0 + a))
                        }
                        _ => {
                            let q = (if u >= 0.0 {
                                1.0 / (1.0 + (-u).exp())
                            } else {
                                u.exp() / (1.0 + u.exp())
                            })
                            .clamp(1e-9, 1.0 - 1e-9);
                            (w * q * (1.0 - q), -t / q)
                        }
                    };
                    grad[0] += sw * resp * s.x[i][0];
                    grad[1] += sw * resp;
                    if loss == DeviceLoss::HingeOrdinary && phase == 1 && (1.0 - u).abs() > 1e-6 {
                        // The majoriser's gradient is the hinge subgradient.
                        let sub = if u < 1.0 { -w * t } else { 0.0 };
                        assert!((sw * resp - sub).abs() <= 1e-9 * w.max(1.0), "row {i}: {}", sw * resp);
                    }
                }
                let tag = format!("{metric:?} phase {phase} tree {k}");
                assert!((a64 - want).abs() <= 1e-12 * want.max(1.0), "{tag}: f64 assembly {a64} vs {want}");
                assert!((l64 - want).abs() <= 1e-12 * want.max(1.0), "{tag}: f64 trial {l64} vs {want}");
                assert!((a32 - want).abs() <= 1e-5 * want.max(1.0), "{tag}: f32 assembly {a32} vs {want}");
                assert!((a32 - l32).abs() <= 1e-12 * want.max(1.0), "{tag}: f32 {a32} vs {l32}");
                for j in 0..2 {
                    let scale = grad[j].abs().max(1.0);
                    assert!((jtr[j] - grad[j]).abs() <= 1e-9 * scale, "{tag}: jtr[{j}] {} vs {}", jtr[j], grad[j]);
                }
            }
        }
    }
}

/// A label-sign launch without the uploaded sign is refused on the device.
#[test]
fn hinge_launch_without_a_sign_is_refused() {
    let s = overlap();
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    assert!(ev.set_fit_sign(&[]), "the sign is cleared");
    let plan = fit::gpu_fit_plan(&affine(0.5, 0.0)).unwrap();
    for (_, loss, _) in FORMS {
        let (_, ms) = ev.fit_batch(&[&plan], 8, loss);
        assert!(ms < 0.0, "{loss:?} ran without a label sign");
        assert!(ev.fit_probe(&[&plan], loss, 0).is_none());
    }
}

/// The #222 parity rule per form: the device objective is no worse than the
/// CPU arm's at equal iterations, on separable and non-separable fixtures.
#[test]
fn hinge_device_matches_cpu_arm() {
    let fixtures = [
        ("overlap", overlap()),
        ("separable", separable()),
        ("imbalanced", imbalanced()),
        ("noisy", noisy()),
    ];
    for (name, s) in &fixtures {
        let Some(mut ev) = evaluator(s) else {
            return;
        };
        for (metric, loss, _) in FORMS {
            // The sigmoid hinge has no finite optimum on separable data.
            if metric == Metric::Hinge && *name == "separable" {
                continue;
            }
            for start in [affine(0.0, 0.0), shaped(0.3, 0.1, 0.2)] {
                let begin = score(&start, s, metric);
                let mut cpu = start.clone();
                let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 8, metric);
                let (dev, status) = device_fit(&mut ev, &start, 8, loss);
                let dev_score = score(&dev, s, metric);
                eprintln!("{metric:?} {name}: start {begin:.9} cpu {cpu_score:.9} device {dev_score:.9} status {status}");
                assert!(fit::fit_status_usable(status), "{metric:?} {name}: status {status}");
                assert!(dev_score <= begin, "{metric:?} {name}: {dev_score} > start {begin}");
                assert!(
                    dev_score <= cpu_score * (1.0 + 1e-5) + 1e-12,
                    "{metric:?} {name}: device {dev_score:.12e} vs cpu {cpu_score:.12e}"
                );
            }
        }
    }
}

/// Constant model under `hinge_squared`: `p* = (n₊ − n₋)/(n₊ + n₋)`, to 1e-6
/// on the device and 1e-9 after the exact refit.
#[test]
fn hinge_squared_constant_model_is_closed_form() {
    for (pos, neg) in [(300usize, 300usize), (420, 180), (60, 540)] {
        let n = pos + neg;
        let s = Sample {
            x: (0..n).map(|i| vec![i as f64 / n as f64]).collect(),
            y: (0..n).map(|i| if i < pos { 1.0 } else { 0.0 }).collect(),
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        };
        let Some(mut ev) = evaluator(&s) else {
            return;
        };
        let want = (pos as f64 - neg as f64) / n as f64;
        let (dev, status) = device_fit(&mut ev, &constant(0.0), 8, DeviceLoss::HingeSquared);
        assert!(fit::fit_status_usable(status), "status {status}");
        let c = constants(&dev)[0];
        assert!((c - want).abs() <= 1e-6 * want.abs().max(1.0), "{pos}:{neg}: device {c} vs {want}");
        let mut refit = vec![dev];
        fit::refit_finalists_exact(&mut refit, &s.x, &s.y, &s.w, 8, Metric::HingeSquared);
        let c = constants(&refit[0])[0];
        assert!((c - want).abs() <= 1e-9 * want.abs().max(1.0), "{pos}:{neg}: refit {c} vs {want}");
    }
}

/// Constant model under `hinge_ordinary`: the optimum is the majority margin
/// `sign(n₊ − n₋)` with objective `2·min(n₊, n₋)/n`; the logistic phase alone
/// stops at `ln(n₊/n₋)`.
#[test]
fn hinge_ordinary_constant_model_reaches_the_majority_margin() {
    for (pos, neg) in [(420usize, 180usize), (180, 420)] {
        let n = pos + neg;
        let s = Sample {
            x: (0..n).map(|i| vec![i as f64 / n as f64]).collect(),
            y: (0..n).map(|i| if i < pos { 1.0 } else { 0.0 }).collect(),
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        };
        let Some(mut ev) = evaluator(&s) else {
            return;
        };
        let want = 2.0 * pos.min(neg) as f64 / n as f64;
        let (dev, status) = device_fit(&mut ev, &constant(0.0), 16, DeviceLoss::HingeOrdinary);
        assert!(fit::fit_status_usable(status), "status {status}");
        let got = score(&dev, &s, Metric::HingeOrdinary);
        assert!((got - want).abs() <= 1e-6 * want, "{pos}:{neg}: device {got} vs {want}");
        let logistic = (pos as f64 / neg as f64).ln();
        let at_logistic = score(&constant(logistic), &s, Metric::HingeOrdinary);
        assert!(got < at_logistic, "{pos}:{neg}: {got} !< {at_logistic}");
    }
}

/// `hinge` judges every logistic trial on `2σ(−u)`: from a start above the
/// log-loss optimum the logistic step lowers log-loss but raises the sigmoid
/// hinge, so the hinge code keeps the start while the log-loss code moves.
#[test]
fn hinge_sigmoid_rejects_a_logloss_only_improvement() {
    let n = 500;
    let pos = 300;
    let s = Sample {
        x: (0..n).map(|i| vec![i as f64 / n as f64]).collect(),
        y: (0..n).map(|i| if i < pos { 1.0 } else { 0.0 }).collect(),
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let start = constant(3.0);
    let (moved, st) = device_fit(&mut ev, &start, 8, DeviceLoss::Logistic);
    assert!(fit::fit_status_usable(st));
    assert!(score(&moved, &s, Metric::LogLoss) < score(&start, &s, Metric::LogLoss));
    assert!(
        score(&moved, &s, Metric::Hinge) > score(&start, &s, Metric::Hinge),
        "the fixture must make the log-loss step worse on the sigmoid hinge"
    );
    let (kept, status) = device_fit(&mut ev, &start, 8, DeviceLoss::HingeSigmoid);
    assert_eq!(bits(&kept), bits(&start), "a log-loss-only improvement was accepted");
    assert_eq!(status, 6, "no_improvement");
    let mut cpu = start.clone();
    fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 8, Metric::Hinge);
    assert_eq!(bits(&cpu), bits(&start), "the CPU arm rejects it too");
}

/// Separable data has a finite optimum with loss 0, and the device reaches it.
#[test]
fn hinge_separable_data_reaches_zero() {
    let s = separable();
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    for (metric, loss) in [
        (Metric::HingeOrdinary, DeviceLoss::HingeOrdinary),
        (Metric::HingeSquared, DeviceLoss::HingeSquared),
    ] {
        let (dev, status) = device_fit(&mut ev, &affine(0.0, 0.0), 16, loss);
        assert!(fit::fit_status_usable(status), "{metric:?}: status {status}");
        assert!(constants(&dev).iter().all(|c| c.is_finite()));
        assert_eq!(score(&dev, &s, metric), 0.0, "{metric:?}: {:?}", constants(&dev));
    }
}

/// A single-class sample is declined before launch; the exact CPU arm keeps
/// the candidate.
#[test]
fn hinge_single_class_is_declined() {
    let s = Sample {
        x: (0..300).map(|i| vec![i as f64 / 300.0]).collect(),
        y: vec![1.0; 300],
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    for (metric, _, _) in FORMS {
        let mut trees = vec![affine(0.2, -0.5)];
        let out = {
            let backend = BatchedGpu::new(&mut ev);
            fit::fit_batch_verified(&backend, &mut trees, &s.x, &s.y, &s.w, 4, metric)
        };
        assert_eq!(out.status, vec![BatchStatus::Unsupported], "{metric:?}");
        let mut cpu = affine(0.2, -0.5);
        let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 4, metric);
        assert_eq!(out.score[0].to_bits(), cpu_score.to_bits(), "{metric:?}");
        assert_eq!(bits(&trees[0]), bits(&cpu), "{metric:?}");
    }
}

/// A zero-weight row is skipped before its prediction or label is read: a NaN
/// target there changes no constant and no status.
#[test]
fn hinge_weighted_zero_rows_are_invisible() {
    let mut base = overlap();
    base.w = vec![1.0; base.len()];
    let mut masked = base.clone();
    masked.x.push(vec![0.3, f64::NAN]);
    masked.y.push(f64::NAN);
    masked.w.push(0.0);
    let Some(mut a) = evaluator(&base) else {
        return;
    };
    let mut b = evaluator(&masked).expect("device evaluator");
    for (metric, loss, _) in FORMS {
        let start = shaped(0.4, -0.1, 0.3);
        let (fa, sa) = device_fit(&mut a, &start, 8, loss);
        let (fb, sb) = device_fit(&mut b, &start, 8, loss);
        assert_eq!(sa, sb, "{metric:?}");
        assert_eq!(bits(&fa), bits(&fb), "{metric:?}");
        assert_eq!(score(&fa, &base, metric).to_bits(), score(&fb, &masked, metric).to_bits());
    }
}

/// The same input twice is bit-identical, and a candidate alone or inside a
/// mixed batch gets the same constants.
#[test]
fn hinge_fit_is_deterministic_and_batch_independent() {
    let s = noisy();
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let model = shaped(0.6, 0.2, -0.3);
    let plan = fit::gpu_fit_plan(&model).unwrap();
    let others: Vec<fit::GpuFitPlan> = (0..7)
        .map(|k| fit::gpu_fit_plan(&affine(0.1 * (k as f64 + 1.0), -0.2 * k as f64)).unwrap())
        .collect();
    for (_, loss, _) in FORMS {
        let (a, _) = ev.fit_batch(&[&plan], 8, loss);
        let (b, _) = ev.fit_batch(&[&plan], 8, loss);
        let mut batch: Vec<&fit::GpuFitPlan> = others.iter().collect();
        batch.insert(3, &plan);
        let (c, _) = ev.fit_batch(&batch, 8, loss);
        let key = |o: &GpuFitOutput| (o.c.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), o.status, o.sse.to_bits());
        assert_eq!(key(&a[0]), key(&b[0]), "{loss:?}: a repeat moved a constant");
        assert_eq!(key(&a[0]), key(&c[3]), "{loss:?}: batch composition changed a candidate");
    }
}

/// SM-2 under `hinge_squared`: a constant that only moves inactive rows has an
/// exactly zero Gauss-Newton diagonal. It is frozen and the other constant is
/// still fitted, as the CPU ridge solve does.
#[test]
fn zero_diagonal_slot_is_frozen_like_the_cpu() {
    // Rows 0..400 overlap on x0 with x1 = 0; rows 400..500 are positives with
    // x1 = 1, which the start `c·x1` with c = 10 puts far past the margin.
    let mut s = overlap();
    s.x.truncate(400);
    s.y.truncate(400);
    for r in s.x.iter_mut() {
        r[1] = 0.0;
    }
    for i in 0..100 {
        s.x.push(vec![(i as f64) / 100.0 - 0.5, 1.0]);
        s.y.push(1.0);
    }
    let Some(mut ev) = evaluator(&s) else {
        return;
    };
    let start = node(
        Op::Add,
        vec![constant(0.0), node(Op::Mul, vec![constant(10.0), input(1)])],
    );
    let (dev, status) = device_fit(&mut ev, &start, 8, DeviceLoss::HingeSquared);
    assert!(fit::fit_status_usable(status), "status {status}");
    let c = constants(&dev);
    assert_eq!(c[1].to_bits(), 10.0f64.to_bits(), "the zero-diagonal slot moved: {c:?}");
    let begin = score(&start, &s, Metric::HingeSquared);
    let got = score(&dev, &s, Metric::HingeSquared);
    assert!(got < begin, "the free constant was not fitted: {got} vs {begin}");
    let mut cpu = start.clone();
    let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 8, Metric::HingeSquared);
    assert!(got <= cpu_score * (1.0 + 1e-5), "device {got} vs cpu {cpu_score}");
}

fn tiny_config(metric: Metric, seed: u32) -> Config {
    Config {
        generations: 3,
        islands: 1,
        pop: 16,
        max_complexity: 10,
        fit_iters: 6,
        patience: 0,
        seed,
        metric,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Tanh, Op::Neg, Op::Square],
        // Under compute-sanitizer the cost model would keep every stage on the
        // CPU and no kernel would run.
        dispatch: crate::dispatch::DispatchMode::Device,
        ..Default::default()
    }
}

/// A search defers its hinge fits to the device, and every reported error is
/// the exact CPU `f64` metric of the stored tree.
#[test]
fn reported_hinge_errors_are_exact_cpu_values() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = cell(160, 240, 1.5, 1.0, 0, 21);
    let test = cell(40, 60, 1.5, 1.0, 0, 22);
    for (k, (metric, _, _)) in FORMS.iter().enumerate() {
        let cfg = Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..tiny_config(*metric, 7 + k as u32)
        };
        let out = search::run(&cfg, &train, &test);
        let g = out.gpu.as_ref().expect("a fit ledger");
        assert!(g.fit_gpu > 0, "{metric:?}: nothing was fitted on the device: {g:?}");
        assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback, "{metric:?}");
        assert!(!out.archive.items.is_empty());
        for item in &out.archive.items {
            assert_eq!(
                item.error.to_bits(),
                score(&item.tree, &train, *metric).to_bits(),
                "{metric:?}: archive row is not the exact CPU value"
            );
        }
    }
}

/// `--fit-iters 0` is a no-op on both backends: the same archive constants.
#[test]
fn hinge_no_fit_is_a_no_op() {
    let train = cell(120, 180, 1.5, 1.0, 0, 31);
    let test = cell(30, 45, 1.5, 1.0, 0, 32);
    for (k, (metric, _, _)) in FORMS.iter().enumerate() {
        let base = Config {
            fit_iters: 0,
            ..tiny_config(*metric, 5 + k as u32)
        };
        let cpu = search::run(&base, &train, &test);
        let gpu = search::run(
            &Config {
                gpu: true,
                fit_strategy: FitStrategy::Gpu,
                ..base
            },
            &train,
            &test,
        );
        assert_eq!(cpu.archive.items.len(), gpu.archive.items.len(), "{metric:?}");
        for (a, b) in cpu.archive.items.iter().zip(&gpu.archive.items) {
            assert_eq!(bits(&a.tree), bits(&b.tree), "{metric:?}: --no-fit moved a constant");
            assert_eq!(a.error.to_bits(), b.error.to_bits(), "{metric:?}");
        }
    }
}

/// A single-class training sample completes with every fit fallback labelled
/// `single_class`.
#[test]
fn a_single_class_hinge_run_is_labelled() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = Sample {
        x: (0..300).map(|i| vec![i as f64 / 300.0]).collect(),
        y: vec![0.0; 300],
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    };
    for (metric, _, _) in FORMS {
        let cfg = Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..tiny_config(metric, 12)
        };
        let out = search::run(&cfg, &train, &train);
        let g = out.gpu.as_ref().expect("a fit ledger");
        if g.fit_attempts > 0 {
            assert_eq!(
                g.fit_fallback_by_cause.get("single_class").copied().unwrap_or(0),
                g.fit_fallback,
                "{metric:?}: {g:?}"
            );
            assert_eq!(g.fit_gpu, 0, "{metric:?}: nothing may be fitted on one class");
        }
    }
}

/// The paired fit-stage measurement: the CPU arm per candidate against one
/// device batch on the same candidates, objective and iterations. Rows from
/// `RCSR_BENCH_ROWS` (default 154 000), metric from `RCSR_BENCH_METRIC`
/// (`hinge_ordinary`, `hinge_squared`, `hinge`; default all three).
#[test]
#[ignore]
fn bench_hinge_fit_stage() {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let rows: usize = std::env::var("RCSR_BENCH_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(154_000);
    let pos = rows * 2 / 5;
    let s = cell(pos, rows - pos, 1.5, 1.0, 9, 41);
    let mut ev = evaluator(&s).expect("device evaluator");
    let starts: Vec<Node> = (0..32)
        .map(|k| {
            let a = 0.2 + 0.05 * k as f64;
            if k % 2 == 0 {
                affine(a, -0.1 * (k % 5) as f64)
            } else {
                shaped(a, 0.05 * (k % 3) as f64, 0.1 + 0.02 * k as f64)
            }
        })
        .collect();
    let plans: Vec<fit::GpuFitPlan> = starts.iter().map(|t| fit::gpu_fit_plan(t).unwrap()).collect();
    let refs: Vec<&fit::GpuFitPlan> = plans.iter().collect();
    let wanted = std::env::var("RCSR_BENCH_METRIC").ok();
    for (metric, loss, _) in FORMS {
        if wanted.as_deref().is_some_and(|m| m != metric.name()) {
            continue;
        }
        let t0 = std::time::Instant::now();
        let cpu: Vec<f64> = starts
            .iter()
            .map(|t| {
                let mut tree = t.clone();
                fit::fit_weighted_metric(&mut tree, &s.x, &s.y, &s.w, 8, metric)
            })
            .collect();
        let cpu_s = t0.elapsed().as_secs_f64();
        let t1 = std::time::Instant::now();
        let (out, kernel_ms) = ev.fit_batch(&refs, 8, loss);
        let dev_s = t1.elapsed().as_secs_f64();
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
            let d = score(&tree, &s, metric);
            let rel = (d - cpu[k]) / cpu[k].abs().max(1e-12);
            worst = worst.max(rel);
            if d <= cpu[k] * (1.0 + 1e-5) + 1e-12 {
                wins += 1;
            }
        }
        eprintln!(
            "bench {}: N={rows} P={} iters=8 cpu={cpu_s:.3}s ({:.2} ms/cand) device={dev_s:.4}s kernel={kernel_ms:.1}ms speedup={:.1}x fallbacks={fallbacks} device<=cpu*(1+1e-5)={wins}/{} worst_rel={worst:.3e}",
            metric.name(),
            starts.len(),
            1e3 * cpu_s / starts.len() as f64,
            cpu_s / dev_s.max(1e-9),
            starts.len()
        );
    }
}
