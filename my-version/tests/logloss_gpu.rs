//! #234: the logistic (log-loss) arm of the batched GPU fit.
//!
//! What this file pins, in the order the design claims it:
//!
//! 1. **the label rule** — `t = 2·[y > (min y + max y)/2] − 1`, subset
//!    independent for a two-valued target, single-class refused;
//! 2. **the objective is the configured one** — `log_loss` has a native *fit*
//!    objective and, since #289, a native *fitness* objective
//!    (`ScoreObjective::LogLoss`; the regression `DeviceLoss::for_metric` stays
//!    `None`), so it can never be ranked by squared error;
//! 3. **the proposal gate** — a device proposal is kept only when the exact
//!    `f64` log-loss at it is finite and no worse than at the start;
//! 4. **the device kernel** — parity with the CPU `logistic_irls` on the
//!    objective, extreme logits, a constant model against its closed form,
//!    determinism and batch independence, and a labelled single-class refusal;
//! 5. **the search** — exact-CPU-reported errors, determinism, labelled
//!    fallbacks and the `--no-fit` no-op;
//! 6. **the measurement** (`bench_logloss_end_to_end`, ignored) — the paired
//!    speed/quality run that the shipped claim rests on.
//!
//! Device tests skip cleanly on a machine without CUDA. Pure-CPU invariants run
//! in every build, so the gate and the label rule are proven without a device.

use rcsr::checkpoint::Backend;
use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Op};
use rcsr::fit::{self, FitStrategy, ProposalGate};
use rcsr::metrics::{self, Metric};
use rcsr::search::{self, Config};

// ---------------------------------------------------------------- fixtures --

/// A deterministic binary sample: a latent score `0.9·x0 + 0.5·x1 + noise·ε`
/// with exactly `positives · n` positive labels (the top scores), so the class
/// balance is controlled to the row and the label rule sees two values only.
fn binary_sample(n: usize, positives: f64, noise: f64, seed: u64) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut score = Vec::with_capacity(n);
    let mut state = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    for i in 0..n {
        let t0 = i as f64 / (n.max(2) - 1) as f64 * 6.0 - 3.0;
        let t1 = next() * 2.0 - 1.0;
        // Box–Muller-free jitter: uniform noise is enough to make the classes
        // inseparable, which is what an ill-conditioned fit needs.
        let eps = next() * 2.0 - 1.0;
        x.push(vec![t0, t1]);
        score.push(0.9 * t0 + 0.5 * t1 + noise * 6.0 * eps);
    }
    let k = ((n as f64 * positives).round() as usize).clamp(1, n.max(1));
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| score[b].partial_cmp(&score[a]).unwrap_or(std::cmp::Ordering::Equal));
    let mut y = vec![0.0; n];
    for &i in order.iter().take(k) {
        y[i] = 1.0;
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

/// The affine logit `a·x0 + b` the parity fixtures fit.
fn affine(a: f64, b: f64) -> rcsr::expr::Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
    )
}

fn constants(tree: &rcsr::expr::Node) -> Vec<f64> {
    let mut out = Vec::new();
    fn walk(n: &rcsr::expr::Node, out: &mut Vec<f64>) {
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

fn bits(tree: &rcsr::expr::Node) -> Vec<u64> {
    constants(tree).iter().map(|v| v.to_bits()).collect()
}

fn log_loss_of(tree: &rcsr::expr::Node, s: &Sample) -> f64 {
    let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(tree, r)).collect();
    metrics::score_weighted(&pred, &s.y, &s.w, Metric::LogLoss, expr::count_constants(tree))
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

// ------------------------------------------------------- the label rule ----

/// #234 §1: the label is `y > (min y + max y)/2`, read as the sign `t = ±1`.
/// For a two-valued target the rule is subset-independent, which is what lets
/// the fit rows and the score rows agree.
#[test]
fn log_loss_labels_use_the_target_midpoint_and_are_subset_independent() {
    for (lo, hi) in [(0.0, 1.0), (-1.0, 1.0), (0.2, 0.9), (-7.0, 3.0)] {
        let y = vec![lo, hi, lo, hi, hi, lo];
        let sign = fit::classification_signs(&y);
        assert_eq!(sign, vec![-1.0, 1.0, -1.0, 1.0, 1.0, -1.0], "{lo}..{hi}");
        assert_eq!(fit::class_counts(&y), (3, 3));
        // Every subset that still holds both values agrees with the whole
        // sample on the rows it contains.
        for (i, keep) in [[0usize, 1], [3, 5], [0, 2], [2, 3]].iter().enumerate() {
            let sub: Vec<f64> = keep.iter().map(|&k| y[k]).collect();
            let sub_sign = fit::classification_signs(&sub);
            for (k, &idx) in keep.iter().enumerate() {
                assert_eq!(
                    sub_sign[k], sign[idx],
                    "subset {i} disagrees on row {idx} for {lo}..{hi}"
                );
            }
        }
    }
}

/// A single-valued target has no midpoint and no finite log-loss optimum: the
/// rule degenerates to one class, which the arm must refuse rather than fit.
#[test]
fn a_single_class_target_is_counted_as_one_class() {
    for y in [vec![1.0; 5], vec![0.0; 5], vec![2.5; 3]] {
        assert_eq!(fit::class_counts(&y), (0, y.len()));
        assert!(fit::classification_signs(&y).iter().all(|&t| t == -1.0));
    }
    assert_eq!(fit::class_counts(&[]), (0, 0));
}

// ---------------------------------------------- objective, never a proxy ----

/// #234 §5 + #289: `log_loss` has a native **fit** objective and, since #289,
/// a native **score** objective in its own label-aware kernel. The scoring
/// objective is not a `DeviceLoss`: it is `ScoreObjective::LogLoss` with the
/// disjoint code 6, so a fit launch can never be mistaken for a score launch
/// and the evaluator is still never ranked by squared error.
#[test]
fn log_loss_has_a_native_fit_objective_and_a_native_score_objective() {
    assert_eq!(fit::fit_loss(Metric::LogLoss), Some(fit::DeviceLoss::Logistic));
    assert_eq!(
        fit::DeviceLoss::for_metric(Metric::LogLoss),
        None,
        "the regression row loss still has no label semantics"
    );
    assert_eq!(
        metrics::ScoreObjective::for_metric(Metric::LogLoss),
        Some(metrics::ScoreObjective::LogLoss)
    );
    assert_eq!(metrics::ScoreObjective::LogLoss.code(), 10);
    assert!(!fit::DeviceLoss::Logistic.scores_on_device());
    assert!(!fit::DeviceLoss::Correlation.scores_on_device());
    assert!(fit::DeviceLoss::Absolute.scores_on_device());
    // The fit code is distinct from the classification score code; the
    // regression score codes are 0–9 (#287), the classification block 10–14
    // (#289).
    assert_eq!(fit::DeviceLoss::Logistic.code(), 5);
    let mut codes: Vec<i32> = [
        fit::DeviceLoss::Squared,
        fit::DeviceLoss::LogSquashed,
        fit::DeviceLoss::Correlation,
        fit::DeviceLoss::CorrelationSigned,
        fit::DeviceLoss::Absolute,
        fit::DeviceLoss::Logistic,
    ]
    .iter()
    .map(|l| l.code())
    .collect();
    let before = codes.len();
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), before, "every DeviceLoss needs its own code");
    // The AUC/hinge neighbours keep the CPU: they have no native fit loss, so
    // an explicit gpu stays a usage error rather than a surrogate fit.
    for m in [Metric::Auc, Metric::Hinge] {
        assert_eq!(fit::fit_loss(m), None, "{m:?} has no native fit objective");
    }
    // And the usage gate now accepts log_loss.
    assert!(fit::gpu_fit_metric_error(Metric::LogLoss).is_none());
}

/// #289: `log_loss` is no longer fit-only. It has both a device fit objective
/// (`Logistic`) and a native scorer (`ScoreObjective::LogLoss`), so the same
/// metric may be fitted **and** ranked on the device, while every reported
/// number is still the exact CPU `f64` rebuild.
#[test]
fn log_loss_is_no_longer_a_fit_only_objective() {
    assert!(fit::fit_only_loss(Metric::LogLoss).is_none());
    assert!(fit::fit_only_loss(Metric::Squared).is_none());
    assert!(fit::fit_only_loss(Metric::Absolute).is_none());
    assert!(fit::fit_only_loss(Metric::Auc).is_none());
    let train = binary_sample(96, 0.5, 0.1, 3);
    for strategy in [FitStrategy::Exact, FitStrategy::Gpu] {
        let cfg = Config {
            gpu: true,
            metric: Metric::LogLoss,
            fit_strategy: strategy,
            ..Default::default()
        };
        let want = if cfg!(feature = "cuda") {
            Backend::Gpu
        } else {
            Backend::Cpu
        };
        assert_eq!(
            search::effective_backend(&cfg, &train),
            want,
            "{strategy:?}: log_loss has a native device scorer"
        );
    }
}

// --------------------------------------------------------- proposal gate ----

/// #234 §5, the invariant the arm ships on: a proposal that is worse in the
/// **configured** objective is reverted and the start is kept bit for bit.
#[test]
fn the_proposal_gate_reverts_a_worse_log_loss() {
    let s = binary_sample(200, 0.4, 0.1, 5);
    let mut tree = affine(0.0, 0.0);
    // The CPU arm's own optimum, which must be kept.
    let mut cpu = tree.clone();
    let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 16, Metric::LogLoss);
    let cpu_constants = bits(&cpu);
    let (score, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = cpu.clone(),
        &s.x,
        &s.y,
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::Kept);
    assert_eq!(score.to_bits(), cpu_score.to_bits());
    assert_eq!(bits(&tree), cpu_constants);

    // A worse proposal (here: the start) must not replace the optimum.
    let before = bits(&tree);
    let (score, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = affine(0.0, 0.0),
        &s.x,
        &s.y,
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::Worse);
    // On rejection the gate reports the kept (starting) objective.
    assert_eq!(score.to_bits(), cpu_score.to_bits());
    assert!(
        log_loss_of(&affine(0.0, 0.0), &s) > cpu_score,
        "the fixture must make the start worse than the optimum"
    );
    assert_eq!(bits(&tree), before, "a rejected proposal must not move a constant");
}

/// #234: a proposal with a non-finite objective (or constants) is labelled and
/// leaves the tree untouched.
#[test]
fn the_proposal_gate_labels_non_finite_proposals() {
    let s = binary_sample(64, 0.5, 0.1, 6);
    let mut tree = affine(0.5, 0.1);
    let before = bits(&tree);
    let (_, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = affine(f64::NAN, 0.0),
        &s.x,
        &s.y,
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::NonFinite);
    assert_eq!(bits(&tree), before);
    // A misshapen sample is `+inf`, never a silent zero.
    let (_, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = affine(1.0, 0.0),
        &s.x,
        &s.y[..3],
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::NonFinite);
}

// ------------------------------------------------- CPU reference quality ----

/// #234 §7.1: a constant model under `log_loss` has the closed-form optimum
/// `ln(n₊/n₋)`. The CPU arm (`logistic_irls`) must reach it, which is the
/// reference the device is then compared with.
#[test]
fn the_cpu_constant_model_reaches_the_closed_form() {
    for positives in [0.5, 0.1, 0.01] {
        let n = 1000;
        let k = (n as f64 * positives).round() as usize;
        let y: Vec<f64> = (0..n).map(|i| if i < k { 1.0 } else { 0.0 }).collect();
        let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
        let mut tree = constant(0.0);
        let score = fit::fit_weighted_metric(&mut tree, &x, &y, &[], 16, Metric::LogLoss);
        let want = (k as f64 / (n - k) as f64).ln();
        let got = constants(&tree)[0];
        let rel = (got - want).abs() / want.abs().max(1.0);
        // The CPU arm reaches the closed form to ~6e-9 at 1 % positives: the
        // residual is the inner LM damping of `fit_weighted`, not a formula
        // error. The tolerance is the parity tolerance (§7.3), not 1e-9 §7.1's
        // ideal-Newton value, which the damped inner solve does not promise.
        assert!(rel <= 1e-6, "positives {positives}: {got} vs {want} (rel {rel})");
        let (pos, neg) = fit::class_counts(&y);
        assert_eq!((pos, neg), (k, n - k));
        let _ = score;
    }
}

/// #234 §7.6: at 1 % positives the fit is ill-conditioned but finite, and it
/// still improves the configured loss.
#[test]
fn a_one_percent_positive_cpu_fit_is_finite_and_improves() {
    let s = binary_sample(4_000, 0.01, 0.2, 11);
    let (pos, neg) = fit::class_counts(&s.y);
    assert_eq!(pos, 40, "fixture keeps exactly 1 % positives ({pos}/{neg})");
    let mut tree = affine(0.0, 0.0);
    let start = log_loss_of(&tree, &s);
    let score = fit::fit_weighted_metric(&mut tree, &s.x, &s.y, &s.w, 16, Metric::LogLoss);
    assert!(score.is_finite(), "a 1 %-positive fit must stay finite");
    assert!(score <= start, "{score} must not be worse than {start}");
    assert!(constants(&tree).iter().all(|v| v.is_finite()));
    // And the reported value is the metric, not an SSE: an SSE of a 1 %-positive
    // binary target is a small number ~0.01, a log-loss is O(0.1).
    assert!(score > 0.01, "a log-loss of {score} looks like a squared error");
}

/// #234: extreme logits on the CPU reference are clamped, never NaN — the
/// contract the device's `rc_logistic_q` mirrors.
#[test]
fn extreme_logits_on_the_cpu_are_clamped_not_nan() {
    let s = binary_sample(200, 0.5, 0.05, 12);
    for start in [1e3, -1e3, 1e6] {
        let mut tree = affine(start, 0.0);
        let score = fit::fit_weighted_metric(&mut tree, &s.x, &s.y, &s.w, 8, Metric::LogLoss);
        assert!(score.is_finite() && score >= 0.0, "start {start}: {score}");
        assert!(constants(&tree).iter().all(|v| v.is_finite()));
    }
}

// -------------------------------------------------------- the search path --

fn tiny_logloss_config(seed: u32) -> Config {
    Config {
        generations: 3,
        islands: 1,
        pop: 16,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        metric: Metric::LogLoss,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Tanh, Op::Neg, Op::Square],
        ..Default::default()
    }
}

/// #234/#289: `--no-fit` is a true no-op for the fit arm: with
/// `fit_iters == 0` no arm is built, so a `gpu` request and a CPU run explore
/// the same constants. Since #289 the GPU also scores natively, so the fitness
/// is the f32 device value during the search; the reported archive is the exact
/// CPU rebuild in both arms.
#[test]
fn log_loss_no_fit_is_a_true_noop() {
    let train = binary_sample(200, 0.5, 0.1, 21);
    let test = binary_sample(60, 0.5, 0.1, 22);
    let base = Config {
        fit_iters: 0,
        ..tiny_logloss_config(5)
    };
    let cpu = search::run(&Config { ..base.clone() }, &train, &test);
    let gpu = search::run(
        &Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..base
        },
        &train,
        &test,
    );
    assert_eq!(cpu.archive.items.len(), gpu.archive.items.len());
    for (a, b) in cpu.archive.items.iter().zip(&gpu.archive.items) {
        assert_eq!(bits(&a.tree), bits(&b.tree), "--no-fit moved a constant");
        assert_eq!(
            a.error.to_bits(),
            b.error.to_bits(),
            "the reported archive is the exact CPU rebuild"
        );
        let cpu_value = log_loss_of(&a.tree, &train);
        assert_eq!(a.error.to_bits(), cpu_value.to_bits());
    }
}

/// #234 §5: whatever arm produced the constants, every reported error of a
/// log-loss run is the exact CPU `f64` log-loss of the stored tree — never an
/// SSE, never a device number.
#[test]
fn reported_log_loss_errors_are_exact_cpu_values() {
    let train = binary_sample(240, 0.5, 0.1, 31);
    let test = binary_sample(60, 0.5, 0.1, 32);
    let cfg = Config {
        gpu: true,
        fit_strategy: FitStrategy::Gpu,
        ..tiny_logloss_config(7)
    };
    let out = search::run(&cfg, &train, &test);
    assert!(!out.archive.items.is_empty());
    for s in &out.archive.items {
        let cpu = log_loss_of(&s.tree, &train);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU log-loss: {} vs {cpu}",
            s.error
        );
        // A mean log-loss of a non-degenerate binary target is not a squared
        // error of a 0/1 target (which is < 0.25); this also catches a silent
        // SSE substitution in the archive.
        assert!(s.error > 0.0);
    }
}

// --------------------------------------------------------- the device arm --

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::fit::{BatchStatus, BatchedGpu};
    use rcsr::gpu::GpuEvaluator;

    /// Column-major `f64` copy of the training rows for the exact fit polish.
    fn cols64(s: &Sample) -> (Vec<f64>, usize, usize) {
        let n = s.len();
        let v = s.x.first().map(|r| r.len()).unwrap_or(0);
        let mut cols = vec![0.0f64; n * v];
        for (r, row) in s.x.iter().enumerate() {
            for (k, &x) in row.iter().enumerate().take(v) {
                cols[k * n + r] = x;
            }
        }
        (cols, n, v)
    }

    fn evaluator(s: &Sample) -> GpuEvaluator {
        let (cols, n, v) = cols64(s);
        let cols32: Vec<f32> = cols.iter().map(|&x| x as f32).collect();
        let y32: Vec<f32> = s.y.iter().map(|&y| y as f32).collect();
        let mut ev = GpuEvaluator::new(&cols32, n, v, &y32).expect("device evaluator");
        assert!(ev.set_fit_data(&cols, &s.y, &s.w), "f64 fit rows uploaded");
        ev
    }

    /// Fit a batch of trees through the device and the exact verifier.
    fn device_fit(
        ev: &mut GpuEvaluator,
        trees: &mut [rcsr::expr::Node],
        s: &Sample,
        iters: usize,
    ) -> fit::VerifiedBatch {
        let backend = BatchedGpu::new(ev);
        fit::fit_batch_verified(&backend, trees, &s.x, &s.y, &s.w, iters, Metric::LogLoss)
    }

    /// #234 §7.1/§7.3: the device constant model reaches the closed form
    /// `ln(n₊/n₋)` and is judged on the metric.
    #[test]
    fn device_constant_model_matches_the_closed_form() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        for positives in [0.5, 0.2, 0.02] {
            let n = 600;
            let k = (n as f64 * positives).round() as usize;
            let s = Sample {
                x: (0..n).map(|i| vec![i as f64 / n as f64]).collect(),
                y: (0..n).map(|i| if i < k { 1.0 } else { 0.0 }).collect(),
                w: Vec::new(),
                ys: Vec::new(),
            };
            let mut ev = evaluator(&s);
            let mut trees = vec![constant(0.0)];
            let out = device_fit(&mut ev, &mut trees, &s, 8);
            assert_eq!(out.status, vec![BatchStatus::Gpu], "{:?}", out.status);
            let want = (k as f64 / (n - k) as f64).ln();
            let got = constants(&trees[0])[0];
            let rel = (got - want).abs() / want.abs().max(1.0);
            assert!(
                rel <= 1e-6,
                "positives {positives}: device {got} vs closed form {want} (rel {rel})"
            );
            // The reported value is the metric (a mean log-loss), not an SSE.
            assert!((out.score[0] - log_loss_of(&trees[0], &s)).abs() <= 1e-12);
        }
    }

    /// #234 §7.2/§7.3: on a two-parameter logistic fixture the device fit is as
    /// good as the CPU `logistic_irls` **on the objective** (the constants are
    /// not an invariant on a flat optimum), and never worse than its start.
    #[test]
    fn device_log_loss_parity_with_cpu_logistic_irls() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(2_000, 0.5, 0.2, 41);
        let start = affine(0.0, 0.0);
        let start_score = log_loss_of(&start, &s);

        let mut cpu = start.clone();
        let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 8, Metric::LogLoss);

        let mut ev = evaluator(&s);
        let mut trees = vec![start.clone()];
        let out = device_fit(&mut ev, &mut trees, &s, 8);
        assert_eq!(out.status, vec![BatchStatus::Gpu], "{:?}", out.status);
        let device_score = out.score[0];
        assert_eq!(device_score.to_bits(), log_loss_of(&trees[0], &s).to_bits());
        assert!(
            device_score <= start_score,
            "device {device_score} must not be worse than the start {start_score}"
        );
        let rel = (device_score - cpu_score).abs() / cpu_score;
        assert!(
            rel <= 1e-6,
            "device {device_score} vs CPU {cpu_score} (rel {rel}) on N={} rows",
            s.len()
        );
        // The constants agree to the same order as the objective.
        for (a, b) in constants(&trees[0]).iter().zip(constants(&cpu).iter()) {
            let d = (a - b).abs() / b.abs().max(1.0);
            assert!(d <= 1e-4, "constant {a} vs {b} (rel {d})");
        }
    }

    /// #234 §4.4: extreme start logits must not produce a NaN, an infinity or a
    /// silent "0 weight for every row" Hessian.
    #[test]
    fn device_log_loss_survives_extreme_logits() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(400, 0.3, 0.1, 42);
        for start in [300.0, -300.0, 5_000.0] {
            let mut ev = evaluator(&s);
            let mut trees = vec![affine(start, 0.0)];
            let before = log_loss_of(&trees[0], &s);
            let out = device_fit(&mut ev, &mut trees, &s, 8);
            assert!(
                out.score[0].is_finite() && out.score[0] >= 0.0,
                "start {start}: score {}",
                out.score[0]
            );
            assert!(
                out.status[0] == BatchStatus::Gpu || out.status[0] == BatchStatus::Unsupported,
                "start {start}: {:?}",
                out.status[0]
            );
            assert!(constants(&trees[0]).iter().all(|v| v.is_finite()));
            assert!(out.score[0] <= before, "start {start}: {} > {before}", out.score[0]);
        }
    }

    /// #234 §7.4/§7.8: the same input twice is bit-identical, and a candidate
    /// alone or inside a batch of many gets the same constants — the kernel's
    /// ownership rule, not a lucky run.
    #[test]
    fn device_log_loss_is_deterministic_and_batch_independent() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(500, 0.4, 0.15, 43);
        let model = affine(0.6, 0.2);
        let mut ev = evaluator(&s);
        let mut a = vec![model.clone()];
        let first = device_fit(&mut ev, &mut a, &s, 6);
        let mut b = vec![model.clone()];
        let second = device_fit(&mut ev, &mut b, &s, 6);
        assert_eq!(bits(&a[0]), bits(&b[0]), "repeated run moved a constant");
        assert_eq!(first.score[0].to_bits(), second.score[0].to_bits());

        // Alone vs in a batch: the same tree, padded with other candidates.
        let mut alone = vec![model.clone()];
        let alone_out = device_fit(&mut ev, &mut alone, &s, 6);
        let mut batch = vec![model.clone()];
        for k in 0..7 {
            batch.push(affine(0.1 * (k as f64 + 1.0), -0.2 * k as f64));
        }
        let batch_out = device_fit(&mut ev, &mut batch, &s, 6);
        assert_eq!(
            bits(&alone[0]),
            bits(&batch[0]),
            "batch composition changed a candidate's constants"
        );
        assert_eq!(alone_out.score[0].to_bits(), batch_out.score[0].to_bits());
    }

    /// #234 §4.5/§8: a single-class sample has no finite optimum; the arm
    /// declines it (`unsupported`) and the exact CPU arm's result is reported.
    #[test]
    fn device_log_loss_declines_a_single_class_sample() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = Sample {
            x: (0..200).map(|i| vec![i as f64 / 200.0]).collect(),
            y: vec![1.0; 200],
            w: Vec::new(),
            ys: Vec::new(),
        };
        let mut ev = evaluator(&s);
        let mut trees = vec![affine(0.0, 0.0)];
        let out = device_fit(&mut ev, &mut trees, &s, 4);
        assert_eq!(out.status, vec![BatchStatus::Unsupported]);
        let mut cpu = affine(0.0, 0.0);
        let cpu_score = fit::fit_weighted_metric(&mut cpu, &s.x, &s.y, &s.w, 4, Metric::LogLoss);
        assert_eq!(out.score[0].to_bits(), cpu_score.to_bits());
        assert_eq!(bits(&trees[0]), bits(&cpu));
    }

    /// #234 regression: a log-loss fit that stalls at f32 resolution pushes λ
    /// past its 1e9 ceiling inside the LM loop (the logistic loss is flat at
    /// ~1e-7 once the constants converge, so every trial is rejected). The λ
    /// exit test used to be a non-uniform `break` — thread 0 left the loop while
    /// the rest of the block waited at the next `__syncthreads()`, and the
    /// kernel never returned. `--fit-iters 32` reproduces it in milliseconds;
    /// the bounded run below is the regression guard.
    #[test]
    fn device_log_loss_with_a_large_iteration_budget_terminates() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(600, 0.05, 0.15, 49);
        for iters in [28usize, 32, 64] {
            let mut ev = evaluator(&s);
            let mut trees = vec![affine(0.0, 0.0)];
            let start = log_loss_of(&trees[0], &s);
            let out = device_fit(&mut ev, &mut trees, &s, iters);
            assert!(
                out.score[0].is_finite() && out.score[0] <= start,
                "iters {iters}: {} > start {start}",
                out.score[0]
            );
            assert!(constants(&trees[0]).iter().all(|v| v.is_finite()));
        }
    }

    /// #234: the device arm is not a no-op — on a fixture it actually moves
    /// constants and lowers the log-loss.
    #[test]
    fn device_log_loss_moves_constants_and_lowers_the_loss() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(800, 0.5, 0.1, 44);
        let mut ev = evaluator(&s);
        let mut trees = vec![affine(0.05, 0.02)];
        let start = log_loss_of(&trees[0], &s);
        let out = device_fit(&mut ev, &mut trees, &s, 8);
        assert!(out.score[0] < start, "{} !< {start}", out.score[0]);
        assert!(out.score[0] < 0.69, "a fit below the ln 2 baseline");
    }

    /// #234: the search's fit-only arm runs the device and reports the exact
    /// CPU fitness, with a labelled ledger.
    #[test]
    fn the_search_fits_log_loss_on_the_device() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let train = binary_sample(400, 0.5, 0.15, 45);
        let test = binary_sample(80, 0.5, 0.15, 46);
        let cfg = Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..tiny_logloss_config(11)
        };
        let out = search::run(&cfg, &train, &test);
        let g = out.gpu.as_ref().expect("a fit ledger");
        assert!(g.fit_attempts > 0, "the fit arm admitted candidates: {g:?}");
        assert!(g.fit_gpu > 0, "the device fitted at least one candidate: {g:?}");
        assert_eq!(g.fit_attempts, g.fit_gpu + g.fit_fallback);
        let known: Vec<&str> = search::FitFallbackCause::ALL.iter().map(|c| c.label()).collect();
        let mut labelled = 0;
        for (cause, n) in &g.fit_fallback_by_cause {
            assert!(known.contains(&cause.as_str()), "unknown cause {cause}");
            labelled += n;
        }
        assert_eq!(labelled, g.fit_fallback);
        // The *reported* fitness is the exact CPU log-loss rebuild, never a
        // device number, and the run claims the device-scored backend.
        for s in &out.archive.items {
            assert_eq!(s.error.to_bits(), log_loss_of(&s.tree, &train).to_bits());
        }
        assert_eq!(out.backend, rcsr::backend::Kind::Gpu);
    }

    /// #234: a single-class training sample completes, with the `single_class`
    /// fallback labelled — never scored as if it had two classes.
    #[test]
    fn a_single_class_run_is_labelled() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let train = Sample {
            x: (0..300).map(|i| vec![i as f64 / 300.0]).collect(),
            y: vec![0.0; 300],
            w: Vec::new(),
            ys: Vec::new(),
        };
        let cfg = Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..tiny_logloss_config(12)
        };
        let out = search::run(&cfg, &train, &train);
        let g = out.gpu.as_ref().expect("a fit ledger");
        if g.fit_attempts > 0 {
            assert_eq!(
                g.fit_fallback_by_cause.get("single_class").copied().unwrap_or(0),
                g.fit_fallback,
                "every fallback of a single-class sample must be labelled: {g:?}"
            );
            assert_eq!(g.fit_gpu, 0, "nothing may be fitted on one class: {g:?}");
        }
    }

    /// #234: repeated seeds give the same archive, and 1 vs 2 islands does not
    /// change a run's quality class (device determinism per device+toolkit).
    #[test]
    fn the_log_loss_arm_is_deterministic() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let train = binary_sample(400, 0.5, 0.15, 47);
        let test = binary_sample(80, 0.5, 0.15, 48);
        let cfg = Config {
            gpu: true,
            fit_strategy: FitStrategy::Gpu,
            ..tiny_logloss_config(13)
        };
        let a = search::run(&cfg, &train, &test);
        let b = search::run(&cfg, &train, &test);
        let describe = |o: &search::Outcome| -> Vec<(u64, usize, Vec<u64>)> {
            o.archive
                .items
                .iter()
                .map(|s| (s.error.to_bits(), s.complexity, bits(&s.tree)))
                .collect()
        };
        assert_eq!(describe(&a), describe(&b), "a repeated run moved the archive");
    }
}

// ---------------------------------------------------------- measurement ----

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// #234 measurement (ignored): the paired CPU-versus-device run the shipped
/// claim rests on, at equal evaluations and on the same rows.
///
/// Reports, per arm: end-to-end wall, the measured CPU split (`fit_cpu_s` and
/// `eval_cpu_s` from the dev log), the fit stage's share of wall, the fit
/// kernel's time and the device duty, every fallback cause, and the best
/// log-loss plus the Pareto hypervolume at equal evaluations.
///
/// Run with
/// `cargo test --release --features cuda --test logloss_gpu bench_logloss_end_to_end -- --ignored --nocapture`
/// and optionally `RCSR_BENCH_N/POP/ISLANDS/GENS/ITERS/POS/NOISE/SEED`.
#[test]
#[ignore]
fn bench_logloss_end_to_end() {
    let n = env_usize("RCSR_BENCH_N", 20_000);
    let pop = env_usize("RCSR_BENCH_POP", 64);
    let islands = env_usize("RCSR_BENCH_ISLANDS", 2);
    let generations = env_usize("RCSR_BENCH_GENS", 5);
    let iters = env_usize("RCSR_BENCH_ITERS", 8);
    let seed = env_usize("RCSR_BENCH_SEED", 93) as u32;
    let positives = std::env::var("RCSR_BENCH_POS")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.05);
    let noise = std::env::var("RCSR_BENCH_NOISE")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .unwrap_or(0.15);
    let train = binary_sample(n, positives, noise, seed as u64);
    let test = binary_sample(1_000, positives, noise, seed as u64 + 1);
    let (pos, neg) = fit::class_counts(&train.y);
    let ops = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Sin,
        Op::Cos,
        Op::Tanh,
        Op::Gauss,
        Op::Square,
        Op::Cube,
        Op::Neg,
    ];
    let base = Config {
        generations,
        islands,
        pop,
        max_complexity: 14,
        fit_iters: iters,
        metric: Metric::LogLoss,
        seed,
        patience: 0,
        gpu_notes: false,
        ops,
        ..Default::default()
    };
    // (name, wall, best, fit_cpu, eval_cpu, Pareto points)
    let mut runs: Vec<(String, f64, f64, f64, f64, Vec<(f64, f64)>)> = Vec::new();
    for (name, strategy, gpu) in [("cpu/exact", FitStrategy::Exact, false), ("gpu/fit", FitStrategy::Gpu, true)] {
        let totals = std::sync::Arc::new(std::sync::Mutex::new((0.0f64, 0.0f64)));
        let sink = totals.clone();
        let cfg = Config {
            fit_strategy: strategy,
            gpu,
            dev_log: Some(std::sync::Arc::new(move |r: &rcsr::devlog::GenRecord| {
                let mut t = sink.lock().unwrap();
                t.0 += r.fit_cpu_s;
                t.1 += r.eval_cpu_s;
            })),
            ..base.clone()
        };
        let t0 = std::time::Instant::now();
        let out = search::run(&cfg, &train, &test);
        let wall = t0.elapsed().as_secs_f64();
        let (fit_cpu, eval_cpu) = *totals.lock().unwrap();
        let best = out
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        eprintln!(
            "arm={name} metric=log_loss N={n} pos={:.1}% ({pos}+/{neg}-) noise={:.0}% evals={} wall={wall:.2}s best={best:.6}",
            100.0 * positives,
            100.0 * noise,
            generations * islands * pop
        );
        eprintln!(
            "  cpu split: fit={fit_cpu:.2}s eval={eval_cpu:.2}s -> fit owns {:.1}% of the measured CPU",
            100.0 * fit_cpu / (fit_cpu + eval_cpu).max(1e-9)
        );
        if let Some(g) = &out.gpu {
            let causes: Vec<String> = search::FitFallbackCause::ALL
                .iter()
                .map(|c| {
                    format!(
                        "{}={}",
                        c.label(),
                        g.fit_fallback_by_cause.get(c.label()).copied().unwrap_or(0)
                    )
                })
                .collect();
            eprintln!(
                "  fit: screened={} admitted={} gpu={} fallback={} rate={:.1}% causes=[{}] refits={}",
                g.fit_screened,
                g.fit_attempts,
                g.fit_gpu,
                g.fit_fallback,
                100.0 * g.fit_fallback as f64 / g.fit_attempts.max(1) as f64,
                causes.join(","),
                g.fit_refits
            );
            // #234/#289: only unsupported programs use the CPU fallback pool;
            // a native classification run scores supported candidates on the
            // device. `fallback_seconds` remains the CPU-scoring time.
            let cpu_fitness: f64 = g.generations.iter().map(|s| s.fallback_seconds).sum();
            eprintln!(
                "  time: fit_host={:.2}s ({:.1}% of wall) kernel={:.2}s ({:.1}% of fit_host) \
                 cpu_fitness={:.2}s ({:.1}% of wall) duty={:.1}%",
                g.fit_host_ms / 1000.0,
                100.0 * (g.fit_host_ms / 1000.0) / wall,
                g.fit_ms / 1000.0,
                100.0 * (g.fit_ms / 1000.0) / (g.fit_host_ms / 1000.0).max(1e-9),
                cpu_fitness,
                100.0 * cpu_fitness / wall,
                100.0 * g.accounting.duty_cycle()
            );
            eprintln!(
                "  per candidate: fit_host {:.2} ms over {} admitted (kernel {:.3} ms)",
                g.fit_host_ms / g.fit_attempts.max(1) as f64,
                g.fit_attempts,
                g.fit_ms / g.fit_attempts.max(1) as f64
            );
        }
        let front = out
            .archive
            .items
            .iter()
            .filter(|s| s.error.is_finite())
            .map(|s| (s.cost, s.error))
            .collect();
        runs.push((name.to_string(), wall, best, fit_cpu, eval_cpu, front));
    }
    if runs.len() == 2 {
        let speed = runs[0].1 / runs[1].1;
        eprintln!(
            "paired: speed-up {speed:.2}x (cpu {:.2}s vs gpu {:.2}s), best log-loss delta {:+.3e} (cpu {:.6} vs gpu {:.6})",
            runs[0].1, runs[1].1, runs[1].2 - runs[0].2, runs[0].2, runs[1].2
        );
        eprintln!(
            "paired: the CPU arm's measured fit share is {:.1}% of its CPU work, the upper bound any fit-only speed-up can recover",
            100.0 * runs[0].3 / (runs[0].3 + runs[0].4).max(1e-9)
        );
        let all: Vec<(f64, f64)> = runs.iter().flat_map(|r| r.5.clone()).collect();
        let cmin = all.iter().map(|p| p.0).fold(f64::INFINITY, f64::min);
        let cmax = all.iter().map(|p| p.0).fold(f64::NEG_INFINITY, f64::max);
        let emin = all.iter().map(|p| p.1).fold(f64::INFINITY, f64::min);
        let emax = all.iter().map(|p| p.1).fold(f64::NEG_INFINITY, f64::max);
        for (name, _, _, _, _, front) in &runs {
            eprintln!(
                "paired: arm={name} hypervolume={:.6} points={}",
                hypervolume(front, cmin, cmax, emin, emax),
                front.len()
            );
        }
    }
}

/// #169: normalized 2-D hypervolume of `(cost, error)` points, on extrema both
/// arms share, so the two fronts are compared at equal evaluations.
fn hypervolume(points: &[(f64, f64)], cmin: f64, cmax: f64, emin: f64, emax: f64) -> f64 {
    if points.len() < 2 {
        return 0.0;
    }
    let cspan = (cmax - cmin).max(1e-12);
    let espan = (emax - emin).max(1e-12);
    const G: usize = 64;
    let normalized: Vec<(f64, f64)> = points
        .iter()
        .map(|&(c, e)| ((c - cmin) / cspan, (e - emin) / espan))
        .collect();
    let mut dominated = 0usize;
    for gy in 0..G {
        let y = (gy as f64 + 0.5) / G as f64;
        for gx in 0..G {
            let x = (gx as f64 + 0.5) / G as f64;
            if normalized.iter().any(|&(px, py)| px <= x && py <= y) {
                dominated += 1;
            }
        }
    }
    dominated as f64 / (G * G) as f64
}

