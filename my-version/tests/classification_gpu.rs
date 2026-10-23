//! #289: weighted rows and native classification fitness on the device.
//!
//! What this file pins, in the order the design claims it:
//!
//! 1. **definitions** — `log_loss` is the stable softplus, `brier` is the
//!    proper Brier score on `σ(p)`, and `hinge` is explicitly the historical
//!    bounded sigmoid surrogate, with `hinge_ordinary`/`hinge_squared` as the
//!    separate SVM forms;
//! 2. **weights** — every classification metric is `Σ wᵢ·lossᵢ / Σwᵢ`, equal
//!    to duplicating a row; a zero-weight row is skipped before its prediction
//!    or label is read, so its `NaN`/`∞` cannot poison the score;
//! 3. **native scoring** — device vs CPU parity on every objective, on the
//!    same f32 rows (`≤1e-6`), with f32 rows within `1e-5` of the f64 model and
//!    no rank inversion;
//! 4. **determinism** — one candidate alone, in a batch, and on a repeated
//!    call yields the same bits; the weighted denominator is `Σw`;
//! 5. **the search** — a classification run claims the device backend, the
//!    archive is the exact CPU `f64` rebuild, and the proposal gate judges the
//!    configured metric (log-loss cannot replace Brier and vice versa);
//! 6. **the measurement** (`bench_classification_fitness`, ignored) — paired
//!    native-fitness vs CPU scorer cost at large `N`.
//!
//! Device tests skip cleanly without CUDA. Every pure-CPU invariant runs in
//! every build.

use rcsr::data::Sample;
use rcsr::expr::{self, constant, input, node, Op};
use rcsr::fit::{self, ProposalGate};
use rcsr::metrics::{self, Metric, ScoreObjective};
use rcsr::search::{self, Config};

// ---------------------------------------------------------------- fixtures --

/// A deterministic binary sample with exactly `positives · n` positive labels
/// (the largest latent scores), so fixtures control class balance to the row.
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

fn affine(a: f64, b: f64) -> rcsr::expr::Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
    )
}

fn bits(tree: &rcsr::expr::Node) -> Vec<u64> {
    fn walk(n: &rcsr::expr::Node, out: &mut Vec<u64>) {
        if n.op == Op::Const {
            out.push(n.val.to_bits());
        }
        for k in &n.kids {
            walk(k, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

/// CPU score of the configured classification metric on the sample's weights.
fn cpu_score(tree: &rcsr::expr::Node, s: &Sample, m: Metric) -> f64 {
    let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(tree, r)).collect();
    metrics::score_weighted(&pred, &s.y, &s.w, m, expr::count_constants(tree))
}

fn classification_metrics() -> [Metric; 5] {
    [
        Metric::LogLoss,
        Metric::Brier,
        Metric::Hinge,
        Metric::HingeOrdinary,
        Metric::HingeSquared,
    ]
}

// ------------------------------------------------------------ definitions ----

/// #289 §6.1/§6.6: the objective map is total on the classification family
/// except `auc`, which is a rank statistic (#288), and every code is distinct
/// from the fit codes 0–5.
#[test]
fn score_objects_are_named_and_disjoint_from_fit_codes() {
    let expected = [
        (Metric::LogLoss, ScoreObjective::LogLoss),
        (Metric::Brier, ScoreObjective::Brier),
        (Metric::Hinge, ScoreObjective::HingeSigmoid),
        (Metric::HingeOrdinary, ScoreObjective::HingeOrdinary),
        (Metric::HingeSquared, ScoreObjective::HingeSquared),
    ];
    let mut codes = Vec::new();
    for (m, want) in expected {
        assert_eq!(ScoreObjective::for_metric(m), Some(want), "{}", m.name());
        assert_eq!(want.metric(), m);
        assert_eq!(want.name(), m.name());
        codes.push(want.code());
    }
    assert_eq!(codes, vec![10, 11, 12, 13, 14]);
    codes.sort_unstable();
    codes.dedup();
    assert_eq!(codes.len(), 5, "each score objective needs its own code");
    assert_eq!(ScoreObjective::for_metric(Metric::Auc), None);
    // The regression score codes are 0–9 (#287) and the fit codes 0–5; the
    // classification block starts after both, so no launch is ambiguous.
    for loss in [
        fit::DeviceLoss::Squared,
        fit::DeviceLoss::LogSquashed,
        fit::DeviceLoss::Correlation,
        fit::DeviceLoss::CorrelationSigned,
        fit::DeviceLoss::Absolute,
        fit::DeviceLoss::Logistic,
    ] {
        assert!(loss.code() < ScoreObjective::LogLoss.code());
    }
}

/// #289 §6.6: improving log-loss does not imply improving Brier.
///
/// The fixture mixes 100 near-saturated positives (logit 2 → 3, where log-loss
/// still moves but Brier barely does) with 10 negatives at the decision
/// boundary (logit 0 → 1, where Brier moves most). The aggregate log-loss falls
/// while the aggregate Brier rises, so the gate must keep the start under
/// `--metric brier` and accept the same proposal under `--metric log_loss`.
#[test]
fn the_gate_judges_the_configured_classification_metric() {
    let s = Sample {
        // x0 = 0 for the positives, 1 for the negatives. The start logit is
        // 2 − 2x0 and the proposal is 3 − 2x0.
        x: (0..110)
            .map(|i| vec![if i < 100 { 0.0 } else { 1.0 }])
            .collect(),
        y: (0..110).map(|i| if i < 100 { 1.0 } else { 0.0 }).collect(),
        w: Vec::new(),
        ys: Vec::new(),
    };
    // `affine(slope, intercept)`: logit 2 − 2·x0 and 3 − 2·x0.
    let start = affine(-2.0, 2.0);
    let proposal = affine(-2.0, 3.0);
    let ll = |t: &rcsr::expr::Node| cpu_score(t, &s, Metric::LogLoss);
    let br = |t: &rcsr::expr::Node| cpu_score(t, &s, Metric::Brier);
    assert!(
        ll(&proposal) < ll(&start),
        "fixture: the proposal must improve log-loss ({} vs {})",
        ll(&proposal),
        ll(&start)
    );
    assert!(
        br(&proposal) > br(&start),
        "fixture: the proposal must worsen Brier ({} vs {})",
        br(&proposal),
        br(&start)
    );
    // Under Brier the worse proposal is rejected, bit for bit.
    let mut tree = start.clone();
    let (score, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = proposal.clone(),
        &s.x,
        &s.y,
        &s.w,
        Metric::Brier,
    );
    assert_eq!(verdict, ProposalGate::Worse);
    assert_eq!(score.to_bits(), br(&start).to_bits());
    assert_eq!(bits(&tree), bits(&start));
    // Under log-loss the same proposal is kept.
    let (score, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = proposal.clone(),
        &s.x,
        &s.y,
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::Kept);
    assert_eq!(score.to_bits(), ll(&proposal).to_bits());
}

/// #289 §6.6: the AUC gate is independent of the log-loss gate. A proposal
/// that fixes one catastrophically mis-ranked positive but flips a
/// correctly-ranked pair improves log-loss and *lowers* AUC, so it must be
/// rejected under `--metric auc` and accepted under `--metric log_loss`.
#[test]
fn log_loss_improvement_cannot_replace_a_worse_auc() {
    // x = 0 (positive, logit -100), 1 (negative, logit 0), 2 (positive, 1).
    let s = Sample {
        x: (0..3).map(|i| vec![i as f64]).collect(),
        y: vec![1.0, 0.0, 1.0],
        w: Vec::new(),
        ys: Vec::new(),
    };
    let levels = |a: f64, b: f64, c: f64| {
        let c1 = node(Op::Less, vec![input(0), constant(0.5)]);
        let c2 = node(Op::Less, vec![input(0), constant(1.5)]);
        node(
            Op::If,
            vec![
                c1,
                constant(a),
                node(Op::If, vec![c2, constant(b), constant(c)]),
            ],
        )
    };
    let start = levels(-100.0, 0.0, 1.0);
    let proposal = levels(-0.5, 2.0, 1.0);
    let auc_of = |t: &rcsr::expr::Node| cpu_score(t, &s, Metric::Auc);
    let ll_of = |t: &rcsr::expr::Node| cpu_score(t, &s, Metric::LogLoss);
    assert!(auc_of(&start) < auc_of(&proposal), "fixture: AUC must fall");
    assert!(ll_of(&proposal) < ll_of(&start), "fixture: log-loss must improve");
    let mut tree = start.clone();
    let (_, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = proposal.clone(),
        &s.x,
        &s.y,
        &s.w,
        Metric::Auc,
    );
    assert_eq!(verdict, ProposalGate::Worse);
    assert_eq!(bits(&tree), bits(&start), "AUC must keep the better ranking");
    let (_, verdict) = fit::gate_proposal(
        &mut tree,
        |t| *t = proposal.clone(),
        &s.x,
        &s.y,
        &s.w,
        Metric::LogLoss,
    );
    assert_eq!(verdict, ProposalGate::Kept);
}

/// #289: the weighted classification family is the documented `Σw` mean, and
/// it is visible in the weight-support list the CLI refuses against.
#[test]
fn weighted_classification_is_supported_and_normalised_by_sum_w() {
    for m in classification_metrics() {
        assert!(metrics::weights_supported(m), "{}", m.name());
        assert!(
            metrics::weighted_metric_names().contains(&m.name()),
            "{} is missing from the CLI list",
            m.name()
        );
    }
    assert!(!metrics::weights_supported(Metric::Auc));
    // Scaling all weights is a no-op up to the last rounding step of the
    // normalisation (the hinge path has one addition more than the others).
    let y = [0.0, 1.0, 1.0, 0.0];
    let p = [-1.0, 0.2, 1.4, -0.3];
    let w = [0.5, 2.0, 0.0, 1.5];
    let scaled: Vec<f64> = w.iter().map(|v| v * 11.0).collect();
    for m in classification_metrics() {
        let a = metrics::score_weighted(&p, &y, &w, m, 0);
        let b = metrics::score_weighted(&p, &y, &scaled, m, 0);
        assert!(
            (a - b).abs() <= 1e-15 + 1e-12 * a.abs(),
            "{}: {a} vs {b}",
            m.name()
        );
    }
}

/// #289: a positive-weight non-finite target is invalid, not a label: the CPU
/// scorer refuses it and the uploaded sign is NaN so the device does too. A
/// masked non-finite target is still ignored.
#[test]
fn a_positive_weight_non_finite_target_is_invalid() {
    let y = [0.0, 1.0, f64::NAN, 1.0];
    let w = [1.0, 1.0, 1.0, 1.0];
    assert!(fit::classification_signs_weighted(&y, &w)
        .iter()
        .all(|v| v.is_nan()));
    assert_eq!(fit::class_counts_weighted(&y, &w), (0, 0));
    assert_eq!(
        metrics::score_weighted(&[0.0, 0.0, 0.0, 0.0], &y, &w, Metric::LogLoss, 0),
        f64::INFINITY
    );
    // Masked: the row is invisible and the other labels are valid.
    let w0 = [1.0, 1.0, 0.0, 1.0];
    let signs = fit::classification_signs_weighted(&y, &w0);
    assert_eq!(signs, vec![-1.0, 1.0, -1.0, 1.0]);
    assert_eq!(fit::class_counts_weighted(&y, &w0), (2, 1));
}

/// #289: on a weighted sample the midpoint and the class counts follow the
/// positive-weight rows only, which is the label the device is given.
#[test]
fn weighted_labels_ignore_masked_rows() {
    let y = [0.0, 1.0, 0.0, 1.0, 1000.0];
    let w = [1.0, 1.0, 1.0, 1.0, 0.0];
    let signs = fit::classification_signs_weighted(&y, &w);
    assert_eq!(signs, vec![-1.0, 1.0, -1.0, 1.0, -1.0]);
    assert_eq!(fit::class_counts_weighted(&y, &w), (2, 2));
    // Unweighted input is exactly the historical full-sample rule.
    assert_eq!(
        fit::classification_signs_weighted(&y, &[]),
        fit::classification_signs(&y)
    );
    assert_eq!(fit::class_counts_weighted(&y, &[]), fit::class_counts(&y));
}

// --------------------------------------------------------- the search gate --

/// #289: `gpu_blocker` admits the classification objectives and refuses `auc`
/// with a named reason instead of the old generic "classification metric". A
/// weighted regression run is still refused because its kernel has no weight
/// path yet.
#[cfg(feature = "cuda")]
#[test]
fn the_device_gate_admits_classification_and_names_auc() {
    let mut s = binary_sample(64, 0.5, 0.1, 1);
    let cfg = |m: Metric| Config {
        gpu: true,
        metric: m,
        ..Default::default()
    };
    for m in classification_metrics() {
        assert_eq!(
            search::backend_blocker_reason(&cfg(m), &s),
            None,
            "{} must be scored natively",
            m.name()
        );
    }
    let why = search::backend_blocker_reason(&cfg(Metric::Auc), &s).unwrap();
    assert!(why.contains("auc") && why.contains("#288"), "{why}");
    // A weighted classification run is admitted; a weighted squared run is not.
    s.w = vec![1.0; s.len()];
    assert_eq!(
        search::backend_blocker_reason(&cfg(Metric::Brier), &s),
        None
    );
    let why = search::backend_blocker_reason(&cfg(Metric::Squared), &s).unwrap();
    assert!(why.contains("weighted"), "{why}");
}

/// #289 §6.6: a classification run with a native GPU scorer claims the device
/// backend, and every reported archive value is the exact CPU `f64` rebuild.
#[cfg(feature = "cuda")]
#[test]
fn the_search_scores_brier_natively_and_reports_exact_cpu_values() {
    let train = binary_sample(240, 0.4, 0.2, 5);
    let test = binary_sample(60, 0.4, 0.2, 6);
    let cfg = Config {
        gpu: true,
        generations: 3,
        islands: 1,
        pop: 16,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 7,
        metric: Metric::Brier,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Tanh, Op::Neg, Op::Square],
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    assert!(!out.archive.items.is_empty());
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu);
    for sol in &out.archive.items {
        assert_eq!(
            sol.error.to_bits(),
            cpu_score(&sol.tree, &train, Metric::Brier).to_bits(),
            "the reported Brier is not the exact CPU rebuild"
        );
        // A Brier score of a 0/1 target is ≤ 0.25, unlike a log-loss.
        assert!(sol.error.is_finite() && sol.error <= 0.2500001);
    }
    // Repeated runs are deterministic.
    let again = search::run(&cfg, &train, &test);
    let describe = |o: &search::Outcome| -> Vec<(u64, usize, Vec<u64>)> {
        o.archive
            .items
            .iter()
            .map(|s| (s.error.to_bits(), s.complexity, bits(&s.tree)))
            .collect()
    };
    assert_eq!(describe(&out), describe(&again));
}

/// #289: a weighted classification run keeps the weights through the device
/// score and reports the exact weighted CPU value.
#[cfg(feature = "cuda")]
#[test]
fn a_weighted_search_scores_the_weighted_mean() {
    let mut train = binary_sample(240, 0.4, 0.2, 8);
    // A mix of frequency weights, a zero-weight row and a non-finite masked
    // target: none of it may change the positive-weight rows.
    train.w = (0..train.len())
        .map(|i| match i % 7 {
            0 => 0.0,
            1 => 2.5,
            _ => 1.0,
        })
        .collect();
    train.y[0] = f64::NAN;
    let test = binary_sample(60, 0.4, 0.2, 9);
    let cfg = Config {
        gpu: true,
        generations: 3,
        islands: 1,
        pop: 16,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 10,
        metric: Metric::LogLoss,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Tanh, Op::Neg],
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    assert!(!out.archive.items.is_empty());
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu);
    for sol in &out.archive.items {
        assert_eq!(
            sol.error.to_bits(),
            cpu_score(&sol.tree, &train, Metric::LogLoss).to_bits(),
            "the reported weighted log-loss is not the exact CPU rebuild"
        );
    }
}

// ---------------------------------------------------------- the device arm --

#[cfg(feature = "cuda")]
mod device {
    use super::*;
    use rcsr::gpu::GpuEvaluator;

    fn device_available() -> bool {
        rcsr::gpu::probe().available()
    }

    fn columns(s: &Sample) -> (Vec<f32>, Vec<f32>, usize, usize) {
        let n = s.len();
        let v = s.x.first().map(|r| r.len()).unwrap_or(0);
        let mut cols = vec![0.0f32; n * v];
        for (r, row) in s.x.iter().enumerate() {
            for (k, &x) in row.iter().enumerate().take(v) {
                cols[k * n + r] = x as f32;
            }
        }
        (cols, s.y.iter().map(|&y| y as f32).collect(), n, v)
    }

    pub(super) fn evaluator(s: &Sample) -> GpuEvaluator {
        let (cols, y32, n, v) = columns(s);
        let mut ev = GpuEvaluator::new(&cols, n, v, &y32).expect("device evaluator");
        assert!(ev.set_fit_sign(&fit::classification_signs_weighted(&s.y, &s.w)));
        if !s.w.is_empty() {
            assert!(ev.set_score_weights(&s.w));
        }
        ev
    }

    /// CPU values on the device's exact f32 prediction rows.
    fn cpu_on_device_rows(
        pred: &[f32],
        s: &Sample,
        m: Metric,
    ) -> f64 {
        let p: Vec<f64> = pred.iter().map(|&v| v as f64).collect();
        metrics::score_weighted(&p, &s.y, &s.w, m, 0)
    }

    /// CPU values on the original f64 model rows (the f32-row bound).
    fn cpu_on_f64_rows(tree: &rcsr::expr::Node, s: &Sample, m: Metric) -> f64 {
        cpu_score(tree, s, m)
    }

    /// #289 §6.2: device vs CPU parity on identical f32 rows for every
    /// objective, with the search's weighted denominator.
    #[test]
    fn device_matches_cpu_for_every_classification_objective() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(2_000, 0.35, 0.25, 21);
        let trees = [
            affine(0.7, -0.2),
            affine(-1.3, 0.4),
            affine(0.0, 0.0),
            node(Op::Tanh, vec![affine(1.1, 0.2)]),
        ];
        for objective in [
            ScoreObjective::LogLoss,
            ScoreObjective::Brier,
            ScoreObjective::HingeSigmoid,
            ScoreObjective::HingeOrdinary,
            ScoreObjective::HingeSquared,
        ] {
            let mut ev = evaluator(&s);
            let pop = rcsr::gpu::Population::from_trees(trees.iter());
            assert_eq!(pop.len(), trees.len());
            let batch = ev.score_population(&pop, objective, &vec![0i32; pop.len()]);
            assert!(batch.ms >= 0.0, "kernel launch failed");
            assert!(batch.all_ok(), "{:?}", batch.status);
            let preds = ev
                .predictions(pop.programs().0, pop.programs().1)
                .expect("raw predictions");
            for (i, tree) in trees.iter().enumerate() {
                let got = batch.values[i];
                let want = cpu_on_device_rows(&preds[i * s.len()..(i + 1) * s.len()], &s, objective.metric());
                let err = (got - want).abs() / want.abs().max(1.0);
                assert!(
                    err <= 1e-6,
                    "{} model {i}: device {got} vs CPU {want} (rel {err:.2e})",
                    objective.name()
                );
                // f32 rows versus the f64 model: the wider bound.
                let f64want = cpu_on_f64_rows(tree, &s, objective.metric());
                let f32err = (got - f64want).abs() / f64want.abs().max(1.0);
                assert!(
                    f32err <= 1e-5,
                    "{} model {i}: f32-row error {f32err:.2e} exceeds 1e-5",
                    objective.name()
                );
            }
        }
    }

    /// #289: weighted rows use `Σw` and skip a zero-weight row **before** its
    /// program runs, so a masked `ln(-1)` = NaN contributes nothing.
    #[test]
    fn zero_weight_rows_are_skipped_before_the_program_runs() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let mut s = binary_sample(500, 0.4, 0.1, 22);
        // ln(x0 + 4) is undefined only in row 0, which has weight 0; every
        // other row has x0 >= -3, so the value is finite there.
        s.x[0][0] = -5.0;
        s.w = vec![1.0; s.len()];
        s.w[0] = 0.0;
        s.w[7] = 3.0;
        let tree = node(
            Op::Ln,
            vec![node(Op::Add, vec![input(0), constant(4.0)])],
        );
        for objective in [ScoreObjective::LogLoss, ScoreObjective::Brier] {
            let mut ev = evaluator(&s);
            let pop = rcsr::gpu::Population::from_trees([&tree]);
            let batch = ev.score_population(&pop, objective, &[0i32]);
            assert!(batch.ms >= 0.0);
            assert!(batch.all_ok(), "{:?}", batch.status);
            let want = cpu_score(&tree, &s, objective.metric());
            assert!(want.is_finite(), "CPU reference must skip the masked row");
            let got = batch.values[0];
            let err = (got - want).abs() / want.abs().max(1.0);
            assert!(err <= 1e-6, "{}: {got} vs {want} (rel {err:.2e})", objective.name());
            assert!(got.is_finite(), "a masked NaN must not poison the score");
        }
    }

    /// #289 §6.4: the weighted denominator is `Σw` and a repeated call / batch
    /// composition cannot move a bit.
    #[test]
    fn device_classification_scores_are_deterministic_and_weighted_by_sum_w() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let mut s = binary_sample(777, 0.3, 0.2, 23);
        s.w = (0..s.len())
            .map(|i| match i % 5 {
                0 => 0.0,
                1 => 2.0,
                _ => 0.5 + (i % 3) as f64,
            })
            .collect();
        let objective = ScoreObjective::LogLoss;
        let model = affine(0.4, -0.1);
        let mut ev = evaluator(&s);
        let pop = rcsr::gpu::Population::from_trees([&model]);
        let a = ev.score_population(&pop, objective, &[0i32]);
        let b = ev.score_population(&pop, objective, &[0i32]);
        assert!(a.all_ok() && b.all_ok());
        assert_eq!(a.values[0].to_bits(), b.values[0].to_bits());
        // Alone versus a batch: the first slot is unchanged.
        let mut batch = vec![model.clone()];
        for k in 0..7 {
            batch.push(affine(0.1 * (k as f64 + 1.0), -0.2 * k as f64));
        }
        let pop_batch = rcsr::gpu::Population::from_trees(batch.iter());
        let c = ev.score_population(&pop_batch, objective, &vec![0i32; pop_batch.len()]);
        assert!(c.all_ok());
        assert_eq!(a.values[0].to_bits(), c.values[0].to_bits(), "batch composition moved a score");
    }

    /// #289: on a fixture with clear gaps the device and CPU order every
    /// candidate the same way (no rank inversion at the parity band).
    #[test]
    fn device_classification_scores_do_not_invert_the_cpu_ranking() {
        if !device_available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
        let s = binary_sample(1_500, 0.3, 0.15, 24);
        let trees: Vec<rcsr::expr::Node> = (0..8)
            .map(|k| affine(0.1 * (k as f64 + 1.0), 0.05 * k as f64 - 0.2))
            .collect();
        let objective = ScoreObjective::Brier;
        let mut ev = evaluator(&s);
        let pop = rcsr::gpu::Population::from_trees(trees.iter());
        let batch = ev.score_population(&pop, objective, &vec![0i32; pop.len()]);
        assert!(batch.ms >= 0.0);
        assert!(batch.all_ok(), "{:?}", batch.status);
        let device: Vec<f64> = batch.values.clone();
        let cpu: Vec<f64> = trees
            .iter()
            .map(|t| cpu_score(t, &s, Metric::Brier))
            .collect();
        let mut dev_order: Vec<usize> = (0..trees.len()).collect();
        dev_order.sort_by(|&a, &b| device[a].partial_cmp(&device[b]).unwrap());
        let mut cpu_order: Vec<usize> = (0..trees.len()).collect();
        cpu_order.sort_by(|&a, &b| cpu[a].partial_cmp(&cpu[b]).unwrap());
        assert_eq!(dev_order, cpu_order, "device {device:?} vs CPU {cpu:?}");
    }
}

// ---------------------------------------------------------- measurement -----

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// #289 measurement (ignored): paired native-fitness versus CPU scorer cost.
///
/// The GPU arm uploads the rows once, compiles `P` models, and pays one
/// `score_population` call per repeat; the CPU arm pays `P` prediction walks
/// plus `P` `metrics::score_weighted` passes on the same rows. Both report the
/// score and the throughput so the measurement is an apples-to-apples cost, not
/// a claim about the whole search.
///
/// Run with
/// `cargo test --release --features cuda --test classification_gpu bench_classification_fitness -- --ignored --nocapture`
/// and optionally `RCSR_BENCH_N`, `RCSR_BENCH_P`, `RCSR_BENCH_REPEATS`,
/// `RCSR_BENCH_METRIC` (`log_loss`, `brier`, `hinge`), `RCSR_BENCH_WEIGHTS=1`.
#[test]
#[ignore]
fn bench_classification_fitness() {
    let n = env_usize("RCSR_BENCH_N", 154_000);
    let p = env_usize("RCSR_BENCH_P", 32);
    let repeats = env_usize("RCSR_BENCH_REPEATS", 5);
    let weighted = std::env::var("RCSR_BENCH_WEIGHTS").as_deref() == Ok("1");
    let metric = Metric::parse(
        &std::env::var("RCSR_BENCH_METRIC").unwrap_or_else(|_| "log_loss".to_string()),
    );
    let Some(objective) = ScoreObjective::for_metric(metric) else {
        eprintln!("metric {} has no native classification scorer", metric.name());
        return;
    };
    let mut train = binary_sample(n, 0.05, 0.15, 289);
    if weighted {
        train.w = (0..n)
            .map(|i| if i % 11 == 0 { 0.0 } else { 0.5 + (i % 4) as f64 })
            .collect();
    }
    let trees: Vec<rcsr::expr::Node> = (0..p)
        .map(|k| affine(0.05 * (k as f64 + 1.0), -0.1 + 0.02 * k as f64))
        .collect();

    // CPU: the same per-candidate prediction + metric pass the search uses.
    let cpu_start = std::time::Instant::now();
    let mut cpu_best = f64::INFINITY;
    for _ in 0..repeats {
        for tree in &trees {
            let v = cpu_score(tree, &train, metric);
            cpu_best = cpu_best.min(v);
        }
    }
    let cpu_s = cpu_start.elapsed().as_secs_f64();

    #[cfg(feature = "cuda")]
    {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping GPU arm: no usable CUDA device");
            return;
        }
        let mut ev = device::evaluator(&train);
        let pop = rcsr::gpu::Population::from_trees(trees.iter());
        assert_eq!(pop.len(), p);
        let mut best = f64::INFINITY;
        let mut kernel_s = 0.0;
        let gpu_start = std::time::Instant::now();
        for _ in 0..repeats {
            let batch = ev.score_population(&pop, objective, &vec![0i32; pop.len()]);
            assert!(batch.ms >= 0.0);
            assert!(batch.all_ok());
            kernel_s += batch.ms as f64 / 1000.0;
            for &v in &batch.values {
                best = best.min(v);
            }
        }
        let gpu_s = gpu_start.elapsed().as_secs_f64();
        let rel = (best - cpu_best).abs() / cpu_best.abs().max(1.0);
        assert!(rel <= 1e-5, "device {best} vs CPU {cpu_best} (rel {rel:.2e})");
        eprintln!(
            "metric={} N={n} P={p} repeats={repeats} weighted={weighted} \
             cpu={cpu_s:.4}s gpu={gpu_s:.4}s kernel={kernel_s:.4}s \
             speedup={:.2}x cpu_row_evals/s={:.3e} gpu_row_evals/s={:.3e} best={best:.9}",
            metric.name(),
            cpu_s / gpu_s.max(1e-9),
            (repeats * p * n) as f64 / cpu_s.max(1e-9),
            (repeats * p * n) as f64 / gpu_s.max(1e-9)
        );
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = objective;
        eprintln!(
            "metric={} N={n} P={p} repeats={repeats} weighted={weighted} \
             cpu={cpu_s:.4}s (build with --features cuda for the device arm) best={cpu_best:.9}",
            metric.name()
        );
    }
}
