//! The CPU arm of the SVM hinge forms.
//!
//! With the label sign `t` and the logit `p`, `hinge_ordinary` is
//! `max(0, 1 − t·p)` and `hinge_squared` its square. Both arms read `t` with
//! the weighted scorer's midpoint rule and keep a round only when the
//! configured hinge strictly falls, so a result is never worse than its start.
//!
//! * `hinge_ordinary`: the [`logistic_irls`] rounds while they lower the hinge,
//!   then majorise-minimise (MM) rounds for the rest of the round budget and
//!   one final MM solve. With `z = 1 − t·p` and `a = max(|z|, 1e-6)` at the
//!   current constants, `max(0, z) ≤ (z + a)²/(4a)` with equality at the
//!   current point, and `z + a = t·(t(1 + a) − p)`: an MM round is weighted
//!   least squares to the target `t(1 + a)` with weight `1/(2a)`. The logistic
//!   rounds are today's rule unchanged, so the arm extends it.
//! * `hinge_squared`: generalized Gauss-Newton. Because `t² = 1` the loss is
//!   `(p − t)²` on the active rows `t·p < 1` and 0 elsewhere, so a round is
//!   weighted least squares to the target `t` with weight `[t·p < 1]`.
//!
//! The device kernel runs the same schedules (fit codes 6 and 7 in
//! `src/cuda/eval.cu`).

use super::*;

/// The MM floor on `|1 − t·p|`, the device kernel's `FT_L1_EPS`.
const MM_EPS: f64 = 1e-6;

/// Inner LM sweeps per round, as in [`logistic_irls`].
const INNER: usize = 2;

/// The round budget [`logistic_irls`] uses for `iters`.
fn round_budget(iters: usize) -> usize {
    iters.clamp(1, 16)
}

fn live(w: &[f64], i: usize) -> bool {
    w.is_empty() || w[i] > 0.0
}

/// MM weight and target of one row; a zero-weight row is inert.
fn mm_row(p: f64, t: f64, live: bool) -> Option<(f64, f64)> {
    if !live {
        return Some((0.0, 0.0));
    }
    if !p.is_finite() || !t.is_finite() {
        return None;
    }
    let a = (1.0 - t * p).abs().max(MM_EPS);
    Some((1.0 / (2.0 * a), t * (1.0 + a)))
}

/// Gauss-Newton weight and target of one row; a zero-weight row is inert.
fn ggn_row(p: f64, t: f64, live: bool) -> Option<(f64, f64)> {
    if !live {
        return Some((0.0, 0.0));
    }
    if !p.is_finite() || !t.is_finite() {
        return None;
    }
    Some((if t * p < 1.0 { 1.0 } else { 0.0 }, t))
}

/// `--metric hinge_ordinary`: logistic rounds, then MM rounds.
pub(super) fn fit_ordinary(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
) -> f64 {
    let (best, accepted) = logistic_irls_counted(tree, x, y, w, iters, Metric::HingeOrdinary);
    if !best.is_finite() || best == 0.0 {
        return best;
    }
    let signs = classification_signs_weighted(y, w);
    let rest = round_budget(iters).saturating_sub(accepted);
    irls_rounds(tree, x, y, w, Metric::HingeOrdinary, rest, INNER, iters.max(4), |p, i| {
        mm_row(p, signs[i], live(w, i))
    })
}

/// `--metric hinge_squared`: Gauss-Newton rounds on the active set.
pub(super) fn fit_squared(
    tree: &mut Node,
    x: &[Vec<f64>],
    y: &[f64],
    w: &[f64],
    iters: usize,
) -> f64 {
    if x.len() != y.len() || (!w.is_empty() && w.len() != x.len()) {
        return f64::INFINITY;
    }
    let signs = classification_signs_weighted(y, w);
    irls_rounds(
        tree,
        x,
        y,
        w,
        Metric::HingeSquared,
        round_budget(iters),
        INNER,
        iters.max(4),
        |p, i| ggn_row(p, signs[i], live(w, i)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::{constant, input, node};

    /// `a·x0 + b`.
    fn affine(a: f64, b: f64) -> Node {
        node(
            Op::Add,
            vec![node(Op::Mul, vec![constant(a), input(0)]), constant(b)],
        )
    }

    fn consts(tree: &Node) -> Vec<f64> {
        let f = flatten(tree);
        let mut c = vec![0.0; f.nconst];
        for i in 0..f.op.len() {
            if f.cslot[i] >= 0 {
                c[f.cslot[i] as usize] = f.val[i];
            }
        }
        c
    }

    fn bits(tree: &Node) -> Vec<u64> {
        consts(tree).iter().map(|v| v.to_bits()).collect()
    }

    /// One labelled cell: positives at `+sep/2`, negatives at `−sep/2`, each
    /// spread by `spread·u` with `u` a centred sum of three uniforms. Every
    /// `flip`-th label is flipped (0: none).
    fn cell(
        pos: usize,
        neg: usize,
        sep: f64,
        spread: f64,
        flip: usize,
        seed: u64,
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
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
            x.push(vec![centre + spread * u]);
            let flipped = flip > 0 && i % flip == flip - 1;
            y.push(if positive != flipped { 1.0 } else { 0.0 });
        }
        (x, y)
    }

    /// The four method cells: overlap 1:2, separable 1:2, overlap 1:19 and
    /// label noise 1:2.
    fn cells() -> Vec<(&'static str, Vec<Vec<f64>>, Vec<f64>)> {
        let mut out = Vec::new();
        let (x, y) = cell(200, 400, 1.0, 1.0, 0, 1);
        out.push(("overlap 1:2", x, y));
        let (x, y) = cell(200, 400, 4.0, 0.5, 0, 2);
        out.push(("separable 1:2", x, y));
        let (x, y) = cell(30, 570, 1.5, 1.0, 0, 3);
        out.push(("overlap 1:19", x, y));
        let (x, y) = cell(200, 400, 2.0, 0.5, 7, 4);
        out.push(("noisy 1:2", x, y));
        out
    }

    fn score(tree: &Node, x: &[Vec<f64>], y: &[f64], w: &[f64], m: Metric) -> f64 {
        metric_score(tree, x, y, w, m)
    }

    /// Each hinge form has its own fit code, reads the label sign, has no
    /// legacy evaluator objective and is accepted for `--fit-strategy gpu`.
    #[test]
    fn hinge_forms_have_distinct_device_fit_codes() {
        let forms = [
            (Metric::HingeSquared, DeviceLoss::HingeSquared, 6),
            (Metric::HingeOrdinary, DeviceLoss::HingeOrdinary, 7),
            (Metric::Hinge, DeviceLoss::HingeSigmoid, 8),
        ];
        for (metric, loss, code) in forms {
            assert_eq!(fit_loss(metric), Some(loss), "{metric:?}");
            assert_eq!(loss.code(), code);
            assert!(loss.needs_label_sign());
            assert!(!loss.scores_on_device());
            assert_eq!(DeviceLoss::for_metric(metric), None);
            assert_eq!(fit_only_loss(metric), None, "{metric:?} is scored natively");
            assert!(gpu_fit_metric_error(metric).is_none(), "{metric:?}");
            assert!(!device_fit_is_gated(metric), "{metric:?}: no host gate");
        }
        assert!(DeviceLoss::Logistic.needs_label_sign());
        assert!(!DeviceLoss::Squared.needs_label_sign());
        assert_eq!(fit_loss(Metric::Brier), None);
    }

    /// Constant model: `hinge_squared` has the closed form
    /// `p* = (n₊ − n₋)/(n₊ + n₋)`.
    #[test]
    fn hinge_squared_constant_model_is_closed_form() {
        for (pos, neg) in [(300usize, 300usize), (420, 180), (60, 540), (13, 7)] {
            let n = pos + neg;
            let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let y: Vec<f64> = (0..n).map(|i| if i < pos { 1.0 } else { 0.0 }).collect();
            let mut tree = constant(0.0);
            let got = fit_weighted_metric(&mut tree, &x, &y, &[], 8, Metric::HingeSquared);
            let want = (pos as f64 - neg as f64) / n as f64;
            let c = consts(&tree)[0];
            let rel = (c - want).abs() / want.abs().max(1.0);
            assert!(rel <= 1e-9, "{pos}:{neg}: {c} vs {want} (rel {rel:e})");
            let loss = (pos as f64 * (1.0 - want).powi(2) + neg as f64 * (1.0 + want).powi(2))
                / n as f64;
            assert!((got - loss).abs() <= 1e-12 * loss.max(1.0), "{got} vs {loss}");
        }
    }

    /// Constant model: `hinge_ordinary` has its optimum at the majority margin
    /// `p* = sign(n₊ − n₋)` with objective `2·min(n₊, n₋)/n`. The MM rate
    /// falls with the class ratio, so the 1:9 case is only required to beat
    /// the logistic stop at `ln(n₊/n₋)`.
    #[test]
    fn hinge_ordinary_constant_model_reaches_the_majority_margin() {
        for (pos, neg) in [(420usize, 180usize), (180, 420), (13, 7), (60, 540)] {
            let n = pos + neg;
            let x: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
            let y: Vec<f64> = (0..n).map(|i| if i < pos { 1.0 } else { 0.0 }).collect();
            let want = 2.0 * pos.min(neg) as f64 / n as f64;
            let margin = if pos > neg { 1.0 } else { -1.0 };
            let mut tree = constant(0.0);
            let got = fit_weighted_metric(&mut tree, &x, &y, &[], 16, Metric::HingeOrdinary);
            let c = consts(&tree)[0];
            let mut lo = constant(0.0);
            let only = logistic_irls(&mut lo, &x, &y, &[], 16, Metric::HingeOrdinary);
            eprintln!("{pos}:{neg}: schedule {got:.12} at {c:.9}, logistic {only:.12}, optimum {want:.12}");
            assert!(only > got, "{pos}:{neg}: logistic {only} must stall above {got}");
            if pos * 4 >= neg && neg * 4 >= pos {
                assert!((got - want).abs() <= 1e-6 * want, "{pos}:{neg}: {got} vs {want}");
                assert!((c - margin).abs() <= 1e-5, "{pos}:{neg}: constant {c}");
            }
        }
    }

    /// The `hinge_ordinary` arm starts with `logistic_irls` and accepts only
    /// improving rounds after it, so it is never worse at equal iterations; on
    /// the non-separable cells it is strictly better.
    #[test]
    fn hinge_ordinary_schedule_is_never_worse_than_logistic_irls() {
        let start = affine(0.0, 0.0);
        let mut strictly = 0;
        for (name, x, y) in cells() {
            for iters in [4usize, 8, 16] {
                let mut lo = start.clone();
                let a = logistic_irls(&mut lo, &x, &y, &[], iters, Metric::HingeOrdinary);
                let mut mm = start.clone();
                let b = fit_weighted_metric(&mut mm, &x, &y, &[], iters, Metric::HingeOrdinary);
                eprintln!("hinge_ordinary {name} iters {iters}: logistic {a:.9} schedule {b:.9}");
                assert!(b <= a, "{name} iters {iters}: {b} > logistic {a}");
                assert_eq!(b.to_bits(), score(&mm, &x, &y, &[], Metric::HingeOrdinary).to_bits());
                if b < a {
                    strictly += 1;
                }
            }
        }
        assert!(strictly >= 6, "the MM rounds improved only {strictly} cells");
    }

    /// `hinge_squared` replaces `logistic_irls` by Gauss-Newton. Measured on
    /// the method cells at equal iterations: no worse on every non-separable
    /// cell. On separable data the logistic rounds grow the scale faster, so
    /// Gauss-Newton may trail them at a small budget; it reaches the zero-loss
    /// optimum within 16 rounds.
    #[test]
    fn hinge_squared_gauss_newton_is_no_worse_than_logistic_irls() {
        let start = affine(0.0, 0.0);
        let mut worse = Vec::new();
        for (name, x, y) in cells() {
            for iters in [4usize, 8, 16] {
                let mut lo = start.clone();
                let a = logistic_irls(&mut lo, &x, &y, &[], iters, Metric::HingeSquared);
                let mut gn = start.clone();
                let b = fit_weighted_metric(&mut gn, &x, &y, &[], iters, Metric::HingeSquared);
                eprintln!("hinge_squared {name} iters {iters}: logistic {a:.9} gauss-newton {b:.9}");
                if b > a {
                    worse.push((name, iters, a, b));
                }
            }
        }
        for &(name, iters, a, b) in &worse {
            assert!(name.starts_with("separable"), "{name} iters {iters}: {b} > logistic {a}");
            assert!(iters < 16 && b <= 1e-4, "{name} iters {iters}: {b} > logistic {a}");
        }
    }

    /// Separable data has a finite optimum with loss 0, and both arms reach it.
    #[test]
    fn hinge_separable_data_reaches_zero() {
        let (x, y) = cell(200, 400, 4.0, 0.5, 0, 2);
        for metric in [Metric::HingeOrdinary, Metric::HingeSquared] {
            let mut tree = affine(0.0, 0.0);
            let got = fit_weighted_metric(&mut tree, &x, &y, &[], 16, metric);
            assert_eq!(got, 0.0, "{metric:?}");
            assert!(consts(&tree).iter().all(|c| c.is_finite()), "{metric:?}");
        }
    }

    /// A zero-weight row is invisible: a NaN target there changes no constant.
    #[test]
    fn hinge_weighted_zero_rows_are_invisible() {
        let (x, y) = cell(150, 250, 1.0, 1.0, 0, 9);
        let w = vec![1.0; x.len()];
        let mut xm = x.clone();
        let mut ym = y.clone();
        let mut wm = w.clone();
        xm.push(vec![0.25]);
        ym.push(f64::NAN);
        wm.push(0.0);
        for metric in [Metric::HingeOrdinary, Metric::HingeSquared, Metric::Hinge] {
            let mut a = affine(0.3, -0.1);
            let sa = fit_weighted_metric(&mut a, &x, &y, &w, 8, metric);
            let mut b = affine(0.3, -0.1);
            let sb = fit_weighted_metric(&mut b, &xm, &ym, &wm, 8, metric);
            assert_eq!(bits(&a), bits(&b), "{metric:?}");
            assert_eq!(sa.to_bits(), sb.to_bits(), "{metric:?}");
        }
    }

    /// `--fit-iters 0` moves nothing and returns the configured score.
    #[test]
    fn hinge_no_fit_is_a_no_op() {
        let (x, y) = cell(100, 100, 1.0, 1.0, 0, 5);
        for metric in [Metric::HingeOrdinary, Metric::HingeSquared, Metric::Hinge] {
            let mut tree = affine(0.4, 0.2);
            let before = bits(&tree);
            let got = fit_weighted_metric(&mut tree, &x, &y, &[], 0, metric);
            assert_eq!(bits(&tree), before, "{metric:?}");
            assert_eq!(got.to_bits(), score(&tree, &x, &y, &[], metric).to_bits());
        }
    }

    /// The bounded sigmoid `hinge` keeps the logistic arm unchanged.
    #[test]
    fn hinge_sigmoid_keeps_the_logistic_arm() {
        let (x, y) = cell(200, 400, 1.0, 1.0, 0, 6);
        let mut a = affine(0.0, 0.0);
        let sa = fit_weighted_metric(&mut a, &x, &y, &[], 8, Metric::Hinge);
        let mut b = affine(0.0, 0.0);
        let sb = logistic_irls(&mut b, &x, &y, &[], 8, Metric::Hinge);
        assert_eq!(bits(&a), bits(&b));
        assert_eq!(sa.to_bits(), sb.to_bits());
    }
}
