//! Implicit relations (issue #14).
//!
//! The target becomes an extra input and the search fits `f(inputs, target) = 0`.
//! These tests pin down the level-set distance `|f| / ||∇f||`, the zero-gradient
//! guard, the plain-residual path, the constant fit, the unit-circle recovery,
//! the CLI warning and that every explicit metric is untouched.

use rcsr::data::{self, Sample};
use rcsr::expr::{self, constant, input, node, Node, Op};
use rcsr::metrics::{self, Metric};
use rcsr::rng::Mt;
use rcsr::search::{self, Config};
use std::path::PathBuf;
use std::process::{Command, Output};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Points near the unit circle, with radial noise. The target column is the
/// implicit one: every row's `y` is 0 and both coordinates are inputs.
fn noisy_circle(n: usize, noise: f64, seed: u32) -> Sample {
    let mut rng = Mt::new(seed);
    let mut x = Vec::with_capacity(n);
    for i in 0..n {
        let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
        let r = 1.0 + rng.range(-noise, noise);
        x.push(vec![r * angle.cos(), r * angle.sin()]);
    }
    Sample {
        x,
        y: vec![0.0; n],
        w: Vec::new(),
        ys: Vec::new(),
    }
}

/// `x0^2 + x1^2 - c`.
fn circle_tree(c: f64) -> Node {
    node(
        Op::Sub,
        vec![
            node(
                Op::Add,
                vec![
                    node(Op::Square, vec![input(0)]),
                    node(Op::Square, vec![input(1)]),
                ],
            ),
            constant(c),
        ],
    )
}

/// The distance of one point to the candidate level set.
fn point_distance(tree: &Node, row: &[f64]) -> f64 {
    rcsr::implicit::distance(tree, row)
}

/// Zero crossing of `f(r, 0)` on `0.5 .. 1.5`, by bisection. Independent of any
/// positive scale factor the search may put in front of the circle.
fn radius_on_x_axis(tree: &Node) -> f64 {
    let mut lo = 0.5;
    let mut hi = 1.5;
    let f = |r: f64| expr::eval(tree, &[r, 0.0]);
    assert!(
        f(lo).is_finite() && f(hi).is_finite() && f(lo) * f(hi) < 0.0,
        "the level set must cross the positive x axis: f(0.5)={}, f(1.5)={}",
        f(lo),
        f(hi)
    );
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if f(lo) * f(mid) <= 0.0 {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    0.5 * (lo + hi)
}

fn config(seed: u32) -> Config {
    Config {
        generations: 60,
        islands: 1,
        pop: 60,
        max_complexity: 30,
        fit_iters: 10,
        patience: 0,
        seed,
        metric: Metric::ImplicitDerivative,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Square],
        ..Default::default()
    }
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

struct TempPath(PathBuf);

impl TempPath {
    fn new(suffix: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        TempPath(std::env::temp_dir().join(format!(
            "rcsr-implicit-{}-{}-{}",
            std::process::id(),
            n,
            suffix
        )))
    }
    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 path")
    }
}

impl Drop for TempPath {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

fn circle_csv() -> TempPath {
    let path = TempPath::new("circle.csv");
    let text = "x,y\n0.8,0.6\n0.6,0.8\n-0.8,0.6\n-0.6,-0.8\n0.8,-0.6\n0,-1\n1,0\n-1,0\n";
    std::fs::write(&path.0, text).expect("write csv");
    path
}

// ---------------------------------------------------------------------------
// The metric
// ---------------------------------------------------------------------------

#[test]
fn level_set_distance_is_scale_free_and_guards_a_zero_gradient() {
    let s = noisy_circle(24, 0.0, 1);
    let on = circle_tree(1.0);
    for row in &s.x {
        assert!(point_distance(&on, row) < 1e-12, "exact circle");
    }
    // Scaling f changes nothing: the metric is geometric, not algebraic.
    let scaled = node(Op::Mul, vec![constant(5.0), on.clone()]);
    for row in &s.x {
        let a = point_distance(&on, row);
        let b = point_distance(&scaled, row);
        assert!(
            (a - b).abs() <= 1e-12 * (1.0 + a.abs()),
            "the metric is scale-free: {a} vs {b}"
        );
    }
    // A constant (or f = 0) has no level set: infeasible, never perfect.
    for tree in [constant(0.0), constant(3.0)] {
        assert_eq!(
            search::eval_error(&tree, &s, Metric::ImplicitDerivative),
            f64::INFINITY,
            "a zero gradient must be infeasible"
        );
    }
    // Non-finite values are infeasible too.
    let bad = node(Op::Ln, vec![input(0)]);
    assert_eq!(point_distance(&bad, &[-2.0, 0.0]), f64::INFINITY);
}

#[test]
fn implicit_residual_is_the_plain_absolute_residual() {
    let s = noisy_circle(20, 0.01, 2);
    let tree = circle_tree(1.2);
    let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(&tree, row)).collect();
    let plain = metrics::score(&pred, &s.y, Metric::Absolute);
    assert_eq!(
        search::eval_error(&tree, &s, Metric::ImplicitResidual).to_bits(),
        plain.to_bits()
    );
    let geom = search::eval_error(&tree, &s, Metric::ImplicitDerivative);
    assert!(
        (geom - plain).abs() > 1e-6,
        "the two implicit metrics are different: residual={plain}, geometric={geom}"
    );
}

#[test]
fn implicit_without_the_metric_trains_on_the_absolute_residual() {
    // `--implicit` alone leaves the metric at absolute, which on the implicit
    // sample is mean |f|: the plain implicit residual.
    let s = noisy_circle(32, 0.02, 3);
    let tree = circle_tree(1.1);
    let plain = search::eval_error(&tree, &s, Metric::Absolute);
    let expected: f64 = s
        .x
        .iter()
        .map(|row| expr::eval(&tree, row).abs())
        .sum::<f64>()
        / s.len() as f64;
    assert_eq!(plain.to_bits(), expected.to_bits());
    // `squared` is a different objective on the same residual; the point is
    // that no implicit machinery is involved unless an implicit metric is
    // chosen.
    let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(&tree, row)).collect();
    assert_eq!(
        search::eval_error(&tree, &s, Metric::Squared).to_bits(),
        metrics::score(&pred, &s.y, Metric::Squared).to_bits()
    );
}

#[test]
fn constants_are_fitted_on_the_geometric_objective() {
    let s = noisy_circle(48, 0.01, 4);
    let mut f = circle_tree(4.0); // deliberately wrong radius (2)
    let before = search::eval_error(&f, &s, Metric::ImplicitDerivative);
    let sse = rcsr::implicit::fit(&mut f, &s.x, &s.w, 30);
    let after = search::eval_error(&f, &s, Metric::ImplicitDerivative);
    assert!(sse.is_finite());
    assert!(
        after < before * 0.01,
        "the geometric fit must improve the score: before={before}, after={after}"
    );
    let r = radius_on_x_axis(&f);
    assert!((r - 1.0).abs() < 0.05, "radius = {r}");
}

// ---------------------------------------------------------------------------
// End-to-end recovery
// ---------------------------------------------------------------------------

#[test]
fn recovers_the_unit_circle_from_noisy_points() {
    let noise = 0.01;
    let train = noisy_circle(40, noise, 11);
    let test = noisy_circle(40, noise, 12);
    // The wrong radius is fitted; the structure is the one under test.
    let seed = circle_tree(3.0);
    let out = search::run_seeded(&config(7), &train, &test, &[seed]);
    let best = out
        .archive
        .items
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
        .expect("a solution");
    assert!(
        best.error < 0.05,
        "best implicit error = {} (noise {noise})",
        best.error
    );
    // The level set contains the data points.
    for row in &train.x {
        let d = point_distance(&best.tree, row);
        assert!(
            d < 0.05,
            "a noisy point is {d} from the recovered level set"
        );
    }
    // The radius is one, read off the x-axis crossing (scale-free).
    let r = radius_on_x_axis(&best.tree);
    assert!((r - 1.0).abs() < 0.05, "recovered radius = {r}");
}

#[test]
fn an_unsmoothed_implicit_run_still_works_but_warns() {
    let path = circle_csv();
    let out = rcsr(&[
        "--data",
        path.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--implicit",
        "--eval-expr",
        "x^2 + y^2 - 1",
        "--quiet",
    ]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("warning: --implicit on unsmoothed data"),
        "stderr: {err}"
    );
    assert!(
        err.contains("implicit: target 'y' is input 2"),
        "stderr: {err}"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("0 = "), "stdout: {stdout}");
}

#[test]
fn implicit_metric_without_implicit_is_a_usage_error() {
    let path = circle_csv();
    let out = rcsr(&[
        "--data",
        path.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--metric",
        "implicit",
        "--eval-expr",
        "x^2 + y^2 - 1",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("requires --implicit"), "stderr: {err}");
}

#[test]
fn implicit_derivative_flag_selects_both() {
    let path = circle_csv();
    let out = rcsr(&[
        "--data",
        path.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--implicit-derivative",
        "--eval-expr",
        "x^2 + y^2 - 1",
        "--quiet",
    ]);
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stdout).contains("0 = "));
}

// ---------------------------------------------------------------------------
// Flag gating
// ---------------------------------------------------------------------------

#[test]
fn every_explicit_metric_is_bit_identical_to_the_old_scoring_path() {
    // The implicit branch in `eval_error` must be invisible to every other
    // metric: this is the pre-#14 formula, compared bit for bit.
    let s = noisy_circle(16, 0.02, 9);
    let tree = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(1.7), input(0)]),
            node(Op::Sin, vec![input(1)]),
        ],
    );
    let pred: Vec<f64> = s.x.iter().map(|row| expr::eval(&tree, row)).collect();
    let k = expr::count_constants(&tree);
    for m in Metric::all().iter().filter(|m| !m.is_implicit()) {
        assert_eq!(
            search::eval_error(&tree, &s, *m).to_bits(),
            metrics::score_weighted(&pred, &s.y, &s.w, *m, k).to_bits(),
            "metric {} changed",
            m.name()
        );
    }
}

#[test]
fn make_implicit_appends_a_normalised_target_and_zeroes_the_target() {
    let train = Sample {
        x: vec![vec![1.0], vec![2.0], vec![3.0]],
        y: vec![2.0, 4.0, 6.0],
        w: Vec::new(),
        ys: Vec::new(),
    };
    let test = Sample {
        x: vec![vec![4.0]],
        y: vec![8.0],
        w: Vec::new(),
        ys: Vec::new(),
    };
    let (mean, std) = data::implicit_target_stats(&train);
    assert!((mean - 4.0).abs() < 1e-12);
    assert!(std > 0.0);
    let explicit = train.clone();
    let ti = train.with_target_as_input(mean, std);
    let te = test.with_target_as_input(mean, std);
    assert_eq!(ti.x[0].len(), 2);
    assert_eq!(ti.y, vec![0.0, 0.0, 0.0]);
    assert_eq!(te.y, vec![0.0]);
    assert_eq!(ti.x.len(), explicit.x.len());
    // Original units are recovered from the appended denormalisation
    // constants: (y - mean)/std * std + mean == y.
    let appended = ti.x[2][1];
    assert!(((appended * std + mean) - 6.0).abs() < 1e-12);
    // A zero-variance target still gets a usable std.
    let flat = Sample {
        x: vec![vec![0.0], vec![1.0]],
        y: vec![2.0, 2.0],
        w: Vec::new(),
        ys: Vec::new(),
    };
    assert_eq!(data::implicit_target_stats(&flat), (2.0, 1.0));
}
