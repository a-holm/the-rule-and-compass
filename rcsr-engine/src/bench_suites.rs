//! The `hard` and `signal` benchmark suites.
//!
//! `hard` holds 20 closed-form problems that the default budget does not solve
//! in every run, so a search change shows up as a change in recovery. `signal`
//! holds three finance-like regression tasks with a low signal-to-noise ratio,
//! heavy-tailed noise, distractor columns and, for one task, autocorrelated
//! inputs. See `docs/benchmarks.md` for the full tables.

use crate::bench::{c, gaussian, u1, u2, x, Problem, DATA_SEED};
use crate::data::Sample;
use crate::expr::{self, Node, Op};
use crate::rng::Mt;

/// Training rows of a `hard` problem.
const HARD_TRAIN: usize = 100;
/// Fresh noiseless test rows of a `hard` problem.
const HARD_TEST: usize = 500;

/// Rows per split of a `signal` task.
pub const SIGNAL_TRAIN: usize = 2000;
pub const SIGNAL_VAL: usize = 1000;
pub const SIGNAL_TEST: usize = 2000;
/// A signal task is recovered when the selected member reaches this R² against
/// the noiseless test signal.
pub const SIGNAL_THRESHOLD: f64 = 0.8;
/// Degrees of freedom of the Student-t noise.
const T_DOF: usize = 4;

fn sq(a: Node) -> Node {
    u1(Op::Square, a)
}
fn cu(a: Node) -> Node {
    u1(Op::Cube, a)
}
fn pow4(a: Node) -> Node {
    sq(sq(a))
}
fn add(a: Node, b: Node) -> Node {
    u2(Op::Add, a, b)
}
fn sub(a: Node, b: Node) -> Node {
    u2(Op::Sub, a, b)
}
fn mul(a: Node, b: Node) -> Node {
    u2(Op::Mul, a, b)
}
fn div(a: Node, b: Node) -> Node {
    u2(Op::Div, a, b)
}

/// The 20 `hard` problems. `truth_cx` is the table value of the design and is
/// checked against the tree by a unit test.
pub fn hard() -> Vec<Problem> {
    use Op::*;
    let h = |name, vars: &[&str], truth, ranges: &[(f64, f64)], cx: usize| {
        Problem::new(name, vars, truth, ranges, false)
            .in_suite("hard", HARD_TRAIN, HARD_TEST)
            .with_truth_cx(cx)
    };
    let u = (-1.0, 1.0);
    let t = (-3.0, 3.0);
    let k = (-50.0, 50.0);
    vec![
        h(
            "nguyen5",
            &["x"],
            sub(mul(u1(Sin, sq(x(0))), u1(Cos, x(0))), c(1.0)),
            &[u],
            8,
        ),
        h(
            "nguyen6",
            &["x"],
            add(u1(Sin, x(0)), u1(Sin, add(x(0), sq(x(0))))),
            &[u],
            8,
        ),
        h(
            "nguyen7",
            &["x"],
            add(u1(Ln, add(x(0), c(1.0))), u1(Ln, add(sq(x(0)), c(1.0)))),
            &[(0.0, 2.0)],
            10,
        ),
        h(
            "nguyen9",
            &["x1", "x2"],
            add(u1(Sin, x(0)), u1(Sin, sq(x(1)))),
            &[u, u],
            6,
        ),
        h(
            "nguyen10",
            &["x1", "x2"],
            mul(mul(c(2.0), u1(Sin, x(0))), u1(Cos, x(1))),
            &[u, u],
            7,
        ),
        h(
            "nguyen12",
            &["x1", "x2"],
            sub(
                add(sub(pow4(x(0)), cu(x(0))), mul(c(0.5), sq(x(1)))),
                x(1),
            ),
            &[u, u],
            13,
        ),
        h(
            "keijzer11",
            &["x1", "x2"],
            add(
                mul(x(0), x(1)),
                u1(Sin, mul(sub(x(0), c(1.0)), sub(x(1), c(1.0)))),
            ),
            &[t, t],
            12,
        ),
        h(
            "keijzer14",
            &["x1", "x2"],
            div(c(8.0), add(add(c(2.0), sq(x(0))), sq(x(1)))),
            &[t, t],
            9,
        ),
        h(
            "keijzer15",
            &["x1", "x2"],
            sub(
                sub(
                    add(mul(c(0.2), cu(x(0))), mul(c(0.5), cu(x(1)))),
                    x(1),
                ),
                x(0),
            ),
            &[t, t],
            13,
        ),
        h(
            "vladislavleva1",
            &["x1", "x2"],
            div(
                u1(Exp, u1(Neg, sq(sub(x(0), c(1.0))))),
                add(c(1.2), sq(sub(x(1), c(2.5)))),
            ),
            &[(0.3, 4.0), (0.3, 4.0)],
            13,
        ),
        h(
            "pagie1",
            &["x1", "x2"],
            add(
                div(pow4(x(0)), add(c(1.0), pow4(x(0)))),
                div(pow4(x(1)), add(c(1.0), pow4(x(1)))),
            ),
            &[(-5.0, 5.0), (-5.0, 5.0)],
            19,
        ),
        h(
            "korns1",
            &["x1", "x2", "x3", "x4", "x5"],
            add(c(1.57), mul(c(24.3), x(3))),
            &[k, k, k, k, k],
            5,
        )
        .with_distractors(&[0, 1, 2, 4]),
        h(
            "korns4",
            &["x1", "x2", "x3", "x4", "x5"],
            add(c(-2.3), mul(c(0.13), u1(Sin, x(2)))),
            &[k, k, k, k, k],
            6,
        )
        .with_distractors(&[0, 1, 3, 4]),
        h(
            "feynman_i6_2",
            &["t", "s"],
            div(
                u1(
                    Exp,
                    mul(c(-0.5), sq(div(x(0), x(1)))),
                ),
                mul(c(2.5066283), x(1)),
            ),
            &[(1.0, 3.0), (1.0, 3.0)],
            11,
        ),
        h(
            "feynman_i12_11",
            &["q", "E", "B", "v", "th"],
            mul(x(0), add(x(1), mul(mul(x(2), x(3)), u1(Sin, x(4))))),
            &[(1.0, 5.0); 5],
            10,
        ),
        h(
            "feynman_i15_3x",
            &["x", "u", "t", "c"],
            div(
                sub(x(0), mul(x(1), x(2))),
                u1(Sqrt, sub(c(1.0), div(sq(x(1)), sq(x(3))))),
            ),
            &[(5.0, 10.0), (1.0, 2.0), (2.0, 5.0), (3.0, 10.0)],
            14,
        ),
        h(
            "feynman_i29_16",
            &["x1", "x2", "th1", "th2"],
            u1(
                Sqrt,
                sub(
                    add(sq(x(0)), sq(x(1))),
                    mul(
                        mul(mul(c(2.0), x(0)), x(1)),
                        u1(Cos, sub(x(2), x(3))),
                    ),
                ),
            ),
            &[(1.0, 5.0); 4],
            17,
        ),
        h(
            "freq_sine",
            &["x"],
            mul(c(1.5), u1(Sin, add(mul(c(2.7), x(0)), c(0.4)))),
            &[t],
            8,
        ),
        h(
            "rational_r1",
            &["x"],
            div(
                cu(add(x(0), c(1.0))),
                add(sub(sq(x(0)), x(0)), c(1.0)),
            ),
            &[u],
            11,
        ),
        h(
            "damped_freq",
            &["x"],
            mul(
                u1(Exp, mul(c(-0.4), x(0))),
                u1(Cos, mul(c(3.0), x(0))),
            ),
            &[(0.0, 6.0)],
            9,
        ),
    ]
}

// ---------------------------------------------------------------------------
// signal suite
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum SignalKind {
    Tanh,
    Regime,
    Ar,
}

/// A low-SNR regression task: noisy target `y = s + e`, noiseless signal `s`.
#[derive(Clone)]
pub struct SignalTask {
    pub name: &'static str,
    pub vars: Vec<String>,
    /// The signal `s` as a tree over `vars`.
    pub truth: Node,
    pub truth_cx: usize,
    /// Input indexes the signal does not use.
    pub distractors: Vec<usize>,
    /// Target `var(s)/var(y)` on the training rows.
    pub oracle_r2: f64,
    /// Rows stay in time order; the splits are consecutive blocks.
    pub ordered: bool,
    kind: SignalKind,
}

impl SignalTask {
    pub fn truth_string(&self) -> String {
        expr::to_string(&self.truth, &self.vars)
    }

    /// Deterministic data: train, validation and test samples with noisy
    /// targets, plus the noiseless signal of each split.
    pub fn generate(&self) -> SignalData {
        let n = SIGNAL_TRAIN + SIGNAL_VAL + SIGNAL_TEST;
        let mut rng = Mt::new(DATA_SEED ^ crate::bench::fnv(self.name.as_bytes()));
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(n);
        match self.kind {
            SignalKind::Tanh => {
                for _ in 0..n {
                    let r1 = gaussian(&mut rng);
                    let vol = (0.5 * gaussian(&mut rng)).exp();
                    let z = gaussian(&mut rng);
                    rows.push(vec![r1, vol, z]);
                }
            }
            SignalKind::Regime => {
                for _ in 0..n {
                    let r1 = gaussian(&mut rng);
                    let vol = (0.5 * gaussian(&mut rng)).exp();
                    rows.push(vec![r1, vol]);
                }
            }
            SignalKind::Ar => {
                let (fast, slow) = (0.5f64, 0.95f64);
                let mut xf = gaussian(&mut rng);
                let mut xs = gaussian(&mut rng);
                for _ in 0..n {
                    xf = fast * xf + (1.0 - fast * fast).sqrt() * gaussian(&mut rng);
                    xs = slow * xs + (1.0 - slow * slow).sqrt() * gaussian(&mut rng);
                    let vol = (0.5 * gaussian(&mut rng)).exp();
                    rows.push(vec![xf, xs, vol]);
                }
            }
        }
        for row in rows.iter_mut() {
            for _ in 0..3 {
                row.push(gaussian(&mut rng));
            }
        }
        let signal: Vec<f64> = rows.iter().map(|r| expr::eval(&self.truth, r)).collect();
        let raw: Vec<f64> = (0..n).map(|_| student_t(&mut rng)).collect();

        let tr = 0..SIGNAL_TRAIN;
        let (vs, ve, cov) = moments(&signal[tr.clone()], &raw[tr]);
        // Choose the noise scale k so that var(s) / var(s + k e) equals the
        // target on the training rows exactly:
        // ve k^2 + 2 cov k + vs (1 - 1/r2) = 0.
        let a = ve;
        let b = 2.0 * cov;
        let cc = vs * (1.0 - 1.0 / self.oracle_r2);
        let k = (-b + (b * b - 4.0 * a * cc).sqrt()) / (2.0 * a);

        let split = |lo: usize, hi: usize| -> (Sample, Vec<f64>) {
            let s = Sample {
                x: rows[lo..hi].to_vec(),
                y: (lo..hi).map(|i| signal[i] + k * raw[i]).collect(),
                w: Vec::new(),
                ys: Vec::new(),
                pairs: None,
            };
            (s, signal[lo..hi].to_vec())
        };
        let (train, s_train) = split(0, SIGNAL_TRAIN);
        let (val, s_val) = split(SIGNAL_TRAIN, SIGNAL_TRAIN + SIGNAL_VAL);
        let (test, s_test) = split(SIGNAL_TRAIN + SIGNAL_VAL, n);
        SignalData {
            train,
            val,
            test,
            s_train,
            s_val,
            s_test,
        }
    }
}

/// The three splits of a signal task and the noiseless signal of each.
pub struct SignalData {
    pub train: Sample,
    pub val: Sample,
    pub test: Sample,
    pub s_train: Vec<f64>,
    pub s_val: Vec<f64>,
    pub s_test: Vec<f64>,
}

impl SignalData {
    /// Realised `var(s) / var(y)` on the training rows.
    pub fn train_signal_ratio(&self) -> f64 {
        let (vs, _, _) = moments(&self.s_train, &self.s_train);
        let mean = self.train.y.iter().sum::<f64>() / self.train.y.len() as f64;
        let vy = self
            .train
            .y
            .iter()
            .map(|v| (v - mean) * (v - mean))
            .sum::<f64>()
            / self.train.y.len() as f64;
        vs / vy
    }
}

/// Population variances of `a` and `b` and their covariance.
fn moments(a: &[f64], b: &[f64]) -> (f64, f64, f64) {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut va = 0.0;
    let mut vb = 0.0;
    let mut cab = 0.0;
    for (p, q) in a.iter().zip(b) {
        va += (p - ma) * (p - ma);
        vb += (q - mb) * (q - mb);
        cab += (p - ma) * (q - mb);
    }
    (va / n, vb / n, cab / n)
}

/// `N(0,1) / sqrt(chi2_4 / 4)`.
fn student_t(rng: &mut Mt) -> f64 {
    let z = gaussian(rng);
    let mut chi2 = 0.0;
    for _ in 0..T_DOF {
        let g = gaussian(rng);
        chi2 += g * g;
    }
    z / (chi2 / T_DOF as f64).sqrt()
}

/// The three `signal` tasks.
pub fn signal() -> Vec<SignalTask> {
    use Op::*;
    let names = |base: &[&str]| -> Vec<String> {
        base.iter()
            .chain(["d1", "d2", "d3"].iter())
            .map(|s| s.to_string())
            .collect()
    };
    let task = |name, vars: Vec<String>, truth: Node, distractors: Vec<usize>, oracle_r2, ordered, kind| {
        let truth_cx = expr::complexity(&truth);
        SignalTask {
            name,
            vars,
            truth,
            truth_cx,
            distractors,
            oracle_r2,
            ordered,
            kind,
        }
    };
    vec![
        task(
            "sig_tanh",
            names(&["r1", "vol", "z"]),
            sub(
                mul(c(0.3), u1(Tanh, mul(c(2.0), x(0)))),
                mul(mul(c(0.2), x(2)), x(1)),
            ),
            vec![3, 4, 5],
            0.20,
            false,
            SignalKind::Tanh,
        ),
        task(
            "sig_regime",
            names(&["r1", "vol"]),
            node3(
                sub(c(1.2), x(1)),
                mul(c(0.25), x(0)),
                mul(c(-0.1), x(0)),
            ),
            vec![2, 3, 4],
            0.10,
            false,
            SignalKind::Regime,
        ),
        task(
            "sig_ar",
            names(&["xf", "xs", "vol"]),
            div(mul(c(0.15), sub(x(0), x(1))), x(2)),
            vec![3, 4, 5],
            0.05,
            true,
            SignalKind::Ar,
        ),
    ]
}

fn node3(a: Node, b: Node, d: Node) -> Node {
    expr::node(Op::If, vec![a, b, d])
}
