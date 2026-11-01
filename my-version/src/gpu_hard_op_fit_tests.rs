//! #499: the piecewise and hard operators in the batched fit kernel.
//!
//! Each family is fitted on the device and on the CPU; the device loss must
//! not exceed the CPU loss by more than `1e-5` relative. The rows sit on an
//! exact f32 grid and hit the kinks of every operator: `x0 == 0`, `x0 == c1`,
//! `x0 == x1`, half-integers for `round` and exact multiples of the modulus.
//! Device tests skip cleanly on a machine without CUDA.

use super::*;
use crate::data::Sample;
use crate::expr::{constant, input, node, Node, Op};
use crate::fit::{self, FitStrategy};
use crate::search::{self, Config};

const ROWS: usize = 640;

fn rows() -> Vec<Vec<f64>> {
    (0..ROWS)
        .map(|i| {
            let x0 = if i % 5 == 0 { 0.0 } else { ((i * 37) % 33) as f64 / 8.0 - 2.0 };
            let x1 = if i % 7 == 0 { x0 } else { ((i * 53) % 29) as f64 / 8.0 - 1.5 };
            let x2 = if x0 > 0.0 { -0.25 - (i % 3) as f64 / 4.0 } else { 0.25 + (i % 4) as f64 / 4.0 };
            vec![x0, x1, x2]
        })
        .collect()
}

fn noise(i: usize) -> f64 {
    (((i * 31) % 11) as f64 - 5.0) * 1e-3
}

fn targets(x: &[Vec<f64>], f: impl Fn(&[f64]) -> f64) -> Vec<f64> {
    x.iter().enumerate().map(|(i, r)| f(r) + noise(i)).collect()
}

/// `c0 * op(args) + c2`.
fn scaled(c0: f64, c2: f64, op: Op, args: Vec<Node>) -> Node {
    node(Op::Add, vec![node(Op::Mul, vec![constant(c0), node(op, args)]), constant(c2)])
}

fn device_fit(x: &[Vec<f64>], y: &[f64], start: &Node, iters: usize) -> (Node, i32) {
    let n = x.len();
    let v = x[0].len();
    let mut cols64 = vec![0.0f64; n * v];
    for (r, row) in x.iter().enumerate() {
        for (k, &val) in row.iter().enumerate() {
            cols64[k * n + r] = val;
        }
    }
    let cols32: Vec<f32> = cols64.iter().map(|&a| a as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&a| a as f32).collect();
    let mut ev = GpuEvaluator::new(&cols32, n, v, &y32).expect("device evaluator");
    assert!(ev.set_fit_data(&cols64, y, &[]), "f64 fit rows uploaded");
    let plan = fit::gpu_fit_plan(start).expect("every operator of the fixture has a fit opcode");
    let (out, ms) = ev.fit_batch(&[&plan], iters, fit::DeviceLoss::Squared);
    assert!(ms >= 0.0, "the launch was refused");
    let mut dev = start.clone();
    if fit::fit_status_usable(out[0].status) && out[0].c.len() == plan.nconst {
        plan.apply(&mut dev, &out[0].c);
    }
    (dev, out[0].status)
}

fn sse(tree: &Node, x: &[Vec<f64>], y: &[f64]) -> f64 {
    let mut t = tree.clone();
    fit::fit_weighted(&mut t, x, y, &[], 0)
}

fn check(name: &str, x: &[Vec<f64>], y: &[f64], start: &Node) {
    let begin = sse(start, x, y);
    let mut cpu = start.clone();
    fit::fit_weighted(&mut cpu, x, y, &[], 8);
    let cpu_sse = fit::fit_weighted(&mut cpu, x, y, &[], 8);
    let (dev, status) = device_fit(x, y, start, 8);
    assert!(fit::fit_status_usable(status), "{name}: status {status}");
    let dev_sse = sse(&dev, x, y);
    eprintln!("{name}: start {begin:.6e} cpu {cpu_sse:.6e} device {dev_sse:.6e} status {status}");
    assert!(dev_sse.is_finite(), "{name}: non-finite device loss");
    assert!(dev_sse <= begin * (1.0 + 1e-5) + 1e-12, "{name}: worse than the start");
    assert!(
        dev_sse <= cpu_sse * (1.0 + 1e-5) + 1e-12,
        "{name}: device {dev_sse:.12e} vs cpu {cpu_sse:.12e}"
    );
}

macro_rules! need_device {
    () => {
        if !probe().available() {
            eprintln!("skipping GPU test: no usable CUDA device");
            return;
        }
    };
}

#[test]
fn hard_op_plan_words_carry_the_scoring_opcodes() {
    let cases = [
        (Op::Step, 19),
        (Op::Floor, 23),
        (Op::Ceil, 24),
        (Op::Round, 25),
        (Op::Mod, 26),
        (Op::Atan2, 27),
        (Op::Min, 30),
        (Op::Max, 31),
        (Op::Sign, 33),
        (Op::Equal, 34),
        (Op::Less, 35),
        (Op::LessOrEqual, 36),
        (Op::Greater, 37),
        (Op::GreaterOrEqual, 38),
        (Op::And, 39),
        (Op::Or, 40),
        (Op::Xor, 41),
        (Op::Not, 42),
    ];
    for (op, code) in cases {
        let args = match op {
            Op::Step | Op::Floor | Op::Ceil | Op::Round | Op::Sign | Op::Not => vec![input(0)],
            _ => vec![input(0), constant(0.5)],
        };
        let t = scaled(1.0, 0.0, op, args);
        let plan = fit::gpu_fit_plan(&t).unwrap_or_else(|e| panic!("{op:?}: {e:?}"));
        assert!(plan.words.chunks(4).any(|r| r[0] == code), "{op:?} is opcode {code}");
        assert_eq!(code, crate::gpu::postfix_opcode(op).expect("a scoring opcode").0);
    }
    let all: Vec<Op> = cases.iter().map(|c| c.0).collect();
    assert!(fit::gpu_fit_op_coverage(&all, false));
}

#[test]
fn min_max_device_fit_matches_the_cpu_at_the_tie_rows() {
    need_device!();
    let x = rows();
    for (name, op, f) in [
        ("min", Op::Min, f64::min as fn(f64, f64) -> f64),
        ("max", Op::Max, f64::max as fn(f64, f64) -> f64),
    ] {
        let y = targets(&x, |r| 1.5 * f(r[0], 0.75) + 0.25);
        // `c1 = 0.5` and `c1 = 0` put grid rows exactly on the tie.
        for c1 in [0.5, 0.0] {
            check(&format!("{name} c1={c1}"), &x, &y, &scaled(1.0, 0.0, op, vec![input(0), constant(c1)]));
        }
        check(
            &format!("{name}(x0, x1)"),
            &x,
            &targets(&x, |r| 0.5 * f(r[0], r[1]) + 1.0),
            &scaled(1.0, 0.0, op, vec![input(0), input(1)]),
        );
        // A max against a constant on the left, as in `max(0, x)`.
        check(
            &format!("{name}(c1, x0)"),
            &x,
            &y,
            &scaled(1.0, 0.0, op, vec![constant(0.5), input(0)]),
        );
    }
}

#[test]
fn atan2_device_fit_matches_the_cpu_including_the_origin() {
    need_device!();
    let x = rows();
    let y = targets(&x, |r| 1.5 * r[0].atan2(1.0) + 0.5);
    // `c1 = 0` makes `x0 = 0` rows hit `atan2(0, 0)`, where the adjoint is zero.
    for c1 in [0.5, 0.0] {
        check(&format!("atan2(x0, c1={c1})"), &x, &y, &scaled(1.0, 0.0, Op::Atan2, vec![input(0), constant(c1)]));
    }
    let y2 = targets(&x, |r| 0.8 * (r[1] + 3.0).atan2(r[0] + 2.5) - 0.2);
    check(
        "atan2(x1 + c1, x0 + c3)",
        &x,
        &y2,
        &scaled(
            1.0,
            0.0,
            Op::Atan2,
            vec![
                node(Op::Add, vec![input(1), constant(2.0)]),
                node(Op::Add, vec![input(0), constant(2.0)]),
            ],
        ),
    );
}

#[test]
fn mod_device_fit_matches_the_cpu_at_exact_multiples() {
    need_device!();
    let x = rows();
    let m = |a: f64, b: f64| a - b * (a / b).floor();
    let y = targets(&x, |r| 0.7 * m(r[0], 1.25) + 0.1);
    // `c1 = 1.5` and `c1 = 1.25` put grid rows exactly on multiples of c1.
    for c1 in [1.5, 1.25] {
        check(&format!("mod(x0, c1={c1})"), &x, &y, &scaled(1.0, 0.0, Op::Mod, vec![input(0), constant(c1)]));
    }
    let y2 = targets(&x, |r| 0.4 * m(r[0], r[1].abs() + 0.5) + 0.3);
    check(
        "mod(x0, |x1| + c1)",
        &x,
        &y2,
        &scaled(
            1.0,
            0.0,
            Op::Mod,
            vec![input(0), node(Op::Add, vec![node(Op::Abs, vec![input(1)]), constant(0.5)])],
        ),
    );
}

#[test]
fn piecewise_constant_device_fit_matches_the_cpu_at_the_jumps() {
    need_device!();
    let x = rows();
    let cases: [(&str, Op, fn(f64) -> f64); 5] = [
        ("step", Op::Step, |a| if a > 0.0 { 1.0 } else { 0.0 }),
        ("sign", Op::Sign, |a| if a > 0.0 { 1.0 } else if a < 0.0 { -1.0 } else { 0.0 }),
        ("floor", Op::Floor, f64::floor),
        ("ceil", Op::Ceil, f64::ceil),
        ("round", Op::Round, f64::round),
    ];
    for (name, op, f) in cases {
        let y = targets(&x, |r| 2.0 * f(r[0]) - 0.5);
        check(name, &x, &y, &scaled(1.0, 0.0, op, vec![input(0)]));
        // The argument shifted by a constant: a value-only operator has a zero
        // adjoint, so only the outer constants move.
        let y2 = targets(&x, |r| 2.0 * f(r[0] + r[1]) - 0.5);
        check(
            &format!("{name}(x0 + x1)"),
            &x,
            &y2,
            &scaled(1.0, 0.0, op, vec![node(Op::Add, vec![input(0), input(1)])]),
        );
    }
}

#[test]
fn comparisons_device_fit_matches_the_cpu_on_equal_rows() {
    need_device!();
    let x = rows();
    assert!(x.iter().filter(|r| r[0] == r[1]).count() >= ROWS / 7, "tie rows present");
    let cases: [(&str, Op, fn(f64, f64) -> bool); 5] = [
        ("equal", Op::Equal, |a, b| a == b),
        ("less", Op::Less, |a, b| a < b),
        ("less_or_equal", Op::LessOrEqual, |a, b| a <= b),
        ("greater", Op::Greater, |a, b| a > b),
        ("greater_or_equal", Op::GreaterOrEqual, |a, b| a >= b),
    ];
    for (name, op, f) in cases {
        let y = targets(&x, |r| if f(r[0], r[1]) { 1.75 } else { -0.5 });
        check(name, &x, &y, &scaled(1.0, 0.0, op, vec![input(0), input(1)]));
        let y2 = targets(&x, |r| if f(r[0], 0.0) { 1.75 } else { -0.5 });
        check(&format!("{name} x0 vs 0"), &x, &y2, &scaled(1.0, 0.0, op, vec![input(0), constant(0.0)]));
    }
}

#[test]
fn logic_device_fit_matches_the_cpu_on_the_product_surrogate() {
    need_device!();
    let x = rows();
    let t = |a: f64| a > 0.0;
    let cases: [(&str, Op, fn(bool, bool) -> bool); 3] = [
        ("and", Op::And, |a, b| a && b),
        ("or", Op::Or, |a, b| a || b),
        ("xor", Op::Xor, |a, b| a != b),
    ];
    for (name, op, f) in cases {
        let y = targets(&x, |r| if f(t(r[0]), t(r[1])) { 1.25 } else { 0.25 });
        check(name, &x, &y, &scaled(1.0, 0.0, op, vec![input(0), input(1)]));
        // A constant operand gets the surrogate partial through the adjoint.
        check(
            &format!("{name}(x0, c1)"),
            &x,
            &y,
            &scaled(1.0, 0.0, op, vec![input(0), node(Op::Add, vec![input(1), constant(0.0)])]),
        );
    }
    let y = targets(&x, |r| if t(r[0]) { 0.25 } else { 1.25 });
    check("not", &x, &y, &scaled(1.0, 0.0, Op::Not, vec![input(0)]));
}

/// A branch of `if` that is `NaN` on the rows it is not taken.
#[test]
fn a_hard_operator_under_if_ignores_a_nan_in_the_untaken_branch() {
    need_device!();
    let x = rows();
    let y = targets(&x, |r| if r[0] > 0.0 { 1.5 * r[1].min(0.5) + 0.25 } else { 0.75 });
    let taken = scaled(1.0, 0.0, Op::Min, vec![input(1), constant(0.25)]);
    let other = node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(1.0), node(Op::Sqrt, vec![input(2)])]), constant(0.5)],
    );
    check("if(x0, c min(x1, c) + c, c sqrt(x2) + c)", &x, &y, &node(Op::If, vec![input(0), taken, other]));
}

/// Every constant sits under a value-only operator: the Jacobian has a zero
/// column. The kernel labels it and keeps the start; it never falls back.
#[test]
fn constants_only_under_a_value_only_operator_are_labelled_not_fallen_back() {
    need_device!();
    let x = rows();
    let y = targets(&x, |r| if r[0] > 0.3 { 1.0 } else { 0.0 });
    for op in [Op::Step, Op::Sign, Op::Floor, Op::Ceil, Op::Round] {
        let start = node(op, vec![node(Op::Mul, vec![constant(0.7), input(0)])]);
        let (dev, status) = device_fit(&x, &y, &start, 8);
        assert!(matches!(status, 4 | 6), "{op:?}: status {status}, want rank_deficient or no_improvement");
        assert_eq!(sse(&dev, &x, &y), sse(&start, &x, &y), "{op:?}: the start is kept");
    }
}

/// Non-finite and kink rows reach every new opcode in both sweeps. The CPU
/// loss is `inf` on such data, so only the launch and the label are checked.
#[test]
fn every_new_opcode_runs_on_nan_inf_and_kink_rows() {
    need_device!();
    let mut x = rows();
    for (i, r) in x.iter_mut().enumerate() {
        match i % 16 {
            3 => r[0] = f64::NAN,
            5 => r[0] = f64::INFINITY,
            9 => r[1] = f64::NEG_INFINITY,
            11 => r[1] = f64::NAN,
            13 => r[0] = 1e-13,
            _ => {}
        }
    }
    let y = targets(&rows(), |r| r[0] + r[1]);
    let unary = [Op::Step, Op::Sign, Op::Floor, Op::Ceil, Op::Round, Op::Not];
    let binary = [
        Op::Min, Op::Max, Op::Atan2, Op::Mod, Op::Equal, Op::Less, Op::LessOrEqual, Op::Greater,
        Op::GreaterOrEqual, Op::And, Op::Or, Op::Xor,
    ];
    for op in unary {
        let (_, status) = device_fit(&x, &y, &scaled(1.0, 0.0, op, vec![input(0)]), 4);
        eprintln!("{op:?}: status {status}");
    }
    for op in binary {
        for second in [input(1), constant(0.0), constant(1e-13)] {
            let (_, status) = device_fit(&x, &y, &scaled(1.0, 0.0, op, vec![input(0), second]), 4);
            eprintln!("{op:?}: status {status}");
        }
    }
}

fn sample(x: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

/// A device-dispatched search over the new operators fits on the device and
/// names no `unsupported_plan` fallback.
#[test]
fn a_search_with_hard_operators_fits_on_the_device() {
    need_device!();
    let x = rows();
    let y = targets(&x, |r| 1.5 * r[0].max(0.25) + if r[1] > r[0] { 1.0 } else { 0.0 });
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
        ops: vec![
            Op::Add, Op::Mul, Op::Min, Op::Max, Op::Step, Op::Sign, Op::Floor, Op::Mod, Op::Atan2,
            Op::Greater, Op::And, Op::Not,
        ],
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
