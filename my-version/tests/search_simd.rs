//! The search must be able to use the SoA/AVX evaluator without changing a
//! single result (issue #27, stage 2).
//!
//! The fast path compiles a candidate once and evaluates whole columns. The
//! historical scalar walk is still there, behind `Config::simd = false`
//! (`--no-simd`), and is the transparent fallback for a tree whose operators the
//! SoA compiler does not implement. These tests prove the three properties the
//! issue asks for:
//!
//! 1. a fixed seed produces a **bit-identical** archive on both paths;
//! 2. an unsupported operator falls back per tree and still agrees;
//! 3. the `--no-simd` flag is accepted and changes nothing but the path.

use std::io::Write;
use std::process::{Command, Stdio};

use rcsr::data::Sample;
use rcsr::eval_soa;
use rcsr::expr::{self, Node, Op};
use rcsr::metrics::{self, Metric};
use rcsr::search::{self, Config, Outcome};

/// A smooth, non-degenerate table with three inputs: 96 rows, no missing
/// values, no rounding traps.
fn sample(rows: usize) -> Sample {
    let mut x = Vec::with_capacity(rows);
    let mut y = Vec::with_capacity(rows);
    for i in 0..rows {
        let a = -1.5 + 3.0 * i as f64 / (rows - 1) as f64;
        let b = (i as f64 * 0.37).sin();
        let c = (i as f64 * 0.11).cos();
        y.push(1.3 * a.sin() + 0.7 * b * b + 0.2 * c);
        x.push(vec![a, b, c]);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

/// The same search except for the evaluator choice.
fn config(simd: bool, ops: Vec<Op>) -> Config {
    Config {
        generations: 8,
        islands: 2,
        pop: 40,
        max_complexity: 16,
        fit_iters: 5,
        patience: 0,
        seed: 27,
        ops,
        simd,
        ..Default::default()
    }
}

/// Every archive entry as bits: error, configured cost, node count and the
/// printed tree. `to_bits` is the strictest comparison available.
fn describe(out: &Outcome) -> Vec<(u64, u64, usize, String)> {
    let names = vec!["x0".to_string(), "x1".to_string(), "x2".to_string()];
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.cost.to_bits(),
                s.complexity,
                expr::to_string(&s.tree, &names),
            )
        })
        .collect()
}

fn test_errors(out: &Outcome) -> Vec<u64> {
    out.test_errors.iter().map(|e| e.to_bits()).collect()
}

#[test]
fn simd_and_scalar_produce_identical_archives() {
    let train = sample(96);
    let test = sample(63);
    // The default set is entirely inside `eval_soa::supports`, so every
    // candidate takes the new path when `simd` is on.
    let ops = Op::default_set();

    let fast = search::run(&config(true, ops.clone()), &train, &test);
    let slow = search::run(&config(false, ops), &train, &test);

    assert!(
        !fast.archive.items.is_empty(),
        "the search must find at least one solution"
    );
    assert_eq!(
        describe(&fast),
        describe(&slow),
        "the SIMD and scalar walks must leave the archive bit-identical"
    );
    assert_eq!(
        test_errors(&fast),
        test_errors(&slow),
        "test errors must be bit-identical too"
    );
    assert_eq!(fast.generations, slow.generations);
}

#[test]
fn unsupported_operator_falls_back_per_tree() {
    let train = sample(61);

    // A tree the SoA compiler rejects, in the middle of an otherwise supported
    // expression. `erf` is deliberately outside the supported set.
    let tree = expr::node(
        Op::Add,
        vec![
            expr::node(Op::Mul, vec![expr::input(0), expr::constant(2.0)]),
            expr::node(Op::Erf, vec![expr::input(1)]),
        ],
    );
    assert!(
        eval_soa::compile(&tree).is_none(),
        "erf must force the scalar fallback"
    );

    let got = search::eval_error(&tree, &train, Metric::Absolute);
    let pred: Vec<f64> = train.x.iter().map(|row| expr::eval(&tree, row)).collect();
    let want = metrics::score_weighted(
        &pred,
        &train.y,
        &train.w,
        Metric::Absolute,
        expr::count_constants(&tree),
    );
    assert_eq!(
        got.to_bits(),
        want.to_bits(),
        "the fallback must return the scalar walk's exact error"
    );

    // A whole search whose operator set mixes supported and unsupported
    // operators: supported trees take the SoA path, unsupported ones fall back.
    let ops = vec![
        Op::Add,
        Op::Sub,
        Op::Mul,
        Op::Div,
        Op::Sin,
        Op::Exp,
        Op::Erf,
        Op::Mod,
    ];
    let fast = search::run(&config(true, ops.clone()), &train, &train);
    let slow = search::run(&config(false, ops), &train, &train);
    assert_eq!(
        describe(&fast),
        describe(&slow),
        "a mixed supported/unsupported run must be identical"
    );
}

/// `y = 2*x0 + x1` on a grid; written as CSV for the binary.
fn csv(rows: usize) -> String {
    let mut s = String::from("x0,x1,y\n");
    for i in 0..rows {
        let a = -2.0 + 4.0 * i as f64 / (rows - 1) as f64;
        let b = (i as f64 * 0.23).cos();
        s.push_str(&format!("{a},{b},{}\n", 2.0 * a + b));
    }
    s
}

fn run(args: &[&str]) -> (Option<i32>, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn rcsr");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(csv(50).as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn no_simd_flag_is_accepted_and_changes_no_output() {
    let base = [
        "--data",
        "-",
        "--target",
        "y",
        "--vars",
        "x0,x1",
        "--generations",
        "4",
        "--pop",
        "24",
        "--seed",
        "5",
        "--no-fit",
        "--quiet",
    ];
    let (code_on, out_on, err_on) = run(&base);
    assert_eq!(code_on, Some(0), "stderr: {err_on}");

    let mut with_flag: Vec<&str> = base.to_vec();
    with_flag.push("--no-simd");
    let (code_off, out_off, err_off) = run(&with_flag);
    assert_eq!(code_off, Some(0), "stderr: {err_off}");

    assert_eq!(
        out_on, out_off,
        "the evaluator choice must not change the reported solutions"
    );

    // The flag is documented where every other search flag is.
    let help = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .arg("--help")
        .output()
        .expect("run --help");
    assert!(String::from_utf8_lossy(&help.stdout).contains("--no-simd"));
}

/// `Config::simd` is a path choice, not a saved search setting: a checkpoint
/// written by one path resumes under the other.
#[test]
fn evaluator_choice_is_not_a_search_setting() {
    let on = config(true, Op::default_set());
    let off = config(false, Op::default_set());
    assert!(on.same_search_settings(&off));
    assert!(on.search_setting_differences(&off).is_empty());
}

/// The fallback is per tree, not per run: a supported tree still compiles even
/// when the operator set also contains unsupported ones.
#[test]
fn supported_trees_still_compile_in_a_mixed_operator_set() {
    let supported = expr::node(
        Op::Sin,
        vec![expr::node(Op::Add, vec![expr::input(0), expr::input(1)])],
    );
    assert!(eval_soa::compile(&supported).is_some());
    let unsupported: Node = expr::node(Op::Erf, vec![expr::input(0)]);
    assert!(eval_soa::compile(&unsupported).is_none());
}
