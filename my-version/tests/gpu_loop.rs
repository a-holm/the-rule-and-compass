//! The GPU/search integration (issue #26).
//!
//! Three guarantees are pinned here:
//!
//! * `--no-gpu` forces the CPU path, with or without the `cuda` feature;
//! * a CUDA build with no usable device falls back with one visible note;
//! * with `--features cuda` the GPU-guided search is deterministic for a seed
//!   and the *reported* law and errors agree with the CPU backend (the numbers
//!   a user reads are always re-scored in `f64` on the full training sample).

#[cfg(feature = "cuda")]
use rcsr::data::Sample;
#[cfg(feature = "cuda")]
use rcsr::expr::{self, Op};
#[cfg(feature = "cuda")]
use rcsr::metrics::Metric;
#[cfg(feature = "cuda")]
use rcsr::search::{self, Config, Outcome};
use std::io::Write;
use std::process::{Command, Stdio};

/// `y = 2x + 1` on a deterministic grid with a little noise, two inputs.
#[cfg(feature = "cuda")]
fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 13) % n) as f64) / (n as f64) * 2.0 - 1.0;
        let noise = (((i * 37) % 7) as f64 - 3.0) * 1e-4;
        x.push(vec![a, b]);
        y.push(2.0 * a + 1.0 + noise);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
    }
}

#[cfg(feature = "cuda")]
fn config(seed: u32, gpu: bool) -> Config {
    Config {
        generations: 80,
        islands: 1,
        pop: 80,
        max_complexity: 18,
        fit_iters: 6,
        patience: 0,
        seed,
        metric: Metric::Squared,
        ops: vec![Op::Add, Op::Sub, Op::Mul, Op::Div],
        gpu,
        ..Default::default()
    }
}

/// A comparable view of the reported front: structural hash, complexity, and
/// the exact `f64` error bits.
#[cfg(feature = "cuda")]
fn describe(out: &Outcome) -> Vec<(u64, usize, u64)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                expr::structural_hash(&s.tree),
                s.complexity,
                s.error.to_bits(),
            )
        })
        .collect()
}

#[cfg(feature = "cuda")]
fn best(out: &Outcome) -> &rcsr::pareto::Sol {
    out.archive
        .items
        .iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap())
        .expect("a non-empty archive")
}

fn csv(n: usize) -> String {
    let mut s = String::from("x,y\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        s.push_str(&format!("{},{}\n", x, x * x + 1.0));
    }
    s
}

fn run_cli(args: &[&str], gpu_env: Option<&str>) -> (Option<i32>, String, String) {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr"));
    cmd.args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(v) = gpu_env {
        cmd.env("RCSR_GPU", v);
    }
    let mut child = cmd.spawn().expect("spawn rcsr");
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(csv(60).as_bytes())
        .unwrap();
    let out = child.wait_with_output().unwrap();
    (
        out.status.code(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn forced_cpu_flag_works_with_and_without_the_feature() {
    let (_code, stdout, stderr) = run_cli(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "8",
            "--pop",
            "24",
            "--seed",
            "3",
            "--metric",
            "squared",
            "--no-gpu",
        ],
        None,
    );
    assert!(!stdout.is_empty(), "the search still reports solutions");
    // #91: CPU is the default, so `--no-gpu` is accepted but silent — there is
    // no GPU to report being switched off.
    assert!(
        !stderr.contains("# gpu:"),
        "a CPU run must not print a GPU note, got: {stderr}"
    );
}

#[test]
fn cuda_build_without_a_device_falls_back_with_one_note() {
    // `RCSR_GPU=0` makes a CUDA build behave as if no device were present.
    let (code, stdout, stderr) = run_cli(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "8",
            "--pop",
            "24",
            "--seed",
            "5",
            "--metric",
            "squared",
            // #91: the GPU is opt-in, so the fallback only happens when asked.
            "--gpu",
        ],
        Some("0"),
    );
    assert_eq!(code, Some(0));
    assert!(!stdout.is_empty());
    if cfg!(feature = "cuda") {
        assert!(
            stderr.contains("# gpu: unavailable (RCSR_GPU=0"),
            "expected a visible device fallback, got: {stderr}"
        );
    } else {
        assert!(
            stderr.contains("# gpu: unavailable (built without --features cuda)"),
            "expected the feature-less note, got: {stderr}"
        );
    }
    // One note, not one per generation.
    assert_eq!(
        stderr.matches("# gpu:").count(),
        1,
        "a fallback must be reported once: {stderr}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn recovered_law_and_reported_errors_agree_with_the_cpu_backend() {
    let train = sample(64);
    let test = sample(64);

    let cpu = search::run(&config(11, false), &train, &test);
    let gpu = search::run(&config(11, true), &train, &test);

    // The selected tree may be a *different but algebraically equivalent*
    // form (`1 + 2x` versus `2(0.5 + x)`), which is what the f32 ranking can
    // change. Compare the law itself: the predictions on the held-out rows.
    let (c, g) = (best(&cpu), best(&gpu));
    let pred = |tree: &rcsr::expr::Node| -> Vec<f64> {
        test.x.iter().map(|row| expr::eval(tree, row)).collect()
    };
    let (pc, pg) = (pred(&c.tree), pred(&g.tree));
    let worst = pc
        .iter()
        .zip(&pg)
        .map(|(a, b)| (a - b).abs() / (1.0 + a.abs().max(b.abs())))
        .fold(0.0f64, f64::max);
    // Observed on this fixture: 0 (the two forms agree bit for bit on these
    // rows). The asserted bound is the f64 rounding tolerance.
    assert!(worst <= 1e-12, "laws differ by {worst:e} on the test rows");

    let rel = |a: f64, b: f64| (a - b).abs() / (1.0 + a.abs().max(b.abs()));
    assert!(
        rel(c.error, g.error) <= 1e-9,
        "train error {} vs {}",
        c.error,
        g.error
    );
    // Reported test errors are recomputed by the CPU even in the GPU run.
    let ct = search::eval_error(&c.tree, &test, Metric::Squared);
    let gt = search::eval_error(&g.tree, &test, Metric::Squared);
    assert!(rel(ct, gt) <= 1e-12, "test error {} vs {}", ct, gt);
    assert!(
        g.error < 1e-6,
        "the fixture is recoverable up to its 1e-4 noise, got {}",
        g.error
    );
}

#[cfg(feature = "cuda")]
#[test]
fn batched_generation_is_deterministic_given_the_seed() {
    let train = sample(48);
    let test = sample(48);
    let a = search::run(&config(17, true), &train, &test);
    let b = search::run(&config(17, true), &train, &test);
    assert_eq!(describe(&a), describe(&b), "the GPU generation path is deterministic");
    assert_eq!(
        a.test_errors
            .iter()
            .map(|e| e.to_bits())
            .collect::<Vec<_>>(),
        b.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
    );
}
