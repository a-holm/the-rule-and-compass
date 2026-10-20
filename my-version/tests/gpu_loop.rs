//! The GPU/search integration (issues #26 and #141).
//!
//! The guarantees pinned here:
//!
//! * a CUDA build with a usable device uses the GPU with no flag (#141);
//! * a non-CUDA build, or a CUDA build with no usable device, defaults to the
//!   CPU, warns once and records `backend`/`backend_reason` (#141);
//! * `--gpu` fails clearly when no device exists; `--no-gpu`/`--cpu` forces the
//!   CPU with or without the feature and is bit-identical across runs;
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
            "3",
            "--metric",
            "squared",
            "--cpu",
            "--json",
        ],
        Some("1"),
    );
    assert_eq!(code, Some(0));
    assert!(stdout.contains("\"backend\":\"cpu\""), "json: {stdout}");
    assert!(
        stdout.contains("\"backend_reason\":\"--cpu/--no-gpu\""),
        "json: {stdout}"
    );
    assert!(!stdout.contains("\"backend\":\"gpu\""), "json: {stdout}");
    // #141: an explicit CPU override is silent; the summary still names it.
    assert!(
        !stderr.contains("# gpu:"),
        "a forced CPU run must not print a GPU note, got: {stderr}"
    );
    assert!(
        stderr.contains("# resource: backend=cpu"),
        "expected the resource summary, got: {stderr}"
    );
}

#[test]
fn unavailable_device_falls_back_by_default_with_a_recorded_reason() {
    // `RCSR_GPU=0` makes a CUDA build behave as if no device were present.
    // No --gpu: the default must fall back, warn once, and record the reason.
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
            "--json",
        ],
        Some("0"),
    );
    assert_eq!(code, Some(0));
    assert!(stdout.contains("\"backend\":\"cpu\""), "json: {stdout}");
    assert!(stdout.contains("\"device\":null"), "json: {stdout}");
    let expected = if cfg!(feature = "cuda") {
        "RCSR_GPU=0"
    } else {
        "built without --features cuda"
    };
    assert!(
        stdout.contains(&format!("\"backend_reason\":\"{}", expected)),
        "expected a recorded fallback reason, json: {stdout}"
    );
    if cfg!(feature = "cuda") {
        assert!(
            stderr.contains("# gpu: unavailable") && stderr.contains(expected),
            "expected a visible device fallback, got: {stderr}"
        );
        // One note, not one per generation.
        assert_eq!(
            stderr.matches("# gpu:").count(),
            1,
            "a fallback must be reported once: {stderr}"
        );
    }
    assert!(
        stderr.contains("# resource: backend=cpu"),
        "expected the resource summary, got: {stderr}"
    );
}

#[test]
fn explicit_gpu_fails_clearly_when_no_device_is_available() {
    // #141: `--gpu` is an assertion, not a preference. `RCSR_GPU=0` makes it
    // fail on a CUDA machine; a non-CUDA build fails the same way.
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
            "--gpu",
        ],
        Some("0"),
    );
    assert_eq!(code, Some(2), "stderr: {stderr}");
    assert!(stdout.is_empty(), "no search output on a usage error");
    assert!(
        stderr.contains("error: --gpu was requested but no GPU is available"),
        "got: {stderr}"
    );
}

#[cfg(not(feature = "cuda"))]
#[test]
fn non_cuda_build_defaults_to_cpu() {
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
            "6",
            "--metric",
            "squared",
            "--json",
        ],
        None,
    );
    assert_eq!(code, Some(0));
    assert!(stdout.contains("\"backend\":\"cpu\""), "json: {stdout}");
    assert!(
        stdout.contains("\"backend_reason\":\"built without --features cuda\""),
        "json: {stdout}"
    );
    assert!(
        stderr.contains("# resource: backend=cpu"),
        "got: {stderr}"
    );
}

#[cfg(feature = "cuda")]
#[test]
fn cuda_build_defaults_to_gpu_when_a_device_is_available() {
    // On a machine with no usable device this test cannot assert the GPU
    // default; the fallback test above covers that case. `RCSR_GPU=1` is not
    // an "off" value, so the probe reads the real device.
    if !rcsr::backend::probe().available() {
        return;
    }
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
            "9",
            "--metric",
            "squared",
            "--json",
        ],
        Some("1"),
    );
    assert_eq!(code, Some(0));
    assert!(stdout.contains("\"backend\":\"gpu\""), "json: {stdout}");
    assert!(
        !stdout.contains("\"device\":null"),
        "a GPU run must name its device, json: {stdout}"
    );
    assert!(
        stderr.contains("# resource: backend=gpu device="),
        "got: {stderr}"
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
    // The two runs may select *different algebraically equivalent* forms, so
    // the rows can differ at f64 rounding level: the observed worst case with
    // archive-parent feedback is 6.6e-12. The bound still rejects a genuinely
    // different law (nine significant digits must agree).
    assert!(worst <= 1e-9, "laws differ by {worst:e} on the test rows");

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
fn cpu_override_is_bit_identical_across_runs() {
    // #141: `--cpu` (Config::gpu = false) is the reproducibility escape hatch.
    // Two CPU runs must agree bit for bit, including the reported f64 errors.
    let train = sample(48);
    let test = sample(48);
    let a = search::run(&config(23, false), &train, &test);
    let b = search::run(&config(23, false), &train, &test);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(
        a.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
        b.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(a.backend, rcsr::backend::Kind::Cpu);
    assert!(a.device.is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn four_island_gpu_run_with_if_is_deterministic() {
    // Mixes the device pool, the parallel CPU fallback (`if` has no opcode)
    // and four island threads. Repeating the seed must give the same archive.
    let train = sample(48);
    let test = sample(48);
    let mut cfg = config(29, true);
    cfg.islands = 4;
    cfg.pop = 40;
    cfg.generations = 20;
    cfg.fit_iters = 0;
    cfg.ops.push(Op::If);
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(
        describe(&a),
        describe(&b),
        "four islands, the device pool and the if fallback are deterministic"
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
