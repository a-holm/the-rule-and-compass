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
        pairs: None,
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
    // An explicit --gpu failure may exit before reading stdin; a broken pipe
    // is expected and must not mask the exit code/stderr assertion.
    if let Some(stdin) = child.stdin.as_mut() {
        let _ = stdin.write_all(csv(60).as_bytes());
    }
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

/// #236/#287: a metric with no native score objective stays on the CPU —
/// visibly, with the score table's named refusal. `--metric slope --gpu` must print the labelled
/// reason, must not claim the device in the resource line or JSON, and must
/// return exactly the `--no-gpu` run: the CPU evaluator is the only one in both
/// runs, so any difference would mean the gate is leaking somewhere else.
#[cfg(feature = "cuda")]
#[test]
fn non_squared_metric_on_gpu_prints_the_reason_and_matches_the_cpu() {
    if !rcsr::backend::probe().available() {
        return;
    }
    let common = [
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
        "21",
        "--metric",
        "slope",
        "--json",
    ];
    let mut gpu_args = common.to_vec();
    gpu_args.extend_from_slice(&["--gpu"]);
    let (code, stdout, stderr) = run_cli(&gpu_args, Some("1"));
    assert_eq!(code, Some(0), "stderr: {stderr}");
    assert!(
        stdout.contains("\"metric\":\"slope\""),
        "the configured objective is still slope, json: {stdout}"
    );
    assert!(stdout.contains("\"backend\":\"cpu\""), "json: {stdout}");
    // #236: the default random split leaves shuffled slope pairs, which the
    // device scorer cannot difference.
    assert!(
        stdout.contains(
            "\"backend_reason\":\"slope pairs are not the consecutive rows of the training sample\""
        ),
        "the fallback must carry its named cause, json: {stdout}"
    );
    assert!(
        stderr.contains(
            "# gpu: off (slope pairs are not the consecutive rows of the training sample)"
        ),
        "the reason must be printed once, got: {stderr}"
    );
    assert!(
        stderr.contains("# resource: backend=cpu"),
        "the resource line must report the evaluator that ran, got: {stderr}"
    );
    assert!(
        !stderr.contains("# resource: backend=gpu"),
        "a CPU-scored run must not claim the device, got: {stderr}"
    );

    let mut cpu_args = common.to_vec();
    cpu_args.extend_from_slice(&["--no-gpu"]);
    let (cpu_code, cpu_stdout, cpu_stderr) = run_cli(&cpu_args, Some("1"));
    assert_eq!(cpu_code, Some(0), "stderr: {cpu_stderr}");
    // The reported front is the last JSON field; everything before it carries
    // wall-clock measurements that legitimately differ between runs.
    let front = |s: &str| s[s.find("\"pareto\":").expect("a JSON front")..].to_string();
    assert_eq!(
        front(&stdout),
        front(&cpu_stdout),
        "the gated GPU run must reproduce the --no-gpu front exactly"
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

/// #121: a generation that mixes native (`asinh` included) and unsupported
/// (`require`) candidates is deterministic, and the JSON reports the split and
/// the CPU fallback time/workers for every GPU generation.
#[cfg(feature = "cuda")]
#[test]
fn mixed_supported_and_unsupported_candidates_are_deterministic_and_reported() {
    if !rcsr::backend::probe().available() {
        return;
    }
    let train = sample(48);
    let test = sample(48);
    let mut cfg = config(37, true);
    cfg.generations = 12;
    cfg.pop = 36;
    cfg.islands = 2;
    cfg.fit_iters = 0;
    cfg.ops = vec![Op::Add, Op::Mul, Op::Asinh, Op::If, Op::Require];
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(describe(&a), describe(&b), "the mixed batch changed the archive");
    let gpu = a.gpu.expect("a GPU run reports gpu stats");
    assert!(!gpu.generations.is_empty());
    let mut saw_unsupported = false;
    for s in &gpu.generations {
        assert_eq!(s.native, s.programs);
        assert_eq!(s.fallbacks, s.unsupported_programs);
        saw_unsupported |= s.unsupported_programs > 0;
        if s.fallback_seconds > 0.0 {
            assert!(s.fallback_workers >= 1);
        }
    }
    assert!(
        saw_unsupported,
        "the fixture must exercise the CPU fallback, got {:?}",
        gpu.generations
    );

    // The CLI surface: the counters are real JSON keys, not just struct fields.
    let (code, stdout, stderr) = run_cli(
        &[
            "--data", "-", "--target", "y", "--vars", "x",
            "--generations", "3", "--pop", "24", "--seed", "37",
            "--metric", "squared", "--ops", "add,mul,asinh,require", "--gpu", "--json",
        ],
        Some("1"),
    );
    assert_eq!(code, Some(0), "stderr: {stderr}");
    for key in [
        "\"native\":",
        "\"supported\":",
        "\"unsupported_programs\":",
        "\"fallback_seconds\":",
        "\"fallback_workers\":",
        "\"totals\":",
    ] {
        assert!(stdout.contains(key), "missing {key} in json: {stdout}");
    }
    assert!(!stdout.contains("\"lowered\":"), "the lowering pass is gone: {stdout}");
}

/// #119: the overlap pipeline must not change a single scored value. The
/// synchronous path (`pipeline_depth = 0`) and the bounded overlap path are
/// compared on the same seed; every archive member, its exact `f64` error bits
/// and the test errors must match. Arrival order across islands is not fixed,
/// but each result is keyed by `(island, slot)`, so the mapping is.
#[cfg(feature = "cuda")]
#[test]
fn overlap_pipeline_is_byte_identical_to_the_synchronous_gpu_path() {
    let train = sample(96);
    let test = sample(96);
    let mut sync = config(29, true);
    sync.pipeline_depth = 0;
    let mut overlap = config(29, true);
    overlap.pipeline_depth = 2;

    let a = search::run(&sync, &train, &test);
    let b = search::run(&overlap, &train, &test);
    assert_eq!(
        describe(&a),
        describe(&b),
        "the overlap queue changed the archive"
    );
    assert_eq!(
        a.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
        b.test_errors.iter().map(|e| e.to_bits()).collect::<Vec<_>>(),
        "the overlap queue changed the reported test errors"
    );
}

/// #119: a CPU-only run cannot be affected by the pipeline depth. With no GPU
/// the queue is never constructed, so the archive is bit-identical for any
/// depth.
#[cfg(feature = "cuda")]
#[test]
fn pipeline_depth_is_a_no_op_on_the_cpu_path() {
    let train = sample(48);
    let test = sample(48);
    let mut sync = config(31, false);
    sync.pipeline_depth = 0;
    let mut deep = config(31, false);
    deep.pipeline_depth = 8;
    let a = search::run(&sync, &train, &test);
    let b = search::run(&deep, &train, &test);
    assert_eq!(describe(&a), describe(&b));
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

/// #118: a CPU run reports `"gpu":null` — never fabricated zeroes, and never a
/// duty or occupancy value it did not measure.
#[test]
fn cpu_json_reports_gpu_null() {
    let (code, stdout, _stderr) = run_cli(
        &[
            "--data",
            "-",
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "2",
            "--pop",
            "16",
            "--seed",
            "3",
            "--no-gpu",
            "--json",
        ],
        None,
    );
    assert_eq!(code, Some(0));
    assert!(
        stdout.contains("\"gpu\":null"),
        "a CPU run must report gpu:null, got: {stdout}"
    );
}

/// #118: a GPU run reports the measured ledger, keeps #131's per-generation
/// opcode report, and the host windows close to the independently measured
/// wall. Occupancy stays null: it needs ncu, not a guess from duty.
#[cfg(feature = "cuda")]
#[test]
fn gpu_run_reports_measured_accounting_not_occupancy() {
    let train = sample(64);
    let test = sample(64);
    let out = search::run(&config(11, true), &train, &test);
    let g = out.gpu.expect("the GPU path scored generations");
    let a = g.accounting;
    assert!(a.wall_s > 0.0, "the full path wall must be measured");
    assert!(a.batches > 0 && a.programs > 0 && a.evals > 0);
    assert!(a.evals == a.programs, "one candidate evaluation per program");
    assert_eq!(a.row_evals, a.programs * a.rows);
    assert!(a.kernel_s >= 0.0 && a.sync_s >= 0.0 && a.alloc_s >= 0.0);
    assert!(a.d2h_bytes > 0, "the SSE output must be downloaded");
    assert!(
        a.decomposition_within(0.01),
        "host windows should explain the wall within 1%: accounted {} vs wall {} (residual {:+.4})",
        a.accounted_s(),
        a.wall_s,
        a.residual_fraction()
    );
    assert!((0.0..=1.0).contains(&a.duty_cycle()));
    assert!(a.quality_per_gpu_hour(1.0, 10.0).is_some());
    // #284: the run ledger closes to its own wall and the run duty is the
    // interval union, not an evaluator-window ratio.
    let l = &g.ledger;
    assert!(l.wall_s() > 0.0);
    assert!(
        l.closure_residual().value.abs() < 0.01,
        "run ledger must close within 1%: stages {} + unaccounted {} vs wall {}",
        l.stage_sum_s(),
        l.unaccounted_s(),
        l.wall_s()
    );
    assert!(l.device_busy_s() <= l.device_busy_sum_s() + 1e-12);
    assert!((0.0..=1.0).contains(&l.duty_run().value));
    assert!((0.0..=1.0).contains(&l.eval_kernel_duty().value));
    assert!((0.0..=1.0).contains(&l.fit_kernel_duty().value));
    assert!(!l.intervals().is_empty(), "the evaluator must record a device interval");
    // #131 must still be present alongside the accounting.
    assert!(
        !g.generations.is_empty(),
        "the per-generation opcode report was removed by the accounting change"
    );
}

/// #118: a failed batch is still timed, adds no evaluations and cannot inflate
/// throughput. `eval_batch` with zero programs fails before the kernel.
#[cfg(feature = "cuda")]
#[test]
fn failed_batch_is_timed_but_not_counted_as_an_eval() {
    let cols = vec![0.0f32, 1.0, 2.0, 3.0];
    let y = vec![0.0f32, 1.0, 2.0, 3.0];
    let mut ev = rcsr::gpu::GpuEvaluator::new(&cols, 4, 1, &y).expect("a CUDA device");
    let (out, ms) = ev.eval_batch(&[], &[0]);
    assert!(out.is_empty());
    assert!(ms < 0.0, "an empty batch must report the error marker");
    let a = ev.stats();
    assert_eq!(a.batches, 1);
    assert_eq!(a.failed_batches, 1);
    assert_eq!(a.evals, 0);
    assert_eq!(a.evals_per_s(), 0.0);
    assert!(a.wall_s >= 0.0);
}

/// #284: a GPU run that writes checkpoints books the time in the `io` stage, its
/// device intervals are placed from CUDA event timestamps, and the ledger still
/// closes to the run wall within 1 %.
#[cfg(feature = "cuda")]
#[test]
fn gpu_run_ledger_books_checkpoint_io_and_uses_event_intervals() {
    let train = sample(64);
    let test = sample(64);
    let mut cfg = config(13, true);
    cfg.generations = 8;
    let mut saved = 0usize;
    let out = search::run_resumable(&cfg, &train, &test, None, 1, |_| {
        saved += 1;
        std::thread::sleep(std::time::Duration::from_millis(20));
    })
    .expect("a resumable GPU run");
    assert!(saved >= 2, "the run must write checkpoints, wrote {saved}");
    let g = out.gpu.expect("the GPU path scored generations");
    let l = &g.ledger;
    use rcsr::gpu_stats::{IntervalSource, Stage};
    assert!(
        l.stage_s(Stage::Io) >= 0.9 * 0.020 * saved as f64,
        "io stage must hold the checkpoint writes: {} s for {saved} writes",
        l.stage_s(Stage::Io)
    );
    assert_eq!(l.stage_s(Stage::Pnl), 0.0);
    assert_eq!(l.stage_s(Stage::Features), 0.0);
    assert!(
        l.closure_residual().value.abs() < 0.01,
        "closure with io booked: stages {} + unaccounted {} vs wall {}",
        l.stage_sum_s(),
        l.unaccounted_s(),
        l.wall_s()
    );
    assert!(!l.intervals().is_empty());
    assert!(
        l.intervals().iter().all(|i| i.source == IntervalSource::Event),
        "every device interval must come from event timestamps: {:?}",
        l.interval_counts()
    );
    assert_eq!(l.interval_basis(), "events");
    assert!(l.device_busy_s() <= l.wall_s() + 1e-9);
    assert!((0.0..=1.0).contains(&l.duty_run().value));
}
