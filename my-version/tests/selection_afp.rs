//! #465: age-fitness Pareto survivor selection (`--selection afp`).

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::search::{self, Config, Selection};
use std::process::Command;

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b * a + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn config(selection: Selection, generations: usize, threads: usize) -> Config {
    Config {
        generations,
        islands: 2,
        pop: 60,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        migration_interval: 3,
        seed: 5,
        threads,
        selection,
        ..Default::default()
    }
}

fn final_state(cfg: &Config, train: &Sample, test: &Sample) -> String {
    let mut text: Option<String> = None;
    search::run_resumable(cfg, train, test, None, 1, |cp| text = Some(cp.to_text()))
        .expect("run");
    text.expect("a checkpoint")
}

fn archive_hash(cfg: &Config, train: &Sample, test: &Sample) -> u64 {
    search::run(cfg, train, test).archive.hash()
}

#[test]
fn afp_is_deterministic() {
    let train = sample(300);
    let test = sample(80);
    let a = config(Selection::Afp, 12, 1);
    let first = archive_hash(&a, &train, &test);
    assert_eq!(first, archive_hash(&a, &train, &test), "same seed, same archive");
    assert_eq!(
        first,
        archive_hash(&config(Selection::Afp, 12, 4), &train, &test),
        "the thread count changed the archive"
    );
    assert_ne!(
        first,
        archive_hash(&config(Selection::Lexicographic, 12, 1), &train, &test),
        "afp must change the trajectory"
    );
}

#[test]
fn afp_checkpoint_resumes_to_the_uninterrupted_archive() {
    let train = sample(300);
    let test = sample(80);
    let full = archive_hash(&config(Selection::Afp, 20, 1), &train, &test);

    let text = final_state(&config(Selection::Afp, 10, 1), &train, &test);
    assert!(text.contains("config selection afp"));
    assert!(text.contains("\nages 0 "));
    let saved = Checkpoint::from_text(&text).expect("parse");
    let resumed = search::run_resumable(
        &config(Selection::Afp, 20, 1),
        &train,
        &test,
        Some(saved),
        0,
        |_| {},
    )
    .expect("resume");
    assert_eq!(full, resumed.archive.hash());
}

#[test]
fn lexicographic_checkpoints_carry_no_selection_or_ages() {
    let train = sample(300);
    let test = sample(80);
    let text = final_state(&config(Selection::Lexicographic, 4, 1), &train, &test);
    assert!(!text.contains("selection"));
    assert!(!text.contains("\nages "));
}

fn rcsr(args: &[&str]) -> std::process::Output {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", data, "--target", "y", "--vars", "x", "--cpu", "--quiet"])
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn afp_with_postfix_variation_exits_two() {
    let out = rcsr(&["--selection", "afp", "--variation", "postfix"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--selection afp runs on the node path only"), "{err}");
}

#[test]
fn unknown_selection_exits_two() {
    let out = rcsr(&["--selection", "nsga"]);
    assert_eq!(out.status.code(), Some(2));
}

/// The pinned `tests/trajectory.rs` command with `--selection lexicographic`
/// spelled out gives the pinned front.
#[test]
fn afp_off_is_bit_identical() {
    let out_path = std::env::temp_dir().join(format!("rcsr-afp-off-{}.tsv", std::process::id()));
    let out = rcsr(&[
        "--seed", "3", "--islands", "1", "--pop", "60", "--generations", "20",
        "--no-front-parents", "--selection", "lexicographic", "--out",
        out_path.to_str().expect("utf-8 path"),
    ]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let got = std::fs::read_to_string(&out_path).expect("front");
    let _ = std::fs::remove_file(&out_path);
    let want = "rank\tcomplexity\ttrain_error\ttest_error\texpression\n\
1\t1\t2.97670764\t3.66924796\t(0.5726 * (0.1305 + x))\n\
2\t2\t2.17852613\t2.42006255\texp((0.5726 * (0.1305 + x)))\n\
3\t3\t1.61548477\t1.58574957\t(3.9826 ^ (0.5726 * (0.1305 + x)))\n\
4\t4\t1.12695731\t1.05986720\tsinh((1.4645 + (0.5726 * (0.1305 + x))))\n\
5\t5\t0.00000045\t0.00000067\t(2.7389 + (2 * (0.1305 + x)))\n";
    assert_eq!(got, want);
}

/// Device scoring and the batched device fit under afp: immigrant slots, the
/// overlap pipeline (depth 0 and 2) and the thread count must not change the
/// archive. Skipped without a device.
#[cfg(feature = "cuda")]
#[test]
fn afp_is_deterministic_on_the_device() {
    use rcsr::dispatch::DispatchMode;
    use rcsr::fit::FitStrategy;
    use rcsr::metrics::Metric;
    let train = sample(96);
    let test = sample(96);
    for depth in [0, 2] {
        let run = |threads| {
            let cfg = Config {
                metric: Metric::Squared,
                gpu: true,
                dispatch: DispatchMode::Device,
                fit_strategy: FitStrategy::Gpu,
                pipeline_depth: depth,
                pop: 48,
                ..config(Selection::Afp, 4, threads)
            };
            search::run(&cfg, &train, &test)
        };
        let a = run(1);
        if a.backend != rcsr::backend::Kind::Gpu {
            eprintln!("skipped: no usable device ({:?})", a.backend_reason);
            return;
        }
        let b = run(1);
        let c = run(4);
        assert_eq!(a.archive.hash(), b.archive.hash(), "depth {depth}: same seed, same archive");
        assert_eq!(a.archive.hash(), c.archive.hash(), "depth {depth}: threads 4 changed the archive");
    }
}
