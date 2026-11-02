//! `--fit-irls` (issue #611): determinism, thread invariance, exact reported
//! members and refusals of the opt-in reweighted L1 schedules.

use rcsr::data::Sample;
use rcsr::fit::{self, FitIrls};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};
use std::path::PathBuf;
use std::process::{Command, Output};

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

struct TempPath(PathBuf);

impl TempPath {
    fn new(tag: &str) -> Self {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        TempPath(std::env::temp_dir().join(format!(
            "rcsr-fit-irls-{}-{}-{}",
            std::process::id(),
            n,
            tag
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

/// The documented baseline command of `tests/trajectory.rs` plus `extra`.
fn front(extra: &[&str]) -> String {
    let out = TempPath::new("front.tsv");
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let mut args = vec![
        "--data",
        data,
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--islands",
        "1",
        "--pop",
        "60",
        "--generations",
        "20",
        "--no-front-parents",
        "--cpu",
        "--out",
        out.as_str(),
        "--quiet",
    ];
    args.extend_from_slice(extra);
    let output = rcsr(&args);
    assert!(
        output.status.success(),
        "rcsr failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    std::fs::read_to_string(&out.0).expect("read the front")
}

#[test]
fn reweighted_is_deterministic_and_thread_invariant() {
    for value in ["reweighted", "reweighted-final"] {
        let first = front(&["--fit-irls", value]);
        let again = front(&["--fit-irls", value]);
        let one = front(&["--fit-irls", value, "--threads", "1"]);
        let eight = front(&["--fit-irls", value, "--threads", "8"]);
        assert_eq!(first, again, "{value}: two runs differ");
        assert_eq!(first, one, "{value}: --threads 1 differs");
        assert_eq!(first, eight, "{value}: --threads 8 differs");
    }
}

/// A known-truth table with Laplace-like noise, so the L1 fit is the one that
/// matters: `y = 1.7 x0 + 0.6 sin(2 x1) + 0.4`.
fn known_truth(rows: usize) -> Sample {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(rows);
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        let a = 4.0 * next() - 2.0;
        let b = 4.0 * next() - 2.0;
        let u = next() - 0.5;
        let noise = -0.05 * u.signum() * (1.0 - 2.0 * u.abs()).ln();
        x.push(vec![a, b]);
        y.push(1.7 * a + 0.6 * (2.0 * b).sin() + 0.4 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

#[test]
fn reported_members_are_exact_refits() {
    let train = known_truth(2000);
    let test = known_truth(400);
    for schedule in [FitIrls::Reweighted, FitIrls::ReweightedFinal] {
        let cfg = Config {
            generations: 12,
            islands: 1,
            pop: 40,
            max_complexity: 12,
            seed: 7,
            gpu: false,
            metric: Metric::Absolute,
            fit_irls: schedule,
            ..Default::default()
        };
        let out = search::run(&cfg, &train, &test);
        assert!(!out.archive.items.is_empty());
        for member in &out.archive.items {
            let mut tree = member.tree.clone();
            let again = fit::fit_weighted_metric(
                &mut tree,
                &train.x,
                &train.y,
                &train.w,
                cfg.fit_iters,
                cfg.metric,
            );
            let scale = member.error.abs().max(f64::MIN_POSITIVE);
            assert!(
                (again - member.error).abs() < 1e-9 * scale,
                "{schedule:?}: member {} error {} changes to {again} on today's IRLS",
                member.complexity,
                member.error
            );
        }
    }
}

#[test]
fn fit_irls_refuses_bad_values() {
    let csv = TempPath::new("line.csv");
    let mut text = String::from("x,y\n");
    for i in 0..40 {
        let x = i as f64 / 10.0;
        text.push_str(&format!("{x},{}\n", 2.0 * x + 1.0 + 0.01 * ((i * 7) % 5) as f64));
    }
    std::fs::write(&csv.0, text).expect("write csv");
    let bad = rcsr(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--fit-irls",
        "fast",
    ]);
    assert_eq!(bad.status.code(), Some(2), "{}", String::from_utf8_lossy(&bad.stderr));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("--fit-irls"));

    let checkpoint = TempPath::new("line.rcsr");
    let common = |extra: &[&str]| {
        let mut args = vec![
            "--data",
            csv.as_str(),
            "--target",
            "y",
            "--vars",
            "x",
            "--seed",
            "1",
            "--islands",
            "1",
            "--pop",
            "20",
            "--cpu",
            "--quiet",
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    let made = common(&["--generations", "3", "--fit-irls", "reweighted", "--save", checkpoint.as_str()]);
    assert!(made.status.success(), "{}", String::from_utf8_lossy(&made.stderr));
    let text = std::fs::read_to_string(&checkpoint.0).expect("read checkpoint");
    assert!(text.contains("config fit_irls reweighted\n"));

    let refused = common(&["--generations", "6", "--fit-irls", "reweighted-final", "--resume", checkpoint.as_str()]);
    assert_eq!(refused.status.code(), Some(2), "{}", String::from_utf8_lossy(&refused.stderr));
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("--fit-irls reweighted-final (saved reweighted)"), "{err}");

    let same = common(&["--generations", "6", "--fit-irls", "reweighted", "--resume", checkpoint.as_str()]);
    assert!(same.status.success(), "{}", String::from_utf8_lossy(&same.stderr));
    let implicit = common(&["--generations", "6", "--resume", checkpoint.as_str()]);
    assert!(implicit.status.success(), "{}", String::from_utf8_lossy(&implicit.stderr));

    let default_text = {
        let plain = TempPath::new("plain.rcsr");
        let made = common(&["--generations", "3", "--save", plain.as_str()]);
        assert!(made.status.success());
        std::fs::read_to_string(&plain.0).expect("read checkpoint")
    };
    assert!(!default_text.contains("fit_irls"), "the default writes no record");
}
