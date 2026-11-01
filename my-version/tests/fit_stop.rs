//! Opt-in relative-improvement stop of the CPU LM (issue #569).

use rcsr::data::Sample;
use rcsr::fit::{self, LmStop};
use rcsr::search::{self, Config};
use rcsr::metrics::Metric;
use std::process::Command;

fn run_rcsr(extra: &[&str], out: &std::path::Path) -> std::process::Output {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", data, "--target", "y", "--vars", "x", "--seed", "3", "--islands", "1",
            "--pop", "60", "--generations", "20", "--no-front-parents", "--cpu", "--out",
            out.to_str().expect("utf-8 path"), "--quiet",
        ])
        .args(extra)
        .output()
        .expect("run rcsr")
}

fn front(extra: &[&str], tag: &str) -> String {
    let out = std::env::temp_dir().join(format!("rcsr-fit-stop-{}-{}.tsv", std::process::id(), tag));
    let output = run_rcsr(extra, &out);
    assert!(output.status.success(), "rcsr failed: {}", String::from_utf8_lossy(&output.stderr));
    let got = std::fs::read_to_string(&out).expect("read the front");
    let _ = std::fs::remove_file(&out);
    got
}

#[test]
fn fit_stop_is_deterministic_and_thread_invariant() {
    // The default metric (absolute) and the squared metric, where the stop fires.
    for metric in [None, Some("squared")] {
        let mut args: Vec<&str> = vec!["--fit-stop", "on"];
        if let Some(m) = metric {
            args.extend(["--metric", m]);
        }
        let tag = metric.unwrap_or("absolute");
        let a = front(&args, &format!("{tag}-a"));
        let b = front(&args, &format!("{tag}-b"));
        assert_eq!(a, b, "{tag}: two runs differ");
        let mut one = args.clone();
        one.extend(["--threads", "1"]);
        let mut eight = args.clone();
        eight.extend(["--threads", "8"]);
        let t1 = front(&one, &format!("{tag}-t1"));
        let t8 = front(&eight, &format!("{tag}-t8"));
        assert_eq!(t1, t8, "{tag}: --threads 1 and 8 differ");
        assert_eq!(t1, a, "{tag}: --threads 1 differs from the default");
    }
}

fn known_truth(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-2;
        x.push(vec![a, b]);
        y.push(2.5 * (-0.7 * a).exp() * b.sin() + 0.8 * b + 0.3 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

#[test]
fn reported_members_are_exact_refits_with_the_stop_on() {
    let train = known_truth(2000);
    let test = known_truth(200);
    let cfg = Config {
        generations: 12,
        islands: 2,
        pop: 40,
        max_complexity: 12,
        metric: Metric::Squared,
        fit_iters: 8,
        seed: 1,
        gpu: false,
        threads: 2,
        fit_stop: Some(LmStop::ON),
        ..Default::default()
    };
    let out = search::run(&cfg, &train, &test);
    assert!(out.fit_work.lm_stops > 0, "the stop never fired: {:?}", out.fit_work);
    assert!(
        out.fit_work.refits >= out.archive.items.len(),
        "refits {} front {}",
        out.fit_work.refits,
        out.archive.items.len()
    );
    let n = train.x.len() as f64;
    for member in &out.archive.items {
        let mut tree = member.tree.clone();
        fit::fit_weighted_metric(&mut tree, &train.x, &train.y, &[], cfg.fit_iters, cfg.metric);
        let mse = train
            .x
            .iter()
            .zip(&train.y)
            .map(|(row, t)| {
                let d = rcsr::expr::eval(&tree, row) - t;
                d * d
            })
            .sum::<f64>()
            / n;
        let rel = (mse - member.error).abs() / member.error.max(1e-12);
        assert!(rel < 1e-9, "member {} error {} refit {}", rcsr::expr::to_string(&member.tree, &["x0".into(), "x1".into()]), member.error, mse);
    }
}

#[test]
fn fit_stop_refuses_bad_values() {
    let out = std::env::temp_dir().join(format!("rcsr-fit-stop-bad-{}.tsv", std::process::id()));
    for bad in ["0", "1", "1e-8,0", "nan", "-1e-8", "x"] {
        let o = run_rcsr(&["--fit-stop", bad], &out);
        assert_eq!(o.status.code(), Some(2), "--fit-stop {bad}: {}", String::from_utf8_lossy(&o.stderr));
    }
    for good in ["off", "on", "1e-8", "1e-8,3"] {
        let o = run_rcsr(&["--fit-stop", good, "--generations", "2"], &out);
        assert!(o.status.success(), "--fit-stop {good}: {}", String::from_utf8_lossy(&o.stderr));
    }
    let _ = std::fs::remove_file(&out);

    let dir = std::env::temp_dir();
    let cp = dir.join(format!("rcsr-fit-stop-cp-{}.rcsr", std::process::id()));
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let base = ["--data", data, "--target", "y", "--vars", "x", "--seed", "3", "--pop", "20", "--cpu", "--quiet"];
    let made = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(base)
        .args(["--generations", "3", "--fit-stop", "on", "--save", cp.to_str().unwrap()])
        .output()
        .expect("run rcsr");
    assert!(made.status.success(), "{}", String::from_utf8_lossy(&made.stderr));
    let text = std::fs::read_to_string(&cp).expect("checkpoint");
    assert!(text.contains("config fit_stop 1e-10,2\n"), "{text}");
    let refused = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(base)
        .args(["--resume", cp.to_str().unwrap(), "--fit-stop", "1e-8"])
        .output()
        .expect("run rcsr");
    assert!(!refused.status.success());
    let err = String::from_utf8_lossy(&refused.stderr);
    assert!(err.contains("--fit-stop"), "{err}");
    let _ = std::fs::remove_file(&cp);
}
