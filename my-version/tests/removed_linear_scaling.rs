//! Flags removed after the default-decision A/B round fail with a message
//! that names the removal. Checkpoints written with them refuse to load.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::search::{self, Config};

fn rcsr(args: &[&str]) -> std::process::Output {
    rcsr_cmd::rcsr()
        .args(args)
        .output()
        .expect("run rcsr")
}

fn bench(args: &[&str]) -> std::process::Output {
    rcsr_cmd::rcsr_bench()
        .args(args)
        .output()
        .expect("run rcsr-bench")
}

fn checkpoint_text() -> String {
    let x: Vec<Vec<f64>> = (0..40).map(|i| vec![i as f64 * 0.1]).collect();
    let y: Vec<f64> = x.iter().map(|r| 2.0 * r[0] + 1.0).collect();
    let train = Sample::new(x.clone(), y.clone());
    let test = Sample::new(x, y);
    let cfg = Config {
        generations: 2,
        pop: 8,
        max_complexity: 8,
        fit_iters: 2,
        seed: 3,
        gpu: false,
        ..Default::default()
    };
    let mut text = None;
    search::run_resumable(&cfg, &train, &test, None, 0, |cp| text = Some(cp.to_text()))
        .expect("fresh run");
    text.expect("a checkpoint callback")
}

fn with_config_line(text: &str, line: &str) -> String {
    let at = text.find("\nconfig ").expect("a config line") + 1;
    format!("{}{}\n{}", &text[..at], line, &text[at..])
}

#[test]
fn removed_linear_scaling_flag_is_a_usage_error() {
    let out = rcsr(&["--linear-scaling", "--data", "missing.csv"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--linear-scaling was removed (#470)"), "{stderr}");
    let out = bench(&["--linear-scaling"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("--linear-scaling was removed (#470)"), "{stderr}");
}

#[test]
fn checkpoint_with_linear_scaling_refuses_by_name() {
    let text = with_config_line(&checkpoint_text(), "config linear_scaling true");
    let err = Checkpoint::from_text(&text).err().expect("a refusal");
    assert!(err.contains("--linear-scaling, which was removed (#470)"), "{err}");
}
