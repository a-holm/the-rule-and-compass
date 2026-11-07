//! The removed --tree-arity flag (#471) fails with a message that names the removal. Checkpoints written with it refuse to load.

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

/// Insert `line` before the first line that starts with `prefix`.
fn with_line_before(text: &str, prefix: &str, line: &str) -> String {
    let at = text.find(&format!("\n{prefix}")).expect("an anchor line") + 1;
    format!("{}{}\n{}", &text[..at], line, &text[at..])
}

fn assert_removed(out: &std::process::Output, needle: &str) {
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains(needle), "{stderr}");
}

#[test]
fn removed_tree_arity_flag_is_a_usage_error() {
    for value in ["balanced", "uniform"] {
        assert_removed(
            &rcsr(&["--tree-arity", value, "--data", "missing.csv"]),
            "--tree-arity was removed (#471)",
        );
    }
    assert_removed(&bench(&["--tree-arity", "balanced"]), "--tree-arity was removed (#471)");
}

#[test]
fn checkpoint_with_tree_arity_refuses_by_name() {
    let bad = with_line_before(&checkpoint_text(), "generations ", "config tree_arity balanced");
    let err = Checkpoint::from_text(&bad).err().expect("a refusal");
    assert!(err.contains("--tree-arity balanced, which was removed (#471)"), "{err}");
}
