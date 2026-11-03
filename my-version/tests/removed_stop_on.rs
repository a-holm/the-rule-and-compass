//! Flags removed after the default-decision A/B round (#464) fail with a message that names the removal. Checkpoints written with them refuse to load.

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::search::{self, Config};
use std::process::Command;

fn rcsr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn bench(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr-bench"))
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
fn removed_stop_flags_are_usage_errors() {
    for args in [
        vec!["--stop-on", "front", "--data", "missing.csv"],
        vec!["--stop-on", "front:5", "--data", "missing.csv"],
    ] {
        assert_removed(&rcsr(&args), "--stop-on was removed (#464)");
    }
    assert_removed(
        &rcsr(&["--patience-tol", "1e-4", "--data", "missing.csv"]),
        "--patience-tol was removed (#464)",
    );
    assert_removed(&bench(&["--stop-on", "front"]), "--stop-on was removed (#464)");
    assert_removed(&bench(&["--patience-tol", "1e-4"]), "--patience-tol was removed (#464)");
}

#[test]
fn checkpoints_written_with_the_removed_settings_refuse_by_name() {
    let text = checkpoint_text();
    for line in ["config patience_tol 1e-4", "config stop_front 3"] {
        let bad = with_line_before(&text, "generations ", line);
        let err = Checkpoint::from_text(&bad).err().expect("a refusal");
        assert!(err.contains("which were removed (#464)"), "{line}: {err}");
    }
    let bad = with_line_before(&text, "data train ", "front_stable 3");
    let err = Checkpoint::from_text(&bad).err().expect("a refusal");
    assert!(err.contains("--stop-on front, which was removed (#464)"), "{err}");
}
