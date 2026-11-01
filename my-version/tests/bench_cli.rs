//! #462: the `rcsr-bench` command line: NDJSON streaming, multi-seed runs and
//! the paired `compare` command.

use rcsr::bench_stats::{parse_json, Json};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::process::{Command, Stdio};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_rcsr-bench"))
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-bench-cli-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir.join(name)
}

const SMALL: [&str; 12] = [
    "--suite",
    "hard",
    "--problems",
    "nguyen9,korns1",
    "--seeds",
    "1-2",
    "--noise",
    "0",
    "--generations",
    "5",
    "--islands",
    "1",
];

fn small_run() -> Vec<String> {
    let out = bin()
        .args(SMALL)
        .args(["--pop", "40", "--ndjson", "--cpu"])
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout)
        .unwrap()
        .lines()
        .map(str::to_string)
        .collect()
}

#[test]
fn ndjson_has_a_line_per_run_and_a_summary() {
    let lines = small_run();
    assert_eq!(lines.len(), 5, "{lines:?}");
    for l in &lines[..4] {
        let j = parse_json(l).unwrap();
        assert_eq!(j.get("type").and_then(Json::as_str), Some("run"));
        assert_eq!(j.get("suite").and_then(Json::as_str), Some("hard"));
        assert_eq!(j.get("backend").and_then(Json::as_str), Some("cpu"));
        assert!(j.get("config").is_some());
    }
    let last = parse_json(&lines[4]).unwrap();
    assert_eq!(last.get("type").and_then(Json::as_str), Some("summary"));
}

#[test]
fn compare_with_itself_has_no_discordant_pairs() {
    let lines = small_run();
    let path = scratch("self.ndjson");
    std::fs::write(&path, lines.join("\n")).unwrap();
    let out = bin()
        .args(["compare", "--json", "--base"])
        .arg(&path)
        .arg("--treat")
        .arg(&path)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let j = parse_json(&String::from_utf8(out.stdout).unwrap()).unwrap();
    assert_eq!(j.get("b").and_then(Json::as_f64), Some(0.0));
    assert_eq!(j.get("c").and_then(Json::as_f64), Some(0.0));
    assert_eq!(j.get("mcnemar_p").and_then(Json::as_f64), Some(1.0));
    assert_eq!(j.get("pairs").and_then(Json::as_f64), Some(4.0));
}

#[test]
fn compare_exits_2_on_mismatched_key_sets() {
    let lines = small_run();
    let a = scratch("keys-a.ndjson");
    let b = scratch("keys-b.ndjson");
    std::fs::write(&a, lines.join("\n")).unwrap();
    std::fs::write(&b, lines[..3].join("\n")).unwrap();
    let out = bin().arg("compare").arg("--base").arg(&a).arg("--treat").arg(&b).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("do not hold the same runs"));
}

#[test]
fn compare_lists_pairs_with_an_undefined_criterion() {
    let run = |seed: u32, oracle: &str| {
        format!("{{\"type\":\"run\",\"suite\":\"hard\",\"problem\":\"p\",\"noise\":0.1,\"seed\":{seed},\"recovered\":true,\"oracle_recovered\":{oracle},\"signal_recovered\":null,\"seconds\":1.0,\"config\":{{\"generator\":\"g\"}}}}")
    };
    let a = scratch("undef-a.ndjson");
    let b = scratch("undef-b.ndjson");
    std::fs::write(&a, [run(1, "true"), run(2, "null")].join("\n")).unwrap();
    std::fs::write(&b, [run(1, "false"), run(2, "true")].join("\n")).unwrap();
    let out = bin()
        .args(["compare", "--criterion", "oracle_recovered", "--base"])
        .arg(&a)
        .arg("--treat")
        .arg(&b)
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("pairs=1"), "{text}");
    assert!(text.contains("hard/p/noise 0.1/seed 2"), "{text}");
}

#[test]
fn compare_exits_2_on_a_changed_data_generator() {
    let lines = small_run();
    let a = scratch("gen-a.ndjson");
    let b = scratch("gen-b.ndjson");
    std::fs::write(&a, lines.join("\n")).unwrap();
    std::fs::write(&b, lines.join("\n").replace("\"generator\":\"bench-v2\"", "\"generator\":\"bench-v3\"")).unwrap();
    let out = bin().arg("compare").arg("--base").arg(&a).arg("--treat").arg(&b).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("different data configuration"));
}

#[test]
fn unknown_problem_is_a_usage_error() {
    let out = bin().args(["--suite", "hard", "--problems", "nope"]).output().unwrap();
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn list_prints_the_new_suites() {
    let hard = bin().args(["--list", "--suite", "hard"]).output().unwrap();
    let text = String::from_utf8(hard.stdout).unwrap();
    assert_eq!(text.lines().filter(|l| l.contains("nguyen") || l.contains("keijzer") || l.contains("feynman") || l.contains("korns") || l.contains("pagie") || l.contains("vladislavleva") || l.contains("freq_sine") || l.contains("rational_r1") || l.contains("damped_freq")).count(), 20);
    assert!(text.contains("truth_cx"));
    let sig = bin().args(["--list", "--suite", "signal"]).output().unwrap();
    let text = String::from_utf8(sig.stdout).unwrap();
    assert!(text.contains("sig_tanh") && text.contains("sig_regime") && text.contains("sig_ar"));
    assert!(text.contains("0.20") && text.contains("0.10") && text.contains("0.05"));
}

/// The first line is readable while the next run is still going.
#[test]
fn ndjson_lines_are_flushed_as_runs_finish() {
    let mut child = bin()
        .args([
            "--suite", "hard", "--problems", "nguyen9,pagie1", "--noise", "0", "--generations",
            "400", "--islands", "1", "--pop", "200", "--patience", "0", "--ndjson", "--cpu",
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut reader = BufReader::new(child.stdout.take().unwrap());
    let mut first = String::new();
    reader.read_line(&mut first).unwrap();
    assert!(first.contains("\"problem\": \"nguyen9\""), "{first}");
    let still_running = child.try_wait().unwrap().is_none();
    let _ = child.kill();
    let _ = child.wait();
    assert!(still_running, "the process had already finished when the first line was read");
}
