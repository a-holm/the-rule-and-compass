//! The tracked launcher contract (issue #140).
//!
//! The engine-side manifest/checkpoint guarantees are pinned elsewhere. These
//! tests keep the *tracked* launchers honest: stable seeds, an explicit backend,
//! a manifest, atomic final JSON, a progress log instead of `/dev/null`, and a
//! real end-to-end run of `scripts/run-long.sh`.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scripts_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../scripts")
}

fn check_launcher_text(name: &str, text: &str, hash_marker: &str) {
    // #211: the development log is on by default; a launcher names one log
    // per run so windows never share `rcsr.dev.ndjson`.
    for flag in ["--run-manifest", "--json-out", "--progress", "--dev-log"] {
        assert!(text.contains(flag), "{name} does not pass {flag}");
    }
    assert!(
        text.contains("--gpu") && text.contains("--cpu"),
        "{name} must make the backend explicit"
    );
    assert!(
        !text.contains("DEVNULL"),
        "{name} redirects output to DEVNULL; keep a log instead"
    );
    assert!(
        text.contains(hash_marker),
        "{name} must derive the seed with a stable SHA-256"
    );
}

#[test]
fn every_tracked_launcher_publishes_a_verifiable_run() {
    let mut seen = 0;
    for entry in fs::read_dir(scripts_dir()).expect("read scripts/") {
        let path = entry.expect("dir entry").path();
        if path.file_name().and_then(|n| n.to_str()) != Some("run-long.sh") {
            continue;
        }
        seen += 1;
        let text = fs::read_to_string(&path).expect("read launcher");
        let name = path.file_name().unwrap().to_string_lossy();
        check_launcher_text(&name, &text, "sha256sum");
    }
    assert!(seen == 1, "expected the tracked shell launcher, found {seen}");
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/bin/rcsr_launch.rs");
    let text = fs::read_to_string(&source).expect("read rcsr-launch source");
    check_launcher_text("rcsr_launch.rs", &text, "sha256_hex");
}

fn launch_dry_run(out: &Path, label: &str) -> String {
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr-launch"))
        .args([
            "--data",
            "unused.csv",
            "--target",
            "y",
            "--vars",
            "x",
            "--backend",
            "cpu",
            "--seed",
            "7",
            "--out-dir",
            out.to_str().unwrap(),
            "--window",
            &format!("{label},100,200"),
            "--dry-run",
        ])
        .output()
        .expect("run rcsr-launch");
    assert!(output.status.success(), "{:?}", output);
    String::from_utf8_lossy(&output.stdout).to_string()
}

#[test]
fn the_seed_is_stable_across_processes() {
    // SHA-256("alpha:7") = 8855c828..., SHA-256("beta:7") = 98e9f534...; the
    // seed is the first 8 hex digits mod 100000.
    let out = unique_dir("seed");
    let a1 = launch_dry_run(&out, "alpha");
    let a2 = launch_dry_run(&out, "alpha");
    let b = launch_dry_run(&out, "beta");
    assert_eq!(a1, a2, "the resolved seed changed between processes");
    assert_ne!(a1, b, "different labels must not share a seed");
    assert!(a1.contains("# resolved_seed=23176 label=alpha backend=cpu\n"), "{a1}");
    assert!(b.contains("# resolved_seed=69492 label=beta backend=cpu\n"), "{b}");
    assert!(
        a1.contains(&format!(
            "--seed 23176 --generations 1000000 --save {0}/alpha.ckpt",
            out.display()
        )),
        "{a1}"
    );
    assert!(
        a1.contains("--train-select 'ts >= 0 and ts < -86300' --test-select 'ts >= 100 and ts < 200'"),
        "{a1}"
    );
    assert!(a1.trim_end().ends_with("--dev-log-level summary --cpu"), "{a1}");
    let _ = fs::remove_dir_all(out);
}

#[test]
fn run_long_writes_manifest_result_and_live_log() {
    let dir = unique_dir("runlong");
    fs::create_dir_all(&dir).expect("create temp dir");
    let csv = dir.join("data.csv");
    let mut text = String::from("x,y\n");
    for i in 0..40 {
        let x = -2.0 + 4.0 * i as f64 / 39.0;
        text.push_str(&format!("{:.6},{:.6}\n", x, 2.0 * x + 1.0));
    }
    fs::write(&csv, text).expect("write csv");

    let rcsr = Path::new(env!("CARGO_BIN_EXE_rcsr"));
    let bin_dir = rcsr.parent().expect("binary dir");
    let path = format!(
        "{}:{}",
        bin_dir.display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let out = dir.join("runs");
    let output = Command::new("bash")
        .arg(scripts_dir().join("run-long.sh"))
        .args([
            "--data",
            csv.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--label",
            "contract",
            "--seed",
            "3",
            "--generations",
            "4",
            "--cpu",
        ])
        .env("OUT_DIR", &out)
        .env("PATH", path)
        .output()
        .expect("run scripts/run-long.sh");
    assert!(
        output.status.success(),
        "launcher failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );

    let result = out.join("contract.result.json");
    let manifest = out.join("contract.manifest.json");
    let log = out.join("contract.stderr.log");
    let command = out.join("contract.command.txt");
    for path in [&result, &manifest, &log, &command] {
        assert!(path.exists(), "missing {}", path.display());
    }
    let json = fs::read_to_string(&result).expect("read result");
    assert!(json.trim_start().starts_with('{'), "result is not JSON: {json}");
    assert!(
        json.contains("\"backend\""),
        "the result must record the backend: {json}"
    );
    let log_text = fs::read_to_string(&log).expect("read log");
    assert!(!log_text.trim().is_empty(), "progress/error log is empty");
    let command_text = fs::read_to_string(&command).expect("read command");
    assert!(command_text.contains("--run-manifest"), "{command_text}");
    assert!(command_text.contains("resolved_seed="), "{command_text}");

    let verify = Command::new(rcsr)
        .args([
            "verify-manifest",
            manifest.to_str().unwrap(),
            "--data",
            csv.to_str().unwrap(),
        ])
        .output()
        .expect("run verifier");
    assert!(
        verify.status.success(),
        "manifest verification failed: {}",
        String::from_utf8_lossy(&verify.stdout)
    );
    assert!(String::from_utf8_lossy(&verify.stdout).contains("PASS"));

    // #256: the launcher keeps the exact binary beside the run, named by its
    // SHA-256, and the manifest verifies against it.
    let sha = rcsr::manifest::sha256_file(rcsr).expect("hash the binary");
    let preserved = out.join(format!("rcsr-{sha}"));
    assert!(preserved.exists(), "missing {}", preserved.display());
    let verify = Command::new(rcsr)
        .args([
            "verify-manifest",
            manifest.to_str().unwrap(),
            "--data",
            csv.to_str().unwrap(),
            "--binary",
            preserved.to_str().unwrap(),
        ])
        .output()
        .expect("run verifier with --binary");
    assert!(
        verify.status.success(),
        "preserved-binary verification failed: {}",
        String::from_utf8_lossy(&verify.stdout)
    );

    let _ = fs::remove_dir_all(dir);
}

fn unique_dir(tag: &str) -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static SEQ: AtomicUsize = AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "rcsr-launcher-{}-{}-{}",
        std::process::id(),
        n,
        tag
    ))
}

#[test]
fn rss_sampler_records_csv_and_summary_for_a_child() {
    let dir = unique_dir("rss");
    fs::create_dir_all(&dir).expect("create temp dir");
    let csv = dir.join("rss.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr-rss"))
        .args(["--csv", csv.to_str().unwrap(), "--interval", "0.02", "--", "sleep", "0.2"])
        .output()
        .expect("run rcsr-rss");
    assert!(output.status.success(), "{:?}", output);
    let text = fs::read_to_string(&csv).expect("read csv");
    let mut lines = text.lines();
    assert_eq!(lines.next(), Some("elapsed_s,vmrss_kib,vmhwm_kib,threads"));
    assert!(lines.count() >= 2, "expected at least 2 samples: {text}");
    let summary = String::from_utf8_lossy(&output.stdout);
    for key in [
        "command",
        "csv",
        "exit_code",
        "interval_s",
        "peak_rss_bytes",
        "peak_rss_kib_sampled",
        "peak_threads",
        "peak_vmhwm_kib",
        "samples",
        "wall_s",
    ] {
        assert!(summary.contains(&format!("\"{key}\": ")), "missing {key}: {summary}");
    }
    assert!(summary.contains("\"exit_code\": 0"), "{summary}");

    let failing = Command::new(env!("CARGO_BIN_EXE_rcsr-rss"))
        .args(["--", "sh", "-c", "exit 3"])
        .output()
        .expect("run rcsr-rss");
    assert_eq!(failing.status.code(), Some(3), "the child's status is propagated");
    let _ = fs::remove_dir_all(dir);
}
