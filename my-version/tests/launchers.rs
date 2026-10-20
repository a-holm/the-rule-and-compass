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

#[test]
fn every_tracked_launcher_publishes_a_verifiable_run() {
    let mut seen = 0;
    for entry in fs::read_dir(scripts_dir()).expect("read scripts/") {
        let path = entry.expect("dir entry").path();
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext != "sh" && ext != "py" {
            continue;
        }
        seen += 1;
        let text = fs::read_to_string(&path).expect("read launcher");
        let name = path.file_name().unwrap().to_string_lossy();
        for flag in ["--run-manifest", "--json-out", "--progress"] {
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
        if ext == "py" {
            assert!(
                !text.contains("hash("),
                "{name} uses Python's salted hash(); derive the seed with SHA-256"
            );
            assert!(
                text.contains("hashlib.sha256"),
                "{name} must derive the seed with SHA-256"
            );
        }
        if ext == "sh" {
            assert!(
                text.contains("sha256sum"),
                "{name} must derive the seed with a stable hash"
            );
        }
    }
    assert!(seen >= 2, "expected the tracked launcher examples, found {seen}");
}

#[test]
fn the_seed_is_stable_across_processes() {
    // Python's hash(label) changes between interpreters because it is salted.
    // Two separate `--dry-run` invocations must produce the same seed, and a
    // different label must produce a different one.
    let py = Command::new("python3").arg("--version").output();
    if py.is_err() {
        eprintln!("python3 is unavailable; skipping the seed-stability check");
        return;
    }
    let out = unique_dir("seed");
    let run = |label: &str| {
        let output = Command::new("python3")
            .arg(scripts_dir().join("launch-windows.py"))
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
            .expect("run launcher");
        assert!(output.status.success(), "{:?}", output);
        String::from_utf8_lossy(&output.stdout).to_string()
    };
    let a1 = run("alpha");
    let a2 = run("alpha");
    let b = run("beta");
    assert_eq!(a1, a2, "the resolved seed changed between processes");
    assert_ne!(a1, b, "different labels must not share a seed");
    assert!(a1.contains("resolved_seed="), "{a1}");
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
