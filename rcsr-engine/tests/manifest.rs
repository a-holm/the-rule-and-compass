//! Run-manifest tests (issue #140).
//!
//! The manifest is evidence only if the verifier actually fails when the
//! binary, data or configuration changes. These tests run the CLI, verify a
//! manifest that should pass, then corrupt each check in turn.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::manifest;
use std::path::PathBuf;
use std::process::Output;

fn rcsr(args: &[&str]) -> Output {
    rcsr_cmd::rcsr()
        .args(args)
        .output()
        .expect("run rcsr")
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

struct TempPath(PathBuf, #[allow(dead_code)] Scratch);

impl TempPath {
    fn new(suffix: &str) -> Self {
        let scratch = Scratch::new("manifest");
        let path = scratch.join(suffix);
        TempPath(path, scratch)
    }
    fn as_str(&self) -> &str {
        self.0.to_str().expect("utf-8 path")
    }
}

fn run_with_manifest(data_path: &str, manifest: &str, json_out: &str) -> Output {
    rcsr(&[
        "--data",
        data_path,
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--generations",
        "4",
        "--pop",
        "20",
        "--max-complexity",
        "8",
        "--run-manifest",
        manifest,
        "--json-out",
        json_out,
        "--quiet",
    ])
}

#[test]
fn manifest_pass_then_each_check_fails_independently() {
    let manifest_path = TempPath::new("run.json");
    let json_path = TempPath::new("result.json");
    let data_path = TempPath::new("data.csv");
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv"),
        &data_path.0,
    )
    .expect("copy data fixture");

    let run = run_with_manifest(data_path.as_str(), manifest_path.as_str(), json_path.as_str());
    assert!(
        run.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    assert!(
        run.stdout.is_empty(),
        "--json-out must not stream the result to stdout: {:?}",
        String::from_utf8_lossy(&run.stdout)
    );
    let result = std::fs::read_to_string(&json_path.0).expect("result json");
    assert!(
        result.trim_start().starts_with('{') && result.trim_end().ends_with('}'),
        "the atomically published result must be one JSON object"
    );

    let text = std::fs::read_to_string(&manifest_path.0).expect("manifest");
    assert!(text.contains("\"completed\":true"), "manifest: {}", text);
    assert!(text.contains("\"end_utc\":\""), "manifest: {}", text);
    assert!(text.contains("\"sha256\":"), "manifest: {}", text);
    assert!(text.contains("\"phases_seconds\""), "manifest: {}", text);

    let verify = rcsr(&["verify-manifest", manifest_path.as_str()]);
    assert!(
        verify.status.success(),
        "a fresh manifest must verify: {}",
        String::from_utf8_lossy(&verify.stderr)
    );

    // 1. A stale binary hash.
    let binary = manifest::object(&text, "binary").expect("binary object").to_string();
    let sha = manifest::string_field(&binary, "sha256").expect("binary sha");
    let bad_binary = binary.replace(&sha, &"0".repeat(64));
    let bad = text.replacen(&binary, &bad_binary, 1);
    std::fs::write(&manifest_path.0, &bad).expect("write stale binary manifest");
    let failed = rcsr(&["verify-manifest", manifest_path.as_str()]);
    assert_eq!(failed.status.code(), Some(1), "stale binary must fail");
    assert!(
        String::from_utf8_lossy(&failed.stderr).contains("FAIL binary sha256"),
        "{}",
        String::from_utf8_lossy(&failed.stderr)
    );

    // 2. A changed config canonical text (hash not updated).
    let changed_config = text.replacen("config seed 3", "config seed 4", 1);
    assert_ne!(changed_config, text, "fixture must contain the config seed line");
    std::fs::write(&manifest_path.0, &changed_config).expect("write changed config");
    let failed = rcsr(&["verify-manifest", manifest_path.as_str()]);
    assert_eq!(failed.status.code(), Some(1), "changed config must fail");
    assert!(
        String::from_utf8_lossy(&failed.stderr).contains("FAIL config hash"),
        "{}",
        String::from_utf8_lossy(&failed.stderr)
    );

    // 3. Changed data: append a row to the file the manifest points at.
    std::fs::write(&manifest_path.0, &text).expect("restore manifest");
    let data_path = manifest::object(&text, "data").expect("data object");
    let path = manifest::string_field(data_path, "path").expect("data path");
    let original = std::fs::read_to_string(&path).expect("read data");
    std::fs::write(&path, format!("{}\n0,0\n", original.trim_end())).expect("append data row");
    let failed = rcsr(&["verify-manifest", manifest_path.as_str()]);
    std::fs::write(&path, original).expect("restore data");
    assert_eq!(failed.status.code(), Some(1), "changed data must fail");
    assert!(
        String::from_utf8_lossy(&failed.stderr).contains("FAIL data sha256"),
        "{}",
        String::from_utf8_lossy(&failed.stderr)
    );
}

#[test]
fn version_prints_a_cuda_line_only_in_a_cuda_build() {
    let out = rcsr(&["--version"]);
    assert!(out.status.success());
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let lines: Vec<&str> = text.lines().collect();
    assert!(lines[0].starts_with("rcsr "), "{}", text);
    if cfg!(feature = "cuda") {
        assert_eq!(lines.len(), 2, "{}", text);
        assert!(lines[1].starts_with("cuda: toolkit "), "{}", text);
        assert!(lines[1].contains(", runtime ") && lines[1].contains(", arch sm_"), "{}", text);
    } else {
        assert_eq!(lines.len(), 1, "{}", text);
    }
}

#[test]
fn manifest_binary_records_the_cuda_build() {
    let manifest_path = TempPath::new("cuda-build.json");
    let json_path = TempPath::new("cuda-build-result.json");
    let data_path = TempPath::new("cuda-build-data.csv");
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv"),
        &data_path.0,
    )
    .expect("copy data fixture");
    let run = run_with_manifest(data_path.as_str(), manifest_path.as_str(), json_path.as_str());
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let text = std::fs::read_to_string(&manifest_path.0).expect("manifest");
    let binary = manifest::object(&text, "binary").expect("binary object");
    if cfg!(feature = "cuda") {
        let build = manifest::object(binary, "cuda_build").expect("cuda_build object");
        for key in ["toolkit", "nvcc", "source", "arch", "runtime"] {
            let value = manifest::string_field(build, key).unwrap_or_default();
            assert!(!value.is_empty(), "cuda_build.{} in {}", key, build);
        }
    } else {
        assert!(binary.contains("\"cuda_build\":null"), "{}", binary);
    }
}
