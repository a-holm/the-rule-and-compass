//! Long-form benchmark for issue #124: Rust walk-forward vs the pinned
//! row-wise Python reference on the 3,943,856 x 56 hourly panel projection.
//!
//! The test is `#[ignore]`d because it deliberately runs the pilot Python
//! reference twice (tens of seconds to minutes each, reading a 4.23 GB CSV).
//! It is the executable form of `docs/eval-benchmark.md` and fails when the
//! q70 threshold disagrees (1e-12) or the median wall ratio is below 10x.
//!
//! Run it through the shared suite lock, on the host that owns the pilot data:
//!
//! ```sh
//! /mnt/lager/tmp/opencode/one-full-suite.sh bash -c \
//!   'cd worktrees/issue-124-benchmark/my-version && \
//!    cargo test --release --test walkforward_bench -- --ignored --nocapture'
//! ```
//!
//! The corpus lives under the git-ignored `arbeidsmappe/crypto-rcsr/pilot/`
//! directory. If it is absent the test prints a skip instead of failing;
//! the production `cargo test --release` never runs it.

use std::path::{Path, PathBuf};
use std::process::Command;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("my-version has a parent")
        .to_path_buf()
}

#[test]
#[ignore = "long benchmark: reads a 4.23 GB CSV twice; run explicitly via the suite lock"]
fn eval_10x_against_pinned_python_reference() {
    let root = repo_root();
    let pilot = root.join("arbeidsmappe/crypto-rcsr/pilot");
    let csv = PathBuf::from(
        std::env::var("RCSR_BENCH_CSV")
            .unwrap_or_else(|_| pilot.join("panel_hourly_compact.csv").display().to_string()),
    );
    let store = PathBuf::from(std::env::var("RCSR_BENCH_STORE").unwrap_or_else(|_| {
        pilot.join("panel_hourly_v2.rcsb").display().to_string()
    }));
    let rcsr = PathBuf::from(std::env::var("RCSR_BENCH_RCSR").unwrap_or_else(|_| {
        root.join("my-version/target/release/rcsr").display().to_string()
    }));
    let driver = root.join("benchmarks/eval_10x.py");
    let python = std::env::var("RCSR_BENCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    for (label, path) in [
        ("CSV projection", &csv),
        (".rcsb store", &store),
        ("release rcsr binary", &rcsr),
        ("benchmark driver", &driver),
    ] {
        if !path.exists() {
            eprintln!("skip #124 10x benchmark: missing {} at {}", label, path.display());
            return;
        }
    }

    let out_dir = root.join("my-version/target/eval-10x");
    std::fs::create_dir_all(&out_dir).expect("create benchmark output directory");
    let json = out_dir.join("issue-124-eval-10x.json");
    let markdown = out_dir.join("issue-124-eval-10x.md");

    let status = Command::new(&python)
        .arg(&driver)
        .arg("--check")
        .arg("--repeats")
        .arg("2")
        .arg("--json")
        .arg(&json)
        .arg("--markdown")
        .arg(&markdown)
        .arg("--csv")
        .arg(&csv)
        .arg("--store")
        .arg(&store)
        .arg("--rcsr")
        .arg(&rcsr)
        .status()
        .expect("run benchmarks/eval_10x.py");
    assert!(
        status.success(),
        "the 10x gate failed or a run errored; see {}",
        json.display()
    );

    let report = std::fs::read_to_string(&json).expect("read benchmark report");
    for needle in [
        "\"passes_10x\": true",
        "\"threshold_equal\": true",
        "\"passed\": true",
    ] {
        assert!(report.contains(needle), "report missing {}", needle);
    }
    eprintln!("#124 10x benchmark report: {}", json.display());
    eprintln!("#124 10x benchmark markdown: {}", markdown.display());
}
