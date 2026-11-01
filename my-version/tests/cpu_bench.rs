//! Smoke tests for the `cpu_bench` binary.

use std::process::Command;

fn bench() -> Command {
    Command::new(env!("CARGO_BIN_EXE_cpu_bench"))
}

#[test]
fn every_mode_runs_on_a_tiny_fixture() {
    let out = bench()
        .args([
            "--mode",
            "all",
            "--rows",
            "500",
            "--threads",
            "1,2",
            "--min-seconds",
            "0",
            "--tsv",
        ])
        .output()
        .expect("run cpu_bench");
    assert!(out.status.success(), "exit {:?}", out.status);
    let text = String::from_utf8_lossy(&out.stdout);
    let headers = [
        "tree_nodes\trows\twalk_us\tsoa_scalar_us\tsoa_avx_us\twalk_mrows\tsoa_scalar_mrows\tsoa_avx_mrows\tbitwise_equal",
        "tree\trows\tconsts\tpass_mrows\tfit_iters\tfit_s\tfits_per_s\thash",
        "metric\trows\tfit_iters\tfit_s\thash",
        "scorer\trows\tscore_ms",
        "rows\tthreads\tfit_jobs\tfits_per_s\tfit_speedup\teval_jobs\teval_mrows\teval_speedup",
    ];
    for h in headers {
        assert!(text.lines().any(|l| l == h), "missing header {h:?} in\n{text}");
    }
    for name in ["arith15", "trans13", "squared", "absolute", "median", "interquartile", "rank_correlation"] {
        assert!(text.lines().any(|l| l.starts_with(name)), "no row for {name}");
    }
}

#[test]
fn bad_arguments_exit_two() {
    for args in [
        vec!["--rows", "0"],
        vec!["--rows", ""],
        vec!["--threads", "0"],
        vec!["--mode", "nope"],
        vec!["--bogus"],
    ] {
        let out = bench().args(&args).output().expect("run cpu_bench");
        assert_eq!(out.status.code(), Some(2), "args {args:?}");
    }
}

#[test]
fn fit_hashes_repeat_between_runs() {
    let run = || {
        let out = bench()
            .args(["--mode", "fit", "--rows", "2000", "--min-seconds", "0", "--tsv"])
            .output()
            .expect("run cpu_bench");
        assert!(out.status.success());
        String::from_utf8_lossy(&out.stdout)
            .lines()
            .filter(|l| l.starts_with("arith15") || l.starts_with("trans13"))
            .map(|l| l.rsplit('\t').next().unwrap_or("").to_string())
            .collect::<Vec<_>>()
    };
    let (a, b) = (run(), run());
    assert_eq!(a.len(), 2);
    assert_eq!(a, b);
}
