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
        "tree_nodes rows walk_us soa_scalar_us soa_avx_us tile_us walk_mrows soa_scalar_mrows soa_avx_mrows tile_mrows tile_over_avx bitwise_equal",
        "tree rows consts pass_mrows fit_iters fit_s fits_per_s hash",
        "metric rows fit_iters fit_s hash",
        "scorer rows score_ms",
        "rows threads fit_jobs fits_per_s fit_speedup eval_jobs eval_mrows eval_speedup",
    ];
    for h in headers {
        let wanted: Vec<&str> = h.split(' ').collect();
        assert!(
            text.lines().any(|l| {
                let cols: Vec<&str> = l.split('\t').collect();
                wanted.iter().all(|w| cols.contains(w))
            }),
            "no header with columns {wanted:?} in\n{text}"
        );
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
