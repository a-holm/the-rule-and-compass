//! #821: a CPU command opens no `/dev/nvidia*` file. The count comes from
//! `strace -f -e trace=openat`; a missing or unusable strace fails the test.

use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const T0: i64 = 1_704_067_200;
const H: i64 = 3600;

/// Successful opens of paths that start with `prefix` by `program args` and its children.
fn strace_opens(path_env: Option<&str>, trace: &Path, prefix: &str, program: &str, args: &[&str]) -> Result<usize, String> {
    let mut cmd = Command::new("strace");
    if let Some(p) = path_env {
        cmd.env("PATH", p);
    }
    let out = cmd
        .args(["-f", "-qq", "-e", "trace=openat", "-o"])
        .arg(trace)
        .arg(program)
        .args(args)
        .env_remove("RCSR_GPU")
        .output()
        .map_err(|e| format!("cannot run strace: {e}"))?;
    let text = std::fs::read_to_string(trace).map_err(|e| format!("strace wrote no trace ({e}); stderr: {}", String::from_utf8_lossy(&out.stderr)))?;
    if !text.contains("openat(") {
        return Err(format!("strace trace has no openat line; stderr: {}", String::from_utf8_lossy(&out.stderr)));
    }
    let needle = format!("\"{prefix}");
    Ok(text.lines().filter(|l| l.contains(&needle) && !l.contains("= -1")).count())
}

fn rcsr_opens(trace: &Path, args: &[&str]) -> usize {
    strace_opens(None, trace, "/dev/nvidia", env!("CARGO_BIN_EXE_rcsr"), args).unwrap_or_else(|e| panic!("{e}"))
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn run_ok(args: &[&str]) {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).env("RCSR_GPU", "0").output().unwrap();
    assert!(out.status.success(), "rcsr {args:?}: {}", String::from_utf8_lossy(&out.stderr));
}

fn fixtures(dir: &Path) -> PathBuf {
    let mut line = String::from("x,y\n");
    for i in 0..60 {
        let x = i as f64 * 0.1;
        line.push_str(&format!("{x},{}\n", 2.0 * x + 1.0));
    }
    std::fs::write(dir.join("d.csv"), line).unwrap();
    let mut panel = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            panel.push_str(&format!("{},s{},{},{},{}\n", T0 + i * H, g, f1, f2, y));
        }
    }
    std::fs::write(dir.join("p.csv"), panel).unwrap();
    std::fs::write(
        dir.join("w.csv"),
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nrolling,w0,{},{},{},{},{}\n",
            T0,
            T0 + 398 * H,
            T0 + 250 * H,
            T0 + 400 * H,
            T0 + 500 * H
        ),
    )
    .unwrap();
    let mut ohlcv = String::from("ts,symbol,open,high,low,close,volume\n");
    for g in 0..2i64 {
        let mut c = 100.0f64;
        for i in 0..400i64 {
            c *= 1.0 + (((i * 7 + g * 3) % 11) as f64 - 5.0) / 500.0;
            ohlcv.push_str(&format!("{},s{},{},{},{},{},{}\n", T0 + i * H, g, c, c * 1.01, c * 0.99, c, 1000 + i));
        }
    }
    std::fs::write(dir.join("o.csv"), ohlcv).unwrap();
    dir.join("trace.txt")
}

#[test]
fn the_counter_sees_opens_and_a_missing_strace_fails() {
    let scratch = Scratch::new("cpu-nodev-control");
    let trace = scratch.path().join("trace.txt");
    let n = strace_opens(None, &trace, "/dev/null", "sh", &["-c", "true < /dev/null"]).unwrap();
    assert!(n >= 1, "the control open of /dev/null was not counted");
    let none = scratch.path().join("empty-path");
    std::fs::create_dir_all(&none).unwrap();
    let err = strace_opens(Some(none.to_str().unwrap()), &scratch.path().join("t2.txt"), "/dev/nvidia", "sh", &["-c", "true"]).unwrap_err();
    assert!(err.contains("cannot run strace"), "{err}");
}

#[test]
fn explicit_cpu_commands_open_no_device() {
    let scratch = Scratch::new("cpu-nodev");
    let dir = scratch.path();
    let trace = fixtures(dir);
    let (d, p, w, o) = (s(&dir.join("d.csv")), s(&dir.join("p.csv")), s(&dir.join("w.csv")), s(&dir.join("o.csv")));
    let store = s(&dir.join("o.rcsb"));
    run_ok(&["import", &o, &store, "--time", "ts", "--quiet"]);

    let search = ["--data", d.as_str(), "--target", "y", "--generations", "1", "--pop", "20", "--quiet"];
    for flag in ["--cpu", "--no-gpu"] {
        let mut a = search.to_vec();
        a.push(flag);
        assert_eq!(rcsr_opens(&trace, &a), 0, "search {flag} opened /dev/nvidia*");
    }
    let out = s(&dir.join("e.rcsb"));
    let enrich = ["enrich", "--data", store.as_str(), "--out", out.as_str(), "--features", "returns", "--label", "long:4,name=h4,fee=5,stride=4"];
    let mut a = enrich.to_vec();
    a.push("--cpu");
    assert_eq!(rcsr_opens(&trace, &a), 0, "enrich --cpu opened /dev/nvidia*");
    let cal = s(&dir.join("cal.json"));
    assert_eq!(rcsr_opens(&trace, &["calibrate", "--cpu", "--out", cal.as_str()]), 0, "calibrate --cpu opened /dev/nvidia*");

    let window = format!("dev,{},{},{},{}", T0, T0 + 150 * H, T0 + 160 * H, T0 + 250 * H);
    let wf = [
        "walkforward", "--data", p.as_str(), "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym", "--horizon", "5",
        "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", window.as_str(),
    ];
    assert_eq!(rcsr_opens(&trace, &wf), 0, "walkforward without device flags opened /dev/nvidia*");
    let lout = s(&dir.join("l.json"));
    let ladder = [
        "ladder", "--data", p.as_str(), "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2", "--windows", w.as_str(),
        "--scheme", "rolling", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50", "--max-fit-rows", "600", "--out", lout.as_str(),
    ];
    assert_eq!(rcsr_opens(&trace, &ladder), 0, "ladder opened /dev/nvidia*");
}
