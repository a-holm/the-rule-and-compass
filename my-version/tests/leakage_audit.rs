//! `rcsr audit` is fail-closed and returns exit code 5 (#129).

use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn write_panel(leak: bool) -> (std::path::PathBuf, String) {
    let panel = rcsr::leakage::synthetic_panel(240, 77, leak);
    let dir = std::env::temp_dir();
    let path = dir.join(format!(
        "rcsr-audit-{}-{}-{}.csv",
        std::process::id(),
        leak,
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    let mut text = String::from("row,time,x,noise,y");
    if leak {
        text.push_str(",future_y");
    }
    text.push('\n');
    for i in 0..panel.len() {
        text.push_str(&format!(
            "{},{},{:?},{:?},{:?}",
            i,
            panel.time[i] * 60.0,
            panel.column("x").unwrap()[i],
            panel.column("noise").unwrap()[i],
            panel.column("y").unwrap()[i]
        ));
        if leak {
            text.push_str(&format!(",{:?}", panel.column("future_y").unwrap()[i]));
        }
        text.push('\n');
    }
    let path_s = path.to_string_lossy().into_owned();
    std::fs::write(&path, text).expect("write fixture");
    (path, path_s)
}

fn audit(path: &str) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "audit", "--data", path, "--target", "y", "--time", "time", "--horizon", "1",
            "--json",
        ])
        .output()
        .expect("run rcsr audit")
}

#[test]
fn a_clean_panel_exits_zero_with_a_machine_readable_report() {
    let (path, p) = write_panel(false);
    let out = audit(&p);
    let _ = std::fs::remove_file(path);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"clean\":true"), "{text}");
    assert!(text.contains("\"effective_n\":240"), "{text}");
    assert!(text.contains("\"spec_hash\""), "{text}");
    assert!(text.contains("\"tzdb_version\":\"2025a\""), "{text}");
}

#[test]
fn a_leaky_panel_exits_five_and_names_the_column() {
    let (path, p) = write_panel(true);
    let out = audit(&p);
    let _ = std::fs::remove_file(path);
    assert_eq!(out.status.code(), Some(5), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"clean\":false"), "{text}");
    assert!(text.contains("future_y"), "{text}");
    assert!(text.contains("\"exit_code\":5"), "{text}");
}

#[test]
fn a_wall_time_gap_is_converted_and_never_silently_dropped() {
    let (path, p) = write_panel(false);
    // 240 one-minute rows: 90m is 90 rows, and the split report says so.
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", &p, "--target", "y", "--vars", "x", "--time", "time",
            "--split", "temporal", "--gap", "90m", "--quiet", "--json",
            "--generations", "2", "--pop", "10",
        ])
        .output()
        .expect("run rcsr");
    let _ = std::fs::remove_file(path);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("\"gap\":90"), "{text}");
}

#[test]
fn a_wall_time_gap_is_dropped_with_a_warning_under_explicit_row_selection() {
    let (path, p) = write_panel(false);
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", &p, "--target", "y", "--vars", "x", "--time", "time",
            "--split", "temporal", "--gap", "90m", "--train-select", "row < 120",
            "--test-select", "row >= 160", "--json",
            "--generations", "2", "--pop", "10",
        ])
        .output()
        .expect("run rcsr");
    let _ = std::fs::remove_file(path);
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--gap is dropped"), "{err}");
    assert!(out.status.success(), "{err}");
}
