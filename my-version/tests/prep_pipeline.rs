//! #123: `rcsr prep` reads raw klines/CSV and writes a column store directly.
//! This covers the first increment: layouts, unit normalisation, validation,
//! numeric de-duplication, deterministic ordering and OHLCV resampling.

use rcsr::prep::{self, Layout};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn tmp(name: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!(
        "rcsr_prep_{}_{}_{}",
        std::process::id(),
        name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    p
}

/// 60 one-minute Binance bars, headerless 12 columns, times in milliseconds.
fn binance_csv() -> String {
    let mut s = String::new();
    for i in 0..60u64 {
        let t = 1_699_999_200_000u64 + i * 60_000;
        let o = 100.0 + i as f64;
        s.push_str(&format!(
            "{t},{o},{},{},{},2.5,{},{},7,1.0,2.0,0\n",
            o + 2.0,
            o - 1.0,
            o + 1.0,
            t + 59_999,
            5.0
        ));
    }
    s
}

#[test]
fn binance_klines_resample_to_15m_with_exact_ohlcv() {
    let path = tmp("binance.csv");
    fs::write(&path, binance_csv()).unwrap();
    let (t, audit) = prep::build(path.to_str().unwrap(), Layout::Auto, "15m", 65536).unwrap();
    assert_eq!(audit.layout, "binance");
    assert_eq!(audit.unit, "ms");
    assert_eq!(audit.unit_converted, 60);
    assert_eq!(audit.rows_in, 60);
    assert_eq!(audit.rows_out, 4);
    assert_eq!(audit.rows_dropped_duplicate, 0);

    let time = t.index_of("time").unwrap();
    let open = t.index_of("open").unwrap();
    let high = t.index_of("high").unwrap();
    let low = t.index_of("low").unwrap();
    let close = t.index_of("close").unwrap();
    let volume = t.index_of("volume").unwrap();
    // First bucket: bars 0..14 -> open 100, high 116, low 99, close 115.
    assert_eq!(t.value(0, time), 1_699_999_200.0);
    assert_eq!(t.value(0, open), 100.0);
    assert_eq!(t.value(0, high), 116.0);
    assert_eq!(t.value(0, low), 99.0);
    assert_eq!(t.value(0, close), 115.0);
    assert_eq!(t.value(0, volume), 15.0 * 2.5);
    // Last bucket starts one hour after the first.
    assert_eq!(t.value(3, time), 1_699_999_200.0 + 3.0 * 900.0);
    assert_eq!(t.value(3, open), 145.0);
    assert_eq!(t.value(3, close), 160.0);
    let _ = fs::remove_file(&path);
}

#[test]
fn aliased_header_layout_sorts_and_drops_numeric_duplicate_instants() {
    let csv = "Timestamp,Open,High,Low,Close,Volume\n\
               1700000120,11,12,10,11.5,1\n\
               1700000000,10,11,9,10.5,1\n\
               1700000000,10,11,9,10.5,1\n\
               1700000060,10.5,12,10,11,1\n";
    let path = tmp("aliases.csv");
    fs::write(&path, csv).unwrap();
    let (t, audit) = prep::build(path.to_str().unwrap(), Layout::Auto, "none", 1024).unwrap();
    assert_eq!(audit.layout, "ohlcv");
    assert_eq!(audit.rows_in, 4);
    assert_eq!(audit.rows_dropped_duplicate, 1);
    assert_eq!(audit.non_monotonic, 1, "the second row goes back in time");
    assert_eq!(audit.rows_out, 3);
    let time = t.index_of("time").unwrap();
    assert_eq!(
        t.col(time).to_vec(),
        vec![1_700_000_000.0, 1_700_000_060.0, 1_700_000_120.0]
    );
    let _ = fs::remove_file(&path);
}

#[test]
fn invalid_ohlcv_is_refused_with_the_row() {
    let csv = "time,open,high,low,close,volume\n\
               1700000000,10,9,11,10,1\n"; // high below low
    let path = tmp("bad.csv");
    fs::write(&path, csv).unwrap();
    let err = prep::build(path.to_str().unwrap(), Layout::Ohlcv, "none", 1024).unwrap_err();
    assert!(err.contains("row 1") && err.contains("high") && err.contains("low"), "{err}");

    fs::write(&path, "time,open,close,volume\n1700000000,1,2,3\n").unwrap();
    let err = prep::build(path.to_str().unwrap(), Layout::Ohlcv, "none", 1024).unwrap_err();
    assert!(err.contains("missing required column"), "{err}");
    let _ = fs::remove_file(&path);
}

#[test]
fn the_same_input_and_spec_produce_byte_identical_stores() {
    let path = tmp("determinism.csv");
    fs::write(&path, binance_csv()).unwrap();
    let a = tmp("det_a.rcsb");
    let b = tmp("det_b.rcsb");
    prep::run(
        path.to_str().unwrap(),
        Layout::Auto,
        "1h",
        4096,
        Some(a.to_str().unwrap()),
        None,
        false,
    )
    .unwrap();
    prep::run(
        path.to_str().unwrap(),
        Layout::Auto,
        "1h",
        4096,
        Some(b.to_str().unwrap()),
        None,
        false,
    )
    .unwrap();
    assert_eq!(fs::read(&a).unwrap(), fs::read(&b).unwrap());
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&a);
    let _ = fs::remove_file(&b);
}

#[test]
fn a_manifest_records_the_spec_and_the_audit() {
    let path = tmp("manifest.csv");
    fs::write(&path, binance_csv()).unwrap();
    let out = tmp("m.rcsb");
    let man = tmp("m.json");
    prep::run(
        path.to_str().unwrap(),
        Layout::Auto,
        "1h",
        4096,
        Some(out.to_str().unwrap()),
        Some(man.to_str().unwrap()),
        false,
    )
    .unwrap();
    let text = fs::read_to_string(&man).unwrap();
    assert!(text.contains("\"spec_hash\""), "{text}");
    assert!(text.contains("\"resample\":\"1h\""), "{text}");
    assert!(text.contains("\"rows_out\":1"), "{text}");
    assert!(text.contains("\"unit\":\"ms\""), "{text}");
    let _ = fs::remove_file(&path);
    let _ = fs::remove_file(&out);
    let _ = fs::remove_file(&man);
}

#[test]
fn dry_run_writes_nothing_and_the_cli_reports_the_audit() {
    let path = tmp("dry.csv");
    fs::write(&path, binance_csv()).unwrap();
    let out = tmp("dry.rcsb");
    let res = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "prep",
            path.to_str().unwrap(),
            "--resample",
            "1h",
            "--out",
            out.to_str().unwrap(),
            "--dry-run",
        ])
        .output()
        .unwrap();
    assert!(res.status.success(), "{}", String::from_utf8_lossy(&res.stderr));
    let stdout = String::from_utf8_lossy(&res.stdout);
    assert!(stdout.contains("rows_in=60") && stdout.contains("rows_out=1"), "{stdout}");
    assert!(!out.exists(), "dry-run must not write the store");
    let _ = fs::remove_file(&path);
}
