//! #486: the protocol locked test declared with the data is refused by the
//! search split, walk-forward and the ladder, and read only through the sealed
//! path. Fixtures are synthetic; the locked range is a stand-in in 2024.

use rcsr::lockbox::{self, Lockbox};
use rcsr::protocol::{Candidate, Ledger};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 2024-01-01T00:00:00Z.
const T0: i64 = 1_704_067_200;
const H: i64 = 3600;
const DEFAULT_PURGE: i64 = 219_600;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rcsr-locked-window-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn locked() -> Lockbox {
    Lockbox { start: T0 + 300 * H, end: T0 + 400 * H }
}

fn series_csv(path: &Path) {
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", T0 + i * H, x, y));
    }
    std::fs::write(path, csv).unwrap();
}

fn search(data: &str, run_json: &str, extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "--data", data, "--target", "y", "--vars", "x", "--time", "ts", "--split", "temporal", "--cpu",
        "--no-dev-log", "--seed", "3", "--generations", "5", "--pop", "20", "--metric", "squared", "--quiet",
        "--json-out", run_json,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

#[test]
fn the_search_refuses_the_declared_locked_test_unless_the_rows_are_excluded() {
    let dir = scratch("search");
    let data = dir.join("d.csv");
    series_csv(&data);
    let run = s(&dir.join("run.json"));
    let d = s(&data);

    // without a declaration the temporal test block is read, as before
    let (code, _, err) = search(&d, &run, &[]);
    assert_eq!(code, 0, "{}", err);

    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    assert_eq!(lockbox::declared_for(&d), Some(locked()));
    let (code, _, err) = search(&d, &run, &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("split has"), "{}", err);
    assert!(err.contains("declared locked test 2024-01-13..2024-01-17 (store manifest"), "{}", err);
    assert!(err.contains(&format!("--exclude 'ts >= {}'", locked().start - DEFAULT_PURGE)), "{}", err);

    let at_start = format!("ts >= {}", locked().start);
    let (code, _, err) = search(&d, &run, &["--exclude", &at_start]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("whose label window reaches the declared locked test"), "{}", err);

    let exclude = format!("ts >= {}", locked().start - DEFAULT_PURGE);
    assert!(err.contains(&format!("--exclude '{}'", exclude)), "{}", err);
    let (code, _, err) = search(&d, &run, &["--exclude", &exclude]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"rows_in_scope\":0"), "{}", json);
    assert!(json.contains("\"evaluated\":false"), "{}", json);
    assert!(json.contains("\"source\":\"store\""), "{}", json);
    assert!(json.contains("\"boundary_purge_source\":\"protocol_default\""), "{}", json);

    // --allow-lockbox opens the 2026H2 box only
    let (code, _, err) = search(&d, &run, &["--allow-lockbox"]);
    assert_eq!(code, 3, "{}", err);
}

#[test]
fn an_explicit_lockbox_flag_declares_the_range_for_a_csv_without_a_sidecar() {
    let dir = scratch("flag");
    let data = dir.join("d.csv");
    series_csv(&data);
    let run = s(&dir.join("run.json"));
    let range = format!("{},{}", locked().start, locked().end);
    let (code, _, err) = search(&s(&data), &run, &["--lockbox", &range]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("(--lockbox;"), "{}", err);
}

fn wf(data: &str, ledger: Option<&str>, extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "walkforward", "--data", data, "--target", "y", "--expr", "x", "--time", "ts", "--horizon", "5",
        "--cost-bps", "5", "--periods-per-year", "365", "--json",
    ];
    if let Some(l) = ledger {
        a.extend_from_slice(&["--ledger", l]);
    }
    a.extend_from_slice(extra);
    rcsr(&a)
}

fn window(label: &str, a: i64, b: i64, c: i64, d: i64) -> String {
    format!("{},{},{},{},{}", label, T0 + a * H, T0 + b * H, T0 + c * H, T0 + d * H)
}

#[test]
fn walkforward_refuses_the_declared_range_without_a_ledger_and_still_reads_it_once() {
    let dir = scratch("wf");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();

    let bad = window("bad", 0, 200, 210, 350);
    let (code, _, err) = wf(&d, None, &["--window", &bad]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("reaches the declared locked test"), "{}", err);

    let dev = window("dev", 0, 150, 160, 250);
    let (code, out, err) = wf(&d, None, &["--window", &dev]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"source\":\"store\""), "{}", out);
    assert!(out.contains("\"evaluated\":false"), "{}", out);

    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let (code, _, err) = wf(&d, Some(&l), &["--window", &dev]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);

    let read = window("locked", 0, 250, 300, 400);
    let args = ["--locked-test", "--sealed", &sealed, "--window", &read];
    let (code, out, err) = wf(&d, Some(&l), &args);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"test_read\":true"), "{}", out);
    assert!(out.contains("\"evaluated\":true"), "{}", out);
    let (code, _, err) = wf(&d, Some(&l), &args);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("already has test_read:true"), "{}", err);
}

fn panel_csv(path: &Path) {
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            csv.push_str(&format!("{},s{},{},{},{}\n", T0 + i * H, g, f1, f2, y));
        }
    }
    std::fs::write(path, csv).unwrap();
}

fn windows_file(path: &Path, scheme: &str) {
    let row = format!(
        "{scheme},{scheme}_0,{},{},{},{},{}\n",
        T0,
        T0 + 398 * H,
        T0 + 250 * H,
        T0 + 400 * H,
        T0 + 500 * H
    );
    std::fs::write(
        path,
        format!("scheme,label,train_start,train_end,validation_start,test_start,test_end\n{row}"),
    )
    .unwrap();
}

fn ladder(data: &str, windows: &str, out: &str, extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "ladder", "--data", data, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
        "--windows", windows, "--scheme", "locked", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws",
        "50", "--max-fit-rows", "600", "--out", out,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

#[test]
fn the_ladder_refuses_the_locked_scheme_and_reads_it_once_through_the_ledger() {
    let dir = scratch("ladder");
    let (data, win) = (dir.join("p.csv"), dir.join("w.csv"));
    panel_csv(&data);
    windows_file(&win, "locked");
    let (d, w, out) = (s(&data), s(&win), s(&dir.join("ladder.json")));
    let range = Lockbox { start: T0 + 400 * H, end: T0 + 500 * H };
    lockbox::write_sidecar(&d, &range, "test").unwrap();

    let (code, _, err) = ladder(&d, &w, &out, &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("declared locked test"), "{}", err);

    let (code, _, err) = ladder(&d, &w, &out, &["--locked-test"]);
    assert_eq!(code, 2, "{}", err);

    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    Ledger::new(dir.join("ledger.jsonl"))
        .add_candidate(&Candidate {
            id: "c1".into(),
            spec_sha256: "a".repeat(64),
            data_sha256: "b".repeat(64),
            build_commit: "build".into(),
            report_sha256: "c".repeat(64),
            expression: "x".into(),
        })
        .unwrap();
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);

    let (code, _, err) = ladder(&d, &w, &out, &["--locked-test", "--ledger", &l, "--sealed", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&out).unwrap();
    assert!(json.contains("\"evaluated\":true"), "{}", json);
    assert!(json.contains("\"rows_in_scope\":"), "{}", json);
    assert!(std::fs::read_to_string(&l).unwrap().contains("\"event\":\"baseline_read\""));
    let (code, vout, err) = rcsr(&["protocol", "verify", "--ledger", &l]);
    assert_eq!(code, 0, "{}{}", vout, err);

    let (code, _, err) = ladder(&d, &w, &out, &["--locked-test", "--ledger", &l, "--sealed", &sealed]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("baseline_read"), "{}", err);
}

#[test]
fn declared_for_reads_each_carrier_and_a_flag_must_cover_the_store_range() {
    let dir = scratch("declared");
    let p = s(&dir.join("a.rcsb"));
    assert_eq!(lockbox::declared_for(&p), None);
    let lb = locked();
    std::fs::write(format!("{p}.manifest.json"), format!("{{\"locked_test\":{}}}", lb.declaration_json("t"))).unwrap();
    assert_eq!(lockbox::declared_for(&p), Some(lb));
    let narrow = format!("{},{}", T0, T0 + H);
    let e = lockbox::resolve(Some(&narrow), &p).unwrap_err();
    assert_eq!(e.exit, 2, "{}", e.message);
    let wide = format!("{},{}", lb.start - H, lb.end + H);
    let d = lockbox::resolve(Some(&wide), &p).unwrap().declared().unwrap();
    assert_eq!((d.source, d.lockbox.start), ("flag", lb.start - H));
    let d = lockbox::resolve(None, &p).unwrap().declared().unwrap();
    assert_eq!(d.source, "store");
    let d = lockbox::resolve(None, "-").unwrap().declared().unwrap();
    assert_eq!(d.source, "protocol_default");
    let d = lockbox::resolve(None, &p).unwrap().declared().unwrap();
    let times = [lb.start - 1, lb.start, lb.end - 1, lb.end];
    let e = d.check_times("test split", "ts", &times, 0).unwrap_err();
    assert!(e.contains("2 row(s)"), "{}", e);
    assert!(d.check_times("test split", "ts", &times[..1], 0).is_ok());
}

/// Hourly panel of 3 symbols, 500 bars. Bars from index 400 on (the locked
/// range) are different when `flip` is set; every earlier bar is identical.
/// The label at bar i is the return over bars i+1 and i+2; features read past
/// bars only.
fn diff_panel(path: &Path, flip: bool) {
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        let ret: Vec<f64> = (0..502i64)
            .map(|i| {
                let r = (((i * 29 + g * 7) % 11) as f64 - 5.0) * 0.0006 + (((i * 13 + g) % 5) as f64 - 2.0) * 0.0002;
                if flip && i >= 400 {
                    -r + 0.004
                } else {
                    r
                }
            })
            .collect();
        for i in 0..500usize {
            let f1 = if i == 0 { 0.0 } else { ret[i - 1] * 100.0 };
            let f2 = (((i as i64 * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = ret[i + 1] + ret[i + 2];
            csv.push_str(&format!("{},s{},{},{},{}\n", T0 + i as i64 * H, g, f1, f2, y));
        }
    }
    std::fs::write(path, csv).unwrap();
}

fn diff_store(dir: &Path, name: &str, flip: bool) -> String {
    let p = dir.join(format!("{name}.csv"));
    diff_panel(&p, flip);
    let d = s(&p);
    lockbox::write_sidecar(&d, &Lockbox { start: T0 + 400 * H, end: T0 + 500 * H }, "test").unwrap();
    d
}

fn numbers(json: &str, key: &str) -> Vec<String> {
    let needle = format!("\"{key}\":");
    json.match_indices(&needle)
        .map(|(i, _)| {
            json[i + needle.len()..]
                .chars()
                .take_while(|c| !matches!(c, ',' | '}' | ']'))
                .collect()
        })
        .collect()
}

fn dev_search(data: &str, run: &str, exclude: &str) -> (i32, String) {
    let (code, _, err) = rcsr(&[
        "--data", data, "--target", "y", "--vars", "f1,f2", "--time", "ts", "--group-by", "sym", "--split",
        "temporal", "--validation-frac", "0.2", "--fill", "drop", "--exclude", exclude, "--eval-expr", "f1",
        "--cpu", "--no-dev-log", "--quiet", "--json-out", run,
    ]);
    (code, err)
}

fn ladder_fold(path: &Path, test_end: i64) {
    std::fs::write(
        path,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\n\
             expanding,f1,{},{},{},{},{}\n",
            T0,
            T0 + 250 * H,
            T0 + 150 * H,
            T0 + 260 * H,
            test_end
        ),
    )
    .unwrap();
}

fn dev_ladder(data: &str, win: &str, out: &str, purge: &str) -> (i32, String) {
    let (code, _, err) = rcsr(&[
        "ladder", "--data", data, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
        "--windows", win, "--scheme", "expanding", "--purge-seconds", purge, "--gbm-trees", "10", "--draws",
        "50", "--max-fit-rows", "600", "--out", out,
    ]);
    (code, err)
}

fn without_timing(json: &str) -> Vec<String> {
    json.split(',')
        .filter(|t| !(t.contains("_s\"") || t.contains("seconds") || t.contains("wall") || t.contains("denominator")))
        .map(str::to_string)
        .collect()
}

#[test]
fn development_outputs_ignore_locked_prices() {
    let dir = scratch("differential");
    let a = diff_store(&dir, "a", false);
    let b = diff_store(&dir, "b", true);
    let s_start = T0 + 400 * H;

    // search: a boundary exclusion at the locked start keeps labels made of locked bars
    let at_start = format!("ts >= {s_start}");
    for d in [&a, &b] {
        let (code, err) = dev_search(d, &s(&dir.join("x.json")), &at_start);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains(&format!("--exclude 'ts >= {}'", s_start - DEFAULT_PURGE)), "{}", err);
    }
    let at_purge = format!("ts >= {}", s_start - DEFAULT_PURGE);
    let (run_a, run_b) = (s(&dir.join("ra.json")), s(&dir.join("rb.json")));
    assert_eq!(dev_search(&a, &run_a, &at_purge).0, 0);
    assert_eq!(dev_search(&b, &run_b, &at_purge).0, 0);
    let (ja, jb) = (std::fs::read_to_string(&run_a).unwrap(), std::fs::read_to_string(&run_b).unwrap());
    for key in ["train_error", "test_error"] {
        let (na, nb) = (numbers(&ja, key), numbers(&jb, key));
        assert!(!na.is_empty(), "{key}: {ja}");
        assert_eq!(na, nb, "{key}");
    }

    // walk-forward: horizon 7200 s; a fold ending at the locked start is refused
    let wf_run = |d: &str, test_end: i64| {
        let window = format!("f1,{},{},{},{}", T0, T0 + 150 * H, T0 + 160 * H, test_end);
        rcsr(&[
            "walkforward", "--data", d, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym",
            "--horizon", "7200", "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", &window,
        ])
    };
    for d in [&a, &b] {
        let (code, _, err) = wf_run(d, s_start);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("reaches the declared locked test"), "{}", err);
    }
    let (ca, oa, ea) = wf_run(&a, s_start - 7200);
    let (cb, ob, eb) = wf_run(&b, s_start - 7200);
    assert_eq!((ca, cb), (0, 0), "{}{}", ea, eb);
    let (ha, hb) = (numbers(&oa, "report_sha256"), numbers(&ob, "report_sha256"));
    assert!(!ha.is_empty(), "{}", oa);
    assert_eq!(ha, hb);

    // ladder: purge 18000 s
    let win = dir.join("w.csv");
    for d in [&a, &b] {
        ladder_fold(&win, s_start - 3600);
        let (code, err) = dev_ladder(d, &s(&win), &s(&dir.join("l.json")), "18000");
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("reaches the declared locked test"), "{}", err);
    }
    ladder_fold(&win, s_start - 18_000);
    let (la, lb) = (s(&dir.join("la.json")), s(&dir.join("lb.json")));
    assert_eq!(dev_ladder(&a, &s(&win), &la, "18000").0, 0);
    assert_eq!(dev_ladder(&b, &s(&win), &lb, "18000").0, 0);
    let (ja, jb) = (std::fs::read_to_string(&la).unwrap(), std::fs::read_to_string(&lb).unwrap());
    assert_eq!(without_timing(&ja), without_timing(&jb));
}

#[test]
fn walkforward_locked_scheme_trains_at_least_the_label_purge_before_the_start() {
    let dir = scratch("locked-train");
    let a = diff_store(&dir, "a", false);
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let dev = format!("dev,{},{},{},{}", T0, T0 + 150 * H, T0 + 160 * H, T0 + 250 * H);
    let run = |extra: &[&str]| {
        let mut args = vec![
            "walkforward", "--data", a.as_str(), "--target", "y", "--expr", "f1", "--time", "ts", "--group-by",
            "sym", "--horizon", "7200", "--cost-bps", "5", "--periods-per-year", "365", "--json", "--ledger", &l,
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    assert_eq!(run(&["--window", &dev]).0, 0);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let late = format!("late,{},{},{},{}", T0, T0 + 399 * H, T0 + 400 * H, T0 + 500 * H);
    let (code, _, err) = run(&["--locked-test", "--sealed", &sealed, "--window", &late]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("trains within the label purge of the locked test"), "{}", err);
}

/// 2025-06-20T00:00:00Z: 400 hourly bars end on 2025-07-07, inside the
/// protocol default locked test.
const T1: i64 = 1_750_377_600;
const DEFAULT_START: i64 = 1_751_328_000;

fn series_csv_at(path: &Path, t0: i64) {
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", t0 + i * H, x, y));
    }
    std::fs::write(path, csv).unwrap();
}

fn panel_csv_at(path: &Path, t0: i64) {
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            csv.push_str(&format!("{},s{},{},{},{}\n", t0 + i * H, g, f1, f2, y));
        }
    }
    std::fs::write(path, csv).unwrap();
}

#[test]
fn undeclared_data_gets_the_protocol_default_in_search_walkforward_and_ladder() {
    let dir = scratch("undeclared");
    let data = dir.join("d.csv");
    series_csv_at(&data, T1);
    let (d, run) = (s(&data), s(&dir.join("run.json")));

    let (code, _, err) = search(&d, &run, &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("declared locked test 2025-07-01..2026-07-01 (protocol default"), "{}", err);
    let (code, _, err) = search(&d, &run, &["--lockbox", "none"]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"source\":\"none\""), "{}", json);
    assert!(json.contains("\"declared_in\":[]"), "{}", json);

    let w = format!("w,{},{},{},{}", T1, T1 + 250 * H, T1 + 260 * H, T1 + 399 * H);
    let (code, _, err) = wf(&d, None, &["--window", &w]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("touches the declared locked test"), "{}", err);
    let (code, out, err) = wf(&d, None, &["--window", &w, "--lockbox", "none"]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"source\":\"none\""), "{}", out);
    let dev = format!("w,{},{},{},{}", T1, T1 + 100 * H, T1 + 110 * H, T1 + 250 * H);
    let (code, out, err) = wf(&d, None, &["--window", &dev]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"source\":\"protocol_default\""), "{}", out);
    assert!(out.contains("\"declared_in\":[]"), "{}", out);

    let (p, win, lad) = (dir.join("p.csv"), dir.join("w.csv"), s(&dir.join("ladder.json")));
    panel_csv_at(&p, T1);
    windows_file_at(&win, "locked", T1);
    let (p, win) = (s(&p), s(&win));
    let (code, _, err) = ladder(&p, &win, &lad, &[]);
    assert_eq!(code, 3, "{}", err);
    let (code, _, err) = ladder(&p, &win, &lad, &["--lockbox", "none"]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&lad).unwrap();
    assert!(json.contains("\"source\":\"none\""), "{}", json);
}

fn windows_file_at(path: &Path, scheme: &str, t0: i64) {
    let row = format!(
        "{scheme},{scheme}_0,{},{},{},{},{}\n",
        t0,
        t0 + 398 * H,
        t0 + 250 * H,
        t0 + 400 * H,
        t0 + 500 * H
    );
    std::fs::write(
        path,
        format!("scheme,label,train_start,train_end,validation_start,test_start,test_end\n{row}"),
    )
    .unwrap();
}

#[test]
fn a_flag_that_narrows_or_removes_the_store_declaration_is_refused() {
    let dir = scratch("narrow");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    let run = s(&dir.join("run.json"));
    let (code, _, err) = search(&d, &run, &["--lockbox", "2030-01-01,2030-02-01"]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("does not cover the declared locked test"), "{}", err);
    let (code, _, err) = search(&d, &run, &["--lockbox", "none"]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("--lockbox none is refused"), "{}", err);
    let wide = format!("{},{}", locked().start - H, locked().end + H);
    let (code, _, err) = search(&d, &run, &["--lockbox", &wide]);
    assert_eq!(code, 3, "{}", err);
}

#[test]
fn conflicting_declaration_files_are_refused_naming_both() {
    let dir = scratch("conflict");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    let a = locked();
    let b = Lockbox { start: a.start + 1000 * H, end: a.end + 1000 * H };
    std::fs::write(format!("{d}.enrich.json"), format!("{{\"locked_test\":{}}}", a.declaration_json("t"))).unwrap();
    lockbox::write_sidecar(&d, &b, "test").unwrap();
    let run = s(&dir.join("run.json"));
    let (code, _, err) = search(&d, &run, &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("conflicting locked-test declarations"), "{}", err);
    assert!(err.contains("d.csv.lockbox.json") && err.contains("d.csv.enrich.json"), "{}", err);
}

#[test]
fn audit_leaves_the_locked_rows_out_and_counts_them() {
    let dir = scratch("audit");
    let data = dir.join("d.csv");
    series_csv_at(&data, T1);
    let d = s(&data);
    let inside = (0..400i64).filter(|i| T1 + i * H >= DEFAULT_START).count();
    assert!(inside > 0 && inside < 400);
    let (_, out, err) = rcsr(&["audit", "--data", &d, "--target", "y", "--time", "ts", "--json"]);
    assert!(out.contains(&format!("\"rows_left_out\":{inside}")), "{}{}", out, err);
    assert!(out.contains("\"source\":\"protocol_default\""), "{}", out);
    let (_, out, err) = rcsr(&["audit", "--data", &d, "--target", "y", "--time", "ts", "--json", "--lockbox", "none"]);
    assert!(out.contains("\"rows_left_out\":0"), "{}{}", out, err);
    assert!(out.contains("\"source\":\"none\""), "{}", out);
}


/// Two series that are byte-identical outside the stand-in locked test and
/// differ inside it (flipped target).
fn ab_csv(a: &Path, b: &Path) {
    let mut ca = String::from("ts,x,y\n");
    let mut cb = ca.clone();
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        let ts = T0 + i * H;
        let yb = if (locked().start..locked().end).contains(&ts) { -y - 0.001 } else { y };
        ca.push_str(&format!("{},{},{}\n", ts, x, y));
        cb.push_str(&format!("{},{},{}\n", ts, x, yb));
    }
    std::fs::write(a, ca).unwrap();
    std::fs::write(b, cb).unwrap();
}

fn plain(args: &[&str]) -> (i32, String, String) {
    let mut a = vec!["--target", "y", "--vars", "x", "--cpu", "--no-dev-log", "--seed", "3", "--quiet"];
    a.extend_from_slice(args);
    rcsr(&a)
}

fn cv_part(json: &str) -> String {
    let a = json.find("\"cross_validation\":").expect("cross_validation in run.json");
    let rest = &json[a..];
    rest[..rest.find(",\"stability\"").unwrap()].to_string()
}

#[test]
fn declared_split_without_time_is_refused() {
    let dir = scratch("notime");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    let run = s(&dir.join("run.json"));

    let (code, _, err) = plain(&["--data", &d, "--eval-expr", "0"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("declared locked test"), "{}", err);
    assert!(err.contains("2024-01-13..2024-01-17"), "{}", err);

    // with --time the random split is checked too
    let (code, _, err) = plain(&["--data", &d, "--time", "ts", "--eval-expr", "0"]);
    assert_eq!(code, 3, "{}", err);

    let exclude = format!("ts >= {}", locked().start - DEFAULT_PURGE);
    let (code, _, err) = plain(&[
        "--data", &d, "--time", "ts", "--split", "temporal", "--exclude", &exclude, "--generations", "3",
        "--pop", "16", "--json-out", &run,
    ]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"checked\":true"), "{}", json);
    assert!(json.contains("\"rows_in_scope\":0"), "{}", json);
}

#[test]
fn declared_selectors_without_time_are_refused() {
    let dir = scratch("selectors");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    let train = format!("ts < {}", T0 + 200 * H);
    let test = format!("ts >= {}", locked().start + H);
    let (code, _, err) = plain(&["--data", &d, "--train-select", &train, "--test-select", &test, "--eval-expr", "0"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("2024-01-13..2024-01-17"), "{}", err);
}

#[test]
fn declared_non_epoch_time_is_refused() {
    let dir = scratch("nonepoch");
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,y\n");
    for i in 0..100i64 {
        csv.push_str(&format!("{},{},{}\n", i, ((i * 37) % 17) as f64, ((i * 13) % 7) as f64));
    }
    std::fs::write(&data, csv).unwrap();
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    let (code, _, err) = plain(&["--data", &d, "--time", "ts", "--split", "temporal", "--eval-expr", "0"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("not epoch seconds"), "{}", err);

    // without a declaration the check is skipped as before
    let free = dir.join("free.csv");
    std::fs::copy(&data, &free).unwrap();
    let (code, _, err) = plain(&["--data", &s(&free), "--time", "ts", "--split", "temporal", "--eval-expr", "0"]);
    assert_eq!(code, 0, "{}", err);
}

#[test]
fn declared_cv_ignores_allow_lockbox() {
    let dir = scratch("cvallow");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    // the split sides avoid the locked rows; the cross-validation folds do not
    let train = format!("ts < {}", T0 + 100 * H);
    let test = format!("(ts >= {}) * (ts < {})", T0 + 100 * H, T0 + 150 * H);
    let base = [
        "--data", d.as_str(), "--time", "ts", "--train-select", train.as_str(), "--test-select", test.as_str(),
        "--cv", "k-fold", "--cv-k", "3", "--generations", "2", "--pop", "8",
    ];
    for allow in [false, true] {
        let mut a = base.to_vec();
        if allow {
            a.push("--allow-lockbox");
        }
        let (code, _, err) = plain(&a);
        assert_eq!(code, 3, "allow={} {}", allow, err);
        assert!(err.contains("cross-validation rows"), "{}", err);
        assert!(err.contains("declared locked test"), "{}", err);
    }
    let exclude = format!("ts >= {}", locked().start);
    let mut a = base.to_vec();
    a.extend_from_slice(&["--exclude", &exclude]);
    let (code, _, err) = plain(&a);
    assert_eq!(code, 0, "{}", err);
}

#[test]
fn cv_folds_honour_exclude() {
    let dir = scratch("cvexclude");
    let (a, b) = (dir.join("a.csv"), dir.join("b.csv"));
    ab_csv(&a, &b);
    let exclude = format!("ts >= {}", locked().start);
    let mut cv = Vec::new();
    for p in [&a, &b] {
        let d = s(p);
        lockbox::write_sidecar(&d, &locked(), "test").unwrap();
        let run = s(&dir.join(format!("{}.json", p.file_stem().unwrap().to_str().unwrap())));
        let train = format!("ts < {}", T0 + 100 * H);
        let test = format!("(ts >= {}) * (ts < {})", T0 + 100 * H, T0 + 150 * H);
        let (code, _, err) = plain(&[
            "--data", &d, "--time", "ts", "--train-select", &train, "--test-select", &test, "--exclude", &exclude,
            "--cv", "k-fold", "--cv-k", "3", "--generations", "3", "--pop", "16", "--json-out", &run,
        ]);
        assert_eq!(code, 0, "{}", err);
        cv.push(cv_part(&std::fs::read_to_string(&run).unwrap()));
    }
    assert!(cv[0].contains("\"folds\":[{"), "{}", cv[0]);
    assert_eq!(cv[0], cv[1]);
}

#[test]
fn a_refused_cv_run_exits_before_the_search() {
    let started = std::time::Instant::now();
    let dir = scratch("cvpreflight");
    let data = dir.join("d.csv");
    series_csv(&data);
    let d = s(&data);
    lockbox::write_sidecar(&d, &locked(), "test").unwrap();
    let run = dir.join("run.json");
    let train = format!("ts < {}", T0 + 100 * H);
    let test = format!("(ts >= {}) * (ts < {})", T0 + 100 * H, T0 + 150 * H);
    let (code, _, err) = plain(&[
        "--data", &d, "--time", "ts", "--train-select", &train, "--test-select", &test, "--cv", "k-fold",
        "--cv-k", "3", "--generations", "1000000", "--pop", "2000", "--json-out", &s(&run),
    ]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("cross-validation rows"), "{}", err);
    assert!(!run.exists(), "run.json written: the search ran");
    assert!(started.elapsed().as_secs() < 30, "the search ran before the refusal");
}

#[test]
fn a_flag_on_undeclared_data_adds_to_the_protocol_default() {
    let dir = scratch("flag-adds");
    let (late, early) = (dir.join("late.csv"), dir.join("early.csv"));
    series_csv_at(&late, T1);
    series_csv(&early);
    let (late, early, run) = (s(&late), s(&early), s(&dir.join("run.json")));

    // a disjoint flag does not switch the default off
    let (code, _, err) = search(&late, &run, &["--lockbox", "2030-01-01,2030-02-01"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("2025-07-01..2026-07-01 (protocol default"), "{}", err);
    // a flag narrower than the default does not either
    let (code, _, err) = search(&late, &run, &["--lockbox", "2025-07-02,2025-07-03"]);
    assert_eq!(code, 3, "{}", err);
    let w = format!("w,{},{},{},{}", T1, T1 + 250 * H, T1 + 260 * H, T1 + 399 * H);
    let (code, _, err) = wf(&late, None, &["--window", &w, "--lockbox", "2030-01-01,2030-02-01"]);
    assert_eq!(code, 3, "{}", err);

    // the flag range is guarded as well
    let range = format!("{},{}", locked().start, locked().end);
    let (code, _, err) = search(&early, &run, &["--lockbox", &range]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("(--lockbox;"), "{}", err);

    // a stand-in window on data that never reaches 2025 still runs, and the report says both
    let (code, _, err) = search(&early, &run, &["--lockbox", "2030-01-01,2030-02-01"]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"source\":\"flag+protocol_default\""), "{}", json);
    assert!(json.contains("\"also\":{"), "{}", json);

    // only none removes the default
    let (code, _, err) = search(&late, &run, &["--lockbox", "none"]);
    assert_eq!(code, 0, "{}", err);
}
