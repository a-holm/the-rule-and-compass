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
/// The embargo of a store whose writer declared none (`protocol::DEFAULT_LOCKBOX_EMBARGO_SECONDS`).
const EXISTING_EMBARGO: i64 = 10_972_800;

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
    assert_eq!(lockbox::stored(&d).unwrap().map(|(_, l)| l), Some(locked()));
    let (code, _, err) = search(&d, &run, &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("split has"), "{}", err);
    assert!(err.contains("declared locked test 2024-01-13..2024-01-17 (store manifest"), "{}", err);
    assert!(err.contains(&format!("--exclude 'ts >= {} and ts < {}'", locked().start - DEFAULT_PURGE, locked().end + EXISTING_EMBARGO)), "{}", err);

    let at_start = format!("ts >= {}", locked().start);
    let (code, _, err) = search(&d, &run, &["--exclude", &at_start]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("whose label window reaches the declared locked test"), "{}", err);

    let exclude = format!("ts >= {}", locked().start - DEFAULT_PURGE);
    assert!(err.contains(&format!("--exclude '{} and ts < {}'", exclude, locked().end + EXISTING_EMBARGO)), "{}", err);
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
            locked_test: Some(rcsr::protocol::LockedRange { start: range.start, end: range.end, source: "store".into() }),
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
fn stored_reads_each_carrier_and_a_flag_must_cover_the_store_range() {
    let dir = scratch("declared");
    let p = s(&dir.join("a.rcsb"));
    assert_eq!(lockbox::stored(&p).unwrap().map(|(_, l)| l), None);
    let lb = locked();
    std::fs::write(format!("{p}.manifest.json"), format!("{{\"locked_test\":{}}}", lb.declaration_json("t"))).unwrap();
    assert_eq!(lockbox::stored(&p).unwrap().map(|(_, l)| l), Some(lb));
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
    assert!(e.contains("3 row(s)"), "{}", e);
    assert!(d.check_times("test split", "ts", &times[..1], 0).is_ok());
}

/// Hourly panel of 3 symbols, 500 bars. Bars from index 400 on (the locked
/// range) are different when `flip` is set; every earlier bar is identical.
/// The label at bar i is the return of bar i+1; features read past
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
            let y = ret[i + 1];
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
        .filter(|t| !(t.contains("_s\"") || t.contains("seconds") || t.contains("wall") || t.contains("denominator") || t.contains("declared_in")))
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
        assert!(err.contains(&format!("--exclude 'ts >= {} and ts < {}'", s_start - DEFAULT_PURGE, s_start + 100 * H + EXISTING_EMBARGO)), "{}", err);
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

    // walk-forward: horizon 3600 s; a fold ending at the locked start is refused
    let wf_run = |d: &str, test_end: i64| {
        let window = format!("f1,{},{},{},{}", T0, T0 + 150 * H, T0 + 160 * H, test_end);
        rcsr(&[
            "walkforward", "--data", d, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym",
            "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", &window,
        ])
    };
    for d in [&a, &b] {
        let (code, _, err) = wf_run(d, s_start);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("reaches the declared locked test"), "{}", err);
    }
    let (ca, oa, ea) = wf_run(&a, s_start - 3600);
    let (cb, ob, eb) = wf_run(&b, s_start - 3600);
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
            "sym", "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365", "--json", "--ledger", &l,
        ];
        args.extend_from_slice(extra);
        rcsr(&args)
    };
    assert_eq!(run(&["--window", &dev]).0, 0);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let late = format!("late,{},{},{},{}", T0, T0 + 399 * H + 1800, T0 + 400 * H, T0 + 500 * H);
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
    assert!(err.contains("reaches the declared locked test"), "{}", err);
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
    let inside = (0..400i64).filter(|i| T1 + i * H >= DEFAULT_START - DEFAULT_PURGE).count();
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
    let exclude = format!("ts >= {}", locked().start - DEFAULT_PURGE);
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
    let exclude = format!("ts >= {}", locked().start - DEFAULT_PURGE);
    let at_start = format!("ts >= {}", locked().start);
    let mut cv = Vec::new();
    for p in [&a, &b] {
        let d = s(p);
        lockbox::write_sidecar(&d, &locked(), "test").unwrap();
        let train0 = format!("ts < {}", T0 + 100 * H);
        let test0 = format!("(ts >= {}) * (ts < {})", T0 + 100 * H, T0 + 150 * H);
        let (code, _, err) = plain(&[
            "--data", &d, "--time", "ts", "--train-select", &train0, "--test-select", &test0, "--exclude", &at_start,
            "--cv", "k-fold", "--cv-k", "3", "--generations", "2", "--pop", "8",
        ]);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("cross-validation rows has"), "{}", err);
        assert!(err.contains("whose label window reaches the declared locked test"), "{}", err);
        assert!(err.contains(&format!("--exclude '{} and ts < {}'", exclude, locked().end + EXISTING_EMBARGO)), "{}", err);
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

fn audit_series(path: &Path, locked_ts: impl Fn(i64) -> i64) {
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        let ts = T1 + i * H;
        let ts = if ts >= DEFAULT_START { locked_ts(i) } else { ts };
        csv.push_str(&format!("{},{},{}\n", ts, x, y));
    }
    std::fs::write(path, csv).unwrap();
}

fn json_part(json: &str, from: &str, to: &str) -> String {
    let a = json.find(from).unwrap_or_else(|| panic!("{from} in {json}"));
    let rest = &json[a..];
    rest[..rest.find(to).unwrap()].to_string()
}

#[test]
fn audit_time_statistics_and_rows_ignore_the_locked_rows() {
    let dir = scratch("audit-time");
    let (a, b, kept) = (dir.join("a.csv"), dir.join("b.csv"), dir.join("kept.csv"));
    // A: regular hourly locked rows. B: locked rows with duplicates, jumps and
    // backwards steps. Identical outside the default locked test.
    audit_series(&a, |i| T1 + i * H);
    audit_series(&b, |i| DEFAULT_START + (i % 5) * 7 * H + (i % 3) * 86_400 + 400 * H * (i % 2));
    let mut text = String::from("ts,x,y\n");
    for line in std::fs::read_to_string(&a).unwrap().lines().skip(1) {
        if line.split(',').next().unwrap().parse::<i64>().unwrap() < DEFAULT_START - DEFAULT_PURGE {
            text.push_str(line);
            text.push('\n');
        }
    }
    std::fs::write(&kept, text).unwrap();
    let run = |p: &Path| rcsr(&["audit", "--data", &s(p), "--target", "y", "--time", "ts", "--json"]);
    let (_, ja, ea) = run(&a);
    let (_, jb, eb) = run(&b);
    let (_, jk, ek) = run(&kept);
    let part = |j: &str| {
        format!(
            "{}|{}|{}",
            json_part(j, "\"rows\":", ","),
            json_part(j, "\"time_audit\":{", "}"),
            json_part(j, "\"leaks\":[", "]")
        )
    };
    assert_eq!(part(&ja), part(&jk), "{ja}{ea}{jk}{ek}");
    assert_eq!(part(&jb), part(&jk), "{jb}{eb}{jk}{ek}");
    assert!(json_part(&jb, "\"time_audit\":{", "}").contains("\"duplicate_instants\":0,\"non_monotonic_rows\":0"), "{jb}");
    let kept_rows = 400 - (0..400i64).filter(|i| T1 + i * H >= DEFAULT_START - DEFAULT_PURGE).count();
    assert!(jb.contains(&format!("\"rows\":{kept_rows},")), "{jb}");
    assert!(jb.contains("\"rows_left_out\":"), "{jb}");
}

#[test]
fn a_second_declared_range_is_guarded_by_every_reader() {
    let dir = scratch("also");
    let extra = Lockbox { start: T0 + 100 * H, end: T0 + 150 * H };
    let default = lockbox::protocol_default();
    for (name, t0) in [("flag.csv", T0), ("default.csv", T1)] {
        let data = dir.join(name);
        series_csv_at(&data, t0);
        let d = s(&data);
        let also = lockbox::also_json(Some(&extra), "test");
        lockbox::write_sidecar_with(&d, &default, "test", &also).unwrap();
        let resolved = lockbox::resolve(None, &d).unwrap();
        assert_eq!(resolved.ranges(), vec![default, extra]);
        let run = s(&dir.join("run.json"));
        let (code, _, err) = search(&d, &run, &[]);
        assert_eq!(code, 3, "{name}: {err}");
        assert!(err.contains("declared locked test"), "{name}: {err}");
        // a flag that covers only one of the two ranges is refused
        let e = lockbox::resolve(Some(&format!("{},{}", default.start, default.end)), &d).unwrap_err();
        assert_eq!(e.exit, 2, "{}", e.message);
        // the audit leaves out the rows of both ranges
        let (_, out, err) = rcsr(&["audit", "--data", &d, "--target", "y", "--time", "ts", "--json"]);
        let inside = (0..400i64)
            .filter(|i| [default, extra].iter().any(|r| (r.start - DEFAULT_PURGE..r.end + EXISTING_EMBARGO).contains(&(t0 + i * H))))
            .count();
        assert!(out.contains(&format!("\"rows_left_out\":{inside}")), "{name}: {out}{err}");
    }
}

// ---------------------------------------------------------------------------
// #582: the lookback embargo after the end of a declared range.
// ---------------------------------------------------------------------------

const EMB_BARS: i64 = 420;
const EMB_S: i64 = 200;
const EMB_E: i64 = 260;
/// 24 bars: the longest trailing window of the fixture features.
const EMB_L: i64 = 86_400;

fn emb_range() -> Lockbox {
    Lockbox { start: T0 + EMB_S * H, end: T0 + EMB_E * H }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Three hourly series of independent price levels. `locked_variant` redraws
/// the levels of `[S, E)` only. `f1` is a 24-bar trailing mean, `f2` a 12-bar
/// one, `f3` a one-bar difference and `y` the next one-bar change, so rows from
/// `E + 24 h` on read no bar of the range.
fn emb_panel(path: &Path, locked_variant: bool) {
    let mut csv = String::from("ts,sym,f1,f2,f3,y,ret,vol\n");
    for g in 0..3u64 {
        let mut st = 0x9E37_79B9 + g;
        let mut alt = 0xC0FF_EE00 + g;
        let mut p: Vec<f64> = Vec::new();
        for i in 0..=EMB_BARS {
            let base = 100.0 + 5.0 * lcg(&mut st);
            let other = 100.0 + 5.0 * lcg(&mut alt);
            let inside = (EMB_S..EMB_E).contains(&i);
            p.push(if locked_variant && inside { other } else { base });
        }
        for i in 0..EMB_BARS as usize {
            let mean = |w: usize| {
                let lo = (i + 1).saturating_sub(w);
                p[lo..=i].iter().sum::<f64>() / (i + 1 - lo) as f64
            };
            let f3 = if i == 0 { 0.0 } else { p[i] - p[i - 1] };
            let y = p[i + 1] - p[i];
            csv.push_str(&format!("{},s{},{},{},{},{},{},{}\n", T0 + i as i64 * H, g, mean(24), mean(12), f3, y, y, 1.0 + f3.abs()));
        }
    }
    std::fs::write(path, csv).unwrap();
}

fn emb_declare(data: &str, seconds: i64) {
    let embargo = lockbox::embargo_json(seconds, "f1", "test", H);
    lockbox::write_sidecar_with(data, &emb_range(), "test", &lockbox::embargo_member(Some(&embargo))).unwrap();
}

fn emb_store(dir: &Path, name: &str, locked_variant: bool, seconds: i64) -> String {
    let p = dir.join(format!("{name}.csv"));
    emb_panel(&p, locked_variant);
    let d = s(&p);
    emb_declare(&d, seconds);
    d
}

fn emb_search(data: &str, run: &str, exclude: &str, extra: &[&str]) -> (i32, String) {
    let mut a = vec![
        "--data", data, "--target", "y", "--vars", "f1,f2", "--time", "ts", "--group-by", "sym", "--split",
        "temporal", "--validation-frac", "0.2", "--fill", "drop", "--exclude", exclude, "--seed", "3",
        "--generations", "5", "--pop", "20", "--cpu", "--no-dev-log", "--quiet", "--json-out", run,
    ];
    a.extend_from_slice(extra);
    let (code, _, err) = rcsr(&a);
    (code, err)
}

fn emb_fold(path: &Path, label: &str, test_start: i64, test_end: i64) {
    std::fs::write(
        path,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\n\
             expanding,{label},{},{},{},{},{}\n",
            T0,
            T0 + 100 * H,
            T0 + 80 * H,
            test_start,
            test_end
        ),
    )
    .unwrap();
}

fn emb_wf(data: &str, test_start: i64, test_end: i64, extra: &[&str]) -> (i32, String, String) {
    let window = format!("f1,{},{},{},{}", T0, T0 + 100 * H, test_start, test_end);
    let mut a = vec![
        "walkforward", "--data", data, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym",
        "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365", "--threshold-fixed", "0", "--json",
        "--window", &window,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

fn emb_ladder(data: &str, win: &str, out: &str, extra: &[&str]) -> (i32, String) {
    let mut a = vec![
        "ladder", "--data", data, "--target", "y", "--return-col", "ret", "--time", "ts", "--group", "sym",
        "--features", "f1,f2", "--windows", win, "--scheme", "expanding", "--horizon-seconds", "3600",
        "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50", "--max-fit-rows", "600", "--out", out,
    ];
    a.extend_from_slice(extra);
    let (code, _, err) = rcsr(&a);
    (code, err)
}

#[test]
fn development_outputs_after_the_range_ignore_locked_prices() {
    let dir = scratch("embargo-differential");
    let a = emb_store(&dir, "a", false, EMB_L);
    let b = emb_store(&dir, "b", true, EMB_L);
    let (e, after) = (emb_range().end, emb_range().end + EMB_L);

    // search: rows from E read locked bars through their trailing features
    for d in [&a, &b] {
        let (code, err) = emb_search(d, &s(&dir.join("x.json")), &format!("ts < {e}"), &[]);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("whose features read the declared locked test"), "{}", err);
        assert!(err.contains(&format!("ts < {after}")), "{}", err);
    }
    let (run_a, run_b) = (s(&dir.join("ra.json")), s(&dir.join("rb.json")));
    let clean = format!("ts < {after}");
    assert_eq!(emb_search(&a, &run_a, &clean, &[]).0, 0);
    assert_eq!(emb_search(&b, &run_b, &clean, &[]).0, 0);
    let (ja, jb) = (std::fs::read_to_string(&run_a).unwrap(), std::fs::read_to_string(&run_b).unwrap());
    for key in ["train_error", "validation_error", "test_error"] {
        let (na, nb) = (numbers(&ja, key), numbers(&jb, key));
        assert!(!na.is_empty(), "{key}: {ja}");
        assert_eq!(na, nb, "{key}");
    }
    assert!(ja.contains(&format!("\"lockbox_embargo_seconds\":{EMB_L},\"lockbox_embargo_source\":\"store\"")), "{ja}");

    // walk-forward: a fold whose test starts at E is refused, one from E + L is not
    for d in [&a, &b] {
        let (code, _, err) = emb_wf(d, e, e + 40 * H, &[]);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("reaches the declared locked test"), "{}", err);
    }
    let (ca, oa, ea) = emb_wf(&a, after, after + 40 * H, &[]);
    let (cb, ob, eb) = emb_wf(&b, after, after + 40 * H, &[]);
    assert_eq!((ca, cb), (0, 0), "{}{}", ea, eb);
    let (ha, hb) = (numbers(&oa, "report_sha256"), numbers(&ob, "report_sha256"));
    assert!(!ha.is_empty(), "{}", oa);
    assert_eq!(ha, hb);

    // ladder
    let win = dir.join("w.csv");
    emb_fold(&win, "leak", e, e + 40 * H);
    for d in [&a, &b] {
        let (code, err) = emb_ladder(d, &s(&win), &s(&dir.join("l.json")), &[]);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("reaches the declared locked test"), "{}", err);
    }
    emb_fold(&win, "clean", after, after + 40 * H);
    let (la, lb) = (s(&dir.join("la.json")), s(&dir.join("lb.json")));
    assert_eq!(emb_ladder(&a, &s(&win), &la, &[]).0, 0);
    assert_eq!(emb_ladder(&b, &s(&win), &lb, &[]).0, 0);
    let (ja, jb) = (std::fs::read_to_string(&la).unwrap(), std::fs::read_to_string(&lb).unwrap());
    assert_eq!(without_timing(&ja), without_timing(&jb));
    assert!(ja.contains(&format!("\"embargo_seconds\":{EMB_L},\"embargo_source\":\"store\"")), "{ja}");
}

#[test]
fn cross_validation_refuses_the_embargo_rows_and_ignores_the_locked_prices_after_it() {
    let dir = scratch("embargo-cv");
    let a = emb_store(&dir, "a", false, EMB_L);
    let b = emb_store(&dir, "b", true, EMB_L);
    let after = emb_range().end + EMB_L;
    let cv = |d: &str, exclude: &str, out: &str| {
        let (code, _, err) = plain(&[
            "--data", d, "--time", "ts", "--vars", "f1,f2", "--cv", "k-fold", "--cv-k", "3", "--exclude", exclude,
            "--generations", "3", "--pop", "16", "--json-out", out,
        ]);
        (code, err)
    };
    for d in [&a, &b] {
        let (code, err) = cv(d, &format!("ts < {}", emb_range().end), &s(&dir.join("x.json")));
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("whose features read the declared locked test"), "{}", err);
    }
    let (ra, rb) = (s(&dir.join("ca.json")), s(&dir.join("cb.json")));
    assert_eq!(cv(&a, &format!("ts < {after}"), &ra).0, 0);
    assert_eq!(cv(&b, &format!("ts < {after}"), &rb).0, 0);
    let (ja, jb) = (std::fs::read_to_string(&ra).unwrap(), std::fs::read_to_string(&rb).unwrap());
    assert_eq!(cv_part(&ja), cv_part(&jb));
}

#[test]
fn the_embargo_flag_widens_the_declared_value_and_never_narrows_it() {
    let dir = scratch("embargo-flag");
    let a = emb_store(&dir, "a", false, EMB_L);
    let run = s(&dir.join("r.json"));
    let after = emb_range().end + EMB_L;
    let (code, err) = emb_search(&a, &run, &format!("ts < {after}"), &["--lockbox-embargo", "3600"]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("a flag may widen it, never narrow it"), "{}", err);
    let wider = EMB_L + 24 * H;
    let (code, err) = emb_search(&a, &run, &format!("ts < {after}"), &["--lockbox-embargo", &wider.to_string()]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains(&format!("lookback embargo {wider} s")), "{}", err);
    let (code, err) = emb_search(&a, &run, &format!("ts < {}", emb_range().end + wider), &["--lockbox-embargo", &wider.to_string()]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains(&format!("\"embargo_seconds\":{wider},\"embargo_source\":\"flag\"")), "{json}");
    assert!(json.contains(&format!("\"lockbox_embargo_seconds\":{wider},\"lockbox_embargo_source\":\"flag\"")), "{json}");
}

#[test]
fn a_store_without_a_declared_embargo_uses_the_protocol_default() {
    let dir = scratch("embargo-default");
    let p = dir.join("a.csv");
    emb_panel(&p, false);
    let d = s(&p);
    lockbox::write_sidecar(&d, &emb_range(), "test").unwrap();
    let run = s(&dir.join("r.json"));
    let (code, err) = emb_search(&d, &run, &format!("ts < {}", emb_range().end + EMB_L), &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("lookback embargo 10972800 s after the end (protocol default)"), "{}", err);
}

#[test]
fn audit_leaves_out_the_purge_band_and_the_embargo_rows() {
    let dir = scratch("embargo-audit");
    let a = emb_store(&dir, "a", false, EMB_L);
    let r = emb_range();
    let (_, out, err) = rcsr(&["audit", "--data", &a, "--target", "y", "--time", "ts", "--json"]);
    let band = (0..EMB_BARS)
        .filter(|i| {
            let t = T0 + i * H;
            t >= r.start - DEFAULT_PURGE && t < r.end + EMB_L
        })
        .count()
        * 3;
    assert!(band > 3 * (EMB_E - EMB_S) as usize, "the band reaches past the range on both sides");
    assert!(out.contains(&format!("\"rows_left_out\":{band}")), "{out}{err}");
    assert!(out.contains(&format!("\"embargo_seconds\":{EMB_L}")), "{out}");
}

#[test]
fn ablation_prepare_segments_the_embargo_and_its_null_columns_ignore_locked_prices() {
    let dir = scratch("embargo-ablation");
    let a = emb_store(&dir, "a", false, EMB_L);
    let b = emb_store(&dir, "b", true, EMB_L);
    let def = dir.join("def.json");
    std::fs::write(
        &def,
        r#"{"schema":"rcsr-ablation/1","b0":["f1"],"groups":{"g":["f2","f3"]},"placebo_template_group":"g",
"placebo_prefix":"placebo_","permute_prefix":"perm_","permute_block_len":24,"generation_budget":1,
"screen_seeds":3,"finalist_seeds":3,"trials":{"windows":1,"horizons":1},
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["g"]},{"name":"A2","add":["g"],"placebo":1},{"name":"A3","add":[],"permute":["g"]}],
"rule":{"min_seeds":3,"spa_alpha":0.05,"fdr_q":0.10,"min_dsr":0.5,"draws":200,"block_len":0,"seed":0,"pbo_blocks":8,"max_pbo":0.5,
"families":[{"name":"g","treatment":"A1","controls":["A0"],"null_arm":"A3","null_control":"A0"}]}}"#,
    )
    .unwrap();
    let mut outs = Vec::new();
    for (tag, d) in [("a", &a), ("b", &b)] {
        let out = s(&dir.join(format!("{tag}.prepared.csv")));
        let (code, _, err) = rcsr(&[
            "ablation", "prepare", "--definition", &s(&def), "--data", d, "--time", "ts", "--group", "sym", "--keep",
            "y", "--seed", "1", "--out", &out,
        ]);
        assert_eq!(code, 0, "{}", err);
        outs.push(out);
    }
    let r = emb_range();
    let side = std::fs::read_to_string(lockbox::sidecar_path(&outs[0])).unwrap();
    assert!(
        side.contains(&format!("\"kind\":\"embargo\",\"start\":{},\"end\":{},", r.end, r.end + EMB_L)),
        "{side}"
    );
    assert_eq!(lockbox::declared_embargo(&outs[0]).unwrap().map(|(_, s)| s), Some(EMB_L));
    let rows = |p: &str| -> Vec<String> {
        let text = std::fs::read_to_string(p).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert!(lines[0].contains("placebo_1") && lines[0].contains("perm_"), "{}", lines[0]);
        lines[1..]
            .iter()
            .filter(|l| l.split(',').next().unwrap().parse::<i64>().unwrap() >= r.end + EMB_L)
            .map(|l| l.to_string())
            .collect()
    };
    let (ra, rb) = (rows(&outs[0]), rows(&outs[1]));
    assert!(ra.len() > 300, "{}", ra.len());
    assert_eq!(ra, rb, "placebo and permuted columns from E + L do not read the range");
    // a search on the prepared rows refuses the embargo band
    let (code, _, err) = rcsr(&[
        "--data", &outs[0], "--target", "y", "--vars", "f1", "--time", "ts", "--group-by", "sym", "--split", "temporal",
        "--exclude", &format!("ts < {}", r.end), "--eval-expr", "f1", "--cpu", "--no-dev-log", "--quiet",
    ]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("whose features read the declared locked test"), "{}", err);
}

const LONG_LABEL_SPEC: &str = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"label\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"ret\",\"return_units\":\"log\",\
\"horizon_seconds\":270000,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\"exit_price\":\"close\",\
\"offset\":0.002,\"scale\":{\"kind\":\"column\",\"column\":\"vol\",\"multiplier\":6.0,\"window_seconds\":3600,\"ends_at_offset_seconds\":0},\
\"clip\":[-1.0,1.0],\"units\":\"log_return/(6*vol)\",\"cost_bps\":null,\"cost_in_label\":false,\
\"construction_columns\":[\"vol\"],\"note\":null}]}";

#[test]
fn ladder_takes_the_target_spec_purge_before_the_range_start() {
    let dir = scratch("embargo-spec-purge");
    let a = emb_store(&dir, "a", false, EMB_L);
    let spec = dir.join("spec.json");
    std::fs::write(&spec, LONG_LABEL_SPEC).unwrap();
    // a fold ending 68 h before S: clear of the 61 h default purge, inside the 75 h label of the spec
    let win = dir.join("w.csv");
    std::fs::write(
        &win,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f,{},{},{},{},{}\n",
            T0,
            T0 + 120 * H,
            T0 + 100 * H,
            T0 + 124 * H,
            T0 + (EMB_S - 68) * H
        ),
    )
    .unwrap();
    let run = |extra: &[&str], out: &str| {
        let win_s = s(&win);
        let mut args = vec![
            "ladder", "--data", a.as_str(), "--target", "y", "--return-col", "ret", "--time", "ts", "--group", "sym",
            "--features", "f1,f2", "--windows", &win_s, "--scheme", "expanding", "--gbm-trees", "10", "--draws", "50",
            "--max-fit-rows", "600", "--out", out,
        ];
        args.extend_from_slice(extra);
        let args: Vec<String> = args.iter().map(|x| x.to_string()).collect();
        let refs: Vec<&str> = args.iter().map(String::as_str).collect();
        rcsr(&refs)
    };
    let free = s(&dir.join("free.json"));
    let (code, _, err) = run(&[], &free);
    assert_eq!(code, 0, "{}", err);
    assert!(std::fs::read_to_string(&free).unwrap().contains("\"purge_source\":\"protocol_default\""));
    let (code, _, err) = run(&["--target-spec", &s(&spec)], &s(&dir.join("spec.out.json")));
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("reaches the declared locked test"), "{}", err);
    assert!(err.contains("label end = stamp + 270000 s"), "{}", err);
}

#[test]
fn a_sealed_read_keeps_every_other_declared_range_sealed() {
    let dir = scratch("embargo-other-range");
    let data = dir.join("d.csv");
    emb_panel(&data, false);
    let d = s(&data);
    let primary = emb_range();
    let other = Lockbox { start: T0 + 100 * H, end: T0 + 140 * H };
    let also = lockbox::also_json(Some(&other), "test");
    let embargo = lockbox::embargo_member(Some(&lockbox::embargo_json(EMB_L, "f1", "test", H)));
    lockbox::write_sidecar_with(&d, &primary, "test", &format!("{also}{embargo}")).unwrap();
    assert_eq!(lockbox::resolve(None, &d).unwrap().ranges(), vec![primary, other]);
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));

    // the training range reaches the other range, not the primary one
    let window = format!("w,{},{},{},{}", T0, T0 + 150 * H, primary.start, primary.end);
    let (code, _, err) = rcsr(&[
        "walkforward", "--data", &d, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym",
        "--horizon", "3600", "--cost-bps", "5", "--periods-per-year", "365", "--json", "--locked-test", "--ledger", &l,
        "--sealed", &sealed, "--window", &window,
    ]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("reaches the declared locked test 2024-01-05"), "{}", err);

    // the same for the ladder: the ledger and the seal are valid, the fold still reaches the other range
    Ledger::new(dir.join("ledger.jsonl"))
        .add_candidate(&Candidate {
            id: "c1".into(),
            spec_sha256: "a".repeat(64),
            data_sha256: "b".repeat(64),
            build_commit: "build".into(),
            report_sha256: "c".repeat(64),
            expression: "x".into(),
            locked_test: Some(rcsr::protocol::LockedRange { start: primary.start, end: primary.end, source: "store".into() }),
        })
        .unwrap();
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let win = dir.join("w.csv");
    std::fs::write(
        &win,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nlocked,w,{},{},{},{},{}\n",
            T0,
            T0 + 150 * H,
            T0 + 120 * H,
            primary.start,
            primary.end
        ),
    )
    .unwrap();
    let (code, _, err) = rcsr(&[
        "ladder", "--data", &d, "--target", "y", "--return-col", "ret", "--time", "ts", "--group", "sym", "--features",
        "f1,f2", "--windows", &s(&win), "--scheme", "locked", "--horizon-seconds", "3600", "--purge-seconds", "7200",
        "--gbm-trees", "10", "--draws", "50", "--max-fit-rows", "600", "--out", &s(&dir.join("o.json")),
        "--locked-test", "--ledger", &l, "--sealed", &sealed,
    ]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("reaches the declared locked test"), "{}", err);
    assert!(!std::fs::read_to_string(&l).unwrap().contains("baseline_read"), "no read is recorded");
}
