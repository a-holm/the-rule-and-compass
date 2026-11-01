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
    assert!(err.contains("test split has"), "{}", err);
    assert!(err.contains("declared locked test 2024-01-13..2024-01-17 (store manifest"), "{}", err);
    assert!(err.contains(&format!("--exclude 'ts >= {}'", locked().start)), "{}", err);

    let exclude = format!("ts >= {}", locked().start);
    let (code, _, err) = search(&d, &run, &["--exclude", &exclude]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"rows_in_scope\":0"), "{}", json);
    assert!(json.contains("\"evaluated\":false"), "{}", json);
    assert!(json.contains("\"source\":\"store\""), "{}", json);

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
    assert!(err.contains("touches the declared locked test"), "{}", err);

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
fn declared_for_reads_each_carrier_and_an_explicit_flag_wins() {
    let dir = scratch("declared");
    let p = s(&dir.join("a.rcsb"));
    assert_eq!(lockbox::declared_for(&p), None);
    let lb = locked();
    std::fs::write(format!("{p}.manifest.json"), format!("{{\"locked_test\":{}}}", lb.declaration_json("t"))).unwrap();
    assert_eq!(lockbox::declared_for(&p), Some(lb));
    let flag = format!("{},{}", T0, T0 + H);
    let d = lockbox::resolve_declared(Some(&flag), &p).unwrap().unwrap();
    assert_eq!((d.source, d.lockbox.start), ("flag", T0));
    let d = lockbox::resolve_declared(None, &p).unwrap().unwrap();
    assert_eq!(d.source, "store");
    assert!(lockbox::resolve_declared(None, "-").unwrap().is_none());
    let times = [lb.start - 1, lb.start, lb.end - 1, lb.end];
    let e = d.check_times("test split", "ts", &times).unwrap_err();
    assert!(e.contains("2 row(s)"), "{}", e);
    assert!(d.check_times("test split", "ts", &times[..1]).is_ok());
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

    let exclude = format!("ts >= {}", locked().start);
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
