//! #924: `rcsr protocol retire` and `--retired`: a locked range leaves the guard
//! only after the ledger proves its one sealed read. Fixtures are synthetic.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::lockbox::{self, Lockbox};
use rcsr::protocol::{Candidate, Ledger, LockedRange};
use scratch::Scratch;
use std::path::{Path, PathBuf};

const T0: i64 = 1_704_067_200;
const H: i64 = 3600;
const DEFAULT_RANGE: &str = "2025-07-01,2026-07-01";
const NEW_RANGE: &str = "2026-07-01,2027-07-01";
/// Disjoint from the protocol default, after the data.
const STAND_IN: &str = "2030-01-01,2030-02-01";

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let s = Scratch::new(name);
    let d = s.path().to_path_buf();
    (s, d)
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = rcsr_cmd::rcsr().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn default_range() -> Lockbox {
    lockbox::protocol_default()
}

fn candidate(tag: &str) -> Candidate {
    let spec = format!("spec-{}", tag);
    Candidate {
        id: Candidate::address(&spec, "data", "build"),
        spec_sha256: spec,
        data_sha256: "data".to_string(),
        build_commit: "build".to_string(),
        report_sha256: format!("report-{}", tag),
        expression: format!("x + {}", tag.len()),
        locked_test: None,
    }
}

/// A ledger with candidates a and b, a seal when `seal`, and one read per tag in `reads`.
fn ledger_with(dir: &Path, seal: bool, reads: &[&str]) -> String {
    let path = dir.join("ledger.jsonl");
    let ledger = Ledger::new(&path);
    let (a, b) = (candidate("a"), candidate("b"));
    ledger.add_candidate(&a).unwrap();
    ledger.add_candidate(&b).unwrap();
    let range = LockedRange::new(&default_range(), "store");
    if seal {
        ledger.seal(&dir.join("sealed.json"), &[]).unwrap();
    }
    for tag in reads {
        let c = if *tag == "a" { &a } else { &b };
        ledger.record_read(&c.id, &format!("locked-{}", tag), Some(&range)).unwrap();
    }
    s(&path)
}

fn retire(ledger: &str, range: &str, out: &str) -> (i32, String, String) {
    rcsr(&["protocol", "retire", "--ledger", ledger, "--range", range, "--out", out])
}

#[test]
fn retire_needs_one_read() {
    let (_s, dir) = scratch("retire-zero");
    let out = s(&dir.join("retired.json"));
    let ledger = ledger_with(&dir, true, &[]);
    let (code, _, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("range not read"), "{}", err);
    assert!(!Path::new(&out).exists());

    // a read of another range does not count
    let (_s2, dir2) = scratch("retire-other-range");
    let ledger = ledger_with(&dir2, true, &["a"]);
    let (code, _, err) = retire(&ledger, NEW_RANGE, &out);
    assert_eq!(code, 2, "{}", err);
    assert!(!Path::new(&out).exists());

    // one read after a seal writes the record
    let (_s3, dir3) = scratch("retire-one");
    let ledger = ledger_with(&dir3, true, &["a"]);
    let out = s(&dir3.join("retired.json"));
    let (code, stdout, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 0, "{}", err);
    assert!(stdout.contains("retired 2025-07-01..2026-07-01"), "{}", stdout);
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.contains("\"schema\":\"rcsr-retired/1\""), "{}", text);
    assert!(text.contains(&format!("\"range\":{{\"start\":{},\"end\":{}}}", default_range().start, default_range().end)), "{}", text);
    assert!(text.contains(&format!("\"ledger_sha256\":\"{}\"", rcsr::manifest::sha256_file(&ledger).unwrap())), "{}", text);
    assert!(text.contains(&format!("\"read_event\":\"{}\"", candidate("a").id)), "{}", text);
    assert!(text.contains("\"verified_reads\":1"), "{}", text);
    assert!(!dir3.join("retired.json.tmp").exists());
}

#[test]
fn retire_refuses_two_reads() {
    let (_s, dir) = scratch("retire-two");
    let out = s(&dir.join("retired.json"));
    let ledger = ledger_with(&dir, true, &["a", "b"]);
    let (code, _, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("2 locked reads"), "{}", err);
    assert!(!Path::new(&out).exists());
}

#[test]
fn retire_refuses_a_read_before_any_seal() {
    let (_s, dir) = scratch("retire-unsealed");
    let out = s(&dir.join("retired.json"));
    let ledger = ledger_with(&dir, false, &["a"]);
    let (code, _, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("follows no seal"), "{}", err);
}

#[test]
fn retire_refuses_an_edited_ledger() {
    let (_s, dir) = scratch("retire-edited");
    let out = s(&dir.join("retired.json"));
    let ledger = ledger_with(&dir, true, &["a"]);
    let text = std::fs::read_to_string(&ledger).unwrap();
    std::fs::write(&ledger, text.replacen("spec-a", "spec-z", 1)).unwrap();
    let (code, _, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 1, "{}", err);
    assert!(err.contains("breaks the hash chain"), "{}", err);
}

/// A record for the default range made from a one-read ledger.
fn record(dir: &Path) -> (String, String) {
    let ledger = ledger_with(dir, true, &["a"]);
    let out = s(&dir.join("retired.json"));
    let (code, _, err) = retire(&ledger, DEFAULT_RANGE, &out);
    assert_eq!(code, 0, "{}", err);
    (ledger, out)
}

#[test]
fn retired_record_binds_to_the_ledger() {
    let (_s, dir) = scratch("bind");
    let (ledger, record) = record(&dir);
    let data = dir.join("d.csv");
    series_csv(&data);
    let run = s(&dir.join("run.json"));
    let (code, _, err) = search(&s(&data), &run, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 0, "{}", err);

    // the ledger gains a line after the record was written
    let l = Ledger::new(&ledger);
    l.record_read(&candidate("b").id, "locked-b", None).unwrap();
    let (code, _, err) = search(&s(&data), &run, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("no longer matches the record"), "{}", err);

    // a missing ledger is a mismatch too
    std::fs::remove_file(&ledger).unwrap();
    let (code, _, err) = search(&s(&data), &run, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 3, "{}", err);
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

fn search(data: &str, run_json: &str, extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "--data", data, "--target", "y", "--vars", "x", "--time", "ts", "--split", "temporal", "--cpu",
        "--no-dev-log", "--seed", "3", "--generations", "5", "--pop", "20", "--metric", "squared", "--quiet",
        "--json-out", run_json,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

fn wf(data: &str, extra: &[&str]) -> (i32, String, String) {
    let w = format!("w,{},{},{},{}", T0, T0 + 100 * H, T0 + 110 * H, T0 + 250 * H);
    let mut a = vec![
        "walkforward", "--data", data, "--target", "y", "--expr", "x", "--time", "ts", "--horizon", "5",
        "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", &w,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

fn ladder(data: &str, win: &str, out: &str, extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "ladder", "--data", data, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
        "--windows", win, "--scheme", "expanding", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws",
        "50", "--max-fit-rows", "600", "--out", out,
    ];
    a.extend_from_slice(extra);
    rcsr(&a)
}

#[test]
fn a_report_names_the_retirement_and_drops_the_default_from_the_guard() {
    let (_s, dir) = scratch("reports");
    let (_ledger, record) = record(&dir);
    let rec_text = std::fs::read_to_string(&record).unwrap();
    let ledger_sha = rec_text.split("\"ledger_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    let event = candidate("a").id;
    let (data, panel, win) = (dir.join("d.csv"), dir.join("p.csv"), dir.join("w.csv"));
    series_csv(&data);
    panel_csv(&panel);
    std::fs::write(
        &win,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,e0,{},{},{},{},{}\n",
            T0,
            T0 + 398 * H,
            T0 + 250 * H,
            T0 + 400 * H,
            T0 + 500 * H
        ),
    )
    .unwrap();
    let (d, p, w) = (s(&data), s(&panel), s(&win));
    let want_retired = format!("\"retired\":{{\"range\":{{\"start\":{},\"end\":{},\"start_date\":\"2025-07-01\",\"end_date\":\"2026-07-01\"}},\"ledger_sha256\":\"{}\",\"read_event\":\"{}\"}}", default_range().start, default_range().end, ledger_sha, event);

    // search
    let run = s(&dir.join("run.json"));
    let (code, _, err) = search(&d, &run, &["--lockbox", STAND_IN]);
    assert_eq!(code, 0, "{}", err);
    let plain = std::fs::read_to_string(&run).unwrap();
    assert!(plain.contains("\"source\":\"flag+protocol_default\"") && plain.contains("\"also\":{"), "{}", plain);
    assert!(!plain.contains("\"retired\""), "{}", plain);
    let (code, _, err) = search(&d, &run, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&run).unwrap();
    assert!(json.contains("\"source\":\"flag\"") && json.contains("\"also\":null"), "{}", json);
    assert!(json.contains(&want_retired), "{}", json);

    // walkforward
    let (code, out, err) = wf(&d, &["--lockbox", STAND_IN]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"source\":\"flag+protocol_default\"") && !out.contains("\"retired\""), "{}", out);
    let (code, out, err) = wf(&d, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"source\":\"flag\"") && out.contains("\"also\":null"), "{}", out);
    assert!(out.contains(&want_retired), "{}", out);

    // ladder
    let rep = s(&dir.join("ladder.json"));
    let (code, _, err) = ladder(&p, &w, &rep, &["--lockbox", STAND_IN]);
    assert_eq!(code, 0, "{}", err);
    let plain = std::fs::read_to_string(&rep).unwrap();
    assert!(plain.contains("\"source\":\"flag+protocol_default\"") && !plain.contains("\"retired\""), "{}", plain);
    let (code, _, err) = ladder(&p, &w, &rep, &["--lockbox", STAND_IN, "--retired", &record]);
    assert_eq!(code, 0, "{}", err);
    let json = std::fs::read_to_string(&rep).unwrap();
    assert!(json.contains("\"source\":\"flag\"") && json.contains("\"also\":null"), "{}", json);
    assert!(json.contains(&want_retired), "{}", json);
}

#[test]
fn a_retired_range_that_overlaps_the_new_locked_range_is_refused() {
    let (_s, dir) = scratch("overlap");
    let (_ledger, record) = record(&dir);
    let data = dir.join("d.csv");
    series_csv(&data);
    let (d, run) = (s(&data), s(&dir.join("run.json")));
    for flag in ["2026-01-01,2026-09-01", "2025-01-01,2027-01-01", DEFAULT_RANGE] {
        let (code, _, err) = search(&d, &run, &["--lockbox", flag, "--retired", &record]);
        assert_eq!(code, 2, "{} {}", flag, err);
        assert!(err.contains("overlaps the retired range"), "{}", err);
    }
    // the new range abuts the retired one
    let (code, _, err) = search(&d, &run, &["--lockbox", NEW_RANGE, "--retired", &record]);
    assert_eq!(code, 0, "{}", err);
    // no flag, or none: there is no new locked range
    let (code, _, err) = search(&d, &run, &["--retired", &record]);
    assert_eq!(code, 2, "{}", err);
    let (code, _, err) = search(&d, &run, &["--lockbox", "none", "--retired", &record]);
    assert_eq!(code, 2, "{}", err);
}

#[test]
fn a_store_that_declares_the_retired_range_refuses_the_record() {
    let (_s, dir) = scratch("declared");
    let (_ledger, record) = record(&dir);
    let data = dir.join("d.csv");
    series_csv(&data);
    let (d, run) = (s(&data), s(&dir.join("run.json")));
    lockbox::write_sidecar(&d, &default_range(), "test").unwrap();
    let (code, _, err) = search(&d, &run, &["--lockbox", "2025-01-01,2030-01-01", "--retired", &record]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("overlaps the retired range"), "{}", err);
    let (code, _, err) = search(&d, &run, &["--retired", &record]);
    assert_eq!(code, 2, "{}", err);
}

#[test]
fn a_bad_record_is_refused() {
    let (_s, dir) = scratch("bad-record");
    let data = dir.join("d.csv");
    series_csv(&data);
    let (d, run) = (s(&data), s(&dir.join("run.json")));
    let (code, _, err) = search(&d, &run, &["--lockbox", STAND_IN, "--retired", &s(&dir.join("missing.json"))]);
    assert_eq!(code, 2, "{}", err);
    let junk = dir.join("junk.json");
    std::fs::write(&junk, "{\"schema\":\"other\"}").unwrap();
    let (code, _, err) = search(&d, &run, &["--lockbox", STAND_IN, "--retired", &s(&junk)]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("rcsr-retired/1"), "{}", err);
}
