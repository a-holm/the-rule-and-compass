//! Windows table, candidate ledger, sealed list and the one-read lockbox.

use rcsr::lockbox::LOCKBOX_START;
use rcsr::protocol::{self, Candidate, Ledger};
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-protocol-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

#[test]
fn windows_table_is_the_committed_file_and_obeys_the_protocol() {
    let committed = std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join("../protocol/windows.csv")).unwrap();
    assert_eq!(committed, protocol::windows_csv(protocol::DEFAULT_PURGE_SECONDS));
    let rows = protocol::parse_windows_csv(&committed, None).unwrap();
    assert_eq!(rows.iter().filter(|r| r.scheme == "expanding").count(), 10);
    assert_eq!(rows.iter().filter(|r| r.scheme == "rolling").count(), 6);
    assert_eq!(rows.iter().filter(|r| r.scheme == "locked").count(), 1);
    let dev_start = rcsr::data::parse_time("2022-01-01").unwrap();
    let dev_end = rcsr::data::parse_time("2025-07-01").unwrap();
    for r in &rows {
        let w = &r.window;
        assert_eq!(w.test_start - w.train_end, protocol::DEFAULT_PURGE_SECONDS, "{}", w.label);
        assert_eq!(w.test_start % 86_400.0, 0.0, "{}", w.label);
        assert!(w.train_start >= dev_start, "warm-up 2021 is never trained on");
        assert!(w.test_end <= LOCKBOX_START as f64, "{} reaches 2026H2", w.label);
        assert!(r.validation_start > w.train_start && r.validation_start < w.train_end);
        if r.scheme != "locked" {
            assert!(
                w.test_end <= dev_end - protocol::DEFAULT_PURGE_SECONDS,
                "{} has a label that reaches the locked test",
                w.label
            );
        } else {
            assert_eq!(w.test_start, dev_end);
            assert_eq!(w.test_end, rcsr::data::parse_time("2026-07-01").unwrap());
        }
    }
    for label in ["expanding-2025Q2", "rolling24-2025Q2"] {
        let w = &rows.iter().find(|r| r.window.label == label).unwrap().window;
        assert_eq!(protocol::iso(w.test_end), "2025-06-28T11:00:00", "{}", label);
    }
    // consecutive quarters inside each scheme
    for scheme in ["expanding", "rolling"] {
        let rs = protocol::parse_windows_csv(&committed, Some(scheme)).unwrap();
        for pair in rs.windows(2) {
            assert_eq!(pair[0].window.test_end, pair[1].window.test_start);
        }
    }
    // expanding folds start after 12 months of training, rolling after 24 with a 24 month window
    let first = &protocol::parse_windows_csv(&committed, Some("expanding")).unwrap()[0];
    assert_eq!(first.window.test_start, rcsr::data::parse_time("2023-01-01").unwrap());
    let roll = protocol::parse_windows_csv(&committed, Some("rolling")).unwrap();
    assert_eq!(roll[0].window.train_start, dev_start);
    assert_eq!(roll[1].window.train_start, rcsr::data::parse_time("2022-04-01").unwrap());
    assert!(protocol::parse_windows_csv(&committed, Some("nope")).is_err());
    assert!(protocol::is_windows_table(&committed));
    assert!(!protocol::is_windows_table("w,0,1,2,3\n"));
}

fn candidate(tag: &str) -> Candidate {
    let spec = format!("spec-{}", tag);
    Candidate {
        id: Candidate::address(&spec, "data", "build"),
        spec_sha256: spec,
        data_sha256: "data".to_string(),
        build_commit: "build".to_string(),
        report_sha256: format!("report-{}", tag),
        expression: format!("x + \"{}\"", tag),
    }
}

#[test]
fn ledger_seals_before_the_read_and_allows_one_read_per_candidate() {
    let dir = scratch("ledger");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let (a, b, c) = (candidate("a"), candidate("b"), candidate("c"));
    ledger.add_candidate(&a).unwrap();
    ledger.add_candidate(&b).unwrap();
    ledger.add_candidate(&c).unwrap();
    assert_eq!(ledger.verify_chain().unwrap(), 3);

    // nothing sealed yet
    std::fs::write(&sealed, "{}").unwrap();
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build").unwrap_err().contains("not sealed"));

    let hash = ledger.seal(&sealed, &[a.id.clone(), b.id.clone()]).unwrap();
    assert_eq!(hash, rcsr::manifest::sha256_file(&sealed).unwrap());
    // c was not sealed
    assert!(ledger.authorize_read(&sealed, &c.id, "data", "build").unwrap_err().contains("not in the sealed list"));
    // a different code or data hash voids the run
    assert!(ledger.authorize_read(&sealed, &a.id, "data2", "build").unwrap_err().contains("void"));
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build2").unwrap_err().contains("void"));

    let entry = ledger.authorize_read(&sealed, &a.id, "data", "build").unwrap();
    assert_eq!(entry.expression, a.expression);
    assert_eq!(ledger.record_read(&a.id, "locked-report-a").unwrap(), 1);
    // test_read:true cannot be read or selected again
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build").unwrap_err().contains("test_read:true"));
    assert_eq!(ledger.read_report_sha(&a.id).as_deref(), Some("locked-report-a"));
    assert_eq!(ledger.record_read(&b.id, "locked-report-b").unwrap(), 2);
    assert_eq!(ledger.read_count().unwrap(), 2);
    let unread: Vec<String> = ledger.unread_candidates().unwrap().into_iter().map(|c| c.id).collect();
    assert_eq!(unread, vec![c.id.clone()]);

    // a sealed file edited after the seal no longer matches the ledger
    let text = std::fs::read_to_string(&sealed).unwrap();
    std::fs::write(&sealed, text.replace("report-a", "report-z")).unwrap();
    assert!(ledger.authorize_read(&sealed, &c.id, "data", "build").is_err());
}

#[test]
fn an_edited_ledger_breaks_the_chain() {
    let dir = scratch("chain");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    ledger.add_candidate(&candidate("a")).unwrap();
    ledger.add_candidate(&candidate("b")).unwrap();
    let path = dir.join("ledger.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    std::fs::write(&path, text.replacen("report-a", "report-edited", 1)).unwrap();
    assert!(ledger.verify_chain().unwrap_err().contains("breaks the hash chain"));
    assert!(ledger.add_candidate(&candidate("c")).is_err());
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

#[test]
fn the_cli_runs_the_whole_protocol_and_refuses_a_second_read() {
    let dir = scratch("cli");
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,target\n");
    for i in 0..300 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", i, x, y));
    }
    std::fs::write(&data, &csv).unwrap();
    let (d, l, s) = (
        data.to_str().unwrap().to_string(),
        dir.join("ledger.jsonl").to_str().unwrap().to_string(),
        dir.join("locked_candidates.json").to_str().unwrap().to_string(),
    );
    let base = |extra: &[&str]| -> (i32, String, String) {
        let mut a = vec![
            "walkforward", "--data", &d, "--target", "target", "--expr", "x", "--time", "ts", "--horizon", "5",
            "--cost-bps", "5", "--lockbox", "200,300", "--ledger", &l, "--periods-per-year", "365", "--json",
        ];
        a.extend_from_slice(extra);
        rcsr(&a)
    };
    // a development window that touches the declared lockbox is refused
    let (code, _, err) = base(&["--window", "bad,0,100,110,210"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("locked test"), "{}", err);
    // the development run is recorded
    let (code, out, err) = base(&["--window", "dev,0,60,70,150"]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"lockbox\":{\"start\":200,\"end\":300"), "{}", out);
    assert!(out.contains("\"read_count\":0"));
    // a locked read before the seal is refused
    let locked = ["--locked-test", "--sealed", &s, "--window", "locked,0,150,210,300"];
    std::fs::write(&s, "{}").unwrap();
    let (code, _, err) = base(&locked);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("not sealed"), "{}", err);
    let (code, out, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &s]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.starts_with("sealed_sha256 "));
    // training into the lockbox is refused even with the seal
    let (code, _, err) = base(&["--locked-test", "--sealed", &s, "--window", "locked,0,250,260,300"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("trains within the label purge of the locked test"), "{}", err);
    // the locked read
    let (code, out, err) = base(&locked);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"test_read\":true"), "{}", out);
    assert!(out.contains("\"read_count\":1"), "{}", out);
    assert!(out.contains("\"first_date\""), "{}", out);
    let hash = out.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    // a second read is refused
    let (code, _, err) = base(&locked);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("already has test_read:true"), "{}", err);
    // the reproduce command matches the recorded hash and records nothing
    let (code, out, err) = base(&["--locked-test", "--verify-read", "--sealed", &s, "--window", "locked,0,150,210,300", "--expect-report-sha256", &hash]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"read_count\":1"));
    let (code, _, _) = base(&["--locked-test", "--verify-read", "--sealed", &s, "--window", "locked,0,150,210,300", "--expect-report-sha256", &"0".repeat(64)]);
    assert_eq!(code, 3);
    let (code, out, _) = rcsr(&["protocol", "verify", "--ledger", &l]);
    assert_eq!(code, 0);
    assert!(out.contains("1 locked read(s)"), "{}", out);
    // changed data is a different candidate: the sealed list does not know it
    std::fs::write(&data, csv.replacen("\n0,", "\n0.5,", 1)).unwrap();
    let (code, _, err) = base(&locked);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("not in the sealed list"), "{}", err);
}

#[test]
fn generated_selectors_leave_the_required_purge_on_a_ragged_panel() {
    let dir = scratch("selectors");
    let purge = 3.0 * 3600.0;
    let t0 = 1_600_000_000.0f64;
    let hour = 3600.0;
    // one window: train [t0, t0 + 120 h - purge), test [t0 + 120 h, t0 + 160 h)
    let test_start = t0 + 120.0 * hour;
    let table = format!(
        "scheme,label,train_start,train_end,validation_start,test_start,test_end\n\
         expanding,w1,{},{},{},{},{}\n",
        t0,
        test_start - purge,
        t0 + 60.0 * hour,
        test_start,
        t0 + 160.0 * hour
    );
    let windows = dir.join("w.csv");
    std::fs::write(&windows, table).unwrap();
    // symbol 0 hourly, symbol 1 listed late, symbol 2 has a hole
    let mut csv = String::from("ts,sym,x,y\n");
    for h in 0..200 {
        let ts = t0 + h as f64 * hour;
        let x = ((h * 7) % 13) as f64 / 13.0;
        csv.push_str(&format!("{},0,{},{}\n", ts, x, 0.5 * x));
        if h >= 110 {
            csv.push_str(&format!("{},1,{},{}\n", ts, x, 0.5 * x));
        }
        if !(30..50).contains(&h) {
            csv.push_str(&format!("{},2,{},{}\n", ts, x, 0.5 * x));
        }
    }
    let data = dir.join("d.csv");
    std::fs::write(&data, csv).unwrap();
    let (code, out, err) = rcsr(&[
        "protocol", "selectors", "--windows", windows.to_str().unwrap(), "--label", "w1", "--time-col", "ts",
    ]);
    assert_eq!(code, 0, "{}", err);
    let tr = out.split("--train-select '").nth(1).unwrap().split('\'').next().unwrap().to_string();
    let te = out.split("--test-select '").nth(1).unwrap().split('\'').next().unwrap().to_string();
    let (code, run, err) = rcsr(&[
        "--data", data.to_str().unwrap(), "--target", "y", "--vars", "x", "--cpu", "--no-dev-log", "--seed", "1",
        "--generations", "4", "--pop", "30", "--metric", "squared", "--json", "--quiet", "--time", "ts",
        "--train-select", &tr, "--test-select", &te, "--lockbox", "1700000000,1700003600",
    ]);
    assert_eq!(code, 0, "{}", err);
    let j = rcsr::devlog::Json::parse(run.trim()).unwrap();
    let split = j.get("split").unwrap();
    assert!(split.get("purge_seconds").unwrap().as_f64().unwrap() >= purge);
    // run.json carries the lockbox object: rows, date range, read count
    let lb = j.get("lockbox").unwrap();
    assert_eq!(lb.get("start").unwrap().as_f64(), Some(1_700_000_000.0));
    assert_eq!(lb.get("rows").unwrap().as_f64(), Some(0.0));
    assert_eq!(lb.get("read_count").unwrap().as_f64(), Some(0.0));
    assert!(lb.get("start_date").is_some() && lb.get("end_date").is_some());
}
