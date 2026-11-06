//! Windows table, candidate ledger, sealed list and the one-read lockbox.

use rcsr::lockbox::LOCKBOX_START;
use rcsr::protocol::{self, Candidate, Ledger};
use std::path::{Path, PathBuf};
use std::process::Command;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(name);
    let path = scratch.path().to_path_buf();
    (scratch, path)
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
        locked_test: None,
    }
}

#[test]
fn ledger_seals_before_the_read_and_allows_one_read_per_candidate() {
    let (_scratch, dir) = scratch("ledger");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let (a, b, c) = (candidate("a"), candidate("b"), candidate("c"));
    ledger.add_candidate(&a).unwrap();
    ledger.add_candidate(&b).unwrap();
    ledger.add_candidate(&c).unwrap();
    assert_eq!(ledger.verify_chain().unwrap(), 3);

    // nothing sealed yet
    std::fs::write(&sealed, "{}").unwrap();
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build", None).unwrap_err().contains("not sealed"));

    let hash = ledger.seal(&sealed, &[a.id.clone(), b.id.clone()]).unwrap();
    assert_eq!(hash, rcsr::manifest::sha256_file(&sealed).unwrap());
    // c was not sealed
    assert!(ledger.authorize_read(&sealed, &c.id, "data", "build", None).unwrap_err().contains("not in the sealed list"));
    // a different code or data hash voids the run
    assert!(ledger.authorize_read(&sealed, &a.id, "data2", "build", None).unwrap_err().contains("void"));
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build2", None).unwrap_err().contains("void"));

    let entry = ledger.authorize_read(&sealed, &a.id, "data", "build", None).unwrap();
    assert_eq!(entry.expression, a.expression);
    assert_eq!(ledger.record_read(&a.id, "locked-report-a", None).unwrap(), 1);
    // test_read:true cannot be read or selected again
    assert!(ledger.authorize_read(&sealed, &a.id, "data", "build", None).unwrap_err().contains("test_read:true"));
    assert_eq!(ledger.read_report_sha(&a.id).unwrap().as_deref(), Some("locked-report-a"));
    assert_eq!(ledger.record_read(&b.id, "locked-report-b", None).unwrap(), 2);
    assert_eq!(ledger.read_count().unwrap(), 2);
    let unread: Vec<String> = ledger.unread_candidates().unwrap().into_iter().map(|c| c.id).collect();
    assert_eq!(unread, vec![c.id.clone()]);

    // a sealed file edited after the seal no longer matches the ledger
    let text = std::fs::read_to_string(&sealed).unwrap();
    std::fs::write(&sealed, text.replace("report-a", "report-z")).unwrap();
    assert!(ledger.authorize_read(&sealed, &c.id, "data", "build", None).is_err());
}

#[test]
fn an_edited_ledger_breaks_the_chain() {
    let (_scratch, dir) = scratch("chain");
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
    let (_scratch, dir) = scratch("cli");
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
            "walkforward", "--data", &d, "--target", "target", "--expr", "x", "--time", "ts", "--horizon", "1",
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
    let (_scratch, dir) = scratch("selectors");
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

#[test]
fn final_seal_blocks_new_candidates_and_limits_reads() {
    let (_scratch, dir) = scratch("final");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let (a, b) = (candidate("a"), candidate("b"));
    ledger.add_candidate(&a).unwrap();
    ledger.add_candidate(&b).unwrap();
    assert!(!ledger.is_final().unwrap());
    assert!(ledger.seal_final(&sealed, &[], 0, false).is_err());
    let hash = ledger.seal_final(&sealed, &[], 1, false).unwrap();
    assert_eq!(hash, rcsr::manifest::sha256_file(&sealed).unwrap());
    let (seal_hash, terms) = ledger.final_seal().unwrap().unwrap();
    assert_eq!(seal_hash, hash);
    assert_eq!((terms.max_reads, terms.report_candidates), (1, false));

    let refused = ledger.add_candidate(&candidate("c")).unwrap_err();
    assert!(refused.contains("the ledger is final"), "{}", refused);
    assert!(ledger.seal(&sealed, &[]).unwrap_err().contains("the ledger is final"));
    assert!(ledger.seal_final(&sealed, &[], 1, false).unwrap_err().contains("the ledger is final"));

    ledger.authorize_read(&sealed, &a.id, "data", "build", None).unwrap();
    ledger.record_read(&a.id, "report-a", None).unwrap();
    let second = ledger.authorize_read(&sealed, &b.id, "data", "build", None).unwrap_err();
    assert!(second.contains("the final seal allows 1 locked read(s)"), "{}", second);
    ledger.verify_chain().unwrap();
}

#[test]
fn final_seal_with_more_reads_allows_exactly_that_many() {
    let (_scratch, dir) = scratch("final2");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let (a, b, c) = (candidate("a"), candidate("b"), candidate("c"));
    for x in [&a, &b, &c] {
        ledger.add_candidate(x).unwrap();
    }
    ledger.seal_final(&sealed, &[], 2, true).unwrap();
    assert!(ledger.final_seal().unwrap().unwrap().1.report_candidates);
    for x in [&a, &b] {
        ledger.authorize_read(&sealed, &x.id, "data", "build", None).unwrap();
        ledger.record_read(&x.id, "r", None).unwrap();
    }
    assert!(ledger.authorize_read(&sealed, &c.id, "data", "build", None).is_err());
}

#[test]
fn final_seal_refuses_a_ledger_that_has_a_seal_or_a_read() {
    let (_scratch, dir) = scratch("final3");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let a = candidate("a");
    ledger.add_candidate(&a).unwrap();
    ledger.seal(&sealed, &[]).unwrap();
    assert!(ledger.seal_final(&sealed, &[], 1, false).unwrap_err().contains("already has a seal"));
}

#[test]
fn the_read_marker_binds_a_store_to_one_ledger_chain() {
    let (_scratch, dir) = scratch("marker");
    let data = dir.join("d.csv");
    std::fs::write(&data, "ts,x\n").unwrap();
    let d = data.to_str().unwrap();
    let first = Ledger::new(dir.join("l1.jsonl"));
    first.add_candidate(&candidate("a")).unwrap();
    let other = Ledger::new(dir.join("l2.jsonl"));
    other.add_candidate(&candidate("a")).unwrap();

    first.check_read_marker(d).unwrap();
    other.check_read_marker(d).unwrap();
    first.write_read_marker(d, "test_read", "r1").unwrap();
    // a retry on the same head writes no second line
    first.write_read_marker(d, "test_read", "r1").unwrap();
    let marker = std::fs::read_to_string(protocol::marker_path(d)).unwrap();
    assert_eq!(marker.lines().count(), 1, "{}", marker);
    assert!(first.verify_read_marker(d).unwrap_err().contains("that the ledger does not record"));
    first.record_read(&candidate("a").id, "r1", None).unwrap();
    assert!(first.verify_read_marker(d).unwrap().contains("1 read line(s)"));

    let err = other.check_read_marker(d).unwrap_err();
    assert!(err.contains("was already read through ledger"), "{}", err);
    assert!(other.write_read_marker(d, "test_read", "r2").is_err());
    assert_eq!(std::fs::read_to_string(protocol::marker_path(d)).unwrap(), marker);

    // the same path with a different chain is refused
    let path = dir.join("l1.jsonl");
    let text = std::fs::read_to_string(&path).unwrap();
    let mut lines: Vec<&str> = text.lines().collect();
    lines.truncate(1);
    std::fs::write(&path, format!("{}\n", lines[0])).unwrap();
    assert!(first.check_read_marker(d).is_ok(), "the head of the first line is still in the chain");
    let rewritten = Ledger::new(dir.join("l3.jsonl"));
    rewritten.add_candidate(&candidate("zzz")).unwrap();
    std::fs::copy(dir.join("l3.jsonl"), &path).unwrap();
    assert!(first.check_read_marker(d).unwrap_err().contains("does not contain the head"));
}

#[test]
fn a_ledger_binds_the_locked_range_and_a_read_refuses_another() {
    use rcsr::protocol::{Event, LockedRange, RangeBinding};

    let (_scratch, dir) = scratch("bound-range");
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let sealed = dir.join("locked_candidates.json");
    let range = |start: i64, end: i64| LockedRange { start, end, source: "store".to_string() };
    let developed = range(1_751_328_000, 1_782_864_000);
    let (mut a, b) = (candidate("a"), candidate("b"));
    a.locked_test = Some(developed.clone());
    ledger.add_candidate(&a).unwrap();
    ledger.add_candidate(&b).unwrap();
    ledger.seal(&sealed, &[a.id.clone(), b.id.clone()]).unwrap();

    // the range round trips through the ledger line and the sealed list
    let events = ledger.events().unwrap();
    assert!(events.iter().any(|e| matches!(e, Event::Candidate(c) if c.id == a.id && c.locked_test.as_ref() == Some(&developed))));
    assert!(std::fs::read_to_string(&sealed).unwrap().contains("\"locked_test\":{\"start\":1751328000,\"end\":1782864000"));

    fn bind(current: &LockedRange, allow_unbound: bool) -> Option<RangeBinding<'_>> {
        Some(RangeBinding { current, allow_unbound })
    }
    let same = range(developed.start, developed.end);
    ledger.authorize_read(&sealed, &a.id, "data", "build", bind(&same, false)).unwrap();
    // the source name does not matter, the bounds do
    let renamed = LockedRange { source: "flag".to_string(), ..same.clone() };
    ledger.authorize_read(&sealed, &a.id, "data", "build", bind(&renamed, false)).unwrap();
    for other in [range(developed.start, developed.end + 3600), range(developed.start - 3600, developed.end)] {
        let err = ledger.authorize_read(&sealed, &a.id, "data", "build", bind(&other, false)).unwrap_err();
        assert!(err.contains("was developed against the locked test 2025-07-01..2026-07-01 (store)"), "{err}");
        assert!(err.contains("this read resolves"), "{err}");
        let err = ledger.authorize_sealed(&sealed, &a.id, "data", "build", bind(&other, true)).unwrap_err();
        assert!(err.contains("was developed against"), "{err}");
    }

    // an old candidate without a range needs the flag
    let err = ledger.authorize_read(&sealed, &b.id, "data", "build", bind(&same, false)).unwrap_err();
    assert!(err.contains("--allow-unbound-ledger"), "{err}");
    let entry = ledger.authorize_read(&sealed, &b.id, "data", "build", bind(&same, true)).unwrap();
    assert_eq!(entry.locked_test, None);
    // the baseline read checks every sealed item
    let err = ledger.authorize_baseline_read(&sealed, bind(&same, false)).unwrap_err();
    assert!(err.contains("--allow-unbound-ledger"), "{err}");
    ledger.authorize_baseline_read(&sealed, bind(&same, true)).unwrap();
    let other = range(0, 1);
    assert!(ledger.authorize_baseline_read(&sealed, bind(&other, true)).is_err());

    // read lines carry the range
    ledger.record_read(&a.id, "report-a", Some(&same)).unwrap();
    ledger.record_baseline_read("data", "report-base", Some(&same)).unwrap();
    let reads: Vec<Option<LockedRange>> = ledger
        .events()
        .unwrap()
        .into_iter()
        .filter_map(|e| match e {
            Event::TestRead { locked_test, .. } | Event::BaselineRead { locked_test, .. } => Some(locked_test),
            _ => None,
        })
        .collect();
    assert_eq!(reads, vec![Some(same.clone()), Some(same)]);
}

const ZERO: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// The two ledger lines of the issue: a final seal and a locked read.
fn issue_lines() -> (String, String) {
    let l1 = format!(
        "{{\"prev\":\"{ZERO}\",\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"max_reads\":1,\"report_candidates\":false,\"test_read\":false}}"
    );
    let h1 = rcsr::manifest::sha256_hex(l1.as_bytes());
    let l2 = format!(
        "{{\"prev\":\"{h1}\",\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":1,\"test_read\":true}}"
    );
    (l1, l2)
}

fn path_str(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

#[test]
fn verify_fails_on_two_objects_in_one_line() {
    let (_scratch, dir) = scratch("torn-join");
    let (l1, l2) = issue_lines();
    let (good, torn) = (dir.join("good.jsonl"), dir.join("torn.jsonl"));
    std::fs::write(&good, format!("{l1}\n{l2}\n")).unwrap();
    std::fs::write(&torn, format!("{l1}{l2}\n")).unwrap();
    let (code, out, err) = rcsr(&["protocol", "verify", "--ledger", &path_str(&good)]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("ledger ok: 2 line(s), 1 locked read(s)"), "{out}");
    let (code, out, err) = rcsr(&["protocol", "verify", "--ledger", &path_str(&torn)]);
    assert_eq!(code, 1, "{out}{err}");
    assert!(!out.contains("ledger ok"), "{out}");
    assert!(err.contains("line 1"), "{err}");
    assert!(err.contains("not one complete JSON object"), "{err}");
}

#[test]
fn verify_refuses_a_ledger_whose_last_line_has_no_newline() {
    let (_scratch, dir) = scratch("torn-end");
    let (l1, l2) = issue_lines();
    let path = dir.join("cut.jsonl");
    std::fs::write(&path, format!("{l1}\n{l2}")).unwrap();
    let p = path_str(&path);
    for args in [
        vec!["protocol", "verify", "--ledger", p.as_str()],
        vec!["protocol", "candidates", "--ledger", p.as_str()],
    ] {
        let (code, out, err) = rcsr(&args);
        assert_eq!(code, 1, "{out}{err}");
        assert!(err.contains("a write was cut"), "{err}");
        assert!(err.contains("has no newline"), "{err}");
    }
    // a half line is refused the same way
    std::fs::write(&path, format!("{l1}\n{}", &l2[..l2.len() / 2])).unwrap();
    assert_eq!(rcsr(&["protocol", "verify", "--ledger", &p]).0, 1);
    // so is a complete line that is not an object
    std::fs::write(&path, format!("{l1}\nnot json\n")).unwrap();
    let (code, _, err) = rcsr(&["protocol", "verify", "--ledger", &p]);
    assert_eq!(code, 1);
    assert!(err.contains("line 2"), "{err}");
}

#[test]
fn seal_refuses_a_ledger_with_a_cut_last_line() {
    let (_scratch, dir) = scratch("seal-cut");
    let path = dir.join("ledger.jsonl");
    let ledger = Ledger::new(&path);
    ledger.add_candidate(&candidate("a")).unwrap();
    ledger.add_candidate(&candidate("b")).unwrap();
    let mut text = std::fs::read_to_string(&path).unwrap();
    assert!(text.ends_with('\n'));
    text.pop();
    std::fs::write(&path, &text).unwrap();
    let out = dir.join("sealed.json");
    for extra in [vec![], vec!["--final"]] {
        let mut a = vec!["protocol", "seal"];
        a.extend_from_slice(&extra);
        let (l, o) = (path_str(&path), path_str(&out));
        a.extend_from_slice(&["--ledger", &l, "--out", &o]);
        let (code, _, err) = rcsr(&a);
        assert_eq!(code, 1, "{err}");
        assert!(err.contains("a write was cut"), "{err}");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), text, "the file is unchanged");
        assert!(!out.exists());
    }
    assert!(ledger.add_candidate(&candidate("c")).unwrap_err().contains("a write was cut"));
    assert_eq!(std::fs::read_to_string(&path).unwrap(), text);
}

#[test]
fn verify_accepts_the_existing_ledger_format() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/protocol");
    let ledger = dir.join("old_ledger.jsonl");
    let (code, out, err) = rcsr(&["protocol", "verify", "--ledger", &path_str(&ledger)]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.contains("ledger ok: 3 line(s), 1 locked read(s)"), "{out}");
    // the fixture keeps its terminated lines and hash chain: a new line extends it
    let (_scratch, tmp) = scratch("old-ledger");
    let copy = tmp.join("old_ledger.jsonl");
    std::fs::copy(&ledger, &copy).unwrap();
    assert_eq!(Ledger::new(&copy).verify_chain().unwrap(), 3);
    let (l1, l2) = issue_lines();
    let issue = tmp.join("issue.jsonl");
    std::fs::write(&issue, format!("{l1}\n{l2}\n")).unwrap();
    assert_eq!(Ledger::new(&issue).verify_chain().unwrap(), 2);
}

#[test]
fn concurrent_appenders_keep_one_chain_of_whole_lines() {
    let (_scratch, dir) = scratch("concurrent");
    let path = dir.join("ledger.jsonl");
    let threads: Vec<_> = (0..6)
        .map(|t| {
            let path = path.clone();
            std::thread::spawn(move || {
                let ledger = Ledger::new(&path);
                for k in 0..8 {
                    ledger.add_candidate(&candidate(&format!("{t}-{k}"))).unwrap();
                }
            })
        })
        .collect();
    for t in threads {
        t.join().unwrap();
    }
    let ledger = Ledger::new(&path);
    assert_eq!(ledger.verify_chain().unwrap(), 48);
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.ends_with('\n'));
    assert_eq!(text.lines().count(), 48);
}

#[test]
fn the_search_refuses_a_ledger_with_a_cut_last_line_before_it_starts() {
    let (_scratch, dir) = scratch("search-cut");
    let data = dir.join("d.csv");
    let mut csv = String::from("ts,x,y\n");
    for i in 0..100 {
        csv.push_str(&format!("{},{},{}\n", i, (i * 7) % 11, (i * 3) % 5));
    }
    std::fs::write(&data, csv).unwrap();
    let (l1, _) = issue_lines();
    let ledger = dir.join("cut.jsonl");
    std::fs::write(&ledger, &l1).unwrap();
    let (d, l) = (path_str(&data), path_str(&ledger));
    let (code, _, err) = rcsr(&[
        "--data", &d, "--target", "y", "--vars", "x", "--time", "ts", "--cpu", "--no-dev-log", "--seed", "3",
        "--generations", "2", "--pop", "10", "--quiet", "--ledger", &l,
    ]);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("a write was cut"), "{err}");
    assert_eq!(std::fs::read_to_string(&ledger).unwrap(), l1);
}

fn tmp_of(p: &Path) -> PathBuf {
    PathBuf::from(format!("{}.tmp", p.display()))
}

#[test]
fn a_kill_at_each_point_of_a_sealed_write_has_a_defined_state_that_a_retry_handles() {
    use protocol::{write_around_record, CrashAt};
    for crash in [CrashAt::AfterStage, CrashAt::AfterRecord, CrashAt::AfterFirstRename] {
        let (_scratch, dir) = scratch("crash");
        let (a, b) = (dir.join("a.json"), dir.join("b.md"));
        std::fs::write(&b, "old b").unwrap();
        let recorded = std::cell::Cell::new(0);
        let files: [(&Path, &[u8]); 2] = [(&a, b"new a"), (&b, b"new b")];
        let err = write_around_record(&files, || { recorded.set(recorded.get() + 1); Ok(()) }, Some(crash)).unwrap_err();
        assert!(err.contains("simulated kill"), "{err}");
        match crash {
            CrashAt::AfterStage => {
                assert_eq!(recorded.get(), 0, "no record before the files are staged");
                assert!(!a.exists());
                assert_eq!(std::fs::read_to_string(&b).unwrap(), "old b");
            }
            CrashAt::AfterRecord => {
                assert_eq!(recorded.get(), 1);
                assert!(!a.exists());
                assert_eq!(std::fs::read_to_string(&b).unwrap(), "old b");
            }
            CrashAt::AfterFirstRename => {
                assert_eq!(recorded.get(), 1);
                assert_eq!(std::fs::read_to_string(&a).unwrap(), "new a");
                assert_eq!(std::fs::read_to_string(&b).unwrap(), "old b");
                assert!(!tmp_of(&a).exists() && tmp_of(&b).exists());
            }
        }
        if crash != CrashAt::AfterFirstRename {
            assert_eq!(std::fs::read_to_string(tmp_of(&a)).unwrap(), "new a");
            assert_eq!(std::fs::read_to_string(tmp_of(&b)).unwrap(), "new b");
        }
        // the retry: after the record the verify route records nothing; before it, a plain rerun records once
        let again = std::cell::Cell::new(0);
        write_around_record(&files, || { again.set(again.get() + 1); Ok(()) }, None).unwrap();
        assert_eq!(again.get(), 1);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "new a");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "new b");
        assert!(!tmp_of(&a).exists() && !tmp_of(&b).exists());
    }
}

#[test]
fn a_sealed_write_that_fails_before_the_record_records_nothing_and_leaves_no_staged_file() {
    use protocol::write_around_record;
    let (_scratch, dir) = scratch("no-record");
    let good = dir.join("good.json");
    let as_dir = dir.join("is-a-dir");
    std::fs::create_dir(&as_dir).unwrap();
    let called = std::cell::Cell::new(false);
    let files: [(&Path, &[u8]); 2] = [(&good, b"x"), (&as_dir, b"y")];
    let err = write_around_record(&files, || { called.set(true); Ok(()) }, None).unwrap_err();
    assert!(err.contains("is a directory") && err.contains("still available"), "{err}");
    assert!(!called.get());
    assert!(!good.exists() && !tmp_of(&good).exists());
    // a record that fails removes the staged files as well
    let files: [(&Path, &[u8]); 1] = [(&good, b"x")];
    let err = write_around_record::<()>(&files, || Err("ledger refused".to_string()), None).unwrap_err();
    assert_eq!(err, "ledger refused");
    assert!(!good.exists() && !tmp_of(&good).exists());
    // two outputs naming one file are refused before anything is staged
    let files: [(&Path, &[u8]); 2] = [(&good, b"x"), (&good, b"y")];
    let err = write_around_record(&files, || Ok(()), None).unwrap_err();
    assert!(err.contains("same file"), "{err}");
    assert!(!tmp_of(&good).exists());
}

#[test]
fn staged_write_crash_points_on_a_ledger_appended_under_the_lock() {
    use protocol::{write_around_record, CrashAt};
    for crash in [CrashAt::AfterStage, CrashAt::AfterRecord, CrashAt::AfterFirstRename] {
        let (_scratch, dir) = scratch("staged-ledger");
        let (a, b) = (dir.join("a.json"), dir.join("b.md"));
        let lp = dir.join("ledger.jsonl");
        let ledger = Ledger::new(&lp);
        ledger.add_candidate(&candidate("a")).unwrap();
        let files: [(&Path, &[u8]); 2] = [(&a, b"new a"), (&b, b"new b")];
        let record = || ledger.record_read(&candidate("a").id, "hash", None);
        let err = write_around_record(&files, record, Some(crash)).unwrap_err();
        assert!(err.contains("simulated kill"), "{err}");
        let reads = if crash == CrashAt::AfterStage { 0 } else { 1 };
        assert_eq!(ledger.read_count().unwrap(), reads, "{crash:?}");
        assert!(std::fs::read_to_string(&lp).unwrap().ends_with('\n'));
        assert_eq!(ledger.verify_chain().unwrap(), 1 + reads as usize);
        // retry: a plain rerun records once; after the record the verify route records nothing
        if reads == 0 {
            write_around_record(&files, record, None).unwrap();
        } else {
            write_around_record(&files, || Ok(0), None).unwrap();
        }
        assert_eq!(ledger.read_count().unwrap(), 1);
        assert_eq!(ledger.verify_chain().unwrap(), 2);
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "new a");
        assert_eq!(std::fs::read_to_string(&b).unwrap(), "new b");
        assert!(!tmp_of(&a).exists() && !tmp_of(&b).exists());
    }
}

#[test]
fn a_staged_write_reaches_its_record_while_the_ledger_lock_is_held_and_completes_after_release() {
    use protocol::write_around_record;
    use std::sync::mpsc;
    let (_scratch, dir) = scratch("staged-lock");
    let (a, lp) = (dir.join("a.json"), dir.join("ledger.jsonl"));
    let ledger = Ledger::new(&lp);
    ledger.add_candidate(&candidate("a")).unwrap();
    let holder = std::fs::OpenOptions::new().read(true).append(true).open(&lp).unwrap();
    holder.lock().unwrap();
    let (about_to_record, recorded_signal) = mpsc::channel();
    let id = candidate("a").id;
    let writer = std::thread::spawn({
        let (a, lp) = (a.clone(), lp.clone());
        move || {
            let ledger = Ledger::new(&lp);
            let files: [(&Path, &[u8]); 1] = [(&a, b"new a")];
            write_around_record(
                &files,
                || {
                    about_to_record.send(()).unwrap();
                    ledger.record_read(&id, "hash", None)
                },
                None,
            )
        }
    });
    // staging finished without the ledger lock; only now is the lock released
    recorded_signal.recv().unwrap();
    assert!(!a.exists(), "the rename waits for the record, which waits for the lock");
    // the lock is still held: an append that got through would show here. `concurrent_appenders_keep_one_chain_of_whole_lines` proves the lock itself.
    assert_eq!(ledger.verify_chain().unwrap(), 1, "the ledger is unchanged while the lock is held");
    assert_eq!(ledger.read_count().unwrap(), 0);
    holder.unlock().unwrap();
    assert_eq!(writer.join().unwrap().unwrap(), 1);
    assert_eq!(std::fs::read_to_string(&a).unwrap(), "new a");
    assert_eq!(ledger.verify_chain().unwrap(), 2);
}

#[test]
fn outputs_that_reach_one_file_are_refused_before_staging_or_recording() {
    use protocol::write_around_record;
    let (_scratch, dir) = scratch("same-file");
    std::fs::create_dir(dir.join("b")).unwrap();
    let a = dir.join("a.json");
    std::fs::write(&a, "old").unwrap();
    std::os::unix::fs::symlink(&a, dir.join("sym.json")).unwrap();
    std::fs::hard_link(&a, dir.join("hard.json")).unwrap();
    let others = [
        dir.join(".").join("a.json"),
        dir.join("b").join("..").join("a.json"),
        dir.join("sym.json"),
        dir.join("hard.json"),
    ];
    for other in others {
        let called = std::cell::Cell::new(false);
        let files: [(&Path, &[u8]); 2] = [(&a, b"x"), (&other, b"y")];
        let err = write_around_record(&files, || { called.set(true); Ok(()) }, None).unwrap_err();
        assert!(err.contains("same file"), "{}: {err}", other.display());
        assert!(!called.get(), "{}", other.display());
        assert_eq!(std::fs::read_to_string(&a).unwrap(), "old");
        for entry in std::fs::read_dir(&dir).unwrap() {
            let name = entry.unwrap().file_name().to_string_lossy().into_owned();
            assert!(!name.ends_with(".tmp"), "{name} left behind for {}", other.display());
        }
    }
    // different files in one directory, and a target that does not exist yet, still work
    let (c, d) = (dir.join("c.json"), dir.join("b").join("..").join("d.md"));
    let files: [(&Path, &[u8]); 2] = [(&c, b"x"), (&d, b"y")];
    write_around_record(&files, || Ok(()), None).unwrap();
    assert_eq!(std::fs::read_to_string(dir.join("d.md")).unwrap(), "y");
}

/// A ledger of the given line bodies, each chained to the hash of the line before.
fn chained_ledger(bodies: &[&str]) -> String {
    let mut prev = ZERO.to_string();
    let mut text = String::new();
    for body in bodies {
        let line = format!("{{\"prev\":\"{prev}\",{body}}}");
        prev = rcsr::manifest::sha256_hex(line.as_bytes());
        text.push_str(&line);
        text.push('\n');
    }
    text
}

const FINAL_SEAL: &str = "\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"max_reads\":1,\"report_candidates\":false,\"test_read\":false";

#[test]
fn an_event_without_the_fields_a_read_depends_on_is_a_ledger_fault() {
    let (_scratch, dir) = scratch("event-fields");
    let path = dir.join("ledger.jsonl");
    let faults: [(&str, &str); 22] = [
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"test_read\":true"),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":\"1\",\"test_read\":true"),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":1.5,\"test_read\":true"),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":-1,\"test_read\":true"),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":null,\"test_read\":true"),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":,\"test_read\":true"),
        ("id", "\"event\":\"test_read\",\"report_sha256\":\"bb\",\"read_count\":1,\"test_read\":true"),
        ("id", "\"event\":\"test_read\",\"id\":\"\",\"report_sha256\":\"bb\",\"read_count\":1,\"test_read\":true"),
        ("report_sha256", "\"event\":\"test_read\",\"id\":\"c\",\"read_count\":1,\"test_read\":true"),
        ("report_sha256", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"\",\"read_count\":1,\"test_read\":true"),
        ("report_sha256", "\"event\":\"baseline_read\",\"data_sha256\":\"dd\",\"test_read\":true"),
        ("report_sha256", "\"event\":\"baseline_read\",\"data_sha256\":\"dd\",\"report_sha256\":\"\",\"test_read\":true"),
        ("data_sha256", "\"event\":\"baseline_read\",\"report_sha256\":\"bb\",\"test_read\":true"),
        ("sealed_sha256", "\"event\":\"seal\",\"final\":true,\"max_reads\":1,\"report_candidates\":false,\"test_read\":false"),
        ("sealed_sha256", "\"event\":\"seal\",\"sealed_sha256\":\"\",\"test_read\":false"),
        ("max_reads", "\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"report_candidates\":false,\"test_read\":false"),
        ("max_reads", "\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"max_reads\":0,\"test_read\":false"),
        ("max_reads", "\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"max_reads\":2.5,\"test_read\":false"),
        ("max_reads", "\"event\":\"seal\",\"sealed_sha256\":\"aa\",\"final\":true,\"max_reads\":\"1\",\"test_read\":false"),
        ("id", "\"event\":\"candidate\",\"spec_sha256\":\"s\",\"data_sha256\":\"d\",\"build_commit\":\"b\",\"report_sha256\":\"r\",\"test_read\":false,\"expression\":\"x\""),
        ("id", "\"event\":\"candidate\",\"id\":\"\",\"spec_sha256\":\"s\",\"data_sha256\":\"d\",\"build_commit\":\"b\",\"report_sha256\":\"r\",\"test_read\":false,\"expression\":\"x\""),
        ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":99999999999999999999,\"test_read\":true"),
    ];
    for (key, bad) in faults {
        std::fs::write(&path, chained_ledger(&[FINAL_SEAL, bad])).unwrap();
        let ledger = Ledger::new(&path);
        let err = ledger.read_count().unwrap_err();
        assert!(err.contains("line 2") && err.contains(&format!("without a valid {key}")), "{bad}: {err}");
        assert!(ledger.events().is_err(), "{bad}");
        let (code, _, err) = rcsr(&["protocol", "verify", "--ledger", &path_str(&path)]);
        assert_eq!(code, 1, "{bad}: {err}");
        assert!(err.contains(&format!("without a valid {key}")), "{bad}: {err}");
    }
}

#[test]
fn a_marker_line_without_its_head_or_report_is_refused() {
    let (_scratch, dir) = scratch("marker-fields");
    let data = path_str(&dir.join("d.csv"));
    let ledger_path = dir.join("ledger.jsonl");
    std::fs::write(&ledger_path, "").unwrap();
    let ledger = Ledger::new(&ledger_path);
    let abs = std::fs::canonicalize(&ledger_path).unwrap();
    let marker = protocol::marker_path(&data);
    let line = |head: Option<&str>, report: Option<&str>| {
        let mut s = format!("{{\"ledger\":\"{}\"", abs.display());
        if let Some(h) = head {
            s.push_str(&format!(",\"ledger_head\":\"{h}\""));
        }
        s.push_str(",\"event\":\"test_read\"");
        if let Some(r) = report {
            s.push_str(&format!(",\"report_sha256\":\"{r}\""));
        }
        s.push_str("}\n");
        s
    };
    std::fs::write(&marker, line(Some(ZERO), Some("bb"))).unwrap();
    ledger.check_read_marker(&data).unwrap();
    for bad in [line(Some(ZERO), None), line(None, Some("bb")), line(Some(ZERO), Some("")), line(Some(""), Some("bb"))] {
        std::fs::write(&marker, &bad).unwrap();
        let err = ledger.check_read_marker(&data).unwrap_err();
        assert!(err.contains("malformed read marker line"), "{bad}: {err}");
    }
}

#[test]
fn a_stat_error_other_than_a_missing_file_refuses_the_output() {
    use protocol::write_around_record;
    use std::os::unix::fs::PermissionsExt;
    let (_scratch, dir) = scratch("stat-error");
    let locked = dir.join("locked");
    std::fs::create_dir(&locked).unwrap();
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();
    let probe = std::fs::metadata(locked.join("a.json"));
    let target = locked.join("a.json");
    let called = std::cell::Cell::new(false);
    let files: [(&Path, &[u8]); 1] = [(&target, b"x")];
    let result = write_around_record(&files, || { called.set(true); Ok(()) }, None);
    std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert_eq!(probe.unwrap_err().kind(), std::io::ErrorKind::PermissionDenied, "the test needs a non-root user");
    let err = result.unwrap_err();
    assert!(err.contains("cannot stat") && err.contains("nothing was recorded"), "{err}");
    assert!(!called.get());
    assert_eq!(std::fs::read_dir(&locked).unwrap().count(), 0);
}
