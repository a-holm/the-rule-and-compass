//! #523: a final seal allows one locked read for one candidate, a read marker
//! beside the store refuses a second ledger, and a final read reports the
//! selected candidate only. Fixtures are synthetic; the locked range is a
//! stand-in in 2024.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::lockbox::{self, Lockbox};
use std::path::{Path, PathBuf};

/// 2024-01-01T00:00:00Z.
const T0: i64 = 1_704_067_200;
const H: i64 = 3600;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(name);
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

/// The children make no device call: `RCSR_GPU=0` skips even the startup device probe.
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

fn window(label: &str, a: i64, b: i64, c: i64, d: i64) -> String {
    format!("{},{},{},{},{}", label, T0 + a * H, T0 + b * H, T0 + c * H, T0 + d * H)
}

/// A store with its declared locked test (hours 300..400) beside it.
fn store(dir: &Path, name: &str) -> String {
    store_rows(dir, name, 400)
}

/// `store` cut to `rows` hours; the declared lockbox stays at hours 300..400.
fn store_rows(dir: &Path, name: &str, rows: i64) -> String {
    let path = dir.join(name);
    let mut csv = String::from("ts,x,y\n");
    for i in 0..rows {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", T0 + i * H, x, y));
    }
    std::fs::write(&path, csv).unwrap();
    let d = s(&path);
    lockbox::write_sidecar(&d, &Lockbox { start: T0 + 300 * H, end: T0 + 400 * H }, "test").unwrap();
    d
}

fn wf(data: &str, ledger: &str, exprs: &[&str], extra: &[&str]) -> (i32, String, String) {
    let mut a = vec![
        "walkforward", "--data", data, "--target", "y", "--time", "ts", "--horizon", "5", "--cost-bps", "5",
        "--periods-per-year", "365", "--json", "--ledger", ledger,
    ];
    for e in exprs {
        a.extend_from_slice(&["--expr", e]);
    }
    a.extend_from_slice(extra);
    rcsr(&a)
}

fn dev(data: &str, ledger: &str, exprs: &[&str], extra: &[&str]) {
    let w = window("dev", 0, 150, 160, 250);
    let mut a = vec!["--window", w.as_str()];
    a.extend_from_slice(extra);
    let (code, _, err) = wf(data, ledger, exprs, &a);
    assert_eq!(code, 0, "{}", err);
}

fn read(data: &str, ledger: &str, sealed: &str, exprs: &[&str], extra: &[&str]) -> (i32, String, String) {
    let w = window("locked", 0, 250, 300, 400);
    let mut a = vec!["--locked-test", "--sealed", sealed, "--window", w.as_str()];
    a.extend_from_slice(extra);
    wf(data, ledger, exprs, &a)
}

fn p_values(json: &str) -> Vec<String> {
    let body = json.split("\"p_values\":[").nth(1).unwrap().split(']').next().unwrap();
    body.split(',').filter(|v| !v.is_empty()).map(str::to_string).collect()
}

#[test]
fn final_ledger_refuses_new_candidates() {
    let (_scratch, dir) = scratch("refuse");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    let (code, out, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.starts_with("sealed_sha256 "), "{}", out);
    let line = std::fs::read_to_string(&l).unwrap().lines().last().unwrap().to_string();
    assert!(line.contains("\"final\":true") && line.contains("\"max_reads\":1"), "{}", line);

    // a development run is refused
    let w = window("dev2", 0, 150, 160, 250);
    let (code, _, err) = wf(&data, &l, &["x"], &["--window", &w]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("the ledger is final"), "{}", err);
    let (code, _, err) = wf(&data, &l, &["x*2"], &["--window", &w]);
    assert_eq!(code, 3, "{}", err);
    // neither seal form works again
    let again = s(&dir.join("sealed2.json"));
    for extra in [&[][..], &["--final"][..]] {
        let mut a = vec!["protocol", "seal", "--ledger", &l, "--out", &again];
        a.extend_from_slice(extra);
        let (code, _, err) = rcsr(&a);
        assert_eq!(code, 1, "{}", err);
        assert!(err.contains("the ledger is final"), "{}", err);
    }
    assert!(!Path::new(&again).exists());
    let (code, out, _) = rcsr(&["protocol", "verify", "--ledger", &l]);
    assert_eq!(code, 0);
    assert!(out.contains("2 line(s), 0 locked read(s)"), "{}", out);
}

#[test]
fn final_seal_needs_a_ledger_without_a_seal_or_a_read() {
    let (_scratch, dir) = scratch("fresh");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 1, "{}", err);
    assert!(err.contains("already has a seal"), "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed, "--max-reads", "2"]);
    assert_eq!(code, 2, "{}", err);
}

#[test]
fn final_ledger_reads_once() {
    let (_scratch, dir) = scratch("once");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    dev(&data, &l, &["x*2"], &[]);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let verify = |n: &str, with_sealed: bool| {
        let mut a = vec!["protocol", "verify", "--ledger", l.as_str(), "--data", data.as_str(), "--expect-reads", n];
        if with_sealed {
            a.extend_from_slice(&["--sealed", sealed.as_str()]);
        }
        rcsr(&a)
    };
    // before the read the gate for one read fails; the gate for zero passes
    assert_eq!(verify("1", false).0, 1);
    assert_eq!(verify("0", false).0, 0);

    let (code, out, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"read_count\":1"), "{}", out);
    let marker = format!("{}.locked_reads.jsonl", data);
    let mtext = std::fs::read_to_string(&marker).unwrap();
    assert_eq!(mtext.lines().count(), 1, "{}", mtext);
    assert!(mtext.contains("\"event\":\"test_read\""), "{}", mtext);

    // the other sealed candidate is refused: the final seal allows one read
    let (code, _, err) = read(&data, &l, &sealed, &["x*2"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("the final seal allows 1 locked read(s)"), "{}", err);
    // the same candidate is refused too
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);

    let (code, out, err) = verify("1", true);
    assert_eq!(code, 0, "{}{}", out, err);
    assert!(out.contains("consistent with the ledger"), "{}", out);
    assert!(out.contains("every read candidate is in the final sealed list"), "{}", out);
    assert_eq!(verify("2", false).0, 1);
    assert_eq!(verify("0", false).0, 1);
    // --expect-reads without --data is a usage error
    assert_eq!(rcsr(&["protocol", "verify", "--ledger", &l, "--expect-reads", "1"]).0, 2);

    // a fresh ledger cannot read the same store again
    let (l2, sealed2) = (s(&dir.join("ledger2.jsonl")), s(&dir.join("sealed2.json")));
    dev(&data, &l2, &["x"], &[]);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l2, "--out", &sealed2]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = read(&data, &l2, &sealed2, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("was already read through ledger"), "{}", err);
    let (code, out, _) = rcsr(&["protocol", "verify", "--ledger", &l2]);
    assert_eq!(code, 0);
    assert!(out.contains("0 locked read(s)"), "{}", out);
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), mtext, "the refused read left the marker unchanged");

    // a copy of the ledger at another path is refused as well
    let l3 = s(&dir.join("ledger3.jsonl"));
    std::fs::copy(&l, &l3).unwrap();
    let (code, _, err) = read(&data, &l3, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);

    // deleting the marker is visible to the gate
    std::fs::remove_file(&marker).unwrap();
    let (code, _, err) = verify("1", false);
    assert_eq!(code, 1, "{}", err);
    assert!(err.contains("without a line"), "{}", err);
}

#[test]
fn final_read_reports_the_selected_candidate() {
    let (_scratch, dir) = scratch("selected");
    let sub = |n: &str| {
        let d = dir.join(n);
        std::fs::create_dir_all(&d).unwrap();
        d
    };
    let (full_dir, final_dir) = (sub("full"), sub("final"));
    let exprs = ["x", "0-x"];
    let flags = ["--validation-frac", "0.3", "--bootstrap", "200"];

    // a non-final ledger keeps the per-candidate columns
    let data = store(&full_dir, "d.csv");
    let (l, sealed) = (s(&full_dir.join("ledger.jsonl")), s(&full_dir.join("sealed.json")));
    dev(&data, &l, &exprs, &flags);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, full, err) = read(&data, &l, &sealed, &exprs, &flags);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(p_values(&full).len(), 2, "{}", full);

    // a final ledger reports the selected candidate only
    let data = store(&final_dir, "d.csv");
    let (l, sealed) = (s(&final_dir.join("ledger.jsonl")), s(&final_dir.join("sealed.json")));
    dev(&data, &l, &exprs, &flags);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = read(&data, &l, &sealed, &exprs, &["--batch-pnl"]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("selected candidate only"), "{}", err);
    assert!(!Path::new(&format!("{}.locked_reads.jsonl", data)).exists(), "a refused read leaves no marker");
    let (code, fin, err) = read(&data, &l, &sealed, &exprs, &flags);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(p_values(&fin).len(), 1, "{}", fin);
    assert!(fin.contains("\"pbo_error\":\"final read reports the selected candidate\""), "{}", fin);
    assert!(fin.contains("\"candidates\":1,"), "{}", fin);
    assert!(!fin.contains("\"batch_pnl\":{"), "{}", fin);
    // the selected candidate's own pooled result equals the one in the full report
    let sharpe = |j: &str| j.split("\"sharpe_annual\":").nth(1).unwrap().split(',').next().unwrap().to_string();
    assert_eq!(sharpe(&fin), sharpe(&full));
    // --verify-read reproduces the recorded final report
    let hash = fin.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    let mut verify_flags = flags.to_vec();
    verify_flags.extend_from_slice(&["--verify-read", "--expect-report-sha256", &hash]);
    let (code, _, err) = read(&data, &l, &sealed, &exprs, &verify_flags);
    assert_eq!(code, 0, "{}", err);
}

#[test]
fn report_candidates_keeps_the_full_report_on_a_final_ledger() {
    let (_scratch, dir) = scratch("keep");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let exprs = ["x", "0-x"];
    let flags = ["--validation-frac", "0.3", "--bootstrap", "200"];
    dev(&data, &l, &exprs, &flags);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--report-candidates", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    assert!(std::fs::read_to_string(&l).unwrap().contains("\"report_candidates\":true"));
    let (code, out, err) = read(&data, &l, &sealed, &exprs, &flags);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(p_values(&out).len(), 2, "{}", out);
}

/// A sealed panel for the ladder: `(scratch, dir, data, windows, out, ledger, sealed)`, one walk-forward
/// development read on the ledger and a final seal, no ladder read yet.
fn ladder_fixture(name: &str) -> (Scratch, PathBuf, String, String, String, String, String) {
    let (scratch, dir) = scratch(name);
    let (data, win) = (dir.join("p.csv"), dir.join("w.csv"));
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            csv.push_str(&format!("{},s{},{},{},{}\n", T0 + i * H, g, f1, f2, y));
        }
    }
    std::fs::write(&data, csv).unwrap();
    std::fs::write(
        &win,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nlocked,locked_0,{},{},{},{},{}\n",
            T0,
            T0 + 398 * H,
            T0 + 250 * H,
            T0 + 400 * H,
            T0 + 500 * H
        ),
    )
    .unwrap();
    let (d, w, out) = (s(&data), s(&win), s(&dir.join("ladder.json")));
    lockbox::write_sidecar(&d, &Lockbox { start: T0 + 400 * H, end: T0 + 500 * H }, "test").unwrap();
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let wfd = window("dev", 0, 150, 160, 250);
    let (code, _, err) = rcsr(&[
        "walkforward", "--data", &d, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym", "--horizon", "5",
        "--cost-bps", "5", "--periods-per-year", "365", "--json", "--ledger", &l, "--window", &wfd,
    ]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    (scratch, dir, d, w, out, l, sealed)
}

fn ladder_read(d: &str, w: &str, out: &str, l: &str, sealed: &str) -> (i32, String, String) {
    rcsr(&[
            "ladder", "--data", &d, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
            "--windows", &w, "--scheme", "locked", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50",
            "--max-fit-rows", "600", "--out", &out, "--locked-test", "--ledger", &l, "--sealed", &sealed,
    ])
}

#[test]
fn ladder_on_a_final_ledger_reads_once() {
    let (_scratch, _dir, d, w, out, l, sealed) = ladder_fixture("ladder");
    let run = || ladder_read(&d, &w, &out, &l, &sealed);
    let (code, _, err) = run();
    assert_eq!(code, 0, "{}", err);
    let ledger = std::fs::read_to_string(&l).unwrap();
    assert!(ledger.contains("\"event\":\"baseline_read\""), "{}", ledger);
    let mtext = std::fs::read_to_string(format!("{}.locked_reads.jsonl", d)).unwrap();
    assert!(mtext.contains("\"event\":\"baseline_read\""), "{}", mtext);
    let (code, _, err) = run();
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("baseline_read"), "{}", err);
}

#[test]
fn a_final_read_never_scores_the_locked_rows_of_the_other_candidates() {
    use rcsr::walkforward::{self, EvalSpec, Policy, ProtocolOptions, Window};
    let mut csv = String::from("ts,x,z,target\n");
    for i in 0..200i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        // z * z is finite on development rows and overflows on the locked rows
        let z = if i >= 150 { 1e200 } else { 1000.0 };
        csv.push_str(&format!("{},{},{},{}\n", i, x, z, y));
    }
    let table = rcsr::data::parse_table(&csv).unwrap();
    let spec = |exprs: &[&str]| EvalSpec {
        expression: exprs[0].to_string(),
        vars: vec!["x".to_string(), "z".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: None,
        windows: vec![Window {
            label: "locked".to_string(),
            train_start: 0.0,
            train_end: 120.0,
            test_start: 150.0,
            test_end: 200.0,
        }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.70,
        threshold_fixed: None,
        validation_frac: 0.3,
        periods_per_year: Some(365.0),
        candidates: if exprs.len() > 1 { exprs.iter().map(|e| e.to_string()).collect() } else { Vec::new() },
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    };
    let both = spec(&["x", "z*z"]);
    let full = walkforward::run_table_with(&table, &both, "t", &ProtocolOptions::default()).unwrap();
    assert_eq!(full.windows[0].candidate, 0);
    assert!(full.windows[0].dropped_rows > 0, "the non-final run scores z*z on the locked rows");

    let opts = ProtocolOptions { final_read: true, ..ProtocolOptions::default() };
    let fin = walkforward::run_table_with(&table, &both, "t", &opts).unwrap();
    assert_eq!(fin.windows[0].candidate, 0);
    assert_eq!(fin.windows[0].dropped_rows, 0, "z*z was not evaluated on a locked row");
    // the result equals the run of the selected candidate alone
    let alone = walkforward::run_table_with(&table, &spec(&["x"]), "t", &ProtocolOptions::default()).unwrap();
    assert_eq!(format!("{:?}", fin.windows[0].metrics), format!("{:?}", alone.windows[0].metrics));
    assert_eq!(fin.windows[0].test_rows, alone.windows[0].test_rows);
    assert_eq!(fin.windows[0].validation_error, full.windows[0].validation_error);
}

#[test]
fn a_locked_read_refuses_a_range_other_than_the_one_developed_against() {
    let (_scratch, dir) = scratch("bound-range");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let ledger_text = std::fs::read_to_string(&l).unwrap();
    let (start, end) = (T0 + 300 * H, T0 + 400 * H);
    let bound = format!("\"locked_test\":{{\"start\":{},\"end\":{},\"source\":\"store\"}}", start, end);
    assert!(ledger_text.contains(&bound), "{}", ledger_text);
    assert!(std::fs::read_to_string(&sealed).unwrap().contains(&bound));

    // the store declaration changes after development: same data bytes, other range
    lockbox::write_sidecar(&data, &Lockbox { start, end: end + H }, "test").unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("was developed against the locked test"), "{}", err);
    // a flag that widens the range is refused the same way
    lockbox::write_sidecar(&data, &Lockbox { start, end }, "test").unwrap();
    let wide = format!("{},{}", start - H, end);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--lockbox", &wide]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("was developed against the locked test"), "{}", err);
    assert!(!Path::new(&format!("{}.locked_reads.jsonl", data)).exists(), "a refused read leaves no marker");

    // the same range reads once, and the read line carries the range
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 0, "{}", err);
    let last = std::fs::read_to_string(&l).unwrap().lines().last().unwrap().to_string();
    assert!(last.contains("\"event\":\"test_read\"") && last.contains(&bound), "{}", last);
}

#[test]
fn a_ledger_without_a_recorded_range_needs_the_unbound_flag() {
    let (_scratch, dir) = scratch("unbound");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    // rewrite the ledger as an old one: no locked_test member, hash chain recomputed
    let mut old: Vec<String> = Vec::new();
    let mut prev = "0".repeat(64);
    for line in std::fs::read_to_string(&l).unwrap().lines() {
        let at = line.find("\"locked_test\":{").unwrap();
        let close = line[at..].find("},").unwrap() + at + 2;
        let stripped = format!("{}{}", &line[..at], &line[close..]);
        let line = format!("{{\"prev\":\"{}\"{}", prev, &stripped[stripped.find(',').unwrap()..]);
        prev = rcsr::manifest::sha256_hex(line.as_bytes());
        old.push(line);
    }
    let l_old = s(&dir.join("old.jsonl"));
    std::fs::write(&l_old, old.join("\n") + "\n").unwrap();
    let (code, _, err) = rcsr(&["protocol", "seal", "--ledger", &l_old, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = read(&data, &l_old, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("--allow-unbound-ledger"), "{}", err);
    let (code, _, err) = read(&data, &l_old, &sealed, &["x"], &["--allow-unbound-ledger"]);
    assert_eq!(code, 0, "{}", err);
    assert!(err.contains("unbound"), "{}", err);
}

#[test]
fn a_sealed_batch_read_verifies() {
    let (_scratch, dir) = scratch("batch");
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let exprs = ["x", "0-x"];
    let flags = ["--validation-frac", "0.3", "--batch-pnl"];
    dev(&data, &l, &exprs, &flags);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--report-candidates", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, out, err) = read(&data, &l, &sealed, &exprs, &flags);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"batch_pnl\":{"), "{}", out);
    let mut verify_flags = flags.to_vec();
    verify_flags.push("--verify-read");
    let (code, _, err) = read(&data, &l, &sealed, &exprs, &verify_flags);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"batch_timings\":{"), "{}", out);
}

/// One sealed ladder read on a fresh copy of the same synthetic panel. Returns
/// the ledger text, the data path and the ledger path.
fn sealed_ladder_read(name: &str) -> (Scratch, String, String, String) {
    let (scratch, dir) = scratch(name);
    let (data, win) = (dir.join("p.csv"), dir.join("w.csv"));
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            csv.push_str(&format!("{},s{},{},{},{}\n", T0 + i * H, g, f1, f2, y));
        }
    }
    std::fs::write(&data, csv).unwrap();
    std::fs::write(
        &win,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nlocked,locked_0,{},{},{},{},{}\n",
            T0,
            T0 + 398 * H,
            T0 + 250 * H,
            T0 + 400 * H,
            T0 + 500 * H
        ),
    )
    .unwrap();
    let (d, w, out) = (s(&data), s(&win), s(&dir.join("ladder.json")));
    lockbox::write_sidecar(&d, &Lockbox { start: T0 + 400 * H, end: T0 + 500 * H }, "test").unwrap();
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let wfd = window("dev", 0, 150, 160, 250);
    let (code, _, err) = rcsr(&[
        "walkforward", "--data", &d, "--target", "y", "--expr", "f1", "--time", "ts", "--group-by", "sym", "--horizon", "5",
        "--cost-bps", "5", "--periods-per-year", "365", "--json", "--ledger", &l, "--window", &wfd,
    ]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&[
        "ladder", "--data", &d, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
        "--windows", &w, "--scheme", "locked", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50",
        "--max-fit-rows", "600", "--out", &out, "--locked-test", "--ledger", &l, "--sealed", &sealed,
    ]);
    assert_eq!(code, 0, "{}", err);
    (scratch, std::fs::read_to_string(&l).unwrap(), d, l)
}

fn baseline_read_sha(ledger: &str) -> String {
    let line = ledger.lines().find(|l| l.contains("\"event\":\"baseline_read\"")).expect("baseline_read line");
    line.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string()
}

#[test]
fn a_sealed_ladder_read_hash_is_reproducible_and_verifies() {
    let (_sa, a, da, la) = sealed_ladder_read("lad");
    let (code, out, err) = rcsr(&["protocol", "verify", "--ledger", &la, "--data", &da]);
    assert_eq!(code, 0, "{}{}", out, err);
    let (_sb, b, _, _) = sealed_ladder_read("lad");
    assert_eq!(baseline_read_sha(&a), baseline_read_sha(&b));
}

#[test]
fn ladder_read_hash_ignores_the_store_directory() {
    let (_sa, a, da, _) = sealed_ladder_read("lad-dir-a");
    let (_sb, b, db, _) = sealed_ladder_read("lad-dir-b");
    assert_ne!(da, db);
    assert_eq!(baseline_read_sha(&a), baseline_read_sha(&b));
    let printed = std::fs::read_to_string(Path::new(&da).with_file_name("ladder.json")).unwrap();
    assert!(printed.contains(&format!("\"file\":\"{}.lockbox.json\"", da)), "{printed}");
}

fn sealed_store(name: &str) -> (Scratch, PathBuf, String, String, String) {
    let (scratch, dir) = scratch(name);
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, &["x"], &[]);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    (scratch, dir, data, l, sealed)
}

#[test]
fn read_refuses_a_marker_with_a_cut_last_line() {
    let (_scratch, _dir, data, l, sealed) = sealed_store("marker-cut");
    let marker = format!("{}.locked_reads.jsonl", data);
    std::fs::write(&marker, "{\"ledger\":\"/x\",\"ledger_head\":\"ab").unwrap();
    let ledger_before = std::fs::read(&l).unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    assert_eq!(std::fs::read_to_string(&marker).unwrap(), "{\"ledger\":\"/x\",\"ledger_head\":\"ab");
    // a complete line without its newline is refused too
    std::fs::write(&marker, "{\"ledger\":\"/x\",\"ledger_head\":\"ab\",\"event\":\"test_read\",\"report_sha256\":\"cd\"}").unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    let (code, _, err) = rcsr(&["protocol", "verify", "--ledger", &l, "--data", &data]);
    assert_eq!(code, 1, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
}

#[test]
fn every_reader_refuses_a_ledger_with_a_cut_last_line() {
    let (_scratch, dir, data, l, sealed) = sealed_store("readers-cut");
    let mut text = std::fs::read_to_string(&l).unwrap();
    text.pop();
    std::fs::write(&l, &text).unwrap();
    // development run, locked read and verify read the same file
    let w = window("dev", 0, 150, 160, 250);
    let (code, _, err) = wf(&data, &l, &["x"], &["--window", &w]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &[]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--verify-read"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("a write was cut"), "{}", err);
    assert_eq!(std::fs::read_to_string(&l).unwrap(), text, "no run changed the ledger");
    assert!(!Path::new(&format!("{}.locked_reads.jsonl", data)).exists());
    drop(dir);
}

fn recorded_read_sha(ledger: &str) -> String {
    let text = std::fs::read_to_string(ledger).unwrap();
    let line = text.lines().find(|l| l.contains("\"event\":\"test_read\"")).expect("test_read line");
    line.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string()
}

#[test]
fn walkforward_report_that_cannot_be_written_leaves_the_read_available() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-unwritable");
    let ledger_before = std::fs::read(&l).unwrap();
    let marker = format!("{}.locked_reads.jsonl", data);
    let dir_out = dir.join("is-a-directory");
    std::fs::create_dir(&dir_out).unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &s(&dir_out)]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("the locked read is still available"), "{}", err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    assert!(!Path::new(&marker).exists());
    let out = s(&dir.join("report.json"));
    let bad_md = s(&dir.join("no-such-dir").join("r.md"));
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out, "--markdown", &bad_md]);
    assert_eq!(code, 3, "{}", err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    assert!(!Path::new(&marker).exists());
    assert!(!Path::new(&out).exists());
    assert!(!Path::new(&format!("{}.tmp", out)).exists());
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out]);
    assert_eq!(code, 0, "{}", err);
    assert!(std::fs::read_to_string(&out).unwrap().contains("\"read_count\":1"));
    assert_eq!(std::fs::read_to_string(&l).unwrap().matches("\"event\":\"test_read\"").count(), 1);
}

#[test]
fn walkforward_report_lost_after_the_record_is_recovered() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-lost");
    let out = s(&dir.join("report.json"));
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out]);
    assert_eq!(code, 0, "{}", err);
    let marker = format!("{}.locked_reads.jsonl", data);
    let (ledger_before, marker_before) = (std::fs::read(&l).unwrap(), std::fs::read(&marker).unwrap());
    let first = std::fs::read(&out).unwrap();
    // the state of a kill between the record and the rename
    std::fs::remove_file(&out).unwrap();
    std::fs::write(format!("{}.tmp", out), &first).unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out]);
    assert_eq!(code, 3, "{}", err);
    assert!(!Path::new(&out).exists());
    let hash = recorded_read_sha(&l);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out, "--verify-read", "--expect-report-sha256", &hash]);
    assert_eq!(code, 0, "{}", err);
    let recovered = std::fs::read_to_string(&out).unwrap();
    assert!(recovered.contains(&format!("\"report_sha256\":\"{hash}\"")), "{recovered}");
    assert!(recovered.contains("\"read_count\":1"), "{recovered}");
    assert!(String::from_utf8_lossy(&first).contains(&hash));
    assert!(!Path::new(&format!("{}.tmp", out)).exists());
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    assert_eq!(std::fs::read(&marker).unwrap(), marker_before);
}

#[test]
fn walkforward_retry_after_a_marker_without_a_ledger_line_records_the_read() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-marker-only");
    let ledger_before = std::fs::read(&l).unwrap();
    let out = s(&dir.join("report.json"));
    assert_eq!(read(&data, &l, &sealed, &["x"], &["--out", &out]).0, 0);
    let marker = format!("{}.locked_reads.jsonl", data);
    let marker_before = std::fs::read(&marker).unwrap();
    // the state of a kill between the marker line and the ledger line
    std::fs::write(&l, &ledger_before).unwrap();
    std::fs::remove_file(&out).unwrap();
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out, "--verify-read"]);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("no recorded read"), "{}", err);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &out]);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(std::fs::read(&marker).unwrap(), marker_before);
    let (code, o, err) = rcsr(&["protocol", "verify", "--ledger", &l, "--data", &data, "--expect-reads", "1"]);
    assert_eq!(code, 0, "{}{}", o, err);
}

#[test]
fn walkforward_verify_read_needs_locked_test() {
    let (_scratch, _dir, data, l, _sealed) = sealed_store("wf-flag");
    let w = window("dev", 0, 150, 160, 250);
    let (code, _, err) = wf(&data, &l, &["x"], &["--window", &w, "--verify-read"]);
    assert_eq!(code, 2, "{}", err);
    assert!(err.contains("--verify-read applies to --locked-test only"), "{}", err);
}

fn no_staged_files(dir: &Path, what: &str) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let name = entry.unwrap().file_name().to_string_lossy().into_owned();
        assert!(!name.ends_with(".tmp"), "{name} left behind for {what}");
    }
}

#[test]
fn walkforward_refuses_outputs_that_name_one_file() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-same-file");
    std::fs::create_dir(dir.join("b")).unwrap();
    let ledger_before = std::fs::read(&l).unwrap();
    let marker = format!("{}.locked_reads.jsonl", data);
    let out = dir.join("a.json");
    let forms = [
        dir.join(".").join("a.json"),
        dir.join("b").join("..").join("a.json"),
    ];
    for md in forms {
        let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &s(&out), "--markdown", &s(&md)]);
        assert_eq!(code, 3, "{}: {}", md.display(), err);
        assert!(err.contains("same file"), "{}", err);
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert!(!Path::new(&marker).exists());
        assert!(!out.exists());
        no_staged_files(&dir, &md.display().to_string());
    }
    // a symlink and a hard link to an existing output
    std::fs::write(&out, "old").unwrap();
    std::os::unix::fs::symlink(&out, dir.join("sym.md")).unwrap();
    std::fs::hard_link(&out, dir.join("hard.md")).unwrap();
    for md in ["sym.md", "hard.md"] {
        let (code, _, err) = read(&data, &l, &sealed, &["x"], &["--out", &s(&out), "--markdown", &s(&dir.join(md))]);
        assert_eq!(code, 3, "{md}: {err}");
        assert!(err.contains("same file"), "{md}: {err}");
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert!(!Path::new(&marker).exists());
        assert_eq!(std::fs::read_to_string(&out).unwrap(), "old");
        no_staged_files(&dir, md);
    }
}

#[test]
fn walkforward_refuses_a_ledger_event_without_a_valid_read_count() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-bad-count");
    let marker = format!("{}.locked_reads.jsonl", data);
    let text = std::fs::read_to_string(&l).unwrap();
    let head = rcsr::manifest::sha256_hex(text.lines().last().unwrap().as_bytes());
    let bad = format!("{{\"prev\":\"{head}\",\"event\":\"test_read\",\"id\":\"zz\",\"report_sha256\":\"bb\",\"test_read\":true}}\n");
    std::fs::write(&l, format!("{text}{bad}")).unwrap();
    let ledger_before = std::fs::read(&l).unwrap();
    let out = s(&dir.join("report.json"));
    for extra in [vec!["--out", out.as_str()], vec!["--out", out.as_str(), "--verify-read"]] {
        let (code, _, err) = read(&data, &l, &sealed, &["x"], &extra);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("line 3 is a test_read event without a valid read_count"), "{}", err);
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert!(!Path::new(&marker).exists() && !Path::new(&out).exists());
    }
}

/// Appends `body` as one ledger line chained to the last line.
fn append_chained(ledger: &str, body: &str) {
    let text = std::fs::read_to_string(ledger).unwrap();
    let head = rcsr::manifest::sha256_hex(text.lines().last().unwrap().as_bytes());
    std::fs::write(ledger, format!("{text}{{\"prev\":\"{head}\",{body}}}\n")).unwrap();
}

/// A faulty event per field a read depends on: `(field, body)`. `read_count` is covered by
/// `walkforward_refuses_a_ledger_event_without_a_valid_read_count`.
const FAULTY_EVENTS: &[(&str, &str)] = &[
    ("id", "\"event\":\"test_read\",\"report_sha256\":\"bb\",\"read_count\":1,\"test_read\":true"),
    ("id", "\"event\":\"test_read\",\"id\":\"\",\"report_sha256\":\"bb\",\"read_count\":1,\"test_read\":true"),
    ("report_sha256", "\"event\":\"test_read\",\"id\":\"c\",\"read_count\":1,\"test_read\":true"),
    ("report_sha256", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"\",\"read_count\":1,\"test_read\":true"),
    ("read_count", "\"event\":\"test_read\",\"id\":\"c\",\"report_sha256\":\"bb\",\"read_count\":\"1\",\"test_read\":true"),
    ("data_sha256", "\"event\":\"baseline_read\",\"report_sha256\":\"bb\""),
    ("report_sha256", "\"event\":\"baseline_read\",\"data_sha256\":\"aa\",\"report_sha256\":\"\""),
    ("sealed_sha256", "\"event\":\"seal\""),
    ("sealed_sha256", "\"event\":\"seal\",\"sealed_sha256\":\"\""),
    ("id", "\"event\":\"candidate\""),
    ("id", "\"event\":\"candidate\",\"id\":\"\""),
];

#[test]
fn walkforward_read_refuses_every_faulty_ledger_field_with_exit_3() {
    let (_scratch, dir, data, l, sealed) = sealed_store("wf-faulty-fields");
    let marker = format!("{}.locked_reads.jsonl", data);
    let original = std::fs::read_to_string(&l).unwrap();
    let out = s(&dir.join("report.json"));
    for (field, body) in FAULTY_EVENTS {
        std::fs::write(&l, &original).unwrap();
        append_chained(&l, body);
        let ledger_before = std::fs::read(&l).unwrap();
        for extra in [vec!["--out", out.as_str()], vec!["--out", out.as_str(), "--verify-read"]] {
            let (code, _, err) = read(&data, &l, &sealed, &["x"], &extra);
            assert_eq!(code, 3, "{field} in {body}: {err}");
            assert!(err.contains(&format!("without a valid {field}")), "{field} in {body}: {err}");
            assert_eq!(std::fs::read(&l).unwrap(), ledger_before, "{body}");
            assert!(!Path::new(&marker).exists() && !Path::new(&out).exists(), "{body}");
        }
        // the verifier reports the same fault with its own documented code
        let (code, _, err) = rcsr(&["protocol", "verify", "--ledger", &l, "--data", &data]);
        assert_eq!(code, 1, "{body}: {err}");
        assert!(err.contains(&format!("without a valid {field}")), "{body}: {err}");
    }
}

#[test]
fn ladder_read_refuses_every_faulty_ledger_field_with_exit_3() {
    let (_scratch, dir, d, w, out, l, sealed) = ladder_fixture("ladder-faulty-fields");
    let marker = format!("{}.locked_reads.jsonl", d);
    let original = std::fs::read_to_string(&l).unwrap();
    for (field, body) in FAULTY_EVENTS {
        std::fs::write(&l, &original).unwrap();
        append_chained(&l, body);
        let ledger_before = std::fs::read(&l).unwrap();
        let (code, _, err) = ladder_read(&d, &w, &out, &l, &sealed);
        assert_eq!(code, 3, "{field} in {body}: {err}");
        assert!(err.contains(&format!("without a valid {field}")), "{field} in {body}: {err}");
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before, "{body}");
        assert!(!Path::new(&marker).exists() && !Path::new(&out).exists(), "{body}");
        no_staged_files(&dir, body);
    }
}

// ---------------------------------------------------------------------------
// #961: the sealed read scores a vote book.
// ---------------------------------------------------------------------------

const VOTE_MEMBERS: [&str; 3] = ["x", "x - 2", "x + 2"];

fn vote_flags<'a>(quorum: &'a str, tail: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["--policy", "vote", "--quorum", quorum, "--threshold-fixed", "0"];
    a.extend_from_slice(tail);
    a
}

/// A vote book developed, sealed (final) and ready to read. Returns the data, ledger and sealed paths.
fn sealed_vote(name: &str, members: &[&str], quorum: &str, tail: &[&str], final_seal: bool) -> (Scratch, String, String, String) {
    let (scratch, dir) = scratch(name);
    let data = store(&dir, "d.csv");
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, members, &vote_flags(quorum, tail));
    let mut a = vec!["protocol", "seal", "--ledger", l.as_str(), "--out", sealed.as_str()];
    if final_seal {
        a.splice(2..2, ["--final", "--max-reads", "2"]);
    }
    let (code, _, err) = rcsr(&a);
    assert_eq!(code, 0, "{}", err);
    (scratch, data, l, sealed)
}

fn test_reads(ledger: &str) -> usize {
    std::fs::read_to_string(ledger).unwrap().lines().filter(|l| l.contains("\"event\":\"test_read\"")).count()
}

#[test]
fn vote_locked_read_scores_sealed_book() {
    let (_s, data, l, sealed) = sealed_vote("vote-read", &VOTE_MEMBERS, "0.67", &[], true);
    let (code, out, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &vote_flags("0.67", &[]));
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"policy\":{\"name\":\"vote\""), "{}", out);
    assert!(out.contains("\"quorum\":0.67"), "{}", out);
    assert!(out.contains("\"votes_required\":3"), "{}", out);
    for m in VOTE_MEMBERS {
        assert!(out.contains(&format!("\"{}\"", m)), "member {} in {}", m, out);
    }
    assert!(out.contains("\"test_read\":true") && out.contains("\"read_count\":1"), "{}", out);
    assert_eq!(test_reads(&l), 1);
    let window_rows: usize = out.split("\"test_rows\":").nth(1).unwrap().split(',').next().unwrap().parse().unwrap();
    assert!(window_rows > 0, "{}", out);
}

#[test]
fn vote_locked_second_read_refused() {
    let (_s, data, l, sealed) = sealed_vote("vote-twice", &VOTE_MEMBERS, "0.67", &[], true);
    let flags = vote_flags("0.67", &[]);
    assert_eq!(read(&data, &l, &sealed, &VOTE_MEMBERS, &flags).0, 0);
    assert_eq!(test_reads(&l), 1);
    let (code, _, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &flags);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("already has test_read:true"), "{}", err);
    assert_eq!(test_reads(&l), 1);
}

#[test]
fn vote_locked_changed_spec_refused() {
    let (_s, data, l, sealed) = sealed_vote("vote-changed", &VOTE_MEMBERS, "0.67", &[], false);
    let order = ["x - 2", "x", "x + 2"];
    let changed = ["x", "x - 2", "x + 3"];
    let subset = ["x", "x - 2"];
    let cases: Vec<(&str, Vec<&str>, Vec<String>)> = vec![
        ("order", order.to_vec(), vote_flags("0.67", &[]).iter().map(|v| v.to_string()).collect()),
        ("member", changed.to_vec(), vote_flags("0.67", &[]).iter().map(|v| v.to_string()).collect()),
        ("subset", subset.to_vec(), vote_flags("0.67", &[]).iter().map(|v| v.to_string()).collect()),
        ("quorum", VOTE_MEMBERS.to_vec(), vote_flags("0.34", &[]).iter().map(|v| v.to_string()).collect()),
        ("band", VOTE_MEMBERS.to_vec(), vote_flags("0.67", &["--band", "1"]).iter().map(|v| v.to_string()).collect()),
        ("min-hold", VOTE_MEMBERS.to_vec(), vote_flags("0.67", &["--min-hold", "2"]).iter().map(|v| v.to_string()).collect()),
        (
            "threshold",
            VOTE_MEMBERS.to_vec(),
            ["--policy", "vote", "--quorum", "0.67", "--threshold-fixed", "0.5"].iter().map(|v| v.to_string()).collect(),
        ),
    ];
    for (what, members, flags) in cases {
        let flags: Vec<&str> = flags.iter().map(String::as_str).collect();
        let (code, _, err) = read(&data, &l, &sealed, &members, &flags);
        assert_eq!(code, 3, "{}: {}", what, err);
        assert!(err.contains("is not in the sealed list; the runner refuses it"), "{}: {}", what, err);
        assert_eq!(test_reads(&l), 0, "{}", what);
    }
    let (code, _, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &vote_flags("0.67", &[]));
    assert_eq!(code, 0, "{}", err);
}

/// The pooled metrics object of a walk-forward report.
fn pooled(json: &str) -> String {
    let rest = json.split("\"pooled\":{").nth(1).expect("pooled object");
    let mut depth = 1;
    let mut end = 0;
    for (i, c) in rest.char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    rest[..end].to_string()
}

#[test]
fn vote_locked_one_member_equals_single() {
    let (_a, dir_a) = scratch("vote-one-single");
    let data_a = store(&dir_a, "d.csv");
    let (la, sa) = (s(&dir_a.join("ledger.jsonl")), s(&dir_a.join("sealed.json")));
    let single = ["--threshold-fixed", "0"];
    dev(&data_a, &la, &["x"], &single);
    assert_eq!(rcsr(&["protocol", "seal", "--final", "--ledger", &la, "--out", &sa]).0, 0);
    let (code, single_out, err) = read(&data_a, &la, &sa, &["x"], &single);
    assert_eq!(code, 0, "{}", err);

    let (_b, data_b, lb, sb) = sealed_vote("vote-one-vote", &["x"], "1", &[], true);
    let (code, vote_out, err) = read(&data_b, &lb, &sb, &["x"], &vote_flags("1", &[]));
    assert_eq!(code, 0, "{}", err);
    assert_eq!(pooled(&single_out), pooled(&vote_out));
}

#[test]
fn vote_locked_read_verifies_by_hash() {
    let (_s, data, l, sealed) = sealed_vote("vote-verify", &VOTE_MEMBERS, "0.67", &[], true);
    let (code, _, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &vote_flags("0.67", &[]));
    assert_eq!(code, 0, "{}", err);
    let sha = recorded_read_sha(&l);
    let mut verify = vote_flags("0.67", &["--verify-read", "--expect-report-sha256", &sha]);
    let (code, _, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &verify);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(test_reads(&l), 1, "verify appends no read");
    let wrong = "0".repeat(64);
    verify = vote_flags("0.67", &["--verify-read", "--expect-report-sha256", &wrong]);
    let (code, _, err) = read(&data, &l, &sealed, &VOTE_MEMBERS, &verify);
    assert_eq!(code, 3, "{}", err);
    assert!(err.contains("differs from the expected"), "{}", err);
}

/// The members and thresholds of a sealed vote read come from rows before the
/// locked range: a store that differs inside the range only gives the same thresholds.
#[test]
fn vote_locked_members_and_thresholds_ignore_locked_rows() {
    let run = |poison: bool| {
        let (_s, dir) = scratch(if poison { "vote-leak-p" } else { "vote-leak-c" });
        let data = dir.join("d.csv");
        let mut csv = String::from("ts,x,y\n");
        for i in 0..400i64 {
            let locked = i >= 300;
            let x = ((i * 37) % 17) as f64 - 8.0;
            let x = if poison && locked { x * 100.0 + 50.0 } else { x };
            let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
            let y = if poison && locked { -y } else { y };
            csv.push_str(&format!("{},{},{}\n", T0 + i * H, x, y));
        }
        std::fs::write(&data, csv).unwrap();
        let d = s(&data);
        lockbox::write_sidecar(&d, &Lockbox { start: T0 + 300 * H, end: T0 + 400 * H }, "test").unwrap();
        let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
        let flags = ["--policy", "vote", "--quorum", "0.67", "--validation-frac", "0.3"];
        dev(&d, &l, &VOTE_MEMBERS, &flags);
        assert_eq!(rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]).0, 0);
        let (code, out, err) = read(&d, &l, &sealed, &VOTE_MEMBERS, &flags);
        assert_eq!(code, 0, "{}", err);
        out
    };
    let (clean, poisoned) = (run(false), run(true));
    let thresholds = |j: &str| j.split("\"member_thresholds\":[").nth(1).unwrap().split(']').next().unwrap().to_string();
    assert_eq!(thresholds(&clean), thresholds(&poisoned));
    assert_ne!(pooled(&clean), pooled(&poisoned), "the locked rows do reach the metrics");
}

fn marker_of(data: &str) -> String {
    format!("{}.locked_reads.jsonl", data)
}

/// Seals a final book on a store; `rows` below 300 leaves no row in the locked range.
fn sealed_book(name: &str, rows: i64, members: &[&str], flags: &[&str]) -> (Scratch, String, String, String) {
    let (scratch, dir) = scratch(name);
    let data = store_rows(&dir, "d.csv", rows);
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    dev(&data, &l, members, flags);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    (scratch, data, l, sealed)
}

fn verify_without_a_recorded_read_scores_nothing(name: &str, members: &[&str], flags: &[&str]) {
    let (_s, data, l, sealed) = sealed_book(name, 300, members, flags);
    let ledger_before = std::fs::read(&l).unwrap();
    let zeros = "0".repeat(64);
    for extra in [vec!["--verify-read"], vec!["--verify-read", "--expect-report-sha256", zeros.as_str()]] {
        let mut a = flags.to_vec();
        a.extend_from_slice(&extra);
        let (code, out, err) = read(&data, &l, &sealed, members, &a);
        assert_eq!(code, 3, "{}", err);
        assert!(err.contains("no recorded read for this candidate"), "{}", err);
        assert!(!err.contains("no finite test rows"), "locked rows were scored: {}", err);
        assert!(!err.contains("report sha256") && !out.contains("report_sha256"), "{}{}", out, err);
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert!(!Path::new(&marker_of(&data)).exists());
    }
}

#[test]
fn verify_read_without_a_recorded_read_scores_nothing() {
    verify_without_a_recorded_read_scores_nothing("verify-unread", &["x"], &[]);
}

#[test]
fn vote_verify_read_without_a_recorded_read_scores_nothing() {
    verify_without_a_recorded_read_scores_nothing("vote-verify-unread", &VOTE_MEMBERS, &vote_flags("0.67", &[]));
}

fn expect_hash_without_verify_is_refused(name: &str, members: &[&str], flags: &[&str]) {
    let (_s, data, l, sealed) = sealed_book(name, 400, members, flags);
    let ledger_before = std::fs::read(&l).unwrap();
    let zeros = "0".repeat(64);
    let mut a = flags.to_vec();
    a.extend_from_slice(&["--expect-report-sha256", zeros.as_str()]);
    for _ in 0..2 {
        let (code, out, err) = read(&data, &l, &sealed, members, &a);
        assert_eq!(code, 2, "{}", err);
        assert!(err.contains("--expect-report-sha256 with --locked-test needs --verify-read"), "{}", err);
        assert!(!err.contains("differs") && !out.contains("report_sha256"), "{}{}", out, err);
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert!(!Path::new(&marker_of(&data)).exists());
    }
    let (code, _, err) = read(&data, &l, &sealed, members, flags);
    assert_eq!(code, 0, "{}", err);
    let sha = recorded_read_sha(&l);
    let mut verify = flags.to_vec();
    verify.extend_from_slice(&["--verify-read", "--expect-report-sha256", sha.as_str()]);
    let (code, _, err) = read(&data, &l, &sealed, members, &verify);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(test_reads(&l), 1);
}

#[test]
fn locked_read_refuses_expect_hash_without_verify() {
    expect_hash_without_verify_is_refused("expect-no-verify", &["x"], &[]);
}

#[test]
fn vote_locked_read_refuses_expect_hash_without_verify() {
    expect_hash_without_verify_is_refused("vote-expect-no-verify", &VOTE_MEMBERS, &vote_flags("0.67", &[]));
}

/// A verify with another window set or another report setting is refused before
/// any locked row is scored: no report hash is printed and nothing is written.
fn verify_with_other_inputs_scores_nothing(name: &str, members: &[&str], flags: &[&str], setting: &[&str]) {
    let (_s, data, l, sealed) = sealed_book(name, 400, members, flags);
    let (code, _, err) = read(&data, &l, &sealed, members, flags);
    assert_eq!(code, 0, "{}", err);
    let sha = recorded_read_sha(&l);
    let (ledger_before, marker_before) = (std::fs::read(&l).unwrap(), std::fs::read(marker_of(&data)).unwrap());
    let other = window("locked", 0, 250, 310, 400);
    let unscorable = window("locked", 0, 3, 300, 400);
    let mut cases: Vec<Vec<&str>> = vec![
        vec!["--window", other.as_str()],
        vec!["--window", unscorable.as_str()],
    ];
    if !setting.is_empty() {
        cases.push(setting.to_vec());
    }
    for case in cases {
        let default_window = window("locked", 0, 250, 300, 400);
        for hash in [None, Some(sha.as_str())] {
            let mut a = vec!["--locked-test", "--sealed", sealed.as_str(), "--verify-read"];
            if let Some(h) = hash {
                a.extend_from_slice(&["--expect-report-sha256", h]);
            }
            if !case.contains(&"--window") {
                a.extend_from_slice(&["--window", default_window.as_str()]);
            }
            a.extend_from_slice(&case);
            a.extend_from_slice(flags);
            let (code, out, err) = wf(&data, &l, members, &a);
            assert_eq!(code, 3, "{:?}: {}{}", case, out, err);
            assert!(err.contains("binds the report"), "{:?}: {}", case, err);
            assert!(!err.contains("report sha256") && !out.contains("report_sha256"), "{:?}: {}{}", case, out, err);
            assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
            assert_eq!(std::fs::read(marker_of(&data)).unwrap(), marker_before);
        }
    }
    let mut verify = flags.to_vec();
    verify.extend_from_slice(&["--verify-read", "--expect-report-sha256", sha.as_str()]);
    let (code, _, err) = read(&data, &l, &sealed, members, &verify);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
}

#[test]
fn verify_read_with_other_windows_scores_nothing() {
    verify_with_other_inputs_scores_nothing("verify-windows", &["x"], &[], &["--bootstrap", "20"]);
}

#[test]
fn vote_verify_read_with_other_windows_scores_nothing() {
    verify_with_other_inputs_scores_nothing("vote-verify-windows", &VOTE_MEMBERS, &vote_flags("0.67", &[]), &[]);
}

/// A read written without a binding digest cannot be compared, so verify refuses it.
#[test]
fn verify_read_of_a_read_without_a_binding_digest_is_refused() {
    let (_s, data, l, sealed) = sealed_book("verify-old-read", 400, &["x"], &[]);
    let text = std::fs::read_to_string(&l).unwrap();
    let id = text.split("\"id\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    rcsr::protocol::Ledger::new(&l).record_read(&id, &"a".repeat(64), None).unwrap();
    let ledger_before = std::fs::read(&l).unwrap();
    let (code, out, err) = read(&data, &l, &sealed, &["x"], &["--verify-read"]);
    assert_eq!(code, 3, "{}{}", out, err);
    assert!(err.contains("stores no binding digest"), "{}", err);
    assert!(!err.contains("report sha256") && !out.contains("report_sha256"), "{}{}", out, err);
    assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
    assert!(!Path::new(&marker_of(&data)).exists());
}

#[test]
fn verify_read_with_gpu_scores_nothing() {
    verify_with_other_inputs_scores_nothing("verify-gpu", &["x"], &[], &["--gpu"]);
}

const FRONT_RUN: &str = r#"{"tool":"rcsr","variables":["x"],"target_spec":{"return_column":"y"},
    "search":{"candidates_evaluated":4321},
    "split":{"train_time":[0,100],"validation_time":[101,200],"required_purge_seconds":3600,"purge_seconds":3600},
    "objective":{"spec":{"threshold":0,"per_side_bps":5}},
    "pareto":[{"expression":"x","expression_exact":"x"}]}"#;

/// The `--front` path is not part of the report hash, so another spelling of
/// the same file verifies; a changed file is another run and is refused.
#[test]
fn verify_read_accepts_another_spelling_of_the_front_path() {
    let (_scratch, dir) = scratch("verify-front");
    let data = store(&dir, "d.csv");
    let run = s(&dir.join("run.json"));
    std::fs::write(&run, FRONT_RUN).unwrap();
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let dev_w = window("dev", 0, 150, 160, 250);
    let (code, _, err) = wf(&data, &l, &[], &["--window", dev_w.as_str(), "--front", run.as_str(), "--validation-frac", "0.2"]);
    assert_eq!(code, 0, "{}", err);
    let (code, _, err) = rcsr(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code, 0, "{}", err);
    let w = window("locked", 0, 250, 300, 400);
    let go = |front: &str, extra: &[&str]| {
        let mut a = vec!["--locked-test", "--sealed", sealed.as_str(), "--window", w.as_str(), "--front", front, "--validation-frac", "0.2"];
        a.extend_from_slice(extra);
        wf(&data, &l, &[], &a)
    };
    let (code, _, err) = go(&run, &[]);
    assert_eq!(code, 0, "{}", err);
    let sha = recorded_read_sha(&l);
    let spelled = format!("{}/./run.json", s(&dir));
    let (code, _, err) = go(&spelled, &["--verify-read", "--expect-report-sha256", &sha]);
    assert_eq!(code, 0, "{}", err);
}

/// #1094: the cut rule of `--threshold-fit fitness` is part of the sealed
/// candidate and of the read binding: the fixed arm, another floor or another
/// lambda is another candidate, and a verify with another rule scores nothing.
#[test]
fn threshold_fit_binds_the_sealed_read() {
    let fit = ["--threshold-fixed", "0", "--threshold-fit", "fitness", "--cut-min-entries", "2", "--cut-blocks", "1"];
    let (_s, data, l, sealed) = sealed_book("fit-binds", 400, &["x"], &fit);
    let floor3 = ["--threshold-fixed", "0", "--threshold-fit", "fitness", "--cut-min-entries", "3", "--cut-blocks", "1"];
    let lambda2 = ["--threshold-fixed", "0", "--threshold-fit", "fitness", "--cut-min-entries", "2", "--cut-blocks", "1", "--cut-lambda", "2"];
    let fixed = ["--threshold-fixed", "0"];
    let others: [&[&str]; 3] = [&fixed, &floor3, &lambda2];
    for other in others {
        let (code, out, err) = read(&data, &l, &sealed, &["x"], other);
        assert_eq!(code, 3, "{:?}: {}{}", other, out, err);
        assert!(err.contains("is not in the sealed list"), "{:?}: {}", other, err);
        assert!(!Path::new(&marker_of(&data)).exists());
    }
    let (code, out, err) = read(&data, &l, &sealed, &["x"], &fit);
    assert_eq!(code, 0, "{}", err);
    assert!(out.contains("\"fit\":\"fitness\""), "{}", out);
    let sha = recorded_read_sha(&l);
    let (ledger_before, marker_before) = (std::fs::read(&l).unwrap(), std::fs::read(marker_of(&data)).unwrap());
    for other in others {
        let mut a = other.to_vec();
        a.extend_from_slice(&["--verify-read", "--expect-report-sha256", sha.as_str()]);
        let (code, out, err) = read(&data, &l, &sealed, &["x"], &a);
        assert_eq!(code, 3, "{:?}: {}{}", other, out, err);
        assert!(!err.contains("report sha256") && !out.contains("report_sha256"), "{:?}: {}{}", other, out, err);
        assert_eq!(std::fs::read(&l).unwrap(), ledger_before);
        assert_eq!(std::fs::read(marker_of(&data)).unwrap(), marker_before);
    }
    let mut verify = fit.to_vec();
    verify.extend_from_slice(&["--verify-read", "--expect-report-sha256", sha.as_str()]);
    let (code, _, err) = read(&data, &l, &sealed, &["x"], &verify);
    assert_eq!(code, 0, "{}", err);
    assert_eq!(test_reads(&l), 1);
}
