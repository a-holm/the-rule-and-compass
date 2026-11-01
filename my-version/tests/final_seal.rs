//! #523: a final seal allows one locked read for one candidate, a read marker
//! beside the store refuses a second ledger, and a final read reports the
//! selected candidate only. Fixtures are synthetic; the locked range is a
//! stand-in in 2024.

use rcsr::lockbox::{self, Lockbox};
use std::path::{Path, PathBuf};
use std::process::Command;

/// 2024-01-01T00:00:00Z.
const T0: i64 = 1_704_067_200;
const H: i64 = 3600;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rcsr-final-seal-{}-{}", std::process::id(), name));
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

fn window(label: &str, a: i64, b: i64, c: i64, d: i64) -> String {
    format!("{},{},{},{},{}", label, T0 + a * H, T0 + b * H, T0 + c * H, T0 + d * H)
}

/// A store with its declared locked test (hours 300..400) beside it.
fn store(dir: &Path, name: &str) -> String {
    let path = dir.join(name);
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
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
    let dir = scratch("refuse");
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
    let dir = scratch("fresh");
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
    let dir = scratch("once");
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
    let dir = scratch("selected");
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
    let dir = scratch("keep");
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

#[test]
fn ladder_on_a_final_ledger_reads_once() {
    let dir = scratch("ladder");
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
    let run = || {
        rcsr(&[
            "ladder", "--data", &d, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2",
            "--windows", &w, "--scheme", "locked", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50",
            "--max-fit-rows", "600", "--out", &out, "--locked-test", "--ledger", &l, "--sealed", &sealed,
        ])
    };
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
