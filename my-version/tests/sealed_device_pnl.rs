//! #823: the pilot's sealed read is `walkforward --locked-test --device-pnl` behind a final seal.
//! The flow below follows `main.rs` step by step with library calls, so the device work runs in
//! the test process and a sanitizer sees it without a child process.

#![cfg(feature = "cuda")]

use rcsr::lockbox::{self, Lockbox};
use rcsr::protocol::{Candidate, Ledger};
use rcsr::walkforward::{self, EvalSpec, Policy, ProtocolOptions, Report, Window};
use std::path::Path;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const T0: i64 = 1_704_067_200;
const H: i64 = 3600;

fn store(dir: &Path) -> String {
    let path = dir.join("d.csv");
    let mut csv = String::from("ts,x,y\n");
    for i in 0..400i64 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        csv.push_str(&format!("{},{},{}\n", T0 + i * H, x, y));
    }
    std::fs::write(&path, csv).unwrap();
    let d = path.to_str().unwrap().to_string();
    lockbox::write_sidecar(&d, &Lockbox { start: T0 + 300 * H, end: T0 + 400 * H }, "test").unwrap();
    d
}

fn spec(device_pnl: bool, label: &str, a: i64, b: i64, c: i64, d: i64) -> EvalSpec {
    EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "y".to_string(),
        time: "ts".to_string(),
        group: None,
        windows: vec![Window {
            label: label.to_string(),
            train_start: (T0 + a * H) as f64,
            train_end: (T0 + b * H) as f64,
            test_start: (T0 + c * H) as f64,
            test_end: (T0 + d * H) as f64,
        }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 5.0,
        threshold_quantile: 0.70,
        threshold_fixed: None,
        validation_frac: 0.3,
        periods_per_year: Some(365.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl,
        control_seed: 0,
        policy: Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    }
}

struct Sealed {
    _scratch: Scratch,
    data: String,
    ledger: Ledger,
    sealed: std::path::PathBuf,
    id: String,
    data_sha: String,
}

/// A development registration and a final seal, as the pilot's `seal` step makes them.
fn sealed_store(device_pnl: bool) -> (Sealed, EvalSpec, ProtocolOptions) {
    let scratch = Scratch::new("sealed-device-pnl");
    let dir = scratch.path().to_path_buf();
    let data = store(&dir);
    let ledger = Ledger::new(dir.join("ledger.jsonl"));
    let dev = spec(device_pnl, "dev", 0, 150, 160, 250);
    let opts = ProtocolOptions { final_read: true, ..ProtocolOptions::default() };
    let data_sha = rcsr::manifest::sha256_file(&data).unwrap();
    let spec_sha = rcsr::manifest::sha256_hex(rcsr::protocol::procedure_spec(&dev, &opts).as_bytes());
    let id = Candidate::address(&spec_sha, &data_sha, "test");
    let dev_report = walkforward::run_path_with(&data, &dev, &ProtocolOptions::default()).unwrap();
    ledger
        .add_candidate(&Candidate {
            id: id.clone(),
            spec_sha256: spec_sha,
            data_sha256: data_sha.clone(),
            build_commit: "test".to_string(),
            report_sha256: rcsr::manifest::sha256_hex(dev_report.hashed_json().as_bytes()),
            expression: "x".to_string(),
            locked_test: None,
        })
        .unwrap();
    let sealed = dir.join("sealed.json");
    ledger.seal_final(&sealed, &[], 1, false).unwrap();
    let locked = spec(device_pnl, "locked", 0, 250, 300, 400);
    (Sealed { _scratch: scratch, data, ledger, sealed, id, data_sha }, locked, opts)
}

fn authorize(s: &Sealed) -> Result<(), String> {
    s.ledger.verify_chain()?;
    s.ledger.authorize_read(&s.sealed, &s.id, &s.data_sha, "test", None)?;
    s.ledger.check_read_marker(&s.data)
}

fn run(s: &Sealed, spec: &EvalSpec, opts: &ProtocolOptions) -> (Report, String) {
    let report = walkforward::run_path_with(&s.data, spec, opts).unwrap();
    let sha = rcsr::manifest::sha256_hex(report.hashed_json().as_bytes());
    (report, sha)
}

#[test]
fn a_sealed_device_pnl_read_runs_on_the_device_and_writes_its_marker() {
    let (s, locked, opts) = sealed_store(false);
    authorize(&s).unwrap();
    let (report, sha) = run(&s, &locked, &opts);
    let summary = report.device_pnl.as_ref().expect("a --device-pnl summary");
    assert_eq!(summary.backend, "cuda", "the reduction fell back to the host");
    assert_eq!(summary.device_windows, summary.windows);
    assert_eq!((summary.bit_mismatch_windows, summary.count_mismatch_windows), (0, 0));
    for w in &report.windows {
        let d = w.device_pnl.as_ref().expect("a device block per window");
        assert!(d.bit_equal && d.counts_equal, "window {}: bit_equal {} counts_equal {}", w.window.label, d.bit_equal, d.counts_equal);
    }
    assert!(!Path::new(&rcsr::protocol::marker_path(&s.data)).exists(), "the marker is written after the run");
    s.ledger.write_read_marker(&s.data, "test_read", &sha).unwrap();
    s.ledger.record_read(&s.id, &sha, None).unwrap();
    let marker = std::fs::read_to_string(rcsr::protocol::marker_path(&s.data)).unwrap();
    assert!(marker.contains("\"event\":\"test_read\"") && marker.contains(&sha), "{marker}");
    assert_eq!(s.ledger.read_count().unwrap(), 1);
    let again = authorize(&s).unwrap_err();
    assert!(again.contains("read once") || again.contains("test_read:true"), "{again}");
}

#[test]
fn a_crash_between_the_marker_and_the_ledger_record_leaves_the_read_retryable() {
    let (s, locked, opts) = sealed_store(true);
    authorize(&s).unwrap();
    let (_, sha) = run(&s, &locked, &opts);
    s.ledger.write_read_marker(&s.data, "test_read", &sha).unwrap();
    assert_eq!(s.ledger.read_count().unwrap(), 0, "the crash happens before the ledger record");
    authorize(&s).expect("a retry after the crash is authorised");
    let (_, again) = run(&s, &locked, &opts);
    assert_eq!(again, sha, "the retry reproduces the report hash");
    s.ledger.write_read_marker(&s.data, "test_read", &again).unwrap();
    let lines = std::fs::read_to_string(rcsr::protocol::marker_path(&s.data)).unwrap().lines().count();
    assert_eq!(lines, 1, "the retry adds no second marker line");
    s.ledger.record_read(&s.id, &again, None).unwrap();
    assert_eq!(s.ledger.read_count().unwrap(), 1);
}
