//! #1010: the guarded box comes from the declared lock, not from the 2026H2 literal.
//! Fixtures are synthetic archives with months 2026-06 to 2026-10.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::lockbox::{self, Lockbox};
use rcsr::panel_v2::{self, ValidateOptions};
use rcsr::protocol::{Candidate, Ledger, LockedRange};
use scratch::Scratch;
use std::path::{Path, PathBuf};

/// 2026-06-01, 2026-07-01, 2026-08-01, 2026-09-01, 2026-10-01 (UTC).
const MONTHS: [(&str, i64); 5] = [
    ("2026-06", 1_780_272_000),
    ("2026-07", 1_782_864_000),
    ("2026-08", 1_785_542_400),
    ("2026-09", 1_788_220_800),
    ("2026-10", 1_790_812_800),
];
const FORWARD: &str = "2027-01-01,2027-07-01";
const OVERLAPPING: &str = "2026-07-01,2027-07-01";
const HOURS_PER_MONTH: i64 = 48;

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let s = Scratch::new(name);
    let d = s.path().to_path_buf();
    (s, d)
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = rcsr_cmd::rcsr().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).to_string(),
        String::from_utf8_lossy(&out.stderr).to_string(),
    )
}

fn quarter(t: i64, k: i64) -> String {
    let p = 10.0 * (1.0 + 0.001 * (k % 4) as f64);
    let (o, c) = (p, p * 1.0005);
    let (h, l) = (c * 1.001, o * 0.999);
    let v = 10.0 + (k % 7) as f64;
    let qv = v * (o + c) / 2.0;
    let raw = t * 1000;
    format!("{raw},{o},{h},{l},{c},{v},{raw},{qv},{},{},{},0", 5 + k % 3, v * 0.4, qv * 0.4)
}

/// One symbol with `HOURS_PER_MONTH` full hours at the start of each month in `months`.
fn archive(root: &Path, months: &[usize]) -> PathBuf {
    let raw = root.join("raw");
    let dir = raw.join("AAAUSDT");
    std::fs::create_dir_all(&dir).unwrap();
    for &i in months {
        let (name, start) = MONTHS[i];
        let lines: Vec<String> = (0..HOURS_PER_MONTH * 4).map(|q| quarter(start + q * 900, q)).collect();
        let file = format!("AAAUSDT-15m-{name}");
        let csv = lines.join("\n") + "\n";
        std::fs::write(dir.join(format!("{file}.zip")), stored_zip(&format!("{file}.csv"), csv.as_bytes())).unwrap();
    }
    raw
}

fn membership(path: &Path) {
    std::fs::write(path, format!("as_of,symbol,member,adv_usd,n_bars\n{},AAAUSDT,1,1000.0,100\n", MONTHS[0].1)).unwrap();
}

fn base_args<'a>(raw: &'a str, out: &'a str, extra: &[&'a str]) -> Vec<&'a str> {
    let mut a = vec!["panel", "base", "--input", raw, "--out", out];
    a.extend_from_slice(extra);
    a
}

fn all_months() -> Vec<usize> {
    vec![0, 1, 2, 3, 4]
}

#[test]
fn lockbox_declared_range_not_literal_base() {
    let (_s, dir) = scratch("literal-base");
    let raw = s(&archive(&dir, &all_months()));
    let out = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &out, &["--locked-test", FORWARD]));
    assert_eq!(code, 0, "{err}");
    let stored = lockbox::stored(&out).unwrap().map(|(_, l)| l);
    assert_eq!(stored, Some(Lockbox::parse(FORWARD).unwrap()));
}

#[test]
fn lockbox_declared_range_not_literal_build() {
    let (_s, dir) = scratch("literal-build");
    let raw = s(&archive(&dir, &all_months()));
    let base = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &base, &["--locked-test", FORWARD]));
    assert_eq!(code, 0, "{err}");
    let m = dir.join("membership.csv");
    membership(&m);
    let panel = s(&dir.join("panel.rcsb"));
    let (code, _, err) = rcsr(&["panel", "build", "--data", &base, "--membership", &s(&m), "--out", &panel]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(lockbox::stored(&panel).unwrap().map(|(_, l)| l), Some(Lockbox::parse(FORWARD).unwrap()));
}

#[test]
fn lockbox_declared_range_not_literal_enrich() {
    let (_s, dir) = scratch("literal-enrich");
    let raw = s(&archive(&dir, &all_months()));
    let base = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &base, &["--locked-test", FORWARD]));
    assert_eq!(code, 0, "{err}");
    let out = s(&dir.join("fam.rcsb"));
    let (code, _, err) = rcsr(&[
        "enrich", "--data", &base, "--out", &out, "--cpu", "--features", "r1", "--label", "long:1h,name=h1,fee=10,vol=20",
        "--locked-test", FORWARD,
    ]);
    assert_eq!(code, 0, "{err}");
}

#[test]
fn lockbox_declared_overlap_refused() {
    let (_s, dir) = scratch("overlap");
    let raw = s(&archive(&dir, &all_months()));
    let out = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &out, &["--locked-test", OVERLAPPING]));
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("2026-07-01..2027-07-01") && err.contains("--locked-test"), "{err}");
    assert!(!Path::new(&out).exists());
    let (code, _, err) = rcsr(&base_args(&raw, &out, &["--locked-test", OVERLAPPING, "--allow-lockbox"]));
    assert_eq!(code, 0, "{err}");

    // the build inherits the declared range and refuses its rows too
    let m = dir.join("membership.csv");
    membership(&m);
    let panel = s(&dir.join("panel.rcsb"));
    let build = ["panel", "build", "--data", &out, "--membership", &s(&m), "--out", &panel];
    let (code, _, err) = rcsr(&build);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("declared lock"), "{err}");
    let mut allowed = build.to_vec();
    allowed.push("--allow-lockbox");
    let (code, _, err) = rcsr(&allowed);
    assert_eq!(code, 0, "{err}");

    // enrich refuses labelled rows inside the declared lock
    let fam = s(&dir.join("fam.rcsb"));
    let enrich = ["enrich", "--data", &out, "--out", &fam, "--cpu", "--features", "r1", "--label", "long:1h,name=h1,fee=10,vol=20"];
    let (code, _, err) = rcsr(&enrich);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("declared lock"), "{err}");
}

#[test]
fn lockbox_legacy_default_unchanged() {
    let (_s, dir) = scratch("legacy");
    let raw = s(&archive(&dir, &all_months()));
    let out = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &out, &[]));
    assert_eq!(code, 3, "{err}");
    let file = dir.join("raw/AAAUSDT/AAAUSDT-15m-2026-07.zip");
    let want = format!(
        "error: panel base ({}): window [1782864000, 1785542400) overlaps the lockbox [1782864000, 1798761600) (2026H2); \
         lock the holdout and use --allow-lockbox only for a declared final test\n",
        file.display()
    );
    assert_eq!(err, want);
    assert!(!Path::new(&out).exists());
    // development months before the box build without a declaration
    let raw_dev = s(&archive(&dir.join("dev"), &[0]));
    let out_dev = s(&dir.join("dev.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw_dev, &out_dev, &[]));
    assert_eq!(code, 0, "{err}");
}

fn candidate(tag: &str) -> Candidate {
    let spec = format!("spec-{tag}");
    Candidate {
        id: Candidate::address(&spec, "data", "build"),
        spec_sha256: spec,
        data_sha256: "data".to_string(),
        build_commit: "build".to_string(),
        report_sha256: format!("report-{tag}"),
        expression: format!("x + {}", tag.len()),
        locked_test: None,
    }
}

/// A verified record that retires the protocol default 2025-07-01..2026-07-01.
fn retired_record(dir: &Path) -> String {
    let path = dir.join("ledger.jsonl");
    let ledger = Ledger::new(&path);
    let a = candidate("a");
    ledger.add_candidate(&a).unwrap();
    ledger.seal(&dir.join("sealed.json"), &[]).unwrap();
    ledger.record_read(&a.id, "locked-a", Some(&LockedRange::new(&lockbox::protocol_default(), "store"))).unwrap();
    let out = s(&dir.join("retired.json"));
    let (code, _, err) = rcsr(&["protocol", "retire", "--ledger", &s(&path), "--range", "2025-07-01,2026-07-01", "--out", &out]);
    assert_eq!(code, 0, "{err}");
    out
}

#[test]
fn lockbox_retired_drops_literal() {
    let (_s, dir) = scratch("retired");
    let raw = s(&archive(&dir, &all_months()));
    let record = retired_record(&dir);
    let out = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &out, &["--locked-test", FORWARD, "--retired", &record]));
    assert_eq!(code, 0, "{err}");
    let text = std::fs::read_to_string(format!("{out}.manifest.json")).unwrap();
    assert!(text.contains("\"retired\""), "{text}");
    // the same record does not free a lock that still covers the data
    let again = s(&dir.join("again.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &again, &["--locked-test", OVERLAPPING, "--retired", &record]));
    assert_eq!(code, 3, "{err}");
}

fn validate(panel: &Path) -> panel_v2::Report {
    let opts = panel_v2::PanelOptions::new("", "", "");
    panel_v2::validate(&ValidateOptions {
        panel: s(panel),
        base: None,
        membership: None,
        replay: 0,
        panel_options: opts,
    })
}

#[test]
fn validate_v11_uses_declared_lock() {
    let (_s, dir) = scratch("v11");
    let raw = s(&archive(&dir, &all_months()));
    let m = dir.join("membership.csv");
    membership(&m);

    // development store: V11 counts nothing inside the forward lock
    let base = s(&dir.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &base, &["--locked-test", FORWARD]));
    assert_eq!(code, 0, "{err}");
    let dev = dir.join("dev.rcsb");
    let (code, _, err) = rcsr(&["panel", "build", "--data", &base, "--membership", &s(&m), "--out", &s(&dev)]);
    assert_eq!(code, 0, "{err}");
    let v11 = validate(&dev).checks.into_iter().find(|c| c.id == "V11").unwrap();
    assert_eq!((v11.status, v11.violations), ("pass", 0), "{}", v11.detail);
    assert!(v11.detail.contains("2027-01-01..2027-07-01"), "{}", v11.detail);

    // a store that holds rows inside its declared lock fails with that row count
    let locked = "2026-08-01,2026-09-01";
    let lbase = s(&dir.join("lbase.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw, &lbase, &["--locked-test", locked, "--allow-lockbox"]));
    assert_eq!(code, 0, "{err}");
    let lpanel = dir.join("lpanel.rcsb");
    let (code, _, err) = rcsr(&["panel", "build", "--data", &lbase, "--membership", &s(&m), "--out", &s(&lpanel), "--allow-lockbox"]);
    assert_eq!(code, 0, "{err}");
    let lb = Lockbox::parse(locked).unwrap();
    let table = rcsr::store::FileStore::open(&s(&lpanel)).unwrap().to_table().unwrap();
    let ts = table.index_of("ts").unwrap();
    let inside = (0..table.len()).filter(|&r| lb.contains(table.value(r, ts) as i64)).count();
    assert!(inside > 0);
    let v11 = validate(&lpanel).checks.into_iter().find(|c| c.id == "V11").unwrap();
    assert_eq!((v11.status, v11.violations), ("fail", inside), "{}", v11.detail);

    // an undeclared store keeps the legacy box
    let plain = dir.join("plain");
    std::fs::create_dir_all(&plain).unwrap();
    let raw_dev = s(&archive(&plain, &[0]));
    let pbase = s(&plain.join("base.rcsb"));
    let (code, _, err) = rcsr(&base_args(&raw_dev, &pbase, &[]));
    assert_eq!(code, 0, "{err}");
    let ppanel = plain.join("panel.rcsb");
    let (code, _, err) = rcsr(&["panel", "build", "--data", &pbase, "--membership", &s(&m), "--out", &s(&ppanel)]);
    assert_eq!(code, 0, "{err}");
    let v11 = validate(&ppanel).checks.into_iter().find(|c| c.id == "V11").unwrap();
    assert_eq!((v11.status, v11.violations), ("pass", 0));
    assert!(v11.detail.contains("legacy default"), "{}", v11.detail);
}

fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let crc = crc32(data);
    out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(data);
    let cd = out.len();
    out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    let cd_size = out.len() - cd;
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(cd_size as u32).to_le_bytes());
    out.extend_from_slice(&(cd as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let m = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & m);
        }
    }
    !crc
}

