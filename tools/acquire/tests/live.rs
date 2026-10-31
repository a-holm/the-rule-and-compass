//! Live check against data.binance.vision. Run with `cargo test --release -- --ignored`.
//! Fetches one zip and its checksum.

use rcsr_acquire::calendar::Month;
use rcsr_acquire::fetch::{self, FetchConfig};
use rcsr_acquire::transport::Curl;
use rcsr_acquire::verify::{self, VerifyConfig};
use rcsr_acquire::DATA_URL;
use std::path::PathBuf;

#[test]
#[ignore]
fn btcusdt_15m_2026_09_matches_the_recorded_vendor_hash() {
    let out = PathBuf::from(std::env::var("ACQUIRE_LIVE_DIR").unwrap_or_else(|_| {
        concat!(env!("CARGO_TARGET_TMPDIR"), "/live").to_string()
    }));
    let month = Month::parse("2026-09").unwrap();
    let curl = Curl::default();
    let cfg = FetchConfig {
        base_url: DATA_URL.into(),
        out: out.clone(),
        interval: "15m".into(),
        symbols: vec!["BTCUSDT".into()],
        from: month,
        to: month,
        jobs: 1,
        skip_404: false,
        retries: 3,
        backoff_ms: 1500,
    };
    let s = fetch::run(&cfg, &curl).unwrap();
    assert_eq!(s.failed, 0);
    let entries = verify::run(
        &VerifyConfig {
            base_url: DATA_URL.into(),
            dir: out,
            interval: "15m".into(),
            symbols: None,
            from: None,
            to: None,
            jobs: 1,
            retries: 3,
            backoff_ms: 1500,
        },
        &curl,
    )
    .unwrap();
    let e = entries.iter().find(|e| e.month == month).expect("entry");
    assert!(e.verified);
    assert_eq!(e.bytes, 157_264);
    assert_eq!(e.local_sha256, "0d452e884c1bdd20bbb414b8feab30753e8e9ebef3e8c821d8a54eef48859cd0");
}
