use rcsr_acquire::calendar::Month;
use rcsr_acquire::fetch::{self, csv_path, zip_path, zip_url, FetchConfig, Summary};
use rcsr_acquire::listing::{list_all, months_from_keys, page_url, symbols};
use rcsr_acquire::sha256::sha256_hex;
use rcsr_acquire::verify::{self, snapshot_json, VerifyConfig};
use std::fs;
use std::path::Path;

mod common;
use common::{fixture, scratch, Fake};

const BASE: &str = "https://example.test";
const LIST: &str = "https://example.test/bucket";

fn m(s: &str) -> Month {
    Month::parse(s).unwrap()
}

fn fetch_cfg(out: &Path, skip_404: bool) -> FetchConfig {
    FetchConfig {
        base_url: BASE.into(),
        out: out.to_path_buf(),
        interval: "15m".into(),
        symbols: vec!["BTCUSDT".into(), "NEWUSDT".into()],
        from: m("2026-08"),
        to: m("2026-09"),
        jobs: 3,
        skip_404,
        retries: 2,
        backoff_ms: 0,
    }
}

fn zip_bytes(tag: u8) -> Vec<u8> {
    vec![tag; 2000]
}

fn serving() -> Fake {
    let mut f = Fake::default();
    f.route(&zip_url(BASE, "BTCUSDT", "15m", m("2026-08")), 200, &[], &zip_bytes(1));
    f.route(&zip_url(BASE, "BTCUSDT", "15m", m("2026-09")), 200, &[], &zip_bytes(2));
    let missing = fixture("missing_2026-10.xml");
    f.route(&zip_url(BASE, "NEWUSDT", "15m", m("2026-08")), 404, &[], &missing);
    f.route(&zip_url(BASE, "NEWUSDT", "15m", m("2026-09")), 404, &[], &missing);
    f
}

fn rows(out: &Path) -> Vec<String> {
    fs::read_to_string(csv_path(out, "15m")).unwrap().lines().map(String::from).collect()
}

#[test]
fn listing_follows_markers_across_pages() {
    let dir = scratch("list_pages");
    let mut f = Fake::default();
    let root = "data/spot/monthly/klines/";
    f.route(&page_url(LIST, root, true, None), 200, &[], &fixture("symbols_page1.xml"));
    f.route(
        &page_url(LIST, root, true, Some("data/spot/monthly/klines/1000SATSUSDT/")),
        200,
        &[],
        &fixture("symbols_page2.xml"),
    );
    let (keys, prefixes) = list_all(&f, LIST, root, true, &dir.join("p.xml"), 1, 0).unwrap();
    assert!(keys.is_empty());
    assert_eq!(symbols(&prefixes, root), ["0GUSDT", "1000SATSUSDT", "BTCUSDT"]);
    assert_eq!(f.count(), 2);
    assert!(!dir.join("p.xml").exists());

    let prefix = "data/spot/monthly/klines/ETHUSDT/15m/";
    let mut f = Fake::default();
    f.route(&page_url(LIST, prefix, false, None), 200, &[], &fixture("keys_page1.xml"));
    f.route(
        &page_url(LIST, prefix, false, Some("data/spot/monthly/klines/ETHUSDT/15m/ETHUSDT-15m-2017-08.zip.CHECKSUM")),
        200,
        &[],
        &fixture("keys_page2.xml"),
    );
    let (keys, _) = list_all(&f, LIST, prefix, false, &dir.join("p.xml"), 1, 0).unwrap();
    let months: Vec<String> = months_from_keys(&keys, "15m").iter().map(|x| x.to_string()).collect();
    assert_eq!(months, ["2017-08", "2017-09"]);
}

#[test]
fn listing_http_error_is_an_error() {
    let dir = scratch("list_error");
    let mut f = Fake::default();
    f.route(&page_url(LIST, "p/", false, None), 403, &[], b"denied");
    assert!(list_all(&f, LIST, "p/", false, &dir.join("p.xml"), 1, 0).is_err());
}

#[test]
fn fetch_lays_out_zips_and_records_every_request() {
    let out = scratch("fetch_layout");
    let f = serving();
    let s = fetch::run(&fetch_cfg(&out, false), &f).unwrap();
    assert_eq!(
        s,
        Summary { planned: 4, present: 0, known_missing: 0, requests: 4, ok: 2, missing: 2, failed: 0 }
    );
    let p = out.join("raw_15m/BTCUSDT/BTCUSDT-15m-2026-09.zip");
    assert_eq!(p, zip_path(&out, "15m", "BTCUSDT", m("2026-09")));
    assert_eq!(fs::read(&p).unwrap(), zip_bytes(2));
    assert!(!out.join("raw_15m/NEWUSDT/NEWUSDT-15m-2026-08.zip").exists());
    assert!(!out.join("raw_15m/BTCUSDT/BTCUSDT-15m-2026-09.zip.part").exists());
    let r = rows(&out);
    assert_eq!(r[0], "symbol,interval,month,http_status,bytes,fetched_at");
    assert_eq!(r.len(), 5);
    let mut body: Vec<String> = r[1..].iter().map(|l| l.rsplitn(2, ',').nth(1).unwrap().to_string()).collect();
    body.sort();
    assert_eq!(
        body,
        [
            "BTCUSDT,15m,2026-08,200,2000",
            "BTCUSDT,15m,2026-09,200,2000",
            "NEWUSDT,15m,2026-08,404,0",
            "NEWUSDT,15m,2026-09,404,0"
        ]
    );
    assert!(r[1].ends_with('Z'));
}

#[test]
fn second_run_requests_no_existing_zip() {
    let out = scratch("fetch_resume");
    let f = serving();
    fetch::run(&fetch_cfg(&out, false), &f).unwrap();
    let before = rows(&out);
    let f2 = serving();
    let s = fetch::run(&fetch_cfg(&out, false), &f2).unwrap();
    let urls = f2.requests.lock().unwrap().clone();
    assert_eq!(s.present, 2);
    assert!(urls.iter().all(|u| u.contains("NEWUSDT")), "{urls:?}");
    assert_eq!(s.requests, 2);
    assert_eq!(rows(&out).len(), before.len() + 2);
}

#[test]
fn skip_404_makes_no_request_for_known_missing_months() {
    let out = scratch("fetch_skip404");
    fetch::run(&fetch_cfg(&out, false), &serving()).unwrap();
    let rows_before = rows(&out).len();
    let f = serving();
    let s = fetch::run(&fetch_cfg(&out, true), &f).unwrap();
    assert_eq!((f.count(), s.requests, s.present, s.known_missing), (0, 0, 2, 2));
    assert_eq!(rows(&out).len(), rows_before);
}

#[test]
fn small_zip_is_replaced_and_a_failed_download_leaves_no_file() {
    let out = scratch("fetch_small");
    let p = zip_path(&out, "15m", "BTCUSDT", m("2026-08"));
    fs::create_dir_all(p.parent().unwrap()).unwrap();
    fs::write(&p, vec![0u8; 1000]).unwrap();
    let mut f = serving();
    f.route(&zip_url(BASE, "BTCUSDT", "15m", m("2026-09")), 503, &[], b"busy");
    let s = fetch::run(&fetch_cfg(&out, false), &f).unwrap();
    assert_eq!(fs::read(&p).unwrap(), zip_bytes(1));
    assert_eq!((s.ok, s.failed), (1, 1));
    assert!(!zip_path(&out, "15m", "BTCUSDT", m("2026-09")).exists());
    assert!(!out.join("raw_15m/BTCUSDT/BTCUSDT-15m-2026-09.zip.part").exists());
    assert!(rows(&out).iter().any(|l| l.starts_with("BTCUSDT,15m,2026-09,503,0,")));
    // 503 is retried: two attempts for the failing month.
    let tries = f.requests.lock().unwrap().iter().filter(|u| u.contains("BTCUSDT-15m-2026-09")).count();
    assert_eq!(tries, 2);
}

#[test]
fn transport_error_is_status_zero() {
    let out = scratch("fetch_neterr");
    let f = Fake::default();
    let s = fetch::run(&fetch_cfg(&out, false), &f).unwrap();
    assert_eq!((s.requests, s.failed), (4, 4));
    assert!(rows(&out).iter().skip(1).all(|l| l.contains(",0,0,")));
}

fn verify_cfg(dir: &Path) -> VerifyConfig {
    VerifyConfig {
        base_url: BASE.into(),
        dir: dir.to_path_buf(),
        interval: "15m".into(),
        symbols: None,
        from: None,
        to: None,
        jobs: 2,
        retries: 1,
        backoff_ms: 0,
    }
}

fn checksum_route(f: &mut Fake, sym: &str, month: &str, sha: &str) {
    let url = format!("{}.CHECKSUM", zip_url(BASE, sym, "15m", m(month)));
    let body = format!("{sha}  {sym}-15m-{month}.zip");
    f.route(&url, 200, &[("ETag", "\"abc\""), ("Last-Modified", "Mon, 05 Oct 2026 11:13:54 GMT")], body.as_bytes());
}

#[test]
fn verify_matches_vendor_checksums_and_writes_snapshot() {
    let out = scratch("verify_ok");
    fetch::run(&fetch_cfg(&out, false), &serving()).unwrap();
    let mut f = Fake::default();
    checksum_route(&mut f, "BTCUSDT", "2026-08", &sha256_hex(&zip_bytes(1)));
    checksum_route(&mut f, "BTCUSDT", "2026-09", &sha256_hex(&zip_bytes(2)));
    let entries = verify::run(&verify_cfg(&out), &f).unwrap();
    assert_eq!(entries.len(), 2);
    assert!(entries.iter().all(|e| e.verified && e.http_status == 200 && e.bytes == 2000));
    assert_eq!(entries[0].month, m("2026-08"));
    assert_eq!(entries[1].etag.as_deref(), Some("\"abc\""));
    assert!(!out.join(".verify-tmp").exists());

    let json = snapshot_json(&entries, "15m", None, "2026-10-31T00:00:00Z");
    assert!(json.starts_with('{') && json.trim_end().ends_with('}'));
    assert!(json.contains("\"coverage_last\": \"2026-09-30\""), "{json}");
    assert!(json.contains("\"complete\": false"));
    assert!(json.contains("\"verified\": 2"));
    assert!(json.contains("\"vendor_url\": \"https://example.test/data/spot/monthly/klines/BTCUSDT/15m/BTCUSDT-15m-2026-09.zip.CHECKSUM\""));
    assert!(json.contains("\"etag\": \"\\\"abc\\\"\""));
    let done = snapshot_json(&entries, "15m", Some("2026-09-30"), "t");
    assert!(done.contains("\"complete\": true"));
    let short = snapshot_json(&entries, "15m", Some("2026-12-31"), "t");
    assert!(short.contains("\"complete\": false"));
}

#[test]
fn verify_flags_mismatch_missing_checksum_and_corruption() {
    let out = scratch("verify_bad");
    fetch::run(&fetch_cfg(&out, false), &serving()).unwrap();
    let mut f = Fake::default();
    checksum_route(&mut f, "BTCUSDT", "2026-08", &sha256_hex(b"other"));
    f.route(&format!("{}.CHECKSUM", zip_url(BASE, "BTCUSDT", "15m", m("2026-09"))), 404, &[], b"<Error/>");
    let entries = verify::run(&verify_cfg(&out), &f).unwrap();
    assert!(entries.iter().all(|e| !e.verified));
    assert_eq!(entries[0].http_status, 200);
    assert_eq!(entries[1].http_status, 404);
    assert!(entries[1].vendor_sha256.is_empty());
    let json = snapshot_json(&entries, "15m", Some("2026-09-30"), "t");
    assert!(json.contains("\"complete\": false") && json.contains("\"coverage_last\": null"));
    assert!(snapshot_json(&[], "15m", None, "t").contains("\"entries\": []"));
}

#[test]
fn verify_filters_symbols_and_months() {
    let out = scratch("verify_filter");
    fetch::run(&fetch_cfg(&out, false), &serving()).unwrap();
    let mut f = Fake::default();
    checksum_route(&mut f, "BTCUSDT", "2026-09", &sha256_hex(&zip_bytes(2)));
    let mut cfg = verify_cfg(&out);
    cfg.symbols = Some(vec!["BTCUSDT".into()]);
    cfg.from = Some(m("2026-09"));
    let entries = verify::run(&cfg, &f).unwrap();
    assert_eq!(entries.len(), 1);
    assert_eq!(f.count(), 1);
}
