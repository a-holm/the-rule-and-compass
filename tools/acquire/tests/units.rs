use rcsr_acquire::calendar::{check_published, iso_from_unix, range, Month};
use rcsr_acquire::json::quote;
use rcsr_acquire::listing::{months_from_keys, page_url, parse_page, symbols};
use rcsr_acquire::sha256::{sha256_file, sha256_hex, Sha256};
use rcsr_acquire::transport::parse_headers;
use rcsr_acquire::verify::parse_checksum;

mod common;
use common::{fixture, fixture_text};

#[test]
fn sha256_fips_vectors() {
    assert_eq!(sha256_hex(b""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert_eq!(sha256_hex(b"abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert_eq!(
        sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
    );
}

#[test]
fn sha256_million_a_and_chunking() {
    let data = vec![b'a'; 1_000_000];
    let want = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    assert_eq!(sha256_hex(&data), want);
    let mut s = Sha256::new();
    for chunk in data.chunks(77) {
        s.update(chunk);
    }
    assert_eq!(s.finalize(), want);
}

#[test]
fn sha256_file_matches_bytes() {
    let dir = common::scratch("sha_file");
    let p = dir.join("x.bin");
    std::fs::write(&p, b"abc").unwrap();
    assert_eq!(sha256_file(&p).unwrap(), sha256_hex(b"abc"));
}

#[test]
fn month_parse_range_and_last_day() {
    let a = Month::parse("2025-11").unwrap();
    let b = Month::parse("2026-02").unwrap();
    let all: Vec<String> = range(a, b).iter().map(|m| m.to_string()).collect();
    assert_eq!(all, ["2025-11", "2025-12", "2026-01", "2026-02"]);
    assert!(range(b, a).is_empty());
    assert_eq!(Month::parse("2024-02").unwrap().last_day(), "2024-02-29");
    assert_eq!(Month::parse("2026-09").unwrap().last_day(), "2026-09-30");
    for bad in ["2026-13", "2026-00", "26-01", "2026/01", "2026-1", ""] {
        assert!(Month::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn unpublished_month_is_refused() {
    let now = Month::parse("2026-10").unwrap();
    assert!(check_published(Month::parse("2026-09").unwrap(), now).is_ok());
    assert!(check_published(now, now).is_err());
    assert!(check_published(Month::parse("2026-12").unwrap(), now).is_err());
}

#[test]
fn iso_timestamps() {
    assert_eq!(iso_from_unix(0), "1970-01-01T00:00:00Z");
    assert_eq!(iso_from_unix(1_792_475_799), "2026-10-20T05:56:39Z");
    assert_eq!(iso_from_unix(951_782_400), "2000-02-29T00:00:00Z");
}

#[test]
fn json_quote_escapes() {
    assert_eq!(quote("a\"b\\c\n\u{1}"), "\"a\\\"b\\\\c\\n\\u0001\"");
}

#[test]
fn headers_of_http2_responses() {
    let r = parse_headers(&fixture_text("checksum_2026-09.headers.txt")).unwrap();
    assert_eq!(r.status, 200);
    assert_eq!(r.header("ETag"), Some("\"8bfefe8388dff21f0f85cd95296a5b9c\""));
    assert_eq!(r.header("last-modified"), Some("Mon, 05 Oct 2026 11:13:54 GMT"));
    let r = parse_headers(&fixture_text("missing_2026-10.headers.txt")).unwrap();
    assert_eq!(r.status, 404);
    let redirected = "HTTP/1.1 301 Moved\r\nLocation: x\r\n\r\nHTTP/1.1 200 OK\r\nETag: \"e\"\r\n\r\n";
    let r = parse_headers(redirected).unwrap();
    assert_eq!((r.status, r.header("etag")), (200, Some("\"e\"")));
    assert!(parse_headers("garbage").is_none());
}

#[test]
fn checksum_file_parses() {
    let body = fixture_text("BTCUSDT-15m-2026-09.zip.CHECKSUM");
    assert_eq!(
        parse_checksum(&body).unwrap(),
        "0d452e884c1bdd20bbb414b8feab30753e8e9ebef3e8c821d8a54eef48859cd0"
    );
    assert_eq!(parse_checksum(&body.to_uppercase()).unwrap().len(), 64);
    assert!(parse_checksum("<Error><Code>NoSuchKey</Code></Error>").is_none());
    assert!(parse_checksum("").is_none());
    assert!(parse_checksum(&format!("{}  f.zip", "g".repeat(64))).is_none());
}

#[test]
fn listing_page_of_btcusdt() {
    let page = parse_page(&fixture_text("listing_btcusdt_15m.xml"));
    assert!(!page.truncated);
    assert_eq!(page.next_marker, None);
    assert_eq!(page.keys.len(), 220);
    let months = months_from_keys(&page.keys, "15m");
    assert_eq!(months.len(), 110);
    assert_eq!(months.first().unwrap().to_string(), "2017-08");
    assert_eq!(months.last().unwrap().to_string(), "2026-09");
    assert!(months_from_keys(&page.keys, "1h").is_empty());
}

#[test]
fn listing_pages_carry_markers_and_prefixes() {
    let p1 = parse_page(&fixture_text("symbols_page1.xml"));
    assert!(p1.truncated);
    assert_eq!(p1.next_marker.as_deref(), Some("data/spot/monthly/klines/1000SATSUSDT/"));
    assert_eq!(p1.prefixes.len(), 2);
    assert!(p1.keys.is_empty());
    let p2 = parse_page(&fixture_text("symbols_page2.xml"));
    assert!(!p2.truncated && p2.next_marker.is_none());
    let all: Vec<String> = p1.prefixes.into_iter().chain(p2.prefixes).collect();
    assert_eq!(symbols(&all, "data/spot/monthly/klines/"), ["0GUSDT", "1000SATSUSDT", "BTCUSDT"]);
    let k1 = parse_page(&fixture_text("keys_page1.xml"));
    assert_eq!(k1.next_marker.as_deref(), Some("data/spot/monthly/klines/ETHUSDT/15m/ETHUSDT-15m-2017-08.zip.CHECKSUM"));
}

#[test]
fn page_url_encodes_prefix_and_marker() {
    let u = page_url("https://h/b", "data/a b/", true, Some("data/a b/x.zip"));
    assert_eq!(u, "https://h/b?prefix=data%2Fa%20b%2F&max-keys=1000&delimiter=%2F&marker=data%2Fa%20b%2Fx.zip");
    assert_eq!(page_url("https://h/b", "p/", false, None), "https://h/b?prefix=p%2F&max-keys=1000");
}

#[test]
fn error_body_is_not_a_listing() {
    let page = parse_page(std::str::from_utf8(&fixture("missing_2026-10.xml")).unwrap());
    assert!(page.keys.is_empty() && page.prefixes.is_empty() && !page.truncated);
}
