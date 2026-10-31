//! S3 `ListObjects` (v1) pages of the archive bucket.

use crate::calendar::Month;
use crate::transport::{get_retry, Transport};
use std::fs;
use std::path::Path;

/// One parsed listing page.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Page {
    pub keys: Vec<String>,
    pub prefixes: Vec<String>,
    pub truncated: bool,
    pub next_marker: Option<String>,
}

fn tag_values<'a>(xml: &'a str, tag: &str) -> Vec<&'a str> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut rest = xml;
    while let Some(i) = rest.find(&open) {
        let after = &rest[i + open.len()..];
        match after.find(&close) {
            Some(j) => {
                out.push(&after[..j]);
                rest = &after[j + close.len()..];
            }
            None => break,
        }
    }
    out
}

fn unescape(s: &str) -> String {
    s.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Parses one listing page. Only `<Key>` inside `<Contents>` and `<Prefix>` inside
/// `<CommonPrefixes>` count: the top-level `<Prefix>` echoes the request.
pub fn parse_page(xml: &str) -> Page {
    let _ = xml;
    Page::default()
}

fn pct_encode(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

/// URL of one listing page.
pub fn page_url(base: &str, prefix: &str, delimiter: bool, marker: Option<&str>) -> String {
    let mut url = format!("{base}?prefix={}&max-keys=1000", pct_encode(prefix));
    if delimiter {
        url.push_str("&delimiter=%2F");
    }
    if let Some(m) = marker {
        url.push_str("&marker=");
        url.push_str(&pct_encode(m));
    }
    url
}

/// All keys and common prefixes under `prefix`, following `marker` until the listing ends.
/// `scratch` is a file path the pages are written to.
pub fn list_all(
    t: &dyn Transport,
    base: &str,
    prefix: &str,
    delimiter: bool,
    scratch: &Path,
    retries: u32,
    backoff_ms: u64,
) -> Result<(Vec<String>, Vec<String>), String> {
    let mut keys = Vec::new();
    let mut prefixes = Vec::new();
    let mut marker: Option<String> = None;
    loop {
        let url = page_url(base, prefix, delimiter, marker.as_deref());
        let resp = get_retry(t, &url, scratch, retries, backoff_ms)?;
        if resp.status != 200 {
            let _ = fs::remove_file(scratch);
            return Err(format!("listing {url}: HTTP {}", resp.status));
        }
        let body = fs::read(scratch).map_err(|e| format!("read listing: {e}"))?;
        let _ = fs::remove_file(scratch);
        let page = parse_page(&String::from_utf8_lossy(&body));
        keys.extend(page.keys);
        prefixes.extend(page.prefixes);
        match page.next_marker {
            Some(m) if marker.as_deref() != Some(m.as_str()) => marker = Some(m),
            Some(_) => return Err(format!("listing {url}: marker did not advance")),
            None => return Ok((keys, prefixes)),
        }
    }
}

/// Symbol names from common prefixes like `data/spot/monthly/klines/BTCUSDT/`.
pub fn symbols(prefixes: &[String], root: &str) -> Vec<String> {
    let _ = (prefixes, root);
    Vec::new()
}

/// Months of the `<SYM>-<interval>-YYYY-MM.zip` keys. `.CHECKSUM` siblings do not match.
pub fn months_from_keys(keys: &[String], interval: &str) -> Vec<Month> {
    let _ = (keys, interval);
    Vec::new()
}
