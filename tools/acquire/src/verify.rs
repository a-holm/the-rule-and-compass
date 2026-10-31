//! Checks local zips against the vendor `.CHECKSUM` sibling and builds the snapshot JSON.

use crate::calendar::{self, Month};
use crate::fetch::{zip_name, zip_path, zip_url};
use crate::json::{quote, quote_opt};
use crate::sha256::sha256_file;
use crate::transport::{get_retry, Transport};
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::thread;

pub struct VerifyConfig {
    pub base_url: String,
    pub dir: PathBuf,
    pub interval: String,
    pub symbols: Option<Vec<String>>,
    pub from: Option<Month>,
    pub to: Option<Month>,
    pub jobs: usize,
    pub retries: u32,
    pub backoff_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub symbol: String,
    pub month: Month,
    pub vendor_url: String,
    pub http_status: u16,
    pub etag: Option<String>,
    pub last_modified: Option<String>,
    pub vendor_sha256: String,
    pub local_sha256: String,
    pub bytes: u64,
    pub verified: bool,
    pub fetched_at: String,
}

/// First token of a `.CHECKSUM` body, lowercased, when it is 64 hex digits.
pub fn parse_checksum(body: &str) -> Option<String> {
    let _ = body;
    None
}

/// Zips present under `<dir>/raw_<interval>/`, filtered by symbols and months, sorted.
pub fn discover(
    dir: &Path,
    interval: &str,
    symbols: Option<&[String]>,
    from: Option<Month>,
    to: Option<Month>,
) -> io::Result<Vec<(String, Month)>> {
    let root = dir.join(format!("raw_{interval}"));
    let mut out = Vec::new();
    for sym_dir in fs::read_dir(&root)? {
        let sym_dir = sym_dir?;
        if !sym_dir.file_type()?.is_dir() {
            continue;
        }
        let symbol = sym_dir.file_name().to_string_lossy().into_owned();
        if symbols.is_some_and(|s| !s.contains(&symbol)) {
            continue;
        }
        for f in fs::read_dir(sym_dir.path())? {
            let name = f?.file_name().to_string_lossy().into_owned();
            let stem = name.strip_prefix(&format!("{symbol}-{interval}-")).and_then(|r| r.strip_suffix(".zip"));
            let Some(month) = stem.and_then(|s| Month::parse(s).ok()) else { continue };
            if from.is_some_and(|m| month < m) || to.is_some_and(|m| month > m) {
                continue;
            }
            out.push((symbol.clone(), month));
        }
    }
    out.sort();
    Ok(out)
}

fn verify_one(cfg: &VerifyConfig, t: &dyn Transport, symbol: &str, month: Month, tmp: &Path) -> Entry {
    let zip = zip_path(&cfg.dir, &cfg.interval, symbol, month);
    let url = zip_url(&cfg.base_url, symbol, &cfg.interval, month);
    let mut e = Entry {
        symbol: symbol.to_string(),
        month,
        vendor_url: format!("{url}.CHECKSUM"),
        http_status: 0,
        etag: None,
        last_modified: None,
        vendor_sha256: String::new(),
        local_sha256: String::new(),
        bytes: fs::metadata(&zip).map(|m| m.len()).unwrap_or(0),
        verified: false,
        fetched_at: String::new(),
    };
    let dest = tmp.join(format!("{}.CHECKSUM", zip_name(symbol, &cfg.interval, month)));
    if let Ok(r) = get_retry(t, &e.vendor_url, &dest, cfg.retries, cfg.backoff_ms) {
        e.http_status = r.status;
        e.etag = r.header("etag").map(str::to_string);
        e.last_modified = r.header("last-modified").map(str::to_string);
        if r.status == 200 {
            if let Ok(body) = fs::read_to_string(&dest) {
                e.vendor_sha256 = parse_checksum(&body).unwrap_or_default();
            }
        }
    }
    let _ = fs::remove_file(&dest);
    e.fetched_at = calendar::now_iso();
    if let Ok(h) = sha256_file(&zip) {
        e.local_sha256 = h;
    }
    e.verified = !e.vendor_sha256.is_empty() && e.vendor_sha256 == e.local_sha256;
    e
}

/// Verifies every selected zip. Entries are sorted by symbol and month.
pub fn run(cfg: &VerifyConfig, t: &dyn Transport) -> io::Result<Vec<Entry>> {
    let _ = (cfg, t);
    Ok(Vec::new())
}

/// Snapshot JSON object embedded by `rcsr prep --snapshot` and `rcsr panel --snapshot`.
/// `coverage_last` is the last day of the newest verified month. `complete` needs every
/// entry verified and `coverage_last >= through`; without `through` it is false.
pub fn snapshot_json(entries: &[Entry], interval: &str, through: Option<&str>, fetched_at: &str) -> String {
    let _ = (entries, interval, through, fetched_at);
    String::new()
}
