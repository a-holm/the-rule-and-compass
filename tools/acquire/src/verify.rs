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
    let token = body.split_whitespace().next()?.to_ascii_lowercase();
    (token.len() == 64 && token.bytes().all(|b| b.is_ascii_hexdigit())).then_some(token)
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
    let items = discover(&cfg.dir, &cfg.interval, cfg.symbols.as_deref(), cfg.from, cfg.to)?;
    let tmp = cfg.dir.join(".verify-tmp");
    fs::create_dir_all(&tmp)?;
    let next = AtomicUsize::new(0);
    let results = Mutex::new(Vec::with_capacity(items.len()));
    let workers = cfg.jobs.clamp(1, items.len().max(1));
    thread::scope(|scope| {
        for _ in 0..workers {
            scope.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some((symbol, month)) = items.get(i) else { break };
                let e = verify_one(cfg, t, symbol, *month, &tmp);
                results.lock().expect("results lock").push(e);
            });
        }
    });
    let _ = fs::remove_dir(&tmp);
    let mut entries = results.into_inner().expect("results lock");
    entries.sort_by(|a, b| (&a.symbol, a.month).cmp(&(&b.symbol, b.month)));
    Ok(entries)
}

/// Snapshot JSON object embedded by `rcsr panel build --snapshot`.
/// `coverage_last` is the last day of the newest verified month. `complete` needs every
/// entry verified and `coverage_last >= through`; without `through` it is false.
pub fn snapshot_json(entries: &[Entry], interval: &str, through: Option<&str>, fetched_at: &str) -> String {
    let verified = entries.iter().filter(|e| e.verified).count();
    let coverage_last = entries.iter().filter(|e| e.verified).map(|e| e.month).max().map(Month::last_day);
    let all_verified = !entries.is_empty() && verified == entries.len();
    let complete = all_verified
        && matches!((&coverage_last, through), (Some(c), Some(t)) if c.as_str() >= t);
    let mut s = String::new();
    s.push_str("{\n");
    s.push_str("  \"source\": \"data.binance.vision\",\n");
    s.push_str("  \"market\": \"spot\",\n");
    s.push_str(&format!("  \"interval\": {},\n", quote(interval)));
    s.push_str(&format!("  \"fetched_at\": {},\n", quote(fetched_at)));
    s.push_str(&format!("  \"coverage_last\": {},\n", quote_opt(coverage_last.as_deref())));
    s.push_str(&format!("  \"through\": {},\n", quote_opt(through)));
    s.push_str(&format!("  \"complete\": {complete},\n"));
    s.push_str(&format!("  \"files\": {},\n", entries.len()));
    s.push_str(&format!("  \"verified\": {verified},\n"));
    s.push_str("  \"entries\": [");
    for (i, e) in entries.iter().enumerate() {
        s.push_str(if i == 0 { "\n    " } else { ",\n    " });
        s.push_str(&format!(
            "{{\"symbol\": {}, \"month\": {}, \"vendor_url\": {}, \"http_status\": {}, \"etag\": {}, \
             \"last_modified\": {}, \"vendor_sha256\": {}, \"local_sha256\": {}, \"bytes\": {}, \
             \"verified\": {}, \"fetched_at\": {}}}",
            quote(&e.symbol),
            quote(&e.month.to_string()),
            quote(&e.vendor_url),
            e.http_status,
            quote_opt(e.etag.as_deref()),
            quote_opt(e.last_modified.as_deref()),
            quote(&e.vendor_sha256),
            quote(&e.local_sha256),
            e.bytes,
            e.verified,
            quote(&e.fetched_at),
        ));
    }
    s.push_str(if entries.is_empty() { "]\n}\n" } else { "\n  ]\n}\n" });
    s
}
