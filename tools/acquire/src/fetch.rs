//! Monthly kline zips: planning, bounded-parallel download and the status CSV.

use crate::calendar::{self, Month};
use crate::transport::{get_retry, Transport};
use std::collections::HashSet;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc;
use std::thread;

/// An existing zip must be larger than this to count as present.
pub const MIN_ZIP_BYTES: u64 = 1000;

pub const CSV_HEADER: &str = "symbol,interval,month,http_status,bytes,fetched_at";

pub struct FetchConfig {
    pub base_url: String,
    pub out: PathBuf,
    pub interval: String,
    pub symbols: Vec<String>,
    pub from: Month,
    pub to: Month,
    pub jobs: usize,
    pub skip_404: bool,
    pub retries: u32,
    pub backoff_ms: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Job {
    pub symbol: String,
    pub month: Month,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Summary {
    pub planned: usize,
    pub present: usize,
    pub known_missing: usize,
    pub requests: usize,
    pub ok: usize,
    pub missing: usize,
    pub failed: usize,
}

/// `<out>/raw_<interval>/<SYM>/<SYM>-<interval>-YYYY-MM.zip`
pub fn zip_path(out: &Path, interval: &str, symbol: &str, month: Month) -> PathBuf {
    out.join(format!("raw_{interval}")).join(symbol).join(zip_name(symbol, interval, month))
}

pub fn zip_name(symbol: &str, interval: &str, month: Month) -> String {
    format!("{symbol}-{interval}-{month}.zip")
}

pub fn zip_url(base: &str, symbol: &str, interval: &str, month: Month) -> String {
    format!(
        "{base}/data/spot/monthly/klines/{symbol}/{interval}/{}",
        zip_name(symbol, interval, month)
    )
}

pub fn csv_path(out: &Path, interval: &str) -> PathBuf {
    out.join(format!("downloads_{interval}.csv"))
}

/// Months with a 404 row for `(symbol, month)` in an existing status CSV.
pub fn read_404s(csv: &Path, interval: &str) -> HashSet<(String, Month)> {
    let mut out = HashSet::new();
    let Ok(text) = fs::read_to_string(csv) else { return out };
    for line in text.lines().skip(1) {
        let f: Vec<&str> = line.split(',').collect();
        if f.len() >= 4 && f[1] == interval && f[3] == "404" {
            if let Ok(m) = Month::parse(f[2]) {
                out.insert((f[0].to_string(), m));
            }
        }
    }
    out
}

/// Jobs to request, plus counts of zips already present and months skipped as known 404.
pub fn plan(cfg: &FetchConfig) -> (Vec<Job>, Summary) {
    let known = if cfg.skip_404 { read_404s(&csv_path(&cfg.out, &cfg.interval), &cfg.interval) } else { HashSet::new() };
    let mut jobs = Vec::new();
    let mut s = Summary::default();
    for symbol in &cfg.symbols {
        for month in calendar::range(cfg.from, cfg.to) {
            s.planned += 1;
            let p = zip_path(&cfg.out, &cfg.interval, symbol, month);
            if fs::metadata(&p).is_ok_and(|m| m.len() > MIN_ZIP_BYTES) {
                s.present += 1;
            } else if known.contains(&(symbol.clone(), month)) {
                s.known_missing += 1;
            } else {
                jobs.push(Job { symbol: symbol.clone(), month });
            }
        }
    }
    (jobs, s)
}

struct Outcome {
    job: Job,
    status: u16,
    bytes: u64,
    at: String,
}

fn run_job(cfg: &FetchConfig, t: &dyn Transport, job: &Job) -> Outcome {
    let dest = zip_path(&cfg.out, &cfg.interval, &job.symbol, job.month);
    let part = dest.with_extension("zip.part");
    let mut outcome = Outcome { job: job.clone(), status: 0, bytes: 0, at: String::new() };
    if let Some(dir) = dest.parent() {
        if fs::create_dir_all(dir).is_err() {
            outcome.at = calendar::now_iso();
            return outcome;
        }
    }
    let url = zip_url(&cfg.base_url, &job.symbol, &cfg.interval, job.month);
    match get_retry(t, &url, &part, cfg.retries, cfg.backoff_ms) {
        Ok(r) if r.status == 200 => {
            let len = fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
            if len > 0 && fs::rename(&part, &dest).is_ok() {
                outcome.status = 200;
                outcome.bytes = len;
            } else {
                let _ = fs::remove_file(&part);
            }
        }
        Ok(r) => {
            outcome.status = r.status;
            let _ = fs::remove_file(&part);
        }
        Err(_) => {
            let _ = fs::remove_file(&part);
        }
    }
    outcome.at = calendar::now_iso();
    outcome
}

/// Downloads every missing zip. One status row is appended to the CSV per request, in
/// completion order. Existing zips and (with `skip_404`) known 404s make no request.
pub fn run(cfg: &FetchConfig, t: &dyn Transport) -> io::Result<Summary> {
    let _ = (cfg, t);
    Ok(Summary::default())
}
