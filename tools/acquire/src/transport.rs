//! The network boundary: one GET into a file, with `curl` as the implementation.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

/// Status and headers of one response. The body is in the destination file.
#[derive(Clone, Debug, Default)]
pub struct Response {
    pub status: u16,
    pub headers: Vec<(String, String)>,
}

impl Response {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

pub trait Transport: Sync {
    /// GET `url` and write the body to `dest` for any HTTP status. `Err` means no HTTP
    /// response was obtained (DNS, connect, timeout).
    fn get(&self, url: &str, dest: &Path) -> Result<Response, String>;
}

/// Transport that runs the `curl` binary.
pub struct Curl {
    pub max_time_secs: u32,
}

impl Default for Curl {
    fn default() -> Self {
        Curl { max_time_secs: 120 }
    }
}

impl Transport for Curl {
    fn get(&self, url: &str, dest: &Path) -> Result<Response, String> {
        let mut hdr = PathBuf::from(dest);
        let mut name = hdr.file_name().map(|n| n.to_os_string()).unwrap_or_default();
        name.push(".hdr");
        hdr.set_file_name(name);
        let out = Command::new("curl")
            .args(["--silent", "--show-error", "--connect-timeout", "20", "--max-time"])
            .arg(self.max_time_secs.to_string())
            .args(["--user-agent", "rcsr-acquire/0.1", "--dump-header"])
            .arg(&hdr)
            .arg("--output")
            .arg(dest)
            .arg(url)
            .output()
            .map_err(|e| format!("cannot run curl: {e}"))?;
        let raw = fs::read_to_string(&hdr).unwrap_or_default();
        let _ = fs::remove_file(&hdr);
        if !out.status.success() {
            let _ = fs::remove_file(dest);
            let err = String::from_utf8_lossy(&out.stderr).trim().to_string();
            return Err(format!("curl failed ({}): {err}", out.status));
        }
        parse_headers(&raw).ok_or_else(|| "curl returned no HTTP status".to_string())
    }
}

/// Parses a `--dump-header` file. The last response block wins, so redirects and
/// `100 Continue` blocks are skipped.
pub fn parse_headers(raw: &str) -> Option<Response> {
    let mut current: Option<Response> = None;
    for line in raw.lines() {
        let line = line.trim_end();
        if let Some(rest) = line.strip_prefix("HTTP/") {
            let status = rest.split_whitespace().nth(1)?.parse().ok()?;
            current = Some(Response { status, headers: Vec::new() });
        } else if let (Some(r), Some((k, v))) = (current.as_mut(), line.split_once(':')) {
            r.headers.push((k.trim().to_string(), v.trim().to_string()));
        }
    }
    current
}

/// GET with retries on transport errors and on statuses other than 200 and 404.
pub fn get_retry(
    t: &dyn Transport,
    url: &str,
    dest: &Path,
    attempts: u32,
    backoff_ms: u64,
) -> Result<Response, String> {
    let mut last: Result<Response, String> = Err("no attempt made".to_string());
    for attempt in 0..attempts.max(1) {
        last = t.get(url, dest);
        if matches!(&last, Ok(r) if r.status == 200 || r.status == 404) {
            break;
        }
        if attempt + 1 < attempts {
            thread::sleep(Duration::from_millis(backoff_ms * u64::from(attempt + 1)));
        }
    }
    last
}
