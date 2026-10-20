//! Reproducible run manifests (issue #140).
//!
//! A long run is only evidence if another person can identify the binary, the
//! data and the configuration that produced it. `--run-manifest <path>` writes
//! a JSON document at the start of a run and updates it atomically at the end
//! with timings. `rcsr verify-manifest <path>` checks that the current binary,
//! commit, data file and configuration still match.
//!
//! Everything here is pure `std`: SHA-256 and FNV-1a are implemented locally so
//! the core crate keeps its no-dependency rule.

use crate::checkpoint::{
    config_canonical, config_hash, hash_config_text, Fingerprint,
};
use crate::search::Config;
use std::path::Path;

/// The facts collected once, before the search starts.
#[derive(Clone)]
pub struct Manifest {
    pub version: String,
    pub start_utc: String,
    pub commit: String,
    pub dirty: bool,
    pub exe_path: String,
    pub exe_sha256: String,
    pub data_path: String,
    /// SHA-256 of the raw data file; `None` for stdin.
    pub data_sha256: Option<String>,
    pub train: Fingerprint,
    pub test: Fingerprint,
    pub config: Config,
    pub backend: String,
    pub device: String,
    pub driver: String,
    pub cuda_version: String,
    pub resumed_from: Option<String>,
    pub backend_change: Option<(String, String)>,
}

impl Manifest {
    /// Render the JSON document. `end_utc`/`phases` are absent at the start.
    pub fn json(&self, end_utc: Option<&str>, phases: &[(&str, f64)]) -> String {
        let mut out = String::new();
        out.push_str("{\n");
        out.push_str(&format!("  \"tool\":\"rcsr\",\"version\":{},\n", jstr(&self.version)));
        out.push_str(&format!(
            "  \"start_utc\":{},\"end_utc\":{},\"completed\":{},\n",
            jstr(&self.start_utc),
            end_utc.map(jstr).unwrap_or_else(|| "null".to_string()),
            end_utc.is_some()
        ));
        out.push_str(&format!(
            "  \"commit\":{},\"dirty\":{},\n",
            jstr(&self.commit),
            self.dirty
        ));
        out.push_str(&format!(
            "  \"binary\":{{\"path\":{},\"sha256\":{}}},\n",
            jstr(&self.exe_path),
            jstr(&self.exe_sha256)
        ));
        out.push_str(&format!(
            "  \"data\":{{\"path\":{},\"sha256\":{},\"train\":{{\"rows\":{},\"fingerprint\":\"{:016x}\"}},\"test\":{{\"rows\":{},\"fingerprint\":\"{:016x}\"}}}},\n",
            jstr(&self.data_path),
            self.data_sha256
                .as_deref()
                .map(jstr)
                .unwrap_or_else(|| "null".to_string()),
            self.train.rows,
            self.train.hash,
            self.test.rows,
            self.test.hash
        ));
        out.push_str(&format!(
            "  \"config\":{{\"hash\":\"{:016x}\",\"seed\":{},\"canonical\":{}}},\n",
            config_hash(&self.config),
            self.config.seed,
            jstr(&config_canonical(&self.config))
        ));
        out.push_str(&format!(
            "  \"backend\":{},\"device\":{{\"name\":{},\"driver\":{},\"cuda\":{}}},\n",
            jstr(&self.backend),
            jstr(&self.device),
            jstr(&self.driver),
            jstr(&self.cuda_version)
        ));
        out.push_str(&format!(
            "  \"resume\":{{\"from\":{},\"backend_change\":{}}},\n",
            self.resumed_from
                .as_deref()
                .map(jstr)
                .unwrap_or_else(|| "null".to_string()),
            match &self.backend_change {
                Some((from, to)) => format!(
                    "{{\"from\":{},\"to\":{},\"allowed\":true}}",
                    jstr(from),
                    jstr(to)
                ),
                None => "null".to_string(),
            }
        ));
        let phase_json: Vec<String> = phases
            .iter()
            .map(|(name, secs)| format!("{}:{:.6}", jstr(name), secs))
            .collect();
        out.push_str(&format!(
            "  \"phases_seconds\":{{{}}}\n}}\n",
            phase_json.join(",")
        ));
        out
    }

    /// Publish the manifest atomically.
    pub fn write(
        &self,
        path: impl AsRef<Path>,
        end_utc: Option<&str>,
        phases: &[(&str, f64)],
    ) -> Result<(), String> {
        crate::atomic::write_text_atomic(path, &self.json(end_utc, phases))
    }
}

/// Collect the facts a manifest needs. External tools are optional: a missing
/// `git` or `nvidia-smi` becomes `"unknown"`, never an error.
pub fn collect(
    version: &str,
    data_path: &str,
    train: Fingerprint,
    test: Fingerprint,
    config: &Config,
    backend: &str,
    resumed_from: Option<String>,
    backend_change: Option<(String, String)>,
) -> Manifest {
    let exe = std::env::current_exe().unwrap_or_default();
    let exe_path = exe.to_string_lossy().into_owned();
    let exe_sha256 = sha256_file(&exe).unwrap_or_else(|_| "unknown".to_string());
    let (commit, dirty) = git_info();
    let data_sha256 = if data_path == "-" || data_path.is_empty() {
        None
    } else {
        sha256_file(Path::new(data_path)).ok()
    };
    let (device, driver, cuda_version) = device_info(backend);
    Manifest {
        version: version.to_string(),
        start_utc: utc_now(),
        commit,
        dirty,
        exe_path,
        exe_sha256,
        data_path: data_path.to_string(),
        data_sha256,
        train,
        test,
        config: config.clone(),
        backend: backend.to_string(),
        device,
        driver,
        cuda_version,
        resumed_from,
        backend_change,
    }
}

/// Verify a manifest against the current binary, commit, data and config.
///
/// Returns one human-readable line per check; any `FAIL` line means the caller
/// must exit non-zero.
pub fn verify(path: impl AsRef<Path>, data_override: Option<&str>) -> Result<Vec<String>, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read manifest '{}': {}", path.display(), e))?;
    let mut lines = Vec::new();
    let mut failed = false;

    let binary = object(&text, "binary")
        .ok_or_else(|| format!("manifest '{}' has no 'binary' object", path.display()))?;
    let saved_exe = string_field(binary, "sha256").unwrap_or_else(|| "unknown".to_string());
    let exe = std::env::current_exe().unwrap_or_default();
    let now_exe = sha256_file(&exe).unwrap_or_else(|_| "unknown".to_string());
    if saved_exe == now_exe {
        lines.push(format!("PASS binary sha256 {}", now_exe));
    } else {
        failed = true;
        lines.push(format!(
            "FAIL binary sha256: manifest {}, current {}",
            saved_exe, now_exe
        ));
    }

    let saved_commit = string_field(&text, "commit").unwrap_or_else(|| "unknown".to_string());
    let (now_commit, _) = git_info();
    if saved_commit == "unknown" || now_commit == "unknown" || saved_commit == now_commit {
        lines.push(format!("PASS commit {}", saved_commit));
    } else {
        failed = true;
        lines.push(format!(
            "FAIL commit: manifest {}, current {}",
            saved_commit, now_commit
        ));
    }

    let data = object(&text, "data")
        .ok_or_else(|| format!("manifest '{}' has no 'data' object", path.display()))?;
    let saved_data_path = string_field(data, "path").unwrap_or_default();
    let check_path = data_override.unwrap_or(&saved_data_path);
    match string_field(data, "sha256") {
        Some(saved_hash) => {
            if check_path == "-" || check_path.is_empty() {
                failed = true;
                lines.push("FAIL data sha256: no data file to check".to_string());
            } else {
                match sha256_file(Path::new(check_path)) {
                    Ok(now_hash) if now_hash == saved_hash => {
                        lines.push(format!("PASS data sha256 {} {}", check_path, now_hash))
                    }
                    Ok(now_hash) => {
                        failed = true;
                        lines.push(format!(
                            "FAIL data sha256 for '{}': manifest {}, current {}",
                            check_path, saved_hash, now_hash
                        ));
                    }
                    Err(e) => {
                        failed = true;
                        lines.push(format!("FAIL data file '{}': {}", check_path, e));
                    }
                }
            }
        }
        None => lines.push("PASS data sha256 null (stdin)".to_string()),
    }

    let config = object(&text, "config")
        .ok_or_else(|| format!("manifest '{}' has no 'config' object", path.display()))?;
    let saved_hash = string_field(config, "hash").unwrap_or_default();
    let canonical = string_field(config, "canonical").unwrap_or_default();
    let computed = format!("{:016x}", hash_config_text(&canonical));
    if saved_hash == computed {
        lines.push(format!("PASS config hash {}", computed));
    } else {
        failed = true;
        lines.push(format!(
            "FAIL config hash: manifest {}, canonical config hashes to {}",
            saved_hash, computed
        ));
    }

    if failed {
        Err(lines.join("\n"))
    } else {
        Ok(lines)
    }
}

/// `(commit, dirty)` for the current directory. `"unknown"` outside git.
pub fn git_info() -> (String, bool) {
    let commit = std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string());
    let dirty = std::process::Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);
    (commit, dirty)
}

fn device_info(backend: &str) -> (String, String, String) {
    if backend != "gpu" {
        return ("cpu".to_string(), "n/a".to_string(), "n/a".to_string());
    }
    #[cfg(feature = "cuda")]
    let device = crate::gpu::GpuEvaluator::device_name();
    #[cfg(not(feature = "cuda"))]
    let device = "unknown".to_string();
    let (driver, cuda) = match std::process::Command::new("nvidia-smi").output() {
        Ok(o) if o.status.success() => {
            let text = String::from_utf8_lossy(&o.stdout);
            let driver = text
                .lines()
                .find_map(|l| l.split("Driver Version:").nth(1))
                .map(|s| s.split_whitespace().next().unwrap_or("").to_string())
                .unwrap_or_else(|| "unknown".to_string());
            let cuda = text
                .lines()
                .find_map(|l| l.split("CUDA Version:").nth(1))
                .map(|s| {
                    s.split(|c: char| c == '|' || c.is_whitespace())
                        .find(|s| !s.is_empty())
                        .unwrap_or("")
                        .to_string()
                })
                .unwrap_or_else(|| "unknown".to_string());
            (driver, cuda)
        }
        _ => ("unknown".to_string(), "unknown".to_string()),
    };
    (device, driver, cuda)
}

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
pub fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let days = secs.div_euclid(86_400);
    let rem = secs.rem_euclid(86_400);
    let (y, m, d) = civil_from_days(days);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        y,
        m,
        d,
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}

fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    if m <= 2 {
        y += 1;
    }
    (y, m as u32, d as u32)
}

/// SHA-256 of a file, as lowercase hex.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, String> {
    let bytes = std::fs::read(path.as_ref())
        .map_err(|e| format!("cannot read '{}': {}", path.as_ref().display(), e))?;
    Ok(sha256_hex(&bytes))
}

/// SHA-256, as lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];
    let mut h = H0;
    let mut msg = bytes.to_vec();
    let bit_len = (bytes.len() as u64).wrapping_mul(8);
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0);
    }
    msg.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in msg.chunks_exact(64) {
        let mut w = [0u32; 64];
        for i in 0..16 {
            w[i] = u32::from_be_bytes([
                chunk[i * 4],
                chunk[i * 4 + 1],
                chunk[i * 4 + 2],
                chunk[i * 4 + 3],
            ]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = String::with_capacity(64);
    for word in h {
        out.push_str(&format!("{:08x}", word));
    }
    out
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn json_unescape(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some('"') => out.push('"'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                let hex: String = chars.by_ref().take(4).collect();
                if let Ok(v) = u32::from_str_radix(&hex, 16) {
                    if let Some(ch) = char::from_u32(v) {
                        out.push(ch);
                    }
                }
            }
            Some(other) => out.push(other),
            None => break,
        }
    }
    out
}

/// Find `"key": value` and return the value slice (trimmed of trailing comma).
fn value_after<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let needle = format!("\"{}\"", key);
    let start = text.find(&needle)? + needle.len();
    let rest = &text[start..];
    let colon = rest.find(':')?;
    let rest = rest[colon + 1..].trim_start();
    Some(rest)
}

/// Extract a JSON string field, honouring escapes.
pub fn string_field(text: &str, key: &str) -> Option<String> {
    let rest = value_after(text, key)?;
    if !rest.starts_with('"') {
        return None;
    }
    let body = &rest[1..];
    let mut chars = body.char_indices();
    while let Some((i, c)) = chars.next() {
        if c == '\\' {
            // Skip the escaped character; a `\"` is not the closing quote.
            let _ = chars.next();
            continue;
        }
        if c == '"' {
            return Some(json_unescape(&body[..i]));
        }
    }
    None
}

/// Extract the slice of a nested object.
pub fn object<'a>(text: &'a str, key: &str) -> Option<&'a str> {
    let rest = value_after(text, key)?;
    if !rest.starts_with('{') {
        return None;
    }
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escaped = false;
    for (i, c) in rest.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
            } else if c == '\\' {
                escaped = true;
            } else if c == '"' {
                in_string = false;
            }
            continue;
        }
        match c {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&rest[..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_known_vectors() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn utc_time_is_well_formed() {
        let s = utc_now();
        assert_eq!(s.len(), 20, "{}", s);
        assert!(s.ends_with('Z'), "{}", s);
    }
}
