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

/// A resolved read projection recorded in the manifest (issue #294).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionRecord {
    pub hash: String,
    pub columns: Vec<String>,
    pub indices: Vec<usize>,
}

/// The facts collected once, before the search starts.
#[derive(Clone)]
pub struct Manifest {
    pub version: String,
    pub start_utc: String,
    /// #256: the commit the binary was built from (recorded by `build.rs`).
    pub commit: String,
    /// #256: work tree state when `build.rs` last ran.
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
    /// #169: batched-GPU fit ledger `(gpu, fallback, exact_refits)`, filled in
    /// after the search. `None` for a run that never used the GPU.
    pub fit: Option<(u64, u64, u64)>,
    /// #290: calibrated dispatch values and the per-stage route ledger,
    /// filled in when the search returns.
    pub dispatch: Option<crate::dispatch::DispatchReport>,
    /// #112: `Some(true)` for full rows, `Some(false)` for an explicit
    /// `--sample-rows` sample, `None` when the source was a CSV or unknown.
    pub full_rows: Option<bool>,
    /// #112: rows selected by `--sample-rows` (`None` for a full load).
    pub sampled_rows: Option<usize>,
    /// #112: the guard's estimate that justified the decision.
    pub estimate_bytes: Option<u64>,
    /// #294: the resolved projected read (hash, column list and header
    /// indices), or `None` when the run loaded every column.
    pub projection: Option<ProjectionRecord>,
    /// #260: the `--project` view (id, seed, view index, columns by family)
    /// as a JSON object, or `None` when no view was requested.
    pub view: Option<String>,
    /// #254: the declared target spec `(source, path, sha256)`, or `None`.
    pub target_spec: Option<(String, String, String)>,
    /// #223: `(source, reason)` when the engine resolved the fit arm.
    pub fit_resolution: Option<(String, String)>,
    /// `(rows, row hash, fixed)` of the `--fit-rows`/`--fit-samples` subsample,
    /// or `None` for a full fit. `fixed` is true when the device fits the same
    /// rows for the whole run.
    pub fit_subsample: Option<(usize, u64, bool)>,
    /// #602: the subsample size came from the row-count default, not `--fit-rows`.
    pub fit_rows_default: bool,
    /// #855: the run was an interim chunk (`--interim-chunk`): the search
    /// state is saved and no report ran. The next full chunk overwrites it.
    pub interim: bool,
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
        if self.interim {
            out.push_str("  \"interim\":true,\n");
        }
        out.push_str(&format!(
            "  \"commit\":{},\"dirty\":{},\n",
            jstr(&self.commit),
            self.dirty
        ));
        out.push_str(&format!(
            "  \"binary\":{{\"path\":{},\"sha256\":{},\"build_commit\":{},\"build_dirty\":{},\"cuda_build\":{}}},\n",
            jstr(&self.exe_path),
            jstr(&self.exe_sha256),
            jstr(&self.commit),
            self.dirty,
            cuda_build_json()
        ));
        out.push_str(&format!(
            "  \"data\":{{\"path\":{},\"sha256\":{},\"full_rows\":{},\"sampled_rows\":{},\
             \"estimate_bytes\":{},\"train\":{{\"rows\":{},\"fingerprint\":\"{:016x}\"}},\
             \"test\":{{\"rows\":{},\"fingerprint\":\"{:016x}\"}}}},\n",
            jstr(&self.data_path),
            self.data_sha256
                .as_deref()
                .map(jstr)
                .unwrap_or_else(|| "null".to_string()),
            self.full_rows.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            self.sampled_rows.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
            self.estimate_bytes.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
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
            "  \"backend\":{},\"device\":{{\"name\":{},\"driver\":{},\"cuda\":{}}},\"threads\":{},\n",
            jstr(&self.backend),
            jstr(&self.device),
            jstr(&self.driver),
            jstr(&self.cuda_version),
            self.config.threads
        ));
        if let Some((gpu, fallback, refits)) = self.fit {
            out.push_str(&format!(
                "  \"fit\":{{\"gpu\":{},\"fallback\":{},\"refits\":{}}},\n",
                gpu, fallback, refits
            ));
        }
        if let Some((source, reason)) = &self.fit_resolution {
            out.push_str(&format!(
                "  \"fit_resolution\":{{\"strategy\":{},\"source\":{},\"reason\":{},\"fit_max_iters\":{}}},\n",
                jstr(self.config.fit_strategy.name()),
                jstr(source),
                jstr(reason),
                self.config.fit_max_iters_effective()
            ));
        }
        if let Some((rows, hash, fixed)) = self.fit_subsample {
            out.push_str(&format!(
                "  \"fit_subsample\":{{\"fit_rows\":{},\"hash\":\"{:016x}\",\"fixed\":{},\"scored_rows\":{},\"source\":\"{}\"}},\n",
                rows,
                hash,
                fixed,
                self.train.rows,
                if self.fit_rows_default { "default" } else { "flag" }
            ));
        }
        if let Some(dispatch) = &self.dispatch {
            out.push_str(&format!("  \"dispatch\":{},\n", dispatch.manifest_json()));
        }
        out.push_str(&format!(
            "  \"prune\":{{\"enabled\":{},\"rows\":{},\"budget\":{}}},\n",
            self.config.prune, self.config.prune_rows, self.config.prune_budget
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
        match &self.projection {
            Some(p) => {
                let cols: Vec<String> = p.columns.iter().map(|c| jstr(c)).collect();
                let idx: Vec<String> = p.indices.iter().map(|i| i.to_string()).collect();
                out.push_str(&format!(
                    "  \"projection\":{{\"hash\":{},\"columns\":[{}],\"indices\":[{}]}},\n",
                    jstr(&p.hash),
                    cols.join(","),
                    idx.join(",")
                ));
            }
            None => out.push_str("  \"projection\":null,\n"),
        }
        match &self.view {
            Some(v) => out.push_str(&format!("  \"view\":{},\n", v)),
            None => out.push_str("  \"view\":null,\n"),
        }
        match &self.target_spec {
            Some((source, path, sha)) => out.push_str(&format!(
                "  \"target_spec\":{{\"source\":{},\"path\":{},\"sha256\":{}}},\n",
                jstr(source),
                jstr(path),
                jstr(sha)
            )),
            None => out.push_str("  \"target_spec\":null,\n"),
        }
        let phase_json: Vec<String> = phases
            .iter()
            .map(|(name, secs)| format!("{}:{}", jstr(name), secs))
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
    full_rows: Option<bool>,
    sampled_rows: Option<usize>,
    estimate_bytes: Option<u64>,
) -> Manifest {
    let exe = std::env::current_exe().unwrap_or_default();
    let exe_path = exe.to_string_lossy().into_owned();
    let exe_sha256 = crate::provenance::exe_sha256();
    let (commit, dirty) = build_info();
    let data_sha256 = if data_path == "-" || data_path.is_empty() {
        None
    } else {
        sha256_file_cached(Path::new(data_path)).ok()
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
        fit: None,
        dispatch: None,
        full_rows,
        sampled_rows,
        estimate_bytes,
        projection: None,
        view: None,
        target_spec: None,
        fit_resolution: None,
        fit_subsample: None,
        fit_rows_default: false,
        interim: false,
    }
}

/// Verify a manifest against the current binary, commit, data and config.
///
/// Returns one human-readable line per check; any `FAIL` line means the caller
/// must exit non-zero.
///
/// `binary_override` names the executable to check instead of the running one
/// (#256): the preserved binary of a finished run can be verified by a newer
/// `rcsr`.
pub fn verify(
    path: impl AsRef<Path>,
    data_override: Option<&str>,
    binary_override: Option<&str>,
) -> Result<Vec<String>, String> {
    let path = path.as_ref();
    let text = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read manifest '{}': {}", path.display(), e))?;
    let mut lines = Vec::new();
    let mut failed = false;

    let binary = object(&text, "binary")
        .ok_or_else(|| format!("manifest '{}' has no 'binary' object", path.display()))?;
    let saved_exe = string_field(binary, "sha256").unwrap_or_else(|| "unknown".to_string());
    let now_exe = match binary_override {
        Some(file) => sha256_file(Path::new(file)).unwrap_or_else(|_| "unknown".to_string()),
        None => crate::provenance::exe_sha256(),
    };
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
    let (now_commit, _) = build_info();
    if binary_override.is_some() {
        // The commit is a label compiled into the binary; the SHA-256 above is
        // what proves the given file is the one that ran.
        lines.push(format!("PASS commit {} (build-time label; identity is the binary sha256)", saved_commit));
    } else if saved_commit == "unknown" || now_commit == "unknown" || saved_commit == now_commit {
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

/// `(commit, dirty)` of the tree this binary was built from. Both come from
/// `build.rs` (#256); no `git` process runs at run time.
pub fn build_info() -> (String, bool) {
    (
        crate::provenance::BUILD_COMMIT.to_string(),
        crate::provenance::build_dirty(),
    )
}

pub(crate) fn device_info(backend: &str) -> (String, String, String) {
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

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn round(a: u32, b: u32, c: u32, d: &mut u32, e: u32, f: u32, g: u32, h: &mut u32, k: u32, w: u32) {
    let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
    let ch = (e & f) ^ (!e & g);
    let t1 = h
        .wrapping_add(s1)
        .wrapping_add(ch)
        .wrapping_add(k)
        .wrapping_add(w);
    let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
    let maj = (a & b) ^ (a & c) ^ (b & c);
    *d = d.wrapping_add(t1);
    *h = t1.wrapping_add(s0.wrapping_add(maj));
}

/// Rolling 16-word message schedule: `w[i % 16]` becomes `W[i]` for `i >= 16`.
#[inline(always)]
fn schedule(w: &mut [u32; 16], i: usize) -> u32 {
    let w15 = w[(i + 1) % 16];
    let w2 = w[(i + 14) % 16];
    let s0 = w15.rotate_right(7) ^ w15.rotate_right(18) ^ (w15 >> 3);
    let s1 = w2.rotate_right(17) ^ w2.rotate_right(19) ^ (w2 >> 10);
    w[i % 16] = w[i % 16]
        .wrapping_add(s0)
        .wrapping_add(w[(i + 9) % 16])
        .wrapping_add(s1);
    w[i % 16]
}

/// Streaming SHA-256 state. The file and in-memory entry points share it, so a
/// multi-gigabyte store is hashed without reading or copying it into RAM.
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
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

    fn new() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let chunk = self.buf;
                self.compress(&chunk);
                self.buf_len = 0;
            }
        }
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            let block: &[u8; 64] = block.try_into().expect("64-byte block");
            self.compress(block);
        }
        let rest = blocks.remainder();
        if !rest.is_empty() {
            self.buf[..rest.len()].copy_from_slice(rest);
            self.buf_len = rest.len();
        }
    }

    fn compress(&mut self, chunk: &[u8; 64]) {
        let mut w = [0u32; 16];
        for (word, bytes) in w.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        macro_rules! r8 {
            ($i:expr, $wf:expr) => {{
                round(a, b, c, &mut d, e, f, g, &mut h, Self::K[$i], $wf($i));
                round(h, a, b, &mut c, d, e, f, &mut g, Self::K[$i + 1], $wf($i + 1));
                round(g, h, a, &mut b, c, d, e, &mut f, Self::K[$i + 2], $wf($i + 2));
                round(f, g, h, &mut a, b, c, d, &mut e, Self::K[$i + 3], $wf($i + 3));
                round(e, f, g, &mut h, a, b, c, &mut d, Self::K[$i + 4], $wf($i + 4));
                round(d, e, f, &mut g, h, a, b, &mut c, Self::K[$i + 5], $wf($i + 5));
                round(c, d, e, &mut f, g, h, a, &mut b, Self::K[$i + 6], $wf($i + 6));
                round(b, c, d, &mut e, f, g, h, &mut a, Self::K[$i + 7], $wf($i + 7));
            }};
        }
        r8!(0, |i: usize| w[i]);
        r8!(8, |i: usize| w[i]);
        r8!(16, |i: usize| schedule(&mut w, i));
        r8!(24, |i: usize| schedule(&mut w, i));
        r8!(32, |i: usize| schedule(&mut w, i));
        r8!(40, |i: usize| schedule(&mut w, i));
        r8!(48, |i: usize| schedule(&mut w, i));
        r8!(56, |i: usize| schedule(&mut w, i));
        for (state, v) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(v);
        }
    }

    fn finalize(mut self) -> String {
        let bits = self.total_len.wrapping_mul(8);
        let rem = self.buf_len;
        let pad_len = if rem < 56 { 56 - rem } else { 120 - rem };
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        self.update(&pad[..pad_len]);
        self.update(&bits.to_be_bytes());
        let mut out = String::with_capacity(64);
        for word in self.h {
            out.push_str(&format!("{:08x}", word));
        }
        out
    }
}

/// Bytes per chunk handed from the reader thread to the hashing thread.
const SHA256_CHUNK: usize = 8 << 20;
/// Direct I/O alignment for offsets, lengths and the buffer start.
#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
const DIRECT_ALIGN: usize = 4096;
/// Default minimum file size for the direct path.
const SHA256_DIRECT_MIN_DEFAULT: u64 = 1 << 30;

/// Which read path a digest used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReadPath {
    Buffered,
    Direct,
}

/// One chunk of file bytes: `buf[start..end]`.
struct Chunk {
    buf: Vec<u8>,
    start: usize,
    end: usize,
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
struct DirectSource {
    file: std::fs::File,
    tail: std::fs::File,
    offset: u64,
    aligned_len: u64,
    len: u64,
}

enum StreamSource {
    Buffered(std::fs::File),
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    Direct(DirectSource),
}

impl StreamSource {
    /// Opens `path`; a file of at least `direct_min` bytes uses O_DIRECT when the
    /// filesystem accepts it, everything else the buffered path.
    fn open(path: &Path, direct_min: u64) -> Result<StreamSource, String> {
        let err = |e: std::io::Error| format!("cannot read '{}': {}", path.display(), e);
        let buffered = std::fs::File::open(path).map_err(err)?;
        #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
        {
            use std::os::unix::fs::OpenOptionsExt;
            let len = buffered.metadata().map_err(err)?.len();
            if len >= direct_min {
                // O_DIRECT is 0o40000 in asm-generic/fcntl.h on x86_64.
                let direct = std::fs::OpenOptions::new()
                    .read(true)
                    .custom_flags(0o40000)
                    .open(path);
                if let Ok(file) = direct {
                    return Ok(StreamSource::Direct(DirectSource {
                        file,
                        tail: buffered,
                        offset: 0,
                        aligned_len: len - len % DIRECT_ALIGN as u64,
                        len,
                    }));
                }
            }
        }
        let _ = direct_min;
        Ok(StreamSource::Buffered(buffered))
    }

    fn path_kind(&self) -> ReadPath {
        match self {
            StreamSource::Buffered(_) => ReadPath::Buffered,
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            StreamSource::Direct(_) => ReadPath::Direct,
        }
    }

    fn next_chunk(&mut self, path: &Path) -> Result<Option<Chunk>, String> {
        let err = |e: std::io::Error| format!("cannot read '{}': {}", path.display(), e);
        match self {
            StreamSource::Buffered(file) => {
                use std::io::Read;
                let mut buf = vec![0u8; SHA256_CHUNK];
                let mut filled = 0;
                while filled < buf.len() {
                    let n = file.read(&mut buf[filled..]).map_err(err)?;
                    if n == 0 {
                        break;
                    }
                    filled += n;
                }
                if filled == 0 {
                    return Ok(None);
                }
                Ok(Some(Chunk {
                    buf,
                    start: 0,
                    end: filled,
                }))
            }
            #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
            StreamSource::Direct(src) => {
                use std::os::unix::fs::FileExt;
                if src.offset < src.aligned_len {
                    let want = (src.aligned_len - src.offset).min(SHA256_CHUNK as u64) as usize;
                    let mut buf = vec![0u8; SHA256_CHUNK + DIRECT_ALIGN];
                    let start = buf.as_ptr().align_offset(DIRECT_ALIGN);
                    src.file
                        .read_exact_at(&mut buf[start..start + want], src.offset)
                        .map_err(err)?;
                    src.offset += want as u64;
                    return Ok(Some(Chunk {
                        buf,
                        start,
                        end: start + want,
                    }));
                }
                if src.offset < src.len {
                    let want = (src.len - src.offset) as usize;
                    let mut buf = vec![0u8; want];
                    src.tail.read_exact_at(&mut buf, src.offset).map_err(err)?;
                    src.offset = src.len;
                    return Ok(Some(Chunk {
                        buf,
                        start: 0,
                        end: want,
                    }));
                }
                Ok(None)
            }
        }
    }
}

/// SHA-256 of a file, as lowercase hex. A reader thread fills 8 MiB chunks while
/// the calling thread hashes, so the file size is never added to the process RSS.
/// On Linux x86_64 a file of at least `RCSR_SHA256_DIRECT_MIN_BYTES` bytes
/// (default 1 GiB) is read with O_DIRECT so the stream does not evict the page
/// cache; a filesystem that refuses O_DIRECT uses the buffered path.
pub fn sha256_file(path: impl AsRef<Path>) -> Result<String, String> {
    let direct_min = std::env::var("RCSR_SHA256_DIRECT_MIN_BYTES")
        .ok()
        .and_then(|v| v.trim().parse::<u64>().ok())
        .unwrap_or(SHA256_DIRECT_MIN_DEFAULT);
    sha256_file_with(path.as_ref(), direct_min).map(|(digest, _)| digest)
}

fn sha256_file_with(path: &Path, direct_min: u64) -> Result<(String, ReadPath), String> {
    let mut source = StreamSource::open(path, direct_min)?;
    let kind = source.path_kind();
    let (tx, rx) = std::sync::mpsc::sync_channel::<Result<Chunk, String>>(2);
    std::thread::scope(|s| {
        s.spawn(move || loop {
            match source.next_chunk(path) {
                Ok(Some(chunk)) => {
                    if tx.send(Ok(chunk)).is_err() {
                        break;
                    }
                }
                Ok(None) => break,
                Err(e) => {
                    let _ = tx.send(Err(e));
                    break;
                }
            }
        });
        let mut state = Sha256::new();
        for chunk in rx {
            let chunk = chunk?;
            state.update(&chunk.buf[chunk.start..chunk.end]);
        }
        Ok((state.finalize(), kind))
    })
}


/// Identity of a data file for digest reuse: size, modification time and (on
/// Unix) inode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FileStamp {
    size: u64,
    mtime_ns: u128,
    inode: u64,
}

fn file_stamp(path: &Path) -> Option<FileStamp> {
    let meta = std::fs::metadata(path).ok()?;
    let mtime_ns = meta
        .modified()
        .ok()?
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_nanos();
    #[cfg(unix)]
    let inode = std::os::unix::fs::MetadataExt::ino(&meta);
    #[cfg(not(unix))]
    let inode = 0;
    Some(FileStamp { size: meta.len(), mtime_ns, inode })
}

/// Path of the digest cache that sits beside a data file.
pub fn digest_sidecar_path(data_path: &Path) -> std::path::PathBuf {
    let mut name = data_path.as_os_str().to_os_string();
    name.push(".sha256.json");
    std::path::PathBuf::from(name)
}

fn number_field(text: &str, key: &str) -> Option<u128> {
    let rest = value_after(text, key)?;
    let end = rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// The digest recorded in `sidecar` when it still describes the file.
fn read_digest_sidecar(sidecar: &Path, stamp: FileStamp) -> Option<String> {
    let text = std::fs::read_to_string(sidecar).ok()?;
    let sha = string_field(&text, "sha256")?;
    let valid = sha.len() == 64 && sha.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'));
    let same = number_field(&text, "size")? == stamp.size as u128
        && number_field(&text, "mtime_ns")? == stamp.mtime_ns
        && number_field(&text, "inode")? == stamp.inode as u128;
    (valid && same).then_some(sha)
}

fn write_digest_sidecar(sidecar: &Path, data_path: &Path, stamp: FileStamp, sha: &str) {
    let text = format!(
        "{{\"path\":{},\"size\":{},\"mtime_ns\":{},\"inode\":{},\"sha256\":{}}}\n",
        jstr(&data_path.to_string_lossy()),
        stamp.size,
        stamp.mtime_ns,
        stamp.inode,
        jstr(sha)
    );
    let mut tmp = sidecar.as_os_str().to_os_string();
    tmp.push(format!(".tmp{}", std::process::id()));
    let tmp = std::path::PathBuf::from(tmp);
    if std::fs::write(&tmp, text).is_err() || std::fs::rename(&tmp, sidecar).is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
}

type DigestMemo = std::sync::Mutex<std::collections::HashMap<std::path::PathBuf, (FileStamp, String)>>;

/// SHA-256 of the run's data file for the run record, hashing at most once per
/// process and at most once per file version across processes.
///
/// A digest is reused only when size, modification time and inode match the
/// ones recorded beside the file in `<data>.sha256.json`. A file that changes
/// while it is hashed gets no sidecar. A directory that cannot be written
/// skips the sidecar silently. Content-verifying callers (lockbox, ladder,
/// panel append) keep using [`sha256_file`].
pub fn sha256_file_cached(path: impl AsRef<Path>) -> Result<String, String> {
    static MEMO: std::sync::OnceLock<DigestMemo> = std::sync::OnceLock::new();
    let path = path.as_ref();
    let Some(before) = file_stamp(path) else {
        return sha256_file(path);
    };
    let memo = MEMO.get_or_init(Default::default);
    if let Some((stamp, sha)) = memo.lock().unwrap().get(path) {
        if *stamp == before {
            return Ok(sha.clone());
        }
    }
    let sidecar = digest_sidecar_path(path);
    if let Some(sha) = read_digest_sidecar(&sidecar, before) {
        memo.lock().unwrap().insert(path.to_path_buf(), (before, sha.clone()));
        return Ok(sha);
    }
    let sha = sha256_file(path)?;
    if file_stamp(path) == Some(before) {
        write_digest_sidecar(&sidecar, path, before, &sha);
        memo.lock().unwrap().insert(path.to_path_buf(), (before, sha.clone()));
    }
    Ok(sha)
}

/// SHA-256, as lowercase hex.
pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut state = Sha256::new();
    state.update(bytes);
    state.finalize()
}

/// Incremental SHA-256 for callers that hash one column group at a time.
///
/// `rcsr enrich` uses this to produce per-column digests without materialising
/// a whole column (or the whole store) in memory.
pub struct Sha256Hasher {
    state: Sha256,
}

impl Default for Sha256Hasher {
    fn default() -> Self {
        Self::new()
    }
}

impl Sha256Hasher {
    pub fn new() -> Self {
        Sha256Hasher {
            state: Sha256::new(),
        }
    }
    pub fn update(&mut self, bytes: &[u8]) {
        self.state.update(bytes);
    }
    pub fn finalize(self) -> String {
        self.state.finalize()
    }
}

/// The `binary.cuda_build` value: a JSON object in a CUDA build, `null` otherwise.
fn cuda_build_json() -> String {
    match crate::provenance::CUDA_BUILD {
        Some(b) => format!(
            "{{\"toolkit\":{},\"nvcc\":{},\"source\":{},\"arch\":{},\"runtime\":{}}}",
            jstr(b.toolkit),
            jstr(b.nvcc),
            jstr(b.source),
            jstr(b.arch),
            jstr(&crate::provenance::runtime_text())
        ),
        None => "null".to_string(),
    }
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

/// #855: whether a manifest was written by an interim chunk.
pub fn is_interim(text: &str) -> bool {
    text.lines().any(|l| l.trim() == "\"interim\":true,")
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
    fn streaming_file_hash_matches_in_memory_for_chunk_boundaries() {
        let scratch = crate::test_scratch::Scratch::new("sha256-stream");
        let path = scratch.join("stream.bin");
        // Around the 64-byte block and 56-byte padding boundaries, and more
        // than one 1 MiB read chunk.
        for len in [0usize, 1, 55, 56, 57, 63, 64, 65, 1_048_575, 1_048_576, 1_048_577] {
            let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            std::fs::write(&path, &bytes).unwrap();
            assert_eq!(
                sha256_file(&path).unwrap(),
                sha256_hex(&bytes),
                "length {len}"
            );
        }
    }

    #[test]
    fn sha256_known_answer_boundaries() {
        assert_eq!(
            sha256_hex(b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
        );
        assert_eq!(
            sha256_hex(&vec![b'a'; 1_000_000]),
            "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 55]),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 56]),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a"
        );
        assert_eq!(
            sha256_hex(&[b'a'; 64]),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb"
        );
    }

    #[test]
    fn direct_io_path_matches_buffered_path() {
        let scratch = crate::test_scratch::Scratch::new("sha256-direct");
        let path = scratch.join("direct.bin");
        let mut direct_ran = false;
        for len in [0usize, 4095, 4096, 4097, 8 << 20, (8 << 20) - 1, (8 << 20) + 4097] {
            let bytes: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
            std::fs::write(&path, &bytes).unwrap();
            let expected = sha256_hex(&bytes);
            let (buffered, kind) = sha256_file_with(&path, u64::MAX).unwrap();
            assert_eq!(kind, ReadPath::Buffered, "length {len}");
            assert_eq!(buffered, expected, "buffered, length {len}");
            let (direct, kind) = sha256_file_with(&path, 0).unwrap();
            assert_eq!(direct, expected, "direct request, length {len} ({kind:?})");
            direct_ran |= kind == ReadPath::Direct;
        }
        // The scratch directory may sit on a filesystem that refuses O_DIRECT; the
        // fallback is then the buffered path, which the digest equality above covers.
        let refused = !direct_ran;
        if refused {
            let (_, kind) = sha256_file_with(&path, 0).unwrap();
            assert_eq!(kind, ReadPath::Buffered);
        }
    }

    #[test]
    fn file_hash_reports_missing_file() {
        let scratch = crate::test_scratch::Scratch::new("sha256-missing");
        let err = sha256_file(scratch.join("absent.bin")).unwrap_err();
        assert!(err.starts_with("cannot read "), "{err}");
    }

    #[test]
    fn sidecar_rejects_changed_file() {
        let dir = crate::test_scratch::Scratch::new("sha-sidecar");
        let path = dir.join("d.bin");
        std::fs::write(&path, b"abc").unwrap();
        let abc = sha256_hex(b"abc");
        assert_eq!(sha256_file_cached(&path).unwrap(), abc);
        let sidecar = digest_sidecar_path(&path);
        let stamp = file_stamp(&path).unwrap();
        assert_eq!(read_digest_sidecar(&sidecar, stamp), Some(abc));
        std::fs::write(&path, b"abcd").unwrap();
        let changed = file_stamp(&path).unwrap();
        assert_eq!(read_digest_sidecar(&sidecar, changed), None);
        assert_eq!(sha256_file_cached(&path).unwrap(), sha256_hex(b"abcd"));
    }

    #[test]
    fn utc_time_is_well_formed() {
        let s = utc_now();
        assert_eq!(s.len(), 20, "{}", s);
        assert!(s.ends_with('Z'), "{}", s);
    }
}
