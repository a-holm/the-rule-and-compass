//! #211: optional engine-level development log, and the reader behind
//! `rcsr dev-report`.
//!
//! The search can emit one NDJSON **development record** per generation plus a
//! start, checkpoint and end event. The file is written only when
//! `--dev-log <path>` is given: without it, nothing here runs and the search
//! takes exactly the trajectory it took before this module existed.
//!
//! # What this is for
//!
//! The final `--json` says *what* the run found. This log says *why it behaved
//! as it did*: how a generation's wall split between preparation, variation,
//! constant fitting, evaluation, selection and the archive; how many candidates
//! were rejected as structural duplicates or too complex; how many GPU batches
//! fell back to the CPU and for what cause; how coverage, entropy, convergence
//! and the Pareto front moved. [`crate::devreport`] turns those events into a
//! human-readable summary and a ranked list of bottlenecks.
//!
//! # Guarantees
//!
//! * **Off by default.** `Config::dev_log` is `None` unless the CLI sets it, and
//!   `None` means every measurement here is skipped.
//! * **Trajectory-neutral.** The log is a *reading*: it is computed from values
//!   the search already produces. It never draws from an RNG, never reorders
//!   candidates and never changes selection, so the archive, the reported
//!   errors and the configuration hash are byte-identical with the log on and
//!   off (tested in `tests/dev_log.rs`).
//! * **Atomic and bounded.** Each event is one `write` of one complete line to
//!   an `O_APPEND` file, so a killed process leaves a valid prefix of complete
//!   lines. Generation and checkpoint events stop at `--dev-log-max-bytes`
//!   (default 64 MiB); `run_start` and `run_end` are always written, so the
//!   file is bounded by `cap + two events` and never grows without limit.
//! * **Stable schema.** Every line carries `"schema":"rcsr-devlog/1"`. Fields
//!   may be *added* within a major version; removing or redefining one needs a
//!   new version. The event reference is `docs/dev-log.md`.
//! * **No raw rows.** The log records counts, timings, hashes and error
//!   statistics. It never writes an input row, a target value or a prediction;
//!   the data file appears only as its path, SHA-256 and the checkpoint
//!   fingerprint of the training/test samples.
//!
//! # What this is not
//!
//! It does not replace the external long-run monitor: process CPU and GPU
//! sampling at 1 Hz belongs outside the engine. This is the per-generation
//! record the engine can produce from the inside, at no cost when off.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

/// `schema` value on every event. Bump the trailing number on a breaking
/// change; the reader in [`crate::devreport`] accepts `rcsr-devlog/1`.
pub const SCHEMA: &str = "rcsr-devlog/1";

/// Major schema version, repeated as `"v":1` for readers that only look at the
/// numeric field.
pub const VERSION: u32 = 1;

/// Default byte cap for generation and checkpoint events (64 MiB). A
/// 200-generation run writes well under 1 MiB; the cap exists so a
/// `--generations 1000000` run cannot fill a disk.
pub const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

/// The file name used when the log path is not given: written next to
/// `--json-out`/`--save`, else in the working directory.
pub const DEFAULT_FILE_NAME: &str = "rcsr.dev.ndjson";

/// #211: how much detail a `generation` event carries.
///
/// The log is on by default at [`Level::Summary`]: a concise record about every
/// 10–25 generations that is enough to watch a long run (error, front size,
/// phase times, fit/GPU/fallback counts, GPU duty) and cheap enough to leave on.
/// [`Level::Full`] writes every generation with the complete record — quantiles,
/// archive hash/hypervolume, variation counters, RNG key, per-island best — for
/// after-the-fact analysis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Level {
    #[default]
    Summary,
    Full,
}

impl Level {
    pub fn parse(s: &str) -> Option<Level> {
        match s {
            "summary" => Some(Level::Summary),
            "full" => Some(Level::Full),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Level::Summary => "summary",
            Level::Full => "full",
        }
    }

    /// The generation cadence for a run of `generations`.
    ///
    /// `full` writes every generation. `summary` writes one event every
    /// `clamp(generations / 10, 10, 25)` generations — so a run produces roughly
    /// 10–40 events however long it is — and every generation when the budget is
    /// 10 or fewer, so a short run still has a trajectory.
    pub fn default_every(self, generations: usize) -> usize {
        match self {
            Level::Full => 1,
            Level::Summary => {
                if generations <= 10 {
                    1
                } else {
                    (generations / 10).clamp(10, 25)
                }
            }
        }
    }
}

/// Events larger than this are reported by the writer's `oversize` counter.
/// One event is normally 400–900 bytes.
pub const MAX_EVENT_BYTES: usize = 8 * 1024;

// ---------------------------------------------------------------------------
// JSON encoding helpers
// ---------------------------------------------------------------------------

/// A finite number as a JSON token; a non-finite one becomes `null` rather than
/// the invalid `NaN`/`inf` that `format!("{}", f64)` would produce.
fn opt_num(v: Option<f64>) -> String {
    match v {
        Some(v) if v.is_finite() => num(v),
        _ => "null".to_string(),
    }
}

pub fn num(v: f64) -> String {
    if v.is_finite() {
        let mut s = format!("{}", v);
        // `format!` prints `1` for 1.0, which is valid JSON but easy to read as
        // an integer; keep it as-is so the file round-trips exactly.
        if s == "-0" {
            s = "-0.0".to_string();
        }
        s
    } else {
        "null".to_string()
    }
}

/// A JSON string literal, escaped. Only the seven mandatory escapes plus
/// `\u00XX` for other control characters; UTF-8 passes through unchanged.
pub fn text(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// `0x`-prefixed hex for the 64-bit hashes the log reports.
pub fn hex(v: u64) -> String {
    format!("\"0x{:016x}\"", v)
}

/// FNV-1a 64 of a byte slice: the cheap file identity a `checkpoint` event
/// carries. Not a cryptographic digest; the run manifest is the place for
/// SHA-256.
pub fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

// ---------------------------------------------------------------------------
// The writer
// ---------------------------------------------------------------------------

/// The bounded, append-only NDJSON sink behind `--dev-log`.
///
/// One instance per run. `run_start` and `run_end` are *terminal* events and are
/// always written; everything else is dropped (and counted) once the byte cap
/// is reached, which keeps the file bounded without rotating and therefore
/// without ever leaving two half-files behind.
pub struct Log {
    path: PathBuf,
    file: File,
    max_bytes: u64,
    bytes: u64,
    events: u64,
    dropped: u64,
    oversize: u64,
}

impl Log {
    /// Create (or truncate) the log file.
    ///
    /// Truncation is deliberate: one file describes one run, so a stale log can
    /// never be mistaken for the current one. An empty `max_bytes` means "no
    /// cap", which the CLI refuses; it is kept for tests.
    pub fn create(path: impl AsRef<Path>, max_bytes: u64) -> Result<Log, String> {
        let path = path.as_ref().to_path_buf();
        // `O_APPEND` is what makes one `write` per event atomic; the truncation
        // is a separate call because this std rejects `truncate` combined with
        // `append` in one `OpenOptions`.
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("cannot open --dev-log '{}': {}", path.display(), e))?;
        file.set_len(0)
            .map_err(|e| format!("cannot truncate --dev-log '{}': {}", path.display(), e))?;
        Ok(Log {
            path,
            file,
            max_bytes,
            bytes: 0,
            events: 0,
            dropped: 0,
            oversize: 0,
        })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Bytes written so far.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// Events written so far.
    pub fn events(&self) -> u64 {
        self.events
    }

    /// Events dropped because the byte cap was reached.
    pub fn dropped(&self) -> u64 {
        self.dropped
    }

    /// Events that exceeded [`MAX_EVENT_BYTES`] (still written; a hint that the
    /// schema gained a large field).
    pub fn oversize(&self) -> u64 {
        self.oversize
    }

    /// Whether another event of roughly `len` bytes still fits.
    pub fn has_room(&self, len: usize) -> bool {
        self.max_bytes == 0 || self.bytes + len as u64 <= self.max_bytes
    }

    /// Append one complete event.
    ///
    /// The line is written with a single `write` (plus the trailing newline in
    /// the same buffer) to a file opened with `O_APPEND`, so a concurrent or
    /// killed writer can never interleave two events. Returns whether the event
    /// was written. `terminal` events bypass the cap.
    pub fn append(&mut self, line: &str, terminal: bool) -> bool {
        if line.len() > MAX_EVENT_BYTES {
            self.oversize += 1;
        }
        if !terminal && !self.has_room(line.len() + 1) {
            self.dropped += 1;
            return false;
        }
        let mut buf = Vec::with_capacity(line.len() + 1);
        buf.extend_from_slice(line.as_bytes());
        buf.push(b'\n');
        // No buffering layer: `write_all` on a `File` is one syscall per event.
        // `fsync` is deliberately not called — a flushed line survives a killed
        // process, and an fsync per generation would be a real cost.
        if let Err(e) = self.file.write_all(&buf) {
            eprintln!(
                "warning: --dev-log '{}': {} (the log may be incomplete)",
                self.path.display(),
                e
            );
            return false;
        }
        self.bytes += buf.len() as u64;
        self.events += 1;
        true
    }
}

// ---------------------------------------------------------------------------
// Event payloads
// ---------------------------------------------------------------------------

/// GPU numbers for one generation, taken as a delta of the run's ledger.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GpuGen {
    /// Evaluator batches in this generation.
    pub batches: u64,
    /// Compiled programs scored.
    pub programs: u64,
    /// GPU-event kernel time.
    pub kernel_s: f64,
    /// Independently measured wall of the evaluator calls.
    pub wall_s: f64,
    /// `kernel_s / wall_s`.
    pub duty: f64,
    /// Programs the kernel accepted verbatim.
    pub native: usize,
    /// Programs that needed the exact lowering pass.
    pub lowered: usize,
    /// Programs scored by the CPU fallback pool.
    pub fallbacks: usize,
    /// Wall the CPU fallback pool spent.
    pub fallback_s: f64,
    /// Worker threads in the fallback pool.
    pub fallback_workers: usize,
}

impl GpuGen {
    fn json(&self) -> String {
        format!(
            "{{\"batches\":{},\"programs\":{},\"kernel_s\":{},\"wall_s\":{},\"duty\":{},\
             \"native\":{},\"lowered\":{},\"fallbacks\":{},\"fallback_s\":{},\"fallback_workers\":{}}}",
            self.batches,
            self.programs,
            num(self.kernel_s),
            num(self.wall_s),
            num(self.duty),
            self.native,
            self.lowered,
            self.fallbacks,
            num(self.fallback_s),
            self.fallback_workers
        )
    }
}

/// Batched-constant-fit counters for one generation (#169 names).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FitGen {
    /// Candidates offered to the batched GPU fit.
    pub attempts: usize,
    /// Candidates the GPU fit returned usable constants for.
    pub gpu: usize,
    /// Candidates refit by the exact CPU arm.
    pub fallback: usize,
    /// Exact CPU f64 refits of finalists in this generation.
    pub refits: usize,
    /// CPU fallback counts labelled by cause.
    pub fallback_by_cause: Vec<(String, usize)>,
    /// #265: fallbacks whose CPU refit lowered the exact error, by cause.
    pub fallback_improved_by_cause: Vec<(String, usize)>,
    /// #256: wall seconds of the whole fit stage in this generation (device
    /// call, CPU fallback, write-back). Booked as `phases.fit`, not `eval`.
    pub host_seconds: f64,
    /// #256: wall seconds of the exact CPU fit-fallback inside `host_seconds`.
    pub fallback_seconds: f64,
    /// #256: workers the fit-fallback batch used. This is the fit pool; the
    /// top-level `fallback.workers` is the evaluator's pool.
    pub fallback_workers: usize,
}

impl FitGen {
    fn json(&self) -> String {
        let causes: Vec<String> = self
            .fallback_by_cause
            .iter()
            .map(|(k, v)| format!("{}:{}", text(k), v))
            .collect();
        let improved: Vec<String> = self
            .fallback_improved_by_cause
            .iter()
            .map(|(k, v)| format!("{}:{}", text(k), v))
            .collect();
        let improved_total: usize = self.fallback_improved_by_cause.iter().map(|(_, v)| *v).sum();
        format!(
            "{{\"attempts\":{},\"gpu\":{},\"fallback\":{},\"refits\":{},\"fallback_by_cause\":{{{}}},\"fallback_improved\":{},\"fallback_improved_by_cause\":{{{}}},\
             \"host_seconds\":{},\"fallback_seconds\":{},\"fallback_workers\":{}}}",
            self.attempts,
            self.gpu,
            self.fallback,
            self.refits,
            causes.join(","),
            improved_total,
            improved.join(","),
            num(self.host_seconds),
            num(self.fallback_seconds),
            self.fallback_workers
        )
    }
}

/// Quality-diversity coverage for one generation (only when `--qd` is on).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct QdGen {
    /// Occupied cells.
    pub occupied: usize,
    /// Cells the descriptor can actually reach.
    pub reachable: usize,
    /// `occupied_reachable / reachable`.
    pub coverage: f64,
    /// Shannon entropy of the visit distribution, bits.
    pub entropy: f64,
}

impl QdGen {
    fn json(&self) -> String {
        format!(
            "{{\"occupied\":{},\"reachable\":{},\"coverage\":{},\"entropy\":{}}}",
            self.occupied,
            self.reachable,
            num(self.coverage),
            num(self.entropy)
        )
    }
}

/// One generation's measured numbers — everything a `generation` event carries.
///
/// All of it is read from state the search already holds at the generation
/// boundary; the search fills this only when a development log is attached.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GenRecord {
    /// 1-based generation number, matching `--progress`.
    pub generation: usize,
    pub total_generations: usize,
    /// Seconds since the search started.
    pub t: f64,
    /// Generation wall up to the archive/convergence boundary.
    pub wall_s: f64,
    // Phase windows, all measured on the search's main thread.
    pub prep_s: f64,
    pub search_s: f64,
    /// Candidate evaluation alone; the fit stage is `fit_s` (#256).
    pub eval_s: f64,
    pub select_s: f64,
    pub archive_s: f64,
    /// #256: the constant-fit stage (GPU fit call, CPU fallback, write-back),
    /// taken out of the windows it ran in. `prep + search + eval + select +
    /// archive + fit = wall`.
    pub fit_s: f64,
    /// Constant-fitting CPU time: the island threads plus the exact CPU
    /// fit-fallback pool (wall seconds).
    pub fit_cpu_s: f64,
    /// Candidate-evaluation CPU time: the island threads plus the evaluator's
    /// CPU fallback pool (wall seconds).
    pub eval_cpu_s: f64,
    /// #463: wall seconds of the parallel fit-and-score phase, summed over
    /// islands.
    pub phase_b_s: f64,
    /// #463: candidates dropped after the fit for exceeding the complexity cap.
    pub post_fit_drops: usize,
    // Error distribution over this generation's population.
    pub best_error: f64,
    pub p05: f64,
    pub p25: f64,
    pub median: f64,
    pub p75: f64,
    pub p95: f64,
    // Archive.
    pub pareto_size: usize,
    pub hypervolume: f64,
    pub archive_hash: u64,
    /// Population candidates offered to the archive this generation.
    pub offers: usize,
    /// Offers the archive accepted as non-dominated at the time they arrived.
    pub admissions: usize,
    // Variation counters.
    pub attempts: usize,
    pub duplicates: usize,
    pub too_complex: usize,
    pub offspring: usize,
    /// Offspring whose constants were fitted on the CPU this generation.
    pub fitted: usize,
    pub refits: usize,
    // Optional sections.
    pub gpu: Option<GpuGen>,
    pub fit: FitGen,
    /// CPU fallback pool time for this generation, seconds.
    pub fallback_s: f64,
    pub fallback_workers: usize,
    pub qd: Option<QdGen>,
    // Convergence (#18) and structure.
    pub confidence: f64,
    pub stability: f64,
    pub maturity: f64,
    pub percent_converged: f64,
    /// Best error of each island's population.
    pub island_best: Vec<f64>,
    /// Hash of the island RNG states at this boundary: two logs with the same
    /// key at the same generation saw the same stream.
    pub rng_key: u64,
}

impl GenRecord {
    /// Structural duplicate rate over this generation's attempts.
    pub fn dedup_rate(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.duplicates as f64 / self.attempts as f64
        }
    }

    /// The complete NDJSON line (without the trailing newline), at `level`.
    pub fn json_at(&self, level: Level) -> String {
        match level {
            Level::Full => self.json(),
            Level::Summary => self.json_summary(),
        }
    }

    /// The concise `summary` event: what a monitor needs to follow a long run.
    ///
    /// It deliberately omits the population quantiles beyond the median, the
    /// archive hash/hypervolume, the variation counters, the per-island best
    /// list and the RNG key — those exist only in a `full` log.
    pub fn json_summary(&self) -> String {
        format!(
            "{{\"schema\":\"{}\",\"v\":{},\"event\":\"generation\",\"level\":\"summary\",\
             \"generation\":{},\"total\":{},\"t\":{},\"wall\":{},\
             \"phases\":{{\"prep\":{},\"search\":{},\"eval\":{},\"select\":{},\"archive\":{},\"fit\":{},\
             \"fit_cpu\":{},\"eval_cpu\":{},\"phase_b\":{}}},\
             \"error\":{{\"best\":{},\"median\":{}}},\
             \"archive\":{{\"size\":{}}},\
             \"fit\":{{\"attempts\":{},\"gpu\":{},\"fallback\":{}}},\
             \"gpu\":{},\"fallback\":{{\"scope\":\"evaluator_pool\",\"seconds\":{},\"workers\":{}}},\
             \"convergence\":{{\"percent_converged\":{}}}}}",
            SCHEMA,
            VERSION,
            self.generation,
            self.total_generations,
            num(self.t),
            num(self.wall_s),
            num(self.prep_s),
            num(self.search_s),
            num(self.eval_s),
            num(self.select_s),
            num(self.archive_s),
            num(self.fit_s),
            num(self.fit_cpu_s),
            num(self.eval_cpu_s),
            num(self.phase_b_s),
            num(self.best_error),
            num(self.median),
            self.pareto_size,
            self.fit.attempts,
            self.fit.gpu,
            self.fit.fallback,
            self.gpu.as_ref().map(|g| g.json()).unwrap_or_else(|| "null".to_string()),
            num(self.fallback_s),
            self.fallback_workers,
            num(self.percent_converged)
        )
    }

    /// The complete NDJSON line (without the trailing newline).
    pub fn json(&self) -> String {
        let islands: Vec<String> = self.island_best.iter().map(|v| num(*v)).collect();
        format!(
            "{{\"schema\":\"{}\",\"v\":{},\"event\":\"generation\",\"level\":\"full\",\
             \"generation\":{},\"total\":{},\
             \"t\":{},\"wall\":{},\
             \"phases\":{{\"prep\":{},\"search\":{},\"eval\":{},\"select\":{},\"archive\":{},\"fit\":{},\
             \"fit_cpu\":{},\"eval_cpu\":{},\"phase_b\":{}}},\
             \"error\":{{\"best\":{},\"p05\":{},\"p25\":{},\"median\":{},\"p75\":{},\"p95\":{}}},\
             \"archive\":{{\"size\":{},\"hypervolume\":{},\"hash\":{},\"offers\":{},\"admissions\":{}}},\
             \"variation\":{{\"attempts\":{},\"duplicates\":{},\"too_complex\":{},\"offspring\":{},\
             \"fitted\":{},\"refits\":{},\"dedup_rate\":{},\"post_fit_drops\":{}}},\
             \"gpu\":{},\"fit\":{},\
             \"fallback\":{{\"scope\":\"evaluator_pool\",\"seconds\":{},\"workers\":{}}},\
             \"qd\":{},\
             \"convergence\":{{\"confidence\":{},\"stability\":{},\"maturity\":{},\
             \"percent_converged\":{}}},\
             \"islands\":[{}],\"rng\":{}}}",
            SCHEMA,
            VERSION,
            self.generation,
            self.total_generations,
            num(self.t),
            num(self.wall_s),
            num(self.prep_s),
            num(self.search_s),
            num(self.eval_s),
            num(self.select_s),
            num(self.archive_s),
            num(self.fit_s),
            num(self.fit_cpu_s),
            num(self.eval_cpu_s),
            num(self.phase_b_s),
            num(self.best_error),
            num(self.p05),
            num(self.p25),
            num(self.median),
            num(self.p75),
            num(self.p95),
            self.pareto_size,
            num(self.hypervolume),
            hex(self.archive_hash),
            self.offers,
            self.admissions,
            self.attempts,
            self.duplicates,
            self.too_complex,
            self.offspring,
            self.fitted,
            self.refits,
            num(self.dedup_rate()),
            self.post_fit_drops,
            self.gpu.as_ref().map(|g| g.json()).unwrap_or_else(|| "null".to_string()),
            self.fit.json(),
            num(self.fallback_s),
            self.fallback_workers,
            self.qd.as_ref().map(|q| q.json()).unwrap_or_else(|| "null".to_string()),
            num(self.confidence),
            num(self.stability),
            num(self.maturity),
            num(self.percent_converged),
            islands.join(","),
            hex(self.rng_key)
        )
    }
}

/// What `run_start` records about the run.
#[derive(Clone, Debug, Default)]
pub struct RunStart {
    /// The command line, verbatim (`--dev-log` included), so a log explains
    /// itself even when it is detached from the shell history.
    pub command: String,
    pub version: String,
    pub commit: String,
    pub dirty: bool,
    pub binary_sha256: String,
    pub data_path: String,
    pub data_sha256: Option<String>,
    pub train_rows: usize,
    pub train_fingerprint: String,
    pub test_rows: usize,
    pub test_fingerprint: String,
    /// #112: `Some(true)` full rows, `Some(false)` explicit sample, `None` CSV.
    pub full_rows: Option<bool>,
    pub sampled_rows: Option<usize>,
    pub estimate_bytes: Option<u64>,
    /// Canonical configuration lines, already `key value` pairs.
    pub config: Vec<(String, String)>,
    pub config_hash: u64,
    pub seed: u32,
    pub backend: String,
    pub device: Option<String>,
    pub threads: usize,
    pub host: String,
    pub arch: String,
    pub start_utc: String,
    /// `--dev-log-level`.
    pub level: Level,
    /// The resolved generation cadence.
    pub every: usize,
    /// `--dev-log-max-bytes`, `0` for no cap.
    pub max_bytes: u64,
    /// Set when this run continues a checkpoint.
    pub resumed_from: Option<String>,
    /// #223: `(source, reason)` when the engine resolved the fit arm.
    pub fit_resolution: Option<(String, String)>,
}

fn config_json(config: &[(String, String)]) -> String {
    let mut fields = Vec::with_capacity(config.len());
    for (k, v) in config {
        // Every canonical line is `<name> <value>`; keep numeric values numeric
        // so a reader can compare them without parsing strings.
        let value = match v.parse::<f64>() {
            Ok(n) if n.is_finite() => num(n),
            _ => text(v),
        };
        fields.push(format!("{}:{}", text(k), value));
    }
    format!("{{{}}}", fields.join(","))
}

/// One line for `run_start`. Terminal: always written, never capped.
pub fn run_start_json(r: &RunStart) -> String {
    let mut json = format!(
        "{{\"schema\":\"{}\",\"v\":{},\"event\":\"run_start\",\"command\":{},\"version\":{},\
         \"commit\":{},\"dirty\":{},\"binary_sha256\":{},\
         \"data\":{{\"path\":{},\"sha256\":{},\"full_rows\":{},\"sampled_rows\":{},\
         \"estimate_bytes\":{},\"train\":{{\"rows\":{},\"fingerprint\":{}}},\
         \"test\":{{\"rows\":{},\"fingerprint\":{}}}}},\
         \"config_hash\":{},\"config\":{},\
         \"seed\":{},\"backend\":{},\"device\":{},\"threads\":{},\
         \"host\":{{\"os\":{},\"arch\":{}}},\
         \"start\":{},\"level\":{},\"every\":{},\"max_bytes\":{},\"resumed_from\":{}}}",
        SCHEMA,
        VERSION,
        text(&r.command),
        text(&r.version),
        text(&r.commit),
        r.dirty,
        text(&r.binary_sha256),
        text(&r.data_path),
        match &r.data_sha256 {
            Some(s) => text(s),
            None => "null".to_string(),
        },
        r.full_rows.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
        r.sampled_rows.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
        r.estimate_bytes.map(|v| v.to_string()).unwrap_or_else(|| "null".to_string()),
        r.train_rows,
        text(&r.train_fingerprint),
        r.test_rows,
        text(&r.test_fingerprint),
        hex(r.config_hash),
        config_json(&r.config),
        r.seed,
        text(&r.backend),
        r.device.as_ref().map(|d| text(d)).unwrap_or_else(|| "null".into()),
        r.threads,
        text(&r.host),
        text(&r.arch),
        text(&r.start_utc),
        text(r.level.as_str()),
        r.every,
        r.max_bytes,
        r.resumed_from.as_ref().map(|p| text(p)).unwrap_or_else(|| "null".into())
    );
    if let Some((source, reason)) = &r.fit_resolution {
        json.pop();
        json.push_str(&format!(
            ",\"fit_strategy_source\":{},\"fit_strategy_reason\":{}}}",
            text(source),
            text(reason)
        ));
    }
    json
}

/// What `run_end` records about the run.
#[derive(Clone, Debug, Default)]
pub struct RunEnd {
    pub generations: usize,
    /// `solved`, `patience`, `time_budget` or `generations`.
    pub stop: String,
    /// Seconds since the CLI took the clock, including data preparation.
    pub wall_s: f64,
    /// The search phase alone (the historical `elapsed_seconds`).
    pub search_s: f64,
    pub pareto_size: usize,
    pub best_error: f64,
    pub archive_hash: u64,
    /// Where the final result JSON went; `null` when it streamed to stdout.
    pub json_path: Option<String>,
    pub peak_rss_bytes: Option<u64>,
    /// GPU full-path wall of the evaluator calls only (the evaluator window).
    pub gpu_wall_s: Option<f64>,
    /// #256: `device_seconds_incl_fit / 3600`.
    pub device_hours: Option<f64>,
    /// #256: evaluator full-path wall plus the fit kernels' device seconds.
    pub device_seconds_incl_fit: Option<f64>,
    /// #256: `device_seconds_incl_fit / wall_s` (the process wall at the end
    /// of the search).
    pub device_fraction_of_run: Option<f64>,
    /// #284: run wall from the ledger.
    pub run_wall_s: Option<f64>,
    /// #284: union of device intervals over the run.
    pub device_busy_s: Option<f64>,
    /// #284: `device_busy_s / run_wall_s`.
    pub duty_run: Option<f64>,
    /// #284: wall not attributed to an exclusive stage.
    pub unaccounted_s: Option<f64>,
    /// #284: fit kernel seconds from raw device events.
    pub fit_kernel_s: Option<f64>,
    /// #496: `fit_kernel_s` apportioned to the LM phase and to the polish phase.
    pub fit_kernel_f32_s: Option<f64>,
    pub fit_kernel_f64_s: Option<f64>,
    /// #284: exact CPU fit-fallback seconds.
    pub fit_fallback_s: Option<f64>,
    /// #495: exact finalist refit seconds.
    pub fit_refit_s: Option<f64>,
    /// #495: exact finalist refits performed.
    pub fit_refits: Option<usize>,
    /// Log bookkeeping: events written, bytes, events dropped by the cap.
    pub events: u64,
    pub bytes: u64,
    pub dropped: u64,
    pub end_utc: String,
}

/// One line for `run_end`. Terminal: always written, never capped.
pub fn run_end_json(r: &RunEnd) -> String {
    format!(
        "{{\"schema\":\"{}\",\"v\":{},\"event\":\"run_end\",\"generations\":{},\"stop\":{},\
         \"wall\":{},\"search\":{},\"archive\":{{\"size\":{},\"best_error\":{},\"hash\":{}}},\
         \"json\":{},\"peak_rss_bytes\":{},\"gpu_wall_s\":{},\"device_hours\":{},\
         \"device_seconds_incl_fit\":{},\"device_fraction_of_run\":{},\
         \"run_wall_s\":{},\"device_busy_s\":{},\"duty_run\":{},\"unaccounted_s\":{},\
         \"fit_kernel_s\":{},\"fit_kernel_f32_s\":{},\"fit_kernel_f64_s\":{},\"fit_fallback_s\":{},\"fit_refit_s\":{},\"fit_refits\":{},\
         \"events\":{},\"bytes\":{},\"dropped\":{},\"end\":{}}}",
        SCHEMA,
        VERSION,
        r.generations,
        text(&r.stop),
        num(r.wall_s),
        num(r.search_s),
        r.pareto_size,
        num(r.best_error),
        hex(r.archive_hash),
        r.json_path.as_ref().map(|p| text(p)).unwrap_or_else(|| "null".into()),
        match r.peak_rss_bytes {
            Some(n) => n.to_string(),
            None => "null".to_string(),
        },
        match r.gpu_wall_s {
            Some(v) => num(v),
            None => "null".to_string(),
        },
        match r.device_hours {
            Some(v) => num(v),
            None => "null".to_string(),
        },
        opt_num(r.device_seconds_incl_fit),
        opt_num(r.device_fraction_of_run),
        opt_num(r.run_wall_s),
        opt_num(r.device_busy_s),
        opt_num(r.duty_run),
        opt_num(r.unaccounted_s),
        opt_num(r.fit_kernel_s),
        opt_num(r.fit_kernel_f32_s),
        opt_num(r.fit_kernel_f64_s),
        opt_num(r.fit_fallback_s),
        opt_num(r.fit_refit_s),
        match r.fit_refits {
            Some(n) => n.to_string(),
            None => "null".to_string(),
        },
        r.events,
        r.bytes,
        r.dropped,
        text(&r.end_utc)
    )
}

/// One line for a `checkpoint` event.
pub fn checkpoint_json(generation: usize, path: &str, bytes: u64, hash: u64) -> String {
    format!(
        "{{\"schema\":\"{}\",\"v\":{},\"event\":\"checkpoint\",\"generation\":{},\"path\":{},\
         \"bytes\":{},\"hash\":{}}}",
        SCHEMA,
        VERSION,
        generation,
        text(path),
        bytes,
        hex(hash)
    )
}

// ---------------------------------------------------------------------------
// Peak RSS
// ---------------------------------------------------------------------------

/// Peak resident set size of this process in bytes, from Linux
/// `/proc/self/status` (`VmHWM`). `None` on a platform without that file.
///
/// This is the kernel's high-water mark for the whole process, so it is
/// comparable between two runs of the same binary even when the log's own
/// buffers are part of it.
pub fn peak_rss_bytes() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
            return Some(kb * 1024);
        }
    }
    None
}

// ---------------------------------------------------------------------------
// A small JSON reader for `dev-report`
// ---------------------------------------------------------------------------

/// The JSON subset the log uses. Written by hand because the crate is pure
/// `std`; it is a reader for events this crate produced, not a general parser.
#[derive(Clone, Debug, PartialEq)]
pub enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    /// Parse one complete line. `None` for anything malformed, including the
    /// trailing partial line a killed run can leave behind.
    pub fn parse(src: &str) -> Option<Json> {
        let mut p = Parser {
            b: src.as_bytes(),
            i: 0,
        };
        p.ws();
        let v = p.value(0)?;
        p.ws();
        if p.i == p.b.len() {
            Some(v)
        } else {
            None
        }
    }

    pub fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(fields) => fields.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Json::Num(v) => Some(*v),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_f64().map(|v| v as usize)
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, Json)]> {
        match self {
            Json::Obj(fields) => Some(fields),
            _ => None,
        }
    }
}

struct Parser<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Parser<'a> {
    fn ws(&mut self) {
        while self.i < self.b.len() && matches!(self.b[self.i], b' ' | b'\t' | b'\r' | b'\n') {
            self.i += 1;
        }
    }

    fn eat(&mut self, c: u8) -> Option<()> {
        if self.i < self.b.len() && self.b[self.i] == c {
            self.i += 1;
            Some(())
        } else {
            None
        }
    }

    fn value(&mut self, depth: usize) -> Option<Json> {
        if depth > 32 {
            return None;
        }
        self.ws();
        match *self.b.get(self.i)? {
            b'{' => self.object(depth),
            b'[' => self.array(depth),
            b'"' => self.string().map(Json::Str),
            b't' => self.literal("true").map(|_| Json::Bool(true)),
            b'f' => self.literal("false").map(|_| Json::Bool(false)),
            b'n' => self.literal("null").map(|_| Json::Null),
            _ => self.number(),
        }
    }

    fn literal(&mut self, word: &str) -> Option<()> {
        if self.b[self.i..].starts_with(word.as_bytes()) {
            self.i += word.len();
            Some(())
        } else {
            None
        }
    }

    fn object(&mut self, depth: usize) -> Option<Json> {
        self.eat(b'{')?;
        let mut fields = Vec::new();
        self.ws();
        if self.eat(b'}').is_some() {
            return Some(Json::Obj(fields));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            self.eat(b':')?;
            let v = self.value(depth + 1)?;
            fields.push((key, v));
            self.ws();
            if self.eat(b',').is_some() {
                continue;
            }
            self.eat(b'}')?;
            return Some(Json::Obj(fields));
        }
    }

    fn array(&mut self, depth: usize) -> Option<Json> {
        self.eat(b'[')?;
        let mut items = Vec::new();
        self.ws();
        if self.eat(b']').is_some() {
            return Some(Json::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.ws();
            if self.eat(b',').is_some() {
                continue;
            }
            self.eat(b']')?;
            return Some(Json::Arr(items));
        }
    }

    fn string(&mut self) -> Option<String> {
        self.eat(b'"')?;
        let mut out = String::new();
        loop {
            let c = *self.b.get(self.i)?;
            self.i += 1;
            match c {
                b'"' => return Some(out),
                b'\\' => {
                    let e = *self.b.get(self.i)?;
                    self.i += 1;
                    match e {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'u' => {
                            let hex = self.b.get(self.i..self.i + 4)?;
                            let code = u32::from_str_radix(
                                std::str::from_utf8(hex).ok()?,
                                16,
                            )
                            .ok()?;
                            self.i += 4;
                            out.push(char::from_u32(code)?);
                        }
                        _ => return None,
                    }
                }
                _ => {
                    // Re-decode UTF-8 from the byte stream one char at a time.
                    let start = self.i - 1;
                    let width = utf8_width(c);
                    let bytes = self.b.get(start..start + width)?;
                    out.push_str(std::str::from_utf8(bytes).ok()?);
                    self.i = start + width;
                }
            }
        }
    }

    fn number(&mut self) -> Option<Json> {
        let start = self.i;
        if self.b.get(self.i) == Some(&b'-') {
            self.i += 1;
        }
        while self
            .b
            .get(self.i)
            .is_some_and(|c| c.is_ascii_digit() || matches!(c, b'.' | b'e' | b'E' | b'+' | b'-'))
        {
            self.i += 1;
        }
        let token = std::str::from_utf8(self.b.get(start..self.i)?).ok()?;
        token.parse::<f64>().ok().map(Json::Num)
    }
}

fn utf8_width(first: u8) -> usize {
    match first {
        0x00..=0x7f => 1,
        0xc0..=0xdf => 2,
        0xe0..=0xef => 3,
        _ => 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec() -> GenRecord {
        GenRecord {
            generation: 3,
            total_generations: 10,
            t: 1.5,
            wall_s: 0.25,
            prep_s: 0.01,
            search_s: 0.2,
            eval_s: 0.02,
            select_s: 0.01,
            archive_s: 0.01,
            fit_s: 0.05,
            fit_cpu_s: 0.18,
            eval_cpu_s: 0.02,
            phase_b_s: 0.0,
            post_fit_drops: 0,
            best_error: 0.5,
            p05: 0.6,
            p25: 0.7,
            median: 0.8,
            p75: 0.9,
            p95: 1.0,
            pareto_size: 4,
            hypervolume: 12.5,
            archive_hash: 0xdead_beef,
            offers: 20,
            admissions: 2,
            attempts: 40,
            duplicates: 15,
            too_complex: 3,
            offspring: 20,
            fitted: 20,
            refits: 2,
            gpu: Some(GpuGen {
                batches: 2,
                programs: 20,
                kernel_s: 0.004,
                wall_s: 0.02,
                duty: 0.2,
                native: 18,
                lowered: 2,
                fallbacks: 1,
                fallback_s: 0.001,
                fallback_workers: 1,
            }),
            fit: FitGen {
                attempts: 20,
                gpu: 18,
                fallback: 2,
                refits: 1,
                fallback_by_cause: vec![("ill_conditioned".to_string(), 2)],
                fallback_improved_by_cause: vec![("ill_conditioned".to_string(), 1)],
                host_seconds: 0.05,
                fallback_seconds: 0.03,
                fallback_workers: 4,
            },
            fallback_s: 0.001,
            fallback_workers: 1,
            qd: Some(QdGen {
                occupied: 4,
                reachable: 9,
                coverage: 4.0 / 9.0,
                entropy: 1.75,
            }),
            confidence: 0.9,
            stability: 0.8,
            maturity: 0.7,
            percent_converged: 0.85,
            island_best: vec![0.5, 0.55],
            rng_key: 0x1234,
        }
    }

    /// The generation line is complete JSON that round-trips through the
    /// crate's own reader, including the nested objects and the hash strings.
    #[test]
    fn generation_line_is_valid_json_that_round_trips() {
        let line = rec().json();
        let parsed = Json::parse(&line).expect("the generation line must parse");
        assert_eq!(parsed.get("schema").and_then(|v| v.as_str()), Some(SCHEMA));
        assert_eq!(parsed.get("event").and_then(|v| v.as_str()), Some("generation"));
        assert_eq!(parsed.get("generation").and_then(|v| v.as_usize()), Some(3));
        let phases = parsed.get("phases").unwrap();
        assert_eq!(phases.get("fit_cpu").and_then(|v| v.as_f64()), Some(0.18));
        assert_eq!(phases.get("fit").and_then(|v| v.as_f64()), Some(0.05));
        let archive = parsed.get("archive").unwrap();
        assert_eq!(archive.get("size").and_then(|v| v.as_usize()), Some(4));
        assert_eq!(
            archive.get("hash").and_then(|v| v.as_str()),
            Some("0x00000000deadbeef")
        );
        let gpu = parsed.get("gpu").unwrap();
        assert_eq!(gpu.get("native").and_then(|v| v.as_usize()), Some(18));
        let fit = parsed.get("fit").unwrap();
        let causes = fit.get("fallback_by_cause").and_then(|v| v.as_obj()).unwrap();
        assert_eq!(causes[0].0, "ill_conditioned");
        assert_eq!(causes[0].1.as_usize(), Some(2));
        assert_eq!(fit.get("fallback_improved").and_then(|v| v.as_usize()), Some(1));
        let improved = fit
            .get("fallback_improved_by_cause")
            .and_then(|v| v.as_obj())
            .unwrap();
        assert_eq!(improved[0].0, "ill_conditioned");
        assert_eq!(improved[0].1.as_usize(), Some(1));
        assert_eq!(fit.get("host_seconds").and_then(|v| v.as_f64()), Some(0.05));
        assert_eq!(fit.get("fallback_seconds").and_then(|v| v.as_f64()), Some(0.03));
        assert_eq!(fit.get("fallback_workers").and_then(|v| v.as_usize()), Some(4));
        let eval_pool = parsed.get("fallback").unwrap();
        assert_eq!(
            eval_pool.get("scope").and_then(|v| v.as_str()),
            Some("evaluator_pool")
        );
        let islands = parsed.get("islands").and_then(|v| match v {
            Json::Arr(a) => Some(a.clone()),
            _ => None,
        });
        assert_eq!(islands.unwrap().len(), 2);
        // A null section (no GPU / no QD) parses as JSON null, not a missing key.
        let mut r = rec();
        r.gpu = None;
        r.qd = None;
        let line = r.json();
        let parsed = Json::parse(&line).unwrap();
        assert_eq!(parsed.get("gpu"), Some(&Json::Null));
        assert_eq!(parsed.get("qd"), Some(&Json::Null));
        assert_eq!(
            parsed.get("variation").and_then(|v| v.get("dedup_rate")).and_then(|v| v.as_f64()),
            Some(15.0 / 40.0)
        );
    }

    /// The summary level is a strict subset: it keeps what a monitor needs and
    /// drops the per-generation detail a full log adds.
    #[test]
    fn the_summary_event_keeps_the_monitoring_fields_only() {
        let r = rec();
        let line = r.json_at(Level::Summary);
        let j = Json::parse(&line).expect("the summary line parses");
        assert_eq!(j.get("level").and_then(|v| v.as_str()), Some("summary"));
        for key in ["generation", "t", "wall", "phases", "error", "archive", "fit", "gpu", "fallback", "convergence"] {
            assert!(j.get(key).is_some(), "summary has no '{}'", key);
        }
        assert_eq!(j.get("error").and_then(|e| e.get("best")).and_then(|v| v.as_f64()), Some(0.5));
        assert_eq!(j.get("error").and_then(|e| e.get("median")).and_then(|v| v.as_f64()), Some(0.8));
        assert_eq!(j.get("archive").and_then(|a| a.get("size")).and_then(|v| v.as_usize()), Some(4));
        assert_eq!(j.get("fit").and_then(|f| f.get("fallback")).and_then(|v| v.as_usize()), Some(2));
        assert_eq!(j.get("gpu").and_then(|g| g.get("duty")).and_then(|v| v.as_f64()), Some(0.2));
        for key in ["variation", "islands", "rng"] {
            assert!(j.get(key).is_none(), "summary must not carry '{}'", key);
        }
        let archive = j.get("archive").unwrap();
        assert!(archive.get("hash").is_none() && archive.get("hypervolume").is_none());
        // The full line of the same record keeps them.
        let full = Json::parse(&r.json()).unwrap();
        assert_eq!(full.get("level").and_then(|v| v.as_str()), Some("full"));
        assert!(full.get("variation").is_some() && full.get("islands").is_some());
        assert!(line.len() < r.json().len(), "summary must be smaller than full");
    }

    /// The default cadence keeps a long run to a bounded number of events.
    #[test]
    fn the_level_cadence_is_bounded() {
        assert_eq!(Level::Full.default_every(5000), 1);
        assert_eq!(Level::Summary.default_every(1), 1);
        assert_eq!(Level::Summary.default_every(10), 1);
        assert_eq!(Level::Summary.default_every(40), 10);
        assert_eq!(Level::Summary.default_every(200), 20);
        assert_eq!(Level::Summary.default_every(10_000), 25);
        // A 10-hour run at 100 000 generations still writes a few thousand
        // events, not one hundred thousand.
        assert!(100_000 / Level::Summary.default_every(100_000) <= 4_000);
        assert_eq!(Level::parse("summary"), Some(Level::Summary));
        assert_eq!(Level::parse("full"), Some(Level::Full));
        assert_eq!(Level::parse("Full"), None);
        assert_eq!(Level::default(), Level::Summary);
    }

    /// A trailing partial line (what a killed process can leave) is rejected by
    /// the parser, so the reader's valid-prefix rule is testable.
    #[test]
    fn a_partial_trailing_line_does_not_parse() {
        let line = rec().json();
        for cut in [1, line.len() / 2, line.len() - 1] {
            assert!(Json::parse(&line[..cut]).is_none(), "cut at {} parsed", cut);
        }
        assert!(Json::parse(&line).is_some());
    }

    #[test]
    fn json_reader_handles_escapes_and_non_finite_numbers() {
        let j = Json::parse(r#"{"a":"x\"y\\z\u00e9","b":[1,-2.5e3],"c":true,"d":null}"#).unwrap();
        assert_eq!(j.get("a").and_then(|v| v.as_str()), Some("x\"y\\zé"));
        assert_eq!(j.get("b").and_then(|v| v.as_f64()), None);
        assert_eq!(j.get("c"), Some(&Json::Bool(true)));
        assert_eq!(j.get("d"), Some(&Json::Null));
        // Non-finite numbers become null, never `NaN`, so the line stays JSON.
        assert!(Json::parse(&format!("[{}]", num(f64::NAN))).is_some());
        assert_eq!(Json::parse("[NaN]"), None);
    }

    #[test]
    fn strings_are_escaped_and_paths_survive() {
        assert_eq!(text("a\"b\\c\nd\t"), "\"a\\\"b\\\\c\\nd\\t\"");
        assert_eq!(text("é"), "\"é\"");
    }

    /// Each event is one line, and the cap bounds generation events while the
    /// terminal events always land.
    #[test]
    fn the_byte_cap_bounds_generation_events_but_never_the_end() {
        let dir = std::env::temp_dir().join(format!("rcsr-devlog-cap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cap.ndjson");
        // Room for the start event plus one full generation event (a
        // generation line is ~1 KiB with every optional section present).
        let cap = run_start_json(&RunStart::default()).len() as u64 + 1300;
        let mut log = Log::create(&path, cap).unwrap();
        assert!(log.append(&run_start_json(&RunStart::default()), true));
        let line = rec().json();
        let mut written = 0;
        for _ in 0..50 {
            if log.append(&line, false) {
                written += 1;
            }
        }
        assert!(written >= 1, "at least one event fits under the cap");
        assert!(log.dropped() > 0, "the cap must drop events");
        assert!(log.bytes() <= cap, "generation events never exceed the cap");
        assert!(log.append(&run_end_json(&RunEnd::default()), true));
        let body = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), written as usize + 2);
        assert!(lines.first().unwrap().contains("run_start"));
        assert!(lines.last().unwrap().contains("run_end"));
        // Every complete line is valid JSON.
        for l in &lines {
            assert!(Json::parse(l).is_some(), "line is not JSON: {}", l);
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A killed run leaves a valid prefix: truncating the file mid-line still
    /// leaves every earlier line parseable.
    #[test]
    fn a_truncated_log_keeps_a_valid_prefix() {
        let dir = std::env::temp_dir().join(format!("rcsr-devlog-kill-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("kill.ndjson");
        let mut log = Log::create(&path, 0).unwrap();
        log.append(&run_start_json(&RunStart::default()), true);
        for _ in 0..3 {
            log.append(&rec().json(), false);
        }
        let full = std::fs::read(&path).unwrap();
        // Cut the file in the middle of the last event, as a SIGKILL would.
        let cut = full.len() - 20;
        std::fs::write(&path, &full[..cut]).unwrap();
        let body = std::fs::read_to_string(&path).unwrap();
        let mut good = 0;
        let mut bad = 0;
        for line in body.split_inclusive('\n') {
            match Json::parse(line.trim_end_matches('\n')) {
                Some(_) => good += 1,
                None => bad += 1,
            }
        }
        assert_eq!(good, 3, "run_start plus two complete generations survive");
        assert_eq!(bad, 1, "only the torn last line is unreadable");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// #211: the cost of one generation event on its own — render the record,
    /// format the JSON and append it to a file — so the overhead claim has a
    /// measured per-event number next to the end-to-end A/B. The bound is
    /// deliberately loose (5 ms); the printed figure is the datum quoted in
    /// docs/dev-log.md and the issue.
    #[test]
    fn one_generation_event_costs_well_under_a_millisecond() {
        let dir = std::env::temp_dir().join(format!("rcsr-devlog-cost-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cost.ndjson");
        let mut log = Log::create(&path, 0).unwrap();
        let record = rec();
        // Warm the formatter and the page cache.
        for _ in 0..100 {
            let line = record.json();
            log.append(&line, false);
        }
        let n = 2000;
        let mut cost = |level: Level| -> (f64, usize) {
            let t = std::time::Instant::now();
            let mut out = String::new();
            for _ in 0..n {
                out = record.json_at(level);
                log.append(&out, false);
            }
            (t.elapsed().as_secs_f64() * 1e6 / n as f64, out.len())
        };
        let (summary_us, summary_bytes) = cost(Level::Summary);
        let (per_event_us, full_bytes) = cost(Level::Full);
        println!(
            "# dev-log cost: summary {:.2} us/event ({} bytes), full {:.2} us/event ({} bytes)",
            summary_us, summary_bytes, per_event_us, full_bytes
        );
        assert!(
            summary_bytes < full_bytes,
            "the summary record must be smaller than the full one"
        );
        assert!(
            per_event_us < 5000.0,
            "one event took {:.0} us; the log would not be free",
            per_event_us
        );
        assert!((log.bytes() as f64) > 0.0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn peak_rss_is_plausible_when_available() {
        if let Some(bytes) = peak_rss_bytes() {
            assert!(bytes > 1024 * 1024, "a running test process has > 1 MiB peak RSS");
        }
    }

    #[test]
    fn run_start_and_end_carry_the_schema_and_never_cap() {
        let start = run_start_json(&RunStart {
            version: "0.1.0".into(),
            commit: "abc1234".into(),
            config: vec![("seed".into(), "3".into()), ("metric".into(), "squared".into())],
            ..RunStart::default()
        });
        assert!(start.starts_with(&format!("{{\"schema\":\"{}\"", SCHEMA)));
        let parsed = Json::parse(&start).unwrap();
        assert_eq!(parsed.get("level").and_then(|v| v.as_str()), Some("summary"));
        assert_eq!(
            parsed.get("config").and_then(|c| c.get("seed")).and_then(|v| v.as_f64()),
            Some(3.0)
        );
        assert_eq!(
            parsed.get("config").and_then(|c| c.get("metric")).and_then(|v| v.as_str()),
            Some("squared")
        );
        let end = run_end_json(&RunEnd {
            stop: "patience".into(),
            ..RunEnd::default()
        });
        assert!(Json::parse(&end).is_some());
        assert_eq!(
            Json::parse(&end).unwrap().get("stop").and_then(|v| v.as_str()),
            Some("patience")
        );
    }
}
