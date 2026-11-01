//! Evaluation protocol: windows, candidate ledger, sealed list, one-read lockbox.
//!
//! Four tiers: training fits, validation selects, development out-of-sample
//! evaluates the procedure, and a locked test is read once per sealed
//! candidate. Everything here is plain files and hashes.
//!
//! # Windows
//!
//! [`default_windows`] builds the UTC table in `protocol/windows.csv`.
//! Warm-up is 2021 and is never trained on. Development is `2022-01-01` to
//! `2025-07-01` (exclusive). Expanding quarterly folds train from `2022-01-01`
//! and start testing after 12 months; rolling folds train on the previous 24
//! months and start testing after 24 months. The locked test is
//! `2025-07-01` to `2026-07-01` (exclusive). `2026H2` stays reserved by
//! [`crate::lockbox`]. Every training range ends `purge_seconds` before its
//! test range and its validation slice is the trailing six calendar months
//! before the test start.
//!
//! # Ledger
//!
//! One JSON object per line, append-only. Each line carries `prev`, the SHA-256
//! of the previous line, so an edit anywhere breaks the chain
//! ([`Ledger::verify_chain`]). Lines:
//!
//! * `candidate`: a development run (`id`, `expression`, `spec_sha256`,
//!   `data_sha256`, `build_commit`, `report_sha256`, `test_read:false`).
//! * `seal`: the SHA-256 of `locked_candidates.json`, written before any read.
//! * `test_read`: one locked read (`id`, `report_sha256`, `read_count`,
//!   `test_read:true`).
//!
//! A locked read needs a `seal` line whose hash matches the sealed file, an id
//! inside that file, no earlier `test_read` for the id, and the same data and
//! build hashes that the sealed entry recorded.

use crate::manifest::sha256_hex;
use crate::walkforward::Window;
use std::io::Write;

const MONTHS_PER_QUARTER: i32 = 3;

/// Default purge: 1 h entry offset plus a 60 h label on the hourly panel.
pub const DEFAULT_PURGE_SECONDS: f64 = 219_600.0;

/// Declared locked test, UTC half-open `[start, end)`.
pub const LOCKED_TEST_START: &str = "2025-07-01";
pub const LOCKED_TEST_END: &str = "2026-07-01";

/// Epoch seconds of `y-m-01T00:00:00` after adding `months` (may be negative).
fn month_start(y: i32, m: i32, months: i32) -> f64 {
    let total = y * 12 + (m - 1) + months;
    let (yy, mm) = (total.div_euclid(12), total.rem_euclid(12) + 1);
    crate::ohlcv::days_from_civil(yy, mm as u32, 1) as f64 * 86_400.0
}

pub fn iso(t: f64) -> String {
    let secs = t.round() as i64;
    let (y, m, d) = crate::ohlcv::civil_from_days(secs.div_euclid(86_400));
    let rem = secs.rem_euclid(86_400);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
        y,
        m,
        d,
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

/// One row of `protocol/windows.csv`.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowRow {
    pub scheme: String,
    pub window: Window,
    pub validation_start: f64,
}

pub const SCHEMES: [&str; 3] = ["expanding", "rolling", "locked"];

pub fn default_windows(purge_seconds: f64) -> Vec<WindowRow> {
    let dev_start = month_start(2022, 1, 0);
    let locked_start = month_start(2025, 7, 0);
    let dev_end = locked_start - purge_seconds;
    let mut rows = Vec::new();
    let quarter = |scheme: &str, label: String, train_start: f64, test_start: f64, test_end: f64| WindowRow {
        scheme: scheme.to_string(),
        window: Window {
            label,
            train_start,
            train_end: test_start - purge_seconds,
            test_start,
            test_end,
        },
        validation_start: {
            let (y, m) = civil_ym(test_start);
            month_start(y, m, -6)
        },
    };
    for step in 0..10 {
        let months = 12 + step * MONTHS_PER_QUARTER;
        let test_start = month_start(2022, 1, months);
        let test_end = month_start(2022, 1, months + MONTHS_PER_QUARTER).min(dev_end);
        let (y, m) = civil_ym(test_start);
        rows.push(quarter(
            "expanding",
            format!("expanding-{}Q{}", y, (m - 1) / 3 + 1),
            dev_start,
            test_start,
            test_end,
        ));
    }
    for step in 0..6 {
        let months = 24 + step * MONTHS_PER_QUARTER;
        let test_start = month_start(2022, 1, months);
        let test_end = month_start(2022, 1, months + MONTHS_PER_QUARTER).min(dev_end);
        let (y, m) = civil_ym(test_start);
        rows.push(quarter(
            "rolling",
            format!("rolling24-{}Q{}", y, (m - 1) / 3 + 1),
            month_start(y, m, -24),
            test_start,
            test_end,
        ));
    }
    rows.push(quarter(
        "locked",
        "locked-test".to_string(),
        dev_start,
        locked_start,
        month_start(2026, 7, 0),
    ));
    rows
}

fn civil_ym(t: f64) -> (i32, i32) {
    let (y, m, _) = crate::ohlcv::civil_from_days((t as i64).div_euclid(86_400));
    (y, m as i32)
}

pub const WINDOWS_HEADER: &str = "scheme,label,train_start,train_end,validation_start,test_start,test_end";

/// The text of `protocol/windows.csv`.
pub fn windows_csv(purge_seconds: f64) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "# UTC, half-open. purge_seconds={}. train_end = test_start - purge_seconds.\n",
        purge_seconds
    ));
    s.push_str(WINDOWS_HEADER);
    s.push('\n');
    for r in default_windows(purge_seconds) {
        s.push_str(&format!(
            "{},{},{},{},{},{},{}\n",
            r.scheme,
            r.window.label,
            iso(r.window.train_start),
            iso(r.window.train_end),
            iso(r.validation_start),
            iso(r.window.test_start),
            iso(r.window.test_end)
        ));
    }
    s
}

/// True when the text is a protocol windows table (it names a `scheme` column).
pub fn is_windows_table(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .map_or(false, |l| l.starts_with("scheme,"))
}

/// Parse a protocol windows table, keeping the rows of `scheme` (all when `None`).
pub fn parse_windows_csv(text: &str, scheme: Option<&str>) -> Result<Vec<WindowRow>, String> {
    let mut rows = Vec::new();
    let mut header = false;
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !header {
            if line != WINDOWS_HEADER {
                return Err(format!("windows table header must be '{}'", WINDOWS_HEADER));
            }
            header = true;
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() != 7 {
            return Err(format!("line {}: expected 7 fields, found {}", n + 1, f.len()));
        }
        let t = |i: usize| {
            crate::data::parse_time(f[i]).ok_or_else(|| format!("line {}: cannot parse time '{}'", n + 1, f[i]))
        };
        if !SCHEMES.contains(&f[0]) {
            return Err(format!("line {}: unknown scheme '{}'", n + 1, f[0]));
        }
        if scheme.map_or(true, |s| s == f[0]) {
            rows.push(WindowRow {
                scheme: f[0].to_string(),
                window: Window {
                    label: f[1].to_string(),
                    train_start: t(2)?,
                    train_end: t(3)?,
                    test_start: t(5)?,
                    test_end: t(6)?,
                },
                validation_start: t(4)?,
            });
        }
    }
    if rows.is_empty() {
        return Err(match scheme {
            Some(s) => format!("windows table has no rows for scheme '{}'", s),
            None => "windows table has no rows".to_string(),
        });
    }
    Ok(rows)
}

/// The declared locked test as epoch seconds.
pub fn default_locked_range() -> (f64, f64) {
    (month_start(2025, 7, 0), month_start(2026, 7, 0))
}

// ---------------------------------------------------------------------------
// Ledger
// ---------------------------------------------------------------------------

fn esc(s: &str) -> String {
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

/// Read a string field written by [`esc`]. Keys are unique per line and the
/// expression, the only free text, is written after every key read back.
fn field_str(line: &str, key: &str) -> Option<String> {
    let pat = format!("\"{}\":\"", key);
    let start = line.find(&pat)? + pat.len();
    let mut out = String::new();
    let mut chars = line[start..].chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => match chars.next()? {
                'n' => out.push('\n'),
                'r' => out.push('\r'),
                't' => out.push('\t'),
                'u' => {
                    let hex: String = chars.by_ref().take(4).collect();
                    out.push(char::from_u32(u32::from_str_radix(&hex, 16).ok()?)?);
                }
                other => out.push(other),
            },
            c => out.push(c),
        }
    }
    None
}

fn field_num(line: &str, key: &str) -> Option<f64> {
    let pat = format!("\"{}\":", key);
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].trim().parse().ok()
}

fn field_bool(line: &str, key: &str) -> bool {
    line.contains(&format!("\"{}\":true", key))
}

/// Terms of a final seal: the locked test is read at most `max_reads` times.
#[derive(Clone, Debug, PartialEq)]
pub struct FinalSeal {
    pub max_reads: u64,
    pub report_candidates: bool,
}

/// The locked test range a ledger line was made against: start and end in
/// epoch seconds (half-open) and where the range came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockedRange {
    pub start: i64,
    pub end: i64,
    pub source: String,
}

impl LockedRange {
    pub fn new(range: &crate::lockbox::Lockbox, source: &str) -> Self {
        LockedRange { start: range.start, end: range.end, source: source.to_string() }
    }

    /// `"locked_test":{...}` as stored on ledger lines and sealed items.
    pub fn json(&self) -> String {
        format!(
            "\"locked_test\":{{\"start\":{},\"end\":{},\"source\":{}}}",
            self.start,
            self.end,
            esc(&self.source)
        )
    }

    /// The `locked_test` object of a ledger line or sealed item, if it has one.
    fn parse(text: &str) -> Option<LockedRange> {
        let obj = crate::manifest::object(text, "locked_test")?;
        Some(LockedRange {
            start: crate::lockbox::int_field(obj, "start")?,
            end: crate::lockbox::int_field(obj, "end")?,
            source: field_str(obj, "source").unwrap_or_default(),
        })
    }

    fn describe(&self) -> String {
        format!("{}..{} ({})", crate::lockbox::date(self.start), crate::lockbox::date(self.end), self.source)
    }
}

/// The range a locked read resolves now, checked against the range a sealed
/// candidate was developed against.
#[derive(Clone, Copy, Debug)]
pub struct RangeBinding<'a> {
    pub current: &'a LockedRange,
    /// Read a candidate of an old ledger that recorded no range.
    pub allow_unbound: bool,
}

impl RangeBinding<'_> {
    fn check(&self, id: &str, developed: Option<&LockedRange>) -> Result<(), String> {
        match developed {
            Some(d) if (d.start, d.end) == (self.current.start, self.current.end) => Ok(()),
            Some(d) => Err(format!(
                "candidate {} was developed against the locked test {}; this read resolves {}",
                id,
                d.describe(),
                self.current.describe()
            )),
            None if self.allow_unbound => Ok(()),
            None => Err(format!(
                "candidate {} has no recorded locked test (the ledger predates range binding); \
                 pass --allow-unbound-ledger to read it unbound",
                id
            )),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Candidate(Candidate),
    Seal { sealed_sha256: String, final_seal: Option<FinalSeal> },
    TestRead { id: String, report_sha256: String, read_count: u64, locked_test: Option<LockedRange> },
    BaselineRead { data_sha256: String, report_sha256: String, locked_test: Option<LockedRange> },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub spec_sha256: String,
    pub data_sha256: String,
    pub build_commit: String,
    pub report_sha256: String,
    pub expression: String,
    /// The locked test the candidate was developed against.
    pub locked_test: Option<LockedRange>,
}

impl Candidate {
    /// The content address: the hash of everything that defines the run.
    pub fn address(spec_sha256: &str, data_sha256: &str, build_commit: &str) -> String {
        sha256_hex(format!("{}\n{}\n{}", spec_sha256, data_sha256, build_commit).as_bytes())
    }
}

pub struct Ledger {
    path: std::path::PathBuf,
}

impl Ledger {
    pub fn new(path: impl Into<std::path::PathBuf>) -> Self {
        Ledger { path: path.into() }
    }

    fn lines(&self) -> Result<Vec<String>, String> {
        match std::fs::read_to_string(&self.path) {
            Ok(t) => Ok(t.lines().filter(|l| !l.trim().is_empty()).map(str::to_string).collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("{}: {}", self.path.display(), e)),
        }
    }

    /// Check that every line names the hash of the line before it.
    pub fn verify_chain(&self) -> Result<usize, String> {
        let lines = self.lines()?;
        let mut prev = "0".repeat(64);
        for (i, line) in lines.iter().enumerate() {
            let got = field_str(line, "prev").unwrap_or_default();
            if got != prev {
                return Err(format!(
                    "{}: line {} breaks the hash chain; the ledger was edited",
                    self.path.display(),
                    i + 1
                ));
            }
            prev = sha256_hex(line.as_bytes());
        }
        Ok(lines.len())
    }

    pub fn events(&self) -> Result<Vec<Event>, String> {
        let mut out = Vec::new();
        for line in self.lines()? {
            let kind = field_str(&line, "event").unwrap_or_default();
            let get = |k: &str| field_str(&line, k).unwrap_or_default();
            out.push(match kind.as_str() {
                "candidate" => Event::Candidate(Candidate {
                    id: get("id"),
                    spec_sha256: get("spec_sha256"),
                    data_sha256: get("data_sha256"),
                    build_commit: get("build_commit"),
                    report_sha256: get("report_sha256"),
                    expression: get("expression"),
                    locked_test: LockedRange::parse(&line),
                }),
                "seal" => Event::Seal {
                    sealed_sha256: get("sealed_sha256"),
                    final_seal: if field_bool(&line, "final") {
                        Some(FinalSeal {
                            max_reads: field_num(&line, "max_reads").unwrap_or(1.0) as u64,
                            report_candidates: field_bool(&line, "report_candidates"),
                        })
                    } else {
                        None
                    },
                },
                "test_read" => Event::TestRead {
                    id: get("id"),
                    report_sha256: get("report_sha256"),
                    read_count: field_num(&line, "read_count").unwrap_or(0.0) as u64,
                    locked_test: LockedRange::parse(&line),
                },
                "baseline_read" => Event::BaselineRead {
                    data_sha256: get("data_sha256"),
                    report_sha256: get("report_sha256"),
                    locked_test: LockedRange::parse(&line),
                },
                other => return Err(format!("unknown ledger event '{}'", other)),
            });
        }
        Ok(out)
    }

    fn append(&self, body: &str) -> Result<(), String> {
        self.verify_chain()?;
        let lines = self.lines()?;
        let prev = lines
            .last()
            .map_or("0".repeat(64), |l| sha256_hex(l.as_bytes()));
        let line = format!("{{\"prev\":\"{}\",{}}}", prev, body);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| format!("{}: {}", self.path.display(), e))?;
        writeln!(f, "{}", line).map_err(|e| format!("{}: {}", self.path.display(), e))
    }

    /// Record a development candidate. Returns its id.
    pub fn add_candidate(&self, c: &Candidate) -> Result<String, String> {
        self.refuse_if_final()?;
        self.append(&format!(
            "\"event\":\"candidate\",\"id\":\"{}\",\"spec_sha256\":\"{}\",\"data_sha256\":\"{}\",\
             \"build_commit\":\"{}\",\"report_sha256\":\"{}\",\"test_read\":false,{}\"expression\":{}",
            c.id,
            c.spec_sha256,
            c.data_sha256,
            c.build_commit,
            c.report_sha256,
            c.locked_test.as_ref().map(|l| format!("{},", l.json())).unwrap_or_default(),
            esc(&c.expression)
        ))?;
        Ok(c.id.clone())
    }

    pub fn read_count(&self) -> Result<u64, String> {
        Ok(self
            .events()?
            .iter()
            .filter(|e| matches!(e, Event::TestRead { .. }))
            .count() as u64)
    }

    /// Candidates that have not been read on the locked test.
    pub fn unread_candidates(&self) -> Result<Vec<Candidate>, String> {
        let events = self.events()?;
        let read: Vec<&str> = events
            .iter()
            .filter_map(|e| match e {
                Event::TestRead { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let mut out: Vec<Candidate> = Vec::new();
        for e in &events {
            if let Event::Candidate(c) = e {
                if !read.contains(&c.id.as_str()) && !out.iter().any(|o| o.id == c.id) {
                    out.push(c.clone());
                }
            }
        }
        Ok(out)
    }

    /// Write `locked_candidates.json` for `ids` (every unread candidate when
    /// empty), then record its hash in the ledger. Returns the hash.
    pub fn seal(&self, out: &std::path::Path, ids: &[String]) -> Result<String, String> {
        self.refuse_if_final()?;
        let hash = self.write_sealed(out, ids)?;
        self.append(&format!("\"event\":\"seal\",\"sealed_sha256\":\"{}\",\"test_read\":false", hash))?;
        Ok(hash)
    }

    /// Final seal: as [`Ledger::seal`], and the ledger takes no new candidate or
    /// seal afterwards. The locked test is read at most `max_reads` times.
    pub fn seal_final(
        &self,
        out: &std::path::Path,
        ids: &[String],
        max_reads: u64,
        report_candidates: bool,
    ) -> Result<String, String> {
        if max_reads == 0 {
            return Err("--max-reads must be at least 1".to_string());
        }
        self.verify_chain()?;
        self.refuse_if_final()?;
        let events = self.events()?;
        if events.iter().any(|e| matches!(e, Event::Seal { .. })) {
            return Err("the ledger already has a seal; a final seal needs a ledger with none".to_string());
        }
        if events.iter().any(|e| matches!(e, Event::TestRead { .. } | Event::BaselineRead { .. })) {
            return Err("the ledger already has a locked read; a final seal needs a ledger with none".to_string());
        }
        let hash = self.write_sealed(out, ids)?;
        self.append(&format!(
            "\"event\":\"seal\",\"sealed_sha256\":\"{}\",\"final\":true,\"max_reads\":{},\
             \"report_candidates\":{},\"test_read\":false",
            hash, max_reads, report_candidates
        ))?;
        Ok(hash)
    }

    /// The final seal of the ledger: its sealed-list hash and terms.
    pub fn final_seal(&self) -> Result<Option<(String, FinalSeal)>, String> {
        Ok(self.events()?.into_iter().find_map(|e| match e {
            Event::Seal { sealed_sha256, final_seal: Some(f) } => Some((sealed_sha256, f)),
            _ => None,
        }))
    }

    pub fn is_final(&self) -> Result<bool, String> {
        Ok(self.final_seal()?.is_some())
    }

    pub fn refuse_if_final(&self) -> Result<(), String> {
        match self.final_seal()? {
            Some((hash, _)) => Err(format!(
                "the ledger is final (sealed {}); development runs need a new protocol, not this ledger",
                hash
            )),
            None => Ok(()),
        }
    }

    fn write_sealed(&self, out: &std::path::Path, ids: &[String]) -> Result<String, String> {
        let unread = self.unread_candidates()?;
        let chosen: Vec<&Candidate> = if ids.is_empty() {
            unread.iter().collect()
        } else {
            let mut v = Vec::new();
            for id in ids {
                v.push(
                    unread
                        .iter()
                        .find(|c| &c.id == id)
                        .ok_or_else(|| format!("candidate {} is not an unread ledger candidate", id))?,
                );
            }
            v
        };
        if chosen.is_empty() {
            return Err("nothing to seal: the ledger has no unread candidate".to_string());
        }
        let head = self
            .lines()?
            .last()
            .map_or("0".repeat(64), |l| sha256_hex(l.as_bytes()));
        let items: Vec<String> = chosen
            .iter()
            .map(|c| {
                format!(
                    "{{\"id\":\"{}\",\"spec_sha256\":\"{}\",\"data_sha256\":\"{}\",\
                     \"build_commit\":\"{}\",\"report_sha256\":\"{}\",{}\"expression\":{}}}",
                    c.id,
                    c.spec_sha256,
                    c.data_sha256,
                    c.build_commit,
                    c.report_sha256,
                    c.locked_test.as_ref().map(|l| format!("{},", l.json())).unwrap_or_default(),
                    esc(&c.expression)
                )
            })
            .collect();
        let text = format!(
            "{{\"version\":1,\"ledger_head\":\"{}\",\"candidates\":[{}]}}\n",
            head,
            items.join(",")
        );
        std::fs::write(out, &text).map_err(|e| format!("{}: {}", out.display(), e))?;
        Ok(sha256_hex(text.as_bytes()))
    }

    /// Check that `id` is in a sealed list that the ledger recorded, and that the
    /// data and build hashes match the sealed entry. Does not look at earlier reads.
    pub fn authorize_sealed(
        &self,
        sealed: &std::path::Path,
        id: &str,
        data_sha256: &str,
        build_commit: &str,
        bind: Option<RangeBinding>,
    ) -> Result<Candidate, String> {
        self.verify_chain()?;
        let bytes = std::fs::read(sealed).map_err(|e| format!("{}: {}", sealed.display(), e))?;
        let hash = sha256_hex(&bytes);
        let events = self.events()?;
        let sealed_ok = events
            .iter()
            .any(|e| matches!(e, Event::Seal { sealed_sha256, .. } if *sealed_sha256 == hash));
        if !sealed_ok {
            return Err(format!(
                "{} (sha256 {}) was not sealed in the ledger before this read",
                sealed.display(),
                hash
            ));
        }
        let text = String::from_utf8_lossy(&bytes).to_string();
        let entry = text
            .split("{\"id\":\"")
            .skip(1)
            .map(|chunk| format!("{{\"id\":\"{}", chunk))
            .find(|chunk| field_str(chunk, "id").as_deref() == Some(id))
            .ok_or_else(|| format!("candidate {} is not in the sealed list; the runner refuses it", id))?;
        let get = |k: &str| field_str(&entry, k).unwrap_or_default();
        let candidate = Candidate {
            id: id.to_string(),
            spec_sha256: get("spec_sha256"),
            data_sha256: get("data_sha256"),
            build_commit: get("build_commit"),
            report_sha256: get("report_sha256"),
            expression: get("expression"),
            locked_test: LockedRange::parse(&entry),
        };
        if candidate.data_sha256 != data_sha256 || candidate.build_commit != build_commit {
            return Err(format!(
                "candidate {}: data or build hash differs from the sealed entry \
                 (data {} vs {}, build {} vs {}); the run is void",
                id, data_sha256, candidate.data_sha256, build_commit, candidate.build_commit
            ));
        }
        if let Some(b) = bind {
            b.check(id, candidate.locked_test.as_ref())?;
        }
        Ok(candidate)
    }

    /// Check that a locked read of `id` is allowed: sealed, matching hashes and
    /// not read before. Returns the sealed entry.
    pub fn authorize_read(
        &self,
        sealed: &std::path::Path,
        id: &str,
        data_sha256: &str,
        build_commit: &str,
        bind: Option<RangeBinding>,
    ) -> Result<Candidate, String> {
        let candidate = self.authorize_sealed(sealed, id, data_sha256, build_commit, bind)?;
        if self.read_report_sha(id).is_some() {
            return Err(format!(
                "candidate {} already has test_read:true; the locked test is read once",
                id
            ));
        }
        if let Some((_, f)) = self.final_seal()? {
            let done = self.read_count()?;
            if done >= f.max_reads {
                return Err(format!(
                    "the final seal allows {} locked read(s); {} done; the locked test is read once",
                    f.max_reads, done
                ));
            }
        }
        Ok(candidate)
    }

    /// Report hash recorded by the read of `id`, if there was one.
    pub fn read_report_sha(&self, id: &str) -> Option<String> {
        self.events().ok()?.into_iter().find_map(|e| match e {
            Event::TestRead { id: read, report_sha256, .. } if read == id => Some(report_sha256),
            _ => None,
        })
    }

    /// Append `test_read:true` for `id`. Returns the new total read count.
    pub fn record_read(&self, id: &str, report_sha256: &str, range: Option<&LockedRange>) -> Result<u64, String> {
        let count = self.read_count()? + 1;
        self.append(&format!(
            "\"event\":\"test_read\",\"id\":\"{}\",\"report_sha256\":\"{}\",\"read_count\":{},{}\"test_read\":true",
            id,
            report_sha256,
            count,
            range.map(|l| format!("{},", l.json())).unwrap_or_default()
        ))?;
        Ok(count)
    }

    /// Check that a baseline ladder may read the locked test: the sealed list
    /// was recorded in the ledger and no baseline read exists yet.
    pub fn authorize_baseline_read(&self, sealed: &std::path::Path, bind: Option<RangeBinding>) -> Result<(), String> {
        self.verify_chain()?;
        let bytes = std::fs::read(sealed).map_err(|e| format!("{}: {}", sealed.display(), e))?;
        let hash = sha256_hex(&bytes);
        let events = self.events()?;
        if !events
            .iter()
            .any(|e| matches!(e, Event::Seal { sealed_sha256, .. } if *sealed_sha256 == hash))
        {
            return Err(format!(
                "{} (sha256 {}) was not sealed in the ledger before this read",
                sealed.display(),
                hash
            ));
        }
        if events.iter().any(|e| matches!(e, Event::BaselineRead { .. })) {
            return Err("the ledger already has a baseline_read; the locked test is read once".to_string());
        }
        if let Some(b) = bind {
            let text = String::from_utf8_lossy(&bytes).to_string();
            for chunk in text.split("{\"id\":\"").skip(1) {
                let id = chunk.split('"').next().unwrap_or_default();
                b.check(id, LockedRange::parse(chunk).as_ref())?;
            }
        }
        Ok(())
    }

    /// Append the `baseline_read` event of a ladder run on the locked test.
    pub fn record_baseline_read(
        &self,
        data_sha256: &str,
        report_sha256: &str,
        range: Option<&LockedRange>,
    ) -> Result<(), String> {
        self.append(&format!(
            "\"event\":\"baseline_read\",\"data_sha256\":\"{}\",\"report_sha256\":\"{}\",{}\"test_read\":true",
            data_sha256,
            report_sha256,
            range.map(|l| format!("{},", l.json())).unwrap_or_default()
        ))
    }
}

/// JSON for the report's `protocol` object.
pub struct LockboxFacts {
    pub start: f64,
    pub end: f64,
    pub rows: usize,
    pub first_ts: Option<f64>,
    pub last_ts: Option<f64>,
    pub read_count: u64,
}

impl LockboxFacts {
    pub fn json(&self) -> String {
        let date = |t: Option<f64>| t.map_or("null".to_string(), |v| format!("\"{}\"", iso(v)));
        let num = |t: Option<f64>| t.map_or("null".to_string(), |v| format!("{}", v));
        format!(
            "{{\"start\":{},\"end\":{},\"start_date\":\"{}\",\"end_date\":\"{}\",\"rows\":{},\
             \"first_ts\":{},\"last_ts\":{},\"first_date\":{},\"last_date\":{},\"read_count\":{}}}",
            self.start,
            self.end,
            iso(self.start),
            iso(self.end),
            self.rows,
            num(self.first_ts),
            num(self.last_ts),
            date(self.first_ts),
            date(self.last_ts),
            self.read_count
        )
    }
}

/// Canonical text of the procedure a candidate stands for: everything that
/// changes the report except the windows and the output options. Its SHA-256
/// is the candidate's spec hash, so a development run and the locked read of
/// the same procedure share one id.
pub fn procedure_spec(spec: &crate::walkforward::EvalSpec, o: &crate::walkforward::ProtocolOptions) -> String {
    let mut s = String::new();
    let exprs: Vec<&str> = if spec.candidates.is_empty() {
        vec![spec.expression.as_str()]
    } else {
        spec.candidates.iter().map(String::as_str).collect()
    };
    for e in exprs {
        s.push_str(&format!("expr={}\n", e));
    }
    s.push_str(&format!(
        "target={}\ntime={}\ngroup={}\nvars={}\nhorizon={}\nembargo={}\ncost_bps={}\nquantile={}\n\
         fixed={:?}\nvalidation_frac={}\npolicy={}\nband={}\nmin_hold={}\nentry_offset={}\ndelay={}\n\
         pool_reset_flat={}\ncontrol_seed={}\nperiods_per_year={:?}\nband_select={}\n",
        spec.target,
        spec.time,
        spec.group.as_deref().unwrap_or(""),
        spec.vars.join(","),
        spec.horizon,
        spec.embargo_seconds(),
        spec.cost_bps,
        spec.threshold_quantile,
        spec.threshold_fixed,
        spec.validation_frac,
        spec.policy.as_str(),
        spec.band_width,
        spec.min_hold,
        o.entry_offset,
        o.delay,
        spec.pool_reset_flat,
        spec.control_seed,
        spec.periods_per_year,
        spec.band_select
    ));
    s
}

fn protocol_usage() -> &'static str {
    "rcsr protocol <windows|seal|verify|candidates|selectors> ...

  windows [--purge-seconds N] [--out FILE]
      Write the UTC windows table (default purge 219600 s: 1 h entry + 60 h label).
  seal --ledger FILE --out locked_candidates.json [--id HEX ...]
      Write the sealed list (every unread candidate by default), then record its
      SHA-256 in the ledger. Prints the hash. Do this before any locked read.
  seal --final [--max-reads N] [--report-candidates] --ledger FILE --out FILE [--id HEX ...]
      Final seal for the single locked read. The ledger takes no new candidate
      and no new seal afterwards, allows at most N (default 1) locked reads and
      one baseline read, and a read of a multi-candidate procedure reports the
      validation-selected candidate only (--report-candidates keeps the full
      report and records the choice). Needs a ledger with no seal and no read.
      The first locked read writes <data>.locked_reads.jsonl beside the store;
      a read through any other ledger is refused.
  selectors --windows FILE --label LABEL --time-col COL [--scheme S]
      Print `--train-select` and `--test-select` for one window. The purge is in
      the train range (train_end = test_start - purge), in time and for every
      symbol; `--gap` does not apply to explicit selectors.
  verify --ledger FILE [--data PATH] [--sealed FILE] [--expect-reads N]
      Check the ledger hash chain and print the line and read counts. --data
      also checks the read marker beside the store against the ledger.
      --expect-reads N (needs --data) exits 1 unless the ledger has N locked
      reads, exactly one seal line that is final, and a consistent marker;
      with --sealed it also checks that the file is the final seal's list and
      that every read id is in it.
  candidates --ledger FILE
      List the candidates without a locked read.
"
}

/// `rcsr protocol ...`.
pub fn run(args: &[String]) -> ! {
    let fail = |msg: String, code: i32| -> ! {
        eprintln!("error: {}", msg);
        std::process::exit(code)
    };
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", protocol_usage());
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }
    let mut purge = DEFAULT_PURGE_SECONDS;
    let mut out: Option<String> = None;
    let mut ledger: Option<String> = None;
    let mut ids: Vec<String> = Vec::new();
    let mut windows_file: Option<String> = None;
    let mut scheme: Option<String> = None;
    let mut label: Option<String> = None;
    let mut time_col: Option<String> = None;
    let mut final_seal = false;
    let mut report_candidates = false;
    let mut max_reads: Option<u64> = None;
    let mut expect_reads: Option<u64> = None;
    let mut data: Option<String> = None;
    let mut sealed: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        let value = |i: usize| -> String {
            args.get(i + 1)
                .cloned()
                .unwrap_or_else(|| fail(format!("{} requires a value", args[i]), 2))
        };
        match args[i].as_str() {
            "--purge-seconds" => {
                purge = value(i)
                    .parse::<f64>()
                    .ok()
                    .filter(|v| v.is_finite() && *v >= 0.0)
                    .unwrap_or_else(|| fail("--purge-seconds needs a non-negative number".to_string(), 2));
                i += 1;
            }
            "--out" => {
                out = Some(value(i));
                i += 1;
            }
            "--ledger" => {
                ledger = Some(value(i));
                i += 1;
            }
            "--id" => {
                ids.push(value(i));
                i += 1;
            }
            "--windows" => {
                windows_file = Some(value(i));
                i += 1;
            }
            "--scheme" => {
                scheme = Some(value(i));
                i += 1;
            }
            "--label" => {
                label = Some(value(i));
                i += 1;
            }
            "--time-col" => {
                time_col = Some(value(i));
                i += 1;
            }
            "--final" => final_seal = true,
            "--report-candidates" => report_candidates = true,
            "--max-reads" => {
                max_reads = Some(
                    value(i)
                        .parse::<u64>()
                        .ok()
                        .filter(|v| *v >= 1)
                        .unwrap_or_else(|| fail("--max-reads needs an integer of at least 1".to_string(), 2)),
                );
                i += 1;
            }
            "--expect-reads" => {
                expect_reads = Some(
                    value(i)
                        .parse::<u64>()
                        .unwrap_or_else(|_| fail("--expect-reads needs a non-negative integer".to_string(), 2)),
                );
                i += 1;
            }
            "--data" => {
                data = Some(value(i));
                i += 1;
            }
            "--sealed" => {
                sealed = Some(value(i));
                i += 1;
            }
            other => fail(format!("unknown protocol option '{}'\n\n{}", other, protocol_usage()), 2),
        }
        i += 1;
    }
    let need_ledger = || Ledger::new(ledger.clone().unwrap_or_else(|| fail("this command needs --ledger".to_string(), 2)));
    match args[0].as_str() {
        "windows" => {
            let text = windows_csv(purge);
            match out {
                Some(p) => std::fs::write(&p, text).unwrap_or_else(|e| fail(format!("{}: {}", p, e), 1)),
                None => print!("{}", text),
            }
        }
        "seal" => {
            let l = need_ledger();
            let path = out.unwrap_or_else(|| fail("seal needs --out".to_string(), 2));
            if !final_seal && (max_reads.is_some() || report_candidates) {
                fail("--max-reads and --report-candidates need --final".to_string(), 2);
            }
            let sealed_result = if final_seal {
                l.seal_final(std::path::Path::new(&path), &ids, max_reads.unwrap_or(1), report_candidates)
            } else {
                l.seal(std::path::Path::new(&path), &ids)
            };
            match sealed_result {
                Ok(h) => println!("sealed_sha256 {}", h),
                Err(e) => fail(e, 1),
            }
        }
        "verify" => {
            let l = need_ledger();
            match (l.verify_chain(), l.read_count()) {
                (Ok(n), Ok(r)) => println!("ledger ok: {} line(s), {} locked read(s)", n, r),
                (Err(e), _) | (_, Err(e)) => fail(e, 1),
            }
            if expect_reads.is_some() && data.is_none() {
                fail("--expect-reads needs --data, the store the marker lives with".to_string(), 2);
            }
            match (&data, expect_reads) {
                (Some(d), Some(n)) => match l.verify_expected(n, d, sealed.as_deref().map(std::path::Path::new)) {
                    Ok(lines) => lines.iter().for_each(|s| println!("{}", s)),
                    Err(e) => fail(e, 1),
                },
                (Some(d), None) => match l.verify_read_marker(d) {
                    Ok(s) => println!("{}", s),
                    Err(e) => fail(e, 1),
                },
                _ => {}
            }
        }
        "selectors" => {
            let file = windows_file.unwrap_or_else(|| fail("selectors needs --windows".to_string(), 2));
            let label = label.unwrap_or_else(|| fail("selectors needs --label".to_string(), 2));
            let col = time_col.unwrap_or_else(|| fail("selectors needs --time-col".to_string(), 2));
            let text = std::fs::read_to_string(&file).unwrap_or_else(|e| fail(format!("{}: {}", file, e), 1));
            let rows = parse_windows_csv(&text, scheme.as_deref()).unwrap_or_else(|e| fail(e, 1));
            let row = rows
                .iter()
                .find(|r| r.window.label == label)
                .unwrap_or_else(|| fail(format!("no window labelled '{}'", label), 1));
            let (train, test) = selector_exprs(row, &col);
            println!("--train-select '{}' --test-select '{}'", train, test);
        }
        "candidates" => {
            let l = need_ledger();
            match l.unread_candidates() {
                Ok(v) => {
                    for c in v {
                        println!("{} {}", c.id, c.expression);
                    }
                }
                Err(e) => fail(e, 1),
            }
        }
        other => fail(format!("unknown protocol command '{}'\n\n{}", other, protocol_usage()), 2),
    }
    std::process::exit(0)
}

/// `--train-select` and `--test-select` expressions for one window, in epoch
/// seconds. The training range already ends one purge before the test start,
/// and the same time bounds apply to every symbol of a ragged panel.
pub fn selector_exprs(row: &WindowRow, time_col: &str) -> (String, String) {
    let w = &row.window;
    (
        format!("{} >= {} and {} < {}", time_col, w.train_start, time_col, w.train_end),
        format!("{} >= {} and {} < {}", time_col, w.test_start, time_col, w.test_end),
    )
}

/// One line of `<data>.locked_reads.jsonl`, the read marker kept with the store.
#[derive(Clone, Debug, PartialEq)]
pub struct MarkerLine {
    pub ledger: String,
    pub ledger_head: String,
    pub event: String,
    pub report_sha256: String,
}

pub fn marker_path(data_path: &str) -> String {
    format!("{}.locked_reads.jsonl", data_path)
}

fn marker_lines(data_path: &str) -> Result<Vec<MarkerLine>, String> {
    let path = marker_path(data_path);
    let text = match std::fs::read_to_string(&path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {}", path, e)),
    };
    let mut out = Vec::new();
    for line in text.lines().filter(|l| !l.trim().is_empty()) {
        let get = |k: &str| field_str(line, k).unwrap_or_default();
        let m = MarkerLine {
            ledger: get("ledger"),
            ledger_head: get("ledger_head"),
            event: get("event"),
            report_sha256: get("report_sha256"),
        };
        if !matches!(m.event.as_str(), "test_read" | "baseline_read") || m.ledger.is_empty() {
            return Err(format!("{}: malformed read marker line", path));
        }
        out.push(m);
    }
    Ok(out)
}

impl Ledger {
    /// Hash of the last line, or 64 zeros for an empty ledger.
    pub fn head(&self) -> Result<String, String> {
        Ok(self.lines()?.last().map_or("0".repeat(64), |l| sha256_hex(l.as_bytes())))
    }

    fn absolute_path(&self) -> Result<String, String> {
        let p = if self.path.exists() {
            std::fs::canonicalize(&self.path)
        } else {
            std::path::absolute(&self.path)
        };
        p.map(|p| p.display().to_string())
            .map_err(|e| format!("{}: {}", self.path.display(), e))
    }

    fn chain_heads(&self) -> Result<Vec<String>, String> {
        let mut out = vec!["0".repeat(64)];
        out.extend(self.lines()?.iter().map(|l| sha256_hex(l.as_bytes())));
        Ok(out)
    }

    /// Refuse a locked read of the store at `data_path` through a ledger that is
    /// not the one that made the first read: the marker names that ledger and a
    /// head of its chain.
    pub fn check_read_marker(&self, data_path: &str) -> Result<(), String> {
        let lines = marker_lines(data_path)?;
        let Some(first) = lines.first() else {
            return Ok(());
        };
        let here = self.absolute_path()?;
        if lines.iter().any(|m| m.ledger != first.ledger) {
            return Err(format!("{}: the read marker names more than one ledger", marker_path(data_path)));
        }
        if first.ledger != here {
            return Err(format!("{} was already read through ledger {}", data_path, first.ledger));
        }
        let heads = self.chain_heads()?;
        if !heads.contains(&first.ledger_head) {
            return Err(format!(
                "{} was already read through ledger {}, and this ledger's chain does not contain the head {} of that read",
                data_path, first.ledger, first.ledger_head
            ));
        }
        Ok(())
    }

    /// Append the marker line of a locked read, before the ledger records it.
    /// `ledger_head` is the ledger head the read extends. A retry after a crash
    /// between marker and ledger finds the same head and writes nothing.
    pub fn write_read_marker(&self, data_path: &str, event: &str, report_sha256: &str) -> Result<(), String> {
        self.check_read_marker(data_path)?;
        let head = self.head()?;
        let lines = marker_lines(data_path)?;
        if lines.last().map_or(false, |m| m.ledger_head == head && m.event == event) {
            return Ok(());
        }
        let path = marker_path(data_path);
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("{}: {}", path, e))?;
        writeln!(
            f,
            "{{\"ledger\":{},\"ledger_head\":\"{}\",\"event\":\"{}\",\"report_sha256\":\"{}\"}}",
            esc(&self.absolute_path()?),
            head,
            event,
            report_sha256
        )
        .map_err(|e| format!("{}: {}", path, e))
    }

    /// Compare the marker with the ledger's reads. Returns one status line.
    pub fn verify_read_marker(&self, data_path: &str) -> Result<String, String> {
        self.check_read_marker(data_path)?;
        let marker = marker_lines(data_path)?;
        let mut want: Vec<(String, String)> = Vec::new();
        for e in self.events()? {
            match e {
                Event::TestRead { report_sha256, .. } => want.push(("test_read".to_string(), report_sha256)),
                Event::BaselineRead { report_sha256, .. } => want.push(("baseline_read".to_string(), report_sha256)),
                _ => {}
            }
        }
        let have: Vec<(String, String)> = marker.iter().map(|m| (m.event.clone(), m.report_sha256.clone())).collect();
        for w in &want {
            if !have.contains(w) {
                return Err(format!(
                    "the ledger records a {} (report {}) without a line in {}",
                    w.0,
                    w.1,
                    marker_path(data_path)
                ));
            }
        }
        for h in &have {
            if !want.contains(h) {
                return Err(format!(
                    "{} has a {} (report {}) that the ledger does not record",
                    marker_path(data_path),
                    h.0,
                    h.1
                ));
            }
        }
        Ok(format!("marker {}: {} read line(s), consistent with the ledger", marker_path(data_path), marker.len()))
    }
}

fn sealed_ids(text: &str) -> Vec<String> {
    text.split("{\"id\":\"")
        .skip(1)
        .filter_map(|chunk| field_str(&format!("{{\"id\":\"{}", chunk), "id"))
        .collect()
}

impl Ledger {
    /// Gate for the single locked read: `expect` reads, one seal line that is
    /// final, a marker that matches the ledger and, with `sealed`, a read id
    /// set inside the final sealed list. Returns the status lines.
    pub fn verify_expected(
        &self,
        expect: u64,
        data_path: &str,
        sealed: Option<&std::path::Path>,
    ) -> Result<Vec<String>, String> {
        self.verify_chain()?;
        let mut out = Vec::new();
        let events = self.events()?;
        let reads = self.read_count()?;
        if reads != expect {
            return Err(format!("expected {} locked read(s), the ledger has {}", expect, reads));
        }
        let seals: Vec<&Event> = events.iter().filter(|e| matches!(e, Event::Seal { .. })).collect();
        let Some((final_hash, terms)) = self.final_seal()? else {
            return Err("the ledger has no final seal".to_string());
        };
        if seals.len() != 1 {
            return Err(format!("expected exactly one seal line, the ledger has {}", seals.len()));
        }
        if reads > terms.max_reads {
            return Err(format!("the ledger has {} locked read(s), the final seal allows {}", reads, terms.max_reads));
        }
        out.push(format!("final seal {} (max_reads {})", final_hash, terms.max_reads));
        out.push(self.verify_read_marker(data_path)?);
        if let Some(path) = sealed {
            let bytes = std::fs::read(path).map_err(|e| format!("{}: {}", path.display(), e))?;
            if sha256_hex(&bytes) != final_hash {
                return Err(format!("{} is not the final sealed list", path.display()));
            }
            let ids = sealed_ids(&String::from_utf8_lossy(&bytes));
            for e in &events {
                if let Event::TestRead { id, .. } = e {
                    if !ids.contains(id) {
                        return Err(format!("read candidate {} is not in the final sealed list", id));
                    }
                }
            }
            out.push("every read candidate is in the final sealed list".to_string());
        } else {
            out.push("sealed list: not checked (pass --sealed)".to_string());
        }
        Ok(out)
    }
}
