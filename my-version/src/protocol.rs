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

#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    Candidate(Candidate),
    Seal { sealed_sha256: String },
    TestRead { id: String, report_sha256: String, read_count: u64 },
    BaselineRead { data_sha256: String, report_sha256: String },
}

#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub id: String,
    pub spec_sha256: String,
    pub data_sha256: String,
    pub build_commit: String,
    pub report_sha256: String,
    pub expression: String,
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
                }),
                "seal" => Event::Seal { sealed_sha256: get("sealed_sha256") },
                "test_read" => Event::TestRead {
                    id: get("id"),
                    report_sha256: get("report_sha256"),
                    read_count: field_num(&line, "read_count").unwrap_or(0.0) as u64,
                },
                "baseline_read" => Event::BaselineRead {
                    data_sha256: get("data_sha256"),
                    report_sha256: get("report_sha256"),
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
        self.append(&format!(
            "\"event\":\"candidate\",\"id\":\"{}\",\"spec_sha256\":\"{}\",\"data_sha256\":\"{}\",\
             \"build_commit\":\"{}\",\"report_sha256\":\"{}\",\"test_read\":false,\"expression\":{}",
            c.id,
            c.spec_sha256,
            c.data_sha256,
            c.build_commit,
            c.report_sha256,
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
                     \"build_commit\":\"{}\",\"report_sha256\":\"{}\",\"expression\":{}}}",
                    c.id,
                    c.spec_sha256,
                    c.data_sha256,
                    c.build_commit,
                    c.report_sha256,
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
        let hash = sha256_hex(text.as_bytes());
        self.append(&format!("\"event\":\"seal\",\"sealed_sha256\":\"{}\",\"test_read\":false", hash))?;
        Ok(hash)
    }

    /// Check that `id` is in a sealed list that the ledger recorded, and that the
    /// data and build hashes match the sealed entry. Does not look at earlier reads.
    pub fn authorize_sealed(
        &self,
        sealed: &std::path::Path,
        id: &str,
        data_sha256: &str,
        build_commit: &str,
    ) -> Result<Candidate, String> {
        self.verify_chain()?;
        let bytes = std::fs::read(sealed).map_err(|e| format!("{}: {}", sealed.display(), e))?;
        let hash = sha256_hex(&bytes);
        let events = self.events()?;
        let sealed_ok = events
            .iter()
            .any(|e| matches!(e, Event::Seal { sealed_sha256 } if *sealed_sha256 == hash));
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
        };
        if candidate.data_sha256 != data_sha256 || candidate.build_commit != build_commit {
            return Err(format!(
                "candidate {}: data or build hash differs from the sealed entry \
                 (data {} vs {}, build {} vs {}); the run is void",
                id, data_sha256, candidate.data_sha256, build_commit, candidate.build_commit
            ));
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
    ) -> Result<Candidate, String> {
        let candidate = self.authorize_sealed(sealed, id, data_sha256, build_commit)?;
        if self.read_report_sha(id).is_some() {
            return Err(format!(
                "candidate {} already has test_read:true; the locked test is read once",
                id
            ));
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
    pub fn record_read(&self, id: &str, report_sha256: &str) -> Result<u64, String> {
        let count = self.read_count()? + 1;
        self.append(&format!(
            "\"event\":\"test_read\",\"id\":\"{}\",\"report_sha256\":\"{}\",\"read_count\":{},\"test_read\":true",
            id, report_sha256, count
        ))?;
        Ok(count)
    }

    /// Check that a baseline ladder may read the locked test: the sealed list
    /// was recorded in the ledger and no baseline read exists yet.
    pub fn authorize_baseline_read(&self, sealed: &std::path::Path) -> Result<(), String> {
        self.verify_chain()?;
        let bytes = std::fs::read(sealed).map_err(|e| format!("{}: {}", sealed.display(), e))?;
        let hash = sha256_hex(&bytes);
        let events = self.events()?;
        if !events
            .iter()
            .any(|e| matches!(e, Event::Seal { sealed_sha256 } if *sealed_sha256 == hash))
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
        Ok(())
    }

    /// Append the `baseline_read` event of a ladder run on the locked test.
    pub fn record_baseline_read(&self, data_sha256: &str, report_sha256: &str) -> Result<(), String> {
        self.append(&format!(
            "\"event\":\"baseline_read\",\"data_sha256\":\"{}\",\"report_sha256\":\"{}\",\"test_read\":true",
            data_sha256, report_sha256
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
  selectors --windows FILE --label LABEL --time-col COL [--scheme S]
      Print `--train-select` and `--test-select` for one window. The purge is in
      the train range (train_end = test_start - purge), in time and for every
      symbol; `--gap` does not apply to explicit selectors.
  verify --ledger FILE
      Check the ledger hash chain and print the line and read counts.
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
            match l.seal(std::path::Path::new(&path), &ids) {
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
