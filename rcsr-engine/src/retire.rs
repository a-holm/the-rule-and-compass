//! Retirement of a locked range after its one sealed read.
//!
//! `rcsr protocol retire` checks a ledger and writes a record. A command given
//! `--retired RECORD` re-hashes the ledger named in the record and then stops
//! guarding the protocol default range the record covers. The range is
//! development data from then on and is never a confirmatory test again.

use crate::lockbox::{date, int_field, Lockbox, ResolveError};
use crate::manifest::sha256_hex;
use crate::protocol::{field_str, Event, Ledger, LockedRange};

pub const SCHEMA: &str = "rcsr-retired/1";

/// A verified retirement: the ledger proved exactly one sealed read of `range`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Retirement {
    pub range: Lockbox,
    /// Absolute path of the ledger the record binds to.
    pub ledger: String,
    pub ledger_sha256: String,
    pub read_event: String,
    pub sealed_sha256: String,
    pub verified_reads: u64,
    pub utc: String,
}

fn refuse(exit: i32, message: String) -> ResolveError {
    ResolveError { exit, message }
}

fn utc_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs() as i64);
    let day = secs.rem_euclid(86_400);
    format!("{}T{:02}:{:02}:{:02}Z", date(secs), day / 3600, day % 3600 / 60, day % 60)
}

fn esc(s: &str) -> String {
    format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
}

impl Ledger {
    /// Check that `range` had exactly one locked read and that it follows a seal.
    /// Exit 1: the ledger is damaged. Exit 2: the range was not read. Exit 3: two
    /// or more reads, or a read before any seal.
    pub fn retire(&self, range: Lockbox) -> Result<Retirement, ResolveError> {
        self.verify_chain().map_err(|e| refuse(1, e))?;
        let events = self.events().map_err(|e| refuse(1, e))?;
        let mut seal: Option<&str> = None;
        let mut reads: Vec<(&str, Option<&str>)> = Vec::new();
        for e in &events {
            match e {
                Event::Seal { sealed_sha256, .. } => seal = Some(sealed_sha256),
                Event::TestRead { id, locked_test: Some(LockedRange { start, end, .. }), .. }
                    if (*start, *end) == (range.start, range.end) =>
                {
                    reads.push((id, seal));
                }
                _ => {}
            }
        }
        let span = format!("{}..{}", date(range.start), date(range.end));
        let ((id, sealed), n) = match reads.as_slice() {
            [] => return Err(refuse(2, format!("range not read: the ledger has no locked read of {}", span))),
            [one] => (*one, 1),
            many => return Err(refuse(3, format!("the ledger has {} locked reads of {}; a range retires after exactly one", many.len(), span))),
        };
        let sealed = sealed.ok_or_else(|| refuse(3, format!("the locked read {} of {} follows no seal event", id, span)))?;
        let (ledger, ledger_sha256) = ledger_identity(&self.path().display().to_string())?;
        Ok(Retirement {
            range,
            ledger,
            ledger_sha256,
            read_event: id.to_string(),
            sealed_sha256: sealed.to_string(),
            verified_reads: n,
            utc: utc_now(),
        })
    }
}

fn ledger_identity(path: &str) -> Result<(String, String), ResolveError> {
    let abs = std::fs::canonicalize(path).map_err(|e| refuse(1, format!("{}: {}", path, e)))?;
    let bytes = std::fs::read(&abs).map_err(|e| refuse(1, format!("{}: {}", abs.display(), e)))?;
    Ok((abs.display().to_string(), sha256_hex(&bytes)))
}

impl Retirement {
    /// The record file text.
    pub fn json(&self) -> String {
        format!(
            "{{\"schema\":\"{}\",\"range\":{{\"start\":{},\"end\":{}}},\"ledger\":{},\"ledger_sha256\":\"{}\",\"read_event\":{},\"sealed_sha256\":\"{}\",\"verified_reads\":{},\"utc\":\"{}\"}}\n",
            SCHEMA,
            self.range.start,
            self.range.end,
            esc(&self.ledger),
            self.ledger_sha256,
            esc(&self.read_event),
            self.sealed_sha256,
            self.verified_reads,
            self.utc
        )
    }

    /// The `retired` object of a report or sidecar.
    pub fn report_json(&self) -> String {
        format!(
            "{{\"range\":{{\"start\":{},\"end\":{},\"start_date\":\"{}\",\"end_date\":\"{}\"}},\"ledger_sha256\":\"{}\",\"read_event\":{}}}",
            self.range.start,
            self.range.end,
            date(self.range.start),
            date(self.range.end),
            self.ledger_sha256,
            esc(&self.read_event)
        )
    }

    /// The `,"retired":{...}` member of a sidecar or manifest.
    pub fn member(&self) -> String {
        format!(",\"retired\":{}", self.report_json())
    }

    /// Read a record and re-hash the ledger it names. A different ledger exits 3.
    pub fn load(path: &str) -> Result<Retirement, ResolveError> {
        let text = std::fs::read_to_string(path).map_err(|e| refuse(2, format!("--retired {}: {}", path, e)))?;
        let bad = |what: &str| refuse(2, format!("--retired {}: {}", path, what));
        if field_str(&text, "schema").as_deref() != Some(SCHEMA) {
            return Err(bad(&format!("not a {} record", SCHEMA)));
        }
        let range_obj = crate::manifest::object(&text, "range").ok_or_else(|| bad("no range"))?;
        let (start, end) = (
            int_field(range_obj, "start").ok_or_else(|| bad("no range start"))?,
            int_field(range_obj, "end").ok_or_else(|| bad("no range end"))?,
        );
        if end <= start {
            return Err(bad("range end is not after its start"));
        }
        let get = |k: &str| field_str(&text, k).ok_or_else(|| bad(&format!("no {}", k)));
        let r = Retirement {
            range: Lockbox { start, end },
            ledger: get("ledger")?,
            ledger_sha256: get("ledger_sha256")?,
            read_event: get("read_event")?,
            sealed_sha256: get("sealed_sha256")?,
            verified_reads: int_field(&text, "verified_reads").unwrap_or(0).max(0) as u64,
            utc: get("utc")?,
        };
        if r.verified_reads != 1 {
            return Err(bad("verified_reads is not 1"));
        }
        let bytes = std::fs::read(&r.ledger).map_err(|e| refuse(3, format!("--retired {}: ledger {}: {}", path, r.ledger, e)))?;
        if sha256_hex(&bytes) != r.ledger_sha256 {
            return Err(refuse(3, format!("--retired {}: the ledger {} no longer matches the record (ledger_sha256)", path, r.ledger)));
        }
        Ok(r)
    }

    /// Whether `other` overlaps the retired range (half-open).
    pub fn overlaps(&self, other: &Lockbox) -> bool {
        other.start < self.range.end && self.range.start < other.end
    }

    /// Refuse a locked range, or an input declaration, that reaches the retired range.
    pub fn check_new_range(&self, new: &Lockbox, what: &str) -> Result<(), ResolveError> {
        if self.overlaps(new) {
            return Err(refuse(
                2,
                format!(
                    "{} {}..{} overlaps the retired range {}..{}: a retired range is development data and never a locked test again",
                    what,
                    date(new.start),
                    date(new.end),
                    date(self.range.start),
                    date(self.range.end)
                ),
            ));
        }
        Ok(())
    }
}

/// Checks of a writer (`panel base`, `panel build`, `panel append`, `prep`,
/// `enrich`): the new range stays clear of the retired one and no input
/// declares a range that reaches it.
pub fn writer_check(retired: Option<&Retirement>, declared: Option<&Lockbox>, inputs: &[&str]) -> Result<(), ResolveError> {
    let Some(r) = retired else { return Ok(()) };
    let Some(new) = declared else {
        return Err(refuse(2, "--retired needs a locked range for the output (--locked-test START,END)".to_string()));
    };
    r.check_new_range(new, "--locked-test")?;
    for input in inputs {
        if let Some(s) = crate::lockbox::stored_all(input)? {
            for l in std::iter::once(&s.range).chain(s.also.iter()) {
                r.check_new_range(l, &format!("{} declares the locked test", s.file))?;
            }
        }
    }
    Ok(())
}

/// Load `--retired` for a writer and check it against the output range and the inputs.
pub fn writer_retirement(path: Option<&str>, declared: Option<&Lockbox>, inputs: &[&str]) -> Result<Option<Retirement>, ResolveError> {
    let retired = path.map(Retirement::load).transpose()?;
    writer_check(retired.as_ref(), declared, inputs)?;
    Ok(retired)
}

/// Put a `"retired"` object first in a JSON object text.
pub fn insert_retired(text: &str, retired: &Retirement) -> String {
    let at = text.find('{').map_or(0, |i| i + 1);
    let mut out = text.to_string();
    out.insert_str(at, &format!("\"retired\":{},", retired.report_json()));
    out
}
