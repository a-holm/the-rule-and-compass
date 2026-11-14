//! The campaign lock record and `rcsr lock check` (#1130).
//!
//! A lock record (`locked-tail-*.json`) names a range of rows that no
//! development step may see. `enrich` and the `panel` commands read it with
//! `--lock-record PATH` and leave the range out of every summary they write
//! next to a store: label reports, per-feature statistics, per-year counts,
//! membership rebalances, archive snapshots and schema extents. The store rows
//! themselves stay as they are; readers guard them. `rcsr lock check` scans
//! sidecars for entries inside the range and prints counts only.

use crate::bench_stats::{parse_json, Json};
use crate::lockbox::{date, Lockbox, ResolveError};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Exit code of `rcsr lock check` when a sidecar holds an entry inside the range.
pub const EXIT_IN_RANGE: i32 = 4;

/// The value of `source` in every declaration this module writes.
pub const SOURCE: &str = "lock record";

/// A parsed lock record: the range it guards and the SHA-256 of its bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LockRecord {
    pub range: Lockbox,
    pub sha256: String,
}

fn usage(message: String) -> ResolveError {
    ResolveError { exit: 2, message }
}

impl LockRecord {
    /// Read `path` and parse it. Exit 3 when the file cannot be read, exit 2
    /// when its content is not a usable record.
    pub fn load(path: &str) -> Result<LockRecord, ResolveError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| ResolveError { exit: 3, message: format!("--lock-record {}: {}", path, e) })?;
        Self::parse(path, &text)
    }

    pub fn parse(path: &str, text: &str) -> Result<LockRecord, ResolveError> {
        let bad = |what: &str| usage(format!("--lock-record {}: {}", path, what));
        let doc = parse_json(text).map_err(|e| bad(&format!("not valid JSON ({})", e)))?;
        let range = doc.get("range").ok_or_else(|| bad("missing field range"))?;
        let epoch = |key: &str| -> Result<i64, ResolveError> {
            match range.get(key) {
                Some(Json::Num(v)) if v.fract() == 0.0 && *v >= 0.0 && *v < 1e12 => Ok(*v as i64),
                Some(_) => Err(bad(&format!("range.{} must be an integer epoch in seconds", key))),
                None => Err(bad(&format!("missing field range.{}", key))),
            }
        };
        let start = epoch("start_epoch")?;
        let end = epoch("end_epoch")?;
        match range.get("end_exclusive") {
            Some(Json::Bool(true)) => {}
            Some(_) => return Err(bad("range.end_exclusive must be true (the range is half-open)")),
            None => return Err(bad("missing field range.end_exclusive")),
        }
        if end <= start {
            return Err(bad("range.end_epoch must be after range.start_epoch"));
        }
        Ok(LockRecord { range: Lockbox { start, end }, sha256: crate::manifest::sha256_hex(text.as_bytes()) })
    }

    /// The `"lock_record":{...}` object a sidecar carries. `rows_left_out` is
    /// the number of rows, buckets or entries the sidecar leaves out.
    pub fn declaration_json(&self, rows_left_out: usize) -> String {
        format!(
            "{{\"start\":{},\"end\":{},\"start_date\":\"{}\",\"end_date\":\"{}\",\"source\":\"{}\",\"record_sha256\":\"{}\",\"rows_left_out\":{}}}",
            self.range.start,
            self.range.end,
            date(self.range.start),
            date(self.range.end),
            SOURCE,
            self.sha256,
            rows_left_out
        )
    }

    /// Whether a label sidecar entry line (`{"target":..,"report":{..}}`) was computed under this record: its
    /// report declares the same range as `locked_test_also`.
    pub fn computed_entry(&self, line: &str) -> bool {
        let Ok(doc) = parse_json(line) else { return false };
        let Some(also) = doc.get("report").and_then(|r| r.get("locked_test_also")) else { return false };
        also.get("start_date").and_then(Json::as_str) == Some(date(self.range.start).as_str())
            && also.get("end_date").and_then(Json::as_str) == Some(date(self.range.end).as_str())
    }

    /// Put `"lock_record"` first in a JSON object text.
    pub fn insert_declaration(&self, text: &str, rows_left_out: usize) -> String {
        let at = text.find('{').map_or(0, |i| i + 1);
        let mut out = text.to_string();
        out.insert_str(at, &format!("\"lock_record\":{},", self.declaration_json(rows_left_out)));
        out
    }
}

/// The ranges a resolved lock and a record guard together. The record takes
/// the second-range slot; it never replaces a declared range. A different
/// second range already in place is refused (exit 2).
pub fn with_record(
    primary: Option<Lockbox>,
    also: Option<Lockbox>,
    record: &Lockbox,
) -> Result<(Option<Lockbox>, Option<Lockbox>), ResolveError> {
    match (primary, also) {
        (None, _) => Ok((Some(*record), also)),
        (Some(p), None) if p == *record => Ok((primary, None)),
        (Some(_), None) => Ok((primary, Some(*record))),
        (Some(_), Some(a)) if a == *record => Ok((primary, also)),
        (Some(p), Some(a)) => Err(usage(format!(
            "--lock-record range {} needs the second-range slot, but the resolved locked ranges are {} + {} (declared by the data, or the protocol default and --locked-test)",
            crate::lockbox::span(record),
            crate::lockbox::span(&p),
            crate::lockbox::span(&a)
        ))),
    }
}

// ---------------------------------------------------------------------------
// rcsr lock check
// ---------------------------------------------------------------------------

/// Keys whose values are sizes or file-system times, never data times.
const NOT_DATA: [&str; 8] = ["bytes", "data_bytes", "output_bytes", "input_bytes", "size", "mtime", "utc", "built_utc"];
/// Objects that name the range itself.
const DECLARATIONS: [&str; 4] = ["locked_test", "locked_test_also", "lock_record", "retired"];
/// Boundary fields of a `lockbox` object.
const BOUNDS: [&str; 4] = ["start", "end", "start_date", "end_date"];

/// Seconds since the epoch of an ISO date or date-time string, else `None`.
fn iso_time(s: &str) -> Option<i64> {
    let b = s.trim().as_bytes();
    let digits = |from: usize, n: usize| -> Option<i64> {
        let part = b.get(from..from + n)?;
        part.iter().all(u8::is_ascii_digit).then(|| part.iter().fold(0i64, |a, d| a * 10 + (d - b'0') as i64))
    };
    if b.len() < 10 || b[4] != b'-' || b[7] != b'-' {
        return None;
    }
    let (y, m, d) = (digits(0, 4)?, digits(5, 2)?, digits(8, 2)?);
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = 0i64;
    let mut at = 10;
    if b.len() > at {
        if b[at] != b'T' && b[at] != b' ' {
            return None;
        }
        at += 1;
        let (h, mi) = (digits(at, 2)?, digits(at + 3, 2)?);
        if b.get(at + 2) != Some(&b':') || h > 23 || mi > 59 {
            return None;
        }
        secs = h * 3600 + mi * 60;
        at += 5;
        if b.get(at) == Some(&b':') {
            let s = digits(at + 1, 2)?;
            if s > 60 {
                return None;
            }
            secs += s;
            at += 3;
        }
        if b.get(at) == Some(&b'.') {
            at += 1;
            while b.get(at).is_some_and(u8::is_ascii_digit) {
                at += 1;
            }
        }
        if b.get(at) == Some(&b'Z') {
            at += 1;
        }
        if at != b.len() {
            return None;
        }
    }
    Some(crate::ohlcv::days_from_civil(y as i32, m as u32, d as u32) * 86_400 + secs)
}

/// Whether an integer-valued number is an epoch in seconds, milliseconds or
/// microseconds inside `range`.
fn number_in(range: &Lockbox, v: f64) -> bool {
    if !v.is_finite() || v.fract() != 0.0 || v < 0.0 || v > 9e15 {
        return false;
    }
    let v = v as i64;
    range.contains(v) || range.contains(v / 1000) && v >= 100_000_000_000 && v < 100_000_000_000_000 || range.contains(v / 1_000_000) && v >= 100_000_000_000_000
}

fn count_json(v: &Json, range: &Lockbox, path: &str, owner: &str, out: &mut BTreeMap<String, usize>) {
    match v {
        Json::Num(x) => {
            if number_in(range, *x) {
                *out.entry(path.to_string()).or_insert(0) += 1;
            }
        }
        Json::Str(s) => {
            if iso_time(s).is_some_and(|t| range.contains(t)) {
                *out.entry(path.to_string()).or_insert(0) += 1;
            }
        }
        Json::Arr(items) => {
            let child = format!("{}[]", path);
            for it in items {
                count_json(it, range, &child, owner, out);
            }
        }
        Json::Obj(kv) => {
            for (k, val) in kv {
                if DECLARATIONS.contains(&k.as_str()) || (owner == "lockbox" && BOUNDS.contains(&k.as_str())) {
                    continue;
                }
                if NOT_DATA.contains(&k.as_str()) && matches!(val, Json::Num(_) | Json::Str(_)) {
                    continue;
                }
                let child = if path.is_empty() { k.clone() } else { format!("{}.{}", path, k) };
                count_json(val, range, &child, k, out);
            }
        }
        Json::Bool(_) | Json::Null => {}
    }
}

/// Entries of a JSON text inside `range`, by key path. Array indexes collapse
/// to `[]`. A lock record itself is skipped.
pub fn check_json(text: &str, range: &Lockbox) -> Result<BTreeMap<String, usize>, String> {
    let doc = parse_json(text)?;
    let mut out = BTreeMap::new();
    if doc.get("schema").and_then(Json::as_str) == Some("rcsr-locked-tail/1") {
        return Ok(out);
    }
    count_json(&doc, range, "", "", &mut out);
    Ok(out)
}

/// Refuse a snapshot that holds an entry inside the record range: a manifest or
/// panel that embeds it would carry the entry along.
pub fn refuse_snapshot(text: &str, rec: &LockRecord) -> Result<(), String> {
    let counts = check_json(text, &rec.range).map_err(|e| format!("snapshot: not valid JSON ({})", e))?;
    if counts.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the snapshot holds entries inside the lock record range at {} key path(s); write it with `rcsr panel snapshot --lock-record` (check with `rcsr lock check`)",
        counts.len()
    ))
}

/// Entries of a CSV text inside `range`, by header name.
pub fn check_csv(text: &str, range: &Lockbox) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    let mut lines = text.lines().peekable();
    let Some(head) = lines.peek().copied() else { return out };
    let cells: Vec<&str> = head.split(',').map(|n| n.trim().trim_matches('"')).collect();
    // A first line holding a number or a date is data: the file has no header and no cell is ever printed as a key.
    let headerless = cells.iter().any(|c| c.parse::<f64>().is_ok() || iso_time(c).is_some());
    let names: Vec<&str> = if headerless { Vec::new() } else { cells };
    if !headerless {
        lines.next();
    }
    for line in lines {
        for (k, cell) in line.split(',').enumerate() {
            let cell = cell.trim().trim_matches('"');
            let hit = match cell.parse::<f64>() {
                Ok(v) => number_in(range, v),
                Err(_) => iso_time(cell).is_some_and(|t| range.contains(t)),
            };
            if hit {
                let name = names.get(k).copied().filter(|n| !n.is_empty()).map_or_else(|| format!("column{}", k + 1), str::to_string);
                if !NOT_DATA.contains(&name.as_str()) {
                    *out.entry(name).or_insert(0) += 1;
                }
            }
        }
    }
    out
}

fn collect(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    let meta = std::fs::symlink_metadata(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    if meta.is_dir() {
        let mut entries: Vec<PathBuf> = std::fs::read_dir(path)
            .map_err(|e| format!("{}: {}", path.display(), e))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .collect();
        entries.sort();
        for e in entries {
            collect(&e, files)?;
        }
    } else if meta.is_file() {
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("");
        if ext == "json" || ext == "csv" {
            files.push(path.to_path_buf());
        }
    }
    Ok(())
}

/// One finding: a file, a key path and the number of entries inside the range.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    pub file: String,
    pub key: String,
    pub count: usize,
}

impl Hit {
    pub fn line(&self) -> String {
        format!("{}:{}:{}", self.file, self.key, self.count)
    }
}

/// Scan every `*.json` and `*.csv` below `roots` (files or directories) and
/// the number of files read. `skip` names a file that is not scanned (the record).
pub fn check_paths(roots: &[String], range: &Lockbox, skip: Option<&Path>) -> Result<(Vec<Hit>, usize), String> {
    let mut files = Vec::new();
    for r in roots {
        collect(Path::new(r), &mut files)?;
    }
    let skip = skip.and_then(|p| std::fs::canonicalize(p).ok());
    let mut hits = Vec::new();
    let mut scanned = 0usize;
    for f in files {
        if skip.is_some() && std::fs::canonicalize(&f).ok() == skip {
            continue;
        }
        let text = std::fs::read_to_string(&f).map_err(|e| format!("{}: {}", f.display(), e))?;
        let counts = if f.extension().is_some_and(|e| e == "csv") {
            check_csv(&text, range)
        } else {
            check_json(&text, range).map_err(|e| format!("{}: not valid JSON ({})", f.display(), e))?
        };
        scanned += 1;
        for (key, count) in counts {
            hits.push(Hit { file: f.display().to_string(), key, count });
        }
    }
    Ok((hits, scanned))
}

pub fn usage_text() -> &'static str {
    "rcsr lock check --record R PATH [PATH...]

Scan the *.json and *.csv sidecars below each PATH (a file or a directory) for
entries inside the range of the lock record R. Epoch numbers (seconds,
milliseconds, microseconds) and ISO dates count. Never *.rcsb, *.bin or column
blocks. A lock record, declaration objects (locked_test, locked_test_also,
lock_record, retired), the bounds of a lockbox object and the keys bytes,
data_bytes, output_bytes, input_bytes, size, mtime, utc and built_utc are not
data times and are skipped.

stdout: one `path:key.path:count` line per finding, never a value. Array indexes
collapse to []. A CSV key is its header name, or column<k> when the first line holds a number or a date (no header).
exit: 0 no entry inside the range, 4 at least one, 2 usage or a bad record
(range.end_exclusive must be true, range.start_epoch < range.end_epoch), 3 an
unreadable file or a *.json that does not parse.
"
}

/// `rcsr lock <subcommand>`.
pub fn run(args: &[String]) -> ! {
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{}", usage_text());
        std::process::exit(if args.is_empty() { 2 } else { 0 });
    }
    if args[0] != "check" {
        eprintln!("error: unknown lock command '{}'\n\n{}", args[0], usage_text());
        std::process::exit(2);
    }
    let mut record: Option<String> = None;
    let mut roots: Vec<String> = Vec::new();
    let mut i = 1usize;
    while i < args.len() {
        match args[i].as_str() {
            "-h" | "--help" => {
                print!("{}", usage_text());
                std::process::exit(0);
            }
            "--record" => {
                record = Some(args.get(i + 1).cloned().unwrap_or_else(|| {
                    eprintln!("error: --record requires a value");
                    std::process::exit(2);
                }));
                i += 1;
            }
            other if other.starts_with("--") => {
                eprintln!("error: unknown lock check option '{}'\n\n{}", other, usage_text());
                std::process::exit(2);
            }
            other => roots.push(other.to_string()),
        }
        i += 1;
    }
    let Some(record) = record else {
        eprintln!("error: lock check needs --record <lock record JSON>");
        std::process::exit(2);
    };
    if roots.is_empty() {
        eprintln!("error: lock check needs at least one PATH");
        std::process::exit(2);
    }
    let rec = LockRecord::load(&record).unwrap_or_else(|e| {
        eprintln!("error: {}", e.message);
        std::process::exit(e.exit);
    });
    match check_paths(&roots, &rec.range, Some(Path::new(&record))) {
        Ok((hits, scanned)) => {
            for h in &hits {
                println!("{}", h.line());
            }
            eprintln!("lock check: {} file(s) scanned, {} location(s) with entries inside the locked range", scanned, hits.len());
            std::process::exit(if hits.is_empty() { 0 } else { EXIT_IN_RANGE });
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const R: Lockbox = Lockbox { start: 1_000_000_000, end: 1_000_100_000 };

    #[test]
    fn iso_strings_parse_strictly() {
        assert_eq!(iso_time("1970-01-02"), Some(86_400));
        assert_eq!(iso_time("1970-01-01T01:02:03Z"), Some(3723));
        assert_eq!(iso_time("1970-01-01 00:01"), Some(60));
        assert_eq!(iso_time("BTCUSDT-15m-2026-08"), None);
        assert_eq!(iso_time("2026-13-01"), None);
        assert_eq!(iso_time("2026-08-01x"), None);
    }

    #[test]
    fn numbers_in_three_units_count() {
        assert!(number_in(&R, 1_000_050_000.0));
        assert!(number_in(&R, 1_000_050_000_000.0));
        assert!(number_in(&R, 1_000_050_000_000_000.0));
        assert!(!number_in(&R, 1_000_100_000.0));
        assert!(!number_in(&R, 1_000_050_000.5));
        assert!(!number_in(&R, 5.0));
    }

    #[test]
    fn record_needs_an_exclusive_end_and_an_ordered_pair() {
        let ok = r#"{"range":{"start_epoch":10,"end_epoch":20,"end_exclusive":true}}"#;
        assert_eq!(LockRecord::parse("r", ok).unwrap().range, Lockbox { start: 10, end: 20 });
        for (text, field) in [
            (r#"{"range":{"start_epoch":10,"end_epoch":20,"end_exclusive":false}}"#, "range.end_exclusive"),
            (r#"{"range":{"start_epoch":10,"end_epoch":20}}"#, "range.end_exclusive"),
            (r#"{"range":{"start_epoch":20,"end_epoch":10,"end_exclusive":true}}"#, "range.end_epoch"),
            (r#"{"range":{"start_epoch":"x","end_epoch":10,"end_exclusive":true}}"#, "range.start_epoch"),
            (r#"{"range":{"end_epoch":10,"end_exclusive":true}}"#, "range.start_epoch"),
        ] {
            let e = LockRecord::parse("r", text).unwrap_err();
            assert_eq!(e.exit, 2);
            assert!(e.message.contains(field), "{}", e.message);
        }
    }

    #[test]
    fn json_findings_skip_declarations_and_sizes() {
        let text = r#"{"locked_test_also":{"start":1000000000},"bytes":1000050000,"entries":[{"last_ts":1000050000,"first_ts":5},{"last_ts":1000050001}],"d":"1970-01-01"}"#;
        let got = check_json(text, &R).unwrap();
        assert_eq!(got.get("entries[].last_ts"), Some(&2));
        assert_eq!(got.len(), 1);
    }

    #[test]
    fn a_record_is_not_scanned() {
        let text = r#"{"schema":"rcsr-locked-tail/1","range":{"start_epoch":1000000000}}"#;
        assert!(check_json(text, &R).unwrap().is_empty());
    }

    #[test]
    fn csv_findings_are_counted_by_header() {
        let got = check_csv("as_of,symbol\n1000050000,A\n5,B\n1000050002,C\n", &R);
        assert_eq!(got.get("as_of"), Some(&2));
    }

    #[test]
    fn a_record_range_fills_the_second_slot_or_is_refused() {
        let r = Lockbox { start: 5, end: 9 };
        let p = Lockbox { start: 1, end: 3 };
        assert_eq!(with_record(Some(p), None, &r).unwrap(), (Some(p), Some(r)));
        assert_eq!(with_record(Some(p), Some(r), &r).unwrap(), (Some(p), Some(r)));
        assert_eq!(with_record(Some(r), None, &r).unwrap(), (Some(r), None));
        assert!(with_record(Some(p), Some(Lockbox { start: 20, end: 30 }), &r).is_err());
    }
}
