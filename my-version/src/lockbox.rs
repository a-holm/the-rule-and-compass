//! The 2026H2 crypto lockbox (issue #145).
//!
//! Every path that can train or evaluate on the holdout must go through
//! [`Guard`]. It refuses a window that overlaps the lockbox unless the caller
//! explicitly overrides it for a declared final test. The window is a UTC
//! half-open range; a window may abut the boundary but not cross it.
//!
//! ```text
//! 2026-07-01T00:00:00Z .. 2027-01-01T00:00:00Z
//! ```
//!
//! The same module also reports *coverage*: how many rows of a panel fall in the
//! lockbox and which UTC months they occupy. Builders (`prep`, `panel build`)
//! need that for provenance and must not silently drop the holdout, while
//! runners (`run`, `walkforward`) must refuse it.

/// Inclusive start of the 2026H2 lockbox (epoch seconds).
pub const LOCKBOX_START: i64 = 1_782_864_000;
/// Exclusive end of the 2026H2 lockbox (epoch seconds).
pub const LOCKBOX_END: i64 = 1_798_761_600;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Lockbox {
    pub start: i64,
    pub end: i64,
}

impl Default for Lockbox {
    fn default() -> Self {
        Lockbox {
            start: LOCKBOX_START,
            end: LOCKBOX_END,
        }
    }
}

/// One UTC calendar month touched by the lockbox, with the rows observed there.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct OverlapWindow {
    pub first_ts: i64,
    pub last_ts: i64,
    pub rows: usize,
}

/// How much of a row set falls inside the lockbox.
///
/// `rows == 0` means no overlap and `first_ts`/`last_ts` are zero. Windows are
/// grouped by UTC month, which is the granularity of the public archives and of
/// the monthly rebalance in a point-in-time panel.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Overlap {
    pub rows: usize,
    pub first_ts: i64,
    pub last_ts: i64,
    pub windows: Vec<OverlapWindow>,
}

impl Overlap {
    pub fn overlaps(&self) -> bool {
        self.rows > 0
    }
}

impl Lockbox {
    /// The pre-registered crypto lockbox.
    pub fn crypto_2026h2() -> Self {
        Self::default()
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        let (a, b) = s
            .split_once(',')
            .ok_or_else(|| format!("lockbox '{}' must be START,END", s))?;
        let start = parse_time(a.trim()).ok_or_else(|| format!("bad lockbox start '{}'", a))?;
        let end = parse_time(b.trim()).ok_or_else(|| format!("bad lockbox end '{}'", b))?;
        if end <= start {
            return Err("lockbox end must be after start".to_string());
        }
        Ok(Lockbox { start, end })
    }

    /// Refuse a half-open `[start, end)` window that intersects the lockbox.
    pub fn check(&self, start: i64, end: i64) -> Result<(), String> {
        if start < self.end && end > self.start {
            return Err(format!(
                "window [{}, {}) overlaps the lockbox [{}, {}) (2026H2); \
                 lock the holdout and use --allow-lockbox only for a declared final test",
                start, end, self.start, self.end
            ));
        }
        Ok(())
    }

    pub fn contains(&self, ts: i64) -> bool {
        self.start <= ts && ts < self.end
    }

    /// Count the timestamps inside the lockbox and group them by UTC month.
    pub fn overlap<I>(&self, times: I) -> Overlap
    where
        I: IntoIterator<Item = i64>,
    {
        let mut out = Overlap::default();
        let mut months: std::collections::BTreeMap<(i32, u32), OverlapWindow> =
            std::collections::BTreeMap::new();
        for ts in times {
            if !self.contains(ts) {
                continue;
            }
            if out.rows == 0 || ts < out.first_ts {
                out.first_ts = ts;
            }
            if out.rows == 0 || ts > out.last_ts {
                out.last_ts = ts;
            }
            out.rows += 1;
            let (y, m, _) = crate::ohlcv::civil_from_days(ts.div_euclid(86_400));
            let w = months.entry((y, m)).or_insert(OverlapWindow {
                first_ts: ts,
                last_ts: ts,
                rows: 0,
            });
            w.first_ts = w.first_ts.min(ts);
            w.last_ts = w.last_ts.max(ts);
            w.rows += 1;
        }
        out.windows = months.into_values().collect();
        out
    }

    /// JSON for a manifest block. `complete` is false for the pre-2027 archive
    /// (Oct–Dec 2026 are not published until 2027-02); `evaluated` is false
    /// unless a declared final test actually ran.
    pub fn json(&self, overlap: &Overlap, permitted: bool, complete: bool, evaluated: bool) -> String {
        let windows: Vec<String> = overlap
            .windows
            .iter()
            .map(|w| {
                format!(
                    "{{\"first_ts\":{},\"last_ts\":{},\"first_date\":{},\"last_date\":{},\"rows\":{}}}",
                    w.first_ts,
                    w.last_ts,
                    jstr(&date(w.first_ts)),
                    jstr(&date(w.last_ts)),
                    w.rows
                )
            })
            .collect();
        format!(
            "{{\"start\":{},\"end\":{},\"start_date\":{},\"end_date\":{},\
             \"permitted\":{},\"complete\":{},\"evaluated\":{},\
             \"rows\":{},\"first_ts\":{},\"last_ts\":{},\"windows\":[{}]}}",
            self.start,
            self.end,
            jstr(&date(self.start)),
            jstr(&date(self.end)),
            permitted,
            complete,
            evaluated,
            overlap.rows,
            overlap.first_ts,
            overlap.last_ts,
            windows.join(",")
        )
    }
}

/// The one guard every runner uses.
#[derive(Clone, Debug)]
pub struct Guard {
    pub lockbox: Lockbox,
    /// Explicit final-test override (`--allow-lockbox`).
    pub allow: bool,
    /// Free-text declaration echoed with the override (e.g. `final-2026h2`).
    pub declaration: Option<String>,
}

impl Default for Guard {
    fn default() -> Self {
        Guard {
            lockbox: Lockbox::crypto_2026h2(),
            allow: false,
            declaration: None,
        }
    }
}

impl Guard {
    pub fn new(allow: bool, declaration: Option<String>) -> Self {
        Guard {
            lockbox: Lockbox::crypto_2026h2(),
            allow,
            declaration,
        }
    }

    /// Refuse a named half-open window unless the override is set.
    pub fn check(&self, what: &str, start: i64, end: i64) -> Result<(), String> {
        if self.allow {
            return Ok(());
        }
        self.lockbox
            .check(start, end)
            .map_err(|e| format!("{}: {}", what, e))
    }

    /// Refuse a row set if any timestamp is inside the lockbox.
    pub fn check_times(&self, what: &str, times: &[i64]) -> Result<Overlap, String> {
        let overlap = self.lockbox.overlap(times.iter().copied());
        if overlap.overlaps() && !self.allow {
            return Err(format!(
                "{}: {} row(s) in [{}, {}) overlap the lockbox [{}, {}) (2026H2); \
                 lock the holdout and use --allow-lockbox only for a declared final test",
                what, overlap.rows, overlap.first_ts, overlap.last_ts, self.lockbox.start, self.lockbox.end
            ));
        }
        Ok(overlap)
    }

    /// Coverage of an arbitrary row set, without refusing. Builders use this to
    /// record what they retained.
    pub fn overlap<I>(&self, times: I) -> Overlap
    where
        I: IntoIterator<Item = i64>,
    {
        self.lockbox.overlap(times)
    }
}

/// A locked test range and where it was declared.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Declared {
    pub lockbox: Lockbox,
    /// `store` (read from the data's manifest, schema or sidecar) or `flag`.
    pub source: &'static str,
}

impl Declared {
    /// `2024-06-01..2024-07-01 (store manifest)` for messages.
    pub fn describe(&self) -> String {
        format!("{}..{} ({})", date(self.lockbox.start), date(self.lockbox.end), self.origin())
    }

    pub fn origin(&self) -> &'static str {
        if self.source == "store" {
            "store manifest"
        } else {
            "--lockbox"
        }
    }

    /// Refuse a row set with any timestamp in the declared range. The error
    /// names the first offending row, the split side and the fix.
    pub fn check_times(&self, side: &str, time_name: &str, times: &[i64]) -> Result<Overlap, String> {
        let overlap = self.lockbox.overlap(times.iter().copied());
        if overlap.overlaps() {
            return Err(format!(
                "{} has {} row(s) in the declared locked test {}..{} ({}; first row {}). \
                 Exclude them: --exclude '{} >= {}', or read the locked test only with \
                 rcsr walkforward --locked-test or rcsr ladder --locked-test",
                side,
                group_digits(overlap.rows),
                date(self.lockbox.start),
                date(self.lockbox.end),
                self.origin(),
                date(overlap.first_ts),
                time_name,
                self.lockbox.start
            ));
        }
        Ok(overlap)
    }

    /// The `lockbox` object fields of a report: declared range, source, rows
    /// in scope and whether the range was read.
    pub fn report_fields(&self, rows_in_scope: usize, evaluated: bool, checked: bool) -> String {
        format!(
            "\"declared\":{{\"start\":{},\"end\":{},\"start_date\":{},\"end_date\":{}}},\
             \"source\":{},\"rows_in_scope\":{},\"evaluated\":{},\"checked\":{}",
            self.lockbox.start,
            self.lockbox.end,
            jstr(&date(self.lockbox.start)),
            jstr(&date(self.lockbox.end)),
            jstr(self.source),
            rows_in_scope,
            evaluated,
            checked
        )
    }

    /// Append a `lockbox` object to a JSON report object.
    pub fn splice_report(&self, json: &str, rows_in_scope: usize, evaluated: bool, checked: bool) -> String {
        let body = json.trim_end();
        let body = body.strip_suffix('}').unwrap_or(body);
        format!(
            "{},\"lockbox\":{{{}}}}}",
            body,
            self.report_fields(rows_in_scope, evaluated, checked)
        )
    }
}

fn group_digits(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

impl Lockbox {
    /// The `locked_test` object stored with a panel, CSV or enrich output.
    pub fn declaration_json(&self, source: &str) -> String {
        format!(
            "{{\"start\":{},\"end\":{},\"start_date\":{},\"end_date\":{},\"source\":{}}}",
            self.start,
            self.end,
            jstr(&date(self.start)),
            jstr(&date(self.end)),
            jstr(source)
        )
    }

    /// Read a `"locked_test":{...}` object from sidecar or manifest text.
    pub fn from_declaration(text: &str) -> Option<Lockbox> {
        let obj = crate::manifest::object(text, "locked_test")?;
        let int = |key: &str| -> Option<i64> {
            let needle = format!("\"{}\":", key);
            let rest = &obj[obj.find(&needle)? + needle.len()..];
            let end = rest
                .char_indices()
                .find(|(i, c)| !(c.is_ascii_digit() || (*i == 0 && *c == '-')))
                .map(|(i, _)| i)
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        };
        let (start, end) = (int("start")?, int("end")?);
        (end > start).then_some(Lockbox { start, end })
    }
}

/// Sidecar that carries a declaration for a CSV (or any file without a
/// manifest of its own).
pub fn sidecar_path(data_path: &str) -> String {
    format!("{}.lockbox.json", data_path)
}

/// The locked test declared with a store or CSV: the `.lockbox.json` sidecar,
/// then the schema, the manifest and the enrich sidecar.
pub fn declared_for(data_path: &str) -> Option<Lockbox> {
    ["lockbox.json", "schema.json", "manifest.json", "enrich.json"]
        .iter()
        .find_map(|ext| {
            let text = std::fs::read_to_string(format!("{}.{}", data_path, ext)).ok()?;
            Lockbox::from_declaration(&text)
        })
}

/// Resolve the locked test of a run: an explicit `--lockbox` wins over the
/// declaration stored with the data.
pub fn resolve_declared(flag: Option<&str>, data_path: &str) -> Result<Option<Declared>, String> {
    match flag {
        Some(s) => Ok(Some(Declared {
            lockbox: Lockbox::parse(s)?,
            source: "flag",
        })),
        None if data_path.is_empty() || data_path == "-" => Ok(None),
        None => Ok(declared_for(data_path).map(|lockbox| Declared { lockbox, source: "store" })),
    }
}

/// Write the CSV sidecar for a declaration.
pub fn write_sidecar(data_path: &str, lockbox: &Lockbox, source: &str) -> Result<(), String> {
    write_sidecar_with(data_path, lockbox, source, "")
}

/// `write_sidecar` with extra top-level JSON members (each starting with a comma).
pub fn write_sidecar_with(data_path: &str, lockbox: &Lockbox, source: &str, extra: &str) -> Result<(), String> {
    let p = sidecar_path(data_path);
    std::fs::write(&p, format!("{{\"locked_test\":{}{}}}\n", lockbox.declaration_json(source), extra))
        .map_err(|e| format!("{}: {}", p, e))
}

/// Parse epoch seconds or `YYYY-MM-DD` / `YYYY-MM-DDTHH:MM:SS` (UTC).
pub fn parse_time(s: &str) -> Option<i64> {
    if let Ok(v) = s.parse::<i64>() {
        return Some(v);
    }
    let s = s.trim_end_matches('Z');
    let (date, time) = match s.split_once('T') {
        Some((d, t)) => (d, t),
        None => (s, "00:00:00"),
    };
    let mut d = date.split('-');
    let y: i32 = d.next()?.parse().ok()?;
    let m: u32 = d.next()?.parse().ok()?;
    let day: u32 = d.next()?.parse().ok()?;
    let mut t = time.split(':');
    let h: i64 = t.next().unwrap_or("0").parse().ok()?;
    let mi: i64 = t.next().unwrap_or("0").parse().ok()?;
    let sec: i64 = t.next().unwrap_or("0").parse().ok()?;
    Some(crate::ohlcv::days_from_civil(y, m, day) * 86_400 + h * 3_600 + mi * 60 + sec)
}

/// Whether a numeric time range is plausible Unix seconds (2000..2100). This
/// keeps the crypto lockbox from rejecting datasets whose time column is a row
/// index, days-since-epoch or a year fraction.
pub fn plausible_epoch_seconds(min: f64, max: f64) -> bool {
    const Y2000: f64 = 946_684_800.0;
    const Y2100: f64 = 4_102_444_800.0;
    min.is_finite() && max.is_finite() && min >= Y2000 && max <= Y2100
}

pub(crate) fn date(ts: i64) -> String {
    let (y, m, d) = crate::ohlcv::civil_from_days(ts.div_euclid(86_400));
    format!("{:04}-{:02}-{:02}", y, m, d)
}

fn jstr(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = std::fmt::Write::write_fmt(&mut out, format_args!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_window_abutting_the_lockbox_is_allowed_but_crossing_it_is_not() {
        let l = Lockbox::crypto_2026h2();
        assert!(l.check(1_700_000_000, l.start).is_ok());
        assert!(l.check(l.end, l.end + 86_400).is_ok());
        assert!(l.check(l.start, l.start + 86_400).is_err());
        assert!(l.check(l.start - 86_400, l.start + 1).is_err());
        assert!(l.check(l.start - 86_400, l.end + 86_400).is_err());
        assert!(l.contains(l.start));
        assert!(!l.contains(l.end));
    }

    #[test]
    fn dates_parse_to_the_published_boundary() {
        let l = Lockbox::parse("2026-07-01,2027-01-01").unwrap();
        assert_eq!(l, Lockbox::crypto_2026h2());
    }

    #[test]
    fn coverage_groups_rows_by_utc_month_and_counts_them() {
        let l = Lockbox::crypto_2026h2();
        let july = l.start + 3_600;
        let august = lockbox_date(2026, 8, 15);
        let outside = l.start - 86_400;
        let o = l.overlap([july, august, outside]);
        assert_eq!(o.rows, 2);
        assert_eq!(o.windows.len(), 2);
        assert_eq!(o.windows[0].rows, 1);
        assert_eq!(o.windows[1].rows, 1);
        assert_eq!(o.first_ts, july);
        assert_eq!(o.last_ts, august);
    }

    #[test]
    fn guard_refuses_rows_and_names_the_override() {
        let g = Guard::default();
        let times = [1_700_000_000, LOCKBOX_START + 10];
        let err = g.check_times("training split", &times).unwrap_err();
        assert!(err.contains("training split"));
        assert!(err.contains("lockbox"));
        let g = Guard::new(true, Some("final-2026h2".to_string()));
        let o = g.check_times("training split", &times).unwrap();
        assert_eq!(o.rows, 1);
        assert!(g.check("test window 'h2'", LOCKBOX_START, LOCKBOX_END).is_ok());
    }

    #[test]
    fn plausible_epoch_rejects_index_and_year_columns() {
        assert!(plausible_epoch_seconds(1_600_000_000.0, 1_700_000_000.0));
        assert!(!plausible_epoch_seconds(0.0, 1_000.0));
        assert!(!plausible_epoch_seconds(2_000.0, 2_026.0));
    }

    fn lockbox_date(y: i32, m: u32, d: u32) -> i64 {
        crate::ohlcv::days_from_civil(y, m, d) * 86_400
    }
}
