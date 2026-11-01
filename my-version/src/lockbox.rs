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
    /// `store`, `flag` or `protocol_default`.
    pub source: &'static str,
    /// The protocol default guarded next to a flag range on undeclared data.
    pub also: Option<Lockbox>,
}

impl Declared {
    /// A range from the store or a flag fails closed (no time column, not epoch
    /// seconds). The protocol default only checks data that has epoch stamps.
    pub fn strict(&self) -> bool {
        self.source != "protocol_default"
    }

    /// `2024-06-01..2024-07-01 (store manifest)` for messages.
    pub fn describe(&self) -> String {
        format!("{}..{} ({})", date(self.lockbox.start), date(self.lockbox.end), self.origin())
    }

    pub fn origin(&self) -> &'static str {
        match self.source {
            "store" => "store manifest",
            "protocol_default" => "protocol default, opt out with --lockbox none",
            "flag+protocol_default" => "--lockbox; the protocol default is guarded too",
            _ => "--lockbox",
        }
    }

    /// Refuse a row set with any timestamp in the declared range. The error
    /// names the first offending row, the split side and the fix.
    pub fn check_times(&self, side: &str, time_name: &str, times: &[i64]) -> Result<Overlap, String> {
        let overlap = self.check_range(&self.lockbox, self.origin(), side, time_name, times)?;
        if let Some(also) = &self.also {
            self.check_range(also, "protocol default, opt out with --lockbox none", side, time_name, times)?;
        }
        Ok(overlap)
    }

    fn check_range(
        &self,
        lockbox: &Lockbox,
        origin: &str,
        side: &str,
        time_name: &str,
        times: &[i64],
    ) -> Result<Overlap, String> {
        let overlap = lockbox.overlap(times.iter().copied());
        if overlap.overlaps() {
            return Err(format!(
                "{} has {} row(s) in the declared locked test {}..{} ({}; first row {}). \
                 Exclude them: --exclude '{} >= {}', or read the locked test only with \
                 rcsr walkforward --locked-test or rcsr ladder --locked-test",
                side,
                group_digits(overlap.rows),
                date(lockbox.start),
                date(lockbox.end),
                origin,
                date(overlap.first_ts),
                time_name,
                lockbox.start
            ));
        }
        Ok(overlap)
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

/// Where the locked test of a run came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Flag,
    FlagAndDefault,
    Store,
    ProtocolDefault,
    None,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Flag => "flag",
            Source::FlagAndDefault => "flag+protocol_default",
            Source::Store => "store",
            Source::ProtocolDefault => "protocol_default",
            Source::None => "none",
        }
    }
}

/// A refused resolution. `exit` is 2 for a usage error and 3 for an input error.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResolveError {
    pub exit: i32,
    pub message: String,
}

fn usage(message: String) -> ResolveError {
    ResolveError { exit: 2, message }
}

fn input(message: String) -> ResolveError {
    ResolveError { exit: 3, message }
}

/// The locked test of a run, its source and every file that declared a range.
#[derive(Clone, Debug)]
pub struct Resolved {
    /// `None` only for an explicit `--lockbox none`.
    pub range: Option<Lockbox>,
    pub source: Source,
    /// The protocol default guarded next to a flag range on undeclared data.
    pub also: Option<Lockbox>,
    pub declared_in: Vec<(String, Lockbox)>,
}

impl Resolved {
    /// The active range as a [`Declared`], or `None` for `--lockbox none`.
    pub fn declared(&self) -> Option<Declared> {
        self.range.map(|lockbox| Declared { lockbox, source: self.source.as_str(), also: self.also })
    }

    /// Every guarded range: the primary one, then the protocol default added to a flag.
    pub fn ranges(&self) -> Vec<Lockbox> {
        self.range.into_iter().chain(self.also).collect()
    }

    /// The single range a derived sidecar declares: the union when the flag
    /// and the protocol default overlap, else the protocol default.
    pub fn sidecar_range(&self) -> Option<Lockbox> {
        match (self.range, self.also) {
            (Some(f), Some(d)) if f.start <= d.end && f.end >= d.start => {
                Some(Lockbox { start: f.start.min(d.start), end: f.end.max(d.end) })
            }
            (Some(_), Some(d)) => Some(d),
            (r, _) => r,
        }
    }

    /// The `lockbox` object fields of a report: declared range, source, the
    /// files that declared it, rows in scope and whether the range was read.
    pub fn report_fields(&self, rows_in_scope: usize, evaluated: bool, checked: bool) -> String {
        let range_json = |r: &Option<Lockbox>| match r {
            Some(l) => format!(
                "{{\"start\":{},\"end\":{},\"start_date\":{},\"end_date\":{}}}",
                l.start,
                l.end,
                jstr(&date(l.start)),
                jstr(&date(l.end))
            ),
            None => "null".to_string(),
        };
        let (declared, also) = (range_json(&self.range), range_json(&self.also));
        let files: Vec<String> = self
            .declared_in
            .iter()
            .map(|(f, l)| {
                format!(
                    "{{\"file\":{},\"start_date\":{},\"end_date\":{}}}",
                    jstr(f),
                    jstr(&date(l.start)),
                    jstr(&date(l.end))
                )
            })
            .collect();
        format!(
            "\"declared\":{},\"also\":{},\"source\":{},\"declared_in\":[{}],\"rows_in_scope\":{},\"evaluated\":{},\"checked\":{}",
            declared,
            also,
            jstr(self.source.as_str()),
            files.join(","),
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

const CARRIERS: [&str; 4] = ["lockbox.json", "schema.json", "manifest.json", "enrich.json"];

/// Every locked-test declaration of `data_path`, in carrier order. A carrier
/// that has a `locked_test` object which does not parse is an input error.
pub fn declarations(data_path: &str) -> Result<Vec<(String, Lockbox)>, ResolveError> {
    let mut out = Vec::new();
    if data_path.is_empty() || data_path == "-" {
        return Ok(out);
    }
    for ext in CARRIERS {
        let file = format!("{}.{}", data_path, ext);
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        if crate::manifest::object(&text, "locked_test").is_none() {
            continue;
        }
        match Lockbox::from_declaration(&text) {
            Some(lb) => out.push((file, lb)),
            None => {
                return Err(input(format!(
                    "{}: unreadable locked_test declaration (needs integer start < end)",
                    file
                )))
            }
        }
    }
    Ok(out)
}

/// The protocol default locked test, 2025-07-01..2026-07-01.
pub fn protocol_default() -> Lockbox {
    let (start, end) = crate::protocol::default_locked_range();
    Lockbox { start: start as i64, end: end as i64 }
}

pub fn span(l: &Lockbox) -> String {
    format!("{}..{}", date(l.start), date(l.end))
}

/// Resolve the locked test of a run: flag, else store declaration, else the
/// protocol default. `--lockbox none` is the only opt-out and is refused when
/// the data declares a range. A flag may widen a store declaration, never
/// narrow it. Conflicting store declarations are refused.
pub fn resolve(flag: Option<&str>, data_path: &str) -> Result<Resolved, ResolveError> {
    let explicit = match flag {
        None => None,
        Some(s) if s.trim().eq_ignore_ascii_case("none") => Some(None),
        Some(s) => Some(Some(Lockbox::parse(s).map_err(|e| usage(format!("--lockbox: {}", e)))?)),
    };
    let declared_in = declarations(data_path)?;
    if let Some((first_file, first)) = declared_in.first() {
        if let Some((other_file, other)) = declared_in.iter().find(|(_, l)| l != first) {
            return Err(input(format!(
                "{}: conflicting locked-test declarations: {} {}, {} {}; remove the stale file",
                data_path,
                first_file,
                span(first),
                other_file,
                span(other)
            )));
        }
    }
    let stored = declared_in.first().map(|(f, l)| (f.clone(), *l));
    match (explicit, stored) {
        (Some(None), Some((file, l))) => Err(usage(format!(
            "--lockbox none is refused: {} declares the locked test {}",
            file,
            span(&l)
        ))),
        (Some(None), None) => Ok(Resolved { range: None, source: Source::None, also: None, declared_in }),
        (Some(Some(f)), Some((file, l))) if !(f.start <= l.start && f.end >= l.end) => Err(usage(format!(
            "--lockbox {} does not cover the declared locked test {} ({})",
            span(&f),
            span(&l),
            file
        ))),
        (Some(Some(f)), Some(_)) => Ok(Resolved { range: Some(f), source: Source::Flag, also: None, declared_in }),
        (Some(Some(f)), None) => {
            let d = protocol_default();
            let (source, also) = if f.start <= d.start && f.end >= d.end {
                (Source::Flag, None)
            } else {
                (Source::FlagAndDefault, Some(d))
            };
            Ok(Resolved { range: Some(f), source, also, declared_in })
        }
        (None, Some((_, l))) => Ok(Resolved { range: Some(l), source: Source::Store, also: None, declared_in }),
        (None, None) => Ok(Resolved {
            range: Some(protocol_default()),
            source: Source::ProtocolDefault,
            also: None,
            declared_in,
        }),
    }
}

/// Write the CSV sidecar for a declaration.
pub fn write_sidecar(data_path: &str, lockbox: &Lockbox, source: &str) -> Result<(), String> {
    let p = sidecar_path(data_path);
    std::fs::write(&p, format!("{{\"locked_test\":{}}}\n", lockbox.declaration_json(source)))
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

    fn resolve_dir(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("rcsr-lockbox-resolve-{}-{}", std::process::id(), name));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn declare(data: &str, ext: &str, lb: &Lockbox) {
        std::fs::write(
            format!("{}.{}", data, ext),
            format!("{{\"locked_test\":{}}}", lb.declaration_json("t")),
        )
        .unwrap();
    }

    fn stored() -> Lockbox {
        Lockbox { start: lockbox_date(2024, 6, 1), end: lockbox_date(2024, 7, 1) }
    }

    #[test]
    fn resolve_flag_on_undeclared_data_adds_to_the_protocol_default() {
        let d = resolve_dir("flag");
        let data = d.join("a.csv").to_str().unwrap().to_string();
        let r = resolve(Some("2030-01-01,2030-02-01"), &data).unwrap();
        assert_eq!(r.source, Source::FlagAndDefault);
        assert_eq!(r.also, Some(protocol_default()));
        assert_eq!(r.ranges().len(), 2);
        let r = resolve(Some("2025-01-01,2027-01-01"), &data).unwrap();
        assert_eq!((r.source, r.also), (Source::Flag, None));
        let r = resolve(Some("2030-01-01,2030-02-01"), &data).unwrap();
        assert_eq!(r.range, Some(Lockbox::parse("2030-01-01,2030-02-01").unwrap()));
        assert!(r.declared_in.is_empty());
        assert_eq!(r.sidecar_range(), r.also);
        let r = resolve(Some("2025-01-01,2025-08-01"), &data).unwrap();
        let u = r.sidecar_range().unwrap();
        assert_eq!((u.start, u.end), (Lockbox::parse("2025-01-01,2025-08-01").unwrap().start, protocol_default().end));
        let e = resolve(Some("garbage"), &data).unwrap_err();
        assert_eq!(e.exit, 2);
        assert!(e.message.starts_with("--lockbox: "), "{}", e.message);
    }

    #[test]
    fn resolve_store_only_uses_the_declaration() {
        let d = resolve_dir("store");
        let data = d.join("a.rcsb").to_str().unwrap().to_string();
        declare(&data, "enrich.json", &stored());
        let r = resolve(None, &data).unwrap();
        assert_eq!(r.source, Source::Store);
        assert_eq!(r.range, Some(stored()));
        assert_eq!(r.declared_in.len(), 1);
        assert!(r.declared_in[0].0.ends_with("a.rcsb.enrich.json"));
    }

    #[test]
    fn resolve_undeclared_data_falls_back_to_the_protocol_default() {
        let d = resolve_dir("default");
        let data = d.join("a.csv").to_str().unwrap().to_string();
        for path in [data.as_str(), "-", ""] {
            let r = resolve(None, path).unwrap();
            assert_eq!(r.source, Source::ProtocolDefault);
            let l = r.range.unwrap();
            assert_eq!((l.start, l.end), (lockbox_date(2025, 7, 1), lockbox_date(2026, 7, 1)));
            assert!(r.declared_in.is_empty());
        }
    }

    #[test]
    fn resolve_none_is_the_only_opt_out_and_is_refused_over_a_declaration() {
        let d = resolve_dir("none");
        let data = d.join("a.csv").to_str().unwrap().to_string();
        let r = resolve(Some("none"), &data).unwrap();
        assert_eq!((r.source, r.range), (Source::None, None));
        assert!(r.declared().is_none());
        assert!(r.report_fields(0, false, false).contains("\"source\":\"none\""));
        declare(&data, "lockbox.json", &stored());
        let e = resolve(Some("none"), &data).unwrap_err();
        assert_eq!(e.exit, 2);
        assert!(e.message.contains("--lockbox none is refused"), "{}", e.message);
        assert!(e.message.contains("2024-06-01..2024-07-01"), "{}", e.message);
    }

    #[test]
    fn resolve_conflicting_files_are_refused_naming_both() {
        let d = resolve_dir("conflict");
        let data = d.join("a.rcsb").to_str().unwrap().to_string();
        declare(&data, "enrich.json", &stored());
        declare(&data, "lockbox.json", &Lockbox::parse("2030-01-01,2030-02-01").unwrap());
        let e = resolve(None, &data).unwrap_err();
        assert_eq!(e.exit, 3);
        assert!(e.message.contains("conflicting locked-test declarations"), "{}", e.message);
        assert!(e.message.contains("a.rcsb.lockbox.json 2030-01-01..2030-02-01"), "{}", e.message);
        assert!(e.message.contains("a.rcsb.enrich.json 2024-06-01..2024-07-01"), "{}", e.message);
        assert!(e.message.ends_with("remove the stale file"), "{}", e.message);
        assert_eq!(resolve(Some("2020-01-01,2031-01-01"), &data).unwrap_err().exit, 3);
    }

    #[test]
    fn resolve_equal_ranges_in_two_files_are_not_a_conflict() {
        let d = resolve_dir("equal");
        let data = d.join("a.rcsb").to_str().unwrap().to_string();
        declare(&data, "enrich.json", &stored());
        declare(&data, "manifest.json", &stored());
        let r = resolve(None, &data).unwrap();
        assert_eq!((r.source, r.range, r.declared_in.len()), (Source::Store, Some(stored()), 2));
    }

    #[test]
    fn resolve_a_narrowing_flag_is_refused_and_a_widening_flag_is_accepted() {
        let d = resolve_dir("narrow");
        let data = d.join("a.rcsb").to_str().unwrap().to_string();
        declare(&data, "enrich.json", &stored());
        for flag in ["2030-01-01,2030-02-01", "2024-06-05,2024-07-01", "2024-06-01,2024-06-30"] {
            let e = resolve(Some(flag), &data).unwrap_err();
            assert_eq!(e.exit, 2, "{}", flag);
            assert!(e.message.contains("does not cover the declared locked test 2024-06-01..2024-07-01"), "{}", e.message);
            assert!(e.message.contains("a.rcsb.enrich.json"), "{}", e.message);
        }
        let r = resolve(Some("2024-05-01,2024-08-01"), &data).unwrap();
        assert_eq!(r.source, Source::Flag);
        assert_eq!(r.range, Some(Lockbox::parse("2024-05-01,2024-08-01").unwrap()));
        assert_eq!(r.declared_in.len(), 1);
        let same = resolve(Some("2024-06-01,2024-07-01"), &data).unwrap();
        assert_eq!(same.source, Source::Flag);
    }

    #[test]
    fn resolve_an_unparsable_declaration_is_an_input_error() {
        let d = resolve_dir("bad");
        let data = d.join("a.rcsb").to_str().unwrap().to_string();
        std::fs::write(
            format!("{}.lockbox.json", data),
            "{\"locked_test\":{\"start\":200,\"end\":100}}",
        )
        .unwrap();
        let e = resolve(None, &data).unwrap_err();
        assert_eq!(e.exit, 3);
        assert!(e.message.contains("unreadable locked_test declaration"), "{}", e.message);
    }

    fn lockbox_date(y: i32, m: u32, d: u32) -> i64 {
        crate::ohlcv::days_from_civil(y, m, d) * 86_400
    }
}
