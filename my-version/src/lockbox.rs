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

    /// Refuse a row set with a timestamp whose label window reaches the
    /// declared range: any stamp in `[start - purge, end)`. The error names the
    /// first offending row, the split side and the fix.
    pub fn check_times(&self, side: &str, time_name: &str, times: &[i64], purge: i64) -> Result<Overlap, String> {
        let overlap = self.check_range(&self.lockbox, self.origin(), side, time_name, times, purge)?;
        if let Some(also) = &self.also {
            self.check_range(also, "protocol default, opt out with --lockbox none", side, time_name, times, purge)?;
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
        purge: i64,
    ) -> Result<Overlap, String> {
        let purge = purge.max(0);
        let reach = Lockbox {
            start: lockbox.start - purge,
            end: lockbox.end,
        };
        let overlap = reach.overlap(times.iter().copied());
        if overlap.overlaps() {
            let subject = if purge == 0 {
                format!("{} row(s) in the declared locked test", group_digits(overlap.rows))
            } else {
                format!(
                    "{} row(s) whose label window reaches the declared locked test",
                    group_digits(overlap.rows)
                )
            };
            let label_end = if purge == 0 {
                String::new()
            } else {
                format!("; label end = stamp + {} s", purge)
            };
            return Err(format!(
                "{} has {} {}..{} ({}{}; first row {}). \
                 Exclude them: --exclude '{} >= {}', or read the locked test only with \
                 rcsr walkforward --locked-test or rcsr ladder --locked-test",
                side,
                subject,
                date(lockbox.start),
                date(lockbox.end),
                origin,
                label_end,
                crate::protocol::iso(overlap.first_ts as f64),
                time_name,
                reach.start
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

    /// Read a `"locked_test_also":{...}` object (a second guarded range).
    pub fn also_from_declaration(text: &str) -> Option<Lockbox> {
        let obj = crate::manifest::object(text, "locked_test_also")?;
        let (start, end) = (int_field(obj, "start")?, int_field(obj, "end")?);
        (end > start).then_some(Lockbox { start, end })
    }

    /// Read a `"locked_test":{...}` object from sidecar or manifest text.
    pub fn from_declaration(text: &str) -> Option<Lockbox> {
        let obj = crate::manifest::object(text, "locked_test")?;
        let (start, end) = (int_field(obj, "start")?, int_field(obj, "end")?);
        (end > start).then_some(Lockbox { start, end })
    }
}

/// An integer JSON field `"key":N`.
pub fn int_field(text: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{}\":", key);
    let rest = &text[text.find(&needle)? + needle.len()..];
    let end = rest
        .char_indices()
        .find(|(i, c)| !(c.is_ascii_digit() || (*i == 0 && *c == '-')))
        .map(|(i, _)| i)
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

/// Sidecar that carries a declaration for a CSV (or any file without a
/// manifest of its own).
pub fn sidecar_path(data_path: &str) -> String {
    format!("{}.lockbox.json", data_path)
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

    /// The range a derived sidecar declares as `locked_test`: the union when
    /// the two ranges overlap or touch. When they are disjoint, a flag on
    /// undeclared data declares the protocol default first; a stored pair
    /// keeps its roles.
    pub fn sidecar_range(&self) -> Option<Lockbox> {
        match (self.range, self.also) {
            (Some(f), Some(d)) if f.start <= d.end && f.end >= d.start => {
                Some(Lockbox { start: f.start.min(d.start), end: f.end.max(d.end) })
            }
            (Some(_), Some(d)) if self.source == Source::FlagAndDefault => Some(d),
            (r, _) => r,
        }
    }

    /// The guarded range `sidecar_range` leaves out when the two are disjoint.
    pub fn sidecar_also(&self) -> Option<Lockbox> {
        match (self.range, self.also) {
            (Some(f), Some(d)) if !(f.start <= d.end && f.end >= d.start) => {
                if self.source == Source::FlagAndDefault {
                    Some(f)
                } else {
                    Some(d)
                }
            }
            _ => None,
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

/// Largest file a sidecar hashes; bigger files are bound by their byte count only.
const SIDECAR_HASH_LIMIT: u64 = 2 << 30;

/// Check a `.lockbox.json` that records the bytes and SHA-256 of the file it
/// declares. A sidecar without the fields (hand-written) is not checked.
fn check_sidecar_binding(data_path: &str, file: &str, text: &str) -> Result<(), ResolveError> {
    let Some(want_bytes) = int_field(text, "data_bytes") else {
        return Ok(());
    };
    let Ok(meta) = std::fs::metadata(data_path) else {
        return Ok(());
    };
    let want_sha = crate::manifest::string_field(text, "data_sha256");
    let stale = |sha: &str| {
        input(format!(
            "stale {}: it declares {} with sha256 {}, the file has {}; \
             remove it or write it again with the data",
            file,
            data_path,
            want_sha.as_deref().unwrap_or("unrecorded"),
            sha
        ))
    };
    if meta.len() as i64 != want_bytes {
        return Err(stale(&format!("{} bytes (the sidecar records {})", meta.len(), want_bytes)));
    }
    if let (Some(want), true) = (&want_sha, meta.len() <= SIDECAR_HASH_LIMIT) {
        let have = crate::manifest::sha256_file(data_path).map_err(input)?;
        if &have != want {
            return Err(stale(&have));
        }
    }
    Ok(())
}

/// Every locked-test declaration of `data_path`, in carrier order. A carrier
/// that has a `locked_test` object which does not parse is an input error, and
/// so is a `.lockbox.json` that no longer matches its data file.
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
        if ext == "lockbox.json" {
            check_sidecar_binding(data_path, &file, &text)?;
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

/// The ranges every declaration of `data_path` agrees on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stored {
    /// The first file that declares them.
    pub file: String,
    pub range: Lockbox,
    /// A second guarded range (`locked_test_also`).
    pub also: Option<Lockbox>,
}

impl Stored {
    pub fn describe(&self) -> String {
        match &self.also {
            Some(a) => format!("{} + {}", span(&self.range), span(a)),
            None => span(&self.range),
        }
    }
}

/// The ranges every declaration of `data_path` agrees on. Two files that
/// disagree are an input error.
pub fn stored_all(data_path: &str) -> Result<Option<Stored>, ResolveError> {
    let declared_in = declarations(data_path)?;
    let with_also: Vec<Stored> = declared_in
        .into_iter()
        .map(|(file, range)| {
            let also = std::fs::read_to_string(&file).ok().and_then(|t| Lockbox::also_from_declaration(&t));
            Stored { file, range, also }
        })
        .collect();
    if let Some(first) = with_also.first() {
        if let Some(other) = with_also.iter().find(|s| s.range != first.range || s.also != first.also) {
            return Err(input(format!(
                "{}: conflicting locked-test declarations: {} {}, {} {}; remove the stale file",
                data_path,
                first.file,
                first.describe(),
                other.file,
                other.describe()
            )));
        }
    }
    Ok(with_also.into_iter().next())
}

/// The primary range every declaration of `data_path` agrees on, with the first
/// file that declares it.
pub fn stored(data_path: &str) -> Result<Option<(String, Lockbox)>, ResolveError> {
    Ok(stored_all(data_path)?.map(|s| (s.file, s.range)))
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
    let stored = stored_all(data_path)?;
    let declared_in = declarations(data_path)?;
    let covers = |f: &Lockbox, l: &Lockbox| f.start <= l.start && f.end >= l.end;
    match (explicit, stored) {
        (Some(None), Some(s)) => Err(usage(format!(
            "--lockbox none is refused: {} declares the locked test {}",
            s.file,
            s.describe()
        ))),
        (Some(None), None) => Ok(Resolved { range: None, source: Source::None, also: None, declared_in }),
        (Some(Some(f)), Some(s)) if !(covers(&f, &s.range) && s.also.map_or(true, |a| covers(&f, &a))) => Err(usage(format!(
            "--lockbox {} does not cover the declared locked test {} ({})",
            span(&f),
            s.describe(),
            s.file
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
        (None, Some(s)) => Ok(Resolved { range: Some(s.range), source: Source::Store, also: s.also, declared_in }),
        (None, None) => Ok(Resolved {
            range: Some(protocol_default()),
            source: Source::ProtocolDefault,
            also: None,
            declared_in,
        }),
    }
}

/// Write the CSV sidecar for a declaration. It records the bytes and SHA-256 of
/// `data_path`, so the declaration cannot outlive a rewrite of the file.
pub fn write_sidecar(data_path: &str, lockbox: &Lockbox, source: &str) -> Result<(), String> {
    write_sidecar_with(data_path, lockbox, source, "")
}

/// `write_sidecar` with extra top-level JSON members (each starting with a comma).
pub fn write_sidecar_with(data_path: &str, lockbox: &Lockbox, source: &str, extra: &str) -> Result<(), String> {
    let p = sidecar_path(data_path);
    let binding = match std::fs::metadata(data_path) {
        Ok(meta) => {
            let sha = if meta.len() <= SIDECAR_HASH_LIMIT {
                format!(",\"data_sha256\":{}", jstr(&crate::manifest::sha256_file(data_path)?))
            } else {
                String::new()
            };
            format!(",\"data_bytes\":{}{}", meta.len(), sha)
        }
        Err(_) => String::new(),
    };
    crate::atomic::write_text_atomic(
        &p,
        &format!("{{\"locked_test\":{}{}{}}}\n", lockbox.declaration_json(source), extra, binding),
    )
    .map_err(|e| format!("{}: {}", p, e))
}

/// What a writer's `--locked-test` said about its output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Keep {
    /// The flag was not given.
    Unset,
    /// `--locked-test none`: drop the declaration.
    Remove,
    /// `--locked-test START,END`.
    Range(Lockbox),
}

impl Keep {
    pub fn parse(s: &str) -> Result<Keep, String> {
        if s.trim().eq_ignore_ascii_case("none") {
            Ok(Keep::Remove)
        } else {
            Lockbox::parse(s).map(Keep::Range)
        }
    }
}

/// The locked test a writer gives its output `out`, built from `input`.
///
/// A writer never drops or changes a declaration silently: when `out` or
/// `input` declares a range, the writer repeats it with `--locked-test START,END`
/// or removes it with `--locked-test none` (refused while `input` declares it).
/// A flag may widen the range of `input`, never narrow it. When `input` declares
/// and the flag is absent, the output inherits the range.
pub fn output_declaration(out: &str, input: Option<&str>, keep: Keep) -> Result<Option<Lockbox>, ResolveError> {
    let old = stored(out)?;
    let inherited = match input {
        Some(p) if p != out => stored(p)?,
        _ => None,
    };
    let refuse = |file: &str, l: &Lockbox, why: &str| {
        usage(format!(
            "{} declares the locked test {} ({}); {}; pass --locked-test {},{} to keep it or --locked-test none to remove it",
            out,
            span(l),
            file,
            why,
            date(l.start),
            date(l.end)
        ))
    };
    match keep {
        Keep::Remove => match inherited {
            Some((file, l)) => Err(usage(format!(
                "--locked-test none is refused: {} declares the locked test {}",
                file,
                span(&l)
            ))),
            None => Ok(None),
        },
        Keep::Range(r) => {
            if let Some((file, l)) = &inherited {
                if !(r.start <= l.start && r.end >= l.end) {
                    return Err(usage(format!(
                        "--locked-test {} does not cover the locked test {} declared by {}",
                        span(&r),
                        span(l),
                        file
                    )));
                }
            }
            match old {
                Some((file, o)) if o != r => Err(refuse(&file, &o, "the run would change it")),
                _ => Ok(Some(r)),
            }
        }
        Keep::Unset => match (old, inherited) {
            (Some((_, o)), Some((_, i))) if o == i => Ok(Some(i)),
            (Some((file, o)), _) => Err(refuse(&file, &o, "the run declares none")),
            (None, Some((_, i))) => Ok(Some(i)),
            (None, None) => Ok(None),
        },
    }
}

/// After a writer replaced `out`: delete the `.lockbox.json` sidecar of the old
/// bytes (when the writer did not write it itself) and refuse any other carrier
/// the writer left that declares another range than `declared`.
pub fn finish_output(out: &str, declared: Option<&Lockbox>, wrote: &[&str]) -> Result<(), ResolveError> {
    for ext in CARRIERS {
        if wrote.contains(&ext) {
            continue;
        }
        let file = format!("{}.{}", out, ext);
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        let Some(old) = Lockbox::from_declaration(&text) else {
            continue;
        };
        if ext == "lockbox.json" {
            std::fs::remove_file(&file).map_err(|e| input(format!("{}: {}", file, e)))?;
        } else if declared != Some(&old) {
            return Err(input(format!(
                "stale {}: it declares the locked test {} but {} now declares {}; remove it",
                file,
                span(&old),
                out,
                declared.map_or("none".to_string(), span)
            )));
        }
    }
    Ok(())
}

/// The second range the input declares, when `declared` is the range it declares.
pub fn inherited_also(input: &str, declared: Option<&Lockbox>) -> Result<Option<Lockbox>, String> {
    Ok(stored_all(input)
        .map_err(|e| e.message)?
        .filter(|s| Some(&s.range) == declared)
        .and_then(|s| s.also))
}

/// The `,"locked_test_also":{...}` member for a second guarded range, or nothing.
pub fn also_json(also: Option<&Lockbox>, source: &str) -> String {
    also.map(|a| format!(",\"locked_test_also\":{}", a.declaration_json(source))).unwrap_or_default()
}

/// Put a `"locked_test"` object (and a second range) first in a JSON object text.
pub fn insert_declaration(text: &str, lockbox: &Lockbox, also: Option<&Lockbox>, source: &str) -> String {
    let at = text.find('{').map_or(0, |i| i + 1);
    let mut out = text.to_string();
    out.insert_str(at, &format!("\"locked_test\":{}{},", lockbox.declaration_json(source), also_json(also, source)));
    out
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
    fn declared_check_with_a_purge_refuses_the_label_window_that_reaches_the_range() {
        let d = Declared {
            lockbox: Lockbox { start: 1_000_000, end: 2_000_000 },
            source: "store",
            also: None,
        };
        let purge = 7_200;
        let e = d.check_times("test split", "ts", &[1_000_000 - purge], purge).unwrap_err();
        assert!(e.contains("1 row(s) whose label window reaches the declared locked test"), "{}", e);
        assert!(e.contains("label end = stamp + 7200 s"), "{}", e);
        assert!(e.contains("--exclude 'ts >= 992800'"), "{}", e);
        assert!(d.check_times("test split", "ts", &[1_000_000 - purge - 1], purge).is_ok());
        assert!(d.check_times("test split", "ts", &[2_000_000], purge).is_ok());
        assert!(d.check_times("test split", "ts", &[1_000_000 - 1], 0).is_ok());
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
    fn a_stored_disjoint_pair_keeps_its_roles_in_a_derived_sidecar() {
        let first = stored();
        let second = Lockbox { start: lockbox_date(2024, 9, 1), end: lockbox_date(2024, 10, 1) };
        let store = Resolved { range: Some(first), source: Source::Store, also: Some(second), declared_in: Vec::new() };
        assert_eq!((store.sidecar_range(), store.sidecar_also()), (Some(first), Some(second)));
        let flag = Resolved { range: Some(second), source: Source::FlagAndDefault, also: Some(first), declared_in: Vec::new() };
        assert_eq!((flag.sidecar_range(), flag.sidecar_also()), (Some(first), Some(second)));
        let touching = Resolved { range: Some(first), source: Source::Store, also: Some(Lockbox { start: first.end, end: second.end }), declared_in: Vec::new() };
        assert_eq!(touching.sidecar_range(), Some(Lockbox { start: first.start, end: second.end }));
        assert_eq!(touching.sidecar_also(), None);
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
