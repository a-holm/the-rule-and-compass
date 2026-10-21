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

fn date(ts: i64) -> String {
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
