//! The 2026H2 crypto lockbox (issue #145).
//!
//! Every runner must refuse a test window that overlaps the lockbox unless the
//! caller explicitly overrides it. The window is a UTC half-open range; a
//! test window may abut the boundary but not cross it.
//!
//! ```text
//! 2026-07-01T00:00:00Z .. 2027-01-01T00:00:00Z
//! ```

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
}
