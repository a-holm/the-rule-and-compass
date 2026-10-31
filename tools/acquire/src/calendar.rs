//! Months and UTC timestamps without a date library.

use std::fmt;
use std::time::{SystemTime, UNIX_EPOCH};

/// A calendar month.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Month {
    pub year: i32,
    pub month: u32,
}

impl fmt::Display for Month {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:04}-{:02}", self.year, self.month)
    }
}

impl Month {
    /// Parses `YYYY-MM`.
    pub fn parse(s: &str) -> Result<Month, String> {
        Err(format!("unimplemented {s}"))
    }

    pub fn next(self) -> Month {
        if self.month == 12 {
            Month { year: self.year + 1, month: 1 }
        } else {
            Month { year: self.year, month: self.month + 1 }
        }
    }

    /// Last calendar day as `YYYY-MM-DD`.
    pub fn last_day(self) -> String {
        let leap = (self.year % 4 == 0 && self.year % 100 != 0) || self.year % 400 == 0;
        let day = match self.month {
            1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
            4 | 6 | 9 | 11 => 30,
            _ if leap => 29,
            _ => 28,
        };
        format!("{self}-{day:02}")
    }
}

/// Months from `from` to `to`, both inclusive. Empty when `from > to`.
pub fn range(from: Month, to: Month) -> Vec<Month> {
    let _ = (from, to);
    Vec::new()
}

/// Rejects a range that reaches the current month or later: the archive publishes a
/// month only after it has ended.
pub fn check_published(to: Month, now: Month) -> Result<(), String> {
    if to >= now {
        Err(format!("month {to} is not published yet; the newest candidate is before {now}"))
    } else {
        Ok(())
    }
}

pub fn now_unix() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// `YYYY-MM-DDTHH:MM:SSZ` for Unix seconds.
pub fn iso_from_unix(secs: i64) -> String {
    let _ = secs;
    String::new()
}

pub fn now_iso() -> String {
    iso_from_unix(now_unix())
}

pub fn current_month() -> Month {
    let (y, m, _) = civil_from_days(now_unix().div_euclid(86_400));
    Month { year: y, month: m }
}

fn civil_from_days(z: i64) -> (i32, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    ((y + i64::from(m <= 2)) as i32, m, d)
}
