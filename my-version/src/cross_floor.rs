//! #1274: the member floor of the cross-section columns.
//!
//! A cross-section statistic is `NaN` at a timestamp with fewer finite point-in-time members than
//! its floor. Mean, share, sd and sum-ratio columns (class M) use `--min-cross-members`; rank
//! columns and `avg_corr` (class R) use `--min-cross-members-rank`. Both default to 20. The values
//! are recorded per column in `<store>.enrich.json` and `<panel>.schema.json` so that columns built
//! with different floors are never mixed.

use std::collections::BTreeMap;

/// Floor both classes use when no flag is given.
pub const DEFAULT: usize = 20;
/// Smallest accepted floor: below it the sd, the rank grid and the correlation proxy carry no information.
pub const MIN: usize = 5;
pub const MAX: usize = 200;

/// Class M columns: mean, share, sd and sum-ratio statistics over the members.
pub const MARKET_COLUMNS: &[&str] = &[
    "mkt_r16",
    "mkt_ew_ret16",
    "breadth",
    "mkt_breadth",
    "dispersion",
    "mkt_dispersion",
    "rel_mkt16",
    "mkt_r96",
    "rel_mkt96",
    "mkt_vol96",
    "mkt_ew_ret1",
    "btc_volume_share",
    "mkt_r672",
    "mkt_trend_ma100_ratio",
    "mkt_trend_hi90_dist",
];

/// Class R columns: ranks and the average-correlation proxy.
pub const RANK_COLUMNS: &[&str] = &[
    "avg_corr",
    "xs_rank_r1",
    "xs_rank_r16",
    "xs_rank_r20",
    "xs_rank_vol20",
    "xs_rank_amihud96",
    "cs_rank_r60",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CrossFloor {
    /// Floor of the class M columns.
    pub members: usize,
    /// Floor of the class R columns.
    pub rank_members: usize,
}

impl Default for CrossFloor {
    fn default() -> Self {
        CrossFloor { members: DEFAULT, rank_members: DEFAULT }
    }
}

impl CrossFloor {
    /// The floor `name` reads, or `None` for a column outside the two classes.
    pub fn of(&self, name: &str) -> Option<usize> {
        if MARKET_COLUMNS.contains(&name) {
            Some(self.members)
        } else if RANK_COLUMNS.contains(&name) {
            Some(self.rank_members)
        } else {
            None
        }
    }

    pub fn json(&self) -> String {
        format!("{{\"members\":{},\"rank_members\":{}}}", self.members, self.rank_members)
    }

    pub fn describe(&self) -> String {
        format!("--min-cross-members {} --min-cross-members-rank {}", self.members, self.rank_members)
    }
}

/// True for a column whose value depends on a floor.
pub fn is_floored(name: &str) -> bool {
    MARKET_COLUMNS.contains(&name) || RANK_COLUMNS.contains(&name)
}

/// Parse the value of `flag`: an integer in `MIN..=MAX`.
pub fn parse_flag(flag: &str, value: &str) -> Result<usize, String> {
    match value.trim().parse::<usize>() {
        Ok(v) if (MIN..=MAX).contains(&v) => Ok(v),
        _ => Err(format!("{} needs an integer in {}..={}, got '{}'", flag, MIN, MAX, value)),
    }
}

fn int_after(obj: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\":", key);
    let rest = &obj[obj.find(&needle)? + needle.len()..];
    let digits: String = rest.trim_start().chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse().ok()
}

/// `min_members` of every column entry of an `.enrich.json` or `.schema.json` text.
pub fn column_floors(text: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    let head = "{\"name\":\"";
    let mut rest = text;
    while let Some(at) = rest.find(head) {
        let tail = &rest[at + head.len()..];
        let Some(end) = tail.find('"') else { break };
        let name = &tail[..end];
        let body_end = tail.find('}').unwrap_or(tail.len());
        if let Some(k) = int_after(&tail[..body_end], "min_members") {
            out.insert(name.to_string(), k);
        }
        rest = &tail[body_end.min(tail.len())..];
    }
    out
}

/// `text` with the registry phrase "N>=20 by default (<flags>)" replaced by "N>=<k>" when a floor is recorded.
pub fn pit_text(text: &str, floor: Option<usize>) -> String {
    let Some(k) = floor else { return text.to_string() };
    let head = "N>=20 by default (";
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(at) = rest.find(head) {
        out.push_str(&rest[..at]);
        let after = &rest[at + head.len()..];
        match after.find(')') {
            Some(end) => {
                out.push_str(&format!("N>={}", k));
                rest = &after[end + 1..];
            }
            None => {
                out.push_str(&rest[at..]);
                rest = "";
            }
        }
    }
    out.push_str(rest);
    out
}

/// The floors recorded next to a store: the top-level `cross_floor` of `<store>.enrich.json`.
pub fn recorded_floor(store_path: &str) -> Option<CrossFloor> {
    let text = std::fs::read_to_string(format!("{}.enrich.json", store_path)).ok()?;
    let obj = crate::manifest::object(&text, "cross_floor")?;
    Some(CrossFloor { members: int_after(obj, "members")?, rank_members: int_after(obj, "rank_members")? })
}

/// The one resolver of a cross column's floor, used by `enrich` and by `panel append` and the schema writers: its
/// per-column record in `<store>.schema.json`, else in `<store>.enrich.json` (which also lists the columns a run
/// carried over as `cross_columns`), else the top-level `cross_floor` of `<store>.enrich.json`, else 20.
pub fn resolved_columns(store_path: &str, names: &[String]) -> BTreeMap<String, usize> {
    let mut out = recorded_columns(store_path);
    let top = recorded_floor(store_path);
    for n in names.iter().filter(|n| is_floored(n)) {
        if let (false, Some(t)) = (out.contains_key(n), top) {
            out.insert(n.clone(), t.of(n).unwrap_or(DEFAULT));
        }
    }
    out
}

/// `--min-cross-members <k>` or `--min-cross-members-rank <k>`, whichever flag sets the floor of `name`.
pub fn flag_text(name: &str, k: usize) -> String {
    if MARKET_COLUMNS.contains(&name) {
        format!("--min-cross-members {}", k)
    } else {
        format!("--min-cross-members-rank {}", k)
    }
}

/// Per-column floors recorded next to a store, from its schema and enrich sidecars.
pub fn recorded_columns(store_path: &str) -> BTreeMap<String, usize> {
    let mut out = BTreeMap::new();
    for ext in ["enrich.json", "schema.json"] {
        if let Ok(text) = std::fs::read_to_string(format!("{}.{}", store_path, ext)) {
            out.extend(column_floors(&text));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flags_accept_only_the_range() {
        assert_eq!(parse_flag("--min-cross-members", "5"), Ok(5));
        assert_eq!(parse_flag("--min-cross-members", "200"), Ok(200));
        for bad in ["0", "4", "201", "x", "", "-5", "5.5"] {
            assert!(parse_flag("--min-cross-members", bad).unwrap_err().contains("5..=200"), "{bad}");
        }
    }

    #[test]
    fn column_floors_read_the_entries_that_carry_one() {
        let text = "{\"features\":[{\"name\":\"a\",\"x\":1},\n{\"name\":\"mkt_r16\",\"min_members\":5,\"sha256\":\"z\"},\n{\"name\":\"xs_rank_r1\",\"min_members\":12}]}";
        let m = column_floors(text);
        assert_eq!(m.len(), 2);
        assert_eq!(m["mkt_r16"], 5);
        assert_eq!(m["xs_rank_r1"], 12);
    }

    #[test]
    fn pit_text_states_the_recorded_floor() {
        let t = "PIT; N>=20 by default (--min-cross-members, --min-cross-members-rank); no forward fill";
        assert_eq!(pit_text(t, Some(5)), "PIT; N>=5; no forward fill");
        assert_eq!(pit_text(t, None), t);
        assert_eq!(pit_text("a, N>=20 by default (--x), b N>=20 by default (--y)", Some(7)), "a, N>=7, b N>=7");
    }

    #[test]
    fn the_two_classes_are_disjoint() {
        for n in MARKET_COLUMNS {
            assert!(!RANK_COLUMNS.contains(n), "{n}");
        }
        let f = CrossFloor { members: 5, rank_members: 9 };
        assert_eq!(f.of("breadth"), Some(5));
        assert_eq!(f.of("avg_corr"), Some(9));
        assert_eq!(f.of("btc_r1"), None);
    }
}
