//! #1173: interval-list splits. A splits table (`split_id,role,start,end`,
//! UTC half-open intervals) assigns time ranges to training, validation and
//! test. Rows next to a test interval are purged (their label window reaches
//! it) or embargoed (their features read its bars); training rows next to a
//! validation interval are purged on both sides. The same predicate decides
//! whether a later scoring window is out of sample for a run trained on
//! interval lists.

/// Header of a splits table.
pub const TABLE_HEADER: &str = "split_id,role,start,end";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    Train,
    Validation,
    Test,
}

impl Role {
    pub fn as_str(self) -> &'static str {
        match self {
            Role::Train => "train",
            Role::Validation => "validation",
            Role::Test => "test",
        }
    }

    fn parse(s: &str) -> Option<Role> {
        match s {
            "train" => Some(Role::Train),
            "validation" => Some(Role::Validation),
            "test" => Some(Role::Test),
            _ => None,
        }
    }
}

/// Where the embargo after a test interval came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EmbargoSource {
    /// `--embargo <seconds>`, at least the automatic value.
    Flag,
    /// The longest lookback of the columns the run reads.
    Lookback,
    /// The label purge (no column reads further back).
    Purge,
}

impl EmbargoSource {
    pub fn as_str(self) -> &'static str {
        match self {
            EmbargoSource::Flag => "flag",
            EmbargoSource::Lookback => "lookback",
            EmbargoSource::Purge => "purge",
        }
    }

    pub fn parse(s: &str) -> Option<EmbargoSource> {
        match s {
            "flag" => Some(EmbargoSource::Flag),
            "lookback" => Some(EmbargoSource::Lookback),
            "purge" => Some(EmbargoSource::Purge),
            _ => None,
        }
    }
}

/// The intervals of one split, sorted by start within each role.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Declared {
    pub split_id: String,
    pub train: Vec<(i64, i64)>,
    pub validation: Vec<(i64, i64)>,
    pub test: Vec<(i64, i64)>,
}

/// True when a table is a splits table (its first non-comment line is the header).
pub fn is_table(text: &str) -> bool {
    text.lines()
        .map(str::trim)
        .find(|l| !l.is_empty() && !l.starts_with('#'))
        .is_some_and(|l| l.replace(' ', "") == TABLE_HEADER)
}

fn stamp(s: &str, line: usize, what: &str) -> Result<i64, String> {
    let v = crate::data::parse_time(s)
        .ok_or_else(|| format!("--split-table line {}: cannot parse {} '{}'", line, what, s))?;
    if !v.is_finite() || v.fract() != 0.0 {
        return Err(format!("--split-table line {}: {} '{}' is not a whole second", line, what, s));
    }
    Ok(v as i64)
}

/// Parse a splits table and pick the split `split_id`. Overlapping intervals
/// (of one role or of two) and an empty interval are refused; touching
/// intervals are allowed. A split needs at least one training interval; a
/// split without a test interval is a final all-data run.
pub fn parse_table(text: &str, split_id: &str) -> Result<Declared, String> {
    let mut header_seen = false;
    let mut ids: Vec<String> = Vec::new();
    let mut out = Declared { split_id: split_id.to_string(), ..Declared::default() };
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !header_seen {
            if line.replace(' ', "") != TABLE_HEADER {
                return Err(format!("--split-table: the header must be '{}', found '{}'", TABLE_HEADER, line));
            }
            header_seen = true;
            continue;
        }
        let f: Vec<&str> = line.split(',').map(str::trim).collect();
        if f.len() != 4 {
            return Err(format!("--split-table line {}: expected 4 fields (split_id,role,start,end), found {}", n + 1, f.len()));
        }
        if !ids.iter().any(|i| i == f[0]) {
            ids.push(f[0].to_string());
        }
        let role = Role::parse(f[1])
            .ok_or_else(|| format!("--split-table line {}: role '{}' is not train, validation or test", n + 1, f[1]))?;
        let start = stamp(f[2], n + 1, "start")?;
        let end = stamp(f[3], n + 1, "end")?;
        if end <= start {
            return Err(format!("--split-table line {}: end {} is not after start {}", n + 1, f[3], f[2]));
        }
        if f[0] != split_id {
            continue;
        }
        match role {
            Role::Train => out.train.push((start, end)),
            Role::Validation => out.validation.push((start, end)),
            Role::Test => out.test.push((start, end)),
        }
    }
    if !header_seen {
        return Err(format!("--split-table: empty table (header '{}')", TABLE_HEADER));
    }
    if out.train.is_empty() && out.validation.is_empty() && out.test.is_empty() {
        return Err(format!("--split-table: no split '{}' (the table holds {})", split_id, ids.join(", ")));
    }
    if out.train.is_empty() {
        return Err(format!("--split-table: split '{}' has no training interval", split_id));
    }
    for v in [&mut out.train, &mut out.validation, &mut out.test] {
        v.sort_unstable();
    }
    let mut all: Vec<(i64, i64, Role)> = Vec::new();
    for (role, v) in [(Role::Train, &out.train), (Role::Validation, &out.validation), (Role::Test, &out.test)] {
        all.extend(v.iter().map(|&(a, b)| (a, b, role)));
    }
    all.sort_unstable_by_key(|&(a, b, _)| (a, b));
    for w in all.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(format!(
                "--split-table: split '{}': the {} interval {}..{} overlaps the {} interval {}..{}",
                split_id,
                w[0].2.as_str(),
                iso(w[0].0),
                iso(w[0].1),
                w[1].2.as_str(),
                iso(w[1].0),
                iso(w[1].1)
            ));
        }
    }
    Ok(out)
}

fn iso(t: i64) -> String {
    crate::protocol::iso(t as f64)
}

/// The label window `[t, t + purge)` of a row stamped `t` meets `[start, end)`,
/// or `t` lies in `[end, end + embargo)` ([`crate::lockbox::Lockbox::reaches`]).
pub fn reaches(start: i64, end: i64, t: i64, purge: i64, embargo: i64) -> bool {
    t + purge.max(1) > start && t < end + embargo.max(0)
}

/// Some stamp of the half-open `[a, b)` reaches `[start, end)`.
pub fn interval_reaches(start: i64, end: i64, a: i64, b: i64, purge: i64, embargo: i64) -> bool {
    a < b && a < end + embargo.max(0) && (b - 1) + purge.max(1) > start
}

/// A resolved split: the intervals, the label purge and the embargo after
/// every test interval, in seconds.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntervalSplit {
    pub declared: Declared,
    pub purge: i64,
    pub embargo: i64,
    pub embargo_source: EmbargoSource,
    /// The column whose lookback set the embargo, when `embargo_source` is `lookback`.
    pub embargo_column: Option<String>,
}

fn inside(v: &[(i64, i64)], t: i64) -> bool {
    let p = v.partition_point(|&(a, _)| a <= t);
    p > 0 && t < v[p - 1].1
}

impl IntervalSplit {
    pub fn new(declared: Declared, purge: i64, embargo: i64, embargo_source: EmbargoSource, embargo_column: Option<String>) -> IntervalSplit {
        IntervalSplit { declared, purge: purge.max(0), embargo: embargo.max(0), embargo_source, embargo_column }
    }

    /// Role of a row stamped `t`: `None` when it lies outside every interval,
    /// when its label window reaches a test interval, when it lies within the
    /// embargo after one, or (training rows) when its label window overlaps a
    /// validation interval on either side.
    pub fn role(&self, t: i64) -> Option<Role> {
        let d = &self.declared;
        if inside(&d.test, t) {
            return Some(Role::Test);
        }
        let role = if inside(&d.train, t) {
            Role::Train
        } else if inside(&d.validation, t) {
            Role::Validation
        } else {
            return None;
        };
        if d.test.iter().any(|&(s, e)| reaches(s, e, t, self.purge, self.embargo)) {
            return None;
        }
        if role == Role::Train && d.validation.iter().any(|&(s, e)| reaches(s, e, t, self.purge, self.purge)) {
            return None;
        }
        Some(role)
    }

    /// The parts of the intervals of `role` whose rows keep that role, sorted
    /// and disjoint. Test intervals are returned whole.
    pub fn effective(&self, role: Role) -> Vec<(i64, i64)> {
        let d = &self.declared;
        let own = match role {
            Role::Train => &d.train,
            Role::Validation => &d.validation,
            Role::Test => return d.test.clone(),
        };
        let p1 = self.purge.max(1);
        let mut zones: Vec<(i64, i64)> = d.test.iter().map(|&(s, e)| (s - p1 + 1, e + self.embargo)).collect();
        if role == Role::Train {
            zones.extend(d.validation.iter().map(|&(s, e)| (s - p1 + 1, e + self.purge)));
        }
        zones.sort_unstable();
        let mut out = Vec::new();
        for &(a, b) in own {
            let mut pieces = vec![(a, b)];
            for &(za, zb) in &zones {
                let mut next = Vec::new();
                for (pa, pb) in pieces {
                    if zb <= pa || za >= pb {
                        next.push((pa, pb));
                        continue;
                    }
                    if pa < za {
                        next.push((pa, za));
                    }
                    if zb < pb {
                        next.push((zb, pb));
                    }
                }
                pieces = next;
            }
            out.extend(pieces);
        }
        out.sort_unstable();
        out
    }

    /// Starts of the maximal contiguous training intervals (touching table
    /// intervals merge): every one starts a book segment under `--fitness pnl`.
    pub fn segment_starts(&self) -> Vec<f64> {
        let mut out: Vec<f64> = Vec::new();
        let mut last_end: Option<i64> = None;
        for &(a, b) in &self.declared.train {
            if last_end != Some(a) {
                out.push(a as f64);
            }
            last_end = Some(b);
        }
        out
    }

    /// Validation interval starts, the cuts of `--select-blocks intervals`.
    pub fn validation_starts(&self) -> Vec<f64> {
        self.declared.validation.iter().map(|&(a, _)| a as f64).collect()
    }

    /// The table intervals that reach a guarded range `[start, end)` with the
    /// label purge before it and `embargo` after it, as `role start..end`.
    pub fn reaching(&self, start: i64, end: i64, purge: i64, embargo: i64) -> Vec<String> {
        let d = &self.declared;
        let mut out = Vec::new();
        for (role, v) in [(Role::Train, &d.train), (Role::Validation, &d.validation), (Role::Test, &d.test)] {
            for &(a, b) in v {
                if interval_reaches(start, end, a, b, purge, embargo) {
                    out.push(format!("{} {}..{}", role.as_str(), iso(a), iso(b)));
                }
            }
        }
        out
    }

    /// `split` members of run.json, with a leading comma.
    pub fn json_fields(&self, table_sha256: &str) -> String {
        let list = |v: &[(i64, i64)]| -> String {
            let items: Vec<String> = v.iter().map(|(a, b)| format!("[{},{}]", a, b)).collect();
            format!("[{}]", items.join(","))
        };
        let d = &self.declared;
        format!(
            ",\"split_id\":{},\"split_table_sha256\":{},\"intervals\":{{\"train\":{},\"validation\":{},\"test\":{}}},\
             \"effective_intervals\":{{\"train\":{},\"validation\":{}}},\"interval_purge_seconds\":{},\"embargo_seconds\":{},\
             \"embargo_source\":{},\"embargo_column\":{}",
            jstr(&d.split_id),
            jstr(table_sha256),
            list(&d.train),
            list(&d.validation),
            list(&d.test),
            list(&self.effective(Role::Train)),
            list(&self.effective(Role::Validation)),
            self.purge,
            self.embargo,
            jstr(self.embargo_source.as_str()),
            self.embargo_column.as_deref().map_or("null".to_string(), jstr)
        )
    }

    /// A fingerprint of the resolved intervals, purge and embargo, for checkpoints.
    pub fn fingerprint(&self) -> String {
        let d = &self.declared;
        let mut text = format!("{}|{}|{}", d.split_id, self.purge, self.embargo);
        for (tag, v) in [("t", &d.train), ("v", &d.validation), ("x", &d.test)] {
            for (a, b) in v {
                text.push_str(&format!("|{}{}:{}", tag, a, b));
            }
        }
        crate::manifest::sha256_hex(text.as_bytes())
    }
}

/// Out-of-sample test of a scoring window `[start, end)` against the kept
/// training and validation intervals of a run: in sample when some kept row
/// stamp reaches the window with the run's purge and embargo.
pub fn window_in_sample(kept: &[(i64, i64)], start: i64, end: i64, purge: i64, embargo: i64) -> bool {
    kept.iter().any(|&(a, b)| interval_reaches(start, end, a, b, purge, embargo))
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

/// Modal positive step between consecutive stamps of one group (ties go to the
/// shorter step): the bar of the data. `None` without two stamps in a group.
pub fn modal_step(times: &[f64], groups: Option<&[usize]>) -> Option<i64> {
    let mut idx: Vec<usize> = (0..times.len()).filter(|&i| times[i].is_finite()).collect();
    let g = |i: usize| groups.map_or(0, |g| g[i]);
    idx.sort_by(|&a, &b| g(a).cmp(&g(b)).then(times[a].total_cmp(&times[b])));
    let mut hist: std::collections::BTreeMap<i64, u64> = std::collections::BTreeMap::new();
    for w in idx.windows(2) {
        if g(w[0]) == g(w[1]) {
            let d = (times[w[1]] - times[w[0]]).round() as i64;
            if d > 0 {
                *hist.entry(d).or_insert(0) += 1;
            }
        }
    }
    hist.into_iter().max_by(|a, b| a.1.cmp(&b.1).then(b.0.cmp(&a.0))).map(|(d, _)| d)
}

/// Lookback of one column in seconds under rule `rcsr-lookback/1`: a
/// `<name>_lag<k>` column adds `k` bars to its base column; a name without a
/// rule counts as the protocol default lookback embargo.
pub fn column_lookback(name: &str, bar_seconds: i64) -> i64 {
    if let Some(l) = crate::enrich::lookback_seconds(name, bar_seconds) {
        return l;
    }
    if let Some((stem, k)) = name.rsplit_once("_lag") {
        if let (Ok(k), Some(l)) = (k.parse::<i64>(), crate::enrich::lookback_seconds(stem, bar_seconds)) {
            return l + k * bar_seconds;
        }
    }
    crate::protocol::DEFAULT_LOCKBOX_EMBARGO_SECONDS
}

/// `--embargo auto` (`flag = None`) or an explicit value: the embargo is the
/// larger of the label purge and the longest column lookback; an explicit
/// value below it is refused.
pub fn resolve_embargo(flag: Option<i64>, purge: i64, columns: &[String], bar_seconds: i64) -> Result<(i64, EmbargoSource, Option<String>), String> {
    let mut best: Option<(i64, &str)> = None;
    for c in columns {
        let l = column_lookback(c, bar_seconds);
        if best.is_none_or(|(b, _)| l > b) {
            best = Some((l, c));
        }
    }
    let (auto, source, column) = match best {
        Some((l, c)) if l > purge => (l, EmbargoSource::Lookback, Some(c.to_string())),
        _ => (purge, EmbargoSource::Purge, None),
    };
    match flag {
        None => Ok((auto, source, column)),
        Some(v) if v >= auto => Ok((v, EmbargoSource::Flag, None)),
        Some(v) => Err(format!(
            "--embargo {} is below the automatic embargo {} s ({}): rows after a test interval would read its bars \
             or share its labels; pass at least {} or --embargo auto",
            v,
            auto,
            match &column {
                Some(c) => format!("lookback of column '{}' at a {} s bar", c, bar_seconds),
                None => "the label purge".to_string(),
            },
            auto
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: i64 = 3600;
    const D: i64 = 86_400;

    fn split(train: &[(i64, i64)], validation: &[(i64, i64)], test: &[(i64, i64)], purge: i64, embargo: i64) -> IntervalSplit {
        let d = Declared { split_id: "s".to_string(), train: train.to_vec(), validation: validation.to_vec(), test: test.to_vec() };
        IntervalSplit::new(d, purge, embargo, EmbargoSource::Flag, None)
    }

    #[test]
    fn table_parses_one_split_and_refuses_overlaps() {
        let text = "split_id,role,start,end\n\
                    s02,train,2018-01-01T00:00:00Z,2019-03-01T00:00:00Z\n\
                    s02,validation,2019-03-01T00:00:00Z,2019-05-01T00:00:00Z\n\
                    s02,test,2019-06-01T00:00:00Z,2020-09-01T00:00:00Z\n\
                    s02,train,2022-01-01T00:00:00Z,2023-10-01T00:00:00Z\n\
                    s03,test,2018-01-01T00:00:00Z,2019-01-01T00:00:00Z\n";
        assert!(is_table(text));
        let d = parse_table(text, "s02").unwrap();
        assert_eq!(d.train, vec![(1_514_764_800, 1_551_398_400), (1_640_995_200, 1_696_118_400)]);
        assert_eq!(d.validation.len(), 1);
        assert_eq!(d.test, vec![(1_559_347_200, 1_598_918_400)]);
        let e = parse_table(text, "s09").unwrap_err();
        assert!(e.contains("no split 's09'") && e.contains("s02, s03"), "{e}");
        let e = parse_table(text, "s03").unwrap_err();
        assert!(e.contains("no training interval"), "{e}");
        let bad = "split_id,role,start,end\na,train,0,100\na,test,50,200\n";
        let e = parse_table(bad, "a").unwrap_err();
        assert!(e.contains("overlaps"), "{e}");
        let bad = "split_id,role,start,end\na,train,0,100\na,train,99,200\n";
        assert!(parse_table(bad, "a").unwrap_err().contains("overlaps"));
        let touch = "split_id,role,start,end\na,train,0,100\na,validation,100,200\n";
        assert!(parse_table(touch, "a").is_ok());
        assert!(parse_table("split_id,role,start,end\na,train,5,5\n", "a").unwrap_err().contains("not after"));
        assert!(parse_table("split_id,role,start,end\na,fit,0,5\n", "a").unwrap_err().contains("role 'fit'"));
    }

    #[test]
    fn role_purges_before_and_embargoes_after_a_test_interval() {
        let s = split(&[(0, 100), (200, 400)], &[], &[(100, 200)], 10, 30);
        assert_eq!(s.role(90), Some(Role::Train));
        assert_eq!(s.role(91), None, "label [91, 101) reaches the test start");
        assert_eq!(s.role(100), Some(Role::Test));
        assert_eq!(s.role(199), Some(Role::Test));
        assert_eq!(s.role(229), None, "inside the embargo after the test end");
        assert_eq!(s.role(230), Some(Role::Train));
        assert_eq!(s.role(400), None, "outside every interval");
        assert_eq!(s.effective(Role::Train), vec![(0, 91), (230, 400)]);
    }

    #[test]
    fn training_is_purged_on_both_sides_of_a_validation_interval() {
        let s = split(&[(0, 100), (200, 300)], &[(100, 200)], &[], 10, 30);
        assert_eq!(s.role(90), Some(Role::Train));
        assert_eq!(s.role(91), None);
        assert_eq!(s.role(150), Some(Role::Validation));
        assert_eq!(s.role(209), None, "within the purge after the validation end");
        assert_eq!(s.role(210), Some(Role::Train));
        assert_eq!(s.effective(Role::Train), vec![(0, 91), (210, 300)]);
        assert_eq!(s.segment_starts(), vec![0.0, 200.0]);
    }

    #[test]
    fn touching_training_intervals_are_one_segment() {
        let s = split(&[(0, 100), (100, 200), (300, 400)], &[], &[(200, 300)], 1, 1);
        assert_eq!(s.segment_starts(), vec![0.0, 300.0]);
    }

    #[test]
    fn the_guard_allows_a_cpcv_test_group_and_refuses_an_overlap() {
        let s = split(&[(0, 1000), (2000, 3000)], &[], &[(1000, 2000)], 10, 300);
        let kept = s.effective(Role::Train);
        assert!(!window_in_sample(&kept, 1000, 2000, 10, 300), "{kept:?}");
        assert!(window_in_sample(&kept, 999, 2000, 10, 300), "one second earlier meets the last training label");
        assert!(window_in_sample(&kept, 1000, 2001, 10, 300), "one second later is read by the first training feature");
        assert!(window_in_sample(&kept, 2500, 2600, 10, 300));
    }

    /// Splitmix64, a deterministic generator for the property test.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    /// Random contiguous groups of hourly bars, each a train, validation, test
    /// or unused interval.
    fn random_split(rng: &mut Rng, purge: i64) -> IntervalSplit {
        let mut d = Declared { split_id: "p".to_string(), ..Declared::default() };
        let mut t = 0i64;
        let groups = 3 + rng.below(6);
        for _ in 0..groups {
            let len = (24 + rng.below(24 * 40) as i64) * H;
            match rng.below(4) {
                0 => d.validation.push((t, t + len)),
                1 => d.test.push((t, t + len)),
                2 => {}
                _ => d.train.push((t, t + len)),
            }
            t += len + (rng.below(3) as i64) * H * 7;
        }
        if d.train.is_empty() {
            d.train.push((t, t + 30 * D));
        }
        let lookback = (rng.below(30 * 24) as i64) * H;
        IntervalSplit::new(d, purge, purge.max(lookback), EmbargoSource::Lookback, None)
    }

    #[test]
    fn interval_split_purges_and_embargoes_every_edge() {
        let mut rng = Rng(1173);
        let mut kept_rows = 0usize;
        for horizon_h in [1i64, 24, 72, 168] {
            let purge = (horizon_h + 1) * H;
            for _ in 0..200 {
                let s = random_split(&mut rng, purge);
                let end = s.declared.train.iter().chain(&s.declared.validation).chain(&s.declared.test).map(|v| v.1).max().unwrap();
                let kept_train = s.effective(Role::Train);
                let kept_validation = s.effective(Role::Validation);
                let mut t = 0i64;
                while t < end {
                    match s.role(t) {
                        Some(Role::Train) | Some(Role::Validation) => {
                            kept_rows += 1;
                            for &(a, b) in &s.declared.test {
                                assert!(!(t + purge > a && t < b), "label of {t} meets test {a}..{b}");
                                assert!(!(t >= b && t < b + s.embargo), "{t} inside the embargo after test {a}..{b}");
                            }
                            if s.role(t) == Some(Role::Train) {
                                for &(a, b) in &s.declared.validation {
                                    assert!(!(t + purge > a && t < b + purge), "training {t} overlaps validation {a}..{b}");
                                }
                                assert!(inside(&kept_train, t));
                            } else {
                                assert!(inside(&kept_validation, t));
                            }
                        }
                        Some(Role::Test) => assert!(inside(&s.declared.test, t)),
                        None => {
                            assert!(!inside(&kept_train, t) && !inside(&kept_validation, t), "{t} dropped but inside a kept interval");
                        }
                    }
                    t += H;
                }
            }
        }
        assert!(kept_rows > 100_000, "the property ran on {kept_rows} kept rows");
    }

    #[test]
    fn a_planted_label_crossing_and_a_planted_lookback_are_excluded() {
        // Hourly rows; test interval [1000 h, 1100 h). A 72 h label (purge 73 h) and a feature that
        // reads 500 h back.
        let purge = 73 * H;
        let lookback = 500 * H;
        let s = split(&[(0, 1000 * H), (1100 * H, 3000 * H)], &[], &[(1000 * H, 1100 * H)], purge, lookback);
        // The label of row 950 h ends at 1023 h: it crosses into the test interval.
        assert_eq!(s.role(950 * H), None);
        // The feature of row 1300 h reads bars from 800 h: inside the test interval.
        assert_eq!(s.role(1300 * H), None);
        // Every kept training row: label before the test start, lookback after the test end.
        for h in 0..3000 {
            let t = h * H;
            if s.role(t) == Some(Role::Train) {
                assert!(t + purge <= 1000 * H || t - lookback >= 1100 * H, "row {h} h leaks");
            }
        }
    }

    #[test]
    fn embargo_follows_the_lookback_and_refuses_a_smaller_flag() {
        let cols = vec!["r672".to_string(), "r16".to_string()];
        let (e, src, col) = resolve_embargo(None, 2 * H, &cols, H).unwrap();
        assert_eq!((e, src, col.as_deref()), (28 * D, EmbargoSource::Lookback, Some("r672")));
        let (e, src, _) = resolve_embargo(None, 169 * H, &cols, H).unwrap();
        assert_eq!((e, src), (28 * D, EmbargoSource::Lookback));
        assert!(resolve_embargo(Some(3600), 2 * H, &cols, H).unwrap_err().contains("below the automatic embargo"));
        assert_eq!(resolve_embargo(Some(40 * D), 2 * H, &cols, H).unwrap().1, EmbargoSource::Flag);
        let (e, src, _) = resolve_embargo(None, 17 * H, &["r16".to_string()], H).unwrap();
        assert_eq!((e, src), (17 * H, EmbargoSource::Purge));
        assert_eq!(column_lookback("not_a_registry_column", H), crate::protocol::DEFAULT_LOCKBOX_EMBARGO_SECONDS);
        assert_eq!(column_lookback("r672_lag24", H), 28 * D + 24 * H);
        assert_eq!(column_lookback("ma_slope672", H), 1343 * H);
    }

    #[test]
    fn modal_step_is_the_bar() {
        let times = [0.0, 3600.0, 7200.0, 0.0, 3600.0, 10_800.0, 14_400.0];
        let groups = [0usize, 0, 0, 1, 1, 1, 1];
        assert_eq!(modal_step(&times, Some(&groups)), Some(3600));
        assert_eq!(modal_step(&[5.0], None), None);
    }

    #[test]
    fn a_table_interval_that_reaches_a_locked_range_is_named() {
        let s = split(&[(0, 1000)], &[], &[(1000, 2000)], 10, 10);
        assert!(s.reaching(2005, 3000, 10, 100).iter().any(|r| r.starts_with("test")));
        assert!(s.reaching(2010, 3000, 10, 100).is_empty());
        assert!(!s.reaching(-200, -95, 10, 100).is_empty(), "the embargo after a range reaches the first interval");
    }
}
