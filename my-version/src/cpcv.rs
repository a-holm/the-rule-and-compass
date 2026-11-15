//! #1174: combinatorial purged cross-validation (CPCV).
//!
//! López de Prado (2018), *Advances in Financial Machine Learning*, ch. 12: the
//! span is cut into N groups and every split tests k of them, so there are
//! C(N, k) splits; every group is tested in C(N-1, k-1) splits and the test
//! results join into phi = k C(N, k) / N = C(N-1, k-1) complete backtest
//! paths. Path j of group g takes the j-th split, in lexicographic order of the
//! test-group tuples, that tests g.
//!
//! `rcsr protocol cpcv` writes the splits as a splits table of
//! [`crate::intervals`]. Groups have equal calendar length over the span with
//! the locked holes removed, at UTC midnights. Training and validation
//! intervals are trimmed by the predicate the search applies
//! ([`IntervalSplit::effective`]): the label purge before and the embargo after
//! every test group, and the purge on both sides of every validation interval.
//!
//! `rcsr trials --cpcv-assemble` joins the per-split walk-forward test series
//! of one configuration into the paths. Seam rule: every group of a path is one
//! book segment that starts flat and pays its final exit, whichever split it
//! comes from, so the split reports must be priced with `--pool-reset-flat`.
//! The splits that test a group usually hold different formulas; priced with
//! `--period-grid target` (#1199) they give the group one period grid.

use crate::devlog::Json;
use crate::intervals::{self, Declared, EmbargoSource, IntervalSplit, Role};
use crate::walkforward::{jnum, jstr};

pub const SCHEMA: &str = "rcsr-cpcv/1";
pub const PATH_SCHEMA: &str = "rcsr-cpcv-path/1";
pub const SUMMARY_SCHEMA: &str = "rcsr-cpcv-summary/1";
pub const SEAM_RULE: &str = "every group of a path is one book segment that starts flat and pays its final exit, \
whichever split it comes from (pooled_seam.reset_flat in every split report)";
const DAY: i64 = 86_400;
/// Most groups and most splits the generator writes.
pub const MAX_GROUPS: usize = 64;
pub const MAX_SPLITS: u64 = 10_000;

type Fail = (i32, String);

fn usage<T>(m: impl Into<String>) -> Result<T, Fail> {
    Err((2, m.into()))
}

// ---------------------------------------------------------------------------
// Combinatorics
// ---------------------------------------------------------------------------

/// `C(n, k)`, saturating at `u64::MAX`.
pub fn binomial(n: usize, k: usize) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r * (n - i) as u128 / (i + 1) as u128;
        if r > u64::MAX as u128 {
            return u64::MAX;
        }
    }
    r as u64
}

/// The k-subsets of `0..n` in lexicographic order.
pub fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    if k == 0 || k > n {
        return out;
    }
    let mut c: Vec<usize> = (0..k).collect();
    loop {
        out.push(c.clone());
        let mut i = k;
        while i > 0 && c[i - 1] == n - k + i - 1 {
            i -= 1;
        }
        if i == 0 {
            return out;
        }
        c[i - 1] += 1;
        for j in i..k {
            c[j] = c[j - 1] + 1;
        }
    }
}

/// phi, the number of complete backtest paths of CPCV(N, k).
pub fn path_count(n: usize, k: usize) -> usize {
    if k == 0 || k > n {
        0
    } else {
        binomial(n - 1, k - 1) as usize
    }
}

/// `out[g][j]`: the index into `test_sets` of the split whose result path `j`
/// takes for group `g`. `test_sets` is in lexicographic order.
pub fn assignment(test_sets: &[Vec<usize>], n: usize) -> Vec<Vec<usize>> {
    (0..n).map(|g| (0..test_sets.len()).filter(|&s| test_sets[s].contains(&g)).collect()).collect()
}

// ---------------------------------------------------------------------------
// Groups and holes
// ---------------------------------------------------------------------------

/// One calendar group: sorted, disjoint pieces of usable time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Group {
    pub pieces: Vec<(i64, i64)>,
}

impl Group {
    pub fn start(&self) -> i64 {
        self.pieces[0].0
    }

    pub fn end(&self) -> i64 {
        self.pieces[self.pieces.len() - 1].1
    }

    pub fn seconds(&self) -> i64 {
        self.pieces.iter().map(|(a, b)| b - a).sum()
    }

    pub fn contains(&self, t: f64) -> bool {
        self.pieces.iter().any(|&(a, b)| a as f64 <= t && t < b as f64)
    }
}

pub fn label(g: usize) -> String {
    format!("G{}", g + 1)
}

fn floor_day(t: i64) -> i64 {
    t.div_euclid(DAY) * DAY
}

fn ceil_day(t: i64) -> i64 {
    (t + DAY - 1).div_euclid(DAY) * DAY
}

/// A locked range and the embargo after its end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Locked {
    pub start: i64,
    pub end: i64,
    pub embargo: i64,
}

/// The time no interval may hold because of a locked range: the stamps whose
/// label window `[t, t + purge)` reaches it and the embargo after its end (at
/// least the purge), widened to UTC midnights.
pub fn hole(l: &Locked, purge: i64) -> (i64, i64) {
    (floor_day(l.start - purge.max(1) + 1), ceil_day(l.end + l.embargo.max(purge).max(0)))
}

fn subtract(pieces: &[(i64, i64)], za: i64, zb: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    for &(a, b) in pieces {
        if zb <= a || za >= b {
            out.push((a, b));
            continue;
        }
        if a < za {
            out.push((a, za));
        }
        if zb < b {
            out.push((zb, b));
        }
    }
    out
}

/// `[start, end)` without the holes.
pub fn usable(start: i64, end: i64, holes: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut pieces = vec![(start, end)];
    for &(a, b) in holes {
        pieces = subtract(&pieces, a, b);
    }
    pieces.retain(|&(a, b)| b > a);
    pieces
}

/// The pieces of usable time between the usable offsets `[lo, hi)`.
fn slice(pieces: &[(i64, i64)], lo: i64, hi: i64) -> Vec<(i64, i64)> {
    let mut out = Vec::new();
    let mut off = 0i64;
    for &(a, b) in pieces {
        let len = b - a;
        let (s, e) = (lo.max(off), hi.min(off + len));
        if e > s {
            out.push((a + s - off, a + e - off));
        }
        off += len;
    }
    out
}

/// N groups of equal usable length in whole days: group g ends at usable day
/// `floor((g + 1) D / N)` of the D usable days, so lengths differ by at most a day.
pub fn calendar_groups(pieces: &[(i64, i64)], n: usize) -> Result<Vec<Group>, String> {
    let days: i64 = pieces.iter().map(|(a, b)| (b - a) / DAY).sum();
    let bound = |g: usize| (g as i64 * days) / n as i64 * DAY;
    let mut out = Vec::with_capacity(n);
    for g in 0..n {
        let p = slice(pieces, bound(g), bound(g + 1));
        if p.is_empty() {
            return Err(format!("group {} is empty: {} usable days for {} groups", label(g), days, n));
        }
        out.push(Group { pieces: p });
    }
    Ok(out)
}

fn overlap_seconds(a: &[(i64, i64)], b: &[(i64, i64)]) -> i64 {
    let mut s = 0;
    for &(x, y) in a {
        for &(u, v) in b {
            s += (y.min(v) - x.max(u)).max(0);
        }
    }
    s
}

// ---------------------------------------------------------------------------
// The generator
// ---------------------------------------------------------------------------

/// Inputs of `rcsr protocol cpcv`, in UTC seconds.
#[derive(Clone, Debug)]
pub struct Plan {
    pub start: i64,
    pub end: i64,
    pub groups: usize,
    pub test_groups: usize,
    pub purge: i64,
    pub embargo: i64,
    pub embargo_source: EmbargoSource,
    pub embargo_column: Option<String>,
    pub validation_share: f64,
    pub locked: Vec<Locked>,
}

/// The splits of a plan, in lexicographic order of their test-group tuples.
#[derive(Clone, Debug)]
pub struct Generated {
    pub groups: Vec<Group>,
    pub test_sets: Vec<Vec<usize>>,
    /// Declared intervals are the table rows: trimmed training and validation, whole test groups.
    pub splits: Vec<IntervalSplit>,
    pub holes: Vec<(i64, i64)>,
}

pub fn split_id(s: usize, count: usize) -> String {
    let width = count.to_string().len().max(2);
    format!("s{:0w$}", s + 1, w = width)
}

fn stamp_text(t: i64) -> String {
    format!("{}Z", crate::protocol::iso(t as f64))
}

/// Exit 2 for a plan that cannot give a complete CPCV table, 3 when an interval
/// reaches a locked range (the #1173 refusal).
pub fn generate(p: &Plan) -> Result<Generated, Fail> {
    let (n, k) = (p.groups, p.test_groups);
    if n < 2 || n > MAX_GROUPS {
        return usage(format!("--groups must be in 2..={}, got {}", MAX_GROUPS, n));
    }
    if k == 0 || k >= n {
        return usage(format!("--test-groups must be in 1..{} (k < N leaves a training group), got {}", n - 1, k));
    }
    if binomial(n, k) > MAX_SPLITS {
        return usage(format!("CPCV({}, {}) has {} splits, more than {}", n, k, binomial(n, k), MAX_SPLITS));
    }
    for (flag, t) in [("--start", p.start), ("--end", p.end)] {
        if t.rem_euclid(DAY) != 0 {
            return usage(format!("{} {} is not at UTC midnight; group boundaries are whole days", flag, stamp_text(t)));
        }
    }
    if p.end <= p.start {
        return usage("--end must be after --start");
    }
    if !(p.validation_share >= 0.0 && p.validation_share < 1.0) {
        return usage(format!("--validation-share must be in [0, 1), got {}", p.validation_share));
    }
    let mut holes: Vec<(i64, i64)> = p.locked.iter().map(|l| hole(l, p.purge)).collect();
    holes.sort_unstable();
    holes.retain(|&(a, b)| b > p.start && a < p.end);
    let pieces = usable(p.start, p.end, &holes);
    if pieces.is_empty() {
        return usage("no usable time is left between --start and --end after the locked ranges");
    }
    let groups = calendar_groups(&pieces, n).map_err(|e| (2, e))?;
    let validation_seconds: Vec<i64> = groups
        .iter()
        .map(|g| ((g.seconds() / DAY) as f64 * p.validation_share + 1e-9).floor() as i64 * DAY)
        .collect();
    if p.validation_share > 0.0 {
        if let Some(g) = validation_seconds.iter().position(|&v| v == 0) {
            return usage(format!(
                "group {} has {} days, too few for --validation-share {} (the validation interval is whole days)",
                label(g),
                groups[g].seconds() / DAY,
                p.validation_share
            ));
        }
    }
    let test_sets = combinations(n, k);
    let count = test_sets.len();
    let mut splits = Vec::with_capacity(count);
    for (s, set) in test_sets.iter().enumerate() {
        let id = split_id(s, count);
        let mut whole = Declared { split_id: id.clone(), ..Declared::default() };
        for (g, group) in groups.iter().enumerate() {
            if set.contains(&g) {
                whole.test.extend(group.pieces.iter().copied());
                continue;
            }
            let total = group.seconds();
            whole.train.extend(slice(&group.pieces, 0, total - validation_seconds[g]));
            whole.validation.extend(slice(&group.pieces, total - validation_seconds[g], total));
        }
        let full = IntervalSplit::new(whole, p.purge, p.embargo, p.embargo_source, p.embargo_column.clone());
        let train = full.effective(Role::Train);
        let validation = full.effective(Role::Validation);
        let tests: Vec<String> = set.iter().map(|&g| label(g)).collect();
        for (g, group) in groups.iter().enumerate() {
            if set.contains(&g) {
                continue;
            }
            if overlap_seconds(&train, &group.pieces) == 0 {
                return usage(format!(
                    "split {}: training group {} has no time left after the purge and the embargo around test groups {}; \
                     use fewer groups, a longer span or a smaller embargo",
                    id,
                    label(g),
                    tests.join(",")
                ));
            }
            if validation_seconds[g] > 0 && overlap_seconds(&validation, &group.pieces) == 0 {
                return usage(format!(
                    "split {}: the validation interval of group {} has no time left after the purge and the embargo around test groups {}",
                    id,
                    label(g),
                    tests.join(",")
                ));
            }
        }
        let declared = Declared { split_id: id.clone(), train, validation, test: full.declared.test.clone() };
        let split = IntervalSplit::new(declared, p.purge, p.embargo, p.embargo_source, p.embargo_column.clone());
        for l in &p.locked {
            let hit = split.reaching(l.start, l.end, p.purge, l.embargo.max(p.purge));
            if !hit.is_empty() {
                return Err((
                    3,
                    format!(
                        "split {}: the interval(s) {} reach the locked range {}..{} (label end = stamp + {} s, embargo {} s after the end)",
                        id,
                        hit.join(", "),
                        stamp_text(l.start),
                        stamp_text(l.end),
                        p.purge,
                        l.embargo.max(p.purge)
                    ),
                ));
            }
        }
        splits.push(split);
    }
    Ok(Generated { groups, test_sets, splits, holes })
}

/// The hash of the test grid (N, k and the group pieces): `window_definition`
/// of every assembled path. Purge, embargo and validation share are left out,
/// so configurations with other lookback embargoes on the same groups compare.
pub fn scheme_sha256(groups: &[Group], k: usize) -> String {
    let mut text = format!("{}|{}|{}", SCHEMA, groups.len(), k);
    for (g, group) in groups.iter().enumerate() {
        for (a, b) in &group.pieces {
            text.push_str(&format!("|{}:{}:{}", g, a, b));
        }
    }
    crate::manifest::sha256_hex(text.as_bytes())
}

/// The splits table: rows of each split in time order.
pub fn table_csv(splits: &[IntervalSplit]) -> String {
    let mut out = format!("{}\n", intervals::TABLE_HEADER);
    for s in splits {
        let d = &s.declared;
        let mut rows: Vec<(i64, i64, Role)> = Vec::new();
        for (role, v) in [(Role::Train, &d.train), (Role::Validation, &d.validation), (Role::Test, &d.test)] {
            rows.extend(v.iter().map(|&(a, b)| (a, b, role)));
        }
        rows.sort_unstable_by_key(|&(a, b, _)| (a, b));
        for (a, b, role) in rows {
            out.push_str(&format!("{},{},{},{}\n", d.split_id, role.as_str(), stamp_text(a), stamp_text(b)));
        }
    }
    out
}

pub fn groups_csv(groups: &[Group]) -> String {
    let mut out = String::from("group,start,end,days\n");
    for (g, group) in groups.iter().enumerate() {
        for &(a, b) in &group.pieces {
            out.push_str(&format!("{},{},{},{}\n", label(g), stamp_text(a), stamp_text(b), (b - a) / DAY));
        }
    }
    out
}

/// Walk-forward windows of one split: one window per test piece, labelled by its
/// group, with a one-second training range that ends `purge` before the test.
pub fn windows_csv(groups: &[Group], set: &[usize], purge: i64) -> String {
    let mut out = String::from("label,train_start,train_end,test_start,test_end\n");
    for &g in set {
        let pieces = &groups[g].pieces;
        for (i, &(a, b)) in pieces.iter().enumerate() {
            let name = if pieces.len() == 1 { label(g) } else { format!("{}.{}", label(g), i + 1) };
            let train_end = a - purge.max(1);
            out.push_str(&format!("{},{},{},{},{}\n", name, stamp_text(train_end - 1), stamp_text(train_end), stamp_text(a), stamp_text(b)));
        }
    }
    out
}

fn intervals_json(v: &[(i64, i64)]) -> String {
    format!("[{}]", v.iter().map(|(a, b)| format!("[{},{}]", a, b)).collect::<Vec<_>>().join(","))
}

/// The run description written as `cpcv.json`.
pub struct Manifest<'a> {
    pub plan: &'a Plan,
    pub generated: &'a Generated,
    pub bar_seconds: Option<i64>,
    pub lock_source: &'a str,
    pub splits_sha256: &'a str,
}

impl Manifest<'_> {
    pub fn json(&self) -> String {
        let (p, g) = (self.plan, self.generated);
        let count = g.test_sets.len();
        let ids: Vec<String> = (0..count).map(|s| split_id(s, count)).collect();
        let groups: Vec<String> = g
            .groups
            .iter()
            .enumerate()
            .map(|(i, gr)| format!("{{\"group\":{},\"pieces\":{},\"days\":{}}}", jstr(&label(i)), intervals_json(&gr.pieces), gr.seconds() / DAY))
            .collect();
        let sets: Vec<String> = g
            .test_sets
            .iter()
            .enumerate()
            .map(|(s, set)| {
                format!(
                    "{{\"split\":{},\"test\":[{}]}}",
                    jstr(&ids[s]),
                    set.iter().map(|&x| jstr(&label(x))).collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        let assign: Vec<String> = assignment(&g.test_sets, g.groups.len())
            .iter()
            .enumerate()
            .map(|(i, a)| {
                format!(
                    "{{\"group\":{},\"splits\":[{}]}}",
                    jstr(&label(i)),
                    a.iter().map(|&s| jstr(&ids[s])).collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        let locked: Vec<String> = p
            .locked
            .iter()
            .map(|l| format!("{{\"start\":{},\"end\":{},\"embargo_seconds\":{}}}", l.start, l.end, l.embargo.max(p.purge)))
            .collect();
        format!(
            "{{\"schema\":{},\"start\":{},\"end\":{},\"groups\":{},\"test_groups\":{},\"splits\":{},\"paths\":{},\
\"purge_seconds\":{},\"embargo_seconds\":{},\"embargo_source\":{},\"embargo_column\":{},\"bar_seconds\":{},\
\"validation_share\":{},\"lock_source\":{},\"locked\":[{}],\"holes\":{},\"scheme_sha256\":{},\"splits_sha256\":{},\
\"seam_rule\":{},\"group_bounds\":[{}],\"test_sets\":[{}],\"assignment\":[{}],\
\"files\":{{\"splits\":\"splits.csv\",\"groups\":\"groups.csv\",\"windows\":[{}]}}}}\n",
            jstr(SCHEMA),
            jstr(&stamp_text(p.start)),
            jstr(&stamp_text(p.end)),
            p.groups,
            p.test_groups,
            count,
            path_count(p.groups, p.test_groups),
            p.purge,
            p.embargo,
            jstr(p.embargo_source.as_str()),
            p.embargo_column.as_deref().map_or("null".to_string(), jstr),
            self.bar_seconds.map_or("null".to_string(), |b| b.to_string()),
            jnum(p.validation_share),
            jstr(self.lock_source),
            locked.join(","),
            intervals_json(&g.holes),
            jstr(&scheme_sha256(&g.groups, p.test_groups)),
            jstr(self.splits_sha256),
            jstr(SEAM_RULE),
            groups.join(","),
            sets.join(","),
            assign.join(","),
            ids.iter().map(|id| jstr(&format!("windows_{}.csv", id))).collect::<Vec<_>>().join(",")
        )
    }
}

pub fn protocol_usage() -> &'static str {
    "  cpcv --start DATE --end DATE --groups N --test-groups K --purge-seconds P --out DIR
       [--embargo-seconds S|auto] [--vars a,b,...] [--bar-seconds B] [--validation-share F]
       [--data PATH] [--lockbox START,END|none] [--lockbox-embargo S]
      Combinatorial purged cross-validation (López de Prado 2018, ch. 12). Writes
      DIR/splits.csv (a --split-table with C(N,K) splits s01.., in lexicographic
      order of their test groups), DIR/groups.csv, DIR/windows_<id>.csv (one
      walk-forward window per test group, labelled G1..GN) and DIR/cpcv.json
      (counts, phi = K C(N,K) / N paths, the path assignment, scheme_sha256).
      START and END are UTC midnights; the N groups have equal length in whole
      days (within one day) over the span with the locked holes removed. The
      training rows of every split are trimmed by the label purge P before and
      the embargo after every test group, and the last share F of each training
      group is a validation interval purged on both sides. --embargo-seconds auto
      (the default) is the larger of P and the longest lookback of the --vars
      columns at a bar of B seconds (rule rcsr-lookback/1; needs --vars and
      --bar-seconds); a smaller explicit value exits 2. The locked ranges of
      --data or --lockbox (as for a search) are holes [start - P, end + max(embargo,
      P)) widened to midnights; an interval that still reaches one exits 3.
      K >= N, a group with no time left after the holes, purge and embargo, or a
      non-midnight START or END exits 2. Then one search per split, one
      `rcsr walkforward --windows DIR/windows_<id>.csv --pool-reset-flat
      --period-grid target` per split (the splits of a group share its period
      grid whatever NaN coverage or gate their formulas have) and
      `rcsr trials --cpcv-assemble`.
"
}

fn int_arg(flag: &str, v: &str) -> Result<i64, Fail> {
    v.trim().parse::<i64>().ok().filter(|x| *x >= 0).map_or_else(|| usage(format!("{} needs whole seconds >= 0, got '{}'", flag, v)), Ok)
}

fn time_arg(flag: &str, v: &str) -> Result<i64, Fail> {
    match crate::data::parse_time(v) {
        Some(t) if t.is_finite() && t.fract() == 0.0 => Ok(t as i64),
        _ => usage(format!("{} cannot parse '{}' (ISO 8601 date or epoch seconds)", flag, v)),
    }
}

fn protocol_inner(args: &[String]) -> Result<String, Fail> {
    let mut start = None;
    let mut end = None;
    let mut groups = None;
    let mut test_groups = None;
    let mut purge = None;
    let mut embargo_arg: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut bar = None;
    let mut share = 0.0f64;
    let mut data = String::new();
    let mut lockbox: Option<String> = None;
    let mut lockbox_embargo: Option<String> = None;
    let mut out: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let Some(v) = args.get(i + 1).cloned() else {
            return usage(format!("{} needs a value", flag));
        };
        match flag {
            "--start" => start = Some(time_arg(flag, &v)?),
            "--end" => end = Some(time_arg(flag, &v)?),
            "--groups" => groups = Some(int_arg(flag, &v)? as usize),
            "--test-groups" => test_groups = Some(int_arg(flag, &v)? as usize),
            "--purge-seconds" => purge = Some(int_arg(flag, &v)?),
            "--embargo-seconds" => embargo_arg = Some(v),
            "--vars" => vars = v.split(',').map(str::trim).filter(|s| !s.is_empty()).map(str::to_string).collect(),
            "--bar-seconds" => {
                let b = int_arg(flag, &v)?;
                if b == 0 {
                    return usage("--bar-seconds must be at least 1");
                }
                bar = Some(b);
            }
            "--validation-share" => {
                share = v.trim().parse::<f64>().map_or_else(|_| usage(format!("--validation-share needs a number, got '{}'", v)), Ok)?
            }
            "--data" => data = v,
            "--lockbox" => lockbox = Some(v),
            "--lockbox-embargo" => lockbox_embargo = Some(v),
            "--out" => out = Some(v),
            other => return usage(format!("unknown cpcv option '{}'", other)),
        }
        i += 2;
    }
    let need = |v: Option<i64>, flag: &str| v.map_or_else(|| usage(format!("cpcv needs {}", flag)), Ok);
    let start = need(start, "--start")?;
    let end = need(end, "--end")?;
    let groups = need(groups.map(|g| g as i64), "--groups")? as usize;
    let test_groups = need(test_groups.map(|g| g as i64), "--test-groups")? as usize;
    let purge = need(purge, "--purge-seconds")?;
    let out = out.map_or_else(|| usage("cpcv needs --out DIR"), Ok)?;
    let flag = match embargo_arg.as_deref().map(str::trim) {
        None | Some("auto") => None,
        Some(s) => Some(int_arg("--embargo-seconds", s)?),
    };
    if flag.is_none() && vars.is_empty() {
        return usage("--embargo-seconds auto needs --vars (the columns whose lookback sets the embargo) and --bar-seconds; or give --embargo-seconds S");
    }
    if !vars.is_empty() && bar.is_none() {
        return usage("--vars needs --bar-seconds, the bar the lookback rule counts in");
    }
    let (embargo, source, column) = intervals::resolve_embargo(flag, purge, &vars, bar.unwrap_or(0)).map_err(|e| (2, e.replace("--embargo ", "--embargo-seconds ")))?;
    let resolved = crate::lockbox::resolve_with(lockbox.as_deref(), lockbox_embargo.as_deref(), &data).map_err(|e| (e.exit, e.message))?;
    let locked: Vec<Locked> = resolved.ranges().iter().map(|l| Locked { start: l.start, end: l.end, embargo: resolved.embargo.seconds }).collect();
    let plan = Plan { start, end, groups, test_groups, purge, embargo, embargo_source: source, embargo_column: column, validation_share: share, locked };
    let generated = generate(&plan)?;
    let dir = std::path::Path::new(&out);
    std::fs::create_dir_all(dir).map_err(|e| (3, format!("{}: {}", out, e)))?;
    let write = |name: &str, text: &str| crate::atomic::write_text_atomic(dir.join(name), text).map_err(|e| (3, e));
    let table = table_csv(&generated.splits);
    let splits_sha = crate::manifest::sha256_hex(table.as_bytes());
    write("splits.csv", &table)?;
    write("groups.csv", &groups_csv(&generated.groups))?;
    let count = generated.test_sets.len();
    for (s, set) in generated.test_sets.iter().enumerate() {
        write(&format!("windows_{}.csv", split_id(s, count)), &windows_csv(&generated.groups, set, purge))?;
    }
    let manifest = Manifest { plan: &plan, generated: &generated, bar_seconds: bar, lock_source: resolved.source.as_str(), splits_sha256: &splits_sha };
    write("cpcv.json", &manifest.json())?;
    Ok(format!(
        "cpcv: {} groups, {} test groups, {} splits, {} paths; purge {} s, embargo {} s ({}); wrote {}/splits.csv (sha256 {})",
        groups,
        test_groups,
        count,
        path_count(groups, test_groups),
        purge,
        embargo,
        source.as_str(),
        out,
        splits_sha
    ))
}

/// `rcsr protocol cpcv ...`.
pub fn protocol_cli(args: &[String]) -> ! {
    match protocol_inner(args) {
        Ok(line) => {
            println!("{}", line);
            std::process::exit(0)
        }
        Err((code, msg)) => {
            eprintln!("error: {}", msg);
            std::process::exit(code)
        }
    }
}

// ---------------------------------------------------------------------------
// Reading a CPCV splits table back
// ---------------------------------------------------------------------------

/// The CPCV structure of a splits table: groups from the test intervals, the
/// test-group set of every split, splits in lexicographic order.
#[derive(Clone, Debug)]
pub struct Scheme {
    pub groups: Vec<Group>,
    pub test_groups: usize,
    pub test_sets: Vec<Vec<usize>>,
    pub ids: Vec<String>,
    pub tests: Vec<Vec<(i64, i64)>>,
}

impl Scheme {
    pub fn paths(&self) -> usize {
        path_count(self.groups.len(), self.test_groups)
    }

    pub fn assignment(&self) -> Vec<Vec<usize>> {
        assignment(&self.test_sets, self.groups.len())
    }

    pub fn sha256(&self) -> String {
        scheme_sha256(&self.groups, self.test_groups)
    }
}

/// A test interval that several splits share is one piece of a group; pieces
/// tested by the same splits form one group. The table must hold every k-subset
/// of the N groups exactly once.
pub fn read_scheme(text: &str) -> Result<Scheme, String> {
    let mut header = false;
    let mut ids: Vec<String> = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !header {
            header = true;
            continue;
        }
        let id = line.split(',').next().unwrap_or("").trim();
        if !ids.iter().any(|i| i == id) {
            ids.push(id.to_string());
        }
    }
    if ids.is_empty() {
        return Err("the splits table holds no split".to_string());
    }
    let declared: Vec<Declared> = ids.iter().map(|id| intervals::parse_table(text, id)).collect::<Result<_, _>>()?;
    if let Some(d) = declared.iter().find(|d| d.test.is_empty()) {
        return Err(format!("split '{}' has no test interval: not a CPCV table", d.split_id));
    }
    let mut pieces: Vec<(i64, i64)> = declared.iter().flat_map(|d| d.test.iter().copied()).collect();
    pieces.sort_unstable();
    pieces.dedup();
    for w in pieces.windows(2) {
        if w[1].0 < w[0].1 {
            return Err(format!(
                "the test intervals {}..{} and {}..{} overlap without being equal: a CPCV table tests whole groups",
                stamp_text(w[0].0),
                stamp_text(w[0].1),
                stamp_text(w[1].0),
                stamp_text(w[1].1)
            ));
        }
    }
    let mut by_members: Vec<(Vec<usize>, Vec<(i64, i64)>)> = Vec::new();
    for &p in &pieces {
        let members: Vec<usize> = (0..declared.len()).filter(|&s| declared[s].test.contains(&p)).collect();
        match by_members.iter_mut().find(|(m, _)| *m == members) {
            Some((_, v)) => v.push(p),
            None => by_members.push((members, vec![p])),
        }
    }
    let groups: Vec<Group> = by_members.iter().map(|(_, v)| Group { pieces: v.clone() }).collect();
    let n = groups.len();
    let mut sets: Vec<(Vec<usize>, usize)> = Vec::with_capacity(declared.len());
    for (s, d) in declared.iter().enumerate() {
        let set: Vec<usize> = (0..n).filter(|&g| groups[g].pieces.iter().all(|p| d.test.contains(p))).collect();
        sets.push((set, s));
    }
    let k = sets[0].0.len();
    if let Some((set, s)) = sets.iter().find(|(set, _)| set.len() != k) {
        return Err(format!("split '{}' tests {} groups, split '{}' tests {}: not a CPCV table", ids[*s], set.len(), ids[sets[0].1], k));
    }
    if k >= n {
        return Err(format!("every split tests all {} groups: not a CPCV table", n));
    }
    sets.sort();
    if let Some(w) = sets.windows(2).find(|w| w[0].0 == w[1].0) {
        return Err(format!("splits '{}' and '{}' test the same groups", ids[w[0].1], ids[w[1].1]));
    }
    let want = binomial(n, k);
    if sets.len() as u64 != want {
        return Err(format!("{} splits of {} groups with {} test groups each; CPCV({}, {}) has {}", sets.len(), n, k, n, k, want));
    }
    Ok(Scheme {
        groups,
        test_groups: k,
        test_sets: sets.iter().map(|(set, _)| set.clone()).collect(),
        ids: sets.iter().map(|(_, s)| ids[*s].clone()).collect(),
        tests: sets.iter().map(|(_, s)| declared[*s].test.clone()).collect(),
    })
}

// ---------------------------------------------------------------------------
// Path assembly
// ---------------------------------------------------------------------------

fn at<'a>(j: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(j, |cur, k| cur.get(k))
}

/// A `Json` value as JSON text.
pub fn to_json(j: &Json) -> String {
    match j {
        Json::Null => "null".to_string(),
        Json::Bool(b) => b.to_string(),
        Json::Num(x) => jnum(*x),
        Json::Str(s) => jstr(s),
        Json::Arr(v) => format!("[{}]", v.iter().map(to_json).collect::<Vec<_>>().join(",")),
        Json::Obj(v) => format!("{{{}}}", v.iter().map(|(k, x)| format!("{}:{}", jstr(k), to_json(x))).collect::<Vec<_>>().join(",")),
    }
}

fn series(j: &Json, path: &[&str], file: &str) -> Result<Vec<f64>, Fail> {
    let name = path.join(".");
    let Some(Json::Arr(items)) = at(j, path) else {
        return usage(format!("field '{}' is missing in {}", name, file));
    };
    items
        .iter()
        .enumerate()
        .map(|(i, v)| match v.as_f64() {
            Some(x) if x.is_finite() => Ok(x),
            _ => usage(format!("field '{}'[{}] is not a finite number in {}", name, i, file)),
        })
        .collect()
}

struct Window {
    test: (i64, i64),
    entries: Option<f64>,
    active_rows: Option<f64>,
    expression: Option<String>,
}

struct SplitReport {
    path: String,
    sha256: String,
    json: Json,
    windows: Vec<Window>,
    time: Vec<f64>,
    net: Vec<f64>,
    /// #1199: `pooled.period_rows` of a target-grid report; `None` for the score grid.
    rows: Option<Vec<f64>>,
}

fn whole(x: Option<f64>) -> Option<i64> {
    x.filter(|v| v.is_finite() && v.fract() == 0.0).map(|v| v as i64)
}

fn load_report(path: &str) -> Result<SplitReport, Fail> {
    let text = std::fs::read_to_string(path).map_err(|e| (3, format!("{}: {}", path, e)))?;
    let json = Json::parse(text.trim()).ok_or_else(|| (3, format!("{}: not a JSON report", path)))?;
    let net = series(&json, &["pooled", "period_net"], path)?;
    let time = series(&json, &["pooled", "period_time"], path)?;
    if net.len() != time.len() {
        return usage(format!("field 'pooled.period_time' has {} values but 'pooled.period_net' has {} in {}", time.len(), net.len(), path));
    }
    let rows = if at(&json, crate::trials::GRID_RULE).and_then(Json::as_str) == Some("target") {
        let rows = series(&json, &["pooled", "period_rows"], path)?;
        if rows.len() != time.len() {
            return usage(format!("field 'pooled.period_rows' has {} values but 'pooled.period_time' has {} in {}", rows.len(), time.len(), path));
        }
        Some(rows)
    } else {
        None
    };
    let Some(Json::Arr(defs)) = json.get("window_definition") else {
        return usage(format!("field 'window_definition' is not a list of windows in {}", path));
    };
    let per: Vec<Json> = match json.get("windows") {
        Some(Json::Arr(w)) => w.clone(),
        _ => Vec::new(),
    };
    let mut windows = Vec::with_capacity(defs.len());
    for (i, d) in defs.iter().enumerate() {
        let (Some(a), Some(b)) = (whole(at(d, &["test_start"]).and_then(Json::as_f64)), whole(at(d, &["test_end"]).and_then(Json::as_f64))) else {
            return usage(format!("field 'window_definition'[{}] has no whole-second test_start and test_end in {}", i, path));
        };
        let w = per.get(i);
        windows.push(Window {
            test: (a, b),
            entries: w.and_then(|w| at(w, &["metrics", "entries"])).and_then(Json::as_f64),
            active_rows: w.and_then(|w| at(w, &["metrics", "active_rows"])).and_then(Json::as_f64),
            expression: w.and_then(|w| w.get("expression")).and_then(Json::as_str).map(str::to_string),
        });
    }
    Ok(SplitReport { path: path.to_string(), sha256: crate::manifest::sha256_hex(text.as_bytes()), json, windows, time, net, rows })
}

/// The guarded fields of `rcsr trials`, without `window_definition` (it differs per split by design).
fn guard_paths() -> Vec<&'static [&'static str]> {
    crate::trials::GUARDS
        .iter()
        .chain(crate::trials::FIT_GUARDS.iter())
        .copied()
        .filter(|g| g[0] != "window_definition")
        .collect()
}

fn require_same(a: &SplitReport, b: &SplitReport) -> Result<(), Fail> {
    for g in guard_paths() {
        let (x, y) = (at(&a.json, g).cloned().unwrap_or(Json::Null), at(&b.json, g).cloned().unwrap_or(Json::Null));
        if x != y {
            return usage(format!("split reports differ in field '{}': {} vs {}", g.join("."), a.path, b.path));
        }
    }
    Ok(())
}

/// One assembled path.
struct BuiltPath {
    splits: Vec<usize>,
    time: Vec<f64>,
    net: Vec<f64>,
    group_net: Vec<f64>,
    group_periods: Vec<usize>,
    entries: Option<f64>,
    active_rows: Option<f64>,
    /// #1199: priced rows per period and empty periods per group, for target-grid reports.
    rows: Vec<f64>,
    group_empty: Vec<usize>,
}

fn sum(x: &[f64]) -> f64 {
    x.iter().sum()
}

fn max_drawdown(x: &[f64]) -> f64 {
    let mut d = crate::risk::Drawdown::identity();
    for &v in x {
        d.push(v);
    }
    d.max_drawdown()
}

fn median(x: &[f64]) -> f64 {
    let mut s: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    s.sort_by(f64::total_cmp);
    crate::inference::quantile_sorted(&s, 0.5)
}

fn finite_min(x: &[f64]) -> f64 {
    x.iter().copied().filter(|v| v.is_finite()).fold(f64::NAN, |m, v| if m.is_nan() || v < m { v } else { m })
}

fn finite_max(x: &[f64]) -> f64 {
    x.iter().copied().filter(|v| v.is_finite()).fold(f64::NAN, |m, v| if m.is_nan() || v > m { v } else { m })
}

fn nums(x: &[f64]) -> String {
    x.iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(",")
}

struct Assembly {
    reports: Vec<String>,
    splits: String,
    out_dir: String,
    summary: Option<String>,
}

fn parse_assembly(args: &[String]) -> Result<Assembly, Fail> {
    let mut a = Assembly { reports: Vec::new(), splits: String::new(), out_dir: String::new(), summary: None };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let val = |i: &mut usize| -> Result<String, Fail> {
            *i += 1;
            args.get(*i).cloned().map_or_else(|| usage(format!("{} needs a value", flag)), Ok)
        };
        match flag {
            "--cpcv-assemble" => {
                while args.get(i + 1).is_some_and(|n| !n.starts_with("--")) {
                    i += 1;
                    a.reports.push(args[i].clone());
                }
            }
            "--splits" => a.splits = val(&mut i)?,
            "--out-dir" => a.out_dir = val(&mut i)?,
            "--summary" => a.summary = Some(val(&mut i)?),
            other => return usage(format!("{} does not apply to --cpcv-assemble (it takes --splits, --out-dir and --summary)", other)),
        }
        i += 1;
    }
    if a.reports.is_empty() {
        return usage("--cpcv-assemble needs the walk-forward report of every split");
    }
    if a.splits.is_empty() {
        return usage("--cpcv-assemble needs --splits <csv>, the table of rcsr protocol cpcv");
    }
    if a.out_dir.is_empty() {
        return usage("--cpcv-assemble needs --out-dir <dir> for the path reports");
    }
    Ok(a)
}

/// `rcsr trials --cpcv-assemble <reports...> --splits <csv> --out-dir <dir> [--summary <file>]`.
pub fn assemble_cli(args: &[String]) -> Result<(), Fail> {
    let a = parse_assembly(args)?;
    let table = std::fs::read_to_string(&a.splits).map_err(|e| (3, format!("{}: {}", a.splits, e)))?;
    let scheme = read_scheme(&table).map_err(|e| (2, format!("--splits {}: {}", a.splits, e)))?;
    let (n, count) = (scheme.groups.len(), scheme.ids.len());
    let mut by_split: Vec<Option<SplitReport>> = (0..count).map(|_| None).collect();
    for path in &a.reports {
        let r = load_report(path)?;
        let mut tests: Vec<(i64, i64)> = r.windows.iter().map(|w| w.test).collect();
        tests.sort_unstable();
        let Some(s) = scheme.tests.iter().position(|t| *t == tests) else {
            let list: Vec<String> = tests.iter().map(|(x, y)| format!("{}..{}", stamp_text(*x), stamp_text(*y))).collect();
            return usage(format!(
                "{}: its test intervals [{}] match no split of {}; walk-forward each split on its own windows file",
                path,
                list.join(", "),
                a.splits
            ));
        };
        if let Some(prev) = &by_split[s] {
            return usage(format!("split {} has two reports: {} and {}", scheme.ids[s], prev.path, path));
        }
        by_split[s] = Some(r);
    }
    if let Some(s) = by_split.iter().position(Option::is_none) {
        return usage(format!("split {} has no report (give one walk-forward report per split of {})", scheme.ids[s], a.splits));
    }
    let reports: Vec<SplitReport> = by_split.into_iter().map(Option::unwrap).collect();
    for r in &reports {
        if at(&r.json, &["pooled_seam", "reset_flat"]) != Some(&Json::Bool(true)) {
            return usage(format!(
                "field 'pooled_seam.reset_flat' is not true in {}: every group of a path starts flat; rerun walkforward with --pool-reset-flat",
                r.path
            ));
        }
        require_same(&reports[0], r)?;
    }
    let ppy = at(&reports[0].json, &["annualization", "periods_per_year"]).and_then(Json::as_f64).unwrap_or(f64::NAN);
    // Group series of every split, as index lists into its pooled series.
    let mut parts: Vec<Vec<Vec<usize>>> = Vec::with_capacity(count);
    for (s, r) in reports.iter().enumerate() {
        let set = &scheme.test_sets[s];
        let mut own: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (i, &t) in r.time.iter().enumerate() {
            let Some(&g) = set.iter().find(|&&g| scheme.groups[g].contains(t)) else {
                return usage(format!("{}: the period at {} lies outside the test groups of split {}", r.path, jnum(t), scheme.ids[s]));
            };
            own[g].push(i);
        }
        if let Some(&g) = set.iter().find(|&&g| own[g].is_empty()) {
            return usage(format!("{}: test group {} of split {} has no period", r.path, label(g), scheme.ids[s]));
        }
        parts.push(own);
    }
    let assign = scheme.assignment();
    for g in 0..n {
        let first = assign[g][0];
        let grid = |s: usize| -> Vec<u64> { parts[s][g].iter().map(|&i| reports[s].time[i].to_bits()).collect() };
        let want = grid(first);
        for &s in &assign[g][1..] {
            if grid(s) != want {
                return usage(format!(
                    "split reports differ in field 'pooled.period_time' on group {}: {} ({} periods) vs {} ({} periods)",
                    label(g),
                    reports[first].path,
                    parts[first][g].len(),
                    reports[s].path,
                    parts[s][g].len()
                ));
            }
        }
    }
    let window_sum = |s: usize, g: usize, f: fn(&Window) -> Option<f64>| -> Option<f64> {
        reports[s].windows.iter().filter(|w| scheme.groups[g].pieces.contains(&w.test)).map(f).sum()
    };
    let phi = scheme.paths();
    let paths: Vec<BuiltPath> = (0..phi)
        .map(|j| {
            let mut p = BuiltPath { splits: Vec::with_capacity(n), time: Vec::new(), net: Vec::new(), group_net: Vec::new(), group_periods: Vec::new(), entries: Some(0.0), active_rows: Some(0.0), rows: Vec::new(), group_empty: Vec::new() };
            for g in 0..n {
                let s = assign[g][j];
                p.splits.push(s);
                let idx = &parts[s][g];
                let net: Vec<f64> = idx.iter().map(|&i| reports[s].net[i]).collect();
                p.time.extend(idx.iter().map(|&i| reports[s].time[i]));
                p.group_net.push(sum(&net));
                p.group_periods.push(net.len());
                if let Some(rows) = &reports[s].rows {
                    let r: Vec<f64> = idx.iter().map(|&i| rows[i]).collect();
                    p.group_empty.push(r.iter().filter(|&&v| v == 0.0).count());
                    p.rows.extend(r);
                }
                p.net.extend(net);
                p.entries = p.entries.zip(window_sum(s, g, |w| w.entries)).map(|(a, b)| a + b);
                p.active_rows = p.active_rows.zip(window_sum(s, g, |w| w.active_rows)).map(|(a, b)| a + b);
            }
            p
        })
        .collect();
    let dir = std::path::Path::new(&a.out_dir);
    std::fs::create_dir_all(dir).map_err(|e| (3, format!("{}: {}", a.out_dir, e)))?;
    let width = if phi >= 10 { phi.to_string().len() } else { 1 };
    let files: Vec<String> = (0..phi).map(|j| format!("path_{:0w$}.json", j + 1, w = width)).collect();
    let scheme_sha = scheme.sha256();
    let splits_sha = crate::manifest::sha256_hex(table.as_bytes());
    let ann = ppy.max(0.0).sqrt();
    let first = &reports[0].json;
    let copy = |k: &str| first.get(k).map_or("null".to_string(), to_json);
    let target = reports[0].rows.is_some();
    for (j, p) in paths.iter().enumerate() {
        let segments: Vec<String> = (0..n)
            .map(|g| {
                let s = p.splits[g];
                let r = &reports[s];
                let exprs: Vec<String> = r
                    .windows
                    .iter()
                    .filter(|w| scheme.groups[g].pieces.contains(&w.test))
                    .filter_map(|w| w.expression.as_deref().map(jstr))
                    .collect();
                format!(
                    "{{\"group\":{},\"start\":{},\"end\":{},\"split\":{},\"report\":{},\"report_sha256\":{},\"expression\":[{}],\"periods\":{},\"net\":{}}}",
                    jstr(&label(g)),
                    scheme.groups[g].start(),
                    scheme.groups[g].end(),
                    jstr(&scheme.ids[s]),
                    jstr(&r.path),
                    jstr(&r.sha256),
                    exprs.join(","),
                    p.group_periods[g],
                    jnum(p.group_net[g])
                )
            })
            .collect();
        let opt = |v: Option<f64>| v.map_or("null".to_string(), jnum);
        let text = format!(
            "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"cpcv_path\",\"schema\":{},\"window_definition\":{},\
\"purge\":{},\"cost_model\":{},\"cost\":{},\"annualization\":{},\"threshold\":{},\"policy\":{},\
\"pooled_seam\":{{\"reset_flat\":true,\"seam_cost\":0,\"seam_trades\":0}},\
\"cpcv\":{{\"scheme_sha256\":{},\"splits_sha256\":{},\"groups\":{},\"test_groups\":{},\"splits\":{},\"paths\":{},\"path\":{},\
\"seam_rule\":{},\"segments\":[{}]}},\
\"pooled\":{{\"sharpe\":{},\"max_drawdown\":{},\"net\":{},\"mean_return\":{},\"periods\":{},\"periods_per_year\":{},\
\"entries\":{},\"active_rows\":{},\"period_net\":[{}],\"period_time\":[{}]{}}}}}\n",
            jstr(env!("CARGO_PKG_VERSION")),
            jstr(PATH_SCHEMA),
            jstr(&scheme_sha),
            copy("purge"),
            copy("cost_model"),
            copy("cost"),
            copy("annualization"),
            copy("threshold"),
            copy("policy"),
            jstr(&scheme_sha),
            jstr(&splits_sha),
            n,
            scheme.test_groups,
            count,
            phi,
            j + 1,
            jstr(SEAM_RULE),
            segments.join(","),
            jnum(crate::inference::sharpe_per_period(&p.net) * ann),
            jnum(max_drawdown(&p.net)),
            jnum(sum(&p.net)),
            jnum(crate::inference::mean(&p.net)),
            p.net.len(),
            jnum(ppy),
            opt(p.entries),
            opt(p.active_rows),
            nums(&p.net),
            nums(&p.time),
            if target {
                format!(
                    ",\"grid\":{{\"rule\":\"target\",\"periods\":{},\"empty_periods\":{}}},\"period_rows\":[{}]",
                    p.net.len(),
                    p.group_empty.iter().sum::<usize>(),
                    nums(&p.rows)
                )
            } else {
                String::new()
            }
        );
        crate::atomic::write_text_atomic(dir.join(&files[j]), &text).map_err(|e| (3, e))?;
    }
    let sharpe: Vec<f64> = paths.iter().map(|p| crate::inference::sharpe_per_period(&p.net) * ann).collect();
    let net: Vec<f64> = paths.iter().map(|p| sum(&p.net)).collect();
    let dd: Vec<f64> = paths.iter().map(|p| max_drawdown(&p.net)).collect();
    let positive = |v: &[f64]| v.iter().filter(|x| **x > 0.0).count() as f64 / v.len().max(1) as f64;
    let per_path: Vec<String> = paths
        .iter()
        .enumerate()
        .map(|(j, p)| {
            format!(
                "{{\"path\":{},\"file\":{},\"periods\":{},\"net\":{},\"net_annual\":{},\"sharpe_annual\":{},\"max_drawdown\":{},\
\"splits\":[{}],\"group_net\":[{}]{}}}",
                j + 1,
                jstr(&files[j]),
                p.net.len(),
                jnum(net[j]),
                jnum(crate::inference::mean(&p.net) * ppy),
                jnum(sharpe[j]),
                jnum(dd[j]),
                p.splits.iter().map(|&s| jstr(&scheme.ids[s])).collect::<Vec<_>>().join(","),
                nums(&p.group_net),
                if target { format!(",\"empty_periods\":{}", p.group_empty.iter().sum::<usize>()) } else { String::new() }
            )
        })
        .collect();
    let per_group: Vec<String> = (0..n)
        .map(|g| {
            let v: Vec<f64> = paths.iter().map(|p| p.group_net[g]).collect();
            format!(
                "{{\"group\":{},\"start\":{},\"end\":{},\"splits\":[{}],\"net\":[{}],\"net_mean\":{},\"net_min\":{},\"positive_share\":{}{}}}",
                jstr(&label(g)),
                jstr(&stamp_text(scheme.groups[g].start())),
                jstr(&stamp_text(scheme.groups[g].end())),
                assign[g].iter().map(|&s| jstr(&scheme.ids[s])).collect::<Vec<_>>().join(","),
                nums(&v),
                jnum(crate::inference::mean(&v)),
                jnum(finite_min(&v)),
                jnum(positive(&v)),
                if target {
                    format!(",\"empty_periods\":[{}]", paths.iter().map(|p| p.group_empty[g].to_string()).collect::<Vec<_>>().join(","))
                } else {
                    String::new()
                }
            )
        })
        .collect();
    let summary = format!(
        "{{\"schema\":{},\"scheme_sha256\":{},\"splits_sha256\":{},\"groups\":{},\"test_groups\":{},\"splits\":{},\"paths\":{},\
\"seam_rule\":{},{}\"cost_per_side_bps\":{},\"periods_per_year\":{},\"out_dir\":{},\
\"sharpe\":{{\"values\":[{}],\"mean\":{},\"sd\":{},\"min\":{},\"median\":{},\"max\":{}}},\
\"net\":{{\"values\":[{}],\"mean\":{},\"median\":{},\"min\":{},\"max\":{},\"positive_share\":{}}},\
\"max_drawdown\":{{\"values\":[{}],\"worst\":{}}},\
\"pbo\":{{\"skipped\":\"PBO compares configurations: run rcsr trials --wf on the paths of every configuration\"}},\
\"per_path\":[{}],\"per_group\":[{}]}}\n",
        jstr(SUMMARY_SCHEMA),
        jstr(&scheme_sha),
        jstr(&splits_sha),
        n,
        scheme.test_groups,
        count,
        phi,
        jstr(SEAM_RULE),
        if target { "\"grid\":{\"rule\":\"target\"},".to_string() } else { String::new() },
        at(first, &["cost_model", "per_side_bps"]).map_or("null".to_string(), to_json),
        jnum(ppy),
        jstr(&a.out_dir),
        nums(&sharpe),
        jnum(crate::inference::mean(&sharpe)),
        jnum(crate::inference::sample_sd(&sharpe)),
        jnum(finite_min(&sharpe)),
        jnum(median(&sharpe)),
        jnum(finite_max(&sharpe)),
        nums(&net),
        jnum(crate::inference::mean(&net)),
        jnum(median(&net)),
        jnum(finite_min(&net)),
        jnum(finite_max(&net)),
        jnum(positive(&net)),
        nums(&dd),
        jnum(finite_min(&dd)),
        per_path.join(","),
        per_group.join(",")
    );
    match &a.summary {
        Some(p) => crate::atomic::write_text_atomic(p, &summary).map_err(|e| (3, e)),
        None => {
            print!("{}", summary);
            Ok(())
        }
    }
}
