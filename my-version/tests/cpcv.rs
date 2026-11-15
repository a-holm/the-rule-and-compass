//! #1174: CPCV splits (`rcsr protocol cpcv`) and backtest-path assembly
//! (`rcsr trials --cpcv-assemble`).

#[path = "common/cpcv_series.rs"]
mod cpcv_series;
#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::cpcv::{self, Locked, Plan};
use rcsr::devlog::Json;
use rcsr::intervals::{EmbargoSource, IntervalSplit, Role};
use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const H: i64 = 3600;
const D: i64 = 86_400;

fn day(s: &str) -> i64 {
    rcsr::data::parse_time(s).unwrap() as i64
}

fn rcsr(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn ok(o: &Output) {
    assert_eq!(code(o), 0, "{}", stderr(o));
}

fn parse(text: &str) -> Json {
    Json::parse(text.trim()).unwrap_or_else(|| panic!("not json: {text}"))
}

fn read_json(path: &Path) -> Json {
    parse(&std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}"));
    }
    cur
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
}

fn arr<'a>(j: &'a Json, path: &[&str]) -> &'a [Json] {
    match at(j, path) {
        Json::Arr(a) => a,
        other => panic!("{path:?} is not an array: {other:?}"),
    }
}

fn floats(j: &Json, path: &[&str]) -> Vec<f64> {
    arr(j, path).iter().map(|v| v.as_f64().unwrap()).collect()
}

fn pilot_plan(k: usize) -> Plan {
    Plan {
        start: day("2017-12-15"),
        end: day("2023-11-01"),
        groups: 6,
        test_groups: k,
        purge: 2 * H,
        embargo: 30 * D,
        embargo_source: EmbargoSource::Flag,
        embargo_column: None,
        validation_share: 0.2,
        locked: Vec::new(),
    }
}

/// Every path takes each group once, and every (split, test group) pair feeds exactly one path.
fn check_assignment(test_sets: &[Vec<usize>], n: usize, k: usize) {
    let a = cpcv::assignment(test_sets, n);
    let phi = cpcv::path_count(n, k);
    assert_eq!(phi as u64 * n as u64, k as u64 * cpcv::binomial(n, k), "phi = k C(N,k) / N");
    let mut used = std::collections::HashSet::new();
    for (g, splits) in a.iter().enumerate() {
        assert_eq!(splits.len(), phi, "group {g} is tested in C(N-1,k-1) splits");
        assert!(splits.windows(2).all(|w| w[0] < w[1]), "path order is the lexicographic split order");
        for &s in splits {
            assert!(test_sets[s].contains(&g));
            assert!(used.insert((s, g)), "split {s} group {g} used twice");
        }
    }
    assert_eq!(used.len(), test_sets.len() * k, "every test result is on one path");
}

fn generate_cli(dir: &Path, extra: &[&str]) -> Output {
    let out = dir.to_string_lossy().into_owned();
    let mut args = vec!["protocol", "cpcv", "--out", out.as_str()];
    args.extend_from_slice(extra);
    rcsr(&args)
}

const PILOT_62: &[&str] = &[
    "--start", "2017-12-15", "--end", "2023-11-01", "--groups", "6", "--test-groups", "2", "--purge-seconds", "7200",
    "--embargo-seconds", "2592000", "--validation-share", "0.2",
];

#[test]
fn cpcv_6_2_counts() {
    let g = cpcv::generate(&pilot_plan(2)).unwrap();
    assert_eq!(g.splits.len(), 15);
    assert_eq!(g.test_sets.len(), 15);
    assert_eq!(cpcv::path_count(6, 2), 5);
    for grp in 0..6 {
        assert_eq!(g.test_sets.iter().filter(|s| s.contains(&grp)).count(), 5, "group {grp}");
    }
    check_assignment(&g.test_sets, 6, 2);
    // The group table of the issue, computed with `date`.
    let bounds = ["2017-12-15", "2018-12-07", "2019-11-30", "2020-11-22", "2021-11-15", "2022-11-08", "2023-11-01"];
    for (i, grp) in g.groups.iter().enumerate() {
        assert_eq!((grp.start(), grp.end()), (day(bounds[i]), day(bounds[i + 1])), "group {}", i + 1);
    }
    // Split s01 tests G1 and G2: G3 trains from the embargo end to the purge before its validation tail.
    let s01 = &g.splits[0].declared;
    assert_eq!(s01.split_id, "s01");
    assert_eq!(s01.test, vec![(day("2017-12-15"), day("2018-12-07")), (day("2018-12-07"), day("2019-11-30"))]);
    assert_eq!(s01.train[0], (day("2019-11-30") + 30 * D, day("2020-09-12") - 2 * H + 1));
    assert_eq!(s01.validation[0], (day("2020-09-12"), day("2020-11-22")));
    assert_eq!(s01.train[1].0, day("2020-11-22") + 2 * H, "training is purged after the validation interval too");

    let s = Scratch::new("cpcv-62");
    let (a, b) = (s.path().join("a"), s.path().join("b"));
    ok(&generate_cli(&a, PILOT_62));
    ok(&generate_cli(&b, PILOT_62));
    let table = std::fs::read_to_string(a.join("splits.csv")).unwrap();
    assert_eq!(table, std::fs::read_to_string(b.join("splits.csv")).unwrap(), "the table is deterministic");
    assert_eq!(table, cpcv::table_csv(&g.splits), "the CLI writes the library table");
    let m = read_json(&a.join("cpcv.json"));
    assert_eq!((num(&m, &["splits"]), num(&m, &["paths"])), (15.0, 5.0));
    assert_eq!(arr(&m, &["files", "windows"]).len(), 15);
    for id in ["s01", "s15"] {
        let w = std::fs::read_to_string(a.join(format!("windows_{id}.csv"))).unwrap();
        assert_eq!(w.lines().count(), 3, "header and one window per test group: {w}");
    }
    let scheme = cpcv::read_scheme(&table).unwrap();
    assert_eq!(scheme.ids.len(), 15);
    assert_eq!(scheme.test_sets, g.test_sets);
    assert_eq!(at(&m, &["scheme_sha256"]).as_str(), Some(scheme.sha256().as_str()));
}

#[test]
fn cpcv_6_1_counts() {
    let g = cpcv::generate(&pilot_plan(1)).unwrap();
    assert_eq!(g.splits.len(), 6);
    assert_eq!(cpcv::path_count(6, 1), 1);
    check_assignment(&g.test_sets, 6, 1);
    for (s, split) in g.splits.iter().enumerate() {
        assert_eq!(split.declared.test, g.groups[s].pieces, "split {} tests group {}", s + 1, s + 1);
    }
    let s = Scratch::new("cpcv-61");
    let mut args = PILOT_62.to_vec();
    args[7] = "1";
    ok(&generate_cli(s.path(), &args));
    let m = read_json(&s.path().join("cpcv.json"));
    assert_eq!((num(&m, &["splits"]), num(&m, &["paths"])), (6.0, 1.0));
}

#[test]
fn cpcv_generator_refusals() {
    let s = Scratch::new("cpcv-refusals");
    let dir = s.path().join("out");
    let base = ["--start", "2020-01-01", "--end", "2021-01-01", "--purge-seconds", "3600", "--embargo-seconds", "86400"];
    let run = |extra: &[&str]| {
        let mut a = base.to_vec();
        a.extend_from_slice(extra);
        generate_cli(&dir, &a)
    };
    let o = run(&["--groups", "4", "--test-groups", "4"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--test-groups"), "{}", stderr(&o));
    // A 100-day embargo after G2 leaves the 61-day G3 of split s01 (G1, G2) nothing.
    let o = run(&["--groups", "6", "--test-groups", "2", "--embargo-seconds", "8640000"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("split s01: training group G3 has no time left"), "{}", stderr(&o));
    let mut late = base.to_vec();
    late[1] = "2020-01-01T06:00:00Z";
    late.extend_from_slice(&["--groups", "4", "--test-groups", "1"]);
    let o = generate_cli(&dir, &late);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("not at UTC midnight"), "{}", stderr(&o));
    let o = run(&["--groups", "4", "--test-groups", "1", "--vars", "r672", "--embargo-seconds", "86400", "--bar-seconds", "3600"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("below the automatic embargo 2419200"), "{}", stderr(&o));
    let o = run(&["--groups", "4", "--test-groups", "1", "--vars", "r16,r672", "--embargo-seconds", "auto", "--bar-seconds", "3600"]);
    ok(&o);
    let m = read_json(&dir.join("cpcv.json"));
    assert_eq!(num(&m, &["embargo_seconds"]), (28 * D) as f64);
    assert_eq!(at(&m, &["embargo_column"]).as_str(), Some("r672"));
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

/// Independent checks of one generated split: interval `[a, b)` holds the stamps a..b-1, a row
/// stamped t carries the label window [t, t + purge).
fn check_split(split: &IntervalSplit, plan: &Plan, groups: &[cpcv::Group], set: &[usize]) {
    let d = &split.declared;
    let (purge, embargo) = (plan.purge, plan.embargo);
    let want: Vec<(i64, i64)> = set.iter().flat_map(|&g| groups[g].pieces.iter().copied()).collect();
    assert_eq!(d.test, want, "{}: test intervals are whole groups", d.split_id);
    let kept: Vec<&(i64, i64)> = d.train.iter().chain(&d.validation).collect();
    for &&(a, b) in &kept {
        assert!(a < b && a >= plan.start && b <= plan.end, "{}: {a}..{b} outside the span", d.split_id);
        for &(s, e) in &d.test {
            assert!(b - 1 + purge <= s || a >= e + embargo, "{}: {a}..{b} meets test {s}..{e} (purge {purge}, embargo {embargo})", d.split_id);
        }
        for l in &plan.locked {
            let le = l.embargo.max(purge);
            assert!(b - 1 + purge.max(1) <= l.start || a >= l.end + le, "{}: {a}..{b} reaches locked {}..{} + {le}", d.split_id, l.start, l.end);
        }
    }
    for &(a, b) in &d.train {
        for &(vs, ve) in &d.validation {
            assert!(b - 1 + purge <= vs || a >= ve + purge, "{}: training {a}..{b} meets validation {vs}..{ve}", d.split_id);
        }
    }
    let again = IntervalSplit::new(d.clone(), purge, embargo, EmbargoSource::Flag, None);
    assert_eq!(again.effective(Role::Train), d.train, "{}: the search's trim keeps the table's training intervals", d.split_id);
    assert_eq!(again.effective(Role::Validation), d.validation);
    for (g, grp) in groups.iter().enumerate() {
        if !set.contains(&g) {
            let inside: i64 = d.train.iter().map(|&(a, b)| grp.pieces.iter().map(|&(u, v)| (b.min(v) - a.max(u)).max(0)).sum::<i64>()).sum();
            assert!(inside > 0, "{}: training group {} kept nothing", d.split_id, g + 1);
        }
    }
}

#[test]
fn cpcv_splits_respect_purge_and_embargo() {
    let mut rng = Rng(1174);
    let mut checked = 0usize;
    for horizon_h in [1i64, 24, 72, 168] {
        let purge = (horizon_h + 1) * H;
        for _ in 0..60 {
            let n = 3 + rng.below(6) as usize;
            let k = 1 + rng.below(n as u64 - 1) as usize;
            if cpcv::binomial(n, k) > 70 {
                continue;
            }
            let start = day("2012-01-01") + rng.below(3000) as i64 * D;
            let end = start + (400 + rng.below(2600) as i64) * D;
            let mut locked = Vec::new();
            for _ in 0..rng.below(3) {
                let a = start + rng.below(((end - start) / D) as u64) as i64 * D + rng.below(24) as i64 * H;
                locked.push(Locked { start: a, end: a + (5 + rng.below(55) as i64) * D, embargo: rng.below(30 * 24) as i64 * H });
            }
            let plan = Plan {
                start,
                end,
                groups: n,
                test_groups: k,
                purge,
                embargo: purge.max(rng.below(100 * 24) as i64 * H),
                embargo_source: EmbargoSource::Flag,
                embargo_column: None,
                validation_share: [0.0, 0.1, 0.2, 0.3][rng.below(4) as usize],
                locked,
            };
            let g = match cpcv::generate(&plan) {
                Ok(g) => g,
                Err((code, msg)) => {
                    assert_eq!(code, 2, "{msg}");
                    assert!(["no time left", "too few", "is empty", "no usable time"].iter().any(|m| msg.contains(m)), "{msg}");
                    continue;
                }
            };
            assert_eq!(g.splits.len() as u64, cpcv::binomial(n, k));
            check_assignment(&g.test_sets, n, k);
            let days: Vec<i64> = g.groups.iter().map(|x| x.seconds()).collect();
            assert!(days.iter().max().unwrap() - days.iter().min().unwrap() <= D, "group lengths {days:?}");
            for (s, split) in g.splits.iter().enumerate() {
                check_split(split, &plan, &g.groups, &g.test_sets[s]);
            }
            let scheme = cpcv::read_scheme(&cpcv::table_csv(&g.splits)).unwrap();
            assert_eq!(scheme.groups, g.groups, "the table gives back its groups");
            assert_eq!(scheme.test_sets, g.test_sets);
            checked += 1;
        }
    }
    assert!(checked >= 120, "the property ran on {checked} plans");
}

#[test]
fn cpcv_skips_locked_holes() {
    let lock = Locked { start: day("2020-03-01"), end: day("2020-06-01"), embargo: 30 * D };
    let plan = Plan { locked: vec![lock], ..pilot_plan(2) };
    let g = cpcv::generate(&plan).unwrap();
    let hole = (day("2020-02-29"), day("2020-07-01"));
    assert_eq!(cpcv::hole(&lock, plan.purge), hole);
    let len: Vec<i64> = g.groups.iter().map(|x| x.seconds()).collect();
    assert!(len.iter().max().unwrap() - len.iter().min().unwrap() <= D, "{len:?}");
    assert_eq!(len.iter().sum::<i64>(), plan.end - plan.start - (hole.1 - hole.0));
    let split_group: Vec<&cpcv::Group> = g.groups.iter().filter(|x| x.pieces.len() == 2).collect();
    assert_eq!(split_group.len(), 1, "one group holds the hole");
    assert_eq!((split_group[0].pieces[0].1, split_group[0].pieces[1].0), hole);
    for (s, split) in g.splits.iter().enumerate() {
        check_split(split, &plan, &g.groups, &g.test_sets[s]);
        assert!(split.reaching(lock.start, lock.end, plan.purge, lock.embargo).is_empty());
    }
    check_assignment(&g.test_sets, 6, 2);

    // The search's #1173 guard accepts every split of a table built around a --lockbox range.
    let s = Scratch::new("cpcv-hole");
    let dir = s.path();
    let data = write_panel(dir, day("2020-01-01"), 425 * 24, 1);
    let spec = write_spec(dir);
    let out = dir.join("t");
    ok(&generate_cli(
        &out,
        &[
            "--start", "2020-01-01", "--end", "2021-03-01", "--groups", "4", "--test-groups", "2", "--purge-seconds", "3600",
            "--embargo-seconds", "345600", "--lockbox", "2020-03-01,2020-03-15",
        ],
    ));
    let m = read_json(&out.join("cpcv.json"));
    assert_eq!(at(&m, &["lock_source"]).as_str(), Some("flag+protocol_default"));
    let holes = arr(&m, &["holes"]);
    assert_eq!(holes.len(), 1, "the protocol default range lies outside the span");
    let table = out.join("splits.csv").to_string_lossy().into_owned();
    for id in ["s01", "s06"] {
        let o = rcsr(&[
            "--data", &data, "--target", "y", "--time", "ts", "--target-spec", &spec, "--cpu", "--no-dev-log", "--seed", "5", "--json",
            "--quiet", "--vars", "r16,r96", "--generations", "1", "--pop", "8", "--split-table", &table, "--split-id", id, "--lockbox",
            "2020-03-01,2020-03-15",
        ]);
        ok(&o);
    }
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Hourly rows of `symbols` symbols from `t0`: `r16` and `r96` are registry names, `r` the log
/// return and `y` the target.
fn write_panel(dir: &Path, t0: i64, hours: usize, symbols: usize) -> String {
    let mut text = String::from("ts,sym,r16,r96,r,y\n");
    for i in 0..hours {
        for g in 0..symbols {
            let f = i as f64 + 0.37 * g as f64;
            let a = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let b = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let r = 0.01 * a + 0.004 * noise;
            text.push_str(&format!("{},S{},{},{},{},{}\n", t0 + H * i as i64, g, a, b, r, r));
        }
    }
    let path = dir.join("panel.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// A one-hour forward-return target spec on `y` with return column `r`.
fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":4.0,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("spec.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// CPCV(4, 2) over 40 days of a 3-symbol panel: one 1-generation search and one walk-forward per
/// split. Returns the table path and the six walk-forward reports in split order.
fn real_fixture(s: &Scratch) -> (String, Vec<String>) {
    let dir = s.path();
    let data = write_panel(dir, day("2020-09-14"), 40 * 24, 3);
    let spec = write_spec(dir);
    let out = dir.join("cpcv42");
    ok(&generate_cli(
        &out,
        &[
            "--start", "2020-09-14", "--end", "2020-10-24", "--groups", "4", "--test-groups", "2", "--purge-seconds", "3600",
            "--embargo-seconds", "auto", "--vars", "r16,r96", "--bar-seconds", "3600", "--validation-share", "0.25",
        ],
    ));
    let table = out.join("splits.csv").to_string_lossy().into_owned();
    let mut reports = Vec::new();
    for s in 1..=6 {
        let id = format!("s0{s}");
        let o = rcsr(&[
            "--data", &data, "--target", "y", "--time", "ts", "--group-by", "sym", "--target-spec", &spec, "--cpu", "--no-dev-log",
            "--seed", "5", "--json", "--quiet", "--vars", "r16,r96", "--generations", "1", "--pop", "8", "--split-table", &table,
            "--split-id", &id, "--objective", "sharpe", "--fitness", "pnl",
        ]);
        ok(&o);
        let run = dir.join(format!("{id}.run.json"));
        std::fs::write(&run, &o.stdout).unwrap();
        let windows = out.join(format!("windows_{id}.csv")).to_string_lossy().into_owned();
        let report = dir.join(format!("wf_{id}.json")).to_string_lossy().into_owned();
        let o = rcsr(&[
            "walkforward", "--data", &data, "--time", "ts", "--group-by", "sym", "--from-run", &run.to_string_lossy(), "--windows",
            &windows, "--pool-reset-flat", "--json", "--out", &report,
        ]);
        ok(&o);
        reports.push(report);
    }
    (table, reports)
}

fn assemble(table: &str, reports: &[String], out_dir: &Path, summary: &Path) -> Output {
    let (o, sm) = (out_dir.to_string_lossy().into_owned(), summary.to_string_lossy().into_owned());
    let mut args = vec!["trials", "--cpcv-assemble"];
    args.extend(reports.iter().map(String::as_str));
    args.extend_from_slice(&["--splits", table, "--out-dir", &o, "--summary", &sm]);
    rcsr(&args)
}

#[test]
fn cpcv_paths_cover_each_group_once() {
    let s = Scratch::new("cpcv-paths");
    let (table, reports) = real_fixture(&s);
    let paths = s.path().join("paths");
    let summary = s.path().join("summary.json");
    // The reports in reverse order: the split of a report comes from its test intervals.
    let reversed: Vec<String> = reports.iter().rev().cloned().collect();
    ok(&assemble(&table, &reversed, &paths, &summary));
    let scheme = cpcv::read_scheme(&std::fs::read_to_string(&table).unwrap()).unwrap();
    let wf: Vec<Json> = reports.iter().map(|r| read_json(Path::new(r))).collect();
    // (G1, G2) s01, (G1, G3) s02, (G1, G4) s03, (G2, G3) s04, (G2, G4) s05, (G3, G4) s06.
    let want = [["s01", "s01", "s02", "s03"], ["s02", "s04", "s04", "s05"], ["s03", "s05", "s06", "s06"]];
    for (j, ids) in want.iter().enumerate() {
        let p = read_json(&paths.join(format!("path_{}.json", j + 1)));
        assert_eq!(at(&p, &["window_definition"]).as_str(), Some(scheme.sha256().as_str()));
        assert_eq!(at(&p, &["pooled_seam", "reset_flat"]), &Json::Bool(true));
        let segs = arr(&p, &["cpcv", "segments"]);
        assert_eq!(segs.len(), 4);
        let mut want_net: Vec<u64> = Vec::new();
        let mut want_time: Vec<u64> = Vec::new();
        for (g, seg) in segs.iter().enumerate() {
            assert_eq!(at(seg, &["group"]).as_str(), Some(format!("G{}", g + 1).as_str()), "path {} covers each group once, in order", j + 1);
            assert_eq!(at(seg, &["split"]).as_str(), Some(ids[g]), "path {} group {}", j + 1, g + 1);
            let r = &wf[ids[g][1..].parse::<usize>().unwrap() - 1];
            let (time, net) = (floats(r, &["pooled", "period_time"]), floats(r, &["pooled", "period_net"]));
            for (t, v) in time.iter().zip(&net) {
                if scheme.groups[g].contains(*t) {
                    want_time.push(t.to_bits());
                    want_net.push(v.to_bits());
                }
            }
        }
        let got_net: Vec<u64> = floats(&p, &["pooled", "period_net"]).iter().map(|v| v.to_bits()).collect();
        let got_time: Vec<u64> = floats(&p, &["pooled", "period_time"]).iter().map(|v| v.to_bits()).collect();
        assert!(!got_net.is_empty());
        assert_eq!(got_net, want_net, "path {}: period_net is the concatenation of the chosen group series", j + 1);
        assert_eq!(got_time, want_time);
    }
    let sm = read_json(&summary);
    assert_eq!(num(&sm, &["paths"]), 3.0);
    assert_eq!(arr(&sm, &["sharpe", "values"]).len(), 3);
    assert_eq!(arr(&sm, &["per_group"]).len(), 4);
    let share = num(&sm, &["net", "positive_share"]);
    assert!((0.0..=1.0).contains(&share));
}

#[test]
fn trials_reads_cpcv_paths() {
    // Synthetic split reports: every split has its own series, so the three paths differ.
    let f = synthetic();
    let paths = f.dir.join("paths");
    ok(&assemble(&f.table, &reports_with(&f, None), &paths, &f.dir.join("summary.json")));
    let files: Vec<String> = (1..=3).map(|j| paths.join(format!("path_{j}.json")).to_string_lossy().into_owned()).collect();
    let first = read_json(Path::new(&files[0]));
    let time = floats(&first, &["pooled", "period_time"]);
    let bh: Vec<String> = (0..time.len()).map(|i| format!("{}", 1e-4 * ((i % 7) as f64 - 3.0))).collect();
    let cash = vec!["0"; time.len()];
    let times: Vec<String> = time.iter().map(|t| format!("{t}")).collect();
    let ladder = f.dir.join("ladder.json");
    std::fs::write(
        &ladder,
        format!(
            "{{\"schema\":\"rcsr-ladder/1\",\"series\":{{\"period_time\":[{}],\"period_net\":{{\"cash\":[{}],\"buy_and_hold\":[{}]}}}}}}",
            times.join(","),
            cash.join(","),
            bh.join(",")
        ),
    )
    .unwrap();
    let mut args = vec!["trials", "--wf"];
    args.extend(files.iter().map(String::as_str));
    let lp = ladder.to_string_lossy().into_owned();
    args.extend_from_slice(&["--baseline", &lp, "--draws", "200"]);
    let o = rcsr(&args);
    ok(&o);
    let j = parse(&String::from_utf8_lossy(&o.stdout));
    assert_eq!(num(&j, &["series"]), 3.0);
    assert!(num(&j, &["clusters"]) >= 1.0, "N_eff");
    let dsr = num(&j, &["best", "dsr", "value"]);
    assert!((0.0..=1.0).contains(&dsr), "{dsr}");
    assert!(at(&j, &["spa"]).get("skipped").is_none(), "{:?}", at(&j, &["spa"]));
    let pbo = num(&j, &["pbo", "value"]);
    assert!((0.0..=1.0).contains(&pbo), "{pbo}");
}

/// Synthetic split reports of CPCV(4, 2) over 8 days of hourly periods.
struct Synthetic {
    _s: Scratch,
    dir: std::path::PathBuf,
    table: String,
    scheme: cpcv::Scheme,
}

fn synthetic() -> Synthetic {
    let s = Scratch::new("cpcv-guards");
    let dir = s.path().to_path_buf();
    let out = dir.join("t");
    ok(&generate_cli(
        &out,
        &["--start", "2021-01-01", "--end", "2021-01-09", "--groups", "4", "--test-groups", "2", "--purge-seconds", "3600", "--embargo-seconds", "3600"],
    ));
    let table = out.join("splits.csv").to_string_lossy().into_owned();
    let scheme = cpcv::read_scheme(&std::fs::read_to_string(&table).unwrap()).unwrap();
    Synthetic { _s: s, dir, table, scheme }
}

struct Fields {
    cost: f64,
    quantile: &'static str,
    reset_flat: bool,
    /// #1203: `book.missing_score`; `None` writes no `book` object.
    book: Option<&'static str>,
}

const FIELDS: Fields = Fields { cost: 10.0, quantile: "0.5", reset_flat: true, book: None };

fn write_report(f: &Synthetic, s: usize, test: &[(i64, i64)], fields: &Fields) -> String {
    let defs: Vec<String> = test.iter().map(|(a, b)| format!("{{\"label\":\"w\",\"train_start\":0,\"train_end\":1,\"test_start\":{a},\"test_end\":{b}}}")).collect();
    let mut time = Vec::new();
    let mut net = Vec::new();
    for &(a, b) in test {
        let mut t = a;
        while t < b {
            time.push(t.to_string());
            net.push(format!("{}", 1e-4 * (((t / H) as usize * (s + 3)) % 11) as f64 - 4e-4));
            t += H;
        }
    }
    let text = format!(
        "{{\"mode\":\"walkforward\",\"window_definition\":[{}],\"purge\":{{\"horizon_seconds\":3600,\"embargo_seconds\":3600}},\
\"cost_model\":{{\"per_side_bps\":{c},\"round_trip_bps\":{rt}}},\"cost\":{{\"per_side_bps\":{c},\"source\":\"flag\"}},\
\"policy\":{{\"name\":\"threshold\",\"band_width\":0,\"min_hold\":1}},\"threshold\":{{\"quantile\":{q},\"fixed\":null}},\
\"annualization\":{{\"periods_per_year\":8760,\"method\":\"explicit\"}},{book}\"pooled_seam\":{{\"reset_flat\":{rf},\"seam_cost\":0,\"seam_trades\":0}},\
\"pooled\":{{\"periods\":{n},\"entries\":10,\"active_rows\":100,\"period_net\":[{net}],\"period_time\":[{time}]}}}}",
        defs.join(","),
        c = fields.cost,
        rt = 2.0 * fields.cost,
        q = fields.quantile,
        rf = fields.reset_flat,
        book = fields.book.map_or(String::new(), |r| format!("\"book\":{{\"missing_score\":\"{r}\"}},")),
        n = net.len(),
        net = net.join(","),
        time = time.join(","),
    );
    let path = f.dir.join(format!("wf{s}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn reports_with(f: &Synthetic, odd: Option<(usize, Fields)>) -> Vec<String> {
    (0..f.scheme.ids.len())
        .map(|s| match &odd {
            Some((i, fields)) if *i == s => write_report(f, s, &f.scheme.tests[s], fields),
            _ => write_report(f, s, &f.scheme.tests[s], &FIELDS),
        })
        .collect()
}

#[test]
fn cpcv_assemble_guards_fields() {
    let f = synthetic();
    let out = f.dir.join("paths");
    let summary = f.dir.join("summary.json");
    ok(&assemble(&f.table, &reports_with(&f, None), &out, &summary));
    let cases: [(Fields, &str); 3] = [
        (Fields { cost: 20.0, ..FIELDS }, "cost_model.per_side_bps"),
        (Fields { quantile: "0.6", ..FIELDS }, "threshold.quantile"),
        (Fields { reset_flat: false, ..FIELDS }, "pooled_seam.reset_flat"),
    ];
    for (fields, field) in cases {
        let o = assemble(&f.table, &reports_with(&f, Some((3, fields))), &out, &summary);
        assert_eq!(code(&o), 2, "{field}: {}", stderr(&o));
        assert!(stderr(&o).contains(field), "{field}: {}", stderr(&o));
    }
    // A missing split report and a report of no split of the table.
    let mut r = reports_with(&f, None);
    r.pop();
    let o = assemble(&f.table, &r, &out, &summary);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("split s06 has no report"), "{}", stderr(&o));
    let foreign = write_report(&f, 9, &[f.scheme.groups[0].pieces[0]], &FIELDS);
    r.push(foreign);
    let o = assemble(&f.table, &r, &out, &summary);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("match no split"), "{}", stderr(&o));
}

/// #1199: hourly rows of three symbols from `t0`; `b` is `a` with every symbol NaN on hours
/// `[gap.0, gap.1)`.
fn write_gap_panel(dir: &Path, t0: i64, hours: usize, gap: (usize, usize)) -> String {
    let mut text = String::from("ts,sym,a,b,y\n");
    for i in 0..hours {
        for g in 0..3 {
            let f = i as f64 + 0.37 * g as f64;
            let a = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let y = 0.01 * (frac(f * 0.754_877_666_246_692_7) - 0.5) + 0.002 * a;
            let b = if (gap.0..gap.1).contains(&i) { String::new() } else { a.to_string() };
            text.push_str(&format!("{},S{},{},{},{}\n", t0 + H * i as i64, g, a, b, y));
        }
    }
    let path = dir.join("gap_panel.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn cpcv_assemble_common_grid() {
    // CPCV(4, 2) over 8 days: G2 is hours 48..96 and is tested by s01, s04 and s05. Split s04
    // prices a formula with no score on hours 60..90; the other splits price one with full coverage.
    let f = synthetic();
    let data = write_gap_panel(&f.dir, day("2021-01-01"), 8 * 24, (60, 90));
    let wf = |id: &str, expr: &str, grid: &[&str]| -> String {
        let windows = f.dir.join("t").join(format!("windows_{id}.csv")).to_string_lossy().into_owned();
        let tag = grid.last().copied().unwrap_or("default");
        let out = f.dir.join(format!("wf_{tag}_{id}.json")).to_string_lossy().into_owned();
        let mut args = vec![
            "walkforward", "--data", &data, "--target", "y", "--time", "ts", "--group-by", "sym", "--expr", expr, "--vars", "a,b",
            "--windows", &windows, "--pool-reset-flat", "--horizon", "3600", "--threshold-fixed", "0", "--json", "--out", &out,
        ];
        args.extend_from_slice(grid);
        let o = rcsr(&args);
        ok(&o);
        out
    };
    let reports = |grid: &[&str]| -> Vec<String> {
        f.scheme.ids.iter().map(|id| wf(id, if id == "s04" { "b" } else { "a" }, grid)).collect()
    };
    let out = f.dir.join("grid_paths");
    let summary = f.dir.join("grid_summary.json");
    let score = reports(&["--period-grid", "score"]);
    let o = assemble(&f.table, &score, &out, &summary);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("'pooled.period_time' on group G2"), "{}", stderr(&o));
    let target = reports(&["--period-grid", "target"]);
    ok(&assemble(&f.table, &target, &out, &summary));
    // G2 of path j comes from the j-th of s01, s04, s05: path 2 holds the 30 empty hours.
    let want_empty = [0.0, 30.0, 0.0];
    let first = floats(&read_json(&out.join("path_1.json")), &["pooled", "period_time"]);
    for (j, want) in want_empty.iter().enumerate() {
        let p = read_json(&out.join(format!("path_{}.json", j + 1)));
        assert_eq!(at(&p, &["pooled", "grid", "rule"]).as_str(), Some("target"));
        assert_eq!(num(&p, &["pooled", "grid", "empty_periods"]), *want, "path {}", j + 1);
        let time = floats(&p, &["pooled", "period_time"]);
        assert_eq!(num(&p, &["pooled", "grid", "periods"]), time.len() as f64);
        assert_eq!(floats(&p, &["pooled", "period_rows"]).len(), time.len());
        assert_eq!(bits(&time), bits(&first), "every path has the same grid");
    }
    let sm = read_json(&summary);
    assert_eq!(at(&sm, &["grid", "rule"]).as_str(), Some("target"));
    let per_path: Vec<f64> = arr(&sm, &["per_path"]).iter().map(|p| num(p, &["empty_periods"])).collect();
    assert_eq!(per_path, want_empty);
    assert_eq!(floats(&arr(&sm, &["per_group"])[1], &["empty_periods"]), want_empty, "G2 on each path");
    assert_eq!(floats(&arr(&sm, &["per_group"])[0], &["empty_periods"]), [0.0; 3]);
    // A score report among target reports: the grid rule is a guarded field.
    let mut mixed = target.clone();
    mixed[5] = score[5].clone();
    let o = assemble(&f.table, &mixed, &out, &summary);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.grid.rule"), "{}", stderr(&o));
    let o = rcsr(&["walkforward", "--data", &data, "--period-grid", "formula"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("--period-grid needs score or target"), "{}", stderr(&o));
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn cpcv_assemble_refuses_mixed_missing_score_rules() {
    let f = synthetic();
    let out = f.dir.join("paths");
    let summary = f.dir.join("summary.json");
    let flat = Fields { book: Some("flat"), ..FIELDS };
    let all_flat: Vec<String> = (0..f.scheme.ids.len()).map(|s| write_report(&f, s, &f.scheme.tests[s], &flat)).collect();
    ok(&assemble(&f.table, &all_flat, &out, &summary));
    let path = read_json(&out.join("path_1.json"));
    assert_eq!(at(&path, &["book", "missing_score"]).as_str(), Some("flat"), "a path keeps the rule of its reports");
    for odd in [Fields { book: Some("hold"), ..FIELDS }, FIELDS] {
        let mut r = all_flat.clone();
        r[3] = write_report(&f, 3, &f.scheme.tests[3], &odd);
        let o = assemble(&f.table, &r, &out, &summary);
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        assert!(stderr(&o).contains("book.missing_score"), "{}", stderr(&o));
    }
}

/// #1236: every path carries each series of the split reports as the concatenation of the
/// chosen group series, and `pooled.entry_times` as the sum of the chosen groups' window counts.
#[test]
fn cpcv_paths_carry_period_series() {
    let s = Scratch::new("cpcv-series");
    let dir = s.path();
    let (table, scheme) = cpcv_series::write_table(dir);
    assert!(scheme.groups.iter().any(|g| g.pieces.len() == 2), "a group with two pieces");
    let reports = cpcv_series::write_reports(dir, "wf", &scheme, &|_| cpcv_series::ALL);
    let paths = dir.join("paths");
    ok(&assemble(&table, &reports, &paths, &dir.join("summary.json")));
    let wf: Vec<Json> = reports.iter().map(|r| read_json(Path::new(r))).collect();
    let assign = cpcv::assignment(&scheme.test_sets, scheme.groups.len());
    let mut stamp_short = 0;
    for j in 0..scheme.paths() {
        let p = read_json(&paths.join(format!("path_{}.json", j + 1)));
        for &key in ["period_net", "period_time"].iter().chain(cpcv_series::SERIES.iter()) {
            let want = cpcv_series::path_values(&wf, &scheme, j, key);
            assert!(!want.is_empty());
            assert_eq!(bits(&floats(&p, &["pooled", key])), bits(&want), "path {}: pooled.{key}", j + 1);
        }
        let segs = arr(&p, &["cpcv", "segments"]);
        for (g, seg) in segs.iter().enumerate() {
            let split = assign[g][j];
            assert_eq!(at(seg, &["split"]).as_str(), Some(scheme.ids[split].as_str()));
            let want = cpcv_series::group_entry_times(&wf[split], &scheme.groups[g]);
            assert_eq!(num(seg, &["entry_times"]), want, "path {} group {}", j + 1, g + 1);
        }
        let entry_times = cpcv_series::path_entry_times(&wf, &scheme, j);
        assert_eq!(num(&p, &["pooled", "entry_times"]), entry_times, "path {}", j + 1);
        let stamps: f64 = floats(&p, &["pooled", "period_entry_stamp"]).iter().sum();
        if stamps < entry_times {
            stamp_short += 1;
        }
    }
    assert!(stamp_short > 0, "the window counts are not the entry-stamp sums");
}

/// #1236: a series on some split reports only, a report without the window counts and a
/// series one value short exit 2 naming the field and the report.
#[test]
fn cpcv_assemble_refuses_mixed_series() {
    let s = Scratch::new("cpcv-series-mixed");
    let dir = s.path();
    let (table, scheme) = cpcv_series::write_table(dir);
    let paths = dir.join("paths");
    let summary = dir.join("summary.json");
    let cases = [
        (cpcv_series::without("period_bh_net"), "pooled.period_bh_net"),
        (cpcv_series::Carry { window_entry_times: false, ..cpcv_series::ALL }, "windows[].entry_times"),
        (cpcv_series::Carry { short: Some("period_cost"), ..cpcv_series::ALL }, "pooled.period_cost"),
    ];
    for (odd, field) in cases {
        let reports = cpcv_series::write_reports(dir, "wf", &scheme, &|s| if s == 3 { odd } else { cpcv_series::ALL });
        let o = assemble(&table, &reports, &paths, &summary);
        assert_eq!(code(&o), 2, "{field}: {}", stderr(&o));
        assert!(stderr(&o).contains(field) && stderr(&o).contains(&reports[3]), "{field}: {}", stderr(&o));
    }
    // A report without any entry count among reports with them.
    let reports = cpcv_series::write_reports(dir, "wf", &scheme, &|s| {
        if s == 0 {
            cpcv_series::Carry { pooled_entry_times: false, window_entry_times: false, ..cpcv_series::ALL }
        } else {
            cpcv_series::ALL
        }
    });
    let o = assemble(&table, &reports, &paths, &summary);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.entry_times") && stderr(&o).contains(&reports[0]), "{}", stderr(&o));
}

/// #1236: split reports without the series give the path files and the summary of the build
/// before the issue, on the score and the target grid. Paths are relative, so the bytes hold no
/// scratch directory.
#[test]
fn cpcv_path_output_unchanged_without_series() {
    let s = Scratch::new("cpcv-series-golden");
    let dir = s.path();
    let (_, scheme) = cpcv_series::write_table(dir);
    let mut lines = Vec::new();
    for (tag, carry) in [("score", cpcv_series::NONE), ("target", cpcv_series::Carry { target: true, ..cpcv_series::NONE })] {
        let reports = cpcv_series::write_reports(dir, tag, &scheme, &|_| carry);
        let names: Vec<String> = reports.iter().map(|r| Path::new(r).file_name().unwrap().to_string_lossy().into_owned()).collect();
        let (out, summary) = (format!("paths_{tag}"), format!("summary_{tag}.json"));
        let mut args = vec!["trials", "--cpcv-assemble"];
        args.extend(names.iter().map(String::as_str));
        args.extend_from_slice(&["--splits", "splits.csv", "--out-dir", &out, "--summary", &summary]);
        let o = rcsr_cmd::rcsr().current_dir(dir).args(&args).output().expect("run rcsr");
        ok(&o);
        for j in 1..=scheme.paths() {
            let file = format!("{out}/path_{j}.json");
            lines.push(format!("{file} {}", rcsr::manifest::sha256_hex(&std::fs::read(dir.join(&file)).unwrap())));
        }
        lines.push(format!("{summary} {}", rcsr::manifest::sha256_hex(&std::fs::read(dir.join(&summary)).unwrap())));
    }
    let all = rcsr::manifest::sha256_hex(lines.join("\n").as_bytes());
    assert_eq!(all, PINNED_PATHS_WITHOUT_SERIES, "{}", lines.join("\n"));
}

const PINNED_PATHS_WITHOUT_SERIES: &str = "8000bc183222e3f46eddc8037d064eaba6a847e8fcdd6a206129dcb90e1b05af";

/// #1236: `rcsr protocol cpcv`, one `walkforward --emit-series` per split, the assembly and the
/// candidate table with the pilot rule on a 3-symbol panel.
#[test]
fn cpcv_emit_series_end_to_end() {
    let f = synthetic();
    let data = write_gap_panel(&f.dir, day("2021-01-01"), 8 * 24, (0, 0));
    let mut reports = Vec::new();
    for id in &f.scheme.ids {
        let windows = f.dir.join("t").join(format!("windows_{id}.csv")).to_string_lossy().into_owned();
        let out = f.dir.join(format!("wf_series_{id}.json")).to_string_lossy().into_owned();
        ok(&rcsr(&[
            "walkforward", "--data", &data, "--target", "y", "--time", "ts", "--group-by", "sym", "--expr", "a", "--vars", "a,b",
            "--windows", &windows, "--pool-reset-flat", "--horizon", "3600", "--threshold-fixed", "0", "--period-grid", "target",
            "--emit-series", "--json", "--out", &out,
        ]));
        let r = read_json(Path::new(&out));
        let per: f64 = arr(&r, &["windows"]).iter().map(|w| num(w, &["entry_times"])).sum();
        assert_eq!(per, num(&r, &["pooled", "entry_times"]), "{id}");
        reports.push(out);
    }
    let paths = f.dir.join("series_paths");
    ok(&assemble(&f.table, &reports, &paths, &f.dir.join("series_summary.json")));
    let cal = f.dir.join("cal.csv");
    std::fs::write(&cal, format!("label,start,end\nbull,{},{}\nbear,{},{}\n", day("2021-01-01"), day("2021-01-05"), day("2021-01-05"), day("2021-01-09"))).unwrap();
    let rule = f.dir.join("rule.json");
    std::fs::write(&rule, cpcv_series::PILOT_RULE).unwrap();
    let files: Vec<String> = (1..=3).map(|j| paths.join(format!("path_{j}.json")).to_string_lossy().into_owned()).collect();
    let (cal, rule) = (cal.to_string_lossy().into_owned(), rule.to_string_lossy().into_owned());
    let mut args = vec!["trials", "--wf"];
    args.extend(files.iter().map(String::as_str));
    args.extend_from_slice(&["--table", "--rule", &rule, "--regimes", &cal, "--draws", "50"]);
    let o = rcsr(&args);
    ok(&o);
    let j = parse(&String::from_utf8_lossy(&o.stdout));
    let rows = arr(&j, &["rows"]);
    assert_eq!(rows.len(), 3);
    for r in rows {
        for m in cpcv_series::PILOT_METRICS {
            assert!(num(r, &["metrics", m]).is_finite(), "{m}");
        }
        let bear = arr(r, &["strata"]).iter().find(|c| at(c, &["label"]).as_str() == Some("bear")).expect("a bear stratum");
        assert!(num(bear, &["alpha_bh"]).is_finite(), "alpha_bh:bear");
    }
}
