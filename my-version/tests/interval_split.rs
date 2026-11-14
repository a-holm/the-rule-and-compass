//! #1173: interval-list splits: purge, lookback embargo, book segments, robust
//! blocks per validation interval and the interval-aware in-sample guard.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::data::{self, SplitMode, SplitSpec, TradingSlice};
use rcsr::devlog::Json;
use rcsr::intervals::{Declared, EmbargoSource, IntervalSplit, Role};
use rcsr::trading::{FitnessBook, TradingRows, TradingSpec};
use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const H: i64 = 3600;
const D: i64 = 86_400;
/// 2020-09-13T12:00:00Z, on the hour.
const T0: i64 = 1_599_998_400;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Hourly rows of one symbol: `r16` and `r96` are registry names (lookbacks of
/// 16 and 96 bars), `r672` reads 28 days back; `r` is the per-row log return and
/// `y` the target.
fn write_series(dir: &Path, hours: usize) -> String {
    let mut text = String::from("ts,r16,r96,r672,r,y\n");
    for i in 0..hours {
        let f = i as f64;
        let a = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let b = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let c = frac(f * 0.569_840_290_998_053_3) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * a + 0.004 * noise;
        text.push_str(&format!("{},{},{},{},{},{}\n", T0 + H * i as i64, a, b, c, r, r));
    }
    let path = dir.join("series.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// A forward-return target spec on `y` with return column `r`.
fn write_spec(dir: &Path, name: &str, horizon: i64, entry_offset: i64) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":{horizon},\"entry_offset_seconds\":{entry_offset},\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":4.0,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}]}}"
    );
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_table(dir: &Path, name: &str, rows: &[(&str, &str, i64, i64)]) -> String {
    let mut text = String::from("split_id,role,start,end\n");
    for (id, role, a, b) in rows {
        text.push_str(&format!("{},{},{},{}\n", id, role, a, b));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn hour(h: i64) -> i64 {
    T0 + h * H
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

fn parse(text: &str) -> Json {
    Json::parse(text.trim()).unwrap_or_else(|| panic!("not json: {text}"))
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

/// A short CPU search; `extra` replaces a default flag it names.
fn search(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--time", "ts", "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "5",
        "--metric", "squared", "--json", "--quiet",
    ];
    for (flag, value) in [("--vars", "r16,r96"), ("--generations", "6"), ("--pop", "24")] {
        if !extra.contains(&flag) {
            args.extend_from_slice(&[flag, value]);
        }
    }
    args.extend_from_slice(extra);
    rcsr(&args)
}

fn ok_json(out: &Output) -> Json {
    assert_eq!(code(out), 0, "{}", stderr(out));
    parse(&String::from_utf8_lossy(&out.stdout))
}

#[test]
fn one_interval_split_equals_validation_frac() {
    let s = Scratch::new("interval-split-identity");
    let dir = s.path();
    let data = write_series(dir, 400);
    // A three-hour label: the purge drops two rows before validation and two before test.
    let spec = write_spec(dir, "spec.json", 3 * H, 0);
    let cp_a = dir.join("a.cp").to_string_lossy().into_owned();
    let cp_b = dir.join("b.cp").to_string_lossy().into_owned();
    // --test-frac 0.25 leaves 300 training hours; --validation-frac 0.2 holds out the last 60.
    let a = ok_json(&search(&data, &spec, &["--split", "temporal", "--validation-frac", "0.2", "--save", &cp_a]));
    let table = write_table(dir, "t.csv", &[("one", "train", hour(0), hour(240)), ("one", "validation", hour(240), hour(300)), ("one", "test", hour(300), hour(400))]);
    let b = ok_json(&search(&data, &spec, &["--split-table", &table, "--split-id", "one", "--save", &cp_b]));
    assert_eq!(at(&b, &["split", "mode"]).as_str(), Some("temporal-intervals"));
    for key in ["train_rows", "validation_rows", "test_rows"] {
        assert_eq!(num(&a, &["split", key]), num(&b, &["split", key]), "{key}");
    }
    assert_eq!(num(&b, &["split", "train_rows"]), 238.0, "two rows purged before validation");
    let front = |j: &Json| format!("{:?}", at(j, &["pareto"]));
    assert_eq!(front(&a), front(&b), "the fronts differ");
    let hash_line = |p: &str| {
        std::fs::read_to_string(p).unwrap().lines().find(|l| l.starts_with("archive_hash ")).expect("archive_hash line").to_string()
    };
    assert_eq!(hash_line(&cp_a), hash_line(&cp_b));
}

#[test]
fn embargo_auto_follows_the_lookback() {
    let s = Scratch::new("interval-split-embargo");
    let dir = s.path();
    let data = write_series(dir, 2400);
    let table = write_table(
        dir,
        "t.csv",
        &[("c", "train", hour(0), hour(720)), ("c", "test", hour(720), hour(960)), ("c", "train", hour(960), hour(2400))],
    );
    let vars = ["--vars", "r16,r672", "--split-table", &table, "--split-id", "c", "--generations", "1", "--pop", "8"];
    let h1 = write_spec(dir, "h1.json", H, H);
    let j = ok_json(&search(&data, &h1, &vars));
    assert_eq!(num(&j, &["split", "embargo_seconds"]), (28 * D) as f64);
    assert_eq!(at(&j, &["split", "embargo_source"]).as_str(), Some("lookback"));
    assert_eq!(at(&j, &["split", "embargo_column"]).as_str(), Some("r672"));
    assert_eq!(num(&j, &["split", "interval_purge_seconds"]), (2 * H) as f64);
    let h168 = write_spec(dir, "h168.json", 168 * H, H);
    let j = ok_json(&search(&data, &h168, &vars));
    assert_eq!(num(&j, &["split", "embargo_seconds"]), (28 * D) as f64, "the 169 h label purge is smaller");
    assert_eq!(num(&j, &["split", "interval_purge_seconds"]), (169 * H) as f64);
    let mut flagged = vars.to_vec();
    flagged.extend_from_slice(&["--embargo", "3600"]);
    let out = search(&data, &h1, &flagged);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("below the automatic embargo 2419200"), "{}", stderr(&out));
    // The kept training rows after the test interval start at its end plus the embargo.
    let kept = at(&j, &["split", "effective_intervals", "train"]);
    assert_eq!(format!("{:?}", kept), format!("{:?}", parse(&format!("[[{},{}],[{},{}]]", hour(0), hour(720) - 169 * H + 1, hour(960) + 28 * D, hour(2400)))));
}

/// The rows of `hours` hourly stamps for two groups, in canonical (group, time) order.
fn two_group_slice(hours: &[i64]) -> TradingSlice {
    let mut time = Vec::new();
    let mut group = Vec::new();
    let mut ret = Vec::new();
    for g in 0..2u64 {
        for &h in hours {
            time.push(hour(h) as f64);
            group.push(g);
            ret.push(0.01 * (frac(h as f64 * 0.618_033_988_749_894_9 + g as f64 * 0.3) - 0.45));
        }
    }
    TradingSlice { rows: TradingRows { time, group, funding: Vec::new() }, ret }
}

fn book_spec() -> TradingSpec {
    TradingSpec { cost_bps: 5.0, min_trades: 0, ..TradingSpec::default() }
}

#[test]
fn training_intervals_are_book_segments() {
    let first: Vec<i64> = (0..120).collect();
    let second: Vec<i64> = (300..420).collect();
    let all: Vec<i64> = first.iter().chain(&second).copied().collect();
    let spec = book_spec();
    let joint = FitnessBook::with_segments(&two_group_slice(&all), &spec, H as f64, 1, 1.0, &[hour(0) as f64, hour(300) as f64]).unwrap();
    let a = FitnessBook::new(&two_group_slice(&first), &spec, H as f64).unwrap();
    let b = FitnessBook::new(&two_group_slice(&second), &spec, H as f64).unwrap();
    assert!((joint.bound() - (a.bound() + b.bound())).abs() <= 1e-12, "bound {} vs {} + {}", joint.bound(), a.bound(), b.bound());
    let carried = FitnessBook::new(&two_group_slice(&all), &spec, H as f64).unwrap();
    assert_ne!(carried.hash(), joint.hash(), "the segments change the book");
    for seed in 0..6u64 {
        // Long in the last row of the first interval and in the first row of the second: a carried
        // book would hold through the gap and skip one exit and one entry.
        let pred = |hours: &[i64]| -> Vec<f64> {
            (0..2)
                .flat_map(|g| hours.iter().map(move |&h| if (h as u64 + seed + g) % 3 == 0 || h == 119 || h == 300 { 1.0 } else { -1.0 }))
                .collect()
        };
        let (pa, pb) = (pred(&first), pred(&second));
        let pj: Vec<f64> = {
            let (mut v, n) = (Vec::new(), first.len());
            for g in 0..2 {
                v.extend_from_slice(&pa[g * n..(g + 1) * n]);
                v.extend_from_slice(&pb[g * second.len()..(g + 1) * second.len()]);
            }
            v
        };
        let (tj, ta, tb) = (joint.totals(&pj), a.totals(&pa), b.totals(&pb));
        assert_eq!(tj.flips, ta.flips + tb.flips);
        assert_eq!(tj.exits, ta.exits + tb.exits);
        assert_eq!(tj.long_rows, ta.long_rows + tb.long_rows);
        assert!((tj.gross - (ta.gross + tb.gross)).abs() <= 1e-12);
        let net = a.net_of(&ta) + b.net_of(&tb);
        let want = (a.bound() + b.bound() - net) / (a.bound() + b.bound());
        assert!((joint.regret(&pj) - want).abs() <= 1e-12, "regret {} vs {}", joint.regret(&pj), want);
        assert_ne!(carried.totals(&pj).exits, tj.exits, "seed {seed}: the carried book pays fewer exits");
    }
    let one = FitnessBook::with_segments(&two_group_slice(&all), &spec, H as f64, 1, 1.0, &[hour(0) as f64]).unwrap();
    assert_eq!(one.hash(), carried.hash(), "one segment is the plain book");
}

/// Train 0..300 h, validation 300..360 h, a test interval, train again and two
/// more validation intervals: three validation intervals in all.
fn cpcv_table(dir: &Path) -> String {
    write_table(
        dir,
        "cpcv.csv",
        &[
            ("g2", "train", hour(0), hour(300)),
            ("g2", "validation", hour(300), hour(360)),
            ("g2", "test", hour(360), hour(600)),
            ("g2", "train", hour(600), hour(900)),
            ("g2", "validation", hour(900), hour(960)),
            ("g2", "train", hour(960), hour(1260)),
            ("g2", "validation", hour(1260), hour(1320)),
        ],
    )
}

fn pnl_search(dir: &Path, data: &str, table: &str, extra: &[&str]) -> (Json, String) {
    let spec = write_spec(dir, "pnl.json", H, 0);
    let mut args = vec![
        "--split-table", table, "--split-id", "g2", "--objective", "sharpe", "--fitness", "pnl", "--arm", "A0",
    ];
    args.extend_from_slice(extra);
    let out = search(data, &spec, &args);
    let j = ok_json(&out);
    let path = dir.join("run.json");
    std::fs::write(&path, &out.stdout).unwrap();
    (j, path.to_string_lossy().into_owned())
}

#[test]
fn robust_rule_uses_validation_intervals() {
    // Library: one block per validation interval; with no cost each block alpha is the alpha of
    // that interval priced alone.
    let hours: Vec<i64> = (300..360).chain(900..960).chain(1260..1320).collect();
    let slice = two_group_slice(&hours);
    let rows = slice.rows.clone();
    let order: Vec<usize> = {
        let mut o: Vec<usize> = (0..rows.time.len()).collect();
        o.sort_by(|&x, &y| rows.time[x].total_cmp(&rows.time[y]).then(x.cmp(&y)));
        o
    };
    let pick = |idx: &[usize]| TradingRows { time: idx.iter().map(|&i| rows.time[i]).collect(), group: idx.iter().map(|&i| rows.group[i]).collect(), funding: Vec::new() };
    let sorted = pick(&order);
    let ret: Vec<f64> = order.iter().map(|&i| slice.ret[i]).collect();
    let pred: Vec<f64> = (0..ret.len()).map(|k| if (k * 7) % 5 < 2 { 1.0 } else { -1.0 }).collect();
    let cuts = [hour(300) as f64, hour(900) as f64, hour(1260) as f64];
    let spec = TradingSpec::default();
    let joint = rcsr::trading::blocked_alpha_cuts(&pred, &ret, &sorted, &spec, &cuts).unwrap();
    for (k, (a, b)) in [(300, 360), (900, 960), (1260, 1320)].iter().enumerate() {
        let idx: Vec<usize> = (0..ret.len()).filter(|&i| sorted.time[i] >= hour(*a) as f64 && sorted.time[i] < hour(*b) as f64).collect();
        let part = pick(&idx.iter().map(|&i| order[i]).collect::<Vec<_>>());
        let alone = rcsr::trading::blocked_alpha(
            &idx.iter().map(|&i| pred[i]).collect::<Vec<_>>(),
            &idx.iter().map(|&i| ret[i]).collect::<Vec<_>>(),
            &part,
            &spec,
            1,
        )
        .unwrap();
        assert!((joint.block_alpha[k] - alone.block_alpha[0]).abs() <= 1e-15, "interval {k}: {} vs {}", joint.block_alpha[k], alone.block_alpha[0]);
    }
    // Search: the run records three blocks taken from the intervals.
    let s = Scratch::new("interval-split-robust");
    let dir = s.path();
    let data = write_series(dir, 1400);
    let table = cpcv_table(dir);
    let (j, _) = pnl_search(dir, &data, &table, &["--select-rule", "robust", "--select-blocks", "intervals"]);
    assert_eq!(num(&j, &["objective", "selection_rule", "blocks"]), 3.0);
    assert_eq!(at(&j, &["objective", "selection_rule", "blocks_source"]).as_str(), Some("intervals"));
    if let Json::Arr(members) = at(&j, &["objective", "selection_rule", "members"]) {
        for m in members {
            match at(m, &["block_alpha"]) {
                Json::Arr(a) => assert_eq!(a.len(), 3),
                _ => panic!("block_alpha is not an array"),
            }
        }
    }
    // Two validation intervals at least.
    let one = write_table(dir, "one.csv", &[("g2", "train", hour(0), hour(300)), ("g2", "validation", hour(300), hour(400))]);
    let spec = write_spec(dir, "pnl1.json", H, 0);
    let out = search(&data, &spec, &["--split-table", &one, "--split-id", "g2", "--objective", "sharpe", "--select-rule", "robust", "--select-blocks", "intervals"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("at least 2 validation intervals"), "{}", stderr(&out));
}

#[test]
fn pnl_fitness_accepts_interval_splits() {
    let s = Scratch::new("interval-split-pnl");
    let dir = s.path();
    let data = write_series(dir, 1400);
    let table = cpcv_table(dir);
    let (j, _) = pnl_search(dir, &data, &table, &[]);
    assert_eq!(at(&j, &["split", "mode"]).as_str(), Some("temporal-intervals"));
    assert_eq!(at(&j, &["search_fitness", "kind"]).as_str(), Some("long_pnl_regret"));
    assert_eq!(num(&j, &["search_fitness", "groups"]), 1.0, "groups count symbols, not book segments");
}

fn walkforward(data: &str, run: &str, window: &[&str]) -> Output {
    let mut args = vec!["walkforward", "--data", data, "--time", "ts", "--from-run", run, "--json"];
    args.extend_from_slice(window);
    rcsr(&args)
}

#[test]
fn interval_guard_allows_cpcv_test_groups() {
    let s = Scratch::new("interval-split-guard-ok");
    let dir = s.path();
    let data = write_series(dir, 1400);
    let table = cpcv_table(dir);
    let (j, run) = pnl_search(dir, &data, &table, &[]);
    // The old rule would refuse it: the test interval starts before the last training row.
    let last_train = match at(&j, &["split", "train_time"]) {
        Json::Arr(a) => a[1].as_f64().unwrap(),
        _ => panic!("split.train_time is not an array"),
    };
    assert!(last_train > hour(360) as f64);
    let out = walkforward(&data, &run, &["--windows", &table, "--split-id", "g2"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    let wf = parse(&String::from_utf8_lossy(&out.stdout));
    let w = match at(&wf, &["windows"]) {
        Json::Arr(a) => a[0].clone(),
        _ => panic!("windows is not an array"),
    };
    assert_eq!(at(&w, &["label"]).as_str(), Some("g2-test1"));
    assert_eq!(num(&w, &["test_start"]), hour(360) as f64);
    assert!(w.get("in_sample").is_none_or(|v| matches!(v, Json::Bool(false))), "{w:?}");
    let run_file = rcsr::run_json::RunFile::load(&run).unwrap();
    assert!(!run_file.in_sample_window(hour(360) as f64, hour(600) as f64));
    assert!(run_file.in_sample(hour(360) as f64), "the last-seen rule refuses the same window");
}

#[test]
fn interval_guard_refuses_overlap() {
    let s = Scratch::new("interval-split-guard-refuse");
    let dir = s.path();
    let data = write_series(dir, 1400);
    let table = cpcv_table(dir);
    let (_, run) = pnl_search(dir, &data, &table, &[]);
    let run_file = rcsr::run_json::RunFile::load(&run).unwrap();
    // One second earlier meets the label of the last kept training row before the test interval;
    // one second later is read by the first kept training row after the embargo.
    assert!(run_file.in_sample_window((hour(360) - 1) as f64, hour(600) as f64));
    let embargo = run_file.intervals.as_ref().unwrap().embargo;
    assert!(run_file.in_sample_window(hour(360) as f64, (hour(600) + 1) as f64), "embargo {embargo}");
    let w = format!("w,{},{},{},{}", hour(0), hour(100), hour(700), hour(800));
    let out = walkforward(&data, &run, &["--window", &w]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("window 'w' tests rows the run trained or selected on"), "{}", stderr(&out));
    let out = walkforward(&data, &run, &["--window", &w, "--allow-in-sample-windows"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
}

#[test]
fn split_table_respects_locked_ranges() {
    let s = Scratch::new("interval-split-locked");
    let dir = s.path();
    let data = write_series(dir, 1400);
    let spec = write_spec(dir, "spec.json", H, 0);
    let table = write_table(dir, "t.csv", &[("a", "train", hour(0), hour(600)), ("a", "test", hour(600), hour(800))]);
    // The test interval ends one hour after the locked range starts.
    let lock = format!("{},{}", hour(799), hour(900));
    let out = search(&data, &spec, &["--split-table", &table, "--split-id", "a", "--lockbox", &lock]);
    assert_eq!(code(&out), 3, "{}", stderr(&out));
    assert!(stderr(&out).contains("reach the declared locked test"), "{}", stderr(&out));
    // The embargo after the range reaches a training interval that starts inside it.
    let after = write_table(dir, "after.csv", &[("a", "train", hour(0), hour(400)), ("a", "train", hour(905), hour(1300))]);
    let lock = format!("{},{}", hour(500), hour(900));
    let out = search(&data, &spec, &["--split-table", &after, "--split-id", "a", "--lockbox", &lock]);
    assert_eq!(code(&out), 3, "{}", stderr(&out));
    // Clear of the range: purge before it and the store embargo after it.
    let clear = write_table(dir, "clear.csv", &[("a", "train", hour(0), hour(400)), ("a", "test", hour(400), hour(499))]);
    let out = search(&data, &spec, &["--split-table", &clear, "--split-id", "a", "--lockbox", &lock]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
}

#[test]
fn split_table_refusals_exit_2() {
    let s = Scratch::new("interval-split-refusals");
    let dir = s.path();
    let data = write_series(dir, 600);
    let spec = write_spec(dir, "spec.json", H, 0);
    let overlap = write_table(dir, "o.csv", &[("a", "train", hour(0), hour(300)), ("a", "test", hour(299), hour(400))]);
    let out = search(&data, &spec, &["--split-table", &overlap, "--split-id", "a"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("overlaps"), "{}", stderr(&out));
    let table = write_table(dir, "t.csv", &[("a", "train", hour(0), hour(300)), ("a", "test", hour(300), hour(400))]);
    for extra in [&["--validation-frac", "0.2"][..], &["--test-frac", "0.3"][..], &["--split", "temporal"][..]] {
        let mut args = vec!["--split-table", table.as_str(), "--split-id", "a"];
        args.extend_from_slice(extra);
        let out = search(&data, &spec, &args);
        assert_eq!(code(&out), 2, "{extra:?}: {}", stderr(&out));
    }
    // A training interval inside the embargo after the test interval has no rows.
    let empty = write_table(dir, "e.csv", &[("a", "train", hour(0), hour(300)), ("a", "test", hour(300), hour(400)), ("a", "train", hour(400), hour(402))]);
    let out = search(&data, &spec, &["--split-table", &empty, "--split-id", "a"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("has no rows after the purge"), "{}", stderr(&out));
    let out = search(&data, &spec, &["--split-table", &table]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    let out = search(&data, &spec, &["--split", "temporal", "--embargo", "3600"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
}

/// The rows a prepared split trains on, as stamps.
fn train_stamps(table: &data::Table, vars: &[&str], iv: IntervalSplit, purge: f64) -> (Vec<i64>, Vec<i64>) {
    let spec = SplitSpec {
        mode: SplitMode::TemporalIntervals,
        time_col: Some(0),
        purge_seconds: Some(purge),
        intervals: Some(iv),
        ..SplitSpec::default()
    };
    let vars: Vec<String> = vars.iter().map(|v| v.to_string()).collect();
    let p = data::prepare_full(table, &vars, "y", &spec, &mut rcsr::rng::Mt::new(1)).unwrap();
    let keys = p.keys.expect("time keys");
    let ts = |v: &[f64]| v.iter().map(|&t| t as i64).collect::<Vec<i64>>();
    (ts(&keys.train), ts(&keys.test))
}

#[test]
fn planted_leaks_are_excluded_from_training() {
    // `peek` at row t is the mean return over [t - 500 h, t]: after the test interval it reads test
    // bars for 500 hours. `y` at row t is the return over [t, t + 72 h): before the test interval it
    // crosses into it for 72 hours.
    let n = 3000i64;
    let mut text = String::from("ts,peek,y\n");
    for h in 0..n {
        text.push_str(&format!("{},{},{}\n", hour(h), frac(h as f64 * 0.618_033_988_749_894_9), frac(h as f64 * 0.75)));
    }
    let table = data::parse_table(&text).unwrap();
    let (test_a, test_b) = (hour(1000), hour(1100));
    let purge = 73 * H;
    let lookback = 500 * H;
    let d = Declared {
        split_id: "leak".to_string(),
        train: vec![(hour(0), test_a), (test_b, hour(n))],
        validation: Vec::new(),
        test: vec![(test_a, test_b)],
    };
    let iv = IntervalSplit::new(d, purge, lookback, EmbargoSource::Lookback, Some("peek".to_string()));
    assert_eq!(iv.role(test_a - 72 * H), None);
    let (train, test) = train_stamps(&table, &["peek"], iv, purge as f64);
    assert_eq!(test.len(), 100);
    assert!(!train.is_empty());
    for &t in &train {
        let label_crosses = t + purge > test_a && t < test_b;
        let feature_peeks = t >= test_b && t - lookback < test_b;
        assert!(!label_crosses, "the label of training row {} reaches the test interval", t);
        assert!(!feature_peeks, "the feature of training row {} reads test bars", t);
    }
    // Every other row of the training intervals is kept.
    assert_eq!(train.len() as i64, (1000 - 72) + (n - 1100 - 500));
    let role_test = IntervalSplit::new(
        Declared { split_id: "x".to_string(), train: vec![(0, 10)], validation: vec![], test: vec![(10, 20)] },
        1,
        1,
        EmbargoSource::Purge,
        None,
    );
    assert_eq!(role_test.role(15), Some(Role::Test));
}

#[test]
fn resume_with_other_intervals_exits_2() {
    let s = Scratch::new("interval-split-resume");
    let dir = s.path();
    let data = write_series(dir, 600);
    let spec = write_spec(dir, "spec.json", H, 0);
    let table = write_table(
        dir,
        "t.csv",
        &[("a", "train", hour(0), hour(300)), ("a", "test", hour(300), hour(400)), ("b", "train", hour(0), hour(290)), ("b", "test", hour(300), hour(400))],
    );
    let cp = dir.join("a.cp").to_string_lossy().into_owned();
    let out = search(&data, &spec, &["--split-table", &table, "--split-id", "a", "--save", &cp, "--generations", "2"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(std::fs::read_to_string(&cp).unwrap().lines().any(|l| l.starts_with("split_intervals ")));
    let out = search(&data, &spec, &["--split-table", &table, "--split-id", "b", "--resume", &cp, "--generations", "3"]);
    assert_eq!(code(&out), 2, "{}", stderr(&out));
    assert!(stderr(&out).contains("split intervals mismatch"), "{}", stderr(&out));
    let plain = dir.join("p.cp").to_string_lossy().into_owned();
    let out = search(&data, &spec, &["--split", "temporal", "--save", &plain, "--generations", "2"]);
    assert_eq!(code(&out), 0, "{}", stderr(&out));
    assert!(!std::fs::read_to_string(&plain).unwrap().contains("split_intervals"), "a run without a table writes no line");
}
