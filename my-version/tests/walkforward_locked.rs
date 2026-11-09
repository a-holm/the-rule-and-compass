//! Walk-forward over a `.rcsb` store never depends on rows inside the declared
//! locked test. Two synthetic stores are identical outside the range and
//! poisoned inside it; every report of a window that ends before the range
//! must be byte-identical. Fixtures are synthetic; the range is a stand-in in
//! 2024.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::lockbox::{self, Lockbox};
use rcsr::{data, store};
use std::path::{Path, PathBuf};

/// 2024-01-01T00:00:00Z.
const T0: i64 = 1_704_067_200;
const H: i64 = 3600;
const LOCK_START: i64 = T0 + 400 * H;
const LOCK_END: i64 = T0 + 500 * H;

#[derive(Clone, Copy, PartialEq)]
enum Fixture {
    Clean,
    Poisoned,
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(name);
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = rcsr_cmd::rcsr().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Hourly panel of 3 symbols, 500 bars each. The label at bar i is the return
/// of bar i+1. The poisoned variant changes every value of the bars inside the
/// locked range (huge, NaN, different returns), drops some of them, adds extra
/// and duplicate rows inside the range, and appends rows far after its end.
fn panel_csv(kind: Fixture) -> String {
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    if kind == Fixture::Poisoned {
        // symbols that exist only inside the range, one sorting and appearing first, one last
        for sym in ["a0", "zz"] {
            for i in 400..450i64 {
                csv.push_str(&format!("{},{sym},{},{},{}\n", T0 + i * H, i, -i, i));
            }
        }
    }
    for g in 0..3i64 {
        for i in 0..500i64 {
            let ts = T0 + i * H;
            let locked = ts >= LOCK_START && ts < LOCK_END;
            let r = |k: i64| (((k * 29 + g * 7) % 11) as f64 - 5.0) * 0.0006 + (((k * 13 + g) % 5) as f64 - 2.0) * 0.0002;
            let f1 = if i == 0 { 0.0 } else { r(i - 1) * 100.0 };
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = r(i + 1);
            if kind == Fixture::Poisoned && locked {
                match i % 7 {
                    0 => continue,
                    1 => csv.push_str(&format!("{ts},s{g},NaN,{f2},NaN\n")),
                    2 => csv.push_str(&format!("{ts},s{g},1e300,-1e300,1e300\n")),
                    3 => csv.push_str(&format!("{ts},s{g},{},{},{}\n", -f1 - 9.0, f2 * 50.0, -y + 0.05)),
                    _ => csv.push_str(&format!("{ts},s{g},{},{},{}\n", f1 + 3.0, f2, y + 0.02)),
                }
                if i % 5 == 0 {
                    csv.push_str(&format!("{},s{g},7,7,7\n", ts + 1));
                    csv.push_str(&format!("{ts},s{g},8,8,8\n"));
                }
            } else {
                csv.push_str(&format!("{ts},s{g},{f1},{f2},{y}\n"));
            }
        }
        if kind == Fixture::Poisoned {
            for k in 0..30i64 {
                csv.push_str(&format!("{},s{g},{},{},{}\n", T0 + (900 + k) * H, 1e9, 1e9, -1e9));
            }
        }
    }
    csv
}

fn write_data(dir: &Path, name: &str, kind: Fixture, declared: bool) -> String {
    let table = data::parse_table(&panel_csv(kind)).unwrap();
    let path = dir.join(format!("{name}.rcsb"));
    let p = s(&path);
    store::write_store_with_time(&p, &table, 97, Some(0)).unwrap();
    if declared {
        lockbox::write_sidecar(&p, &Lockbox { start: LOCK_START, end: LOCK_END }, "test").unwrap();
    }
    p
}

fn window(test_end: i64) -> String {
    format!("f1,{},{},{},{}", T0, T0 + 150 * H, T0 + 160 * H, test_end)
}

const RUN: &str = r#"{"tool":"rcsr","variables":["f1","f2"],"target_spec":{"return_column":"y"},
    "split":{"train_time":[0,100],"validation_time":[101,200],"required_purge_seconds":3600,"purge_seconds":3600},
    "search":{"candidates_evaluated":4321},
    "objective":{"spec":{"threshold":0,"per_side_bps":5}},
    "pareto":[{"expression":"f1"},{"expression":"f1 * 2 + f2"},{"expression":"0.5 - f2"}]}"#;

/// The candidate sources the same run is repeated with.
fn variants(dir: &Path) -> Vec<(&'static str, Vec<String>)> {
    let run = dir.join("run.json");
    std::fs::write(&run, RUN).unwrap();
    let v = |a: &[&str]| a.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    vec![
        ("plain", v(&["--expr", "f1", "--vars", "f1,f2"])),
        (
            "candidates",
            v(&["--expr", "f1", "--expr", "f2 - f1", "--vars", "f1,f2", "--validation-frac", "0.2"]),
        ),
        (
            "front",
            vec!["--front".to_string(), s(&run), "--validation-frac".to_string(), "0.2".to_string()],
        ),
        (
            "front-batch",
            vec!["--front".to_string(), s(&run), "--validation-frac".to_string(), "0.2".to_string(), "--batch-pnl".to_string()],
        ),
        (
            "bootstrap",
            v(&["--expr", "f1", "--vars", "f1,f2", "--bootstrap", "100", "--trials", "50"]),
        ),
    ]
}

fn wf(data: &str, window: &str, extra: &[String]) -> (i32, String, String) {
    let mut a: Vec<&str> = vec![
        "walkforward", "--data", data, "--target", "y", "--time", "ts", "--group-by", "sym", "--horizon", "3600",
        "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", window,
    ];
    a.extend(extra.iter().map(String::as_str));
    rcsr(&a)
}

/// Drops the tokens that carry wall clock, timing-derived ledger ratios and the data path.
fn stable(out: &str, data: &str) -> String {
    out.replace(data, "<data>")
        .split(',')
        .filter(|t| {
            !(t.contains("report_sha256")
                || t.contains("_s\"")
                || t.contains("seconds")
                || t.contains("wall")
                || t.contains("elapsed")
                || t.contains("denominator")
                || t.contains("\"numerator\"")
                || t.contains("\"device_fraction\"")
                || t.contains("\"duty_run\"")
                || t.contains("declared_in")
                || t.contains("bytes_per_s")
                || t.contains("rows_per_s"))
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn sha(out: &str) -> String {
    let at = out.find("\"report_sha256\":").unwrap_or_else(|| panic!("no hash in {out}"));
    out[at..].chars().take(90).collect()
}

#[test]
fn the_store_path_ignores_every_value_inside_the_declared_locked_range() {
    let (_scratch, dir) = scratch("diff");
    let clean = write_data(&dir, "clean", Fixture::Clean, true);
    let poisoned = write_data(&dir, "poisoned", Fixture::Poisoned, true);
    assert_ne!(std::fs::read(&clean).unwrap(), std::fs::read(&poisoned).unwrap());

    // the last window row is one horizon before the locked start
    for end in [LOCK_START - H, LOCK_START - 2 * H] {
        for (name, extra) in variants(&dir) {
            let w = window(end);
            let (ca, oa, ea) = wf(&clean, &w, &extra);
            let (cb, ob, eb) = wf(&poisoned, &w, &extra);
            assert_eq!((ca, cb), (0, 0), "{name}: {ea}{eb}");
            assert!(oa.contains("\"report_sha256\":"), "{name}: {oa}");
            assert_eq!(stable(&oa, &clean), stable(&ob, &poisoned), "{name} end {end}");
            // the batch report embeds its own timings, so only the other reports have a reproducible hash
            if !extra.iter().any(|a| a == "--batch-pnl") {
                assert_eq!(sha(&oa), sha(&ob), "{name} end {end}");
            }
        }
    }
}

#[test]
fn the_poison_changes_the_report_when_the_range_is_not_declared() {
    let (_scratch, dir) = scratch("power");
    let clean = write_data(&dir, "clean", Fixture::Clean, false);
    let poisoned = write_data(&dir, "poisoned", Fixture::Poisoned, false);
    // 2024 data is outside the protocol default range, so this window is legal here
    let w = window(LOCK_END - 10 * H);
    let (_, extra) = variants(&dir).remove(0);
    let (ca, _, ea) = wf(&clean, &w, &extra);
    let (cb, _, eb) = wf(&poisoned, &w, &extra);
    assert_eq!(ca, 0, "{ea}");
    assert_eq!(cb, 2, "{eb}");
    assert!(eb.contains("overlapping returns"), "{eb}");
}

#[test]
fn a_window_that_reaches_the_range_is_refused_without_the_sealed_read() {
    let (_scratch, dir) = scratch("refuse");
    let poisoned = write_data(&dir, "poisoned", Fixture::Poisoned, true);
    let (_, extra) = variants(&dir).remove(0);
    for end in [LOCK_START, LOCK_START + 10 * H, LOCK_END] {
        let (code, out, err) = wf(&poisoned, &window(end), &extra);
        assert_eq!(code, 3, "{out}{err}");
        assert!(err.contains("reaches the declared locked test"), "{err}");
        assert!(out.trim().is_empty(), "{out}");
    }
    // a test block inside the range is refused too, and the sealed flag alone does not lift it
    let inside = format!("f1,{},{},{},{}", T0, T0 + 150 * H, LOCK_START + H, LOCK_START + 20 * H);
    let (code, _, err) = wf(&poisoned, &inside, &extra);
    assert_eq!(code, 3, "{err}");
    let mut sealed = extra.clone();
    sealed.push("--locked-test".to_string());
    let (code, _, err) = wf(&poisoned, &inside, &sealed);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("--locked-test needs --ledger and --sealed"), "{err}");
    // the escape hatch for the range does not exist for a declared store
    let mut none = extra.clone();
    none.extend(["--lockbox".to_string(), "none".to_string()]);
    let (code, _, err) = wf(&poisoned, &window(LOCK_START + 10 * H), &none);
    assert_ne!(code, 0, "{err}");
}

#[test]
fn locked_ranges_are_dropped_at_read_time_on_the_store_and_the_table_path() {
    use rcsr::walkforward::{self, EvalSpec, ProtocolOptions, Window};
    let mut csv = String::from("ts,sym,x,target\n");
    for i in 0..40 {
        csv.push_str(&format!("{},{},{},{}\n", i, if i % 2 == 0 { "A" } else { "B" }, i, 0.001 * (i % 5) as f64));
    }
    let table = data::parse_table(&csv).unwrap();
    let (_scratch, dir) = scratch("library");
    let path = s(&dir.join("t.rcsb"));
    store::write_store_with_time(&path, &table, 7, Some(0)).unwrap();
    let spec = EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: Some("sym".to_string()),
        windows: vec![Window { label: "w".to_string(), train_start: 0.0, train_end: 20.0, test_start: 20.0, test_end: 40.0 }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.5,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(4.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    };
    let open = ProtocolOptions::default();
    let locked = ProtocolOptions { locked_ranges: vec![(30, 36)], locked_embargo: 2, ..ProtocolOptions::default() };
    let rows = |r: &walkforward::Report| r.windows[0].test_rows;
    assert_eq!(rows(&walkforward::run_path_with(&path, &spec, &open).unwrap()), 20);
    assert_eq!(rows(&walkforward::run_table_with(&table, &spec, "t", &open).unwrap()), 20);
    // the library guard refuses a window that reaches [S - P, E + L) of a sealed range
    let reaches = |w: &Window| EvalSpec { windows: vec![w.clone()], ..spec.clone() };
    let window = |train_end: f64, test_start: f64, test_end: f64| Window { label: "w".to_string(), train_start: 0.0, train_end, test_start, test_end };
    for refused in [window(20.0, 20.0, 40.0), window(20.0, 37.0, 40.0), window(20.0, 30.0, 33.0)] {
        let spec = reaches(&refused);
        for err in [
            walkforward::run_path_with(&path, &spec, &locked).unwrap_err(),
            walkforward::run_table_with(&table, &spec, "t", &locked).unwrap_err(),
        ] {
            assert!(err.contains("reaches the declared locked test"), "{err}");
        }
    }
    let after = reaches(&window(20.0, 38.0, 40.0));
    assert_eq!(rows(&walkforward::run_path_with(&path, &after, &locked).unwrap()), 2);
    assert_eq!(rows(&walkforward::run_table_with(&table, &after, "t", &locked).unwrap()), 2);
}

#[test]
fn front_read_verifies_with_another_path_spelling() {
    let (_scratch, dir) = scratch("front-spelling");
    let data = write_data(&dir, "p", Fixture::Clean, false);
    std::fs::write(dir.join("run.json"), RUN).unwrap();
    let w = window(T0 + 300 * H);
    let front = |spelling: &str, extra: &[&str]| {
        let mut a: Vec<&str> = vec![
            "walkforward", "--data", &data, "--target", "y", "--time", "ts", "--group-by", "sym", "--horizon", "3600",
            "--cost-bps", "5", "--periods-per-year", "365", "--json", "--window", &w, "--front", spelling,
            "--validation-frac", "0.2",
        ];
        a.extend_from_slice(extra);
        let out = rcsr_cmd::rcsr().current_dir(&dir).args(&a).output().unwrap();
        (
            out.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&out.stdout).into_owned(),
            String::from_utf8_lossy(&out.stderr).into_owned(),
        )
    };
    let abs = s(&dir.join("run.json"));
    let (code, out, err) = front(&abs, &[]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains(&format!("\"run\":\"{abs}\"")), "the printed report keeps the --front path");
    let hash = out.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
    let (code, out, err) = front("run.json", &["--expect-report-sha256", &hash]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\"run\":\"run.json\""), "the printed report keeps the --front path");
}

#[test]
fn the_report_filter_drops_timing_ledger_fields_and_keeps_data_fields() {
    let a = r#"{"device_fraction":0.1,"duty_run":{"value":0.2,"numerator":0.3,"denominator":4,"units":"s/s","scope":"run"},"sharpe":1.5,"n_rows":10}"#;
    let b = r#"{"device_fraction":0.9,"duty_run":{"value":0.8,"numerator":0.7,"denominator":5,"units":"s/s","scope":"run"},"sharpe":1.5,"n_rows":10}"#;
    assert_eq!(stable(a, "/data.rcsb"), stable(b, "/data.rcsb"));
    assert_ne!(stable(a, "/data.rcsb"), stable(&a.replace("1.5", "1.6"), "/data.rcsb"));
    assert_ne!(stable(a, "/data.rcsb"), stable(&a.replace("10", "11"), "/data.rcsb"));
}

// ---------------------------------------------------------------------------
// #931: the lockbox reach at 168 and 336 hours.
// ---------------------------------------------------------------------------

#[path = "common/long_horizon.rs"]
mod lh;

#[test]
fn locked_window_inside_h168_purge_exits_3() {
    let (_scratch, dir) = scratch("h168-lock");
    let (s0, e0) = (T0 + 4000 * H, T0 + 4100 * H);
    let bars = lh::bars(T0, 168 * 30, 0.0);
    let spec = lh::spec(&lh::src(168, ",stride=168"));
    let l = rcsr::labels::build(&bars, &spec, None).unwrap();
    let mut csv = String::from("ts,sym,x,vol96,lbl_h168_y,lbl_h168_fwd\n");
    for (t, b) in bars.iter().enumerate() {
        csv.push_str(&format!("{},s0,{},0.01,{},{}\n", b.time, ((t * 7) % 11) as f64, lh::fmt(l.y[t]), lh::fmt(l.fwd[t])));
    }
    let table = data::parse_table(&csv).unwrap();
    let path = s(&dir.join("h168.rcsb"));
    store::write_store_with_time(&path, &table, 512, Some(0)).unwrap();
    lockbox::write_sidecar(&path, &Lockbox { start: s0, end: e0 }, "test").unwrap();
    let spec_path = dir.join("spec.json");
    std::fs::write(&spec_path, lh::spec_file(&spec)).unwrap();
    let wf = |test_start: i64, test_end: i64| {
        let w = format!("w,{},{},{},{}", T0, T0 + 1500 * H, test_start, test_end);
        rcsr(&[
            "walkforward", "--data", &path, "--target", "lbl_h168_y", "--target-spec", s(&spec_path).as_str(), "--expr", "x",
            "--time", "ts", "--periods-per-year", "52", "--window", &w, "--json",
        ])
    };
    let (code, out, err) = wf(s0 - 608_399, s0 - 1);
    assert_eq!(code, 3, "{out}{err}");
    assert!(err.contains("reaches the declared locked test") && err.contains("label end = stamp + 608400 s"), "{err}");
    let (code, _, err) = wf(T0 + 1500 * H + 608_400, s0 - 608_400);
    assert_eq!(code, 0, "{err}");
    // The ladder refuses a fold whose test range starts inside the purge the same way.
    let iso = |t: i64| rcsr::protocol::iso(t as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f0,{},{},{},{},{}\n",
            iso(T0),
            iso(T0 + 1500 * H),
            iso(T0 + 1000 * H),
            iso(s0 - 608_399),
            iso(s0 - 1)
        ),
    )
    .unwrap();
    let (code, _, err) = rcsr(&[
        "ladder", "--data", &path, "--target", "lbl_h168_y", "--time", "ts", "--group", "sym", "--features", "x", "--windows",
        s(&windows).as_str(), "--target-spec", s(&spec_path).as_str(), "--draws", "100", "--gbm-trees", "20", "--json",
    ]);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("reaches the declared locked test"), "{err}");
}

#[test]
fn no_h336_fit_label_reaches_the_lockbox() {
    use rcsr::ladder::{self, Panel};
    use rcsr::walkforward::{self, EvalSpec, ProtocolOptions, Window};
    let h = 336usize;
    let purge = 1_213_200i64;
    let bars = lh::bars(T0, h * 40, 0.0);
    let l = lh::build(&bars, &lh::src(h, ",stride=336"));
    let (s0, e0) = (T0 + (18 * h) as i64 * H, T0 + (18 * h) as i64 * H + 500 * H);
    let n = bars.len();
    let x: Vec<f64> = (0..n).map(|t| ((t * 7) % 11) as f64).collect();
    let time: Vec<f64> = bars.iter().map(|b| b.time as f64).collect();
    let panel = Panel::new(vec!["x".into()], vec![x.clone()], time.clone(), vec![0; n], l.y.clone(), l.fwd.clone(), vec![]).unwrap();
    let after = (e0 + purge) as f64;
    let fold = |train_end: f64| rcsr::protocol::WindowRow {
        scheme: "expanding".to_string(),
        window: Window {
            label: "f0".to_string(),
            train_start: T0 as f64,
            train_end,
            test_start: after,
            test_end: after + (8 * h) as f64 * H as f64,
        },
        validation_start: train_end - (6 * h) as f64 * H as f64,
    };
    let edge = (s0 - purge) as f64;
    let e = ladder::make_folds(&panel, &[fold(edge + H as f64)], purge as f64, &[(s0 as f64, e0 as f64)], 0.0).unwrap_err();
    assert!(e.contains("reaches the declared locked test"), "{e}");
    let folds = ladder::make_folds(&panel, &[fold(edge)], purge as f64, &[(s0 as f64, e0 as f64)], 0.0).unwrap();
    let seen: Vec<usize> = folds[0].fit.iter().chain(&folds[0].valid).copied().collect();
    assert!(seen.len() >= 15, "{}", seen.len());
    for &i in &seen {
        assert!(l.t1[i] <= s0 as f64, "row {i}: label end {} reaches the locked start", l.t1[i]);
    }
    let reaching: Vec<usize> = (0..n).filter(|&t| l.fwd[t].is_finite() && time[t] < s0 as f64 && time[t] + purge as f64 > s0 as f64).collect();
    assert!(!reaching.is_empty(), "the fixture holds a row whose label reaches the range");
    assert!(reaching.iter().all(|i| !seen.contains(i)));

    // Walk-forward: the same boundary through the library guard.
    let mut csv = String::from("ts,x,target\n");
    for t in 0..n {
        csv.push_str(&format!("{},{},{}\n", bars[t].time, x[t], lh::fmt(l.fwd[t])));
    }
    let table = data::parse_table(&csv).unwrap();
    let spec = |train_end: f64| EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: None,
        windows: vec![fold(train_end).window],
        horizon: (h as i64 * H) as f64,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.5,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: Some(26.0),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    };
    let opts = ProtocolOptions { entry_offset: H as f64, locked_ranges: vec![(s0, e0)], locked_embargo: 0, ..ProtocolOptions::default() };
    let err = walkforward::run_table_with(&table, &spec(edge + H as f64), "t", &opts).unwrap_err();
    assert!(err.contains("reaches the declared locked test"), "{err}");
    let r = walkforward::run_table_with(&table, &spec(edge), "t", &opts).unwrap();
    let last = r.windows[0].train_last_time.unwrap();
    assert!(last + purge as f64 <= s0 as f64, "last training stamp {last}");
}

#[test]
fn short_declared_embargo_keeps_h168_labels_after_the_end() {
    let (_scratch, dir) = scratch("h168-short-embargo");
    let (s0, e0) = (T0 + 1000 * H, T0 + 1100 * H);
    let purge = 608_400i64;
    let bars = lh::bars(T0, 168 * 30, 0.0);
    let spec = lh::spec(&lh::src(168, ",stride=168"));
    let l = rcsr::labels::build(&bars, &spec, None).unwrap();
    let mut csv = String::from("ts,x,vol96,lbl_h168_y,lbl_h168_fwd\n");
    for (t, b) in bars.iter().enumerate() {
        csv.push_str(&format!("{},{},0.01,{},{}\n", b.time, ((t * 7) % 11) as f64, lh::fmt(l.y[t]), lh::fmt(l.fwd[t])));
    }
    let path = s(&dir.join("short.rcsb"));
    store::write_store_with_time(&path, &data::parse_table(&csv).unwrap(), 512, Some(0)).unwrap();
    let embargo = lockbox::embargo_json(3600, "x", "test", 3600);
    lockbox::write_sidecar_with(&path, &Lockbox { start: s0, end: e0 }, "test", &lockbox::embargo_member(Some(&embargo))).unwrap();
    std::fs::write(dir.join("short.targets.json"), lh::spec_file(&spec)).unwrap();
    // The search: rows from one hour after the end read no locked bar, but their labels
    // share returns with the last locked labels until one purge after the end.
    let search = |from: i64| {
        let exclude = format!("ts < {from}");
        rcsr(&[
            "--data", &path, "--target", "lbl_h168_y", "--vars", "x", "--time", "ts", "--split", "temporal", "--cpu",
            "--no-dev-log", "--seed", "3", "--generations", "2", "--pop", "8", "--quiet", "--exclude", &exclude,
        ])
    };
    let (code, _, err) = search(e0 + H);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("lookback embargo 608400 s after the end (the label purge of the target spec)"), "{err}");
    let (code, _, err) = search(e0 + purge);
    assert_eq!(code, 0, "{err}");
    // Walk-forward refuses a window that starts inside the purge after the end.
    let spec_path = s(&dir.join("short.targets.json"));
    let wf = |train_start: i64| {
        let w = format!("w,{},{},{},{}", train_start, train_start + 1500 * H, train_start + 1500 * H + purge, train_start + 3000 * H + purge);
        rcsr(&[
            "walkforward", "--data", &path, "--target", "lbl_h168_y", "--target-spec", &spec_path, "--expr", "x", "--time", "ts",
            "--periods-per-year", "52", "--window", &w, "--json",
        ])
    };
    let (code, _, err) = wf(e0 + H);
    assert_eq!(code, 3, "{err}");
    assert!(err.contains("reaches the declared locked test"), "{err}");
    let (code, _, err) = wf(e0 + purge);
    assert_eq!(code, 0, "{err}");
}
