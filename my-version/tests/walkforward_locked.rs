//! Walk-forward over a `.rcsb` store never depends on rows inside the declared
//! locked test. Two synthetic stores are identical outside the range and
//! poisoned inside it; every report of a window that ends before the range
//! must be byte-identical. Fixtures are synthetic; the range is a stand-in in
//! 2024.

use rcsr::lockbox::{self, Lockbox};
use rcsr::{data, store};
use std::path::{Path, PathBuf};
use std::process::Command;

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

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rcsr-wf-locked-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn rcsr(args: &[&str]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().unwrap();
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

/// Drops the lines that carry pure wall clock and the data path.
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
    let dir = scratch("diff");
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
    let dir = scratch("power");
    let clean = write_data(&dir, "clean", Fixture::Clean, false);
    let poisoned = write_data(&dir, "poisoned", Fixture::Poisoned, false);
    // 2024 data is outside the protocol default range, so this window is legal here
    let w = window(LOCK_END - 10 * H);
    let (_, extra) = variants(&dir).remove(0);
    let (ca, oa, ea) = wf(&clean, &w, &extra);
    let (cb, ob, eb) = wf(&poisoned, &w, &extra);
    assert_eq!(ca, 0, "{ea}");
    assert_eq!(cb, 2, "{eb}");
    assert!(eb.contains("overlapping returns"), "{eb}");
}

#[test]
fn a_window_that_reaches_the_range_is_refused_without_the_sealed_read() {
    let dir = scratch("refuse");
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
