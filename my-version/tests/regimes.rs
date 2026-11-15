//! #1175: regime strata of walk-forward and trials reports, the causal regime
//! expression and the regime calendar writer. All fixtures are synthetic.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use rcsr::lockbox::{self, Lockbox};
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const H: i64 = 3600;
const D: i64 = 86_400;

fn run(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

fn json_of(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    Json::parse(stdout(o).trim()).expect("stdout is JSON")
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

fn items<'a>(j: &'a Json, path: &[&str]) -> &'a [Json] {
    match at(j, path) {
        Json::Arr(v) => v,
        _ => panic!("{path:?} is not an array"),
    }
}

fn series(j: &Json, path: &[&str]) -> Vec<f64> {
    items(j, path).iter().map(|v| v.as_f64().expect("number")).collect()
}

fn scratch(name: &str) -> (Scratch, PathBuf) {
    let s = Scratch::new(name);
    let p = s.path().to_path_buf();
    (s, p)
}

fn s(p: &Path) -> String {
    p.to_str().unwrap().to_string()
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Three symbols on an hourly grid: `x` is a score, `g` a regime column equal
/// across symbols that is positive in blocks of 7 hours and negative in the
/// next, `target` the forward return. `keep` filters rows by the sign of `g`.
fn panel_csv(dir: &Path, name: &str, hours: i64, keep: impl Fn(f64) -> bool) -> String {
    let mut text = String::from("ts,sym,x,g,target\n");
    for i in 0..hours {
        let m = 0.5 + frac(i as f64 * 0.618_033_988_749_894_9);
        let g = if (i / 7) % 2 == 0 { m } else { -m };
        if !keep(g) {
            continue;
        }
        for (k, sym) in ["A", "B", "C"].iter().enumerate() {
            let k = k as i64;
            let x = ((i * 37 + k * 11) % 17) as f64 - 8.0;
            let y = 0.0003 * x + (((i * 13 + k * 5) % 7) as f64 - 3.0) * 0.0004;
            text.push_str(&format!("{},{},{},{},{}\n", T0 + i * H, sym, x, g, y));
        }
    }
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    s(&p)
}

const HOURS: i64 = 600;

fn window() -> String {
    format!("w,{},{},{},{}", T0, T0 + 100 * H, T0 + 100 * H, T0 + HOURS * H)
}

/// The walk-forward call every test shares: a fixed cut, no cost unless given.
fn wf(data: &str, expr: &str, cost: &str, extra: &[&str]) -> Output {
    let w = window();
    let mut a = vec![
        "walkforward", "--data", data, "--target", "target", "--expr", expr, "--vars", "x", "--time", "ts", "--group-by", "sym",
        "--window", &w, "--threshold-fixed", "0", "--cost-bps", cost, "--periods-per-year", "8760", "--json",
    ];
    a.extend_from_slice(extra);
    run(&a)
}

fn stratum<'a>(strata: &'a [Json], label: &str) -> &'a Json {
    strata
        .iter()
        .find(|x| x.get("label").and_then(Json::as_str) == Some(label))
        .unwrap_or_else(|| panic!("no stratum {label}"))
}

#[test]
fn regime_expr_strata_match_filtered_runs() {
    let (_s, dir) = scratch("regimes-expr");
    let all = panel_csv(&dir, "all.csv", HOURS, |_| true);
    let on = panel_csv(&dir, "on.csv", HOURS, |g| g > 0.0);
    let off = panel_csv(&dir, "off.csv", HOURS, |g| g < 0.0);
    let j = json_of(&wf(&all, "x", "0", &["--regime-expr", "g", "--regime-name", "up", "--regime-expr", "0 - g", "--regime-name", "down"]));
    let exprs = items(&j, &["regimes", "expressions"]);
    assert_eq!(exprs.len(), 2);
    for (k, (name, filtered)) in [("up", &on), ("down", &off)].iter().enumerate() {
        assert_eq!(exprs[k].get("name").and_then(Json::as_str), Some(*name));
        let strata = items(&exprs[k], &["strata"]);
        let this = stratum(strata, "on");
        let f = json_of(&wf(filtered, "x", "0", &[]));
        let net: f64 = series(&f, &["pooled", "period_net"]).iter().sum();
        assert!((num(this, &["net_sum"]) - net).abs() < 1e-12, "{name}: {} vs {}", num(this, &["net_sum"]), net);
        assert!((num(this, &["exposure"]) - num(&f, &["pooled", "exposure"])).abs() < 1e-12, "{name}");
        assert_eq!(num(this, &["periods"]), num(&f, &["pooled", "periods"]), "{name}");
        // the off stratum of one expression is the on stratum of the other
        let other = stratum(items(&exprs[1 - k], &["strata"]), "on");
        assert_eq!(num(stratum(strata, "off"), &["periods"]), num(other, &["periods"]));
    }
}

#[test]
fn calendar_strata_sum_to_pooled() {
    let (_s, dir) = scratch("regimes-cal");
    let data = panel_csv(&dir, "all.csv", HOURS, |_| true);
    let iso = |h: i64| rcsr::protocol::iso((T0 + h * H) as f64) + "Z";
    let cal = dir.join("cal.csv");
    // two labels over three rows, a hole of 40 hours and a tail left undated
    std::fs::write(
        &cal,
        format!(
            "label,start,end\nbull,{},{}\nbear,{},{}\nbull,{},{}\n",
            iso(100),
            iso(250),
            iso(250),
            iso(400),
            iso(440),
            iso(560)
        ),
    )
    .unwrap();
    let j = json_of(&wf(&data, "x", "10", &["--regimes", &s(&cal), "--bootstrap", "200"]));
    let strata = items(&j, &["regimes", "calendar", "strata"]);
    let labels: Vec<&str> = strata.iter().map(|x| x.get("label").and_then(Json::as_str).unwrap()).collect();
    assert_eq!(labels, vec!["bull", "bear", "undated"]);
    let net = series(&j, &["pooled", "period_net"]);
    let cost: f64 = series(&j, &["pooled", "period_gross"]).iter().zip(&net).map(|(g, n)| g - n).sum();
    assert_eq!(strata.iter().map(|x| num(x, &["periods"])).sum::<f64>(), net.len() as f64);
    assert_eq!(num(&j, &["pooled", "periods"]), net.len() as f64);
    let net_sum: f64 = strata.iter().map(|x| num(x, &["net_sum"])).sum();
    let cost_sum: f64 = strata.iter().map(|x| num(x, &["cost_sum"])).sum();
    assert!((net_sum - net.iter().sum::<f64>()).abs() < 1e-12, "{net_sum}");
    assert!((cost_sum - cost).abs() < 1e-12, "{cost_sum} vs {cost}");
    assert!(cost > 0.0);
    assert_eq!(num(stratum(strata, "bull"), &["periods"]), 150.0 + 120.0);
    assert_eq!(num(stratum(strata, "undated"), &["periods"]), 40.0 + 40.0);
    assert!(items(stratum(strata, "bull"), &["mean_ci"]).len() == 2);
}

#[test]
fn always_long_has_zero_alpha_in_each_stratum() {
    let (_s, dir) = scratch("regimes-alpha");
    let data = panel_csv(&dir, "all.csv", HOURS, |_| true);
    let iso = |h: i64| rcsr::protocol::iso((T0 + h * H) as f64) + "Z";
    let cal = dir.join("cal.csv");
    std::fs::write(&cal, format!("label,start,end\nbull,{},{}\nbear,{},{}\nsideways,{},{}\n", iso(100), iso(300), iso(300), iso(450), iso(450), iso(500))).unwrap();
    let j = json_of(&wf(&data, "1", "0", &["--regimes", &s(&cal), "--regime-expr", "g"]));
    let mut all: Vec<&Json> = items(&j, &["regimes", "calendar", "strata"]).iter().collect();
    all.extend(items(&j, &["regimes", "expressions"])[0].get("strata").map(|s| match s {
        Json::Arr(v) => v.iter().collect::<Vec<_>>(),
        _ => panic!(),
    }).unwrap());
    assert!(all.len() >= 5);
    for st in all {
        if num(st, &["periods"]) >= 2.0 {
            assert!(num(st, &["alpha_bh_ann"]).abs() < 1e-12, "{}: {}", at(st, &["label"]).as_str().unwrap(), num(st, &["alpha_bh_ann"]));
            assert!((num(st, &["exposure"]) - 1.0).abs() < 1e-12);
        }
    }
}

#[test]
fn switch_count_of_a_daily_flip() {
    let (_s, dir) = scratch("regimes-flip");
    let hours = 100 + 24 * 365 + 1;
    let mut text = String::from("ts,x,g,target\n");
    for i in 0..hours {
        let g = if ((i - 100).max(0) / 24) % 2 == 0 { 1.0 } else { -1.0 };
        text.push_str(&format!("{},{},{},{}\n", T0 + i * H, ((i * 37) % 17) as f64 - 8.0, g, 0.0003 * (((i * 13) % 7) as f64 - 3.0)));
    }
    let data = dir.join("flip.csv");
    std::fs::write(&data, text).unwrap();
    let w = format!("w,{},{},{},{}", T0, T0 + 100 * H, T0 + 100 * H, T0 + hours * H);
    let o = run(&[
        "walkforward", "--data", &s(&data), "--target", "target", "--expr", "x", "--vars", "x", "--time", "ts", "--window", &w,
        "--threshold-fixed", "0", "--cost-bps", "0", "--periods-per-year", "8760", "--json", "--regime-expr", "g",
    ]);
    let j = json_of(&o);
    let e = &items(&j, &["regimes", "expressions"])[0];
    assert_eq!(num(e, &["switches"]), 365.0);
    assert!((num(e, &["switches_per_year"]) - 365.0).abs() < 1e-9, "{}", num(e, &["switches_per_year"]));
    assert!((num(e, &["mean_state_periods", "on"]) - 24.0).abs() < 0.2);
}

#[test]
fn regime_expr_refuses_label_columns() {
    let (_s, dir) = scratch("regimes-d13");
    let data = panel_csv(&dir, "all.csv", HOURS, |_| true);
    for col in ["target", "lbl_h24a_y", "g + lbl_h24a_y"] {
        let o = wf(&data, "x", "0", &["--regime-expr", col]);
        assert_eq!(code(&o), 5, "{col}: {}", stderr(&o));
        assert!(stderr(&o).contains("D13") && stderr(&o).contains("--regime-expr"), "{}", stderr(&o));
    }
    let ok = wf(&data, "x", "0", &["--regime-expr", "g"]);
    assert_eq!(code(&ok), 0, "{}", stderr(&ok));
    for (flag_args, why) in [
        (vec!["--regime-expr", "mean(g)"], "aggregate"),
        (vec!["--regime-expr", "g", "--regime-band", "-1"], "--regime-band must be"),
        (vec!["--regime-expr", "g", "--regime-dwell", "0"], "--regime-dwell must be"),
        (vec!["--regime-band", "0.1"], "--regime-band needs --regime-expr"),
        (vec!["--regime-name", "a"], "--regime-name follows"),
    ] {
        let o = wf(&data, "x", "0", &flag_args);
        assert_eq!(code(&o), 2, "{flag_args:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(why), "{}", stderr(&o));
    }
    let missing = wf(&data, "x", "0", &["--regime-expr", "nope"]);
    assert_eq!(code(&missing), 3, "{}", stderr(&missing));
}

#[test]
fn overlapping_calendar_rows_exit_two() {
    let (_s, dir) = scratch("regimes-overlap");
    let data = panel_csv(&dir, "all.csv", HOURS, |_| true);
    let cal = dir.join("cal.csv");
    std::fs::write(&cal, "label,start,end\na,2020-09-14,2020-10-01\nb,2020-09-30,2020-10-10\n").unwrap();
    let o = wf(&data, "x", "0", &["--regimes", &s(&cal)]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("calendar rows overlap"), "{}", stderr(&o));
}

#[test]
fn the_regime_flags_do_not_change_the_pooled_report() {
    let (_s, dir) = scratch("regimes-additive");
    let data = panel_csv(&dir, "all.csv", HOURS, |_| true);
    let sha = |o: &Output| {
        let t = stdout(o);
        t.split("\"report_sha256\":\"").nth(1).unwrap().split('"').next().unwrap().to_string()
    };
    let plain = wf(&data, "x", "10", &[]);
    assert_eq!(code(&plain), 0, "{}", stderr(&plain));
    assert!(!stdout(&plain).contains("\"regimes\""));
    assert!(!stdout(&plain).contains("period_exposure"));
    let with = wf(&data, "x", "10", &["--regime-expr", "g"]);
    let (a, b) = (json_of(&plain), json_of(&with));
    for k in ["windows", "candidates"] {
        assert_eq!(format!("{:?}", at(&a, &[k])), format!("{:?}", at(&b, &[k])), "{k}");
    }
    assert_eq!(series(&a, &["pooled", "period_net"]), series(&b, &["pooled", "period_net"]));
    assert_ne!(sha(&plain), sha(&with), "the strata are part of the hashed report");
    let again = wf(&data, "x", "10", &["--regime-expr", "g"]);
    assert_eq!(sha(&with), sha(&again), "deterministic");
}

// ---------------------------------------------------------------------------
// rcsr trials
// ---------------------------------------------------------------------------

fn arr(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

fn trial_report(dir: &Path, name: &str, net: &[f64], extra: bool) -> String {
    let n = net.len();
    let time: Vec<f64> = (0..n).map(|i| (T0 + i as i64 * D) as f64).collect();
    let extras = if extra {
        let gross: Vec<f64> = net.iter().map(|v| v + 0.0001).collect();
        format!(
            ",\"period_gross\":[{}],\"period_exposure\":[{}],\"period_entries\":[{}],\"period_bh\":[{}]",
            arr(&gross),
            arr(&vec![1.0; n]),
            arr(&vec![0.0; n]),
            arr(net)
        )
    } else {
        String::new()
    };
    let text = format!(
        "{{\"mode\":\"walkforward\",\"window_definition\":[{{\"label\":\"w1\",\"test_start\":1}}],\
\"purge\":{{\"horizon_seconds\":3600,\"embargo_seconds\":3600}},\
\"cost_model\":{{\"per_side_bps\":10,\"round_trip_bps\":20}},\
\"cost\":{{\"per_side_bps\":10,\"source\":\"flag\",\"return_units\":\"log\"}},\
\"policy\":{{\"name\":\"threshold\",\"band_width\":0,\"min_hold\":1}},\
\"threshold\":{{\"quantile\":0.5,\"fixed\":null}},\
\"annualization\":{{\"periods_per_year\":365,\"method\":\"explicit\"}},\
\"pooled\":{{\"periods\":{n},\"entries\":10,\"active_rows\":100,\"period_net\":[{}],\"period_time\":[{}]{extras}}}}}",
        arr(net),
        arr(&time)
    );
    let p = dir.join(format!("{name}.json"));
    std::fs::write(&p, text).unwrap();
    s(&p)
}

fn noise(n: usize, seed: u64) -> Vec<f64> {
    (0..n).map(|i| 0.01 * (frac((i as f64 + 1.0) * 0.754_877_666_246_692_7 + seed as f64 * 0.318_309_886) - 0.5)).collect()
}

fn iso_day(d: i64) -> String {
    rcsr::protocol::iso((T0 + d * D) as f64) + "Z"
}

#[test]
fn trials_regimes_adds_strata_and_nothing_else() {
    let (_s, dir) = scratch("regimes-trials");
    let n = 400;
    let a = trial_report(&dir, "a", &noise(n, 1), true);
    let b = trial_report(&dir, "b", &noise(n, 2), false);
    let cal = dir.join("labels.csv");
    // any label set: three years-like blocks and a hole
    std::fs::write(&cal, format!("label,start,end\nfirst,{},{}\nsecond,{},{}\n", iso_day(0), iso_day(150), iso_day(150), iso_day(350))).unwrap();
    let base = ["trials", "--wf", &a, &b, "--draws", "50", "--block-len", "8"];
    let plain = run(&base);
    let mut with = base.to_vec();
    let cal_path = s(&cal);
    with.extend(["--regimes", &cal_path]);
    let with = run(&with);
    let (jp, jw) = (json_of(&plain), json_of(&with));
    assert!(!stdout(&plain).contains("regimes"));
    // everything of the plain report is unchanged
    for k in ["series", "distinct_books", "clusters", "best", "spa", "pbo", "periods", "block_len"] {
        assert_eq!(format!("{:?}", at(&jp, &[k])), format!("{:?}", at(&jw, &[k])), "{k}");
    }
    assert_eq!(items(&jw, &["regimes_calendar", "labels"]).len(), 2);
    let members = items(&jw, &["members"]);
    for m in members {
        let strata = items(m, &["regimes"]);
        let labels: Vec<&str> = strata.iter().map(|x| x.get("label").and_then(Json::as_str).unwrap()).collect();
        assert_eq!(labels, vec!["first", "second", "undated"]);
        assert_eq!(strata.iter().map(|x| num(x, &["periods"])).sum::<f64>(), n as f64);
        assert_eq!(num(&strata[0], &["periods"]), 150.0);
        assert_eq!(num(&strata[2], &["periods"]), 50.0);
        assert!(items(&strata[0], &["mean_ci"]).len() == 2);
    }
    // the report with the extra series carries alpha; one without leaves it null
    assert!(at(&items(&members[0], &["regimes"])[0], &["alpha_bh_ann"]).as_f64().is_some());
    assert_eq!(at(&items(&members[1], &["regimes"])[0], &["alpha_bh_ann"]), &Json::Null);
    // a stratum of one series equals the same stratum computed by hand
    let net = noise(n, 1);
    let mean: f64 = net[..150].iter().sum::<f64>() / 150.0;
    assert!((num(&items(&members[0], &["regimes"])[0], &["net_mean"]) - mean).abs() < 1e-15);
    let again = run(&["trials", "--wf", &a, &b, "--draws", "50", "--block-len", "8", "--regimes", &s(&cal)]);
    assert_eq!(stdout(&with), stdout(&again), "deterministic");
    let compare = run(&["trials", "--compare", &a, &b, "--regimes", &s(&cal)]);
    assert_eq!(code(&compare), 2, "{}", stderr(&compare));
}

// ---------------------------------------------------------------------------
// rcsr protocol regimes
// ---------------------------------------------------------------------------

/// Daily closes of one symbol: trough on day 20, peak on day 50, trough on day 80.
fn planted(day: usize) -> f64 {
    if day <= 20 {
        100.0 - 2.0 * day as f64
    } else if day <= 50 {
        60.0 + 2.0 * (day - 20) as f64
    } else if day <= 80 {
        120.0 - 2.0 * (day - 50) as f64
    } else {
        60.0 + 1.5 * (day - 80) as f64
    }
}

/// Hourly rows, day by day, of `days` days from 2019-01-01 (a UTC midnight).
fn hourly_csv(dir: &Path, name: &str, start: i64, days: usize, price: impl Fn(usize, usize) -> f64) -> String {
    let mut text = String::from("ts,sym,close\n");
    for d in 0..days {
        for h in 0..24 {
            text.push_str(&format!("{},BTC,{}\n", start + d as i64 * D + h as i64 * H, price(d, h)));
        }
    }
    let p = dir.join(name);
    std::fs::write(&p, text).unwrap();
    s(&p)
}

const START: i64 = 1_546_300_800;

fn regimes(data: &str, out: &str, extra: &[&str]) -> Output {
    let mut a = vec![
        "protocol", "regimes", "--data", data, "--time-col", "ts", "--price-col", "close", "--symbol-col", "sym", "--symbol", "BTC", "--out",
        out,
    ];
    a.extend_from_slice(extra);
    run(&a)
}

fn rows(path: &str) -> Vec<(String, String, String)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .filter(|l| !l.starts_with('#') && *l != "label,start,end")
        .map(|l| {
            let f: Vec<&str> = l.split(',').collect();
            (f[0].to_string(), f[1].to_string(), f[2].to_string())
        })
        .collect()
}

fn day_iso(start: i64, d: i64) -> String {
    rcsr::protocol::iso((start + d * D) as f64) + "Z"
}

#[test]
fn regime_dating_finds_planted_phases() {
    let (_s, dir) = scratch("regimes-plant");
    let data = hourly_csv(&dir, "d.csv", START, 101, |d, _| planted(d));
    let out = s(&dir.join("cal.csv"));
    let o = regimes(&data, &out, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "0"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let want = vec![
        ("bear".to_string(), day_iso(START, 0), day_iso(START, 20)),
        ("bull".to_string(), day_iso(START, 20), day_iso(START, 50)),
        ("bear".to_string(), day_iso(START, 50), day_iso(START, 80)),
        ("bull".to_string(), day_iso(START, 80), day_iso(START, 101)),
    ];
    assert_eq!(rows(&out), want);
    let text = std::fs::read_to_string(&out).unwrap();
    assert!(text.starts_with("# mode=ex_post"), "{text}");
    assert!(stdout(&o).contains("calendar_sha256 "));
    // deterministic
    let out2 = s(&dir.join("cal2.csv"));
    assert_eq!(code(&regimes(&data, &out2, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "0"])), 0);
    assert_eq!(std::fs::read(&out).unwrap(), std::fs::read(&out2).unwrap());
    // the equal-weight index of one symbol is that symbol
    let idx = s(&dir.join("idx.csv"));
    let o = run(&[
        "protocol", "regimes", "--data", &data, "--time-col", "ts", "--price-col", "close", "--symbol-col", "sym", "--index", "equal-weight",
        "--out", &idx, "--locked-test", "none", "--min-days", "5", "--sideways-days", "0",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(rows(&idx), want);
    // causal mode: the label of a day is independent of later closes
    let causal = s(&dir.join("causal.csv"));
    assert_eq!(code(&regimes(&data, &causal, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "0", "--mode", "causal"])), 0);
    assert!(std::fs::read_to_string(&causal).unwrap().starts_with("# mode=causal"));
    let cut = hourly_csv(&dir, "cut.csv", START, 60, |d, _| planted(d));
    let causal_cut = s(&dir.join("causal-cut.csv"));
    assert_eq!(code(&regimes(&cut, &causal_cut, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "0", "--mode", "causal"])), 0);
    let (full, part) = (rows(&causal), rows(&causal_cut));
    for r in &part[..part.len() - 1] {
        assert!(full.contains(r), "{r:?} is not in the full-series causal calendar");
    }
    // the ex-post calendar of the cut series differs: its first labels use later closes
    let post_cut = s(&dir.join("post-cut.csv"));
    assert_eq!(code(&regimes(&cut, &post_cut, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "0"])), 0);
    assert_eq!(rows(&post_cut).len(), 2, "the cut series ends before the third phase is confirmed");
}

#[test]
fn regime_dating_labels_a_sideways_span() {
    let (_s, dir) = scratch("regimes-side");
    let data = hourly_csv(&dir, "d.csv", START, 220, |d, _| {
        if d <= 30 {
            100.0 + 5.0 * d as f64
        } else if d <= 130 {
            250.0 + if d % 2 == 0 { 3.0 } else { -3.0 }
        } else {
            253.0 + 5.0 * (d - 130) as f64
        }
    });
    let out = s(&dir.join("cal.csv"));
    let o = regimes(&data, &out, &["--locked-test", "none", "--min-days", "5", "--sideways-days", "60", "--sideways-band", "0.25"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let r = rows(&out);
    assert!(r.iter().any(|x| x.0 == "sideways"), "{r:?}");
    let cal = cal_rows(&out);
    for w in cal.windows(2) {
        assert!(w[0].end <= w[1].start);
    }
    let side = cal.iter().find(|x| x.label == "sideways").unwrap();
    assert!((side.end - side.start) / 86_400.0 >= 60.0);
}

#[test]
fn regime_dating_refuses_locked_rows() {
    let (_s, dir) = scratch("regimes-lock");
    let days = 700usize;
    let lock_start = START + 400 * D;
    let lock_end = START + 430 * D;
    let spiky = |poison: f64| move |d: usize, _h: usize| if (400..430).contains(&d) { poison } else { planted(d % 100) + 50.0 };
    let a = hourly_csv(&dir, "a.csv", START, days, spiky(1.0));
    let b = hourly_csv(&dir, "b.csv", START, days, spiky(1.0e9));
    let range = format!("{},{}", lock_start, lock_end);
    let (oa, ob) = (s(&dir.join("a-cal.csv")), s(&dir.join("b-cal.csv")));
    let extra = ["--locked-test", range.as_str(), "--min-days", "5", "--sideways-days", "0"];
    let (ra, rb) = (regimes(&a, &oa, &extra), regimes(&b, &ob, &extra));
    assert_eq!(code(&ra), 0, "{}", stderr(&ra));
    assert_eq!(code(&rb), 0, "{}", stderr(&rb));
    // no row inside the zone changes the calendar
    assert_eq!(std::fs::read(&oa).unwrap(), std::fs::read(&ob).unwrap());
    assert!(stdout(&ra).contains("dropped 3840 rows"), "{}", stdout(&ra));
    // the span is written as not_dated: the purge before the range, the embargo after it
    let purge = 219_600i64;
    let embargo = 10_972_800i64;
    let zs = ((lock_start - purge) as f64 / D as f64).floor() as i64 * D;
    let ze = (((lock_end + embargo) as f64) / D as f64).ceil() as i64 * D;
    let r = rows(&oa);
    let zone = r.iter().find(|x| x.0 == "not_dated").expect("not_dated row");
    assert_eq!(zone.1, rcsr::protocol::iso(zs as f64) + "Z");
    assert_eq!(zone.2, rcsr::protocol::iso(ze as f64) + "Z");
    let cal = cal_rows(&oa);
    for row in cal.iter().filter(|x| x.label != "not_dated") {
        assert!(row.end <= zs as f64 || row.start >= ze as f64, "{row:?} reaches the zone");
    }
    assert!(cal.iter().any(|x| x.label != "not_dated" && x.end <= zs as f64), "development rows before the zone are dated");
}

/// A sealed, read and final store whose declared lock is 4 days of daily closes.
fn read_store(dir: &Path) -> (String, String, String, i64, i64) {
    let start = T0 - T0 % D;
    let lock = (start + 12 * D, start + 17 * D);
    let mut text = String::from("ts,sym,x,close,y\n");
    for i in 0..(17 * 24) as i64 {
        let day = (i / 24) as usize;
        let close = if day >= 12 { [100.0, 150.0, 100.0, 150.0, 100.0][day - 12] } else { 100.0 + day as f64 };
        let x = ((i * 37) % 17) as f64 - 8.0;
        let y = 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004;
        text.push_str(&format!("{},BTC,{},{},{}\n", start + i * H, x, close, y));
    }
    let data = dir.join("store.csv");
    std::fs::write(&data, text).unwrap();
    let data = s(&data);
    lockbox::write_sidecar(&data, &Lockbox { start: lock.0, end: lock.1 }, "test").unwrap();
    let (l, sealed) = (s(&dir.join("ledger.jsonl")), s(&dir.join("sealed.json")));
    let wfa = |extra: &[&str]| {
        let mut a = vec![
            "walkforward", "--data", data.as_str(), "--target", "y", "--time", "ts", "--horizon", "5", "--cost-bps", "5", "--periods-per-year", "365",
            "--json", "--ledger", l.as_str(), "--expr", "x",
        ];
        a.extend_from_slice(extra);
        run(&a)
    };
    let dev = format!("dev,{},{},{},{}", start, start + 100 * H, start + 110 * H, start + 200 * H);
    let o = wfa(&["--window", &dev]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = run(&["protocol", "seal", "--final", "--ledger", &l, "--out", &sealed]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    (data, l, sealed, lock.0, lock.1)
}

#[test]
fn regime_dating_after_read_needs_the_final_reads() {
    let (_s, dir) = scratch("regimes-after");
    let (data, l, sealed, lock_start, lock_end) = read_store(&dir);
    let out = s(&dir.join("lock-cal.csv"));
    let args = |expect: &'static str| {
        vec![
            "protocol", "regimes", "--data", data.as_str(), "--time-col", "ts", "--price-col", "close", "--symbol-col", "sym", "--symbol", "BTC",
            "--out", out.as_str(), "--up", "0.3", "--down", "0.3", "--min-days", "0", "--sideways-days", "0", "--after-read", "--ledger", l.as_str(),
            "--sealed", sealed.as_str(), "--expect-reads", expect,
        ]
    };
    // no read yet: refused, nothing written, the ledger is unchanged
    let before = std::fs::read(&l).unwrap();
    let o = run(&args("1"));
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(stderr(&o).contains("every read the final seal allows"), "{}", stderr(&o));
    assert!(!Path::new(&out).exists());
    assert_eq!(std::fs::read(&l).unwrap(), before);
    // without --after-read the locked range stays undated
    let plain = s(&dir.join("plain.csv"));
    let o = regimes(&data, &plain, &["--min-days", "0", "--sideways-days", "0"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(rows(&plain).iter().all(|r| r.0 == "not_dated" || r.2 <= rcsr::protocol::iso((lock_start) as f64) + "Z"));
    // the one read
    let read = format!("locked,{},{},{},{}", lock_start - 100 * H, lock_start - 10 * H, lock_start, lock_end);
    let o = run(&[
        "walkforward", "--data", &data, "--target", "y", "--time", "ts", "--horizon", "5", "--cost-bps", "5", "--periods-per-year", "365", "--json",
        "--ledger", &l, "--expr", "x", "--locked-test", "--sealed", &sealed, "--window", &read,
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = run(&args("2"));
    assert_eq!(code(&o), 1, "{}", stderr(&o));
    assert!(!Path::new(&out).exists());
    let o = run(&args("1"));
    assert_eq!(code(&o), 0, "{}{}", stdout(&o), stderr(&o));
    let r = rows(&out);
    assert!(r.iter().any(|x| x.0 == "bull" || x.0 == "bear"), "{r:?}");
    let cal = cal_rows(&out);
    for row in &cal {
        assert!(row.start >= lock_start as f64 && row.end <= lock_end as f64, "{row:?} is outside the sealed range");
    }
    let sha = stdout(&o).lines().find_map(|l| l.strip_prefix("calendar_sha256 ")).unwrap().to_string();
    let ledger = std::fs::read_to_string(&l).unwrap();
    let last = ledger.lines().last().unwrap();
    assert!(last.contains("\"event\":\"regime_dating\"") && last.contains(&sha), "{last}");
    let v = run(&["protocol", "verify", "--ledger", &l, "--data", &data, "--expect-reads", "1", "--sealed", &sealed]);
    assert_eq!(code(&v), 0, "{}{}", stdout(&v), stderr(&v));
}

#[derive(Debug)]
struct CalRow {
    label: String,
    start: f64,
    end: f64,
}

fn cal_rows(path: &str) -> Vec<CalRow> {
    rows(path)
        .into_iter()
        .map(|(label, a, b)| CalRow { label, start: rcsr::data::parse_time(&a).unwrap(), end: rcsr::data::parse_time(&b).unwrap() })
        .collect()
}
