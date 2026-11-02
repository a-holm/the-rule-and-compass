//! #493: the cross-sectional rank target (`rcsr panel rank`, target kind
//! `xs_rank`) and the top-tercile long-only book (`--position top_tercile`,
//! `rcsr walkforward --policy top-tercile`).

use rcsr::devlog::Json;
use rcsr::registry;
use rcsr::store::{self, FileStore};
use rcsr::target;
use rcsr::trading::{self, PositionMap, TradingRows, TradingSpec};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;
const STEP: i64 = 3600;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-xs-rank-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn p(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json(o: &Output) -> Json {
    let text = String::from_utf8_lossy(&o.stdout);
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}\n{}", stderr(o)))
}

fn read_json(path: &str) -> Json {
    Json::parse(std::fs::read_to_string(path).unwrap().trim()).unwrap()
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

fn close(a: f64, b: f64) {
    assert!((a - b).abs() <= 1e-12, "{a} != {b}");
}

fn column(path: &str, name: &str) -> Vec<f64> {
    let s = FileStore::open(path).unwrap();
    let c = s.header().column_index(name).unwrap();
    let mut out = Vec::new();
    for g in 0..s.n_groups() {
        out.extend(s.read_f64_group(g, c).unwrap());
    }
    out
}

// ---------------------------------------------------------------------------
// rcsr panel rank
// ---------------------------------------------------------------------------

const FWD_SPEC: &str = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"lbl_h1_y\",\"name\":\"h1\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"lbl_h1_fwd\",\"return_units\":\"log\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open[t+1]\",\
\"exit_price\":\"close[t+h]\",\"offset\":0.0,\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\
\"units\":\"log_return\",\"cost_bps\":20.0,\"cost_in_label\":false,\"construction_columns\":[\"lbl_h1_fwd\"],\
\"note\":null,\"long_only\":true}]}\n";

/// 5 timestamps x 4 symbols, one tie at t2, one NaN at t3 and a thin timestamp t4.
fn panel_fixture(dir: &Path) -> (String, Vec<f64>) {
    let nan = f64::NAN;
    let fwd: [[f64; 4]; 5] = [
        [0.03, -0.01, 0.02, 0.00],
        [0.01, 0.01, 0.05, -0.02],
        [nan, 0.02, -0.03, 0.04],
        [nan, nan, 0.01, 0.02],
        [-0.01, 0.02, 0.03, 0.04],
    ];
    let mut text = String::from("ts,symbol,lbl_h1_fwd\n");
    for (t, row) in fwd.iter().enumerate() {
        for (k, v) in row.iter().enumerate() {
            text.push_str(&format!("{},{},{}\n", T0 + STEP * t as i64, ["AAA", "BBB", "CCC", "DDD"][k], v));
        }
    }
    let table = rcsr::data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = p(dir, "panel.rcsb");
    store::write_store_with_time(&path, &table, 7, Some(0)).unwrap();
    std::fs::write(p(dir, "panel.targets.json"), FWD_SPEC).unwrap();
    let t = 1.0 / 3.0;
    let want = vec![
        1.0, -1.0, t, -t, // t0
        0.0, 0.0, 1.0, -1.0, // t1: A and B tie
        nan, 0.0, -1.0, 1.0, // t2: one NaN, three finite
        nan, nan, nan, nan, // t3: two finite < --min-members 3
        -1.0, -t, t, 1.0, // t4
    ];
    (path, want)
}

#[test]
fn panel_rank_reproduces_the_hand_table_and_keeps_existing_blocks() {
    let dir = scratch();
    let (panel, want) = panel_fixture(&dir);
    let before = registry::write_schema(&panel, "2").map(|_| registry::read_hashes(&registry::schema_path(&panel)).unwrap()).unwrap();
    let panel_sha = rcsr::manifest::sha256_file(&panel).unwrap();
    let out = p(&dir, "ranked.rcsb");
    let o = rcsr(&[
        "panel", "rank", "--data", &panel, "--column", "lbl_h1_fwd", "--name", "lbl_h1_xs", "--out", &out,
        "--min-members", "3",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let got = column(&out, "lbl_h1_xs");
    assert_eq!(got.len(), want.len());
    for (i, (g, w)) in got.iter().zip(&want).enumerate() {
        if w.is_nan() {
            assert!(g.is_nan(), "row {i}: {g}");
        } else {
            assert!((g - w).abs() <= 1e-12, "row {i}: {g} != {w}");
        }
    }

    // The input is untouched and every existing column keeps its SHA-256.
    assert_eq!(rcsr::manifest::sha256_file(&panel).unwrap(), panel_sha);
    let after = registry::read_hashes(&registry::schema_path(&out)).unwrap();
    for (name, h) in &before {
        assert_eq!(after.get(name), Some(h), "{name} changed");
    }
    assert!(after.contains_key("lbl_h1_xs"));
    assert!(registry::verify_schema(&out).unwrap().is_empty());

    // The target spec entry joins <out stem>.targets.json beside the carried entries.
    let specs = target::parse_file(&std::fs::read_to_string(p(&dir, "ranked.targets.json")).unwrap()).unwrap();
    assert_eq!(specs.len(), 2);
    let xs = specs.iter().find(|s| s.column == "lbl_h1_xs").unwrap();
    assert_eq!(xs.kind, "xs_rank");
    assert_eq!(xs.return_column.as_deref(), Some("lbl_h1_fwd"));
    assert_eq!(xs.horizon_seconds, 3600.0);
    assert_eq!(xs.return_units, "log");
    assert_eq!(xs.cost_bps, Some(20.0));
    assert_eq!(xs.construction_columns, vec!["lbl_h1_fwd".to_string()]);
    assert_eq!(xs.required_purge_seconds(), 3600.0);
    assert!(specs.iter().any(|s| s.column == "lbl_h1_y" && s.kind == "forward_return"));

    // Registry family label, lead column.
    let r = registry::lookup("lbl_h1_xs").unwrap();
    assert_eq!(r.family, "label");
    assert_eq!(r.causal_offset, 1);
}

#[test]
fn panel_rank_refuses_bad_input() {
    let dir = scratch();
    let (panel, _) = panel_fixture(&dir);
    let out = p(&dir, "o.rcsb");
    let bad_name = rcsr(&["panel", "rank", "--data", &panel, "--column", "lbl_h1_fwd", "--name", "rank_h1", "--out", &out]);
    assert_eq!(code(&bad_name), 3);
    assert!(stderr(&bad_name).contains("lbl_<label>_xs"), "{}", stderr(&bad_name));
    let no_col = rcsr(&["panel", "rank", "--data", &panel, "--column", "nope", "--name", "lbl_h1_xs", "--out", &out]);
    assert_eq!(code(&no_col), 3);
    let no_spec_entry = rcsr(&["panel", "rank", "--data", &panel, "--column", "ts", "--name", "lbl_h1_xs", "--out", &out]);
    assert_eq!(code(&no_spec_entry), 3);
    assert!(stderr(&no_spec_entry).contains("no forward_return entry"), "{}", stderr(&no_spec_entry));
    let one = rcsr(&["panel", "rank", "--data", &panel, "--column", "lbl_h1_fwd", "--name", "lbl_h1_xs", "--out", &out, "--min-members", "1"]);
    assert_eq!(code(&one), 3);
    let same = rcsr(&["panel", "rank", "--data", &panel, "--column", "lbl_h1_fwd", "--name", "lbl_h1_xs", "--out", &panel]);
    assert_eq!(code(&same), 3);
}

#[test]
fn the_rank_column_is_a_label_never_an_input_and_never_projected() {
    // --vars (D13a): the label-family check and the registry refuse it.
    let e = registry::check_vars("/nonexistent/panel.rcsb", &["lbl_h1_xs".to_string()]).unwrap_err();
    assert!(e.contains("label-family"), "{e}");

    // --project: a lead column is not a view candidate.
    let mut cols: Vec<String> = rcsr::enrich::default_registry()
        .into_iter()
        .filter(|e| e.available && e.source != "input")
        .map(|e| e.name)
        .take(120)
        .collect();
    cols.push("lbl_h1_xs".to_string());
    for view in 0..4 {
        let v = registry::select_view(&cols, 24, 7, view).unwrap();
        assert!(!v.columns.iter().any(|c| c == "lbl_h1_xs"), "view {view}");
        assert!(v.omitted.iter().any(|(n, f, why)| n == "lbl_h1_xs" && f == "label" && why.contains("lead column")));
    }

    // The search: the target column or its return column in --vars is D13a (exit 5).
    let dir = scratch();
    let data = write_search_csv(&dir, "d.csv", None);
    let spec = write_xs_spec(&dir, "s.json");
    for vars in ["x,lbl_h1_xs", "x,r"] {
        let o = search_vars(&data, &spec, vars, &["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765"]);
        assert_eq!(code(&o), 5, "{vars}: {}", stderr(&o));
        assert!(stderr(&o).contains("D13a"), "{vars}: {}", stderr(&o));
    }
}

// ---------------------------------------------------------------------------
// the top-tercile book
// ---------------------------------------------------------------------------

/// 3 timestamps x 6 symbols. Top two scores per timestamp: t0 sym0 + sym3,
/// t1 sym5 + sym1, t2 sym2 + sym3.
const SCORES: [[f64; 6]; 3] = [
    [0.9, 0.1, 0.5, 0.7, 0.3, 0.2],
    [0.2, 0.8, 0.6, 0.4, 0.1, 0.9],
    [0.4, 0.3, 0.9, 0.8, 0.1, 0.2],
];
const RETURNS: [[f64; 6]; 3] = [
    [0.04, -0.01, 0.00, 0.03, -0.02, -0.02],
    [-0.01, 0.05, -0.01, 0.00, 0.01, 0.06],
    [0.00, -0.02, 0.05, 0.04, 0.01, -0.03],
];

// Hand arithmetic at 10 bp per side (c = 0.001), long rows
//   sym0: t0   sym1: t1   sym2: t2   sym3: t0 and t2   sym5: t1
// gross = (0.04 + 0.03) + (0.05 + 0.06) + (0.05 + 0.04) = 0.27
// |delta p| with the two final exits (sym2, sym3) = 2 + 2 + 2 + 4 + 2 = 12
// cost = 12 * 0.001 = 0.012, net = 0.258
// period nets: t0 (0.07 - 2c)/6, t1 (0.11 - 4c)/6, t2 (0.09 - 6c)/6
const GROSS: f64 = 0.27;
const COST: f64 = 0.012;
const NET: f64 = 0.258;

fn flat(a: &[[f64; 6]; 3]) -> Vec<f64> {
    // time-major rows: symbol varies fastest
    a.iter().flatten().copied().collect()
}

#[test]
fn top_tercile_book_matches_hand_fixture() {
    // Library: positions, P&L, trades, exposure and turnover.
    let pred = flat(&SCORES);
    let ret = flat(&RETURNS);
    let time: Vec<f64> = (0..18).map(|i| (T0 + STEP * (i / 6) as i64) as f64).collect();
    let group: Vec<u64> = (0..18).map(|i| (i % 6) as u64).collect();
    let pos = trading::top_tercile_positions(&pred, &time);
    let long: Vec<usize> = (0..18).filter(|&i| pos[i] == 1.0).collect();
    assert_eq!(long, vec![0, 3, 7, 11, 14, 15]);
    assert!(pos.iter().all(|&v| v == 0.0 || v == 1.0));

    // The lib API orders rows (group, time) itself: feed time-major rows.
    let rows = TradingRows { time: time.clone(), group: group.clone(), funding: Vec::new() };
    let spec = TradingSpec {
        cost_bps: 10.0,
        position: PositionMap::TopTercile,
        periods_per_year: Some(365.0),
        ..Default::default()
    };
    let s = trading::evaluate(&pred, &ret, &rows, &spec).unwrap();
    close(s.gross_pnl, GROSS);
    close(s.total_cost, COST);
    close(s.net_pnl, NET);
    assert_eq!((s.trades, s.entries, s.exits), (10, 6, 2));
    close(s.exposure, 6.0 / 18.0);
    assert_eq!(s.periods, 3);
    close(s.turnover, 10.0 * 365.0 / 3.0);
    close(s.period.mean_return, NET / 18.0);

    // A NaN prediction is never long and leaves the cross-section to the finite rows.
    let mut with_nan = pred.clone();
    with_nan[0] = f64::NAN;
    let pos_nan = trading::top_tercile_positions(&with_nan, &time);
    assert_eq!(pos_nan[0], 0.0);
    assert_eq!(pos_nan[3], 1.0);
    assert_eq!(pos_nan[2], 0.0, "five finite predictions hold floor(5/3) = 1 row");
    // Fewer than 3 finite predictions at a timestamp hold nothing.
    let thin = [0.5, 0.1, f64::NAN];
    assert_eq!(trading::top_tercile_positions(&thin, &[1.0, 1.0, 1.0]), vec![0.0, 0.0, 0.0]);
    // Ties go to the earlier row.
    assert_eq!(trading::top_tercile_positions(&[1.0, 1.0, 1.0], &[1.0, 1.0, 1.0]), vec![1.0, 0.0, 0.0]);

    // The same fixture through `rcsr walkforward --policy top-tercile`.
    let dir = scratch();
    let data = write_hand_csv(&dir, "wf.csv");
    let wf_path = p(&dir, "wf.json");
    let w = format!("w,{},{},{},{}", T0, T0 + STEP * 9, T0 + STEP * 10, T0 + STEP * 13);
    let o = rcsr(&[
        "walkforward", "--data", &data, "--target", "r", "--expr", "x", "--vars", "x", "--time", "t",
        "--group-by", "sym", "--horizon", "3600", "--cost-bps", "10", "--window", &w,
        "--periods-per-year", "365", "--policy", "top-tercile", "--out", &wf_path, "--json",
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = read_json(&wf_path);
    assert_eq!(at(&j, &["policy", "name"]).as_str(), Some("top_tercile"));
    let win = match at(&j, &["windows"]) {
        Json::Arr(a) => &a[0],
        _ => panic!(),
    };
    assert_eq!(at(win, &["threshold_source"]).as_str(), Some("none"));
    let m = at(win, &["metrics"]);
    close(num(m, &["total_cost"]), COST);
    close(num(m, &["total_abs_delta"]), 10.0);
    close(num(m, &["turnover"]), 10.0 * 365.0 / 3.0);
    close(num(m, &["exposure"]), 6.0 / 18.0);
    assert_eq!(num(m, &["trades"]), 10.0);
    assert_eq!(num(m, &["entries"]), 6.0);
    close(num(m, &["mean_return"]) * num(m, &["periods"]), NET / 6.0);

    // The search objective block prices the same book on its validation and test slices.
    let spec_path = write_xs_spec(&dir, "s.json");
    let csv = write_search_csv(&dir, "search.csv", None);
    let o = search(&csv, &spec_path, &["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765", "--test-frac", "0.15"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = json(&o);
    let ob = at(&j, &["objective"]);
    assert_eq!(at(ob, &["position"]).as_str(), Some("top_tercile"));
    assert_eq!(at(ob, &["spec", "position"]).as_str(), Some("top_tercile"));
    for book in ["validation", "test"] {
        let b = at(ob, &[book]);
        assert_eq!(num(b, &["rows"]), 18.0, "{book}");
        close(num(b, &["net_pnl"]), NET);
        close(num(b, &["gross_pnl"]), GROSS);
        close(num(b, &["total_cost"]), COST);
        assert_eq!(num(b, &["trades"]), 10.0, "{book}");
        close(num(b, &["exposure"]), 6.0 / 18.0);
    }
}

fn write_hand_csv(dir: &Path, name: &str) -> String {
    // 10 training timestamps with arbitrary values, then the hand block.
    let mut text = String::from("t,sym,x,r\n");
    for t in 0..10 {
        for k in 0..6 {
            let x = ((t * 7 + k * 3) % 11) as f64 / 11.0;
            let r = ((t * 5 + k * 2) % 9) as f64 / 900.0 - 0.004;
            text.push_str(&format!("{},S{},{},{}\n", T0 + STEP * t as i64, k, x, r));
        }
    }
    for t in 0..3 {
        for k in 0..6 {
            text.push_str(&format!("{},S{},{},{}\n", T0 + STEP * (10 + t) as i64, k, SCORES[t][k], RETURNS[t][k]));
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// Columns `t,sym,x,r,lbl_h1_xs`; `lbl_h1_xs = x` so the search fits `x`.
/// Rows: 14 training timestamps, the hand block as validation, the hand block as test.
fn write_search_csv(dir: &Path, name: &str, rotate_test: Option<usize>) -> String {
    write_search_csv_with(dir, name, rotate_test, false)
}

/// `delist`: symbol 5 has no label and no return at the 2nd and 3rd timestamp of each hand block.
fn write_search_csv_with(dir: &Path, name: &str, rotate_test: Option<usize>, delist: bool) -> String {
    let mut rows: Vec<(i64, usize, f64, f64)> = Vec::new();
    for t in 0..14 {
        for k in 0..6 {
            let x = (((t * 7 + k * 3) % 11) as f64 + 1.0) / 12.0 + k as f64 * 1e-4;
            let r = ((t * 5 + k * 2) % 9) as f64 / 900.0 - 0.004;
            rows.push((T0 + STEP * t as i64, k, x, r));
        }
    }
    for block in 0..2 {
        for t in 0..3 {
            for k in 0..6 {
                rows.push((T0 + STEP * (14 + block * 3 + t) as i64, k, SCORES[t][k], RETURNS[t][k]));
            }
        }
    }
    if let Some(rot) = rotate_test {
        let n = rows.len();
        let mut tail: Vec<f64> = rows[n - 18..].iter().map(|r| r.3).collect();
        tail.rotate_left(rot);
        for (row, r) in rows[n - 18..].iter_mut().zip(tail) {
            row.3 = r;
        }
    }
    let mut text = String::from("t,sym,x,r,lbl_h1_xs\n");
    for (t, k, x, r) in rows {
        let step = ((t - T0) / STEP) as usize;
        let gone = delist && k == 5 && step >= 14 && matches!((step - 14) % 3, 1 | 2);
        if gone {
            text.push_str(&format!("{},S{},{},NaN,NaN\n", t, k, x));
        } else {
            text.push_str(&format!("{},S{},{},{},{}\n", t, k, x, r, x));
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_xs_spec(dir: &Path, name: &str) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"lbl_h1_xs\",\"name\":\"h1_xs\",\
\"kind\":\"xs_rank\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"simple\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open[t+1]\",\
\"exit_price\":\"close[t+h]\",\"offset\":0.0,\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\
\"units\":\"rank_in_[-1,1]\",\"cost_bps\":20.0,\"cost_in_label\":false,\"construction_columns\":[\"r\"],\
\"note\":null,\"long_only\":true}]}\n";
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn search(data: &str, spec: &str, extra: &[&str]) -> Output {
    search_vars(data, spec, "x", extra)
}

fn search_vars(data: &str, spec: &str, vars: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "lbl_h1_xs", "--vars", vars, "--time", "t", "--group-by", "sym", "--split",
        "temporal", "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "12",
        "--pop", "24", "--metric", "squared", "--json", "--quiet", "--test-frac", "0.15",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

#[test]
fn xs_rank_spec_parses_and_the_objective_runs_on_it() {
    let dir = scratch();
    let spec_path = write_xs_spec(&dir, "s.json");
    let specs = target::parse_file(&std::fs::read_to_string(&spec_path).unwrap()).unwrap();
    assert_eq!(specs[0].kind, target::KIND_XS_RANK);
    assert_eq!(specs[0].null_prediction(f64::NAN), 0.0);

    // An xs_rank spec without return_column, with a clip or a column scale is refused.
    let text = std::fs::read_to_string(&spec_path).unwrap();
    for (from, to, needle) in [
        ("\"return_column\":\"r\"", "\"return_column\":null", "needs a return_column"),
        ("\"clip\":null", "\"clip\":[-1.0,1.0]", "scale constant 1, no clip"),
        ("\"construction_columns\":[\"r\"]", "\"construction_columns\":[]", "construction_columns"),
    ] {
        let e = target::parse_file(&text.replace(from, to)).unwrap_err();
        assert!(e.contains(needle), "{e}");
    }

    let csv = write_search_csv(&dir, "d.csv", None);
    let o = search(&csv, &spec_path, &["--objective", "sharpe", "--position", "top_tercile", "--validation-frac", "0.1765"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = json(&o);
    assert_eq!(at(&j, &["objective", "name"]).as_str(), Some("sharpe"));
    assert_eq!(at(&j, &["target_spec", "kind"]).as_str(), Some("xs_rank"));
    assert_eq!(num(&j, &["objective", "cost", "per_side_bps"]), 10.0);
    assert_eq!(at(&j, &["objective", "cost", "source"]).as_str(), Some("target_spec"));

    // An xs_rank target needs the top-tercile map; the threshold map is refused.
    let o = search(&csv, &spec_path, &["--objective", "pnl", "--validation-frac", "0.1765"]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("--position top_tercile"), "{}", stderr(&o));

    // The other refusals name their flag.
    let refusals: [(&[&str], &str); 4] = [
        (&["--split", "temporal-group"], "--split temporal"),
        (&["--threshold", "0.1"], "no threshold"),
        (&["--fitness", "pnl"], "--fitness pnl"),
        (&["--max-position", "0.5"], "top_tercile book is the unit"),
    ];
    for (extra, needle) in refusals {
        let mut args = vec!["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765"];
        args.extend_from_slice(extra);
        let o = search(&csv, &spec_path, &args);
        assert_eq!(code(&o), 2, "{needle}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{needle}: {}", stderr(&o));
    }
}

#[test]
fn a_forward_return_spec_keeps_working_with_both_position_maps() {
    let dir = scratch();
    let csv = write_search_csv(&dir, "d.csv", None);
    let text = std::fs::read_to_string(write_xs_spec(&dir, "s.json")).unwrap();
    let fwd = text.replace("\"xs_rank\"", "\"forward_return\"").replace("\"construction_columns\":[\"r\"]", "\"construction_columns\":[]");
    let spec = p(&dir, "fwd.json");
    std::fs::write(&spec, fwd).unwrap();
    for extra in [vec![], vec!["--position", "top_tercile"], vec!["--position", "long_only"]] {
        let mut args = vec!["--objective", "pnl", "--validation-frac", "0.1765"];
        args.extend(extra.iter().copied());
        let o = search(&csv, &spec, &args);
        assert_eq!(code(&o), 0, "{extra:?}: {}", stderr(&o));
    }
}

#[test]
fn the_finalist_does_not_depend_on_the_test_block() {
    let dir = scratch();
    let spec = write_xs_spec(&dir, "s.json");
    let plain = write_search_csv(&dir, "a.csv", None);
    let shuffled = write_search_csv(&dir, "b.csv", Some(7));
    let args = ["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765"];
    let (a, b) = (search(&plain, &spec, &args), search(&shuffled, &spec, &args));
    assert_eq!(code(&a), 0, "{}", stderr(&a));
    assert_eq!(code(&b), 0, "{}", stderr(&b));
    let (ja, jb) = (json(&a), json(&b));
    let (oa, ob) = (at(&ja, &["objective"]), at(&jb, &["objective"]));
    assert_eq!(at(oa, &["finalist", "expression"]).as_str(), at(ob, &["finalist", "expression"]).as_str());
    assert_eq!(num(oa, &["finalist", "validation_objective"]), num(ob, &["finalist", "validation_objective"]));
    assert_eq!(num(oa, &["validation", "net_pnl"]), num(ob, &["validation", "net_pnl"]));
    assert_ne!(num(oa, &["test", "net_pnl"]), num(ob, &["test", "net_pnl"]));
}

#[test]
fn walkforward_from_run_prices_the_top_tercile_book_of_the_search() {
    let dir = scratch();
    let spec = write_xs_spec(&dir, "s.json");
    let csv = write_search_csv(&dir, "d.csv", None);
    let o = search(&csv, &spec, &["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765", "--arm", "A0"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let run = p(&dir, "run.json");
    std::fs::write(&run, &o.stdout).unwrap();
    let book = at(&json(&o), &["objective", "validation"]).clone();

    // The validation slice is the 3 timestamps after 14 training timestamps.
    let w = format!("v,{},{},{},{}", T0, T0 + STEP * 13, T0 + STEP * 14, T0 + STEP * 17);
    let out = p(&dir, "wf.json");
    let o = rcsr(&["walkforward", "--data", &csv, "--time", "t", "--group-by", "sym", "--from-run", &run, "--window", &w, "--out", &out, "--allow-in-sample-windows"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = read_json(&out);
    assert_eq!(at(&j, &["policy", "name"]).as_str(), Some("top_tercile"));
    let m = at(match at(&j, &["windows"]) { Json::Arr(a) => &a[0], _ => panic!() }, &["metrics"]);
    for key in ["total_cost", "exposure", "trades", "entries", "rows"] {
        close(num(m, &[key]), num(&book, &[key]));
    }
    close(num(m, &["mean_return"]) * num(m, &["periods"]) * 6.0, num(&book, &["net_pnl"]));

    // The quantile flags are refused with the top-tercile policy, and so is a threshold run.
    for flag in [["--threshold-quantile", "0.7"], ["--threshold-fixed", "0"]] {
        let o = rcsr(&["walkforward", "--data", &csv, "--target", "r", "--expr", "x", "--time", "t", "--horizon", "3600", "--window", &w, "--policy", "top-tercile", flag[0], flag[1]]);
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        assert!(stderr(&o).contains("no threshold"), "{}", stderr(&o));
    }
    for flag in ["--device-pnl", "--batch-pnl"] {
        let o = rcsr(&["walkforward", "--data", &csv, "--target", "r", "--expr", "x", "--time", "t", "--horizon", "3600", "--window", &w, "--policy", "top-tercile", flag]);
        assert_ne!(code(&o), 0);
        assert!(stderr(&o).contains("top-tercile"), "{flag}: {}", stderr(&o));
    }

    // The ablation prices the threshold rule only.
    let windows = p(&dir, "windows.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f1,{},{},{},{},{}\n",
            T0,
            T0 + STEP * 13,
            T0 + STEP * 10,
            T0 + STEP * 14,
            T0 + STEP * 17
        ),
    )
    .unwrap();
    let ab = rcsr(&["ablation", "score", "--from-run", &run, "--data", &csv, "--time", "t", "--windows", &windows, "--purge-seconds", "3600"]);
    assert_eq!(code(&ab), 2, "{}", stderr(&ab));
    assert!(stderr(&ab).contains("top_tercile"), "{}", stderr(&ab));
}

#[test]
fn search_and_walkforward_rank_the_same_cross_section() {
    // Symbol 5 is delisted: no label and no return at the 2nd and 3rd timestamp of the
    // validation block and of the test block. Its score still ranks in both paths, it
    // earns nothing and pays nothing where the return is missing.
    let dir = scratch();
    let spec = write_xs_spec(&dir, "s.json");
    let csv = write_search_csv_with(&dir, "d.csv", None, true);
    let o = search(
        &csv,
        &spec,
        &["--objective", "pnl", "--position", "top_tercile", "--validation-frac", "0.1765", "--fill", "drop"],
    );
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let run = p(&dir, "run.json");
    std::fs::write(&run, &o.stdout).unwrap();
    let j = json(&o);

    // Hand arithmetic at 10 bp per side. Longs: t0 sym0 + sym3; t1 sym5 (no return) + sym1;
    // t2 sym2 + sym3. gross = 0.07 + 0.05 + 0.09 = 0.21, |delta p| with exits = 2 + 2 + 2 + 4 = 10,
    // net = 0.21 - 0.010 = 0.200 on 16 usable rows, 5 of them long.
    // Period means: (0.07 - 0.002) / 6, (0.05 - 0.003) / 5, (0.09 - 0.005) / 5.
    let period_sum = 0.068 / 6.0 + 0.047 / 5.0 + 0.085 / 5.0;
    for (book, start) in [("validation", 14), ("test", 17)] {
        let b = at(&j, &["objective", book]);
        assert_eq!(num(b, &["rows"]), 16.0, "{book}");
        close(num(b, &["net_pnl"]), 0.2);
        close(num(b, &["total_cost"]), 0.01);
        close(num(b, &["exposure"]), 5.0 / 16.0);
        assert_eq!(num(b, &["trades"]), 8.0, "{book}");

        let w = format!("w,{},{},{},{}", T0, T0 + STEP * (start - 1), T0 + STEP * start, T0 + STEP * (start + 3));
        let out = p(&dir, &format!("wf-{book}.json"));
        let o = rcsr(&["walkforward", "--data", &csv, "--time", "t", "--group-by", "sym", "--from-run", &run, "--window", &w, "--out", &out, "--allow-in-sample-windows"]);
        assert_eq!(code(&o), 0, "{book}: {}", stderr(&o));
        let m = at(match at(&read_json(&out), &["windows"]) { Json::Arr(a) => &a[0], _ => panic!() }, &["metrics"]).clone();
        assert_eq!(num(&m, &["rows"]), 16.0, "{book}");
        assert_eq!(num(&m, &["active_rows"]), 5.0, "{book}");
        close(num(&m, &["total_cost"]), num(b, &["total_cost"]));
        close(num(&m, &["exposure"]), num(b, &["exposure"]));
        assert_eq!(num(&m, &["trades"]), num(b, &["trades"]), "{book}");
        assert_eq!(num(&m, &["entries"]), num(b, &["entries"]), "{book}");
        close(num(&m, &["mean_return"]) * num(&m, &["periods"]), period_sum);
    }
}
