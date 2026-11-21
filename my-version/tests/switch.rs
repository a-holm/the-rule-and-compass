//! #1185: `rcsr switch`, the gate-cut profile of `if(greater(g, c), A, B)` and
//! the k-switch bound on the training rows of a `--fitness pnl` run.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;
const STAMPS: usize = 480;
const SYMBOLS: [&str; 3] = ["a", "b", "c"];

fn scratch() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("switch");
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// The regime gate of stamp `k`: positive in alternate 48-stamp blocks, the
/// same for every symbol.
fn regime(k: usize) -> f64 {
    let m = 0.5 + 0.4 * frac(k as f64 * 0.324_717_957_244_746);
    if (k / 48) % 2 == 0 {
        m
    } else {
        -m
    }
}

/// Three symbols on a 900 s grid from `T0 + first * STEP`. `x1` pays while the
/// gate `g` is positive and `0 - x1` while it is negative; `h` varies inside a
/// timestamp; `lbl_h24a_y` is a label-family column no gate may read. `alter`
/// rewrites the gate and the return of a row from its stamp.
fn write_csv(dir: &Path, name: &str, first: i64, alter: impl Fn(i64) -> Option<(f64, f64)>) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y,lbl_h24a_y\n");
    for k in first..STAMPS as i64 {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x1 = frac(i * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let g = if k >= 0 { regime(k as usize) } else { 0.3 };
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1 + 0.0005 * noise;
            let (g, r) = alter(k).unwrap_or((g, r));
            text.push_str(&format!("{},{},{},{},{},{},{},{},{}\n", T0 + STEP * k, sym, x1, x2, g, h, r, r, r));
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("s.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn rcsr(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
}

/// `rcsr switch` with the flags of a `--fitness pnl` run plus `extra`.
fn switch(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "switch", "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--split", "temporal",
        "--target-spec", spec, "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--test-frac", "0.2",
        "--no-dev-log", "--quiet",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn json(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    let text = String::from_utf8_lossy(&o.stdout);
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
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

fn arr<'a>(j: &'a Json, path: &[&str]) -> &'a Vec<Json> {
    match at(j, path) {
        Json::Arr(a) => a,
        _ => panic!("{path:?} is not an array"),
    }
}

const GATES: [&str; 6] = ["--a", "x1", "--b", "0 - x1", "--gate", "g"];

#[test]
fn switch_reports_the_profile_and_the_bound() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let out_path = dir.join("switch.json").to_string_lossy().into_owned();
    let mut args: Vec<&str> = GATES.to_vec();
    args.extend(["--gate", "h", "--cuts", "8", "--switches", "0,2,8,100000", "--json", "--out", &out_path]);
    let first = switch(&data, &spec, &args);
    let j = json(&first);
    assert_eq!(at(&j, &["schema"]).as_str(), Some("rcsr-switch/1"));
    assert_eq!(at(&j, &["rows"]).as_str(), Some("train"));
    assert_eq!(num(&j, &["groups"]), 3.0);
    assert_eq!(num(&j, &["train_rows"]), 3.0 * num(&j, &["timestamps"]));
    let reference = num(&j, &["a", "net"]).max(num(&j, &["b", "net"]));
    assert_eq!(num(&j, &["bound", "reference"]), reference);
    assert_eq!(at(&j, &["bound", "is_evidence"]), &Json::Bool(false));
    assert_eq!(at(&j, &["bound", "exact"]), &Json::Bool(true));
    let ks = arr(&j, &["bound", "k"]);
    assert_eq!(ks.len(), 4);
    assert!((num(&ks[0], &["value"]) - reference).abs() <= 1e-12, "V_0 is the better branch");
    for w in ks.windows(2) {
        assert!(num(&w[1], &["value"]) >= num(&w[0], &["value"]));
    }
    let last = &ks[3];
    assert_eq!(num(last, &["requested"]), 100000.0);
    assert_eq!(num(last, &["k"]), num(&j, &["timestamps"]) - 1.0);
    assert_eq!(at(last, &["clamped"]), &Json::Bool(true));
    assert_eq!(num(last, &["value"]), num(&j, &["bound", "free"]));

    let gates = arr(&j, &["gates"]);
    assert_eq!(gates.len(), 2);
    let g = &gates[0];
    assert_eq!(at(g, &["time_level"]), &Json::Bool(true));
    let best = at(g, &["best"]);
    let net = num(best, &["net"]);
    assert!(net > reference, "the regime gate beats both branches: {net} vs {reference}");
    let k = num(best, &["switches"]);
    let bound_k = ks.iter().find(|b| num(b, &["k"]) >= k).expect("a budget at or above K");
    assert!(net <= num(bound_k, &["value"]) + 1e-12);
    let seam = at(best, &["seam"]);
    assert!((num(seam, &["sum"]) - net).abs() <= 1e-9, "seam identity at the best cut");
    assert!((num(seam, &["composite"]) - net).abs() <= 1e-12);
    let grid_best = num(g, &["grid", "best", "net"]);
    assert!(grid_best <= net);
    assert_eq!(arr(g, &["grid", "profile"]).len(), 7);
    let plateau = at(best, &["plateau"]);
    assert!(num(plateau, &["lo"]) <= num(best, &["index"]) && num(best, &["index"]) <= num(plateau, &["hi"]));
    assert!(at(best, &["expression"]).as_str().unwrap().starts_with("if(greater(g, "));
    let h = &gates[1];
    assert_eq!(at(h, &["time_level"]), &Json::Bool(false));
    assert_eq!(at(h, &["best", "switches"]), &Json::Null);

    let written = std::fs::read_to_string(&out_path).unwrap();
    assert_eq!(written.trim(), String::from_utf8_lossy(&first.stdout).trim(), "--out holds the printed report");
    let second = switch(&data, &spec, &args);
    assert_eq!(first.stdout, second.stdout, "the report is deterministic");
    let plain = switch(&data, &spec, &GATES);
    assert_eq!(code(&plain), 0, "{}", stderr(&plain));
    let text = String::from_utf8_lossy(&plain.stdout);
    assert!(text.contains("pruning bound, not evidence") && text.contains("expression: if(greater(g, "), "{text}");
}

#[test]
fn the_printed_expression_reproduces_the_book() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    for b in ["0 - x1", "flat"] {
        let j = json(&switch(&data, &spec, &["--a", "x1", "--b", b, "--gate", "g", "--json"]));
        let best = at(&j, &["gates"]);
        let best = match best {
            Json::Arr(a) => at(&a[0], &["best"]),
            _ => panic!(),
        };
        let expr = at(best, &["expression"]).as_str().unwrap().to_string();
        let first = num(&j, &["time_first"]) as i64;
        let last = num(&j, &["time_last"]) as i64;
        assert_eq!(first, T0);
        // The walk-forward data has 30 earlier stamps for its training window;
        // its test window is exactly the training rows of the switch.
        let wf_data = write_csv(&dir, "wf.csv", -30, |_| None);
        let w = format!("w,{},{},{},{}", T0 - 30 * STEP, T0 - STEP, T0, last + 1);
        let wf_out = dir.join("wf.json").to_string_lossy().into_owned();
        let wf = rcsr(&[
            "walkforward", "--data", &wf_data, "--target", "y", "--target-spec", &spec, "--expr", &expr, "--vars", "g,x1,x2",
            "--time", "t", "--group-by", "s", "--window", &w, "--threshold-fixed", "0", "--cost-bps", "5", "--out", &wf_out,
        ]);
        assert_eq!(code(&wf), 0, "{expr}: {}", stderr(&wf));
        let report = Json::parse(std::fs::read_to_string(&wf_out).unwrap().trim()).unwrap();
        let m = at(&arr(&report, &["windows"])[0], &["metrics"]);
        let turns = num(best, &["flips"]) + num(best, &["exits"]);
        assert_eq!(num(m, &["rows"]), num(&j, &["train_rows"]), "{b}: the window holds the training rows");
        assert_eq!(num(m, &["entries"]) * 2.0, turns, "{b}: entries");
        assert!((num(m, &["total_cost"]) - 0.0005 * turns).abs() <= 1e-12, "{b}: cost {} for {turns} turns", num(m, &["total_cost"]));
        // Three symbols at every stamp: the period mean is a third of the row sum.
        let wf_net = num(m, &["mean_return"]) * num(m, &["periods"]) * 3.0;
        assert!((wf_net - num(best, &["net"])).abs() <= 1e-10, "{b}: walkforward net {wf_net}, switch net {}", num(best, &["net"]));
    }
}

#[test]
fn cuts_read_only_the_training_rows() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let args = ["--a", "x1", "--b", "0 - x1", "--gate", "g", "--gate", "h", "--json"];
    let base = switch(&data, &spec, &args);
    let last = num(&json(&base), &["time_last"]) as i64;
    let k_last = (last - T0) / STEP;
    let scrambled = write_csv(&dir, "scrambled.csv", 0, |k| (k > k_last).then(|| (frac(k as f64 * 0.77) - 0.5, 0.02 * (frac(k as f64 * 0.41) - 0.5))));
    let other = switch(&scrambled, &spec, &args);
    assert_eq!(code(&other), 0, "{}", stderr(&other));
    assert_eq!(base.stdout, other.stdout, "gate values and returns after the training rows change nothing");
}

#[test]
fn switch_refusals_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let refused = |extra: &[&str], want: i32, needle: &str| {
        let o = switch(&data, &spec, extra);
        assert_eq!(code(&o), want, "{extra:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{extra:?}: want '{needle}' in {}", stderr(&o));
    };
    for col in ["y", "r", "lbl_h24a_y", "g + lbl_h24a_y"] {
        refused(&["--a", "x1", "--b", "flat", "--gate", col], 5, "--gate");
        refused(&["--a", "x1", "--b", "flat", "--gate", col], 5, "D13");
    }
    refused(&["--a", "x1 + y", "--b", "flat", "--gate", "g"], 5, "--a");
    refused(&["--a", "x1", "--b", "flat", "--gate", "nope"], 3, "nope");
    refused(&["--a", "x1", "--b", "flat", "--gate", "mean(g)"], 2, "aggregate");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g +"], 2, "--gate");
    refused(&["--a", "x1", "--b", "flat", "--gate", "1"], 2, "distinct");
    refused(&["--a", "x1", "--b", "x1", "--gate", "g"], 2, "same book");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--band", "0.1"], 2, "not plain");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--pnl-mask", "g"], 2, "--pnl-mask");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--gpu"], 2, "--gpu");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--fitness", "metric"], 2, "--fitness must be pnl");
    refused(&["--a", "x1", "--b", "flat"], 2, "--gate <expr> is required");
    refused(&["--b", "flat", "--gate", "g"], 2, "--a <expr|flat> is required");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--cuts", "1"], 2, "--cuts");
    refused(&["--a", "x1", "--b", "flat", "--gate", "g", "--switches", "2,x"], 2, "--switches");
}

#[test]
fn switch_help_lists_its_options() {
    let o = rcsr(&["switch", "--help"]);
    assert_eq!(code(&o), 0);
    let help = String::from_utf8_lossy(&o.stdout);
    let doc = rcsr::schema::json();
    let start = doc.find("\"switch\":{").expect("schema documents rcsr switch");
    let section = &doc[start..];
    let section = &section[..section.find("]}").unwrap()];
    let mut names = 0;
    for part in section.split("\"name\":\"--").skip(1) {
        let name = format!("--{}", part.split('"').next().unwrap());
        assert!(help.contains(&name), "rcsr switch --help is missing {name}");
        names += 1;
    }
    assert_eq!(names, 7);
    let top = rcsr(&["--help"]);
    assert!(String::from_utf8_lossy(&top.stdout).contains("rcsr switch"));
}

/// `x1` and the log return of row `(k, j)` of [`write_csv`]; the book prices `exp_m1` of it.
fn row(k: usize, j: usize) -> (f64, f64) {
    let i = (k + 40) as f64 * 3.0 + j as f64;
    let x1 = frac(i * 0.618_033_988_749_894_9) - 0.5;
    let noise = frac(i * 0.324_717_957_244_746) - 0.5;
    (x1, (if regime(k) > 0.0 { 0.01 } else { -0.01 }) * x1 + 0.0005 * noise)
}

/// #1232: under `--pnl-excess xs-mean` the profile is on the excess per row
/// `(r - m_t) - funding`. The report says so; the counts that depend on the
/// positions only are those of the absolute run at every grid cut, and the
/// gross of a cut is the absolute gross less the sum over its long rows of the
/// mean return of the three symbols at the row's stamp, by hand.
#[test]
fn switch_json_reports_excess_label() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let args = ["--a", "x1", "--b", "0 - x1", "--gate", "g", "--cuts", "8"];
    let none = json(&switch(&data, &spec, &[&args[..], &["--json"]].concat()));
    let xs = json(&switch(&data, &spec, &[&args[..], &["--json", "--pnl-excess", "xs-mean"]].concat()));
    assert!(none.get("excess").is_none());
    assert_eq!(at(&xs, &["excess"]).as_str(), Some("xs_mean"));
    let stamps = num(&xs, &["timestamps"]) as usize;
    assert_eq!(num(&xs, &["train_rows"]), 3.0 * stamps as f64);
    let mean = |k: usize| (0..SYMBOLS.len()).map(|j| row(k, j).1.exp_m1()).sum::<f64>() / SYMBOLS.len() as f64;
    let profile = |j: &Json| arr(&arr(j, &["gates"])[0], &["grid", "profile"]).clone();
    let (pn, px) = (profile(&none), profile(&xs));
    assert_eq!(pn.len(), 7);
    for (a, b) in pn.iter().zip(&px) {
        for key in ["index", "value", "rows_a", "flips", "exits", "long_rows", "entry_times", "gate_flips", "switches", "seam_pairs"] {
            assert_eq!(at(a, &[key]), at(b, &[key]), "{key}");
        }
        let c = num(a, &["value"]);
        let mut held = 0.0f64;
        for k in 0..stamps {
            for j in 0..SYMBOLS.len() {
                let (x1, _) = row(k, j);
                let v = if regime(k) > c { x1 } else { 0.0 - x1 };
                if v > 0.0 {
                    held += mean(k);
                }
            }
        }
        let want = num(a, &["gross"]) - held;
        assert!((num(b, &["gross"]) - want).abs() <= 1e-12, "cut {c}: xs gross {} vs gross - sum m {want}", num(b, &["gross"]));
        let cost = num(a, &["gross"]) - num(a, &["net"]);
        assert!((num(b, &["gross"]) - num(b, &["net"]) - cost).abs() <= 1e-12, "cut {c}: the cost is on positions only");
        assert!((num(b, &["net"]) - num(a, &["net"])).abs() > 1e-6, "cut {c}: the net is on the excess");
    }
    for side in ["a", "b"] {
        for key in ["flips", "exits", "long_rows"] {
            assert_eq!(at(&none, &[side, key]), at(&xs, &[side, key]), "{side}.{key}");
        }
    }
    let text = switch(&data, &spec, &[&args[..], &["--pnl-excess", "xs-mean"]].concat());
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    assert!(String::from_utf8_lossy(&text.stdout).contains("excess: xs_mean"), "{}", String::from_utf8_lossy(&text.stdout));
    let plain = switch(&data, &spec, &args);
    assert!(!String::from_utf8_lossy(&plain.stdout).contains("excess:"));
}

/// #1232 (with #1178): `--pnl-stat` and `--pnl-cvar-q` change the block fitness
/// of the cuts and nothing else, and the report names the statistic.
#[test]
fn switch_reports_the_block_statistic() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let args = ["--a", "x1", "--b", "0 - x1", "--gate", "g", "--cuts", "8", "--pnl-blocks", "8"];
    let run = |extra: &[&str]| json(&switch(&data, &spec, &[&args[..], extra, &["--json"]].concat()));
    let sd = run(&[]);
    assert!(at(&sd, &["book"]).get("stat").is_none());
    for (extra, stat, q) in [(&["--pnl-stat", "cvar", "--pnl-cvar-q", "0.25"][..], "cvar", Some(0.25)), (&["--pnl-stat", "mean-semi"][..], "mean-semi", None)] {
        let j = run(extra);
        assert_eq!(at(&j, &["book", "stat"]).as_str(), Some(stat));
        assert_eq!(at(&j, &["book"]).get("cvar_q").and_then(Json::as_f64), q);
        let (gs, gj) = (&arr(&sd, &["gates"])[0], &arr(&j, &["gates"])[0]);
        let mut changed = 0;
        for (a, b) in arr(gs, &["grid", "profile"]).iter().zip(arr(gj, &["grid", "profile"])) {
            for key in ["index", "value", "net", "gross", "flips", "exits", "long_rows", "switches"] {
                assert_eq!(at(a, &[key]), at(b, &[key]), "{stat}: {key}");
            }
            if num(a, &["fitness"]) != num(b, &["fitness"]) {
                changed += 1;
            }
        }
        assert!(changed > 0, "{stat}: the statistic changes the fitness");
        assert_eq!(at(gs, &["best", "index"]), at(gj, &["best", "index"]), "{stat}: the best cut is by net");
        assert_eq!(at(&sd, &["bound"]), at(&j, &["bound"]), "{stat}");
        let text = switch(&data, &spec, &[&args[..], extra].concat());
        assert!(String::from_utf8_lossy(&text.stdout).contains(&format!("fitness statistic: {stat}")), "{}", String::from_utf8_lossy(&text.stdout));
    }
}

/// #1232: the default report (`--pnl-excess none`, mean-sd), JSON and text, is
/// the report of the build before the fitness flags were accepted.
#[test]
fn switch_none_output_is_unchanged() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let args = ["--a", "x1", "--b", "0 - x1", "--gate", "g", "--gate", "h", "--cuts", "8", "--switches", "0,2,8", "--pnl-blocks", "4"];
    let text = switch(&data, &spec, &args);
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    let j = switch(&data, &spec, &[&args[..], &["--json"]].concat());
    assert_eq!(code(&j), 0, "{}", stderr(&j));
    let got = (rcsr::manifest::sha256_hex(&j.stdout), rcsr::manifest::sha256_hex(&text.stdout));
    assert_eq!(got, (PINNED_SWITCH_JSON.to_string(), PINNED_SWITCH_TEXT.to_string()), "the default switch report (JSON, text) changed");
}

const PINNED_SWITCH_JSON: &str = "e7aebc2bc216490a0e28773cd82b965b78bc36ff5ed89c33fb0a48d1eeedd2a8";
const PINNED_SWITCH_TEXT: &str = "d2ad87eaa79a0e1b03a04651a45cb04cae3a0f3a7e8ab18a33890e865160f129";

/// #1248: the sweep budget. The table guard refuses a small `--max-memory` first
/// on any input that fits the test, so the refusal is a unit test of
/// `gate_sweep_within` in `src/switch.rs`.
const SWEEP_ARGS: [&str; 8] = ["--a", "x1", "--b", "0 - x1", "--gate", "g", "--pnl-blocks", "64"];

#[test]
fn switch_reports_sweep_bytes() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let o = switch(&data, &spec, &[&SWEEP_ARGS[..], &["--max-memory", "1GiB", "--json"]].concat());
    let j = json(&o);
    let rows = num(&j, &["train_rows"]) as usize;
    assert_eq!(num(&j, &["book", "sweep_bytes"]) as u64, rcsr::switch::gate_sweep_bytes(rows, 64));
    let text = switch(&data, &spec, &[&SWEEP_ARGS[..], &["--max-memory", "1GiB"]].concat());
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    assert!(String::from_utf8_lossy(&text.stdout).contains("gate sweep memory: an estimated"));
}

#[test]
fn switch_force_skips_sweep_budget() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", 0, |_| None);
    let spec = write_spec(&dir);
    let o = switch(&data, &spec, &[&SWEEP_ARGS[..], &["--max-memory", "1GiB", "--force"]].concat());
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}


/// #1248: `stamps` timestamps of three symbols and six columns; only the row
/// count matters here.
fn write_long_csv(dir: &Path, stamps: usize) -> String {
    let mut text = String::from("t,s,x1,g,r,y\n");
    for k in 0..stamps {
        for sym in SYMBOLS {
            let i = k as f64 * 3.0 + (sym.as_bytes()[0] - b'a') as f64;
            let x1 = frac(i * 0.618_033_988_749_894_9) - 0.5;
            let g = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let r = 0.01 * x1 * g.signum();
            text.push_str(&format!("{},{},{:.4},{:.4},{:.5},{:.5}\n", T0 + STEP * k as i64, sym, x1, g, r, r));
        }
    }
    let path = dir.join("long.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn long_switch(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "switch", "--data", data, "--target", "y", "--vars", "x1", "--time", "t", "--group-by", "s", "--split", "temporal",
        "--target-spec", spec, "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.02", "--test-frac", "0.02",
        "--no-dev-log", "--quiet", "--a", "x1", "--b", "0 - x1", "--gate", "g", "--pnl-blocks", "64",
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

/// 21,000 rows, 64 blocks: the table estimate is about 9.0 MiB and the gate
/// sweep estimate is 12,847,016 bytes (about 12.3 MiB), so this budget passes
/// the table guard and refuses the sweep.
const BETWEEN_BUDGET: &str = "11MiB";
const BETWEEN_STAMPS: usize = 7_000;

#[test]
fn switch_cli_refuses_a_sweep_over_budget() {
    let (_scratch, dir) = scratch();
    let spec = write_spec(&dir);
    let data = write_long_csv(&dir, BETWEEN_STAMPS);
    let o = long_switch(&data, &spec, &["--max-memory", BETWEEN_BUDGET]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let err = stderr(&o);
    let need = rcsr::switch::gate_sweep_bytes(20_166, 64);
    for needle in [need.to_string(), (11u64 << 20).to_string(), "64 blocks".to_string(), "--pnl-blocks".to_string()] {
        assert!(err.contains(&needle), "want '{needle}' in {err}");
    }
}

#[test]
fn switch_cli_force_runs_a_sweep_over_budget() {
    let (_scratch, dir) = scratch();
    let spec = write_spec(&dir);
    let data = write_long_csv(&dir, BETWEEN_STAMPS);
    let o = long_switch(&data, &spec, &["--max-memory", BETWEEN_BUDGET, "--force", "--json"]);
    let j = json(&o);
    assert_eq!(num(&j, &["train_rows"]) as usize, 20_166);
    assert_eq!(num(&j, &["book", "sweep_bytes"]) as u64, rcsr::switch::gate_sweep_bytes(20_166, 64));
}
