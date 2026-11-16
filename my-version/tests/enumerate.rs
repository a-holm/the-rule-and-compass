//! #1183: `rcsr enumerate`, the exhaustive scoring of every expression of a
//! small class on the training book of a `--fitness pnl` run, on a planted panel.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use rcsr::{data, store};
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 3600;
const STAMPS: usize = 600;
const SYMBOLS: [&str; 4] = ["a", "b", "c", "d"];
const OPS10: &str = "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss";
/// The planted cut: `x1 - x2 < Q` holds on 24.5 % of the rows (x1 and x2 are
/// uniform on (-0.5, 0.5)), the third share of the 16-cut grid.
const Q: f64 = -0.3;

fn scratch() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("enumerate");
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

#[derive(Clone, Copy, PartialEq)]
enum Panel {
    /// The return pays only where `x1 - x2 < Q`; x1 and x2 are uniform.
    Planted,
    /// x1 and x2 are whole numbers from -2 to 2 and the return pays only where
    /// `x1 - x2 <= -1`: every cut of `x1 - x2` is on the grid.
    Discrete,
    /// The return does not depend on the features.
    Flat,
}

/// Four symbols on an hourly grid.
fn write_csv(dir: &Path, panel: Panel) -> String {
    let mut text = String::from("t,s,x1,x2,x3,r,y\n");
    for k in 0..STAMPS as i64 {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 4.0 + j as f64;
            let (mut x1, mut x2) = (frac(i * 0.618_033_988_749_894_9) - 0.5, frac(i * 0.754_877_666_246_692_7) - 0.5);
            let x3 = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            if panel == Panel::Discrete {
                x1 = ((x1 + 0.5) * 5.0).floor() - 2.0;
                x2 = ((x2 + 0.5) * 5.0).floor() - 2.0;
            }
            let pays = match panel {
                Panel::Planted => x1 - x2 < Q,
                Panel::Discrete => x1 - x2 <= -1.0,
                Panel::Flat => frac(i * 0.281_718_171_540_954_8) < 0.5,
            };
            let r = (if pays { 0.004 } else { -0.004 }) + 0.002 * noise;
            text.push_str(&format!("{},{},{},{},{},{},{}\n", T0 + STEP * k, sym, x1, x2, x3, r, r));
        }
    }
    let name = match panel {
        Panel::Planted => "planted.csv",
        Panel::Discrete => "discrete.csv",
        Panel::Flat => "flat.csv",
    };
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
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

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn stdout(o: &Output) -> String {
    String::from_utf8_lossy(&o.stdout).into_owned()
}

/// The run flags every command of a test shares: data, target, split, cost and
/// the identity input transform (so a printed expression is the searched one).
fn run_flags<'a>(data: &'a str, spec: &'a str) -> Vec<&'a str> {
    vec![
        "--data", data, "--target-spec", spec, "--target", "y", "--vars", "x1,x2,x3", "--time", "t", "--group-by", "s", "--split", "temporal",
        "--validation-frac", "0.3", "--test-frac", "0.2", "--objective", "pnl", "--cost-bps", "5", "--center", "none", "--scale", "none",
        "--no-dev-log", "--quiet",
    ]
}

fn enumerate(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec!["enumerate"];
    args.extend(run_flags(data, spec));
    args.extend_from_slice(extra);
    rcsr(&args)
}

fn json(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    let text = stdout(o);
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    path.iter().fold(j, |v, k| v.get(k).unwrap_or_else(|| panic!("no {k} in {path:?}")))
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
}

fn arr<'a>(j: &'a Json, path: &[&str]) -> &'a [Json] {
    match at(j, path) {
        Json::Arr(v) => v,
        _ => panic!("{path:?} is not an array"),
    }
}

fn str_at<'a>(j: &'a Json, path: &[&str]) -> &'a str {
    at(j, path).as_str().unwrap_or_else(|| panic!("{path:?} is not a string"))
}

#[test]
fn class_counts_and_thread_invariance() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Planted), write_spec(&dir));
    let base = ["--ops", "add,sub,mul,abs", "--max-size", "3", "--json"];
    let mut reports = Vec::new();
    for threads in ["1", "4"] {
        let mut extra = base.to_vec();
        extra.extend(["--threads", threads]);
        let o = enumerate(&data, &spec, &extra);
        let j = json(&o);
        // Hand count, 3 variables, unary abs, binary add, sub, mul. Raw: 3 inputs,
        // 3 abs(x), 3 abs(abs(x)) + 3 x 9 ordered pairs = 36. Canonical: 3 + 3 +
        // (3 abs(abs(x)) + 6 add with left <= right + 6 sub without sub(a, a) + 6 mul) = 27.
        assert_eq!(num(&j, &["class", "raw"]), 36.0);
        assert_eq!(num(&j, &["class", "canonical"]), 27.0);
        let sizes: Vec<(f64, f64)> = arr(&j, &["class", "sizes"]).iter().map(|s| (num(s, &["raw"]), num(s, &["canonical"]))).collect();
        assert_eq!(sizes, vec![(3.0, 3.0), (3.0, 3.0), (30.0, 21.0)]);
        assert_eq!(num(&j, &["timing", "threads"]), threads.parse::<f64>().unwrap());
        let text = stdout(&o);
        let cut = text.rfind(",\"timing\":").expect("the report ends with its timing");
        reports.push(text[..cut].to_string());
    }
    assert_eq!(reports[0], reports[1], "the report does not depend on --threads");
}

#[test]
fn the_planted_rule_is_the_class_best() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Planted), write_spec(&dir));
    let j = json(&enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "3", "--json"]));
    let best = &arr(&j, &["best"])[2];
    assert_eq!(num(best, &["size"]), 3.0);
    let (tree, side) = (str_at(best, &["tree"]), str_at(best, &["side"]));
    assert!(matches!((tree, side), ("(x1 - x2)", "below") | ("(x2 - x1)", "above")), "{tree} {side}");
    let cut = num(best, &["cut"]) * if side == "below" { 1.0 } else { -1.0 };
    let shares: Vec<f64> = arr(&j, &["class", "shares"]).iter().map(|s| s.as_f64().unwrap()).collect();
    let share = num(best, &["share"]);
    let k = shares.iter().position(|&s| s == share).expect("a grid share");
    assert!(k.abs_diff(2) <= 1, "share {share} (grid index {k}); the planted share is grid index 2");
    assert!((cut - Q).abs() < 0.05, "cut {cut}, planted {Q}");
    // The planted tree is the best of the whole class.
    let top = &arr(&j, &["top"])[0];
    assert_eq!(num(top, &["fitness"]).to_bits(), num(best, &["fitness"]).to_bits());
    assert_eq!(str_at(top, &["expression"]), str_at(best, &["expression"]));
}

#[test]
fn the_family_null_certifies_the_planted_rule() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Planted), write_spec(&dir));
    let j = json(&enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "3", "--null-shifts", "19", "--seed", "11", "--json"]));
    let null = at(&j, &["null"]);
    assert_eq!(num(null, &["shifts"]), 19.0);
    assert_eq!(arr(null, &["best"]).len(), 19);
    let observed = num(null, &["observed"]);
    assert_eq!(observed.to_bits(), num(&arr(&j, &["top"])[0], &["fitness"]).to_bits());
    assert!(arr(null, &["best"]).iter().all(|f| f.as_f64().is_some_and(|f| f > observed)));
    assert!(num(null, &["p_family"]) <= 0.05, "p_family {}", num(null, &["p_family"]));
    assert_eq!(num(&arr(&j, &["top"])[0], &["p_family"]), num(null, &["p_family"]));
    // The shifts come from the walk-forward range of the training periods.
    let periods = num(null, &["periods"]);
    assert_eq!(periods * 4.0, num(&j, &["split", "train_rows"]), "four symbols at every training timestamp");
    let lo = num(null, &["shift_lo"]);
    assert_eq!(lo, num(null, &["block_len"]).min((periods / 4.0).floor()).max(1.0));
    let count = num(null, &["shift_count"]);
    assert_eq!(count, periods + 1.0 - 2.0 * lo);
    assert!(arr(null, &["nums"]).iter().all(|k| k.as_f64().is_some_and(|k| k >= lo && k < lo + count)));
}

#[test]
fn gap_against_a_run() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Discrete), write_spec(&dir));
    let search = |out: &Path, extra: &[&str]| {
        let mut args = run_flags(&data, &spec);
        args.extend(["--fitness", "pnl", "--cpu", "--threads", "1", "--seed", "3", "--generations", "1", "--pop", "16", "--json"]);
        args.extend_from_slice(extra);
        let o = rcsr(&args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        std::fs::write(out, stdout(&o)).unwrap();
        out.to_string_lossy().into_owned()
    };
    let seeds = dir.join("seeds.txt").to_string_lossy().into_owned();
    let early = search(&dir.join("early.json"), &[]);
    let j = json(&enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "3", "--top", "3", "--emit-seeds", &seeds, "--compare-run", &early, "--json"]));
    let planted = arr(&j, &["best"])[2].clone();
    let rows = arr(&j, &["compare", "rows"]);
    assert!(!rows.is_empty());
    assert!(
        rows.iter().any(|r| r.get("gap").and_then(Json::as_f64).is_some_and(|g| g > 0.0)),
        "a run stopped at generation 1 leaves fitness inside the class: {}",
        stdout(&enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "3", "--compare-run", &early]))
    );
    for r in rows {
        let field = |k: &str| r.get(k).and_then(Json::as_f64);
        if let (Some(size), Some(search)) = (field("class_size"), field("search_fitness")) {
            let best = arr(&j, &["best"])[..size as usize].iter().filter_map(|b| b.get("fitness").and_then(Json::as_f64)).fold(f64::INFINITY, f64::min);
            assert_eq!(num(r, &["class_best"]).to_bits(), best.to_bits(), "the best member with at most {size} nodes");
            let nodes = num(r, &["complexity"]) - if num(r, &["constants"]) == 0.0 { 0.0 } else { 2.0 };
            assert_eq!(size, nodes);
            assert_eq!(num(r, &["gap"]), search - best);
        }
    }
    // --emit-seeds: the top members, best first, read by --seed-file.
    let lines: Vec<String> = std::fs::read_to_string(&seeds).unwrap().lines().map(str::to_string).collect();
    assert_eq!(lines.len(), 3);
    assert_eq!(lines[0], str_at(&planted, &["expression"]));
    let seeded = search(&dir.join("seeded.json"), &["--seed-file", &seeds, "--no-fit"]);
    let j = json(&enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "3", "--compare-run", &seeded, "--json"]));
    // The planted book is the optimum of the class; the front holds it at tree size 3, as
    // the seed (c - (x1 - x2)) at complexity 5 or as a constant-free (x2 - x1) at complexity 3.
    let at3: Vec<&Json> = arr(&j, &["compare", "rows"]).iter().filter(|r| r.get("class_size").and_then(Json::as_f64) == Some(3.0)).collect();
    assert!(!at3.is_empty(), "the seeded front holds a member of tree size 3");
    assert_eq!(num(at3[0], &["gap"]), 0.0, "the front holds the planted rule: {}", str_at(at3[0], &["expression"]));
    assert_eq!(num(at3[0], &["search_fitness"]).to_bits(), num(&planted, &["fitness"]).to_bits());
    // walkforward --expr accepts the first member.
    let wf_out = dir.join("wf.json").to_string_lossy().into_owned();
    let w = format!("w,{},{},{},{}", T0, T0 + 99 * STEP, T0 + 101 * STEP, T0 + 300 * STEP);
    let wf = rcsr(&[
        "walkforward", "--data", &data, "--target", "y", "--target-spec", &spec, "--expr", &lines[0], "--vars", "x1,x2,x3", "--time", "t",
        "--group-by", "s", "--window", &w, "--threshold-fixed", "0", "--cost-bps", "5", "--out", &wf_out,
    ]);
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let report = Json::parse(std::fs::read_to_string(&wf_out).unwrap().trim()).unwrap();
    assert!(num(&arr(&report, &["windows"])[0], &["metrics", "rows"]) > 0.0);
    // A run on another book is refused.
    let other = search(&dir.join("other.json"), &["--pnl-blocks", "2"]);
    let o = enumerate(&data, &spec, &["--ops", OPS10, "--compare-run", &other]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("book_hash"), "{}", stderr(&o));
}

#[test]
fn enumerate_refuses_what_it_cannot_price() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Planted), write_spec(&dir));
    for (extra, needle) in [
        (vec!["--ops", OPS10, "--gpu"], "--gpu"),
        (vec!["--ops", OPS10, "--band", "0.1"], "--band"),
        (vec!["--ops", OPS10, "--min-holding", "3"], "--min-holding"),
        (vec!["--ops", OPS10, "--pnl-mask", "x3"], "--pnl-mask"),
        (vec!["--ops", OPS10, "--cash-return", "0.0001"], "--cash-return"),
        (vec!["--ops", OPS10, "--missing-score", "hold"], "--missing-score"),
        (vec!["--ops", OPS10, "--pnl-mask-band", "0.1"], "--pnl-mask-band"),
        (vec!["--ops", OPS10, "--pnl-mask-dwell", "3"], "--pnl-mask-dwell"),
        (vec!["--ops", OPS10, "--band-units", "sd"], "--band-units"),
        (vec!["--ops", OPS10, "--seed-expr", "x1"], "--seed-expr"),
        (vec!["--ops", OPS10, "--seed-file", "seeds.txt"], "--seed-file"),
        (vec!["--ops", OPS10, "--eval-expr", "x1"], "--eval-expr"),
        (vec!["--ops", OPS10, "--resume", "run.ckpt"], "--resume"),
        (vec!["--ops", OPS10, "--save", "run.ckpt"], "--save"),
        (vec!["--ops", OPS10, "--reselect", "run.ckpt"], "--reselect"),
        (vec!["--ops", OPS10, "--reselect-run", "run.json"], "--reselect-run"),
        (vec!["--ops", OPS10, "--gate-search"], "--gate-search"),
        (vec!["--ops", OPS10, "--branch-on", "cash"], "--branch-on"),
        (vec!["--ops", OPS10, "--branch-off", "cash"], "--branch-off"),
        (vec!["--ops", OPS10, "--gate-band", "0.1"], "--gate-band"),
        (vec!["--ops", OPS10, "--gate-dwell", "3"], "--gate-dwell"),
        (vec!["--ops", "add,if"], "'if'"),
        (vec!["--max-size", "3"], "--ops"),
        (vec!["--ops", OPS10, "--fitness", "metric"], "--fitness"),
        (vec!["--ops", OPS10, "--cuts", "15"], "--cuts"),
        (vec!["--ops", OPS10, "--null-shifts", "5"], "--null-shifts 5"),
        (vec!["--ops", OPS10, "--null-shifts", "400"], "distinct shifts"),
        (vec!["--ops", OPS10, "--max-size", "5"], "canonical trees"),
    ] {
        let o = enumerate(&data, &spec, &extra);
        assert_eq!(code(&o), 2, "{extra:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{extra:?}: {}", stderr(&o));
    }
    let o = enumerate(&data, &spec, &["--ops", OPS10, "--max-size", "5"]);
    assert!(stderr(&o).contains("book passes") && stderr(&o).contains("--allow-large"), "{}", stderr(&o));
}

#[test]
fn enumerate_reads_no_locked_row() {
    let (_s, dir) = scratch();
    let (data, spec) = (write_csv(&dir, Panel::Flat), write_spec(&dir));
    // The rows run from 2020-09-13 for 25 days; the lockbox covers some of them.
    let lock = ["--lockbox", "2020-09-30,2020-10-31"];
    let mut args = run_flags(&data, &spec);
    args.extend(["--fitness", "pnl", "--cpu", "--generations", "1", "--pop", "8"]);
    args.extend(lock);
    let searched = rcsr(&args);
    assert_ne!(code(&searched), 0, "the search refuses rows inside the lockbox");
    let mut extra = vec!["--ops", "add,sub"];
    extra.extend(lock);
    let o = enumerate(&data, &spec, &extra);
    assert_eq!(code(&o), code(&searched), "{}", stderr(&o));
    assert!(stderr(&searched).to_lowercase().contains("lockbox"), "{}", stderr(&searched));
    assert!(stderr(&o).to_lowercase().contains("lockbox"), "{}", stderr(&o));
}

/// `--sample-rows` exits 2 before any row is read: the class optimum and the
/// shared-time-shift null are claims about every training row of a contiguous series.
#[test]
fn enumerate_refuses_sample_rows() {
    let (_s, dir) = scratch();
    let (csv, spec) = (write_csv(&dir, Panel::Planted), write_spec(&dir));
    let table = data::parse_table(&std::fs::read_to_string(&csv).unwrap()).unwrap();
    let rcsb = dir.join("d.rcsb").to_string_lossy().into_owned();
    store::write_store(&rcsb, &table, 251).unwrap();
    let (out, seeds) = (dir.join("e.json").to_string_lossy().into_owned(), dir.join("seeds.txt").to_string_lossy().into_owned());
    let files = |dir: &Path| {
        let mut names: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    };
    let before = files(&dir);
    // A row count below the store's 2,400 rows and one above it: the flag is what is refused.
    for n in ["1000", "999999"] {
        let o = enumerate(&rcsb, &spec, &["--ops", OPS10, "--sample-rows", n, "--out", &out, "--emit-seeds", &seeds, "--json"]);
        let err = stderr(&o);
        assert_eq!(code(&o), 2, "--sample-rows {n}: {err}");
        assert!(err.contains("--sample-rows") && err.contains("rcsr enumerate"), "--sample-rows {n}: {err}");
        assert!(!err.contains("sampling (explicit)"), "--sample-rows {n}: {err}");
        assert!(stdout(&o).is_empty());
        assert_eq!(files(&dir), before, "the refusal writes nothing");
    }
    let o = enumerate(&rcsb, &spec, &["--ops", OPS10, "--json"]);
    assert_eq!(code(&o), 0, "the store runs without --sample-rows: {}", stderr(&o));
}
