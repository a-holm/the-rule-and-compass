//! #1181: `rcsr compose`, the regime-switching composition of specialist pools
//! into one exact formula, on a planted panel.

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
const SIDE: f64 = 0.0005;

fn scratch() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("compose");
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// The regime gate of stamp `k`: positive in alternate 48-stamp blocks, the
/// same for every symbol.
fn regime(k: i64) -> f64 {
    let m = 0.5 + 0.4 * frac(k as f64 * 0.324_717_957_244_746);
    if (k / 48) % 2 == 0 {
        m
    } else {
        -m
    }
}

fn x1(k: i64, j: usize) -> f64 {
    frac(((k + 40) as f64 * 3.0 + j as f64) * 0.618_033_988_749_894_9) - 0.5
}

/// Three symbols on a 900 s grid. `x1` pays while the gate `g` is positive and
/// `0 - x1` while it is negative; `h` varies inside a timestamp;
/// `lbl_h24a_y` is a label-family column no gate may read. `alter` rewrites
/// the gate, the return and `x1` of a row from its stamp.
fn write_csv(dir: &Path, name: &str, alter: impl Fn(i64) -> Option<(f64, f64, f64)>) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y,lbl_h24a_y\n");
    for k in 0..STAMPS as i64 {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let g = regime(k);
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1(k, j) + 0.0005 * noise;
            let (g, r, x) = alter(k).unwrap_or((g, r, x1(k, j)));
            text.push_str(&format!("{},{},{},{},{},{},{},{},{}\n", T0 + STEP * k, sym, x, x2, g, h, r, r, r));
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

/// A gate menu of `(name, codeword, expr, band, dwell)` entries.
fn write_menu(dir: &Path, name: &str, gates: &[(&str, &str, &str, f64, usize)], budget: Option<usize>) -> String {
    let list: Vec<String> = gates
        .iter()
        .map(|(n, c, e, b, d)| format!("{{\"name\":\"{n}\",\"codeword\":\"{c}\",\"expr\":\"{e}\",\"band\":{b},\"dwell\":{d}}}"))
        .collect();
    let budget = budget.map_or("null".to_string(), |k| k.to_string());
    let path = dir.join(name);
    std::fs::write(&path, format!("{{\"schema\":\"rcsr-gate-menu/1\",\"switch_budget\":{budget},\"gates\":[{}]}}", list.join(","))).unwrap();
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

const RUN_FLAGS: [&str; 20] = [
    "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--split", "temporal", "--objective", "pnl", "--cost-bps", "5",
    "--validation-frac", "0.3", "--test-frac", "0.2", "--no-dev-log", "--quiet",
];

/// A pool run: a short real search on `data` (its split, target spec and cost
/// blocks are the engine's), with its front replaced by `exprs` in archive order.
fn pool_run(dir: &Path, name: &str, data: &str, spec: &str, extra: &[&str], exprs: &[&str]) -> String {
    let mut args = vec!["--data", data, "--target-spec", spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--cpu", "--seed", "3", "--generations", "2", "--pop", "16", "--json"]);
    args.extend_from_slice(extra);
    let o = rcsr(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let body = String::from_utf8_lossy(&o.stdout).trim().to_string();
    let cut = body.rfind(",\"pareto\":[").expect("run.json ends with its pareto");
    let members: Vec<String> = exprs
        .iter()
        .enumerate()
        .map(|(k, e)| format!("{{\"archive_rank\":{},\"expression\":\"{e}\",\"expression_exact\":\"{e}\",\"complexity\":{}}}", k + 1, e.len()))
        .collect();
    let path = dir.join(name);
    std::fs::write(&path, format!("{},\"pareto\":[{}]}}\n", &body[..cut], members.join(","))).unwrap();
    path.to_string_lossy().into_owned()
}

fn compose(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec!["compose", "--data", data, "--target-spec", spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend_from_slice(extra);
    rcsr(&args)
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

fn s<'a>(j: &'a Json, path: &[&str]) -> &'a str {
    at(j, path).as_str().unwrap_or_else(|| panic!("{path:?} is not a string"))
}

/// The planted pools: the bull run holds `x1` first, the bear run `0 - x1`.
struct Planted {
    _scratch: Scratch,
    dir: PathBuf,
    data: String,
    spec: String,
    bull: String,
    bear: String,
}

fn planted(bull: &[&str], bear: &[&str]) -> Planted {
    planted_with(bull, bear, &[])
}

/// The planted pools searched with `extra` added to both pool searches.
fn planted_with(bull: &[&str], bear: &[&str], extra: &[&str]) -> Planted {
    let (scratch, dir) = scratch();
    let data = write_csv(&dir, "d.csv", |_| None);
    let spec = write_spec(&dir);
    let bull = pool_run(&dir, "bull.json", &data, &spec, &[&["--pnl-mask", "g"][..], extra].concat(), bull);
    let bear = pool_run(&dir, "bear.json", &data, &spec, &[&["--pnl-mask", "0 - g"][..], extra].concat(), bear);
    Planted { _scratch: scratch, dir, data, spec, bull, bear }
}

const XS: [&str; 2] = ["--pnl-excess", "xs-mean"];

const BULL: [&str; 3] = ["x1", "x2", "x1 + x2"];
const BEAR: [&str; 3] = ["0 - x1", "x2", "x2 - x1"];

fn composite<'a>(j: &'a Json, id: usize) -> &'a Json {
    let list = arr(j, &["compose", "composites"]);
    list.iter().find(|c| num(c, &["id"]) == id as f64).expect("composite id")
}

#[test]
fn compose_finds_the_planted_switch() {
    let p = planted(&BULL, &BEAR);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1)], None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let args = ["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--top", "3", "--json", "--out", &out];
    let first = compose(&p.data, &p.spec, &args);
    let j = json(&first);
    assert_eq!(s(&j, &["compose", "schema"]), "rcsr-compose/1");
    assert_eq!(s(&j, &["compose", "selection", "verdict"]), "selected");
    let finalist = s(&j, &["objective", "finalist", "expression_exact"]);
    assert_eq!(finalist, "if(greater(g, 0), x1, 0 - x1)");
    assert_eq!(s(&j, &["compose", "chosen", "type"]), "C2");
    assert_eq!(num(&j, &["objective", "spec", "threshold"]), 0.0);
    let best = num(&j, &["objective", "validation", "net_pnl"]);
    // The branches alone and every if(g, A, cash) pair earn less on the validation rows.
    for b in arr(&j, &["compose", "chosen", "branches"]) {
        assert!(best > num(b, &["validation", "net"]), "branch {}: {best}", num(b, &["member"]));
    }
    let cash: Vec<&Json> = arr(&j, &["compose", "composites"]).iter().filter(|c| s(c, &["type"]) == "C1").collect();
    assert_eq!(cash.len(), 2 * 3 * 2);
    for c in cash {
        assert!(best > num(c, &["validation", "net"]), "{}: {best}", s(c, &["expression"]));
    }
    // Both gate states earn: the regime condition holds for the finalist.
    let regime = at(&j, &["compose", "chosen", "validation", "regime"]);
    assert!(num(regime, &["states", "on", "alpha"]) > 0.0 && num(regime, &["states", "off", "alpha"]) > 0.0);
    let parts = num(regime, &["states", "on", "net"]) + num(regime, &["states", "off", "net"]) + num(regime, &["seam"]);
    assert!((parts - best).abs() <= 1e-12, "state nets plus the seam make the composite net: {parts} vs {best}");
    assert!(num(regime, &["seam"]) >= -1e-15);
    let seam = at(&j, &["compose", "chosen", "train_seam"]);
    assert!((num(seam, &["sum"]) - num(seam, &["composite"])).abs() <= 1e-9);
    // The written file is the printed run.json, and a second run prints the same bytes.
    assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), String::from_utf8_lossy(&first.stdout).trim());
    let second = compose(&p.data, &p.spec, &args);
    assert_eq!(first.stdout, second.stdout, "compose is deterministic");
    let text = compose(&p.data, &p.spec, &args[..8]);
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    assert!(String::from_utf8_lossy(&text.stdout).contains("finalist C2"), "{}", String::from_utf8_lossy(&text.stdout));
}

#[test]
fn composite_prices_the_same_in_walkforward() {
    let p = planted(&BULL, &BEAR);
    replays_in_walkforward(&p, &[]);
}

/// #1232: pools searched under `--pnl-excess xs-mean` compose with the flag, and
/// the run prices the absolute book: walk-forward reproduces its nets.
#[test]
fn compose_xs_mean_prices_the_same_in_walkforward() {
    let p = planted_with(&BULL, &BEAR, &XS);
    let j = replays_in_walkforward(&p, &XS);
    assert_eq!(s(&j, &["objective", "spec", "pnl_excess"]), "xs_mean");
    assert_eq!(s(&j, &["objective", "finalist", "expression_exact"]), "if(greater(g, 0), x1, 0 - x1)");
    for slot in ["validation", "test"] {
        let b = at(&j, &["objective", slot]);
        assert_ne!(num(b, &["xs_excess_net"]), num(b, &["net_pnl"]), "{slot}");
    }
}

/// Composes `p` (gate `g`, `extra` added) and checks that `walkforward
/// --from-run` on the test rows and `walkforward --expr` on the validation rows
/// give the run's absolute nets to 1e-12. Returns the run.
fn replays_in_walkforward(p: &Planted, extra: &[&str]) -> Json {
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let mut args = vec!["--gates", menu.as_str(), "--bull", p.bull.as_str(), "--bear", p.bear.as_str(), "--top", "3", "--json", "--out", out.as_str()];
    args.extend_from_slice(extra);
    let j = json(&compose(&p.data, &p.spec, &args));
    let expr = s(&j, &["objective", "finalist", "expression_exact"]).to_string();
    let test_rows = num(&j, &["test_rows"]) as i64;
    let k_test = STAMPS as i64 - test_rows / SYMBOLS.len() as i64;
    let first_val = num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64;
    let last_val = num(&arr(&j, &["split", "validation_time"])[1], &[]) as i64;
    let net_of = |report: &Json| {
        let m = at(&arr(report, &["windows"])[0], &["metrics"]);
        num(m, &["mean_return"]) * num(m, &["periods"]) * SYMBOLS.len() as f64
    };
    // The run's book on its test rows.
    let wf_out = p.dir.join("wf.json").to_string_lossy().into_owned();
    let w = format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64);
    let wf = rcsr(&[
        "walkforward", "--data", &p.data, "--target", "y", "--from-run", &out, "--time", "t", "--group-by", "s", "--window", &w, "--out", &wf_out,
    ]);
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let report = Json::parse(std::fs::read_to_string(&wf_out).unwrap().trim()).unwrap();
    let test_net = num(&j, &["objective", "test", "net_pnl"]);
    assert!((net_of(&report) - test_net).abs() <= 1e-12, "walkforward {} vs objective.test {test_net}", net_of(&report));
    assert_eq!(num(&at(&arr(&report, &["windows"])[0], &["metrics"]), &["rows"]), test_rows as f64);
    // The printed expression priced directly on the validation rows is the composite's validation book.
    let wv_out = p.dir.join("wv.json").to_string_lossy().into_owned();
    let w = format!("v,{},{},{},{}", T0, first_val - STEP, first_val, last_val + 1);
    let wv = rcsr(&[
        "walkforward", "--data", &p.data, "--target", "y", "--target-spec", &p.spec, "--expr", &expr, "--vars", "g,x1,x2", "--time", "t", "--group-by", "s",
        "--window", &w, "--threshold-fixed", "0", "--cost-bps", "5", "--out", &wv_out,
    ]);
    assert_eq!(code(&wv), 0, "{}", stderr(&wv));
    let report = Json::parse(std::fs::read_to_string(&wv_out).unwrap().trim()).unwrap();
    let val_net = num(&j, &["objective", "validation", "net_pnl"]);
    assert!((net_of(&report) - val_net).abs() <= 1e-12, "walkforward {} vs objective.validation {val_net}", net_of(&report));
    j
}

/// The log return of row `(k, j)` of [`write_csv`]; the book prices `exp_m1` of it.
fn ret(k: i64, j: usize) -> f64 {
    let i = (k + 40) as f64 * 3.0 + j as f64;
    let noise = frac(i * 0.324_717_957_244_746) - 0.5;
    (if regime(k) > 0.0 { 0.01 } else { -0.01 }) * x1(k, j) + 0.0005 * noise
}

/// #1232: under xs-mean every priced book reports `xs_excess_net`; the
/// finalist's is its absolute net less the sum over rows of its position times
/// the mean return of the three symbols at the row's stamp, by hand. A
/// re-selection of the run needs the same flag.
#[test]
fn compose_xs_mean_reports_excess_net() {
    let p = planted_with(&["x1"], &["0 - x1"], &XS);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let args = ["--gates", menu.as_str(), "--bull", p.bull.as_str(), "--bear", p.bear.as_str(), "--pnl-excess", "xs-mean"];
    let j = json(&compose(&p.data, &p.spec, &[&args[..], &["--json", "--out", out.as_str()]].concat()));
    assert_eq!(s(&j, &["objective", "finalist", "expression_exact"]), "if(greater(g, 0), x1, 0 - x1)");
    assert_eq!(s(&j, &["objective", "spec", "pnl_excess"]), "xs_mean");
    assert_eq!(s(&j, &["compose", "chosen", "excess"]), "xs_mean");
    for r in arr(&j, &["compose", "runs"]) {
        assert_eq!(s(r, &["pnl_excess"]), "xs_mean");
        assert!(r.get("stat").is_none());
    }
    let pos = |k: i64, jj: usize| -> f64 {
        let v = if regime(k) > 0.0 { x1(k, jj) } else { 0.0 - x1(k, jj) };
        if v > 0.0 {
            1.0
        } else {
            0.0
        }
    };
    let mean = |k: i64| (0..SYMBOLS.len()).map(|jj| ret(k, jj).exp_m1()).sum::<f64>() / SYMBOLS.len() as f64;
    let held = |k0: i64, k1: i64| -> f64 { (k0..=k1).map(|k| (0..SYMBOLS.len()).map(|jj| pos(k, jj)).sum::<f64>() * mean(k)).sum() };
    let k0 = (num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64 - T0) / STEP;
    let k1 = (num(&arr(&j, &["split", "validation_time"])[1], &[]) as i64 - T0) / STEP;
    let kt = STAMPS as i64 - num(&j, &["test_rows"]) as i64 / SYMBOLS.len() as i64;
    for (slot, a, b) in [("validation", k0, k1), ("test", kt, STAMPS as i64 - 1)] {
        let book = at(&j, &["objective", slot]);
        let net = num(book, &["net_pnl"]);
        let got = num(book, &["xs_excess_net"]);
        let want = net - held(a, b);
        assert!((got - want).abs() <= 1e-12, "{slot}: xs_excess_net {got} vs net - sum p m {want}");
        assert!((got - net).abs() > 1e-6, "{slot}: the excess differs from the net");
        assert_eq!(num(&j, &["compose", "chosen", slot, "xs_excess_net"]), got, "{slot}");
        assert_eq!(num(&j, &["compose", "chosen", slot, "net"]), net, "{slot}");
    }
    for b in arr(&j, &["compose", "chosen", "branches"]) {
        for slot in ["validation", "test"] {
            assert!(num(b, &[slot, "xs_excess_net"]).is_finite(), "branch {}: {slot}", num(b, &["member"]));
        }
    }
    for c in arr(&j, &["compose", "composites"]) {
        assert!(num(c, &["validation", "xs_excess_net"]).is_finite(), "{}", s(c, &["expression"]));
    }
    let text = compose(&p.data, &p.spec, &args);
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    let text = String::from_utf8_lossy(&text.stdout);
    assert!(text.contains("excess: xs_mean") && text.contains("xs_excess_net"), "{text}");
    let reselect = |extra: &[&str]| {
        let mut args = vec!["--data", p.data.as_str(), "--target-spec", p.spec.as_str()];
        args.extend_from_slice(&RUN_FLAGS);
        args.extend_from_slice(extra);
        args.extend(["--reselect-run", out.as_str(), "--select-rule", "robust", "--json"]);
        rcsr(&args)
    };
    let again = json(&reselect(&XS));
    for key in ["expression_exact", "switch"] {
        assert_eq!(at(&again, &["objective", "finalist", key]), at(&j, &["objective", "finalist", key]), "{key}");
    }
    let (a, b) = (num(&again, &["objective", "validation", "xs_excess_net"]), num(&j, &["objective", "validation", "xs_excess_net"]));
    assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
    let refused = reselect(&[]);
    assert_eq!(code(&refused), 2, "{}", stderr(&refused));
    assert!(stderr(&refused).contains("objective.spec.pnl_excess"), "{}", stderr(&refused));
}

/// #1232: selection is on absolute net and alpha. The same pool expressions
/// searched and composed under xs-mean give the finalist, the ranking and the
/// validation books of the run without it; the training pruning reads the
/// training excess.
#[test]
fn compose_selection_is_absolute_under_xs_mean() {
    let plain = planted(&BULL, &BEAR);
    let xs = planted_with(&BULL, &BEAR, &XS);
    let aw = ["x2", "0 - x1", "x1 + x2"];
    let plain_aw = pool_run(&plain.dir, "aw.json", &plain.data, &plain.spec, &[], &aw);
    let xs_aw = pool_run(&xs.dir, "aw.json", &xs.data, &xs.spec, &XS, &aw);
    let gates = [("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1)];
    let run = |p: &Planted, aw: &str, budget: Option<usize>, extra: &[&str]| -> Json {
        let menu = write_menu(&p.dir, "menu.json", &gates, budget);
        let mut args = vec!["--gates", menu.as_str(), "--bull", p.bull.as_str(), "--bear", p.bear.as_str(), "--all-weather", aw, "--top", "3", "--json"];
        args.extend_from_slice(extra);
        json(&compose(&p.data, &p.spec, &args))
    };
    let (a, b) = (run(&plain, &plain_aw, None, &[]), run(&xs, &xs_aw, None, &XS));
    assert_eq!(at(&a, &["objective", "finalist"]), at(&b, &["objective", "finalist"]));
    assert_eq!(at(&a, &["compose", "selection"]), at(&b, &["compose", "selection"]));
    for slot in ["validation", "test"] {
        assert_eq!(num(&a, &["objective", slot, "net_pnl"]), num(&b, &["objective", slot, "net_pnl"]), "{slot}");
        assert!(at(&a, &["objective", slot]).get("xs_excess_net").is_none());
        assert!(num(&b, &["objective", slot, "xs_excess_net"]).is_finite());
        let (x, y) = (at(&a, &["compose", "chosen", "best_all_weather", slot]), at(&b, &["compose", "chosen", "best_all_weather", slot]));
        assert_eq!(num(x, &["net"]), num(y, &["net"]), "best all-weather {slot}");
        assert!(x.get("xs_excess_net").is_none() && num(y, &["xs_excess_net"]).is_finite());
    }
    let (ca, cb) = (arr(&a, &["compose", "composites"]), arr(&b, &["compose", "composites"]));
    assert_eq!(ca.len(), cb.len());
    for (x, y) in ca.iter().zip(cb) {
        for key in ["id", "rank", "type", "expression", "pruned", "bits"] {
            assert_eq!(at(x, &[key]), at(y, &[key]), "{key} of {}", s(x, &["expression"]));
        }
        for key in ["net", "objective", "pooled_alpha", "alpha_on", "alpha_off", "eligible", "robust_eligible"] {
            assert_eq!(at(x, &["validation", key]), at(y, &["validation", key]), "validation {key} of {}", s(x, &["expression"]));
        }
        assert!(at(x, &["validation"]).get("xs_excess_net").is_none());
    }
    assert_eq!(arr(&a, &["pareto"]).iter().map(|m| s(m, &["expression"]).to_string()).collect::<Vec<_>>(), arr(&b, &["pareto"]).iter().map(|m| s(m, &["expression"]).to_string()).collect::<Vec<_>>());
    let (a8, b8) = (run(&plain, &plain_aw, Some(8), &[]), run(&xs, &xs_aw, Some(8), &XS));
    let gains = |j: &Json| -> Vec<Option<f64>> { arr(j, &["compose", "composites"]).iter().map(|c| at(c, &["bound_gain"]).as_f64()).collect() };
    let (ga, gb) = (gains(&a8), gains(&b8));
    assert!(ga.iter().flatten().count() > 0 && ga.iter().flatten().count() == gb.iter().flatten().count());
    assert_ne!(ga, gb, "the pruning bound is on the training excess under xs-mean");
}

/// #1232: a pool run whose `objective.spec.pnl_excess` differs from the flag, in
/// either direction, exits 2 and names the field; so do mixed pools.
#[test]
fn compose_refuses_pool_excess_mismatch() {
    let p = planted(&BULL, &BEAR);
    let xs_bull = pool_run(&p.dir, "xs_bull.json", &p.data, &p.spec, &["--pnl-mask", "g", "--pnl-excess", "xs-mean"], &BULL);
    let xs_bear = pool_run(&p.dir, "xs_bear.json", &p.data, &p.spec, &["--pnl-mask", "0 - g", "--pnl-excess", "xs-mean"], &BEAR);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let refused = |bull: &str, bear: &str, extra: &[&str], needle: &str| {
        let o = compose(&p.data, &p.spec, &[&["--gates", menu.as_str(), "--bull", bull, "--bear", bear][..], extra].concat());
        assert_eq!(code(&o), 2, "{bull} {bear} {extra:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{bull} {bear} {extra:?}: want '{needle}' in {}", stderr(&o));
    };
    refused(&p.bull, &p.bear, &XS, "differs from this invocation in objective.spec.pnl_excess");
    refused(&xs_bull, &xs_bear, &[], "differs from this invocation in objective.spec.pnl_excess");
    refused(&p.bull, &xs_bear, &XS, "differ in objective.spec.pnl_excess");
    refused(&xs_bull, &p.bear, &[], "differ in objective.spec.pnl_excess");
    let ok = compose(&p.data, &p.spec, &["--gates", menu.as_str(), "--bull", xs_bull.as_str(), "--bear", xs_bear.as_str(), "--pnl-excess", "xs-mean"]);
    assert_eq!(code(&ok), 0, "{}", stderr(&ok));
}

/// #1232 (with #1178): pools searched under another block statistic compose with
/// the same `--pnl-stat` (and `--pnl-cvar-q`). The choice reads no statistic, so
/// the run is the run of mean-sd pools of the same expressions, apart from
/// `compose.runs[].stat`; a statistic that differs from the flag exits 2.
#[test]
fn compose_accepts_block_statistic_pools() {
    let p = planted(&BULL, &BEAR);
    let pool = |name: &str, mask: &str, extra: &[&str], exprs: &[&str]| pool_run(&p.dir, name, &p.data, &p.spec, &[&["--pnl-mask", mask][..], extra].concat(), exprs);
    let semi = ["--pnl-stat", "mean-semi"];
    let cvar = ["--pnl-stat", "cvar", "--pnl-cvar-q", "0.5"];
    let (semi_bull, semi_bear) = (pool("semi_bull.json", "g", &semi, &BULL), pool("semi_bear.json", "0 - g", &semi, &BEAR));
    let (cvar_bull, cvar_bear) = (pool("cvar_bull.json", "g", &cvar, &BULL), pool("cvar_bear.json", "0 - g", &cvar, &BEAR));
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1)], Some(8));
    let args = |bull: &str, bear: &str, extra: &[&str]| -> Output {
        compose(&p.data, &p.spec, &[&["--gates", menu.as_str(), "--bull", bull, "--bear", bear, "--top", "3", "--json"][..], extra].concat())
    };
    let base = json(&args(&p.bull, &p.bear, &[]));
    for r in arr(&base, &["compose", "runs"]) {
        assert!(r.get("stat").is_none() && r.get("cvar_q").is_none() && r.get("pnl_excess").is_none());
    }
    let same = |j: &Json, stat: &str, q: Option<f64>| {
        assert_eq!(without(&base, &["command", "compose.runs"]), without(j, &["command", "compose.runs"]), "{stat}: the statistic changes nothing but compose.runs");
        for r in arr(j, &["compose", "runs"]) {
            assert_eq!(s(r, &["stat"]), stat);
            assert_eq!(r.get("cvar_q").and_then(Json::as_f64), q);
        }
    };
    same(&json(&args(&semi_bull, &semi_bear, &semi)), "mean-semi", None);
    same(&json(&args(&cvar_bull, &cvar_bear, &cvar)), "cvar", Some(0.5));
    let refused = |o: Output, needle: &str| {
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        assert!(stderr(&o).contains(needle), "want '{needle}' in {}", stderr(&o));
    };
    refused(args(&semi_bull, &semi_bear, &[]), "differs from this invocation in search_fitness.stat");
    refused(args(&p.bull, &p.bear, &semi), "differs from this invocation in search_fitness.stat");
    refused(args(&cvar_bull, &cvar_bear, &["--pnl-stat", "cvar", "--pnl-cvar-q", "0.25"]), "differs from this invocation in search_fitness.stat");
    refused(args(&semi_bull, &cvar_bear, &semi), "differ in search_fitness.stat");
}

/// #1232: the default output of compose (pools and invocation without
/// `--pnl-excess` and `--pnl-stat`), JSON and text, is the output of the build
/// before those flags were accepted. The scratch directory, the pool file hashes
/// and the version are replaced before hashing.
#[test]
fn compose_none_output_is_unchanged() {
    let p = planted(&BULL, &BEAR);
    let aw = pool_run(&p.dir, "aw.json", &p.data, &p.spec, &[], &["x2", "0 - x1", "x1 + x2"]);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1)], Some(8));
    let args = ["--gates", menu.as_str(), "--bull", p.bull.as_str(), "--bear", p.bear.as_str(), "--all-weather", aw.as_str(), "--top", "3"];
    let text = compose(&p.data, &p.spec, &args);
    assert_eq!(code(&text), 0, "{}", stderr(&text));
    let j = compose(&p.data, &p.spec, &[&args[..], &["--json"]].concat());
    assert_eq!(code(&j), 0, "{}", stderr(&j));
    let mut body = String::from_utf8_lossy(&j.stdout).into_owned();
    for f in [&p.bull, &p.bear, &aw] {
        body = body.replace(&rcsr::manifest::sha256_hex(&std::fs::read(f).unwrap()), "<sha256>");
    }
    let body = body.replace(&*p.dir.to_string_lossy(), "<dir>").replace(&format!("\"version\":\"{}\"", env!("CARGO_PKG_VERSION")), "\"version\":\"<version>\"");
    let got = (rcsr::manifest::sha256_hex(body.as_bytes()), rcsr::manifest::sha256_hex(&text.stdout));
    assert_eq!(got, (PINNED_COMPOSE_JSON.to_string(), PINNED_COMPOSE_TEXT.to_string()), "the default compose output (run.json, summary) changed");
}

const PINNED_COMPOSE_JSON: &str = "246fd7729f55a5c2fa1f1c877c9eaf502f8d30f0a68b9492fe64c49c685bc550";
const PINNED_COMPOSE_TEXT: &str = "25d6add1c7bfa1316484cc48bd3db1c2db6bba1cca8bf8a0e911a44411046357";

#[test]
fn hysteresis_gate_is_carried() {
    let p = planted(&BULL, &BEAR);
    let masked = pool_run(&p.dir, "masked.json", &p.data, &p.spec, &["--pnl-mask", "h", "--pnl-mask-band", "0.02", "--pnl-mask-dwell", "3"], &BULL);
    let band = write_menu(&p.dir, "band.json", &[("h", "011", "h", 0.02, 3)], None);
    let plain = write_menu(&p.dir, "plain.json", &[("h", "011", "h", 0.0, 1)], None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let j = json(&compose(&p.data, &p.spec, &["--gates", &band, "--bull", &masked, "--bear", &p.bear, "--top", "2", "--json", "--out", &out]));
    // #1214: the finalist and every C1 and C2 carry their switch; no gate is carried.
    assert!(at(&j, &["compose", "gate"]).get("carried").is_none());
    let sw = at(&j, &["objective", "finalist", "switch"]);
    assert_eq!((s(sw, &["gate", "expr"]), num(sw, &["gate", "band"]), num(sw, &["gate", "dwell"])), ("h", 0.02, 3.0));
    assert_eq!(s(sw, &["gate", "state_rule"]), "rcsr-gate-state/1");
    let pareto = arr(&j, &["pareto"]);
    assert_eq!(pareto.len() as f64, num(&j, &["compose", "count", "composites"]), "pareto holds every composite");
    for m in pareto {
        assert_eq!(m.get("switch").is_some(), s(m, &["type"]) != "C0", "{}", s(m, &["expression"]));
        assert!(matches!(at(m, &["pruned"]), Json::Bool(_)));
    }
    assert!(at(&j, &["compose", "chosen", "validation", "regime", "start"]).get("warmup_rows").is_some());
    // The validation states are the --pnl-mask states of the same gate under the same rule.
    let mask = Json::parse(std::fs::read_to_string(&masked).unwrap().trim()).unwrap();
    assert_eq!(s(&mask, &["objective", "spec", "pnl_mask", "state_rule"]), "rcsr-gate-state/1");
    let mask_share = num(&mask, &["objective", "validation", "mask", "on_share"]);
    let share = num(&j, &["compose", "chosen", "validation", "regime", "states", "on", "share"]);
    assert_eq!(share, mask_share, "compose and --pnl-mask give the same gate states");
    let pure = json(&compose(&p.data, &p.spec, &["--gates", &plain, "--bull", &masked, "--bear", &p.bear, "--top", "2", "--json"]));
    assert_ne!(num(&pure, &["compose", "chosen", "validation", "regime", "states", "on", "share"]), share, "the band and the dwell change the states");
    // walkforward --from-run prices the stateful composite on its test rows.
    let test_rows = num(&j, &["test_rows"]) as i64;
    let k_test = STAMPS as i64 - test_rows / SYMBOLS.len() as i64;
    let first_val = num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64;
    let w = format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64);
    let priced = |run: &str| {
        let wf = rcsr(&["walkforward", "--data", &p.data, "--target", "y", "--from-run", run, "--time", "t", "--group-by", "s", "--window", &w, "--json"]);
        assert_eq!(code(&wf), 0, "{run}: {}", stderr(&wf));
        let r = Json::parse(String::from_utf8_lossy(&wf.stdout).trim()).unwrap();
        let m = at(&arr(&r, &["windows"])[0], &["metrics"]).clone();
        (m, r)
    };
    let (m, r) = priced(&out);
    let net = num(&m, &["mean_return"]) * num(&m, &["periods"]) * SYMBOLS.len() as f64;
    let want = num(&j, &["objective", "test", "net_pnl"]);
    assert!((net - want).abs() <= 1e-12, "walkforward {net} vs objective.test {want}");
    assert_eq!(s(&r, &["gate", "source"]), "objective.finalist.switch");
    let wf = rcsr(&["walkforward", "--data", &p.data, "--target", "y", "--front", &out, "--time", "t", "--group-by", "s", "--window", &w, "--validation-frac", "0.3", "--json"]);
    assert_eq!(code(&wf), 0, "--front: {}", stderr(&wf));
    // A composed run cannot be a pool: nested switches exit 2.
    let again = compose(&p.data, &p.spec, &["--gates", &plain, "--bull", &out, "--bear", &p.bear, "--json"]);
    assert_eq!(code(&again), 2, "{}", stderr(&again));
    assert!(stderr(&again).contains("nested switches"), "{}", stderr(&again));
}

#[test]
fn compose_counts_every_composite() {
    let bull = ["x1", "x2", "x1 + x2", "x1 - x2", "0 - x2"];
    let bear = ["0 - x1", "x2", "x2 - x1", "0 - x1 - x2", "x1 * x2"];
    let p = planted(&bull, &bear);
    let aw = pool_run(&p.dir, "aw.json", &p.data, &p.spec, &[], &["x1", "x2", "0 - x1", "x1 + x2", "x2 - x1"]);
    let gates = [("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1), ("x2", "010", "x2", 0.0, 1), ("g2", "111", "g - 0.1", 0.0, 1)];
    let menu = write_menu(&p.dir, "menu.json", &gates, None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let j = json(&compose(&p.data, &p.spec, &["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--all-weather", &aw, "--json", "--out", &out]));
    let n = 4 * (5 * 5 + 5 + 5) + 5;
    assert_eq!(n, 145);
    assert_eq!(num(&j, &["search", "candidates_evaluated"]), n as f64);
    assert_eq!(num(&j, &["compose", "candidates_evaluated"]), n as f64);
    assert_eq!(num(&j, &["compose", "count", "composites"]), n as f64);
    let list = arr(&j, &["compose", "composites"]);
    assert_eq!(list.len(), n);
    let mut ranks: Vec<usize> = list.iter().map(|c| num(c, &["rank"]) as usize).collect();
    ranks.sort_unstable();
    assert_eq!(ranks, (1..=n).collect::<Vec<_>>());
    for c in list {
        assert!(num(c, &["bits"]).is_finite() && num(c, &["bits"]) > 0.0, "{}", s(c, &["expression"]));
    }
    // The composite code: C0 = 1 + (1 + tree), C1 = 2 + 3 + 1 + (1 + tree), C2 = 3 + 3 + (1 + tree) + (1 + tree).
    let c0 = num(composite(&j, 0), &["bits"]);
    let first_gate = 5;
    let c1_on = num(composite(&j, first_gate), &["bits"]);
    let c1_off = num(composite(&j, first_gate + 5), &["bits"]);
    let c2 = num(composite(&j, first_gate + 10), &["bits"]);
    assert_eq!(s(composite(&j, first_gate + 10), &["type"]), "C2");
    assert!((c1_on - (c0 + 5.0)).abs() <= 1e-9, "C1 {c1_on}, C0 {c0}");
    assert!((c2 - (c1_on + c1_off - 6.0)).abs() <= 1e-9, "C2 {c2}, C1 {c1_on} {c1_off}");
    assert_eq!(at(&j, &["compose", "code", "sha256"]).as_str().map(str::len), Some(64));
    // The count is the trial count a walk-forward of the run uses on rows it selected on, and 1 out of sample.
    let first_val = num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64;
    let last_val = num(&arr(&j, &["split", "validation_time"])[1], &[]) as i64;
    let trials = |w: &str, extra: &[&str]| {
        let mut args = vec!["walkforward", "--data", &p.data, "--target", "y", "--from-run", &out, "--time", "t", "--group-by", "s", "--window", w, "--json", "--bootstrap", "100"];
        args.extend_from_slice(extra);
        let o = rcsr(&args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        let r = Json::parse(String::from_utf8_lossy(&o.stdout).trim()).unwrap();
        let m = at(&r, &["inference", "multiple_testing"]).clone();
        (num(&m, &["trials"]), s(&m, &["n_trials_source"]).to_string())
    };
    let inside = format!("v,{},{},{},{}", T0, first_val - STEP, first_val, last_val + 1);
    assert_eq!(trials(&inside, &["--allow-in-sample-windows"]), (n as f64, "run.candidates_evaluated".to_string()));
    let test_rows = num(&j, &["test_rows"]) as i64;
    let k_test = STAMPS as i64 - test_rows / 3;
    let outside = format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64);
    assert_eq!(trials(&outside, &[]), (1.0, "out_of_sample_book".to_string()));
}

#[test]
fn compose_reports_switch_cost() {
    let p = planted(&["x1"], &["0 - x1"]);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let j = json(&compose(&p.data, &p.spec, &["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--json"]));
    assert_eq!(s(&j, &["objective", "finalist", "expression_exact"]), "if(greater(g, 0), x1, 0 - x1)");
    let k0 = (num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64 - T0) / STEP;
    let k1 = (num(&arr(&j, &["split", "validation_time"])[1], &[]) as i64 - T0) / STEP;
    // Hand values: the state flips where the sign of g changes; the cost of a
    // flip period is the position changes of the three symbols at that stamp.
    let pos = |k: i64, j: usize| -> f64 {
        let v = if regime(k) > 0.0 { x1(k, j) } else { 0.0 - x1(k, j) };
        if v > 0.0 { 1.0 } else { 0.0 }
    };
    let mut switches = 0usize;
    let mut cost = 0.0f64;
    let mut total = 0.0f64;
    for j in 0..SYMBOLS.len() {
        let mut prev = 0.0;
        for k in k0..=k1 {
            let p = pos(k, j);
            let flip = k > k0 && (regime(k) > 0.0) != (regime(k - 1) > 0.0);
            if flip {
                cost += SIDE * (p - prev).abs();
                if j == 0 {
                    switches += 1;
                }
            }
            total += SIDE * (p - prev).abs();
            prev = p;
        }
        total += SIDE * prev;
    }
    assert!(switches >= 2, "the validation rows hold {switches} switches");
    let r = at(&j, &["compose", "chosen", "validation", "regime"]);
    assert_eq!(num(r, &["switches", "switches"]), switches as f64);
    assert_eq!(num(r, &["switches", "periods"]), (k1 - k0 + 1) as f64);
    assert!((num(r, &["switch_cost"]) - cost).abs() <= 1e-15, "switch cost {} vs {cost}", num(r, &["switch_cost"]));
    assert!((num(r, &["total_cost"]) - total).abs() <= 1e-12, "total cost {} vs {total}", num(r, &["total_cost"]));
    assert!((num(r, &["switch_cost_share"]) - cost / total).abs() <= 1e-12);
    let per_year = switches as f64 * num(&j, &["objective", "validation", "periods_per_year"]) / (k1 - k0) as f64;
    assert!((num(r, &["switches", "per_year"]) - per_year).abs() <= 1e-9 * per_year, "{} vs {per_year}", num(r, &["switches", "per_year"]));
}

#[test]
fn compose_reads_no_test_rows() {
    let p = planted(&BULL, &BEAR);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1), ("h", "001", "h", 0.0, 1)], Some(8));
    let args = ["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--top", "3", "--json"];
    let base = json(&compose(&p.data, &p.spec, &args));
    let last_val = (num(&arr(&base, &["split", "validation_time"])[1], &[]) as i64 - T0) / STEP;
    let poisoned = write_csv(&p.dir, "poisoned.csv", |k| {
        (k > last_val).then(|| (-regime(k) * 3.0, 0.03 * (frac(k as f64 * 0.41) - 0.5), frac(k as f64 * 0.77) - 0.5))
    });
    let other = json(&compose(&poisoned, &p.spec, &args));
    for path in [&["compose", "composites"][..], &["compose", "selection"], &["compose", "members"], &["objective", "validation"], &["objective", "finalist"]] {
        assert_eq!(at(&base, path), at(&other, path), "{path:?} changed with the test rows");
    }
    assert_eq!(at(&base, &["compose", "chosen", "validation"]), at(&other, &["compose", "chosen", "validation"]));
    assert_ne!(num(&base, &["objective", "test", "net_pnl"]), num(&other, &["objective", "test", "net_pnl"]), "the test rows did change");
    assert!(num(&base, &["compose", "pruned"]) < num(&base, &["compose", "candidates_evaluated"]));
}

#[test]
fn compose_refuses_mismatched_runs() {
    let p = planted(&BULL, &BEAR);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let refused = |data: &str, extra: &[&str], want: i32, needle: &str| {
        let o = compose(data, &p.spec, extra);
        assert_eq!(code(&o), want, "{extra:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{extra:?}: want '{needle}' in {}", stderr(&o));
    };
    // A bear run of another split, another cost, or a split that differs from this invocation.
    let other_split = {
        let mut args = vec!["--data", p.data.as_str(), "--target-spec", p.spec.as_str()];
        let mut flags = RUN_FLAGS.to_vec();
        let at = flags.iter().position(|f| *f == "--test-frac").unwrap();
        flags[at + 1] = "0.25";
        args.extend(flags);
        args.extend(["--fitness", "pnl", "--cpu", "--generations", "2", "--pop", "16", "--json"]);
        let o = rcsr(&args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        let path = p.dir.join("split.json");
        std::fs::write(&path, String::from_utf8_lossy(&o.stdout).as_bytes()).unwrap();
        path.to_string_lossy().into_owned()
    };
    refused(&p.data, &["--gates", &menu, "--bull", &p.bull, "--bear", &other_split], 2, "split.");
    let cost = pool_run(&p.dir, "cost.json", &p.data, &p.spec, &["--cost-bps", "7"], &BEAR);
    refused(&p.data, &["--gates", &menu, "--bull", &p.bull, "--bear", &cost], 2, "cost");
    refused(&p.data, &["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--validation-frac", "0.25"], 2, "split.");
    refused(&p.data, &["--gates", &menu, "--bull", &p.bull, "--bear", &p.bear, "--cost-bps", "7"], 2, "cost");
    // Gates that read a label, a missing column or a bad menu.
    for (expr, want, needle) in [("lbl_h24a_y", 5, "D13"), ("y", 5, "--gates"), ("nope", 3, "nope")] {
        let m = write_menu(&p.dir, "bad.json", &[("bad", "000", expr, 0.0, 1)], None);
        refused(&p.data, &["--gates", &m, "--bull", &p.bull, "--bear", &p.bear], want, needle);
    }
    let dup = write_menu(&p.dir, "dup.json", &[("a", "000", "g", 0.0, 1), ("b", "000", "h", 0.0, 1)], None);
    refused(&p.data, &["--gates", &dup, "--bull", &p.bull], 2, "share codeword");
    refused(&p.data, &["--gates", "/nonexistent/menu.json", "--bull", &p.bull], 3, "--gates");
    refused(&p.data, &["--bull", &p.bull], 2, "--gates <menu.json> is required");
    refused(&p.data, &["--gates", &menu], 2, "--bull <run.json> or --bear <run.json> is required");
    for flag in [&["--threshold", "0.1"][..], &["--band", "0.1"], &["--pnl-mask", "g"], &["--gpu"], &["--select-rule", "occam"], &["--fitness", "metric"], &["--top", "0"]] {
        let mut extra = vec!["--gates", menu.as_str(), "--bull", p.bull.as_str()];
        extra.extend_from_slice(flag);
        refused(&p.data, &extra, 2, flag[0]);
    }
}

#[test]
fn compose_help_lists_its_options() {
    let o = rcsr(&["compose", "--help"]);
    assert_eq!(code(&o), 0);
    let help = String::from_utf8_lossy(&o.stdout);
    let doc = rcsr::schema::json();
    let start = doc.find("\"compose\":{").expect("schema documents rcsr compose");
    let section = &doc[start..];
    let section = &section[..section.find("]}").unwrap()];
    let mut names = 0;
    for part in section.split("\"name\":\"--").skip(1) {
        let name = format!("--{}", part.split('"').next().unwrap());
        assert!(help.contains(&name), "rcsr compose --help is missing {name}");
        names += 1;
    }
    assert_eq!(names, 7);
    let top = rcsr(&["--help"]);
    assert!(String::from_utf8_lossy(&top.stdout).contains("rcsr compose"));
}

/// `j` without the fields `drop` (top-level keys, or `parent.key`).
fn without(j: &Json, drop: &[&str]) -> Json {
    let Json::Obj(fields) = j else { panic!("an object") };
    let kept = fields
        .iter()
        .filter(|(k, _)| !drop.contains(&k.as_str()))
        .map(|(k, v)| {
            let inner: Vec<&str> = drop.iter().filter_map(|d| d.strip_prefix(&format!("{k}."))).collect();
            (k.clone(), if inner.is_empty() { v.clone() } else { without(v, &inner) })
        })
        .collect();
    Json::Obj(kept)
}

/// #1237: a walk-forward of a composed run states the composite (D19) and nothing else
/// changes: the same run.json without its compose block gives the same report without the
/// marker, and a plain run (no compose block, no switch) gives the report of the build
/// before the marker plus `selection.run_sha256` (#1238).
#[test]
fn from_run_of_compose_marks_composite() {
    let p = planted(&BULL, &BEAR);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let out = p.dir.join("composed.json").to_string_lossy().into_owned();
    let j = json(&compose(&p.data, &p.spec, &["--gates", &menu, "--bull", &p.bull, "--top", "3", "--json", "--out", &out]));
    assert_eq!(s(&j, &["compose", "chosen", "type"]), "C1");
    let write = |name: &str, doc: &Json| -> String {
        let path = p.dir.join(name).to_string_lossy().into_owned();
        std::fs::write(&path, format!("{}\n", rcsr::cpcv::to_json(doc))).unwrap();
        path
    };
    let unmarked = write("unmarked.json", &without(&j, &["compose"]));
    let plain = write("plain.json", &without(&j, &["compose", "objective.finalist.switch"]));
    let test_rows = num(&j, &["test_rows"]) as i64;
    let k_test = STAMPS as i64 - test_rows / SYMBOLS.len() as i64;
    let first_val = num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64;
    let w = format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64);
    let report = |run: &str, name: &str| -> Json {
        let path = p.dir.join(name).to_string_lossy().into_owned();
        let o = rcsr(&["walkforward", "--data", &p.data, "--target", "y", "--from-run", run, "--time", "t", "--group-by", "s", "--window", &w, "--out", &path]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        Json::parse(std::fs::read_to_string(&path).unwrap().trim()).unwrap()
    };
    let composed = report(&out, "wf_composed.json");
    let m = at(&composed, &["selection", "composite"]);
    assert_eq!(s(m, &["source"]), "rcsr compose");
    assert_eq!(s(m, &["type"]), "C1");
    assert_eq!(s(m, &["form"]), s(&j, &["compose", "chosen", "form"]));
    assert_eq!(s(m, &["gate"]), "g");
    // The switch prices both runs; the marker is the only difference.
    let switch = report(&unmarked, "wf_unmarked.json");
    assert_eq!(s(&switch, &["gate", "source"]), "objective.finalist.switch");
    assert!(at(&switch, &["selection"]).get("composite").is_none());
    // `ledger` holds wall-clock timings; the two run files differ, so do their names and hashes.
    assert_eq!(
        without(&composed, &["selection.composite", "selection.run", "selection.run_sha256", "protocol", "ledger"]),
        without(&switch, &["selection.run", "selection.run_sha256", "protocol", "ledger"]),
        "the marker changes nothing else"
    );
    // A plain run: no marker, no gate, the bytes of the base build plus the run file hash.
    let plain_report = report(&plain, "wf_plain.json");
    assert!(plain_report.get("gate").is_none() && at(&plain_report, &["selection"]).get("composite").is_none());
    let sha = |b: &[u8]| rcsr::manifest::sha256_hex(b);
    assert_eq!(s(&plain_report, &["selection", "run_sha256"]), sha(&std::fs::read(&plain).unwrap()));
    let text = std::fs::read_to_string(p.dir.join("wf_plain.json")).unwrap();
    let hashed = rcsr::verdict::hashed_report_text(&text).unwrap();
    assert_eq!(s(&plain_report, &["protocol", "report_sha256"]), sha(hashed.as_bytes()));
    let before = rcsr::verdict::remove_member(&hashed, &["selection", "run_sha256"]).unwrap();
    assert_eq!(sha(before.as_bytes()), PINNED_PLAIN_FROM_RUN);
}

const PINNED_PLAIN_FROM_RUN: &str = "22e9d27e4d0f59c0e35b4597d6265b71bf343a8cb8a1b0f01309b9437079ebe2";

#[test]
fn compose_refuses_masked_sd_band_run() {
    let p = planted(&BULL, &BEAR);
    let masked = pool_run(&p.dir, "masked-sd.json", &p.data, &p.spec, &["--pnl-mask", "g", "--band", "0.5", "--band-units", "sd"], &BULL);
    let menu = write_menu(&p.dir, "menu.json", &[("g", "000", "g", 0.0, 1)], None);
    let o = compose(&p.data, &p.spec, &["--gates", &menu, "--bull", &masked]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("priced a band or a minimum hold"), "{}", stderr(&o));
}

// ---- #1268: compose and the pools read one training state ----

/// A gate positive for 72 stamps and negative for the next 24, the same for every symbol.
fn uneven_regime(k: i64) -> f64 {
    regime(k).abs() * if k % 96 < 72 { 1.0 } else { -1.0 }
}

/// The planted panel on `STAMPS * stride` stamps with the gate `uneven_regime` and the label `y` on every
/// `stride`-th stamp only, so `--fill drop` keeps one stamp in `stride` while every stamp stays a gate row.
fn write_strided_csv(dir: &Path, name: &str, stride: i64) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y,lbl_h24a_y\n");
    for k in 0..STAMPS as i64 * stride {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let g = uneven_regime(k);
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1(k, j) + 0.0005 * noise;
            let y = if k % stride == 0 { r.to_string() } else { String::new() };
            text.push_str(&format!("{},{},{},{},{},{},{},{},{}\n", T0 + STEP * k, sym, x1(k, j), x2, g, h, r, y, r));
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// A short real pool search on `data` with its own front.
fn searched_pool(dir: &Path, name: &str, data: &str, spec: &str, extra: &[&str]) -> (String, Json) {
    let mut args = vec!["--data", data, "--target-spec", spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--cpu", "--seed", "3", "--generations", "2", "--pop", "16", "--json"]);
    args.extend_from_slice(extra);
    let o = rcsr(&args);
    let j = json(&o);
    let path = dir.join(name);
    std::fs::write(&path, &o.stdout).unwrap();
    (path.to_string_lossy().into_owned(), j)
}

/// The training net of a pool's top member (the lowest archive rank), on its masked book.
fn top_train_net(run: &Json) -> f64 {
    let front = arr(run, &["pareto"]);
    let top = front.iter().min_by_key(|m| num(m, &["archive_rank"]) as usize).expect("a front member");
    num(top, &["fitness", "train_net"])
}

#[test]
fn compose_training_states_equal_the_pool_masks() {
    // A label on every 4th stamp and dwell 24: the state of every stamp follows each 24-stamp
    // negative run of the gate, while 24 labelled rows would hold a state for 96 stamps.
    const STRIDE: i64 = 4;
    const DWELL: usize = 24;
    let (_scratch, dir) = scratch();
    let data = write_strided_csv(&dir, "strided.csv", STRIDE);
    let spec = write_spec(&dir);
    let dwell = DWELL.to_string();
    let (bull, bull_run) = searched_pool(&dir, "bull.json", &data, &spec, &["--fill", "drop", "--pnl-mask", "g", "--pnl-mask-dwell", &dwell]);
    let (bear, bear_run) = searched_pool(&dir, "bear.json", &data, &spec, &["--fill", "drop", "--pnl-mask", "0 - g", "--pnl-mask-dwell", &dwell]);
    // Each pool's training mask is the evaluation state of its gate: the kernel over every stamp.
    let stamps = STAMPS * STRIDE as usize;
    let gate: Vec<f64> = (0..stamps as i64).map(uneven_regime).collect();
    let state = rcsr::regime_gate::hysteresis_states(&gate, &vec![0; stamps], 0.0, DWELL);
    let range = arr(&bull_run, &["split", "train_time"]);
    let (lo, hi) = (num(&range[0], &[]), num(&range[1], &[]));
    let train: Vec<usize> = (0..stamps).filter(|&k| k as i64 % STRIDE == 0 && (lo..=hi).contains(&((T0 + STEP * k as i64) as f64))).collect();
    let rows = (SYMBOLS.len() * train.len()) as f64;
    let off = (SYMBOLS.len() * train.iter().filter(|&&k| !state[k]).count()) as f64;
    assert!(off > 0.0 && off < rows, "{off} of {rows}");
    assert_eq!(num(&bull_run, &["search_fitness", "mask", "off_rows"]), off, "bull training mask");
    assert_eq!(num(&bear_run, &["search_fitness", "mask", "off_rows"]), rows - off, "bear training mask");
    // The training seam of the composite reads the same states as the pools' masked books.
    let menu = write_menu(&dir, "menu.json", &[("g", "000", "g", 0.0, DWELL)], None);
    let j = json(&compose(&data, &spec, &["--fill", "drop", "--gates", &menu, "--bull", &bull, "--bear", &bear, "--top", "1", "--json"]));
    let chosen = composite(&j, num(&j, &["compose", "chosen", "id"]) as usize);
    let seam = at(&j, &["compose", "chosen", "train_seam"]);
    let mut compared = 0;
    for (side, masked, run) in [("on", "masked_on", &bull_run), ("off", "masked_off", &bear_run)] {
        if at(chosen, &[side]).as_f64().is_some() {
            let (a, b) = (num(seam, &[masked]), top_train_net(run));
            assert!((a - b).abs() <= 1e-12, "{masked} {a} vs the pool's training net {b}");
            compared += 1;
        }
    }
    assert!(compared > 0, "the chosen composite holds a branch: {:?}", chosen);
}
