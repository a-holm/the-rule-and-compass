//! #1182: `rcsr --gate-search`, the search of a regime gate between two frozen
//! branches, on a planted panel.

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

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// The planted gate of stamp `k`: positive in alternate 40-stamp blocks, the
/// same for every symbol, so the training rows (the first 240 stamps) hold as
/// many positive as negative blocks.
fn regime(k: i64) -> f64 {
    let m = 0.5 + 0.4 * frac(k as f64 * 0.324_717_957_244_746);
    if (k / 40) % 2 == 0 {
        m
    } else {
        -m
    }
}

fn x1(k: i64, j: usize) -> f64 {
    frac(((k + 40) as f64 * 3.0 + j as f64) * 0.618_033_988_749_894_9) - 0.5
}

/// One row of the panel: `(stamp, symbol index, x1, x2, g, h, r)`.
type Row = (i64, usize, f64, f64, f64, f64, f64);

/// Three symbols on a 900 s grid. `x1` pays while `g` is positive and `0 - x1`
/// while it is negative; `h` and `x2` are noise; `lbl_h24a_y` is a label-family
/// column no gate may read. `alter` rewrites the gate, the return and `x1` of a
/// row from its stamp.
fn panel(alter: impl Fn(i64) -> Option<(f64, f64, f64)>) -> Vec<Row> {
    let mut rows = Vec::new();
    for k in 0..STAMPS as i64 {
        for j in 0..SYMBOLS.len() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let g = regime(k);
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1(k, j) + 0.0005 * noise;
            let (g, r, x) = alter(k).unwrap_or((g, r, x1(k, j)));
            rows.push((k, j, x, x2, g, h, r));
        }
    }
    rows
}

fn write_csv(dir: &Path, name: &str, rows: &[Row]) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y,lbl_h24a_y\n");
    for &(k, j, x, x2, g, h, r) in rows {
        text.push_str(&format!("{},{},{},{},{},{},{},{},{}\n", T0 + STEP * k, SYMBOLS[j], x, x2, g, h, r, r, r));
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

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

/// The flags every run here shares: data rows, split, objective and cost.
const RUN_FLAGS: [&str; 18] = [
    "--target", "y", "--time", "t", "--group-by", "s", "--split", "temporal", "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3",
    "--test-frac", "0.2", "--no-dev-log", "--quiet",
];

/// A short search's gate-search flags.
const SHORT: [&str; 12] = ["--seed", "3", "--generations", "20", "--pop", "24", "--islands", "2", "--max-complexity", "5", "--ops", "add,sub,mul"];

struct Planted {
    _scratch: Scratch,
    dir: PathBuf,
    rows: Vec<Row>,
    data: String,
    spec: String,
}

fn planted() -> Planted {
    let scratch = Scratch::new("gate-search");
    let dir = scratch.path().to_path_buf();
    let rows = panel(|_| None);
    let data = write_csv(&dir, "d.csv", &rows);
    let spec = write_spec(&dir);
    Planted { _scratch: scratch, dir, rows, data, spec }
}

fn gate_search(p: &Planted, data: &str, vars: &str, extra: &[&str]) -> Output {
    let mut args = vec!["--data", data, "--target-spec", p.spec.as_str(), "--vars", vars];
    args.extend_from_slice(&RUN_FLAGS);
    args.push("--gate-search");
    args.extend_from_slice(extra);
    rcsr(&args)
}

/// A branch run: a short real search on the panel (its split, target spec and
/// cost blocks are the engine's), with its finalist replaced by `expr`.
fn branch_run(p: &Planted, name: &str, extra: &[&str], expr: &str) -> String {
    let mut args = vec!["--data", p.data.as_str(), "--target-spec", p.spec.as_str(), "--vars", "x1,x2"];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--cpu", "--seed", "3", "--generations", "2", "--pop", "16", "--json"]);
    args.extend_from_slice(extra);
    let o = rcsr(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let body = String::from_utf8_lossy(&o.stdout).trim().to_string();
    let at = body.find("\"finalist\":{").expect("an objective finalist");
    let key = "\"expression_exact\":\"";
    let start = at + body[at..].find(key).expect("finalist expression_exact") + key.len();
    let end = start + body[start..].find('"').unwrap();
    let path = p.dir.join(name);
    std::fs::write(&path, format!("{}{}{}\n", &body[..start], expr, &body[end..])).unwrap();
    path.to_string_lossy().into_owned()
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

/// The states of a run's switch gate on every panel row, by the shared kernel.
fn switch_states(rows: &[Row], sw: &Json) -> Vec<bool> {
    let gate = rcsr::regime_gate::GateSpec::parse(s(sw, &["gate", "expr"]), num(sw, &["gate", "band"]), num(sw, &["gate", "dwell"]) as usize).unwrap();
    let value = |name: &str, r: &Row| match name {
        "x1" => r.2,
        "x2" => r.3,
        "g" => r.4,
        "h" => r.5,
        other => panic!("gate column {other}"),
    };
    let columns: Vec<Vec<f64>> = gate.columns.iter().map(|c| rows.iter().map(|r| value(c, r)).collect()).collect();
    let group: Vec<u64> = rows.iter().map(|r| r.1 as u64).collect();
    let time: Vec<f64> = rows.iter().map(|r| r.0 as f64).collect();
    rcsr::regime_gate::gate_states(&gate, &group, &time, &columns).unwrap()
}

fn net_of(report: &Json) -> f64 {
    let m = at(&arr(report, &["windows"])[0], &["metrics"]);
    num(m, &["mean_return"]) * num(m, &["periods"]) * SYMBOLS.len() as f64
}

#[test]
fn gate_search_recovers_planted_gate() {
    let p = planted();
    let o = gate_search(
        &p,
        &p.data,
        "g,h,x2",
        &["--branch-on", "x1", "--branch-off", "0 - x1", "--seed", "3", "--generations", "100", "--patience", "0", "--pop", "32", "--islands", "2", "--max-complexity", "5", "--ops", "add,sub,mul", "--json"],
    );
    let j = json(&o);
    assert_eq!(s(&j, &["gate_search", "schema"]), "rcsr-gate-search/1");
    assert_eq!(num(&j, &["gate_search", "generations"]), 100.0);
    assert_eq!(s(&j, &["compose", "chosen", "type"]), "C3");
    assert_eq!(s(&j, &["compose", "chosen", "form"]), "if(gate, f_on, f_off)");
    let sw = at(&j, &["objective", "finalist", "switch"]);
    assert_eq!(s(sw, &["gate", "state_rule"]), "rcsr-gate-state/1");
    assert_eq!((s(sw, &["on"]), s(sw, &["off"])), ("x1", "0 - x1"));
    let states = switch_states(&p.rows, sw);
    let agree = p.rows.iter().zip(&states).filter(|(r, &on)| on == (regime(r.0) > 0.0)).count();
    let share = agree as f64 / p.rows.len() as f64;
    assert!(share >= 0.95, "the gate {} agrees with the planted states on {share} of the rows", s(sw, &["gate", "expr"]));
}

#[test]
fn gate_search_fitness_matches_compose() {
    let p = planted();
    // One variable and complexity 1: the front holds the gate `g` alone.
    let mut args = vec!["--branch-on", "x1", "--branch-off", "0 - x1", "--gate-band", "0.02", "--gate-dwell", "3", "--max-complexity", "1", "--json"];
    args.extend_from_slice(&SHORT[..8]);
    let j = json(&gate_search(&p, &p.data, "g", &args));
    let gates = arr(&j, &["gate_search", "gates"]);
    assert_eq!(gates.len(), 1, "the front of a one-variable complexity-1 search is one gate");
    let gate = &gates[0];
    assert!(matches!(at(gate, &["constant"]), Json::Null), "the gate g is not constant");
    let sw = at(&j, &["objective", "finalist", "switch"]);
    let expr = s(sw, &["gate", "expr"]).to_string();
    // The same gate and pair under rcsr compose.
    let menu = p.dir.join("menu.json");
    std::fs::write(&menu, format!("{{\"schema\":\"rcsr-gate-menu/1\",\"switch_budget\":null,\"gates\":[{{\"name\":\"g\",\"codeword\":\"000\",\"expr\":\"{expr}\",\"band\":0.02,\"dwell\":3}}]}}")).unwrap();
    let pool = |name: &str, member: &str| -> String {
        let run = branch_run(&p, name, &[], member);
        let body = std::fs::read_to_string(&run).unwrap();
        let cut = body.rfind(",\"pareto\":[").unwrap();
        let fixed = format!("{},\"pareto\":[{{\"archive_rank\":1,\"expression\":\"{member}\",\"expression_exact\":\"{member}\",\"complexity\":3}}]}}\n", &body[..cut]);
        std::fs::write(&run, fixed).unwrap();
        run
    };
    let bull = pool("bull.json", "x1");
    let bear = pool("bear.json", "0 - x1");
    let mut cargs = vec!["compose", "--data", p.data.as_str(), "--target-spec", p.spec.as_str(), "--vars", "x1,g"];
    cargs.extend_from_slice(&RUN_FLAGS);
    let menu = menu.to_string_lossy().into_owned();
    cargs.extend(["--gates", menu.as_str(), "--bull", bull.as_str(), "--bear", bear.as_str(), "--top", "1", "--json"]);
    let c = json(&rcsr(&cargs));
    let c2 = arr(&c, &["compose", "composites"]).iter().find(|m| s(m, &["type"]) == "C2").expect("a C2 composite").clone();
    assert_eq!(s(&c2, &["expression"]), s(&j, &["objective", "finalist", "expression_exact"]));
    let close = |a: f64, b: f64, what: &str| assert!((a - b).abs() <= 1e-12, "{what}: gate search {a} vs compose {b}");
    close(num(&j, &["objective", "validation", "net_pnl"]), num(&c2, &["validation", "net"]), "validation net");
    assert_eq!(s(&c, &["compose", "chosen", "type"]), "C2");
    close(num(&j, &["objective", "test", "net_pnl"]), num(&c, &["objective", "test", "net_pnl"]), "test net");
    // The training fitness of the search is the regret of compose's training composite.
    let train_net = num(&c, &["compose", "chosen", "train_seam", "composite"]);
    close(num(gate, &["train_net"]), train_net, "training net");
    let bound = num(&j, &["gate_search", "bound"]);
    close(num(gate, &["fitness"]), (bound - train_net) / bound, "training fitness");
    close(num(gate, &["search_fitness"]), num(gate, &["fitness"]), "search fitness");
    close(num(gate, &["fitness"]), num(&arr(&j, &["pareto"])[0], &["gate_search", "fitness"]), "pareto fitness");
}

#[test]
fn cash_branch_is_flat() {
    let p = planted();
    let mut args = vec!["--branch-on", "x1", "--branch-off", "cash", "--json"];
    args.extend_from_slice(&SHORT);
    let j = json(&gate_search(&p, &p.data, "g,h", &args));
    assert_eq!(s(&j, &["gate_search", "form"]), "C1");
    assert_eq!(num(&j, &["gate_search", "branches_train", "off", "net"]), 0.0);
    for c in arr(&j, &["compose", "composites"]) {
        assert_eq!((s(c, &["type"]), s(c, &["form"])), ("C3", "if(gate, f, cash)"));
    }
    for m in arr(&j, &["pareto"]) {
        assert_eq!(s(m, &["type"]), "C3");
        assert_eq!(s(m, &["switch", "off"]), "-1.0");
    }
    for slice in ["validation", "test"] {
        let off = at(&j, &["compose", "chosen", slice, "regime", "states", "off"]);
        assert!(num(off, &["rows"]) > 0.0, "{slice}: the gate has off rows");
        assert_eq!(num(off, &["net"]), 0.0, "{slice}: the book is flat in the off state");
        assert_eq!(num(off, &["exposure"]), 0.0, "{slice}: no long row in the off state");
    }
}

#[test]
fn gate_search_refuses_mismatched_branches() {
    let p = planted();
    let refused = |vars: &str, extra: &[&str], want: i32, needle: &str| {
        let o = gate_search(&p, &p.data, vars, extra);
        assert_eq!(code(&o), want, "{extra:?}: {}", stderr(&o));
        assert!(stderr(&o).contains(needle), "{extra:?}: want '{needle}' in {}", stderr(&o));
    };
    let on = branch_run(&p, "on.json", &[], "x1");
    let split = branch_run(&p, "split.json", &["--test-frac", "0.25"], "0 - x1");
    let cost = branch_run(&p, "cost.json", &["--cost-bps", "7"], "0 - x1");
    refused("g", &["--branch-on", &on, "--branch-off", &split], 2, "split.");
    refused("g", &["--branch-on", &on, "--branch-off", &cost], 2, "cost");
    refused("g", &["--branch-on", &on, "--branch-off", "cash", "--validation-frac", "0.25"], 2, "split.");
    refused("g", &["--branch-on", &on, "--branch-off", "cash", "--cost-bps", "7"], 2, "cost");
    refused("g", &["--branch-on", "x1", "--branch-off", "x1"], 2, "nothing to switch");
    refused("g", &["--branch-on", "cash", "--branch-off", "cash"], 2, "nothing to switch");
    refused("g", &["--branch-on", "x1", "--branch-off", "x1 + 0"], 2, "same positions");
    // Gate variables or branches that read a label, the target or a missing column.
    refused("g,lbl_h24a_y", &["--branch-on", "x1", "--branch-off", "cash"], 5, "D13");
    refused("g,y", &["--branch-on", "x1", "--branch-off", "cash"], 5, "D13");
    refused("g", &["--branch-on", "lbl_h24a_y", "--branch-off", "cash"], 5, "--branch-on");
    // Refused flags: the device, interest, masks and pricing changes, checkpoints.
    for flag in [&["--gpu"][..], &["--cash-return", "0.0001"], &["--cash-return", "train-mean"], &["--pnl-mask", "g"], &["--threshold", "0.1"], &["--band", "0.1"], &["--save", "x.ckpt"], &["--fitness", "metric"], &["--select-rule", "occam"], &["--gate-band", "-1"], &["--gate-dwell", "0"]] {
        let mut extra = vec!["--branch-on", "x1", "--branch-off", "cash"];
        extra.extend_from_slice(flag);
        refused("g", &extra, 2, flag[0]);
    }
    refused("g", &["--branch-on", "x1", "--branch-off", "cash", "--ops", "add,complexity"], 2, "causal");
    refused("g", &["--branch-off", "cash"], 2, "--branch-on");
    let o = rcsr(&["--data", &p.data, "--target", "y", "--branch-on", "x1"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("needs --gate-search"), "{}", stderr(&o));
    // A composed run cannot be a branch: nested switches exit 2.
    let mut args = vec!["--branch-on", "x1", "--branch-off", "0 - x1", "--json"];
    args.extend_from_slice(&SHORT);
    let composed = p.dir.join("composed.json");
    let o = gate_search(&p, &p.data, "g", &args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    std::fs::write(&composed, &o.stdout).unwrap();
    refused("g", &["--branch-on", composed.to_str().unwrap(), "--branch-off", "cash"], 2, "nested switches");
}

#[test]
fn gate_search_prices_the_same_in_walkforward() {
    let p = planted();
    let on = branch_run(&p, "on.json", &["--threshold", "0.05"], "x1");
    let out = p.dir.join("gate.json").to_string_lossy().into_owned();
    let mut args = vec!["--branch-on", on.as_str(), "--branch-off", "0 - x1", "--gate-band", "0.02", "--gate-dwell", "3", "--json", "--out", out.as_str()];
    args.extend_from_slice(&SHORT);
    let o = gate_search(&p, &p.data, "g,h", &args);
    let j = json(&o);
    assert_eq!(std::fs::read_to_string(&out).unwrap().trim(), String::from_utf8_lossy(&o.stdout).trim(), "--out holds the printed run.json");
    let sw = at(&j, &["objective", "finalist", "switch"]);
    assert_eq!((num(sw, &["gate", "band"]), num(sw, &["gate", "dwell"])), (0.02, 3.0));
    assert_eq!(s(sw, &["on"]), "(x1) - (0.05)");
    let test_rows = num(&j, &["test_rows"]) as i64;
    let k_test = STAMPS as i64 - test_rows / SYMBOLS.len() as i64;
    let first_val = num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64;
    let last_val = num(&arr(&j, &["split", "validation_time"])[1], &[]) as i64;
    let priced = |window: &str, extra: &[&str]| -> Json {
        let mut a = vec!["walkforward", "--data", p.data.as_str(), "--target", "y", "--from-run", out.as_str(), "--time", "t", "--group-by", "s", "--window", window, "--json"];
        a.extend_from_slice(extra);
        let wf = rcsr(&a);
        assert_eq!(code(&wf), 0, "{window}: {}", stderr(&wf));
        Json::parse(String::from_utf8_lossy(&wf.stdout).trim()).unwrap()
    };
    let test = priced(&format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64), &[]);
    assert_eq!(s(&test, &["gate", "source"]), "objective.finalist.switch");
    let want = num(&j, &["objective", "test", "net_pnl"]);
    assert!((net_of(&test) - want).abs() <= 1e-12, "walkforward {} vs objective.test {want}", net_of(&test));
    let validation = priced(&format!("v,{},{},{},{}", T0, first_val - STEP, first_val, last_val + 1), &["--allow-in-sample-windows"]);
    let want = num(&j, &["objective", "validation", "net_pnl"]);
    assert!((net_of(&validation) - want).abs() <= 1e-12, "walkforward {} vs objective.validation {want}", net_of(&validation));
    // --reselect-run of the gate-search run reruns compose's choice over its composites.
    let mut r = vec!["--data", p.data.as_str(), "--target-spec", p.spec.as_str(), "--vars", "g,h"];
    r.extend_from_slice(&RUN_FLAGS);
    r.extend(["--reselect-run", out.as_str(), "--select-rule", "robust", "--json"]);
    let again = json(&rcsr(&r));
    assert!(at(&again, &["objective", "selection_rule"]).get("reselected_from").is_some());
    assert_eq!(at(&again, &["objective", "finalist", "switch"]), sw);
    assert_eq!(at(&again, &["gate_search"]), at(&j, &["gate_search"]));
    assert_eq!(num(&again, &["search", "candidates_evaluated"]), num(&j, &["search", "candidates_evaluated"]));
    // A run searched under the per-segment training kernel has no train_state_rule and is refused.
    assert_eq!(s(&j, &["gate_search", "train_state_rule"]), "rcsr-gate-state/1");
    let old = p.dir.join("old.json").to_string_lossy().into_owned();
    let body = std::fs::read_to_string(&out).unwrap();
    assert!(body.contains("\"train_state_rule\":\"rcsr-gate-state/1\","));
    std::fs::write(&old, body.replace("\"train_state_rule\":\"rcsr-gate-state/1\",", "")).unwrap();
    let window = format!("t,{},{},{},{}", T0, first_val, T0 + STEP * k_test, T0 + STEP * STAMPS as i64);
    let refused = rcsr(&["walkforward", "--data", p.data.as_str(), "--target", "y", "--from-run", old.as_str(), "--time", "t", "--group-by", "s", "--window", window.as_str(), "--json"]);
    assert_ne!(code(&refused), 0);
    assert!(stderr(&refused).contains("gate_search.train_state_rule") && stderr(&refused).contains("rcsr-gate-state/1"), "{}", stderr(&refused));
}

#[test]
fn gate_search_reads_no_test_rows() {
    let p = planted();
    let mut args = vec!["--branch-on", "x1", "--branch-off", "0 - x1", "--json"];
    args.extend_from_slice(&SHORT);
    let base = json(&gate_search(&p, &p.data, "g,h", &args));
    let last_val = (num(&arr(&base, &["split", "validation_time"])[1], &[]) as i64 - T0) / STEP;
    let rows = panel(|k| (k > last_val).then(|| (-regime(k) * 3.0, 0.03 * (frac(k as f64 * 0.41) - 0.5), frac(k as f64 * 0.77) - 0.5)));
    let poisoned = write_csv(&p.dir, "poisoned.csv", &rows);
    let other = json(&gate_search(&p, &poisoned, "g,h", &args));
    for path in [&["gate_search"][..], &["compose", "composites"], &["compose", "selection"], &["objective", "validation"], &["objective", "finalist"]] {
        assert_eq!(at(&base, path), at(&other, path), "{path:?} changed with the test rows");
    }
    assert_ne!(num(&base, &["objective", "test", "net_pnl"]), num(&other, &["objective", "test", "net_pnl"]), "the test rows did change");
}

#[test]
fn gate_search_is_deterministic_and_keeps_branches_frozen() {
    let p = planted();
    let on = branch_run(&p, "on.json", &["--threshold", "0.05"], "x1");
    let off = branch_run(&p, "off.json", &[], "0 - x1");
    let before: Vec<Vec<u8>> = [&on, &off].iter().map(|f| std::fs::read(f).unwrap()).collect();
    let mut args = vec!["--branch-on", on.as_str(), "--branch-off", off.as_str(), "--json"];
    args.extend_from_slice(&SHORT);
    let first = gate_search(&p, &p.data, "g,h", &args);
    let second = gate_search(&p, &p.data, "g,h", &args);
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    assert_eq!(first.stdout, second.stdout, "the gate search is deterministic");
    let after: Vec<Vec<u8>> = [&on, &off].iter().map(|f| std::fs::read(f).unwrap()).collect();
    assert_eq!(before, after, "the branch runs are not touched");
    let j = json(&first);
    // Every composite holds the branch trees and constants of the input runs, byte for byte.
    assert_eq!(s(&j, &["gate_search", "branch_on", "expression_exact"]), "x1");
    assert_eq!(s(&j, &["gate_search", "branch_on", "branch"]), "(x1) - (0.05)");
    assert_eq!(s(&j, &["gate_search", "branch_off", "branch"]), "0 - x1");
    let members = arr(&j, &["compose", "members"]);
    assert_eq!((s(&members[0], &["expression_exact"]), s(&members[1], &["expression_exact"])), ("x1", "0 - x1"));
    let pareto = arr(&j, &["pareto"]);
    assert!(!pareto.is_empty());
    for m in pareto {
        assert_eq!((s(m, &["switch", "on"]), s(m, &["switch", "off"])), ("(x1) - (0.05)", "0 - x1"), "{}", s(m, &["expression"]));
    }
    // A plain run without --gate-search has no gate-search block.
    let mut plain = vec!["--data", p.data.as_str(), "--target-spec", p.spec.as_str(), "--vars", "g,h"];
    plain.extend_from_slice(&RUN_FLAGS);
    plain.extend(["--fitness", "pnl", "--cpu", "--json"]);
    plain.extend_from_slice(&SHORT);
    let o = rcsr(&plain);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(!String::from_utf8_lossy(&o.stdout).contains("\"gate_search\":"), "a plain run has no gate_search block");
}

#[test]
fn gate_search_pays_full_bits_and_counts_every_candidate() {
    let p = planted();
    let mut args = vec!["--branch-on", "x1", "--branch-off", "0 - x1", "--json"];
    args.extend_from_slice(&SHORT);
    let j = json(&gate_search(&p, &p.data, "g,h", &args));
    let trials = num(&j, &["search", "candidates_evaluated"]);
    let composites = arr(&j, &["compose", "composites"]).len() as f64;
    assert!(trials > composites, "{trials} candidates for {composites} composites");
    assert_eq!(num(&j, &["gate_search", "candidates_evaluated"]), trials);
    assert_eq!(num(&j, &["compose", "candidates_evaluated"]), trials);
    assert_eq!(num(&j, &["objective", "candidates"]), trials);
    // The code: the search alphabet and variables plus what the branches read.
    let ops = rcsr::expr::Op::parse_list_checked("add,sub,mul").unwrap();
    let code = rcsr::occam::CodeSpec::v1(&ops, 3);
    let vars = ["g".to_string(), "h".to_string(), "x1".to_string()];
    let member_bits: Vec<f64> = arr(&j, &["compose", "members"]).iter().map(|m| num(m, &["bits"])).collect();
    for (k, gate) in arr(&j, &["gate_search", "gates"]).iter().enumerate() {
        let tree = rcsr::expr::parse(s(gate, &["search_tree"]), &vars).unwrap();
        let gate_bits = code.description_bits(&tree).map(|b| b.total).unwrap_or_else(|_| code.candidates(&tree).into_iter().max_by_key(|c| c.tier).unwrap().bits.total);
        assert!((num(gate, &["gate_bits"]) - gate_bits).abs() <= 1e-9, "gate {k}: {} vs {gate_bits}", num(gate, &["gate_bits"]));
        // C3: type selector 3, form 1; the searched gate: flag 1, its tree, the 4-bit cut; each branch: flag 1 and its tree.
        let want = 3.0 + 1.0 + (1.0 + gate_bits + 4.0) + (1.0 + member_bits[0]) + (1.0 + member_bits[1]);
        assert!((num(gate, &["bits"]) - want).abs() <= 1e-9, "gate {k}: bits {} vs {want}", num(gate, &["bits"]));
        let composite = arr(&j, &["compose", "composites"]).iter().find(|c| num(c, &["id"]) == k as f64).unwrap();
        assert_eq!(num(composite, &["bits"]), num(gate, &["bits"]));
    }
    // Help names every option.
    let help = String::from_utf8_lossy(&rcsr(&["--help"]).stdout).to_string();
    for name in ["--gate-search", "--branch-on", "--branch-off", "--gate-band", "--gate-dwell"] {
        assert!(help.contains(name), "--help is missing {name}");
    }
}

const HOUR: i64 = 3600;
const STRIDED_DAYS: usize = 160;

/// Hourly rows of symbols a and b with a gate `g` that changes sign every 3 days, a day later for
/// symbol b. The label `y` and the return `r` are finite at hour 0 of each day only (a stride-24
/// label), so `--fill drop` keeps one row per symbol and day.
fn strided_csv(dir: &Path) -> String {
    let mut text = String::from("t,s,x1,x2,g,r,y\n");
    for h in 0..(STRIDED_DAYS * 24) as i64 {
        for (j, sym) in ["a", "b"].iter().enumerate() {
            let f = (2 * h + j as i64) as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let g = if ((h + 24 * j as i64) / 72) % 2 == 0 { 1.0 } else { -1.0 };
            let r = g * 0.01 * x1 + 0.0005 * noise;
            let (r, y) = if h % 24 == 0 { (r.to_string(), (r + 0.05 * x2).to_string()) } else { (String::new(), String::new()) };
            text.push_str(&format!("{},{},{},{},{},{},{}\n", T0 + HOUR * h, sym, x1, x2, g, r, y));
        }
    }
    let path = dir.join("strided.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn daily_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":86400,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("daily.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// A gate search of the one-variable gate `g` on the strided fixture.
fn strided_gate_search(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target-spec", spec, "--vars", "g", "--target", "y", "--time", "t", "--group-by", "s", "--fill", "drop", "--split", "temporal", "--objective", "pnl", "--cost-bps", "5",
        "--validation-frac", "0.3", "--test-frac", "0.2", "--no-dev-log", "--quiet", "--gate-search", "--branch-on", "x1", "--branch-off", "0 - x1", "--gate-dwell", "24", "--max-complexity", "1", "--json",
    ];
    args.extend_from_slice(&SHORT[..8]);
    args.extend_from_slice(extra);
    rcsr(&args)
}

#[test]
fn gate_search_training_states_equal_evaluation_states_on_strided_labels() {
    let scratch = Scratch::new("gate-search-strided");
    let dir = scratch.path().to_path_buf();
    let data = strided_csv(&dir);
    let spec = daily_spec(&dir);
    let j = json(&strided_gate_search(&data, &spec, &[]));
    let gates = arr(&j, &["gate_search", "gates"]);
    assert_eq!(gates.len(), 1);
    let cut = num(&gates[0], &["cut"]);
    let gate = rcsr::regime_gate::GateSpec::parse(&format!("({}) - ({:?})", s(&gates[0], &["expression"]), cut), 0.0, 24).unwrap();
    let text = std::fs::read_to_string(&data).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
    let group: Vec<u64> = rows.iter().map(|r| u64::from(r[1] == "b")).collect();
    let time: Vec<f64> = rows.iter().map(|r| r[0].parse().unwrap()).collect();
    let g: Vec<f64> = rows.iter().map(|r| r[4].parse().unwrap()).collect();
    let (lo, hi) = (num(&arr(&j, &["split", "train_time"])[0], &[]), num(&arr(&j, &["split", "train_time"])[1], &[]));
    let train: Vec<usize> = (0..rows.len()).filter(|&i| !rows[i][6].is_empty() && time[i] >= lo && time[i] <= hi).collect();
    assert_eq!(train.len() as f64, num(&j, &["train_rows"]));
    // The evaluation rule: the kernel over every row of the file, read on the training rows.
    let every = rcsr::regime_gate::gate_states(&gate, &group, &time, &[g.clone()]).unwrap();
    let on = train.iter().filter(|&&i| every[i]).count();
    assert!(on > 0 && on < train.len(), "{on} of {}", train.len());
    // The rule before the change: the kernel over the training rows alone.
    let (tg, tgroup): (Vec<f64>, Vec<u64>) = train.iter().map(|&i| (g[i], group[i])).unzip();
    let alone = rcsr::regime_gate::hysteresis_states(&tg.iter().map(|v| v - cut).collect::<Vec<_>>(), &tgroup, 0.0, 24);
    assert_ne!(alone.iter().filter(|&&o| o).count(), on, "the fixture separates the row sets: 24 training rows are 24 days");
    let share = on as f64 / train.len() as f64;
    assert!((num(&gates[0], &["on_share"]) - share).abs() <= 1e-12, "training on share {} vs the evaluation rule {share}", num(&gates[0], &["on_share"]));
    assert_eq!(s(&j, &["gate_search", "train_state_rule"]), "rcsr-gate-state/1");
}

#[test]
fn searched_gate_training_states_equal_evaluation_states_row_for_row() {
    use rcsr::data::{FillMethod, PrepSpec, SplitMode, SplitSpec, TradingColumns};
    use rcsr::gate_search::{Cut, GateLayer};
    use rcsr::regime_gate::GateSpec;
    use rcsr::trading::{FitnessBook, TradingSpec};
    let scratch = Scratch::new("gate-search-states");
    let data = strided_csv(scratch.path());
    let table = rcsr::data::parse_table(&std::fs::read_to_string(&data).unwrap()).unwrap();
    let cols = TradingColumns { return_column: Some("r".to_string()), log_returns: true, train: true, gate_rows: vec!["g".to_string()], ..Default::default() };
    let spec = SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        group_col: Some(1),
        test_frac: 0.2,
        validation_frac: 0.3,
        trading: Some(cols),
        canonical_train_order: true,
        ..SplitSpec::default()
    };
    let prep = PrepSpec { fill: FillMethod::Drop, ..PrepSpec::default() };
    let p = rcsr::data::prepare_full_with(&table, &["g".to_string()], "y", &spec, &mut rcsr::rng::Mt::new(1), &prep).unwrap();
    let split = p.trading.as_ref().expect("trading split");
    let rows = split.gate_rows.as_ref().expect("gate rows");
    let train = split.train.as_ref().expect("training rows");
    let n = train.ret.len();
    let tspec = TradingSpec { cost_bps: 5.0, ..Default::default() };
    let book = FitnessBook::new(train, &tspec, 86_400.0).expect("a profitable training book");
    let on: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let off: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    // The gate `g` with band 0 and dwell 24 on every gate row.
    let (band, dwell) = (0.0, 24);
    let plain = GateSpec::parse("g", band, dwell).unwrap();
    let g_all = rows.values(&plain).unwrap();
    let layer = GateLayer::new(book, on, off, rows.plan().unwrap(), None, band, dwell, "t").unwrap();
    let choice = layer.choose(&g_all).expect("finite gate values");
    let Cut::Grid { value, .. } = choice.cut else { panic!("a constant gate: {:?}", choice.cut) };
    let gate = GateSpec::parse(&format!("(g) - ({:?})", value), band, dwell).unwrap();
    let evaluation = rows.states(&gate).unwrap().train.expect("training states");
    assert_eq!(choice.states.len(), n);
    assert_eq!(choice.states, evaluation, "the search's training states, row for row");
    assert!(evaluation.iter().any(|&s| s) && evaluation.iter().any(|&s| !s));
    // The fixture separates the row sets: the kernel over the training rows alone differs.
    let train_g: Vec<f64> = rows.plan().unwrap().train_rows().iter().map(|&r| g_all[r] - value).collect();
    let group: Vec<u64> = train.rows.group.clone();
    assert_ne!(rcsr::regime_gate::hysteresis_states(&train_g, &group, band, dwell), evaluation);
}

#[test]
fn gate_search_output_is_identical_at_one_and_four_threads() {
    let scratch = Scratch::new("gate-search-threads");
    let data = strided_csv(scratch.path());
    let spec = daily_spec(scratch.path());
    let one = strided_gate_search(&data, &spec, &["--threads", "1"]);
    let four = strided_gate_search(&data, &spec, &["--threads", "4"]);
    assert_eq!(code(&one), 0, "{}", stderr(&one));
    assert_eq!(code(&four), 0, "{}", stderr(&four));
    // Only the echoed thread count differs.
    let strip = |o: &Output| String::from_utf8_lossy(&o.stdout).replace("--threads 1", "--threads N").replace("--threads 4", "--threads N").replace("\"threads\":1", "\"threads\":N").replace("\"threads\":4", "\"threads\":N");
    assert_eq!(strip(&one), strip(&four));
}

const DAY: i64 = 24 * HOUR;

/// The rows of symbol `sym` of the CSV `data`, written beside it.
fn symbol_csv(data: &str, sym: &str) -> String {
    let text = std::fs::read_to_string(data).unwrap();
    let mut lines = text.lines();
    let mut out = format!("{}\n", lines.next().unwrap());
    for line in lines.filter(|l| l.split(',').nth(1) == Some(sym)) {
        out.push_str(line);
        out.push('\n');
    }
    let path = format!("{data}.{sym}.csv");
    std::fs::write(&path, out).unwrap();
    path
}

/// Window 0 of `walkforward --from-run run` on `data` with `window`.
fn walk_window(data: &str, run: &str, window: &str) -> Json {
    let args = ["walkforward", "--data", data, "--target", "y", "--time", "t", "--group-by", "s", "--from-run", run, "--window", window, "--allow-in-sample-windows", "--json"];
    let report = json(&rcsr(&args));
    assert_eq!(s(&report, &["gate", "source"]), "objective.finalist.switch");
    match at(&report, &["windows"]) {
        Json::Arr(a) => a[0].clone(),
        _ => panic!("windows is not an array"),
    }
}

/// #1279: `walkforward --from-run` of a gate-search run on a stride-24 label prices the run's
/// validation and test books. The two-symbol window gives rows, trades and entries; the net is the
/// sum of the windows of each symbol alone, because a period return is the mean over the rows
/// priced at its stamp and the two symbols' gates differ.
#[test]
fn strided_gate_search_walkforward_reproduces_objective_validation_and_test() {
    let scratch = Scratch::new("gate-search-strided-wf");
    let dir = scratch.path().to_path_buf();
    let data = strided_csv(&dir);
    let spec = daily_spec(&dir);
    let run = dir.join("run.json").to_string_lossy().into_owned();
    let j = json(&strided_gate_search(&data, &spec, &["--out", run.as_str()]));
    let sw = at(&j, &["objective", "finalist", "switch"]);
    assert_eq!(num(sw, &["gate", "dwell"]), 24.0);
    let one = [symbol_csv(&data, "a"), symbol_csv(&data, "b")];
    let vt = arr(&j, &["split", "validation_time"]);
    let (vlo, vhi) = (num(&vt[0], &[]) as i64, num(&vt[1], &[]) as i64);
    // The test rows are the last test_rows / 2 labelled stamps of each symbol.
    let test_start = T0 + DAY * (STRIDED_DAYS as i64 - num(&j, &["test_rows"]) as i64 / 2);
    let test_end = T0 + DAY * STRIDED_DAYS as i64;
    // A training range that ends two days before the block.
    let windows = [
        ("validation", format!("v,{T0},{},{vlo},{}", vlo - 2 * DAY, vhi + 1)),
        ("test", format!("t,{T0},{},{test_start},{test_end}", test_start - 2 * DAY)),
    ];
    for (slot, window) in windows {
        let want = at(&j, &["objective", slot]);
        assert!(num(want, &["trades"]) > 0.0 && num(want, &["entries"]) > 0.0, "{slot}: the book must trade");
        let m = at(&walk_window(&data, &run, &window), &["metrics"]).clone();
        for key in ["rows", "trades", "entries"] {
            assert_eq!(num(&m, &[key]), num(want, &[key]), "{slot} {key}");
        }
        let net: f64 = one
            .iter()
            .map(|d| {
                let m = at(&walk_window(d, &run, &window), &["metrics"]).clone();
                if num(&m, &["rows"]) == 0.0 { 0.0 } else { num(&m, &["mean_return"]) * num(&m, &["periods"]) }
            })
            .sum();
        assert!((net - num(want, &["net_pnl"])).abs() <= 1e-12, "{slot}: walkforward net {net} vs objective {}", num(want, &["net_pnl"]));
    }
}
