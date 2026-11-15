//! #1214: walk-forward, the ladder, `--reselect-run` and `--front` price a
//! composite whose gate has a band or a dwell under one gate-state rule
//! (`rcsr-gate-state/1`). Synthetic fixtures only.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use rcsr::regime_gate;
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

/// 2020-09-13T12:00:00Z, on the hour.
const T0: i64 = 1_599_998_400;
const STEP: i64 = 3600;
const STAMPS: usize = 480;
const SYMBOLS: [&str; 3] = ["a", "b", "c"];
const SIDE: f64 = 0.0005;
const BAND: f64 = 0.02;
const DWELL: usize = 5;

fn scratch(tag: &str) -> (Scratch, PathBuf) {
    let scratch = Scratch::new(tag);
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

fn frac(v: f64) -> f64 {
    v - v.floor()
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

fn json(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    let text = String::from_utf8_lossy(&o.stdout);
    Json::parse(text.trim()).unwrap_or_else(|| panic!("stdout is not json: {text}"))
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

fn arr<'a>(j: &'a Json, path: &[&str]) -> &'a Vec<Json> {
    match at(j, path) {
        Json::Arr(a) => a,
        _ => panic!("{path:?} is not an array"),
    }
}

fn s<'a>(j: &'a Json, path: &[&str]) -> &'a str {
    at(j, path).as_str().unwrap_or_else(|| panic!("{path:?} is not a string"))
}

/// Compact text of a parsed value, for edited run.json copies.
fn text(j: &Json) -> String {
    match j {
        Json::Null => "null".to_string(),
        Json::Bool(b) => b.to_string(),
        Json::Num(v) => rcsr::devlog::num(*v),
        Json::Str(v) => rcsr::devlog::text(v),
        Json::Arr(a) => format!("[{}]", a.iter().map(text).collect::<Vec<_>>().join(",")),
        Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}:{}", rcsr::devlog::text(k), text(v))).collect::<Vec<_>>().join(",")),
    }
}

/// Sets (or with `None` removes) the member at `path` of an object tree.
fn set(j: &mut Json, path: &[&str], value: Option<Json>) {
    let Json::Obj(members) = j else { panic!("not an object at {path:?}") };
    if path.len() == 1 {
        members.retain(|(k, _)| k != path[0]);
        if let Some(v) = value {
            members.push((path[0].to_string(), v));
        }
        return;
    }
    let child = members.iter_mut().find(|(k, _)| k == path[0]).map(|(_, v)| v).unwrap_or_else(|| panic!("missing '{}'", path[0]));
    set(child, &path[1..], value);
}

fn x1(k: i64, j: usize) -> f64 {
    frac(((k + 40) as f64 * 3.0 + j as f64) * 0.618_033_988_749_894_9) - 0.5
}

/// The planted regime of the returns: positive in alternate 48-stamp blocks.
fn g(k: i64) -> f64 {
    let m = 0.5 + 0.4 * frac(k as f64 * 0.324_717_957_244_746);
    if (k / 48) % 2 == 0 {
        m
    } else {
        -m
    }
}

/// The gate `h` (band 0.02, dwell 5). Around each slice start `s` it is off for
/// 6 stamps, on at `s - 2` and off again from `s - 1`, so the state at `s` has
/// held 2 rows of a 5-row dwell and a restart at `s` would take the raw off
/// state. Elsewhere it alternates in 17-stamp blocks per symbol, inside the band
/// on every seventh stamp.
fn h(k: i64, j: usize, starts: &[i64]) -> f64 {
    let sign = |on: bool| if on { 1.0 } else { -1.0 };
    for &s in starts {
        if k == s - 2 {
            return 0.3 + 0.01 * j as f64;
        }
        if (s - 8..s + 12).contains(&k) {
            return -0.3 - 0.01 * j as f64;
        }
    }
    let on = ((k + 5 * j as i64) / 17) % 2 == 0;
    if k % 7 == 3 {
        return 0.01 * sign(!on);
    }
    sign(on) * (0.05 + 0.2 * frac(k as f64 * 0.414_213_562_373_095 + j as f64 * 0.17))
}

#[derive(Clone, Copy)]
struct Cell {
    x1: f64,
    x2: f64,
    g: f64,
    h: f64,
    r: f64,
}

fn cell(k: i64, j: usize, starts: &[i64]) -> Cell {
    let i = (k + 40) as f64 * 3.0 + j as f64;
    let noise = frac(i * 0.324_717_957_244_746) - 0.5;
    let r = if g(k) > 0.0 { 0.01 } else { -0.01 } * x1(k, j) + 0.0005 * noise;
    Cell { x1: x1(k, j), x2: frac(i * 0.754_877_666_246_692_7) - 0.5, g: g(k), h: h(k, j, starts), r }
}

/// Three symbols on an hourly grid; `y` is the target, `r` its return column and
/// `lbl_h24a_y` a label-family column no gate may read.
fn write_csv(dir: &Path, name: &str, row: impl Fn(i64, usize) -> Cell) -> String {
    let mut out = String::from("t,s,x1,x2,g,h,r,y,lbl_h24a_y\n");
    for k in 0..STAMPS as i64 {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let c = row(k, j);
            out.push_str(&format!("{},{},{},{},{},{},{},{},{}\n", T0 + STEP * k, sym, c.x1, c.x2, c.g, c.h, c.r, c.r, c.r));
        }
    }
    let path = dir.join(name);
    std::fs::write(&path, out).unwrap();
    path.to_string_lossy().into_owned()
}

const SPEC_OBJECT: &str = "{\"column\":\"y\",\"name\":\"raw_target\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\
     \"return_units\":\"simple\",\"horizon_seconds\":3600,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\"exit_price\":\"close\",\
     \"offset\":0.0,\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"return\",\"cost_bps\":null,\"cost_in_label\":false,\
     \"construction_columns\":[],\"note\":null}";

fn write_spec(dir: &Path) -> String {
    let path = dir.join("s.json");
    std::fs::write(&path, format!("{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{}]}}", SPEC_OBJECT)).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_menu(dir: &Path, band: f64, dwell: usize) -> String {
    let path = dir.join(format!("menu-{band}-{dwell}.json"));
    std::fs::write(&path, format!("{{\"schema\":\"rcsr-gate-menu/1\",\"switch_budget\":null,\"gates\":[{{\"name\":\"h\",\"codeword\":\"011\",\"expr\":\"h\",\"band\":{band},\"dwell\":{dwell}}}]}}")).unwrap();
    path.to_string_lossy().into_owned()
}

const TEMPORAL: [&str; 6] = ["--split", "temporal", "--validation-frac", "0.3", "--test-frac", "0.2"];

fn base_flags<'a>(split: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec!["--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--objective", "pnl", "--cost-bps", "5", "--no-dev-log", "--quiet"];
    v.extend_from_slice(split);
    v
}

/// A pool run: a short real search (its split, target spec and cost blocks are
/// the engine's), with its front replaced by `exprs` in archive order.
fn pool_run(dir: &Path, name: &str, data: &str, spec: &str, split: &[&str], extra: &[&str], exprs: &[&str]) -> String {
    let mut args = vec!["--data", data, "--target-spec", spec];
    args.extend(base_flags(split));
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

fn compose(data: &str, spec: &str, split: &[&str], extra: &[&str]) -> Output {
    let mut args = vec!["compose", "--data", data, "--target-spec", spec];
    args.extend(base_flags(split));
    args.extend_from_slice(extra);
    rcsr(&args)
}

/// The validation and test starts (stamps) of a temporal run.
fn slice_starts(run: &str) -> (i64, i64) {
    let j = read_json(run);
    let vs = (num(&arr(&j, &["split", "validation_time"])[0], &[]) as i64 - T0) / STEP;
    let ts = STAMPS as i64 - num(&j, &["test_rows"]) as i64 / SYMBOLS.len() as i64;
    (vs, ts)
}

/// A composed run: gate `h` (band 0.02, dwell 5) over bull `x1` and bear `0 - x1`,
/// on data whose gate is inside a running dwell at the validation and test starts.
struct Composed {
    _scratch: Scratch,
    dir: PathBuf,
    data: String,
    spec: String,
    run: String,
    j: Json,
    vs: i64,
    ts: i64,
}

fn composed_with(tag: &str, row: impl Fn(i64, usize, &[i64]) -> Cell) -> Composed {
    let (scratch, dir) = scratch(tag);
    let spec = write_spec(&dir);
    let plain = write_csv(&dir, "plain.csv", |k, j| cell(k, j, &[]));
    let probe = pool_run(&dir, "probe.json", &plain, &spec, &TEMPORAL, &[], &["x1"]);
    let (vs, ts) = slice_starts(&probe);
    let data = write_csv(&dir, "d.csv", |k, j| row(k, j, &[vs, ts]));
    let bull = pool_run(&dir, "bull.json", &data, &spec, &TEMPORAL, &["--pnl-mask", "g"], &["x1"]);
    let bear = pool_run(&dir, "bear.json", &data, &spec, &TEMPORAL, &["--pnl-mask", "0 - g"], &["0 - x1"]);
    let menu = write_menu(&dir, BAND, DWELL);
    let run = dir.join("composed.json").to_string_lossy().into_owned();
    let j = json(&compose(&data, &spec, &TEMPORAL, &["--gates", &menu, "--bull", &bull, "--bear", &bear, "--json", "--out", &run]));
    Composed { _scratch: scratch, dir, data, spec, run, j, vs, ts }
}

fn composed(tag: &str) -> Composed {
    composed_with(tag, cell)
}

fn stamp(k: i64) -> i64 {
    T0 + STEP * k
}

impl Composed {
    fn last_validation(&self) -> i64 {
        num(&arr(&self.j, &["split", "validation_time"])[1], &[]) as i64
    }

    /// A window whose test block is the run's validation rows.
    fn validation_window(&self) -> String {
        format!("v,{},{},{},{}", T0, stamp(self.vs) - STEP, stamp(self.vs), self.last_validation() + 1)
    }

    /// A window whose test block is the run's test rows.
    fn test_window(&self) -> String {
        format!("t,{},{},{},{}", T0, stamp(self.vs), stamp(self.ts), stamp(STAMPS as i64))
    }

    fn walk(&self, run: &str, window: &str, extra: &[&str]) -> Output {
        let mut args = vec!["walkforward", "--data", &self.data, "--target", "y", "--from-run", run, "--time", "t", "--group-by", "s", "--window", window, "--allow-in-sample-windows", "--json"];
        args.extend_from_slice(extra);
        rcsr(&args)
    }
}

/// The net of a window: its period returns are equal-weight means over the three symbols.
fn net_of(window: &Json) -> f64 {
    let m = at(window, &["metrics"]);
    num(m, &["mean_return"]) * num(m, &["periods"]) * SYMBOLS.len() as f64
}

fn first_window(report: &Json) -> &Json {
    &arr(report, &["windows"])[0]
}

#[test]
fn walkforward_prices_a_stateful_composite() {
    let c = composed("sg-price");
    let sw = at(&c.j, &["objective", "finalist", "switch"]);
    assert_eq!((s(sw, &["gate", "expr"]), num(sw, &["gate", "band"]), num(sw, &["gate", "dwell"])), ("h", BAND, DWELL as f64));
    assert_eq!(s(sw, &["gate", "state_rule"]), "rcsr-gate-state/1");
    for (window, slot) in [(c.validation_window(), "validation"), (c.test_window(), "test")] {
        let r = json(&c.walk(&c.run, &window, &[]));
        let w = first_window(&r);
        let want = num(&c.j, &["objective", slot, "net_pnl"]);
        assert!((net_of(w) - want).abs() <= 1e-12, "{slot}: walkforward {} vs objective {want}", net_of(w));
        assert_eq!(num(w, &["metrics", "rows"]), num(&c.j, &["compose", "rows", slot]), "{slot} rows");
        assert_eq!(s(&r, &["gate", "source"]), "objective.finalist.switch");
        assert_eq!((s(&r, &["gate", "expr"]), s(&r, &["gate", "state_rule"])), ("h", "rcsr-gate-state/1"));
        assert_eq!(s(w, &["expression"]), s(&c.j, &["objective", "finalist", "expression_exact"]));
    }
}

#[test]
fn stateful_composite_regime_matches_compose() {
    let c = composed("sg-regime");
    for (window, slot) in [(c.validation_window(), "validation"), (c.test_window(), "test")] {
        let r = json(&c.walk(&c.run, &window, &[]));
        let got = at(first_window(&r), &["gate", "regime"]);
        let want = at(&c.j, &["compose", "chosen", slot, "regime"]);
        for path in [&["rows"][..], &["states", "on", "rows"], &["states", "off", "rows"], &["switches", "switches"], &["switches", "periods"]] {
            assert_eq!(num(got, path), num(want, path), "{slot} {path:?}");
        }
        for path in [&["states", "on", "net"][..], &["states", "off", "net"], &["seam"], &["switch_cost"], &["total_cost"]] {
            assert!((num(got, path) - num(want, path)).abs() <= 1e-12, "{slot} {path:?}: {} vs {}", num(got, path), num(want, path));
        }
        assert!(num(want, &["switches", "switches"]) >= 2.0, "{slot}: the gate flips inside the slice");
        // The window and compose state the same warm-up: every group inside a running dwell.
        let start = at(first_window(&r), &["gate", "start"]);
        assert_eq!(start, at(want, &["start"]), "{slot} warm-up");
        assert_eq!((num(start, &["on_at_start"]), num(start, &["dwell_running_at_start"]), num(start, &["groups"])), (3.0, 3.0, 3.0), "{slot}");
    }
}

/// One-group and two-group hand fixtures: `g` is off, on at stamp `K` alone,
/// off after; symbol `b` stays off. `a` is 1 everywhere, so the book is long
/// wherever the mask or the switch lets it.
const K: i64 = 30;

fn hand_csv(dir: &Path, rows: i64, row: impl Fn(i64, usize) -> (f64, f64, f64, f64)) -> String {
    let mut out = String::from("t,s,a,b,g,r,y\n");
    for k in 0..rows {
        for (j, sym) in ["a", "b"].iter().enumerate() {
            let (a, b, gate, r) = row(k, j);
            out.push_str(&format!("{},{},{},{},{},{},{}\n", stamp(k), sym, a, b, gate, r, r));
        }
    }
    let path = dir.join("hand.csv");
    std::fs::write(&path, out).unwrap();
    path.to_string_lossy().into_owned()
}

fn ret(k: i64) -> f64 {
    0.001 * ((k % 7) + 1) as f64
}

/// A run.json the evaluators read: the finalist `expr` with threshold 0, cost 5
/// bp per side, trained before stamp 10, with an optional mask and switch.
fn hand_run(dir: &Path, name: &str, expr: &str, mask: Option<&str>, switch: Option<&str>) -> String {
    let spec = format!(
        "{{\"threshold\":0.0,\"per_side_bps\":5.0,\"max_position\":1.0,\"cash_return\":0.0,\"min_holding\":1,\"tau\":null,\"funding_col\":null{}}}",
        mask.map_or(String::new(), |m| format!(",\"pnl_mask\":{m}"))
    );
    let finalist = format!("{{\"expression\":{e},\"expression_exact\":{e}{}}}", switch.map_or(String::new(), |s| format!(",\"switch\":{s}")), e = rcsr::devlog::text(expr));
    let body = format!(
        "{{\"tool\":\"rcsr\",\"target\":\"y\",\"variables\":[\"a\",\"b\"],\"split\":{{\"mode\":\"temporal\",\"train_time\":[{},{}],\"required_purge_seconds\":3600}},\
         \"target_spec\":{},\"search\":{{\"candidates_evaluated\":1}},\"objective\":{{\"spec\":{},\"cost\":{{\"source\":\"flag\"}},\"finalist\":{}}},\
         \"pareto\":[{{\"archive_rank\":1,\"expression\":{e},\"expression_exact\":{e},\"complexity\":1}}]}}",
        T0,
        stamp(5),
        SPEC_OBJECT,
        spec,
        finalist,
        e = rcsr::devlog::text(expr)
    );
    let path = dir.join(name);
    std::fs::write(&path, body).unwrap();
    path.to_string_lossy().into_owned()
}

fn hand_walk(data: &str, run: &str, window: &str, extra: &[&str]) -> Output {
    let mut args = vec!["walkforward", "--data", data, "--target", "y", "--from-run", run, "--time", "t", "--group-by", "s", "--window", window, "--json"];
    args.extend_from_slice(extra);
    rcsr(&args)
}

fn dwell_fixture(dir: &Path) -> String {
    hand_csv(dir, 60, |k, j| {
        let on = j == 0 && k == K;
        (1.0, 1.0, if on { 1.0 } else { -1.0 }, ret(k))
    })
}

#[test]
fn window_start_inside_a_dwell_keeps_the_prefix_state() {
    let (_s, dir) = scratch("sg-dwell");
    let data = dwell_fixture(&dir);
    let run = hand_run(&dir, "masked.json", "a", Some("{\"expr\":\"g\",\"band\":0.0,\"dwell\":5}"), None);
    // The test block starts 2 rows into the 5-row dwell that began at K.
    let w = format!("w,{},{},{},{}", stamp(10), stamp(K), stamp(K + 2), stamp(50));
    let r = json(&hand_walk(&data, &run, &w, &[]));
    let m = at(first_window(&r), &["metrics"]);
    // On rows K + 2, K + 3 and K + 4 of symbol a: one on-run, an entry and an exit.
    let hand = ret(K + 2) + ret(K + 3) + ret(K + 4) - 2.0 * SIDE;
    let net = num(m, &["mean_return"]) * num(m, &["periods"]);
    assert!((net - hand).abs() <= 1e-12, "net {net} vs hand {hand}");
    assert_eq!(num(m, &["rows"]), 3.0);
    assert_eq!(s(&r, &["gate", "source"]), "objective.spec.pnl_mask");
}

#[test]
fn report_states_the_gate_warm_up() {
    let (_s, dir) = scratch("sg-warm");
    let data = dwell_fixture(&dir);
    let run = hand_run(&dir, "masked.json", "a", Some("{\"expr\":\"g\",\"band\":0.0,\"dwell\":5}"), None);
    let w = format!("w,{},{},{},{}", stamp(10), stamp(K), stamp(K + 2), stamp(50));
    let r = json(&hand_walk(&data, &run, &w, &[]));
    let start = at(first_window(&r), &["gate", "start"]);
    // Rows 0..K+1 of both symbols, from the first row of the file, not the window's training start.
    assert_eq!(num(start, &["warmup_rows"]), 2.0 * (K + 2) as f64);
    assert_eq!(num(start, &["warmup_first_time"]), T0 as f64);
    assert_eq!(num(start, &["warmup_last_time"]), stamp(K + 1) as f64);
    assert_eq!(num(start, &["groups"]), 2.0);
    assert_eq!(num(start, &["on_at_start"]), 1.0, "symbol a entered on at K");
    assert_eq!(num(start, &["dwell_running_at_start"]), 1.0, "symbol a has held 2 of 5 rows; b has held K + 2");
    assert_eq!(num(&r, &["gate", "gate_rows_read"]), 2.0 * 50.0, "the pass reads every row before the window end");
    for b in arr(&r, &["gate", "barriers"]) {
        assert!(num(b, &["start"]) >= stamp(60) as f64, "a barrier inside the data: {:?}", b);
    }
}

#[test]
fn a_switch_pays_only_the_position_change() {
    let (_s, dir) = scratch("sg-cost");
    // Symbol b has no target, so every period holds symbol a alone. Gate g: on to stamp 17, off from 18 with a
    // one-row blip at 19 the dwell of 2 holds off, on from 24. Branch a is long to 17 and flat
    // after; branch b is long from 18 to 23.
    let data = hand_csv(&dir, 30, |k, j| {
        if j == 1 {
            return (-1.0, -1.0, -1.0, f64::NAN);
        }
        let gate = if k < 18 || k == 19 || k >= 24 { 1.0 } else { -1.0 };
        let a = if k < 18 { 1.0 } else { -1.0 };
        let b = if (18..24).contains(&k) { 1.0 } else { -1.0 };
        (a, b, gate, ret(k))
    });
    let sw = "{\"gate\":{\"schema\":\"rcsr-gate/1\",\"expr\":\"g\",\"columns\":[\"g\"],\"band\":0,\"dwell\":2,\"state_rule\":\"rcsr-gate-state/1\"},\"on\":\"a\",\"off\":\"b\"}";
    let run = hand_run(&dir, "switch.json", "if(greater(g, 0), a, b)", None, Some(sw));
    let w = format!("w,{},{},{},{}", stamp(0), stamp(10), stamp(10), stamp(30));
    let r = json(&hand_walk(&data, &run, &w, &[]));
    let m = at(first_window(&r), &["metrics"]);
    // Entry at 10; at the flip to b at 18 both branches are long: no cost; at the flip to a at 24
    // the book goes flat: one side.
    let one_book = 2.0 * SIDE;
    // Two masked books: a's on-run 10..17 pays an entry and an exit, b's off-run 18..23 too.
    let masked = 4.0 * SIDE;
    assert!((num(m, &["total_cost"]) - one_book).abs() <= 1e-15, "total cost {} vs {one_book}", num(m, &["total_cost"]));
    assert!((num(m, &["total_cost"]) - masked).abs() > 1e-6);
    let regime = at(first_window(&r), &["gate", "regime"]);
    assert_eq!(num(regime, &["switches", "switches"]), 2.0);
    assert!((num(regime, &["switch_cost"]) - SIDE).abs() <= 1e-15, "switch cost {}", num(regime, &["switch_cost"]));
    assert!((num(regime, &["total_cost"]) - one_book).abs() <= 1e-15);
    let parts = num(regime, &["states", "on", "net"]) + num(regime, &["states", "off", "net"]) + num(regime, &["seam"]);
    let net = num(m, &["mean_return"]) * num(m, &["periods"]);
    assert!((parts - net).abs() <= 1e-12, "state nets plus the seam make the composite net: {parts} vs {net}");
}

/// Usable rows of a CSV (finite target and `x1`) as `(stamp, symbol, state)`: the kernel over
/// every row of the file in file order, per symbol (no barrier: the data has no locked range).
fn usable_states(path: &str) -> Vec<(i64, usize, bool)> {
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<Vec<String>> = text.lines().skip(1).map(|l| l.split(',').map(str::to_string).collect()).collect();
    let parse = |v: &str| v.parse::<f64>().unwrap_or(f64::NAN);
    let group: Vec<u64> = rows.iter().map(|r| SYMBOLS.iter().position(|s| *s == r[1]).unwrap() as u64).collect();
    let gate: Vec<f64> = rows.iter().map(|r| parse(&r[5])).collect();
    let states = regime_gate::hysteresis_states(&gate, &group, BAND, DWELL);
    rows.iter()
        .zip(&states)
        .zip(&group)
        .filter(|((r, _), _)| parse(&r[7]).is_finite() && parse(&r[2]).is_finite())
        .map(|((r, &st), &g)| (((parse(&r[0]) as i64) - T0) / STEP, g as usize, st))
        .collect()
}

#[test]
fn gate_state_ignores_labels_and_row_drops() {
    let damaged = |k: i64, j: usize, starts: &[i64]| {
        let mut c = cell(k, j, starts);
        let (vs, ts) = (starts[0], starts[1]);
        // The target of 10 rows and the branch column of 10 other rows, next to the slice starts.
        if (j == 0 && (vs..vs + 5).contains(&k)) || (j == 1 && (ts..ts + 5).contains(&k)) {
            c.r = f64::NAN;
        }
        if j == 2 && ((vs + 1..vs + 6).contains(&k) || (ts + 1..ts + 6).contains(&k)) {
            c.x1 = f64::NAN;
        }
        c
    };
    for (tag, fixture) in [("sg-labels-a", composed("sg-labels-a")), ("sg-labels-b", composed_with("sg-labels-b", damaged))] {
        let c = fixture;
        let usable = usable_states(&c.data);
        let (v0, v1) = ((num(&arr(&c.j, &["split", "validation_time"])[0], &[]) as i64 - T0) / STEP, (c.last_validation() - T0) / STEP);
        let test_rows = num(&c.j, &["test_rows"]) as usize;
        let count = |rows: &[&(i64, usize, bool)]| (rows.len() as f64, rows.iter().filter(|r| r.2).count() as f64);
        let validation: Vec<&(i64, usize, bool)> = usable.iter().filter(|r| (v0..=v1).contains(&r.0)).collect();
        let test: Vec<&(i64, usize, bool)> = usable.iter().rev().take(test_rows).collect();
        for (slot, rows) in [("validation", &validation), ("test", &test)] {
            let regime = at(&c.j, &["compose", "chosen", slot, "regime"]);
            assert_eq!((num(regime, &["rows"]), num(regime, &["states", "on", "rows"])), count(rows), "{tag} compose {slot}");
        }
        let ts = test.iter().map(|r| r.0).min().unwrap();
        let w = format!("t,{},{},{},{}", T0, stamp(v0), stamp(ts), stamp(STAMPS as i64));
        let r = json(&c.walk(&c.run, &w, &[]));
        let regime = at(first_window(&r), &["gate", "regime"]);
        assert_eq!((num(regime, &["rows"]), num(regime, &["states", "on", "rows"])), count(&test), "{tag} walkforward test");
    }
}

#[test]
fn selection_states_do_not_read_test_rows() {
    let (_s, dir) = scratch("sg-intervals");
    let spec = write_spec(&dir);
    let table = dir.join("t.csv");
    std::fs::write(
        &table,
        format!("split_id,role,start,end\ns,train,{},{}\ns,test,{},{}\ns,validation,{},{}\n", stamp(0), stamp(150), stamp(160), stamp(220), stamp(240), stamp(330)),
    )
    .unwrap();
    let table = table.to_string_lossy().into_owned();
    let split = ["--split-table", table.as_str(), "--split-id", "s"];
    let data = write_csv(&dir, "d.csv", |k, j| cell(k, j, &[240]));
    let bull = pool_run(&dir, "bull.json", &data, &spec, &split, &["--pnl-mask", "g"], &["x1"]);
    let bear = pool_run(&dir, "bear.json", &data, &spec, &split, &["--pnl-mask", "0 - g"], &["0 - x1"]);
    let menu = write_menu(&dir, BAND, DWELL);
    let args = ["--gates", menu.as_str(), "--bull", bull.as_str(), "--bear", bear.as_str(), "--json"];
    let base = json(&compose(&data, &spec, &split, &args));
    // The gate column of every test row changes; the training, validation and other rows do not.
    let other = write_csv(&dir, "other.csv", |k, j| {
        let mut c = cell(k, j, &[240]);
        if (160..220).contains(&k) {
            c.h = -c.h * 3.0;
        }
        c
    });
    let changed = json(&compose(&other, &spec, &split, &args));
    for path in [&["compose", "composites"][..], &["compose", "selection"], &["objective", "validation"], &["objective", "finalist"], &["compose", "chosen", "validation"]] {
        assert_eq!(at(&base, path), at(&changed, path), "{path:?} changed with the test rows");
    }
    // The validation state restarts at the first row after the test interval plus the embargo.
    let embargo = num(&base, &["split", "embargo_seconds"]) as i64;
    let restart = (stamp(220) + embargo + STEP - 1) / STEP * STEP - T0;
    let start = at(&base, &["compose", "chosen", "validation", "regime", "start"]);
    assert_eq!(num(start, &["warmup_first_time"]), (T0 + restart) as f64, "embargo {embargo}");
    assert_eq!(num(start, &["warmup_rows"]), (3 * (stamp(240) - T0 - restart) / STEP) as f64);
}

#[test]
fn stateful_gate_refuses_label_columns() {
    let c = composed("sg-refuse");
    let body = std::fs::read_to_string(&c.run).unwrap();
    let edited = body.replace("\"expr\":\"h\"", "\"expr\":\"y\"").replace("\"columns\":[\"h\"]", "\"columns\":[\"y\"]").replace("greater(h, 0)", "greater(y, 0)");
    assert_ne!(edited, body);
    let bad = c.dir.join("bad.json").to_string_lossy().into_owned();
    std::fs::write(&bad, &edited).unwrap();
    let wf = c.walk(&bad, &c.test_window(), &[]);
    assert_eq!(code(&wf), 5, "{}", stderr(&wf));
    assert!(stderr(&wf).contains("D13"), "{}", stderr(&wf));
    let windows = ladder_windows(&c);
    let lad = rcsr(&["ladder", "--data", &c.data, "--target", "y", "--time", "t", "--group", "s", "--windows", &windows, "--features", "x1,x2", "--purge-seconds", "3600", "--target-spec", &c.spec, "--cost-bps", "5", "--cost-grid", "5", "--draws", "50", "--gbm-trees", "10", "--sr-run", &bad, "--out", "/dev/null"]);
    assert_eq!(code(&lad), 5, "{}", stderr(&lad));
    let mut args = vec!["--data", c.data.as_str(), "--target-spec", c.spec.as_str()];
    args.extend(base_flags(&TEMPORAL));
    args.extend(["--reselect-run", bad.as_str(), "--select-rule", "robust", "--json"]);
    let re = rcsr(&args);
    assert_eq!(code(&re), 5, "{}", stderr(&re));
    // The device arms and the band, vote and fitted-cut books exit 2 with a switch.
    for flags in [&["--device-pnl"][..], &["--batch-pnl"], &["--threshold-fit", "fitness"], &["--policy", "band", "--band", "0.1"], &["--unmasked"]] {
        let o = c.walk(&c.run, &c.test_window(), flags);
        assert_eq!(code(&o), 2, "{flags:?}: {}", stderr(&o));
    }
    // A composed run written before the switch block exits 3 and asks for a new compose run.
    let mut old = read_json(&c.run);
    set(&mut old, &["objective", "finalist", "switch"], None);
    set(&mut old, &["compose", "gate", "carried"], Some(Json::Bool(true)));
    let stale = c.dir.join("stale.json").to_string_lossy().into_owned();
    std::fs::write(&stale, text(&old)).unwrap();
    let o = c.walk(&stale, &c.test_window(), &[]);
    assert_eq!(code(&o), 3, "{}", stderr(&o));
    assert!(stderr(&o).contains("run rcsr compose again"), "{}", stderr(&o));
}

/// One expanding fold whose test block is the composed run's test rows.
fn ladder_windows(c: &Composed) -> String {
    let path = c.dir.join("windows.csv");
    std::fs::write(
        &path,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,f1,{},{},{},{},{}\n",
            T0,
            stamp(c.vs) - STEP,
            stamp(c.vs / 2),
            stamp(c.ts),
            stamp(STAMPS as i64)
        ),
    )
    .unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn ladder_prices_a_switch() {
    let c = composed("sg-ladder");
    let windows = ladder_windows(&c);
    let out = c.dir.join("ladder.json").to_string_lossy().into_owned();
    let o = rcsr(&["ladder", "--data", &c.data, "--target", "y", "--time", "t", "--group", "s", "--windows", &windows, "--features", "x1,x2", "--purge-seconds", "3600", "--target-spec", &c.spec, "--cost-bps", "5", "--cost-grid", "5", "--draws", "50", "--gbm-trees", "10", "--sr-run", &c.run, "--out", &out]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let lad = read_json(&out);
    let rung = arr(&lad, &["rungs"]).iter().find(|r| at(r, &["name"]).as_str() == Some("symbolic_regression")).expect("symbolic_regression rung");
    assert_eq!(s(rung, &["status"]), "ok");
    // Ladder periods are equal-weight means over the three symbols.
    let net = num(&arr(rung, &["net_by_cost"])[0], &["net"]) * SYMBOLS.len() as f64;
    let want = num(&c.j, &["objective", "test", "net_pnl"]);
    assert!((net - want).abs() <= 1e-12, "ladder {net} vs objective.test {want}");
    assert_eq!(s(&lad, &["sr_run", "switch", "gate", "expr"]), "h");
}

#[test]
fn reselect_run_reproduces_compose() {
    let c = composed("sg-reselect");
    let reselect = |extra: &[&str]| {
        let mut args = vec!["--data", c.data.as_str(), "--target-spec", c.spec.as_str()];
        args.extend(base_flags(&TEMPORAL));
        args.extend(["--reselect-run", c.run.as_str(), "--select-rule", "robust", "--json"]);
        args.extend_from_slice(extra);
        json(&rcsr(&args))
    };
    let same = reselect(&[]);
    assert_eq!(at(&same, &["objective", "finalist", "switch"]), at(&c.j, &["objective", "finalist", "switch"]));
    assert_eq!(at(&same, &["objective", "finalist", "expression_exact"]), at(&c.j, &["objective", "finalist", "expression_exact"]));
    let (a, b) = (num(&same, &["objective", "validation", "net_pnl"]), num(&c.j, &["objective", "validation", "net_pnl"]));
    assert!((a - b).abs() <= 1e-12, "{a} vs {b}");
    assert_eq!(s(&same, &["objective", "selection_rule", "reselected_from", "kind"]), "run");
    let other = reselect(&["--select-se", "2.5"]);
    let original = arr(&c.j, &["compose", "composites"]);
    let rescored = arr(&other, &["compose", "composites"]);
    assert_eq!(rescored.len(), original.len());
    let mut compared = 0;
    for (o, r) in original.iter().zip(rescored) {
        assert_eq!(num(o, &["id"]), num(r, &["id"]));
        if let (Some(x), Some(y)) = (o.get("validation").and_then(|v| v.get("net")).and_then(Json::as_f64), r.get("validation").and_then(|v| v.get("net")).and_then(Json::as_f64)) {
            assert!((x - y).abs() <= 1e-12, "composite {}: {x} vs {y}", num(o, &["id"]));
            compared += 1;
        }
    }
    assert!(compared >= 3, "{compared} composites re-scored");
    // A split other than the run's exits 2.
    let mut args = vec!["--data", c.data.as_str(), "--target-spec", c.spec.as_str()];
    args.extend(base_flags(&["--split", "temporal", "--validation-frac", "0.25", "--test-frac", "0.2"]));
    args.extend(["--reselect-run", c.run.as_str(), "--select-rule", "robust", "--json"]);
    assert_eq!(code(&rcsr(&args)), 2);
}

#[test]
fn front_prices_switch_members() {
    let c = composed("sg-front");
    let pareto = arr(&c.j, &["pareto"]);
    assert_eq!(pareto.len() as f64, num(&c.j, &["compose", "count", "composites"]));
    for m in pareto {
        assert_eq!(m.get("switch").is_some(), s(m, &["type"]) != "C0", "{}", s(m, &["expression"]));
        assert!(matches!(at(m, &["pruned"]), Json::Bool(_)));
    }
    let mut args = vec!["walkforward", "--data", c.data.as_str(), "--target", "y", "--front", c.run.as_str(), "--time", "t", "--group-by", "s"];
    let w = c.test_window();
    args.extend(["--window", w.as_str(), "--validation-frac", "0.3", "--json"]);
    let front = json(&rcsr(&args));
    let chosen = num(first_window(&front), &["candidate"]) as usize;
    let member = &pareto[chosen];
    let mut copy = read_json(&c.run);
    let exact = at(member, &["expression_exact"]).clone();
    set(&mut copy, &["objective", "finalist", "expression"], Some(exact.clone()));
    set(&mut copy, &["objective", "finalist", "expression_exact"], Some(exact));
    set(&mut copy, &["objective", "finalist", "switch"], member.get("switch").cloned());
    let path = c.dir.join("member.json").to_string_lossy().into_owned();
    std::fs::write(&path, text(&copy)).unwrap();
    let single = json(&c.walk(&path, &w, &[]));
    let (a, b) = (net_of(first_window(&front)), net_of(first_window(&single)));
    assert!((a - b).abs() <= 1e-12, "--front member {chosen}: {a} vs --from-run {b}");
}
