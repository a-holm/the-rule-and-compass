//! `--fitness pnl`: the exact long/flat regret book, the search wiring and the
//! CLI refusals (#480).

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/shape.rs"]
mod shape;

use rcsr::data::TradingSlice;
use rcsr::devlog::Json;
use rcsr::trading::{self, FitnessBook, TradingRows, TradingSpec};
use std::path::PathBuf;
use std::process::Output;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// Groups of `sizes` rows, 900 s apart, with deterministic returns.
fn slice(sizes: &[usize]) -> TradingSlice {
    let mut rows = TradingRows::default();
    let mut ret = Vec::new();
    let mut k = 0usize;
    for (g, &n) in sizes.iter().enumerate() {
        for i in 0..n {
            rows.group.push(g as u64);
            rows.time.push(1.6e9 + 900.0 * i as f64);
            ret.push(0.004 * (frac(k as f64 * 0.618_033_988_749_894_9) - 0.45));
            k += 1;
        }
    }
    TradingSlice { rows, ret }
}

fn spec() -> TradingSpec {
    TradingSpec {
        cost_bps: 5.0,
        slippage_bps: 1.0,
        min_trades: 0,
        ..Default::default()
    }
}

/// Predictions that flip often; the last row of every group is long.
fn prediction(s: &TradingSlice) -> Vec<f64> {
    let n = s.ret.len();
    (0..n)
        .map(|i| {
            let last = i + 1 == n || s.rows.group[i + 1] != s.rows.group[i];
            if last {
                1.0
            } else {
                frac(i as f64 * 0.754_877_666_246_692_7) - 0.4
            }
        })
        .collect()
}

#[test]
fn fitness_net_matches_the_exact_book() {
    let s = slice(&[200, 200, 1, 200, 1]);
    let sp = spec();
    let book = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let pred = prediction(&s);
    let t = book.totals(&pred);
    let exact = trading::evaluate(&pred, &s.ret, &s.rows, &sp).unwrap();
    let net = book.net_of(&t);
    assert!((net - exact.net_pnl).abs() <= 1e-12 * exact.net_pnl.abs().max(1.0), "{net} vs {}", exact.net_pnl);
    assert_eq!(t.flips as usize, exact.trades);
    assert_eq!(t.exits as usize, exact.exits);
    assert_eq!(t.exits, 5, "every group ends long");
    assert_eq!(book.groups(), 5);
}

#[test]
fn the_net_is_the_excess_over_cash_and_funding() {
    let mut s = slice(&[200, 200, 1, 200, 1]);
    s.rows.funding = (0..s.ret.len()).map(|i| 0.0001 * frac(i as f64 * 0.31)).collect();
    let sp = TradingSpec {
        cash_return: 0.0002,
        ..spec()
    };
    let book = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let pred = prediction(&s);
    let t = book.totals(&pred);
    let exact = trading::evaluate(&pred, &s.ret, &s.rows, &sp).unwrap();
    let net = book.net_of(&t) + sp.cash_return * s.ret.len() as f64;
    assert!((net - exact.net_pnl).abs() <= 1e-12 * exact.net_pnl.abs().max(1.0), "{net} vs {}", exact.net_pnl);
}

#[test]
fn the_segment_tree_matches_the_exact_book_over_many_segments() {
    let s = slice(&[9000, 8192, 1, 9000]);
    let sp = spec();
    let book = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let pred = prediction(&s);
    let t = book.totals(&pred);
    let exact = trading::evaluate(&pred, &s.ret, &s.rows, &sp).unwrap();
    let net = book.net_of(&t);
    assert!((net - exact.net_pnl).abs() <= 1e-9 * exact.net_pnl.abs().max(1.0), "{net} vs {}", exact.net_pnl);
    assert_eq!(t.flips as usize, exact.trades);
}

#[test]
fn regret_of_cash_is_one() {
    let s = slice(&[200, 200, 1, 200]);
    let book = FitnessBook::new(&s, &spec(), 900.0).unwrap();
    assert_eq!(book.regret(&vec![-1.0; s.ret.len()]), 1.0);
}

/// The hindsight long/flat path of one group as predictions (+1 long, -1 flat).
fn hindsight(s: &TradingSlice, sp: &TradingSpec) -> Vec<f64> {
    let side = sp.per_side_bps() * 1e-4;
    let n = s.ret.len();
    let mut out = vec![-1.0; n];
    let mut lo = 0;
    while lo < n {
        let mut hi = lo;
        while hi < n && s.rows.group[hi] == s.rows.group[lo] {
            hi += 1;
        }
        let m = hi - lo;
        let mut f = vec![[f64::NEG_INFINITY; 2]; m];
        let mut from = vec![[0usize; 2]; m];
        for t in 0..m {
            let e = s.ret[lo + t] - sp.cash_return;
            for p in 0..2 {
                let gain = if p == 1 { e } else { 0.0 };
                if t == 0 {
                    f[t][p] = gain - side * p as f64;
                } else {
                    for q in 0..2 {
                        let v = f[t - 1][q] - side * (p as f64 - q as f64).abs() + gain;
                        if v > f[t][p] {
                            f[t][p] = v;
                            from[t][p] = q;
                        }
                    }
                }
            }
        }
        let mut p = if f[m - 1][1] - side > f[m - 1][0] { 1 } else { 0 };
        for t in (0..m).rev() {
            out[lo + t] = if p == 1 { 1.0 } else { -1.0 };
            p = from[t][p];
        }
        lo = hi;
    }
    out
}

#[test]
fn regret_of_the_hindsight_path_is_zero() {
    let s = slice(&[200, 200, 1, 200, 1]);
    let sp = spec();
    let book = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let r = book.regret(&hindsight(&s, &sp));
    assert!(r.abs() <= 1e-12, "regret {r}");
}

#[test]
fn a_non_finite_prediction_scores_infinity() {
    let s = slice(&[50, 50]);
    let book = FitnessBook::new(&s, &spec(), 900.0).unwrap();
    let mut pred = prediction(&s);
    assert!(book.regret(&pred).is_finite());
    pred[10] = f64::NAN;
    assert_eq!(book.regret(&pred), f64::INFINITY);
    pred[10] = f64::INFINITY;
    assert_eq!(book.regret(&pred), f64::INFINITY);
}

#[test]
fn fewer_flips_than_min_trades_scores_infinity() {
    let s = slice(&[50, 50]);
    let sp = TradingSpec {
        min_trades: 3,
        ..spec()
    };
    let book = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let mut pred = vec![-1.0; 100];
    for p in pred.iter_mut().skip(10).take(5) {
        *p = 1.0;
    }
    assert_eq!(book.totals(&pred).flips, 2);
    assert_eq!(book.regret(&pred), f64::INFINITY);
    pred[70] = 1.0;
    assert_eq!(book.totals(&pred).flips, 4);
    assert!(book.regret(&pred).is_finite());
}

#[test]
fn the_book_refuses_what_it_cannot_price() {
    let s = slice(&[50, 50]);
    let mut shuffled = s.clone();
    shuffled.rows.group.swap(10, 60);
    assert!(FitnessBook::new(&shuffled, &spec(), 900.0).unwrap_err().contains("canonical"));
    assert!(FitnessBook::new(&s, &spec(), 1800.0).unwrap_err().contains("overlapping returns"));
    let ramp = TradingSpec {
        tau: Some(0.5),
        ..spec()
    };
    assert!(FitnessBook::new(&s, &ramp, 900.0).unwrap_err().contains("no tau"));
    let mut losing = s.clone();
    losing.ret.iter_mut().for_each(|r| *r = -0.01);
    assert!(FitnessBook::new(&losing, &spec(), 900.0).unwrap_err().contains("no profitable"));
}

#[test]
fn the_hash_covers_rows_cost_threshold_and_trade_floor() {
    let s = slice(&[50, 50]);
    let base = FitnessBook::new(&s, &spec(), 900.0).unwrap().hash();
    assert_eq!(base, FitnessBook::new(&s, &spec(), 900.0).unwrap().hash());
    for other in [
        TradingSpec { cost_bps: 6.0, ..spec() },
        TradingSpec { threshold: 0.1, ..spec() },
        TradingSpec { min_trades: 2, ..spec() },
    ] {
        assert_ne!(base, FitnessBook::new(&s, &other, 900.0).unwrap().hash());
    }
    let mut moved = s.clone();
    moved.ret[3] += 1e-6;
    assert_ne!(base, FitnessBook::new(&moved, &spec(), 900.0).unwrap().hash());
}

// ---- CLI ----

const T0: i64 = 1_600_000_000;
const N: usize = 600;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch() -> (Scratch, PathBuf) {
    let scratch = Scratch::new("pnl-fitness");
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

/// `r = 0.01 x1 + noise` is the realised return; `y = r + 0.05 x2` is the
/// regression target, so the squared error prefers `x2` and the P&L `x1`.
fn write_csv(dir: &std::path::Path, step: i64) -> String {
    write_csv_rows(dir, step, N)
}

fn write_csv_rows(dir: &std::path::Path, step: i64, rows: usize) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..rows {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", T0 + step * i as i64, x1, x2, r, r + 0.05 * x2));
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &std::path::Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":900,\"delay_seconds\":0,\
         \"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("s.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run(data: &str, spec: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "20",
        "--pop", "48", "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
        "--cost-bps", "5", "--validation-frac", "0.3",
    ];
    args.extend_from_slice(extra);
    rcsr_cmd::rcsr().args(&args).output().expect("run rcsr")
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

fn usage_error(o: &Output, needle: &str) {
    assert_eq!(code(o), 2, "{}", stderr(o));
    assert!(stderr(o).contains(needle), "want '{needle}' in: {}", stderr(o));
}

fn pareto(j: &Json) -> &Vec<Json> {
    match at(j, &["pareto"]) {
        Json::Arr(a) => a,
        _ => panic!("pareto is not an array"),
    }
}

#[test]
fn pnl_fitness_prefers_the_tradable_signal() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);

    let metric = json(&run(&data, &spec, &[]));
    let pnl = json(&run(&data, &spec, &["--fitness", "pnl"]));

    assert_eq!(at(&metric, &["search_fitness", "kind"]).as_str(), Some("metric"));
    let f = at(&pnl, &["search_fitness"]);
    assert_eq!(f.get("kind").and_then(Json::as_str), Some("long_pnl_regret"));
    assert_eq!(f.get("backend").and_then(Json::as_str), Some("cpu_f64"));
    assert!(num(f, &["bound"]) > 0.0);
    assert_eq!(at(&pnl, &["objective", "finalist", "search_fitness"]).as_str(), Some("long_pnl_regret"));

    let best = pareto(&pnl)
        .iter()
        .min_by(|a, b| num(a, &["fitness", "train_regret"]).total_cmp(&num(b, &["fitness", "train_regret"])))
        .unwrap();
    let expression = at(best, &["expression"]).as_str().unwrap();
    assert!(expression.contains("x1"), "best by regret: {expression}");
    assert!(num(best, &["fitness", "train_net"]) > 0.0);
    assert!(num(best, &["fitness", "train_regret"]) < 1.0);

    let v_pnl = num(&pnl, &["objective", "validation", "net_pnl"]);
    let v_metric = num(&metric, &["objective", "validation", "net_pnl"]);
    assert!(v_pnl > v_metric, "validation net {v_pnl} (pnl) vs {v_metric} (metric)");
}

#[test]
fn pnl_fitness_is_identical_for_any_thread_count() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let one = json(&run(&data, &spec, &["--fitness", "pnl", "--threads", "1"]));
    let four = json(&run(&data, &spec, &["--fitness", "pnl", "--threads", "4"]));
    let key = |j: &Json| {
        pareto(j)
            .iter()
            .map(|m| {
                (
                    at(m, &["expression"]).as_str().unwrap().to_string(),
                    num(m, &["fitness", "train_regret"]).to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&one), key(&four));
    assert_eq!(
        num(&one, &["search", "candidates_evaluated"]),
        num(&four, &["search", "candidates_evaluated"])
    );
}

#[test]
fn fitness_pnl_refusals_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);

    let no_objective = rcsr_cmd::rcsr()
        .args([
            "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", &spec, "--cpu", "--no-dev-log", "--generations", "2", "--fitness", "pnl",
        ])
        .output()
        .unwrap();
    usage_error(&no_objective, "--fitness pnl needs --objective");

    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--predictor-samples", "100"]),
        "--predictor-samples",
    );
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--band-units", "sd"]), "--band-units needs --band");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--band", "-0.5"]), "band must be finite and non-negative");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--band", "nan"]), "--band needs a finite number");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--band", "1", "--band-units", "ticks"]), "--band-units must be score or sd");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--min-holding", "0"]), "min_holding must be at least 1");
    usage_error(&run(&data, &spec, &["--band", "0.5", "--position", "top_tercile"]), "--band is not supported with --position top_tercile");
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--fit-samples", "50"]),
        "--fit-samples",
    );
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--cv", "k-fold"]), "--fitness pnl is not supported with --cv");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--stability", "1"]), "--fitness pnl is not supported with --stability");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--null-runs", "1"]), "--fitness pnl is not supported with --null-runs");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--null-block", "4"]), "--fitness pnl is not supported with --null-runs");
    usage_error(&run(&data, &spec, &["--fitness", "other"]), "--fitness must be");

    let overlapping = write_csv(&dir, 300);
    usage_error(&run(&overlapping, &spec, &["--fitness", "pnl"]), "overlapping returns");
}

#[test]
fn resume_needs_the_same_book() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let cp = dir.join("cp.rcsr").to_string_lossy().into_owned();

    let first = run(&data, &spec, &["--fitness", "pnl", "--generations", "10", "--save", &cp]);
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let straight = json(&run(&data, &spec, &["--fitness", "pnl", "--generations", "20"]));
    let resumed = json(&run(&data, &spec, &["--fitness", "pnl", "--generations", "20", "--resume", &cp]));
    let key = |j: &Json| {
        pareto(j)
            .iter()
            .map(|m| (at(m, &["expression"]).as_str().unwrap().to_string(), num(m, &["fitness", "train_regret"]).to_bits()))
            .collect::<Vec<_>>()
    };
    assert_eq!(key(&straight), key(&resumed));

    let other_cost = run(&data, &spec, &["--fitness", "pnl", "--generations", "20", "--resume", &cp, "--cost-bps", "7"]);
    usage_error(&other_cost, "fitness");
    let no_flag = run(&data, &spec, &["--generations", "20", "--resume", &cp]);
    usage_error(&no_flag, "--fitness pnl");
}

// ---- #484: time-block consistency term ----

/// Rows 0..REGIME_ROWS pay `+0.08 x1`, later rows pay `-0.02 x1`; `x2` pays
/// `+0.008` in every row.
const REGIME_ROWS: usize = 90;

fn write_regime_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let c1 = if i < REGIME_ROWS { 0.08 } else { -0.02 };
        let r = c1 * x1 + 0.008 * x2 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r, r));
    }
    let path = dir.join("regime.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn members(j: &Json) -> Vec<(String, u64)> {
    pareto(j)
        .iter()
        .map(|m| (at(m, &["expression"]).as_str().unwrap().to_string(), num(m, &["fitness", "train_regret"]).to_bits()))
        .collect()
}

fn best_expression(j: &Json) -> String {
    pareto(j)
        .iter()
        .min_by(|a, b| num(a, &["fitness", "train_regret"]).total_cmp(&num(b, &["fitness", "train_regret"])))
        .map(|m| at(m, &["expression"]).as_str().unwrap().to_string())
        .unwrap()
}

#[test]
fn one_block_is_the_unblocked_fitness() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let plain = run(&data, &spec, &["--fitness", "pnl"]);
    let one = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "1", "--pnl-block-lambda", "4"]);
    let (a, b) = (json(&plain), json(&one));
    assert_eq!(members(&a), members(&b));
    assert_eq!(at(&a, &["search_fitness"]).get("book_hash"), at(&b, &["search_fitness"]).get("book_hash"));
    assert!(at(&b, &["search_fitness"]).get("blocks").is_none());
    assert!(pareto(&b).iter().all(|m| at(m, &["fitness"]).get("block_regret").is_none()));
    assert_eq!(without_digits(&plain), without_digits(&one));
}

#[test]
fn blocks_report_per_block_regret_and_are_thread_invariant() {
    let (_scratch, dir) = scratch();
    let data = write_regime_csv(&dir);
    let spec = write_spec(&dir);
    let args = ["--fitness", "pnl", "--pnl-blocks", "4", "--pnl-block-lambda", "1"];
    let one = json(&run(&data, &spec, &[&args[..], &["--threads", "1"]].concat()));
    let four = json(&run(&data, &spec, &[&args[..], &["--threads", "4"]].concat()));
    assert_eq!(members(&one), members(&four));
    let f = at(&one, &["search_fitness"]);
    assert_eq!(num(f, &["blocks"]), 4.0);
    assert_eq!(num(f, &["lambda"]), 1.0);
    match at(f, &["block_bounds"]) {
        Json::Arr(a) => assert!(a.len() == 4 && a.iter().all(|v| v.as_f64().unwrap() > 0.0)),
        _ => panic!("block_bounds is not an array"),
    }
    for m in pareto(&one) {
        let r: Vec<f64> = match at(m, &["fitness", "block_regret"]) {
            Json::Arr(a) => a.iter().map(|v| v.as_f64().unwrap()).collect(),
            _ => panic!("block_regret is not an array"),
        };
        assert_eq!(r.len(), 4);
        let mean = r.iter().sum::<f64>() / 4.0;
        let sd = (r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 3.0).sqrt();
        let fit = num(m, &["fitness", "train_regret"]);
        assert!((fit - (mean + sd)).abs() <= 1e-9, "{fit} vs {}", mean + sd);
    }
}

#[test]
fn the_block_term_prefers_the_steady_signal() {
    let (_scratch, dir) = scratch();
    let data = write_regime_csv(&dir);
    let spec = write_spec(&dir);
    let pooled = json(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "1", "--seed", "4"]));
    let blocked = json(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "4", "--pnl-block-lambda", "1", "--seed", "4"]));
    let (p, b) = (best_expression(&pooled), best_expression(&blocked));
    assert!(p.contains("x1"), "pooled best: {p}");
    assert!(b.contains("x2") && !b.contains("x1"), "blocked best: {b}");
}

#[test]
fn block_flags_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    usage_error(&run(&data, &spec, &["--pnl-blocks", "4"]), "--pnl-blocks needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--pnl-block-lambda", "2"]), "--pnl-block-lambda needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "65"]), "--pnl-blocks must be");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "0"]), "--pnl-blocks must be");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-block-lambda", "-1"]), "--pnl-block-lambda must be");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-block-lambda", "nan"]), "--pnl-block-lambda must be");
}

#[test]
fn resume_needs_the_same_blocks() {
    let (_scratch, dir) = scratch();
    let data = write_regime_csv(&dir);
    let spec = write_spec(&dir);
    let cp = dir.join("cp.rcsr").to_string_lossy().into_owned();
    let blocked = ["--fitness", "pnl", "--pnl-blocks", "3"];
    let first = run(&data, &spec, &[&blocked[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let straight = json(&run(&data, &spec, &[&blocked[..], &["--generations", "20"]].concat()));
    let resumed = json(&run(&data, &spec, &[&blocked[..], &["--generations", "20", "--resume", &cp]].concat()));
    assert_eq!(members(&straight), members(&resumed));
    let other = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "4", "--generations", "20", "--resume", &cp]);
    usage_error(&other, "fitness");
}

// ---- #1178: up to 64 blocks and the block statistic ----

fn block_regrets(m: &Json) -> Vec<f64> {
    match at(m, &["fitness", "block_regret"]) {
        Json::Arr(a) => a.iter().map(|v| v.as_f64().unwrap()).collect(),
        _ => panic!("block_regret is not an array"),
    }
}

#[test]
fn block_stat_flags_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    usage_error(&run(&data, &spec, &["--pnl-stat", "cvar"]), "--pnl-stat needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--pnl-cvar-q", "0.5"]), "--pnl-cvar-q needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-stat", "median"]), "--pnl-stat must be mean-sd, mean-semi or cvar");
    for q in ["0", "-0.1", "1.5", "nan", "x"] {
        usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-stat", "cvar", "--pnl-cvar-q", q]), "--pnl-cvar-q must be");
    }
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-cvar-q", "0.5"]), "--pnl-cvar-q needs --pnl-stat cvar");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-stat", "mean-semi", "--pnl-cvar-q", "0.5"]), "--pnl-cvar-q needs --pnl-stat cvar");
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--pnl-stat", "cvar", "--pnl-block-lambda", "2"]),
        "--pnl-block-lambda has no effect under --pnl-stat cvar",
    );
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "65"]), "--pnl-blocks must be an integer from 1 to 64");
    // 80 rows leave fewer than 64 distinct training timestamps.
    let short = write_csv_rows(&dir, 900, 80);
    usage_error(&run(&short, &spec, &["--fitness", "pnl", "--pnl-blocks", "64"]), "distinct training timestamps");
}

/// The acceptance run: 48 blocks, 20 generations, `search_fitness.stat`, and
/// every member's fitness is the statistic of its 48 block regrets.
#[test]
fn k48_search_writes_the_stat() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 2000);
    let spec = write_spec(&dir);
    for (extra, stat) in [
        (&["--pnl-blocks", "48"][..], "mean-sd"),
        (&["--pnl-blocks", "48", "--pnl-stat", "mean-semi", "--pnl-block-lambda", "2"][..], "mean-semi"),
        (&["--pnl-blocks", "48", "--pnl-stat", "cvar", "--pnl-cvar-q", "0.25"][..], "cvar"),
    ] {
        let j = json(&run(&data, &spec, &[&["--fitness", "pnl"][..], extra].concat()));
        assert_eq!(num(&j, &["generations_run"]), 20.0, "{stat}");
        let f = at(&j, &["search_fitness"]);
        assert_eq!(f.get("stat").and_then(Json::as_str), Some(stat));
        assert_eq!(num(f, &["blocks"]), 48.0);
        match at(f, &["block_bounds"]) {
            Json::Arr(a) => assert!(a.len() == 48 && a.iter().all(|v| v.as_f64().unwrap() > 0.0)),
            _ => panic!("block_bounds is not an array"),
        }
        assert_eq!(f.get("cvar_q").and_then(Json::as_f64), (stat == "cvar").then_some(0.25), "{stat}");
        let lambda = num(f, &["lambda"]);
        let mut finite = 0;
        for m in pareto(&j) {
            let r = block_regrets(m);
            assert_eq!(r.len(), 48);
            let fit = num(m, &["fitness", "train_regret"]);
            let mean = r.iter().sum::<f64>() / 48.0;
            let want = match stat {
                "mean-sd" => mean + lambda * (r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / 47.0).sqrt(),
                "mean-semi" => mean + lambda * (r.iter().map(|v| (v - 1.0).max(0.0).powi(2)).sum::<f64>() / 48.0).sqrt(),
                _ => {
                    let mut s = r.clone();
                    s.sort_by(|a, b| b.total_cmp(a));
                    s[..12].iter().sum::<f64>() / 12.0
                }
            };
            assert!((fit - want).abs() <= 1e-9, "{stat}: {fit} vs {want}");
            finite += 1;
        }
        assert!(finite > 0, "{stat}: an empty front");
    }
}

#[test]
fn block_stat_search_is_thread_invariant() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 2000);
    let spec = write_spec(&dir);
    let args = ["--fitness", "pnl", "--pnl-blocks", "20", "--pnl-stat", "cvar", "--pnl-cvar-q", "0.3"];
    let one = json(&run(&data, &spec, &[&args[..], &["--threads", "1"]].concat()));
    let four = json(&run(&data, &spec, &[&args[..], &["--threads", "4"]].concat()));
    assert_eq!(members(&one), members(&four));
}

/// A run without the new flags reports no `stat`, and `--pnl-stat mean-sd`
/// spelled out is that run.
#[test]
fn explicit_mean_sd_is_the_default_run() {
    let (_scratch, dir) = scratch();
    let data = write_regime_csv(&dir);
    let spec = write_spec(&dir);
    for blocks in ["1", "4", "8"] {
        let plain = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", blocks]);
        let spelled = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", blocks, "--pnl-stat", "mean-sd"]);
        let (a, b) = (json(&plain), json(&spelled));
        assert_eq!(members(&a), members(&b), "{blocks} blocks");
        let (fa, fb) = (at(&a, &["search_fitness"]), at(&b, &["search_fitness"]));
        assert_eq!(fa.get("book_hash"), fb.get("book_hash"));
        assert!(fa.get("stat").is_none() && fa.get("cvar_q").is_none(), "{blocks} blocks");
        assert_eq!(without_digits(&plain), without_digits(&spelled), "{blocks} blocks");
    }
}

/// A resume with another block count, statistic or q exits 2.
#[test]
fn book_hash_covers_block_stat() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 2000);
    let spec = write_spec(&dir);
    let cp = dir.join("stat.rcsr").to_string_lossy().into_owned();
    let saved = ["--fitness", "pnl", "--pnl-blocks", "12", "--pnl-stat", "cvar", "--pnl-cvar-q", "0.25"];
    let first = run(&data, &spec, &[&saved[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let straight = json(&run(&data, &spec, &[&saved[..], &["--generations", "20"]].concat()));
    let resumed = json(&run(&data, &spec, &[&saved[..], &["--generations", "20", "--resume", &cp]].concat()));
    assert_eq!(members(&straight), members(&resumed));
    let resume = |extra: &[&str]| run(&data, &spec, &[&["--fitness", "pnl", "--generations", "20", "--resume", &cp][..], extra].concat());
    usage_error(&resume(&["--pnl-blocks", "10", "--pnl-stat", "cvar", "--pnl-cvar-q", "0.25"]), "fitness");
    usage_error(&resume(&["--pnl-blocks", "12", "--pnl-stat", "mean-semi"]), "fitness");
    usage_error(&resume(&["--pnl-blocks", "12", "--pnl-stat", "cvar", "--pnl-cvar-q", "0.5"]), "fitness");
    usage_error(&resume(&["--pnl-blocks", "12"]), "fitness");
}

#[test]
fn prune_fits_candidates_under_pnl_above_prune_rows() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 3000);
    let spec = write_spec(&dir);
    let j = json(&run(
        &data,
        &spec,
        &["--fitness", "pnl", "--prune", "--prune-rows", "1000", "--generations", "8"],
    ));
    let fits = num(&j, &["prune", "fits"]);
    assert!(fits > 0.0, "prune made {fits} fits");
}

// ---- #848: reject trees whose position cannot depend on the inputs ----

const FIXED_OPS: &str = "add,sub,mul,div,abs,exp,square,tanh,logistic,gauss";

fn run_fixed(data: &str, spec: &str, log: &std::path::Path, extra: &[&str]) -> Output {
    let log = log.to_string_lossy().into_owned();
    let args = [
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--seed", "11", "--generations", "20", "--pop", "48",
        "--ops", FIXED_OPS, "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
        "--cost-bps", "5", "--validation-frac", "0.3", "--dev-log", &log, "--dev-log-level", "full",
        "--dev-log-every", "1",
    ];
    rcsr_cmd::rcsr()
        .args(args)
        .args(extra)
        .output()
        .expect("run rcsr")
}

/// Offspring the run rejected for a fixed position, summed over its generations.
fn fixed_rejections(log: &std::path::Path) -> f64 {
    std::fs::read_to_string(log)
        .expect("dev log")
        .lines()
        .filter_map(Json::parse)
        .filter(|e| e.get("event").and_then(Json::as_str) == Some("generation"))
        .map(|e| num(&e, &["variation", "fixed_position_rejected"]))
        .sum()
}

fn fixed_members(j: &Json) -> usize {
    let vars = ["x1".to_string(), "x2".to_string()];
    pareto(j)
        .iter()
        .filter(|m| {
            let text = at(m, &["expression_exact"]).as_str().unwrap();
            let tree = rcsr::parse::parse(text, &vars).unwrap_or_else(|e| panic!("{text}: {e:?}"));
            rcsr::expr::fixed_position(&tree, 0.0).is_some()
        })
        .count()
}

#[test]
fn reject_fixed_position_leaves_no_fixed_member_on_the_front() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 90_000);
    let spec = write_spec(&dir);
    let log = dir.join("on.jsonl");
    let on = json(&run_fixed(&data, &spec, &log, &["--fitness", "pnl", "--reject-fixed-position"]));
    assert!(fixed_rejections(&log) > 0.0, "the filter rejected nothing");
    assert!(!pareto(&on).is_empty());
    assert_eq!(fixed_members(&on), 0);
    assert!(matches!(at(&on, &["search", "fixed_position_front_fallback"]), Json::Bool(false)));
}

#[test]
fn the_flag_has_no_search_effect_under_the_metric_fitness() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 600);
    let spec = write_spec(&dir);
    let key = |j: &Json| {
        pareto(j)
            .iter()
            .map(|m| (at(m, &["expression"]).as_str().unwrap().to_string(), num(m, &["train_error"]).to_bits()))
            .collect::<Vec<_>>()
    };
    let off = json(&run_fixed(&data, &spec, &dir.join("off.jsonl"), &[]));
    let log = dir.join("on.jsonl");
    let on = json(&run_fixed(&data, &spec, &log, &["--reject-fixed-position"]));
    assert_eq!(key(&off), key(&on));
    assert_eq!(fixed_rejections(&log), 0.0);
}

#[test]
fn resume_needs_the_same_reject_fixed_position() {
    let (_scratch, dir) = scratch();
    let data = write_csv_rows(&dir, 900, 600);
    let spec = write_spec(&dir);
    let cp = dir.join("cp.rcsr").to_string_lossy().into_owned();
    let first = run_fixed(
        &data,
        &spec,
        &dir.join("a.jsonl"),
        &["--fitness", "pnl", "--reject-fixed-position", "--generations", "10", "--save", &cp],
    );
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    assert!(std::fs::read_to_string(&cp).unwrap().contains("config reject_fixed_position true"));
    let same = run_fixed(&data, &spec, &dir.join("b.jsonl"), &["--fitness", "pnl", "--reject-fixed-position", "--resume", &cp]);
    assert_eq!(code(&same), 0, "{}", stderr(&same));
    let plain = dir.join("plain.rcsr").to_string_lossy().into_owned();
    let off = run_fixed(&data, &spec, &dir.join("c.jsonl"), &["--fitness", "pnl", "--generations", "10", "--save", &plain]);
    assert_eq!(code(&off), 0, "{}", stderr(&off));
    assert!(!std::fs::read_to_string(&plain).unwrap().contains("reject_fixed_position"));
    let other = run_fixed(&data, &spec, &dir.join("d.jsonl"), &["--fitness", "pnl", "--reject-fixed-position", "--resume", &plain]);
    usage_error(&other, "--reject-fixed-position");
}

// ---- #849: fixed-position report and finalist refusal ----

/// A rising market: every row pays `0.003` plus a small `x1` term, so buy and
/// hold has the best validation Sharpe and a positive constant is the finalist.
fn write_drift_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..600 {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.003 + 0.002 * x1 + 0.0004 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r, r + 0.05 * x2));
    }
    let path = dir.join("drift.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run_drift(data: &str, spec: &str, extra: &[&str]) -> Json {
    let args = [
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "20", "--pop", "48",
        "--metric", "squared", "--json", "--quiet", "--objective", "sharpe", "--cost-bps", "5",
        "--validation-frac", "0.3",
    ];
    let out = rcsr_cmd::rcsr().args(args).args(extra).output().expect("run rcsr");
    json(&out)
}

fn member_fixed(m: &Json) -> Option<String> {
    match at(m, &["validation_book", "fixed_position"]) {
        Json::Str(s) => Some(s.clone()),
        Json::Null => None,
        other => panic!("fixed_position is neither a string nor null: {other:?}"),
    }
}

fn finalist_fixed(j: &Json) -> Option<String> {
    match at(j, &["objective", "finalist", "fixed_position"]) {
        Json::Str(s) => Some(s.clone()),
        Json::Null => None,
        other => panic!("fixed_position is neither a string nor null: {other:?}"),
    }
}

#[test]
fn every_front_member_reports_fixed_position() {
    let (_scratch, dir) = scratch();
    let data = write_drift_csv(&dir);
    let spec = write_spec(&dir);
    for fitness in ["metric", "pnl"] {
        let j = run_drift(&data, &spec, &["--fitness", fitness]);
        let front = pareto(&j);
        assert!(!front.is_empty());
        let vars = ["x1".to_string(), "x2".to_string()];
        for m in front {
            let exact = at(m, &["expression_exact"]).as_str().unwrap();
            let tree = rcsr::parse::parse(exact, &vars).unwrap();
            let want = rcsr::expr::fixed_position(&tree, 0.0).map(|long| if long { "long" } else { "flat" });
            assert_eq!(member_fixed(m).as_deref(), want, "{fitness}: {exact}");
        }
        assert!(
            front.iter().any(|m| member_fixed(m).as_deref() == Some("long")),
            "{fitness}: the rising market keeps a positive constant on the front"
        );
        if fitness == "metric" {
            assert!(front.iter().any(|m| member_fixed(m).is_none()), "no member decided by the inputs");
        }
        let fin = at(&j, &["objective", "finalist", "expression_exact"]).as_str().unwrap().to_string();
        let tree = rcsr::parse::parse(&fin, &vars).unwrap();
        let want = rcsr::expr::fixed_position(&tree, 0.0).map(|long| if long { "long" } else { "flat" });
        assert_eq!(finalist_fixed(&j).as_deref(), want);
    }
}

fn member_realised(m: &Json) -> Option<String> {
    match at(m, &["validation_book", "fixed_position_realised"]) {
        Json::Str(s) => Some(s.clone()),
        Json::Null => None,
        other => panic!("fixed_position_realised is neither a string nor null: {other:?}"),
    }
}

/// #849: the realised validation book is reported at any threshold: `"long"` when
/// the book was long on every row, `"flat"` when on none, else `null`.
#[test]
fn the_realised_position_follows_the_validation_exposure() {
    let (_scratch, dir) = scratch();
    let data = write_drift_csv(&dir);
    let spec = write_spec(&dir);
    for threshold in ["0", "1e-30"] {
        let j = run_drift(&data, &spec, &["--fitness", "metric", "--threshold", threshold]);
        let mut seen_long = false;
        for m in pareto(&j) {
            let exposure = num(m, &["validation_book", "exposure"]);
            let want = if exposure == 1.0 { Some("long") } else if exposure == 0.0 { Some("flat") } else { None };
            assert_eq!(member_realised(m).as_deref(), want, "threshold {threshold}, exposure {exposure}");
            seen_long |= want == Some("long");
        }
        assert!(seen_long, "threshold {threshold}: the rising market has an always-long member");
        let fin = match at(&j, &["objective", "finalist", "fixed_position_realised"]) {
            Json::Str(s) => Some(s.as_str()),
            _ => None,
        };
        let e = num(&j, &["objective", "validation", "exposure"]);
        assert_eq!(fin, if e == 1.0 { Some("long") } else if e == 0.0 { Some("flat") } else { None });
    }
}

#[test]
fn reject_fixed_position_refuses_the_finalist() {
    let (_scratch, dir) = scratch();
    let data = write_drift_csv(&dir);
    let spec = write_spec(&dir);
    for fitness in ["metric", "pnl"] {
        let off = run_drift(&data, &spec, &["--fitness", fitness]);
        assert_eq!(finalist_fixed(&off).as_deref(), Some("long"), "{fitness}: buy and hold wins without the flag");
        let on = run_drift(&data, &spec, &["--fitness", fitness, "--reject-fixed-position"]);
        assert_eq!(finalist_fixed(&on), None, "{fitness}: the flag refuses the fixed finalist");
        assert!(num(&on, &["objective", "validation", "exposure"]) < 1.0, "{fitness}");
        for m in pareto(&on) {
            if member_fixed(m).is_some() || member_realised(m).is_some() {
                assert!(matches!(at(m, &["validation_book", "objective"]), Json::Null), "{fitness}");
            }
        }
        if fitness == "metric" {
            let usable = |j: &Json| num(j, &["objective", "usable_candidates"]);
            let refused = pareto(&off)
                .iter()
                .filter(|m| (member_fixed(m).is_some() || member_realised(m).is_some()) && !matches!(at(m, &["validation_book", "objective"]), Json::Null))
                .count() as f64;
            assert_eq!(usable(&on), usable(&off) - refused);
            assert_eq!(pareto(&on).len(), pareto(&off).len(), "the metric search is unchanged");
        }
    }
}

// ---- #925: the no-trade band and the minimum hold ----

use rcsr::trading::{BandSpec, BandUnits};

fn band_spec(band: f64, units: BandUnits, hold: usize) -> TradingSpec {
    TradingSpec {
        band: BandSpec { units, resolved: if units == BandUnits::Score { band } else { 0.0 }, ..BandSpec::score(band) },
        min_holding: hold,
        ..spec()
    }
}

/// A persistent score per group (AR(1), coefficient 0.8, LCG innovations), so
/// a band and a hold change the book.
fn persistent_scores(s: &TradingSlice, scale: f64) -> Vec<f64> {
    let mut state = 0x2545_f491_4f6c_dd1du64;
    let mut out = Vec::with_capacity(s.ret.len());
    let mut v = 0.0;
    for i in 0..s.ret.len() {
        if i == 0 || s.rows.group[i] != s.rows.group[i - 1] {
            v = 0.0;
        }
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        let u = (state >> 11) as f64 / (1u64 << 53) as f64;
        v = 0.8 * v + (u - 0.5);
        out.push(scale * v);
    }
    out
}

#[test]
fn band_book_refusals() {
    let s = slice(&[50, 50]);
    let negative = band_spec(-0.1, BandUnits::Score, 1);
    assert!(FitnessBook::new(&s, &negative, 900.0).unwrap_err().contains("band must be finite"));
    let ramp = TradingSpec { tau: Some(0.5), ..band_spec(0.2, BandUnits::Score, 1) };
    assert!(ramp.validate().unwrap_err().contains("no tau"));
    let zero = band_spec(0.2, BandUnits::Score, 0);
    assert!(zero.validate().unwrap_err().contains("min_holding"));
}

/// The search book against the book `walkforward --policy band` reports and
/// against the objective's book, on the same rows and scores.
#[test]
fn band_fitness_matches_walkforward_book() {
    let s = slice(&[500, 500, 500]);
    let scores = persistent_scores(&s, 1.0);
    let names = ["A", "B", "C"];
    let mut text = String::from("ts,sym,x,target\n");
    for (g, name) in names.iter().enumerate() {
        for i in 0..20 {
            text.push_str(&format!("{},{},{},{}\n", i, name, 0.0, 0.0));
        }
        for i in 0..500 {
            let k = g * 500 + i;
            text.push_str(&format!("{},{},{},{}\n", 1000 + i, name, scores[k], s.ret[k]));
        }
    }
    let table = rcsr::data::parse_table(&text).unwrap();
    let mut rows = TradingRows::default();
    for (g, _) in names.iter().enumerate() {
        for i in 0..500 {
            rows.group.push(g as u64);
            rows.time.push(1000.0 + i as f64);
        }
    }
    let book_slice = TradingSlice { rows, ret: s.ret.clone() };
    for (band, hold) in [(0.3, 1usize), (0.0, 8), (0.5, 24)] {
        let sp = band_spec(band, BandUnits::Score, hold);
        let book = FitnessBook::new(&book_slice, &sp, 0.0).unwrap();
        assert!(!book.is_plain());
        let t = book.totals(&scores);
        let net = book.net_of(&t);
        let walk_spec = rcsr::walkforward::EvalSpec {
            expression: "x".to_string(),
            vars: vec!["x".to_string()],
            target: "target".to_string(),
            time: "ts".to_string(),
            group: Some("sym".to_string()),
            windows: vec![rcsr::walkforward::Window {
                label: "w".to_string(),
                train_start: 0.0,
                train_end: 20.0,
                test_start: 1000.0,
                test_end: 1500.0,
            }],
            horizon: 0.0,
            embargo: None,
            cost_bps: sp.per_side_bps(),
            threshold_quantile: 0.7,
            threshold_fixed: Some(sp.threshold),
            validation_frac: 0.0,
            periods_per_year: Some(4.0),
            candidates: Vec::new(),
            band_select: false,
            pool_reset_flat: false,
            allow_lockbox: false,
            device_pnl: false,
            control_seed: 0,
            policy: rcsr::walkforward::Policy::Band,
            band_width: band,
            min_hold: hold,
            gpu: false,
        };
        let report = rcsr::walkforward::run_table(&table, &walk_spec, "test").unwrap();
        let walk = report.windows[0].band.as_ref().expect("the band book").net;
        assert!((net - walk).abs() <= 1e-12, "band {band} hold {hold}: fitness net {net} walkforward {walk}");
        let objective = trading::evaluate(&scores, &book_slice.ret, &book_slice.rows, &sp).unwrap();
        assert!((objective.net_pnl - walk).abs() <= 1e-12, "objective net {} walkforward {walk}", objective.net_pnl);
        assert!(t.flips > 10 && t.exits > 0, "band {band} hold {hold}: {t:?}");
    }
}

#[test]
fn band_zero_hold_one_is_the_plain_book() {
    let s = slice(&[300, 1, 300]);
    let pred = persistent_scores(&s, 1.0);
    let plain = FitnessBook::new(&s, &spec(), 900.0).unwrap();
    let banded = FitnessBook::new(&s, &band_spec(0.0, BandUnits::Score, 1), 900.0).unwrap();
    assert!(banded.is_plain());
    assert_eq!(plain.hash(), banded.hash());
    // The band pass at half-width 0 and hold 1 is the plain tree.
    assert_eq!(plain.band_pass(&pred, 0.0, false)[0], plain.totals(&pred));
    let blocked = FitnessBook::with_blocks(&s, &spec(), 900.0, 3, 1.0).unwrap();
    assert_eq!(blocked.band_pass(&pred, 0.0, true), blocked.block_totals(&pred));
}

#[test]
fn the_band_book_hash_covers_band_units_and_hold() {
    let s = slice(&[50, 50]);
    let hashes: Vec<u64> = [
        band_spec(0.0, BandUnits::Score, 1),
        band_spec(0.5, BandUnits::Score, 1),
        band_spec(0.5, BandUnits::Sd, 1),
        band_spec(0.5, BandUnits::Score, 3),
        band_spec(0.0, BandUnits::Score, 3),
        band_spec(0.0, BandUnits::Sd, 1),
    ]
    .iter()
    .map(|sp| FitnessBook::new(&s, sp, 900.0).unwrap().hash())
    .collect();
    for i in 0..hashes.len() {
        for j in i + 1..hashes.len() {
            assert_ne!(hashes[i], hashes[j], "specs {i} and {j}");
        }
    }
}

#[test]
fn band_sd_is_scale_invariant() {
    let s = slice(&[400, 400, 1, 400]);
    let pred = persistent_scores(&s, 1.0);
    let scaled: Vec<f64> = pred.iter().map(|p| p * 1000.0).collect();
    let sd_book = FitnessBook::new(&s, &band_spec(1.0, BandUnits::Sd, 4), 900.0).unwrap();
    let (a, b) = (sd_book.totals(&pred), sd_book.totals(&scaled));
    assert!(a.flips > 10, "{a:?}");
    assert_eq!((a.flips, a.exits, a.long_rows), (b.flips, b.exits, b.long_rows));
    assert_eq!(a.gross.to_bits(), b.gross.to_bits());
    assert_eq!(sd_book.net_of(&a).to_bits(), sd_book.net_of(&b).to_bits());
    let score_book = FitnessBook::new(&s, &band_spec(0.5, BandUnits::Score, 4), 900.0).unwrap();
    let (c, d) = (score_book.totals(&pred), score_book.totals(&scaled));
    assert_ne!((c.flips, c.long_rows), (d.flips, d.long_rows), "a fixed score band is not scale-free");
    // The objective's book with the band resolved on the same rows.
    let at = |p: &[f64]| {
        let band = BandSpec::from_training(BandUnits::Sd, 1.0, p, &s.rows.time);
        let sp = TradingSpec { band, min_holding: 4, ..spec() };
        trading::evaluate(p, &s.ret, &s.rows, &sp).unwrap()
    };
    let (e, f) = (at(&pred), at(&scaled));
    assert_eq!((e.trades, e.exposure.to_bits()), (f.trades, f.exposure.to_bits()));
    assert_eq!(e.net_pnl.to_bits(), f.net_pnl.to_bits());
}

/// The sd-unit fitness book priced on validation rows with the member's own
/// training-resolved band equals the objective's validation book: net, trades,
/// exposure and cost.
#[test]
fn band_sd_fitness_book_matches_validation_book() {
    let train = slice(&[400, 400, 1, 400]);
    let valid = slice(&[300, 250, 300]);
    let hold = 4usize;
    let sp = band_spec(1.0, BandUnits::Sd, hold);
    let train_book = FitnessBook::new(&train, &sp, 900.0).unwrap();
    let valid_book = FitnessBook::new(&valid, &sp, 900.0).unwrap();
    // (training scale, validation scale): the validation scale differs, so a band
    // re-resolved on validation rows would price a different book.
    for (train_scale, valid_scale) in [(1.0, 1.0), (1.0, 3.0), (2.0, 1.2)] {
        let train_scores = persistent_scores(&train, train_scale);
        let valid_scores = persistent_scores(&valid, valid_scale);
        let band = BandSpec::from_training(BandUnits::Sd, 1.0, &train_scores, &train.rows.time);
        assert!(band.resolved > 0.0);
        assert_eq!(train_book.resolve_band(&train_scores).to_bits(), band.resolved.to_bits());
        let t = valid_book.band_pass(&valid_scores, band.resolved, false)[0];
        let net = valid_book.net_of(&t);
        let objective_spec = TradingSpec { band, min_holding: hold, ..spec() };
        let objective = trading::evaluate(&valid_scores, &valid.ret, &valid.rows, &objective_spec).unwrap();
        let label = format!("scales {train_scale}/{valid_scale}");
        assert!(t.flips > 5 && t.exits > 0, "{label}: {t:?}");
        assert!((net - objective.net_pnl).abs() <= 1e-12, "{label}: fitness net {net} objective {}", objective.net_pnl);
        assert_eq!(t.flips as usize, objective.trades, "{label}");
        assert_eq!(t.exits as usize, objective.exits, "{label}");
        let exposure = t.long_rows as f64 / valid.ret.len() as f64;
        assert!((exposure - objective.exposure).abs() <= 1e-12, "{label}: exposure {exposure} vs {}", objective.exposure);
        let cost = t.gross - net;
        assert!((cost - objective.total_cost).abs() <= 1e-12, "{label}: cost {cost} vs {}", objective.total_cost);
        let reresolved = valid_book.totals(&valid_scores);
        if valid_scale != train_scale {
            assert_ne!((reresolved.flips, reresolved.long_rows), (t.flips, t.long_rows), "{label}: the band is not re-resolved on validation");
        }
    }
}
#[test]
fn band_sd_degenerate_is_zero_band() {
    let s = slice(&[200, 200]);
    let n = s.ret.len();
    let constant = vec![0.3; n];
    let band = BandSpec::from_training(BandUnits::Sd, 1.0, &constant, &s.rows.time);
    assert_eq!(band.sd, None);
    assert_eq!(band.resolved, 0.0);
    assert_eq!(band.sd_rows, Some(n));
    let json = band.json_fields();
    assert!(json.contains("\"band_sd\":null") && json.contains("\"band_resolved\":0") && json.contains("\"band_sd_degenerate\":true"), "{json}");
    let mut one = vec![f64::NAN; n];
    one[7] = 1.0;
    let single = BandSpec::from_training(BandUnits::Sd, 1.0, &one, &s.rows.time);
    assert_eq!((single.sd, single.sd_rows, single.resolved), (None, Some(1), 0.0));
    assert_eq!(single.sd_max_time, Some(s.rows.time[7]));
    let book = FitnessBook::new(&s, &band_spec(1.0, BandUnits::Sd, 3), 900.0).unwrap();
    assert_eq!(book.resolve_band(&constant), 0.0);
    let pred = persistent_scores(&s, 1.0);
    let hold_only = FitnessBook::new(&s, &band_spec(0.0, BandUnits::Score, 3), 900.0).unwrap();
    assert_eq!(book.band_pass(&pred, 0.0, false), hold_only.band_pass(&pred, 0.0, false));
    let score = BandSpec::score(0.5).json_fields();
    assert!(score.contains("\"band_sd\":null") && score.contains("\"band_resolved\":0.5") && score.contains("\"band_sd_degenerate\":null"), "{score}");
}

/// Two candidate groups priced on the same training and validation rows:
/// clean candidates (score sd 0.02 around the threshold) that carry the
/// signal, and exploding ones (score sd 5) with a weaker signal. The search
/// ranks members by the training regret and the finalist is the member with
/// the best validation objective (net P&L), as in the CLI.
#[test]
fn band_units_score_and_sd_select_different_sets() {
    let rows = 1600usize;
    let mut slice_rows = TradingRows::default();
    let mut ret = Vec::with_capacity(rows);
    let mut signal = Vec::with_capacity(rows);
    let mut noise = Vec::with_capacity(rows);
    let (mut s, mut w) = (0.0f64, 0.0f64);
    for i in 0..rows {
        let g = (i >= rows / 2) as u64;
        if i == rows / 2 {
            s = 0.0;
            w = 0.0;
        }
        slice_rows.group.push(g);
        slice_rows.time.push(1.6e9 + 900.0 * (i % (rows / 2)) as f64);
        s = 0.95 * s + 0.3 * (frac(i as f64 * 0.618_033_988_749_894_9) - 0.5);
        w = 0.9 * w + 0.4 * (frac(i as f64 * 0.754_877_666_246_692_7) - 0.5);
        signal.push(s);
        noise.push(w);
        ret.push(0.004 * s + 0.0015 * w + 0.002 * (frac(i as f64 * 0.324_717_957_244_746) - 0.5));
    }
    let all = TradingSlice { rows: slice_rows, ret };
    // Training: the first 300 rows of each group; validation: the rest.
    let pick = |train: bool| {
        let idx: Vec<usize> = (0..rows).filter(|&i| ((i % (rows / 2)) < 300) == train).collect();
        let sl = TradingSlice {
            rows: TradingRows {
                time: idx.iter().map(|&i| all.rows.time[i]).collect(),
                group: idx.iter().map(|&i| all.rows.group[i]).collect(),
                funding: Vec::new(),
            },
            ret: idx.iter().map(|&i| all.ret[i]).collect(),
        };
        (idx, sl)
    };
    let (train_idx, train) = pick(true);
    let (valid_idx, valid) = pick(false);
    let sd_of = |v: &[f64]| trading::score_sd(v).sd.unwrap();
    let base_sd = sd_of(&signal);
    let wsd = sd_of(&noise);
    let mut candidates: Vec<(&str, Vec<f64>)> = Vec::new();
    for (k, shift) in [0.0, 0.1, -0.1].iter().enumerate() {
        let clean: Vec<f64> = signal.iter().map(|v| 0.02 * (v / base_sd + shift)).collect();
        candidates.push((["clean0", "clean1", "clean2"][k], clean));
        let exploding: Vec<f64> = noise.iter().map(|v| 5.0 * (v / wsd + shift)).collect();
        candidates.push((["exploding0", "exploding1", "exploding2"][k], exploding));
    }
    for (_, c) in &candidates {
        let sd = sd_of(c);
        assert!(sd > 0.015 && sd < 0.025 || sd > 4.0 && sd < 6.0, "candidate sd {sd}");
    }
    let select = |band: f64, units: BandUnits| {
        let base = band_spec(band, units, 1);
        let book = FitnessBook::new(&train, &base, 900.0).unwrap();
        let mut trading = std::collections::BTreeSet::new();
        let mut train_flips = Vec::new();
        let mut best: Option<(f64, &str)> = None;
        for (name, pred) in &candidates {
            let tp: Vec<f64> = train_idx.iter().map(|&i| pred[i]).collect();
            let vp: Vec<f64> = valid_idx.iter().map(|&i| pred[i]).collect();
            let t = book.totals(&tp);
            train_flips.push((*name, t.flips));
            let resolved = BandSpec::from_training(units, band, &tp, &train.rows.time);
            let sp = TradingSpec { band: resolved, ..base.clone() };
            let v = trading::evaluate(&vp, &valid.ret, &valid.rows, &sp).unwrap();
            if v.trades > 0 {
                trading.insert(name.to_string());
            }
            if book.regret_of(&t).is_finite() && v.trades > 0 && best.map_or(true, |(net, _)| v.net_pnl > net) {
                best = Some((v.net_pnl, name));
            }
        }
        (trading, train_flips, best.map(|(_, n)| n.to_string()))
    };
    let (score_set, score_flips, score_final) = select(0.5, BandUnits::Score);
    let (sd_set, sd_flips, sd_final) = select(1.0, BandUnits::Sd);
    for (name, flips) in &score_flips {
        if name.starts_with("clean") {
            assert_eq!(*flips, 0, "{name} trades under the score band");
        }
    }
    assert!(score_set.iter().all(|n| n.starts_with("exploding")), "{score_set:?}");
    let score_final = score_final.expect("a finalist under the score band");
    assert!(score_final.starts_with("exploding"), "{score_final}");
    for (name, flips) in &sd_flips {
        if name.starts_with("clean") {
            assert!(*flips > 0, "{name} has no trades under the sd band");
        }
    }
    assert!(sd_set.iter().any(|n| n.starts_with("clean")), "{sd_set:?}");
    assert_ne!(score_set, sd_set);
    let sd_final = sd_final.expect("a finalist under the sd band");
    assert_ne!(score_final, sd_final, "both settings select {sd_final}");
}

// -- CLI --

fn run_logged(data: &str, spec: &str, log: &std::path::Path, extra: &[&str]) -> Output {
    let log = log.to_string_lossy().into_owned();
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
        "--target-spec", spec, "--cpu", "--dev-log", &log, "--seed", "3", "--generations", "20",
        "--pop", "48", "--metric", "squared", "--json", "--quiet", "--objective", "pnl",
        "--cost-bps", "5", "--validation-frac", "0.3",
    ];
    args.extend_from_slice(extra);
    rcsr_cmd::rcsr().args(&args).output().expect("run rcsr")
}

/// The `archive.hash` of the dev log's `run_end` event.
fn archive_hash(log: &std::path::Path) -> String {
    std::fs::read_to_string(log)
        .expect("dev log")
        .lines()
        .filter_map(Json::parse)
        .find(|e| e.get("event").and_then(Json::as_str) == Some("run_end"))
        .and_then(|e| e.get("archive")?.get("hash")?.as_str().map(str::to_string))
        .expect("run_end archive hash")
}

fn without_digits(o: &Output) -> String {
    shape::timing_free_shape(&json(o))
}

#[test]
fn band_zero_hold_one_is_identity() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let plain = run(&data, &spec, &["--fitness", "pnl"]);
    let zero = run(&data, &spec, &["--fitness", "pnl", "--band", "0", "--min-holding", "1"]);
    let (a, b) = (json(&plain), json(&zero));
    assert_eq!(members(&a), members(&b));
    assert_eq!(at(&a, &["search_fitness"]).get("book_hash"), at(&b, &["search_fitness"]).get("book_hash"));
    assert_eq!(without_digits(&plain), without_digits(&zero));
    assert_eq!(num(&b, &["search_fitness", "band"]), 0.0);
    assert_eq!(num(&b, &["search_fitness", "min_holding"]), 1.0);
    assert_eq!(at(&b, &["search_fitness", "band_units"]).as_str(), Some("score"));
    let (la, lb) = (dir.join("plain.jsonl"), dir.join("zero.jsonl"));
    assert_eq!(code(&run_logged(&data, &spec, &la, &["--fitness", "pnl"])), 0);
    assert_eq!(code(&run_logged(&data, &spec, &lb, &["--fitness", "pnl", "--band", "0", "--min-holding", "1"])), 0);
    assert_eq!(archive_hash(&la), archive_hash(&lb));
}

#[test]
fn band_units_score_is_default() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let args = ["--fitness", "pnl", "--band", "0.004", "--min-holding", "3"];
    let implicit = run(&data, &spec, &args);
    let explicit = run(&data, &spec, &[&args[..], &["--band-units", "score"]].concat());
    let (a, b) = (json(&implicit), json(&explicit));
    assert_eq!(members(&a), members(&b));
    assert_eq!(at(&a, &["search_fitness"]).get("book_hash"), at(&b, &["search_fitness"]).get("book_hash"));
    assert_eq!(without_digits(&implicit), without_digits(&explicit));
    assert_eq!(at(&b, &["objective", "spec", "band_units"]).as_str(), Some("score"));
    assert_eq!(num(&b, &["objective", "spec", "band_resolved"]), 0.004);
    let (la, lb) = (dir.join("implicit.jsonl"), dir.join("explicit.jsonl"));
    assert_eq!(code(&run_logged(&data, &spec, &la, &args)), 0);
    assert_eq!(code(&run_logged(&data, &spec, &lb, &[&args[..], &["--band-units", "score"]].concat())), 0);
    assert_eq!(archive_hash(&la), archive_hash(&lb));
}

#[test]
fn resume_refuses_changed_band_units() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let cp = dir.join("band.rcsr").to_string_lossy().into_owned();
    let saved = ["--fitness", "pnl", "--band", "0.5", "--min-holding", "2"];
    let first = run(&data, &spec, &[&saved[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let resume = |extra: &[&str]| run(&data, &spec, &[&["--fitness", "pnl", "--generations", "20", "--resume", &cp][..], extra].concat());
    usage_error(&resume(&["--band", "0.5", "--min-holding", "2", "--band-units", "sd"]), "fitness");
    usage_error(&resume(&["--band", "0.4", "--min-holding", "2"]), "fitness");
    usage_error(&resume(&["--band", "0.5", "--min-holding", "3"]), "fitness");
    usage_error(&resume(&[]), "fitness");
    let same = resume(&["--band", "0.5", "--min-holding", "2"]);
    assert_eq!(code(&same), 0, "{}", stderr(&same));
}

const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// One FNV-1a step over the eight little-endian bytes of `v`, as the book hash.
fn fnv_u64(mut h: u64, v: u64) -> u64 {
    for b in v.to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// The hash state before [`fnv_u64`]`(h, v)`: the prime is odd, so it has an
/// inverse modulo 2^64.
fn unfnv_u64(mut h: u64, v: u64) -> u64 {
    let mut inv = FNV_PRIME;
    for _ in 0..5 {
        inv = inv.wrapping_mul(2u64.wrapping_sub(FNV_PRIME.wrapping_mul(inv)));
    }
    assert_eq!(inv.wrapping_mul(FNV_PRIME), 1);
    for b in v.to_le_bytes().iter().rev() {
        h = h.wrapping_mul(inv) ^ *b as u64;
    }
    h
}

/// The book hash in a checkpoint's `fitness long_pnl:<hash>` field.
fn saved_book_hash(cp: &str) -> u64 {
    let text = std::fs::read_to_string(cp).unwrap();
    let at = text.find("long_pnl:").expect("a long_pnl fitness checkpoint") + "long_pnl:".len();
    u64::from_str_radix(&text[at..at + 16], 16).unwrap()
}

/// #945: the one-pass sd moves an sd-unit book's rows in the last bits, so its
/// book hash carries the sd algorithm. The #925 hash of the same book is the
/// score-unit hash with the units word 1 instead of 0; a checkpoint holding it
/// refuses to resume (exit 2).
#[test]
fn resume_refuses_a_two_pass_sd_checkpoint() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let save = |name: &str, units: &str| {
        let cp = dir.join(name).to_string_lossy().into_owned();
        let args = ["--fitness", "pnl", "--band", "0.5", "--band-units", units, "--min-holding", "2", "--generations", "10", "--save", &cp];
        let out = run(&data, &spec, &args);
        assert_eq!(code(&out), 0, "{}", stderr(&out));
        cp
    };
    let sd_cp = save("sd.rcsr", "sd");
    let score_cp = save("score.rcsr", "score");
    let (sd_hash, score_hash) = (saved_book_hash(&sd_cp), saved_book_hash(&score_cp));
    let band_prefix = unfnv_u64(unfnv_u64(score_hash, 2), 0);
    let two_pass = fnv_u64(fnv_u64(band_prefix, 1), 2);
    assert_ne!(two_pass, sd_hash, "the sd-unit book hash still equals the #925 hash");
    assert_eq!(fnv_u64(two_pass, rcsr::trading::SCORE_SD_ALGORITHM), sd_hash, "the sd-algorithm word is the last one hashed");
    // The checkpoint a #925 binary writes: its book hash, and the config hash
    // over the config lines that hold it.
    let old_cp = dir.join("two-pass.rcsr");
    let text = std::fs::read_to_string(&sd_cp).unwrap().replace(&format!("long_pnl:{sd_hash:016x}"), &format!("long_pnl:{two_pass:016x}"));
    let block: String = text.lines().filter(|l| l.starts_with("config ")).map(|l| format!("{l}\n")).collect();
    let config_hash = format!("config_hash {:016x}", rcsr::checkpoint::hash_config_text(&block));
    let old: String = text.lines().map(|l| if l.starts_with("config_hash ") { format!("{config_hash}\n") } else { format!("{l}\n") }).collect();
    std::fs::write(&old_cp, old).unwrap();
    let resume = |cp: &str| {
        let args = ["--fitness", "pnl", "--band", "0.5", "--band-units", "sd", "--min-holding", "2", "--generations", "20", "--resume", cp];
        run(&data, &spec, &args)
    };
    usage_error(&resume(&old_cp.to_string_lossy()), "fitness");
    let same = resume(&sd_cp);
    assert_eq!(code(&same), 0, "{}", stderr(&same));
}

/// The rows of `split.validation_time` as `(time, x1, x2, simple return)`.
fn csv_rows(path: &str) -> Vec<(f64, f64, f64, f64)> {
    std::fs::read_to_string(path)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| {
            let v: Vec<f64> = l.split(',').map(|c| c.parse().unwrap()).collect();
            (v[0], v[1], v[2], v[3].exp_m1())
        })
        .collect()
}

fn time_range(j: &Json, key: &str) -> (f64, f64) {
    match at(j, &["split", key]) {
        Json::Arr(a) => (a[0].as_f64().unwrap(), a[1].as_f64().unwrap()),
        _ => panic!("split.{key} is not a range"),
    }
}

/// A persistent fast signal: `x1` is AR(1) with coefficient 0.95 and unit
/// variance, the realised return has correlation 0.05 with it, `x2` is
/// independent noise of the same kind.
fn write_fast_csv(dir: &std::path::Path, rows: usize) -> String {
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut unit = || {
        state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    let mut normal = || {
        let (u, v) = (unit(), unit());
        (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
    };
    let (phi, rho, sd) = (0.95f64, 0.05f64, 0.003f64);
    let (mut x1, mut x2) = (0.0f64, 0.0f64);
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..rows {
        x1 = phi * x1 + (1.0 - phi * phi).sqrt() * normal();
        x2 = phi * x2 + (1.0 - phi * phi).sqrt() * normal();
        let r = sd * (rho * x1 + (1.0 - rho * rho).sqrt() * normal());
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r, r));
    }
    let path = dir.join("fast.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn band_fitness_rescues_fast_signal() {
    let (_scratch, dir) = scratch();
    let data = write_fast_csv(&dir, 120_000);
    let spec = write_spec(&dir);
    let run_json = json(&run(
        &data,
        &spec,
        &["--fitness", "pnl", "--band", "0.5", "--min-holding", "12", "--cost-bps", "10", "--ops", "add,sub,mul,tanh", "--max-complexity", "7"],
    ));
    let best = pareto(&run_json)
        .iter()
        .min_by(|a, b| num(a, &["fitness", "train_regret"]).total_cmp(&num(b, &["fitness", "train_regret"])))
        .unwrap();
    let exact = at(best, &["expression_exact"]).as_str().unwrap();
    let tree = rcsr::parse::parse(exact, &["x1".to_string(), "x2".to_string()]).unwrap();
    let (lo, hi) = time_range(&run_json, "validation_time");
    let rows: Vec<_> = csv_rows(&data).into_iter().filter(|r| r.0 >= lo && r.0 <= hi).collect();
    let pred: Vec<f64> = rows.iter().map(|r| rcsr::expr::eval(&tree, &[r.1, r.2])).collect();
    let ret: Vec<f64> = rows.iter().map(|r| r.3).collect();
    let book_rows = TradingRows { time: rows.iter().map(|r| r.0).collect(), group: vec![0; rows.len()], funding: Vec::new() };
    let priced = |band: f64, hold: usize| {
        let sp = TradingSpec { cost_bps: 10.0, slippage_bps: 0.0, min_trades: 1, ..band_spec(band, BandUnits::Score, hold) };
        trading::evaluate(&pred, &ret, &book_rows, &sp).unwrap()
    };
    let banded = priced(0.5, 12);
    let plain = priced(0.0, 1);
    let reported = num(best, &["validation_book", "net_pnl"]);
    assert!((banded.net_pnl - reported).abs() <= 1e-9 * reported.abs().max(1e-6), "recomputed {} reported {reported}", banded.net_pnl);
    let turnover = banded.trades as f64 / rows.len() as f64;
    assert!(banded.net_pnl > 0.0, "{exact}: band validation net {}", banded.net_pnl);
    assert!(turnover < 0.05, "{exact}: turnover per row {turnover}");
    assert!(plain.net_pnl < 0.0, "{exact}: plain validation net {}", plain.net_pnl);
}

fn write_extreme_csv(src: &str, dst: &std::path::Path, from: f64, scale: f64, shift: f64) -> String {
    let text = std::fs::read_to_string(src).unwrap();
    let mut out = String::new();
    for (k, line) in text.lines().enumerate() {
        if k == 0 {
            out.push_str(line);
            out.push('\n');
            continue;
        }
        let v: Vec<f64> = line.split(',').map(|c| c.parse().unwrap()).collect();
        if v[0] >= from {
            let r = v[3] * scale;
            out.push_str(&format!("{},{},{},{},{}\n", v[0], v[1] * scale + shift, v[2] * scale - shift, r, r + 0.05 * v[2] * scale));
        } else {
            out.push_str(line);
            out.push('\n');
        }
    }
    std::fs::write(dst, out).unwrap();
    dst.to_string_lossy().into_owned()
}

const SD_ARGS: [&str; 6] = ["--fitness", "pnl", "--band", "1.0", "--band-units", "sd"];

#[test]
fn band_sd_ignores_rows_after_training_end() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let (la, lb) = (dir.join("a.jsonl"), dir.join("b.jsonl"));
    let a_out = run_logged(&data, &spec, &la, &SD_ARGS);
    let a = json(&a_out);
    let (_, train_end) = time_range(&a, "train_time");
    let (validation_start, _) = time_range(&a, "validation_time");
    let other = write_extreme_csv(&data, &dir.join("extreme.csv"), train_end + 1.0, 40.0, 7.0);
    let b = json(&run_logged(&other, &spec, &lb, &SD_ARGS));
    // Every member's band is resolved on its training rows: the same in both runs.
    let bands = |j: &Json| {
        pareto(j)
            .iter()
            .map(|m| {
                let v = at(m, &["validation_book"]);
                (
                    at(m, &["expression_exact"]).as_str().unwrap().to_string(),
                    v.get("band_sd").and_then(Json::as_f64).map(f64::to_bits),
                    num(v, &["band_resolved"]).to_bits(),
                )
            })
            .collect::<Vec<_>>()
    };
    // The display order follows the validation books, which differ; compare by expression.
    let sorted = |mut v: Vec<(String, Option<u64>, u64)>| {
        v.sort();
        v
    };
    let (ba, bb) = (sorted(bands(&a)), sorted(bands(&b)));
    assert!(ba.iter().filter(|m| m.1.is_some()).count() >= 2, "{ba:?}");
    assert_eq!(ba, bb);
    let (mut ma, mut mb) = (members(&a), members(&b));
    ma.sort();
    mb.sort();
    assert_eq!(ma, mb);
    assert!(at(&a, &["objective", "spec", "band_sd"]).as_f64().is_some());
    assert_eq!(archive_hash(&la), archive_hash(&lb));
    let max_time = num(&a, &["objective", "spec", "band_sd_max_time"]);
    assert!(max_time <= train_end && max_time < validation_start, "band_sd_max_time {max_time}, training end {train_end}");
    assert_ne!(
        num(&a, &["objective", "validation", "net_pnl"]).to_bits(),
        num(&b, &["objective", "validation", "net_pnl"]).to_bits(),
        "the validation rows differ"
    );
}

#[test]
fn band_sd_not_recomputed_on_test() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let a = json(&run(&data, &spec, &SD_ARGS));
    let (_, validation_end) = time_range(&a, "validation_time");
    let wide = write_extreme_csv(&data, &dir.join("wide.csv"), validation_end + 1.0, 100.0, 0.0);
    let b = json(&run(&wide, &spec, &SD_ARGS));
    assert_eq!(at(&a, &["objective", "finalist", "expression_exact"]), at(&b, &["objective", "finalist", "expression_exact"]));
    for k in ["band_sd", "band_resolved"] {
        assert_eq!(num(&a, &["objective", "spec", k]).to_bits(), num(&b, &["objective", "spec", k]).to_bits(), "{k}");
    }
    assert_ne!(
        num(&a, &["objective", "test", "net_pnl"]).to_bits(),
        num(&b, &["objective", "test", "net_pnl"]).to_bits(),
        "the test rows differ"
    );
}

#[test]
fn band_sd_recorded_in_run_json() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let j = json(&run(&data, &spec, &[&SD_ARGS[..], &["--min-holding", "2"]].concat()));
    assert_eq!(at(&j, &["search_fitness", "band_units"]).as_str(), Some("sd"));
    assert_eq!(num(&j, &["search_fitness", "band"]), 1.0);
    assert_eq!(num(&j, &["search_fitness", "min_holding"]), 2.0);
    let s = at(&j, &["objective", "spec"]);
    assert_eq!(s.get("band_units").and_then(Json::as_str), Some("sd"));
    assert_eq!(num(s, &["band"]), 1.0);
    assert_eq!(num(s, &["min_holding"]), 2.0);
    let sd = num(s, &["band_sd"]);
    assert!(sd.is_finite() && sd > 0.0, "band_sd {sd}");
    assert_eq!(num(s, &["band_resolved"]).to_bits(), (1.0 * sd).to_bits());
    let rows = num(s, &["band_sd_rows"]);
    assert!(rows >= 2.0 && rows <= num(&j, &["split", "train_rows"]), "band_sd_rows {rows}");
    let (_, train_end) = time_range(&j, "train_time");
    assert!(num(s, &["band_sd_max_time"]) <= train_end);
    assert_eq!(s.get("band_sd_degenerate"), Some(&Json::Bool(false)));
}

/// Robust finalist selection prices each member with its own sd band: its
/// trade count on the validation rows is the member's validation book count.
#[test]
fn band_sd_robust_selection_uses_member_bands() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let j = json(&run(&data, &spec, &[&SD_ARGS[..], &["--select-rule", "robust"]].concat()));
    assert_eq!(at(&j, &["objective", "selection_rule", "rule"]).as_str(), Some("robust"));
    let robust = match at(&j, &["objective", "selection_rule", "members"]) {
        Json::Arr(a) => a,
        _ => panic!("selection_rule.members is not an array"),
    };
    let mut compared = 0usize;
    for r in robust {
        let rank = num(r, &["archive_rank"]);
        let member = pareto(&j).iter().find(|m| num(m, &["archive_rank"]) == rank).expect("a front member per rank");
        let book = at(member, &["validation_book"]);
        if book.get("objective").and_then(Json::as_f64).is_some_and(f64::is_finite) && num(r, &["trades"]) > 0.0 {
            assert_eq!(num(r, &["trades"]), num(book, &["trades"]), "archive rank {rank}");
            assert!(book.get("band_sd").and_then(Json::as_f64).is_some());
            compared += 1;
        }
    }
    assert!(compared >= 2, "{compared} members compared");
    let finalist = at(&j, &["objective", "finalist", "expression_exact"]).as_str().unwrap().to_string();
    let member = pareto(&j)
        .iter()
        .find(|m| at(m, &["expression_exact"]).as_str() == Some(finalist.as_str()))
        .expect("the finalist is a front member");
    assert_eq!(
        num(at(member, &["validation_book"]), &["band_resolved"]).to_bits(),
        num(&j, &["objective", "spec", "band_resolved"]).to_bits()
    );
}

// ---- #1084: book-space QD descriptor ----

const BOOK_QD: [&str; 5] = ["--fitness", "pnl", "--qd-policy", "--qd-descriptor", "book"];

fn front_key(j: &Json) -> Vec<(String, u64)> {
    pareto(j)
        .iter()
        .map(|m| (at(m, &["expression"]).as_str().unwrap().to_string(), num(m, &["fitness", "train_regret"]).to_bits()))
        .collect()
}

#[test]
fn qd_book_descriptor_cpu_run_reports_book_without_fallbacks() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let book = json(&run(&data, &spec, &BOOK_QD));
    let q = at(&book, &["qd"]);
    assert_eq!(q.get("descriptor").and_then(Json::as_str), Some("book"));
    assert_eq!(num(q, &["descriptor_fallbacks"]), 0.0);
    assert!(num(q, &["occupied"]) > 1.0, "{q:?}");
    assert_eq!(num(q, &["reachable"]), num(q, &["cells"]));
    let moments = json(&run(&data, &spec, &["--fitness", "pnl", "--qd-policy"]));
    assert!(at(&moments, &["qd"]).get("descriptor").is_none(), "the moments report keeps its keys");
    assert_ne!(at(&moments, &["qd"]), q, "the two kinds key the archive differently");
}

#[test]
fn qd_book_descriptor_is_deterministic_and_thread_independent() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let with = |threads: &str| {
        let mut args = BOOK_QD.to_vec();
        args.extend(["--threads", threads]);
        json(&run(&data, &spec, &args))
    };
    let (one, again, four) = (with("1"), with("1"), with("4"));
    for other in [&again, &four] {
        assert_eq!(at(&one, &["qd"]), at(other, &["qd"]));
        assert_eq!(front_key(&one), front_key(other));
        assert_eq!(
            num(&one, &["search", "candidates_evaluated"]),
            num(other, &["search", "candidates_evaluated"])
        );
    }
}

#[test]
fn a_passive_book_descriptor_keeps_the_front() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let plain = json(&run(&data, &spec, &["--fitness", "pnl"]));
    let passive = json(&run(&data, &spec, &["--fitness", "pnl", "--qd", "--qd-descriptor", "book"]));
    assert_eq!(front_key(&plain), front_key(&passive));
    assert_eq!(at(&passive, &["qd", "descriptor"]).as_str(), Some("book"));
    assert_eq!(num(&passive, &["qd", "descriptor_fallbacks"]), 0.0);
}

#[test]
fn qd_book_descriptor_refusals_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    usage_error(&run(&data, &spec, &["--qd", "--qd-descriptor", "book"]), "--qd-descriptor book needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--qd-descriptor", "book"]), "--qd-descriptor book needs a QD archive");
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--qd", "--qd-descriptor", "usage"]),
        "--qd-descriptor must be moments or book",
    );
}

#[test]
fn qd_book_descriptor_is_saved_and_resumes() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let cp = dir.join("book.rcsr").to_string_lossy().into_owned();
    let passive = ["--fitness", "pnl", "--qd", "--qd-descriptor", "book"];
    let mut first = passive.to_vec();
    first.extend(["--generations", "10", "--save", &cp]);
    let saved = run(&data, &spec, &first);
    assert_eq!(code(&saved), 0, "{}", stderr(&saved));
    assert!(std::fs::read_to_string(&cp).unwrap().contains("\nconfig qd_descriptor book\n"));

    let straight = json(&run(&data, &spec, &passive));
    let resumed = json(&run(&data, &spec, &["--fitness", "pnl", "--resume", &cp]));
    assert_eq!(at(&resumed, &["qd", "descriptor"]).as_str(), Some("book"));
    assert_eq!(front_key(&straight), front_key(&resumed));
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--resume", &cp, "--qd-descriptor", "moments"]),
        "--qd-descriptor moments (saved qd_descriptor is book)",
    );

    let moments_cp = dir.join("moments.rcsr").to_string_lossy().into_owned();
    let m = run(&data, &spec, &["--fitness", "pnl", "--qd", "--generations", "2", "--save", &moments_cp]);
    assert_eq!(code(&m), 0, "{}", stderr(&m));
    assert!(!std::fs::read_to_string(&moments_cp).unwrap().contains("qd_descriptor"));
}

/// #1023: a two-symbol panel with four time blocks gives eight sub-book cases;
/// every generation of `--selection lexicase` selects on all of them, and the
/// run resumes to the uninterrupted front.
#[test]
fn lexicase_selects_on_two_symbols_times_four_blocks() {
    let (_scratch, dir) = scratch();
    let mut text = String::from("t,sym,x1,x2,r,y\n");
    for s in 0..2 {
        for i in 0..300 {
            let f = (i + 1000 * s) as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let r = 0.01 * x1 + 0.0005 * noise;
            text.push_str(&format!("{},{},{},{},{},{}\n", T0 + 900 * i as i64, ["a", "b"][s], x1, x2, r, r + 0.05 * x2));
        }
    }
    let data = dir.join("panel.csv");
    std::fs::write(&data, text).unwrap();
    let data = data.to_string_lossy().into_owned();
    let spec = write_spec(&dir);
    let log = dir.join("dev.ndjson").to_string_lossy().into_owned();
    let cp = dir.join("lex.rcsr").to_string_lossy().into_owned();
    let lex = |extra: &[&str]| {
        let mut args = vec![
            "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "sym", "--split",
            "temporal", "--target-spec", &spec, "--cpu", "--seed", "3", "--pop", "48", "--metric", "squared",
            "--json", "--quiet", "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--fitness",
            "pnl", "--pnl-blocks", "4", "--selection", "lexicase",
        ];
        args.extend_from_slice(extra);
        rcsr_cmd::rcsr().args(&args).output().expect("run rcsr")
    };
    let full = json(&lex(&["--generations", "12", "--dev-log", &log]));
    assert_eq!(num(at(&full, &["search_fitness"]), &["blocks"]), 4.0);
    let cases: Vec<f64> = std::fs::read_to_string(&log)
        .unwrap()
        .lines()
        .filter(|l| l.contains("\"event\":\"generation\""))
        .map(|l| num(&Json::parse(l).expect("dev-log line"), &["selection", "cases"]))
        .collect();
    assert!(!cases.is_empty() && cases.iter().all(|&c| c == 8.0), "{cases:?}");

    let saved = lex(&["--generations", "6", "--no-dev-log", "--save", &cp]);
    assert_eq!(code(&saved), 0, "{}", stderr(&saved));
    let resumed = json(&lex(&["--generations", "12", "--no-dev-log", "--resume", &cp]));
    // A resumed run reports the front rebuilt on every row, which takes the
    // canonical tree among equal regrets; compare the regrets.
    let regrets = |j: &Json| pareto(j).iter().map(|m| num(m, &["fitness", "train_regret"]).to_bits()).collect::<Vec<_>>();
    assert_eq!(regrets(&full), regrets(&resumed));
    usage_error(&lex(&["--generations", "12", "--no-dev-log", "--resume", &cp, "--lexicase-cases", "4"]), "--lexicase-cases 4 (saved lexicase_cases is 256)");
}

/// #1166: a `--qd-policy --qd-descriptor book` run resumed at generation 4
/// equals the 8-generation run: front, QD report and exit checkpoint apart
/// from the `build` lines.
#[test]
fn a_qd_book_policy_run_and_its_resume_match_exactly() {
    let (_scratch, dir) = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);
    let path = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let (straight_cp, half_cp, resumed_cp) = (path("straight.rcsr"), path("half.rcsr"), path("resumed.rcsr"));
    let mut straight = BOOK_QD.to_vec();
    straight.extend(["--generations", "8", "--save", &straight_cp]);
    let a = json(&run(&data, &spec, &straight));
    let mut half = BOOK_QD.to_vec();
    half.extend(["--generations", "4", "--save", &half_cp]);
    json(&run(&data, &spec, &half));
    let b = json(&run(&data, &spec, &["--fitness", "pnl", "--generations", "8", "--resume", &half_cp, "--save", &resumed_cp]));
    assert_eq!(at(&a, &["qd", "descriptor"]).as_str(), Some("book"));
    assert_eq!(at(&a, &["pareto"]), at(&b, &["pareto"]));
    assert_eq!(at(&a, &["qd"]), at(&b, &["qd"]));
    let read = |p: &str| -> String {
        std::fs::read_to_string(p).unwrap().lines().filter(|l| !l.starts_with("build ")).map(|l| format!("{l}\n")).collect()
    };
    assert!(read(&half_cp).contains("\nqd_archive 1 cvt "), "the book checkpoint carries the archive");
    assert_eq!(read(&straight_cp), read(&resumed_cp));
}

// ---- #1180: regime-masked fitness ----

use trading::BookMask;

/// On in alternate `run`-row runs, counted from the first row of each group.
fn alternate_mask(s: &TradingSlice, run: usize) -> Vec<bool> {
    let mut out = Vec::new();
    let mut at = 0usize;
    for i in 0..s.ret.len() {
        if i > 0 && s.rows.group[i] != s.rows.group[i - 1] {
            at = 0;
        }
        out.push((at / run) % 2 == 0);
        at += 1;
    }
    out
}

fn masked(s: &TradingSlice, sp: &TradingSpec, blocks: usize, on: &[bool]) -> Result<FitnessBook, String> {
    FitnessBook::with_segments(s, sp, 900.0, blocks, 1.0, &[], Some(BookMask { on, identity: "test-gate" }))
}

/// The maximal on-runs as `(group, start, end)` row ranges.
fn on_runs(s: &TradingSlice, on: &[bool]) -> Vec<(usize, usize)> {
    let mut runs = Vec::new();
    let mut start: Option<usize> = None;
    for i in 0..=on.len() {
        let live = i < on.len() && on[i] && (i == 0 || s.rows.group[i] == s.rows.group[i - 1] || start.is_none());
        let seam = i > 0 && i < on.len() && s.rows.group[i] != s.rows.group[i - 1];
        if let Some(a) = start {
            if !live || seam {
                runs.push((a, i));
                start = None;
            }
        }
        if i < on.len() && on[i] && start.is_none() {
            start = Some(i);
        }
    }
    runs
}

fn sub_slice(s: &TradingSlice, a: usize, b: usize) -> TradingSlice {
    TradingSlice {
        rows: TradingRows { time: s.rows.time[a..b].to_vec(), group: vec![0; b - a], funding: Vec::new() },
        ret: s.ret[a..b].to_vec(),
    }
}

#[test]
fn all_on_mask_book_is_the_plain_book() {
    let s = slice(&[300, 280]);
    let all = vec![true; s.ret.len()];
    let pred = prediction(&s);
    let specs = [spec(), TradingSpec { min_holding: 3, band: trading::BandSpec::score(0.1), ..spec() }];
    for sp in &specs {
        for blocks in [1usize, 3] {
            let plain = FitnessBook::with_blocks(&s, sp, 900.0, blocks, 1.0).unwrap();
            let all_on = masked(&s, sp, blocks, &all).unwrap();
            assert!(!all_on.is_masked());
            assert_eq!(plain.hash(), all_on.hash());
            assert_eq!(plain.flags(), all_on.flags());
            assert_eq!(plain.bound().to_bits(), all_on.bound().to_bits());
            assert_eq!(plain.regret(&pred).to_bits(), all_on.regret(&pred).to_bits());
        }
    }
}

#[test]
fn mask_runs_are_book_segments() {
    let s = slice(&[400, 400]);
    let pred = prediction(&s);
    let on = alternate_mask(&s, 50);
    let runs = on_runs(&s, &on);
    assert_eq!(runs.len(), 8);
    let specs = [spec(), TradingSpec { min_holding: 3, band: trading::BandSpec::score(0.1), ..spec() }];
    for sp in &specs {
        let book = masked(&s, sp, 1, &on).unwrap();
        assert!(book.is_masked());
        assert_eq!(book.masked_rows(), on.iter().filter(|&&o| !o).count());
        let (mut net, mut bound, mut flips, mut exits, mut long_rows) = (0.0f64, 0.0f64, 0u64, 0u64, 0u64);
        for &(a, b) in &runs {
            let part = FitnessBook::new(&sub_slice(&s, a, b), sp, 900.0).unwrap();
            let t = part.totals(&pred[a..b]);
            net += part.net_of(&t);
            bound += part.bound();
            flips += t.flips;
            exits += t.exits;
            long_rows += t.long_rows;
        }
        let t = book.totals(&pred);
        assert_eq!((t.flips, t.exits, t.long_rows), (flips, exits, long_rows));
        let masked_net = book.net_of(&t);
        assert!((masked_net - net).abs() <= 1e-12, "{masked_net} vs {net}");
        assert!((book.bound() - bound).abs() <= 1e-12, "{} vs {bound}", book.bound());
        let regret = (bound - net) / bound;
        assert!((book.regret(&pred) - regret).abs() <= 1e-12, "{} vs {regret}", book.regret(&pred));
    }
}

#[test]
fn masked_book_is_the_plain_book_with_the_rows_outside_forced_flat() {
    let s = slice(&[400, 400]);
    let sp = spec();
    let on = alternate_mask(&s, 50);
    let book = masked(&s, &sp, 1, &on).unwrap();
    let plain = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let pred = prediction(&s);
    let forced: Vec<f64> = pred.iter().zip(&on).map(|(&p, &o)| if o { p } else { -1.0 }).collect();
    let (m, p) = (book.totals(&pred), plain.totals(&forced));
    assert_eq!(m.gross.to_bits(), p.gross.to_bits());
    assert_eq!(m.flips + m.exits, p.flips + p.exits, "a seam exit replaces the flip to flat");
    assert_eq!(book.net_of(&m).to_bits(), plain.net_of(&p).to_bits());
    assert_eq!(m.long_rows, p.long_rows);
}

#[test]
fn masked_bound_matches_brute_force() {
    let s = slice(&[24, 24]);
    let sp = spec();
    let side = sp.per_side_bps() * 1e-4;
    let mut on = vec![false; 48];
    for i in (5..13).chain(18..24).chain(24..31).chain(36..44) {
        on[i] = true;
    }
    let book = masked(&s, &sp, 1, &on).unwrap();
    let runs = on_runs(&s, &on);
    assert_eq!(runs.len(), 4);
    let mut want = 0.0f64;
    for &(a, b) in &runs {
        let m = b - a;
        let mut best = f64::NEG_INFINITY;
        for bits in 0u32..(1 << m) {
            let mut net = 0.0;
            let mut prev = 0.0;
            for k in 0..m {
                let p = ((bits >> k) & 1) as f64;
                net += p * s.ret[a + k] - side * (p - prev).abs();
                prev = p;
            }
            net -= side * prev;
            best = best.max(net);
        }
        want += best;
    }
    assert!((book.bound() - want).abs() <= 1e-12, "{} vs {want}", book.bound());
}

#[test]
fn a_mask_changes_the_hash_and_refuses_what_it_cannot_price() {
    let s = slice(&[300]);
    let sp = spec();
    let on = alternate_mask(&s, 50);
    let plain = FitnessBook::new(&s, &sp, 900.0).unwrap();
    let a = FitnessBook::with_segments(&s, &sp, 900.0, 1, 1.0, &[], Some(BookMask { on: &on, identity: "a" })).unwrap();
    let b = FitnessBook::with_segments(&s, &sp, 900.0, 1, 1.0, &[], Some(BookMask { on: &on, identity: "b" })).unwrap();
    assert_ne!(a.hash(), plain.hash());
    assert_ne!(a.hash(), b.hash());
    assert!(masked(&s, &sp, 1, &on[1..]).unwrap_err().contains("regime mask has"));
    let none = vec![false; 300];
    assert!(masked(&s, &sp, 1, &none).unwrap_err().contains("masked"));
    let first_third: Vec<bool> = (0..300).map(|i| i < 100).collect();
    let e = masked(&s, &sp, 3, &first_third).unwrap_err();
    assert!(e.contains("time block 2 of 3") && e.contains("regime mask"), "{e}");
}

#[test]
fn a_non_finite_score_outside_the_mask_is_ignored() {
    let s = slice(&[400, 400]);
    let sp = spec();
    let on = alternate_mask(&s, 50);
    let book = masked(&s, &sp, 1, &on).unwrap();
    let pred = prediction(&s);
    let off = on.iter().position(|&o| !o).unwrap();
    let on_row = on.iter().position(|&o| o).unwrap();
    let mut bad_off = pred.clone();
    bad_off[off] = f64::NAN;
    assert_eq!(book.regret(&bad_off).to_bits(), book.regret(&pred).to_bits());
    let mut bad_on = pred.clone();
    bad_on[on_row] = f64::NAN;
    assert_eq!(book.regret(&bad_on), f64::INFINITY);
}

#[test]
fn masked_cases_hold_on_rows_only_and_the_sd_band_reads_them() {
    let s = slice(&[400, 400]);
    let on = alternate_mask(&s, 50);
    let sp = band_spec(0.5, trading::BandUnits::Sd, 1);
    let book = masked(&s, &sp, 2, &on).unwrap();
    let cases = book.cases();
    assert!(!cases.kept.is_empty());
    for &c in &cases.kept {
        let (a, b) = cases.ranges[c];
        assert!(on[a..b].iter().all(|&o| o), "case {c} holds an off row");
    }
    let pred = prediction(&s);
    assert_eq!(book.case_errors(&pred).len(), cases.kept.len());
    let on_scores: Vec<f64> = pred.iter().zip(&on).filter(|(_, &o)| o).map(|(&p, _)| p).collect();
    let want = 0.5 * trading::score_sd(&on_scores).sd.unwrap();
    assert_eq!(book.resolve_band(&pred).to_bits(), want.to_bits());
}

#[test]
fn masked_fitness_is_thread_independent() {
    let s = slice(&[400, 400]);
    let on = alternate_mask(&s, 50);
    let book = masked(&s, &spec(), 4, &on).unwrap();
    let pred = prediction(&s);
    let want = book.regret(&pred).to_bits();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4).map(|_| scope.spawn(|| book.regret(&pred).to_bits())).collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), want);
        }
    });
}

/// `g` is +1 in alternate 60-row runs and -1 between them; `lbl_h24a_y` is a
/// label-family column that no gate may read.
fn write_gate_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,x1,x2,r,y,g,lbl_h24a_y\n");
    for i in 0..N {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        let g = if (i / 60) % 2 == 0 { 1 } else { -1 };
        text.push_str(&format!("{},{},{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r, r + 0.05 * x2, g, r));
    }
    let path = dir.join("gate.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn all_on_mask_is_the_plain_run() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let plain = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "3"]);
    let all_on = run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "3", "--pnl-mask", "1"]);
    let (a, b) = (json(&plain), json(&all_on));
    assert_eq!(members(&a), members(&b));
    assert_eq!(at(&a, &["search_fitness"]).get("book_hash"), at(&b, &["search_fitness"]).get("book_hash"));
    assert!(at(&b, &["search_fitness"]).get("mask").is_none());
    assert_eq!(at(&a, &["objective", "validation"]).get("net_pnl"), at(&b, &["objective", "validation"]).get("net_pnl"));
    assert_eq!(at(&a, &["objective", "test"]).get("net_pnl"), at(&b, &["objective", "test"]).get("net_pnl"));
    let (la, lb) = (dir.join("plain.jsonl"), dir.join("mask1.jsonl"));
    assert_eq!(code(&run_logged(&data, &spec, &la, &["--fitness", "pnl", "--pnl-blocks", "3"])), 0);
    assert_eq!(code(&run_logged(&data, &spec, &lb, &["--fitness", "pnl", "--pnl-blocks", "3", "--pnl-mask", "1"])), 0);
    assert_eq!(archive_hash(&la), archive_hash(&lb));
}

#[test]
fn mask_refuses_label_columns() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    for col in ["y", "r", "lbl_h24a_y", "x1 + lbl_h24a_y"] {
        let o = run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", col]);
        assert_eq!(code(&o), 5, "{col}: {}", stderr(&o));
        assert!(stderr(&o).contains("D13") && stderr(&o).contains("--pnl-mask"), "{}", stderr(&o));
    }
    let ok = run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "g"]);
    assert_eq!(code(&ok), 0, "{}", stderr(&ok));
}

#[test]
fn mask_flags_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    usage_error(&run(&data, &spec, &["--pnl-mask", "g"]), "--pnl-mask needs --fitness pnl");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask-band", "0.1"]), "--pnl-mask-band needs --pnl-mask");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask-dwell", "3"]), "--pnl-mask-dwell needs --pnl-mask");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "g", "--pnl-mask-band", "-1"]), "--pnl-mask-band must be");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "g", "--pnl-mask-dwell", "0"]), "--pnl-mask-dwell must be");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "mean(g)"]), "aggregate");
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "g +"]), "--pnl-mask");
    let missing = run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "nope"]);
    assert_eq!(code(&missing), 3, "{}", stderr(&missing));
    assert!(stderr(&missing).contains("mask column") && stderr(&missing).contains("nope"), "{}", stderr(&missing));
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "g", "--band", "0.5", "--band-units", "sd", "--gpu"]),
        "--pnl-mask with --band-units sd and --gpu",
    );
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-mask", "0 - g", "--position", "top_tercile"]), "top_tercile");
}

#[test]
fn a_masked_search_reports_the_mask_and_is_thread_independent() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let args = ["--fitness", "pnl", "--pnl-mask", "g", "--pnl-mask-band", "0.5", "--pnl-mask-dwell", "5"];
    let one = json(&run(&data, &spec, &[&args[..], &["--threads", "1"]].concat()));
    let four = json(&run(&data, &spec, &[&args[..], &["--threads", "4"]].concat()));
    assert_eq!(members(&one), members(&four));
    let sf = at(&one, &["search_fitness"]);
    assert_eq!(at(sf, &["mask", "expr"]).as_str(), Some("g"));
    assert_eq!(num(sf, &["mask", "band"]), 0.5);
    assert_eq!(num(sf, &["mask", "dwell"]), 5.0);
    assert!(num(sf, &["mask", "off_rows"]) > 0.0);
    for slot in ["validation", "test"] {
        let b = at(&one, &["objective", slot]);
        let share = num(b, &["mask", "on_share"]);
        assert!(share > 0.0 && share < 1.0, "{slot} on_share {share}");
        assert!(num(b, &["mask", "runs"]) >= 1.0);
        assert!(at(b, &["unmasked_net_pnl"]).as_f64().is_some(), "{slot}");
    }
    let plain = json(&run(&data, &spec, &["--fitness", "pnl"]));
    assert_ne!(
        at(&one, &["search_fitness"]).get("book_hash"),
        at(&plain, &["search_fitness"]).get("book_hash")
    );
}

#[test]
fn resume_needs_the_same_mask() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let cp = dir.join("mask.rcsr").to_string_lossy().into_owned();
    let saved = ["--fitness", "pnl", "--pnl-mask", "g", "--pnl-mask-band", "0.5", "--pnl-mask-dwell", "5"];
    let first = run(&data, &spec, &[&saved[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let straight = json(&run(&data, &spec, &[&saved[..], &["--generations", "20"]].concat()));
    let resumed = json(&run(&data, &spec, &[&saved[..], &["--generations", "20", "--resume", &cp]].concat()));
    assert_eq!(members(&straight), members(&resumed));
    let resume = |extra: &[&str]| run(&data, &spec, &[&["--fitness", "pnl", "--generations", "20", "--resume", &cp][..], extra].concat());
    usage_error(&resume(&["--pnl-mask", "0 - g", "--pnl-mask-band", "0.5", "--pnl-mask-dwell", "5"]), "fitness");
    usage_error(&resume(&["--pnl-mask", "g", "--pnl-mask-band", "0.4", "--pnl-mask-dwell", "5"]), "fitness");
    usage_error(&resume(&["--pnl-mask", "g", "--pnl-mask-band", "0.5", "--pnl-mask-dwell", "6"]), "fitness");
    usage_error(&resume(&[]), "fitness");
}

/// #1203: the validation and test books (`trading::evaluate`) and the walk-forward test book
/// apply one missing-score rule; the search fitness book prices no missing score.
#[test]
fn missing_score_rule_is_shared_by_search_and_walkforward() {
    use rcsr::trading::MissingScore;
    // Two groups; each has a leading, two inner and a trailing stretch with no finite prediction.
    let s = slice(&[120, 120]);
    let mut pred = prediction(&s);
    for (i, p) in pred.iter_mut().enumerate() {
        let k = i % 120;
        if k < 3 || (40..47).contains(&k) || k == 60 || k >= 117 {
            *p = f64::NAN;
        }
    }
    let ppy = rcsr::walkforward::SECONDS_PER_YEAR / 900.0;
    let mut nets = Vec::new();
    for rule in [MissingScore::Flat, MissingScore::Hold] {
        let sp = TradingSpec { cost_bps: 7.5, min_trades: 0, missing_score: rule, ..TradingSpec::default() };
        let got = trading::evaluate(&pred, &s.ret, &s.rows, &sp).unwrap();
        let want = rcsr::walkforward::reference_metrics(&s.rows.time, &s.rows.group, &s.ret, &pred, 0.0, 7.5e-4, ppy, rule);
        assert_eq!(format!("{:?}", got.period), format!("{:?}", want), "{rule:?}");
        nets.push(got.net_pnl);
    }
    assert_ne!(nets[0], nets[1], "the two rules price different books");
    let book = FitnessBook::new(&s, &spec(), 900.0).unwrap();
    assert_eq!(book.regret(&pred), f64::INFINITY, "a non-finite prediction is refused by the fitness");
}

// ---- #1179: the cross-sectional excess (`--pnl-excess xs-mean`) ----

use trading::PnlExcess;

fn xs_spec() -> TradingSpec {
    TradingSpec { pnl_excess: PnlExcess::XsMean, ..spec() }
}

/// `groups` symbols on `steps` shared timestamps 900 s apart, in canonical
/// (group, time) order.
fn panel(groups: usize, steps: usize, ret: impl Fn(usize, usize) -> f64) -> TradingSlice {
    let mut rows = TradingRows::default();
    let mut out = Vec::new();
    for g in 0..groups {
        for t in 0..steps {
            rows.group.push(g as u64);
            rows.time.push(1.6e9 + 900.0 * t as f64);
            out.push(ret(g, t));
        }
    }
    TradingSlice { rows, ret: out }
}

/// The mean of the finite returns of the rows `admit` takes at each row's
/// timestamp, by a scan over every row.
fn hand_means(s: &TradingSlice, admit: impl Fn(usize) -> bool) -> Vec<f64> {
    let n = s.ret.len();
    (0..n)
        .map(|i| {
            let (mut sum, mut count) = (0.0, 0usize);
            for j in 0..n {
                if s.rows.time[j] == s.rows.time[i] && s.ret[j].is_finite() && admit(j) {
                    sum += s.ret[j];
                    count += 1;
                }
            }
            if count == 0 { f64::NAN } else { sum / count as f64 }
        })
        .collect()
}

const DYADIC: f64 = 1.0 / 1024.0;

/// A rising market path; every value is a multiple of 2^-12, so sums and means
/// over four symbols are exact.
fn common(t: usize) -> f64 {
    ((t * 7) % 11) as f64 * DYADIC / 4.0
}

/// Four symbols' dispersion around their timestamp mean, multiples of 2^-12.
fn spread(g: usize, t: usize) -> f64 {
    let base = |g: usize| ((g * 5 + t * 3 + (g * t) % 4) % 9) as f64;
    let mean = (0..4).map(base).sum::<f64>() / 4.0;
    (base(g) - mean) * DYADIC
}

#[test]
fn xs_mean_removes_common_drift() {
    // The same return path for every symbol leaves no relative return: no
    // profitable path, exit 2 at the CLI.
    let same = panel(4, 64, |_, t| common(t));
    assert!(FitnessBook::new(&same, &xs_spec(), 900.0).unwrap_err().contains("no profitable"));
    // The common path over a dispersion is the book of the dispersion alone, bit for bit.
    let drift = panel(4, 64, |g, t| common(t) + spread(g, t));
    let book = FitnessBook::new(&drift, &xs_spec(), 900.0).unwrap();
    let alone = FitnessBook::new(&panel(4, 64, spread), &xs_spec(), 900.0).unwrap();
    let bits = |b: &FitnessBook| b.excess().iter().map(|e| e.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&book), bits(&alone));
    assert_eq!(book.bound().to_bits(), alone.bound().to_bits());
    assert_eq!(book.hash(), alone.hash());
    let n = drift.ret.len();
    assert_eq!(book.regret(&vec![-1.0; n]).to_bits(), 1.0f64.to_bits(), "all-flat is exactly 1");
    // A market-timing rule holds every symbol or none at a timestamp: its gross
    // is exactly 0, so no such tree scores below 1.
    for seed in 0..32usize {
        let pred: Vec<f64> = (0..n).map(|i| if ((i % 64) * 13 + seed * 7) % 5 < 2 + seed % 3 { 1.0 } else { -1.0 }).collect();
        assert_eq!(book.totals(&pred).gross, 0.0, "seed {seed}");
        assert!(book.regret(&pred) >= 1.0, "seed {seed}: {}", book.regret(&pred));
    }
    // Without the flag the always-long book earns the drift; with it, it pays its costs.
    let long = vec![1.0; n];
    assert!(FitnessBook::new(&drift, &spec(), 900.0).unwrap().regret(&long) < 1.0);
    assert!(book.regret(&long) > 1.0);
    assert_ne!(book.hash(), FitnessBook::new(&drift, &spec(), 900.0).unwrap().hash());
}

#[test]
fn xs_mean_rewards_relative_edge() {
    // Symbol 0 beats the cross-section by `a` on every row of a rising market.
    let a = 2.0 * DYADIC;
    let s = panel(4, 48, |g, t| common(t) + if g == 0 { a } else { 0.0 });
    let book = FitnessBook::new(&s, &xs_spec(), 900.0).unwrap();
    let side = spec().per_side_bps() * 1e-4;
    for (i, &e) in book.excess().iter().enumerate() {
        assert_eq!(e, if i < 48 { 0.75 * a } else { -0.25 * a }, "row {i}");
    }
    // The bound holds symbol 0 on all 48 rows and every other symbol flat.
    let bound = 36.0 * a - 2.0 * side;
    assert!((book.bound() - bound).abs() <= 1e-12, "bound {} vs {bound}", book.bound());
    let n = s.ret.len();
    let rule = |steps: usize| -> Vec<f64> { (0..n).map(|i| if i < steps { 1.0 } else { -1.0 }).collect() };
    let full = book.regret(&rule(48));
    assert!(full.abs() <= 1e-12, "long symbol 0 throughout reaches the bound: {full}");
    let half = book.regret(&rule(24));
    let hand = 18.0 * a / bound;
    assert!(half < 1.0 && (half - hand).abs() <= 1e-12, "long symbol 0 for 24 rows: {half} vs {hand}");
    let long = book.regret(&vec![1.0; n]);
    assert!((long - (1.0 + 8.0 * side / bound)).abs() <= 1e-12, "long every symbol: {long}");
}

#[test]
fn xs_mean_bound_matches_brute_force() {
    // Rows (group, time): (0, t0), (0, t1), (1, t0), (1, t1).
    let base = TradingSlice {
        rows: TradingRows { time: vec![1.6e9, 1.6e9 + 900.0, 1.6e9, 1.6e9 + 900.0], group: vec![0, 0, 1, 1], funding: Vec::new() },
        ret: vec![0.010, -0.004, -0.002, 0.006],
    };
    let mut funded = base.clone();
    funded.rows.funding = vec![0.0001, 0.0002, 0.0003, 0.0004];
    let side = spec().per_side_bps() * 1e-4;
    for s in [&base, &funded] {
        let f = |i: usize| if s.rows.funding.is_empty() { 0.0 } else { s.rows.funding[i] };
        let m = [(s.ret[0] + s.ret[2]) / 2.0, (s.ret[1] + s.ret[3]) / 2.0];
        let e: Vec<f64> = (0..4).map(|i| (s.ret[i] - m[i % 2]) - f(i)).collect();
        // The best net over all 16 paths; each run starts flat and pays its exit.
        let best = |runs: &[&[usize]]| -> f64 {
            (0..16u32)
                .map(|bits| {
                    let p = |i: usize| ((bits >> i) & 1) as f64;
                    runs.iter()
                        .map(|run| {
                            let (mut prev, mut net) = (0.0, 0.0);
                            for &i in run.iter() {
                                net += p(i) * e[i] - side * (p(i) - prev).abs();
                                prev = p(i);
                            }
                            net - side * prev
                        })
                        .sum::<f64>()
                })
                .fold(f64::NEG_INFINITY, f64::max)
        };
        let want = best(&[&[0, 1], &[2, 3]]);
        assert!(want > 0.0);
        let book = FitnessBook::new(s, &xs_spec(), 900.0).unwrap();
        assert!((book.bound() - want).abs() <= 1e-15, "bound {} vs brute force {want}", book.bound());
        let blocked = FitnessBook::with_blocks(s, &xs_spec(), 900.0, 2, 1.0).unwrap();
        let want_blocks = [best(&[&[0], &[2]]), best(&[&[1], &[3]])];
        assert_eq!(blocked.block_bounds().len(), 2);
        for (b, w) in want_blocks.iter().enumerate() {
            assert!((blocked.block_bounds()[b] - w).abs() <= 1e-15, "block {b}: {} vs {w}", blocked.block_bounds()[b]);
        }
    }
}

/// The search books keep the absolute net; `xs_excess_net` is the net of the same
/// book on the returns minus their timestamp mean, under both missing-score rules
/// and with interval segments, and it equals the fitness net on training rows.
#[test]
fn xs_excess_net_is_the_net_over_the_timestamp_mean() {
    use rcsr::trading::MissingScore;
    let mut s = slice(&[150, 150, 120, 1]);
    s.rows.funding = (0..s.ret.len()).map(|i| 0.0001 * frac(i as f64 * 0.31)).collect();
    for i in [5, 151, 152, 400] {
        s.ret[i] = f64::NAN;
    }
    let mut pred = prediction(&s);
    for i in [7, 8, 9, 200, 301] {
        pred[i] = f64::NAN;
    }
    let m = hand_means(&s, |_| true);
    let demeaned: Vec<f64> = s.ret.iter().zip(&m).map(|(r, m)| r - m).collect();
    let starts = [1.6e9 + 900.0 * 60.0, 1.6e9 + 900.0 * 100.0];
    for rule in [MissingScore::Flat, MissingScore::Hold] {
        for segments in [&[][..], &starts[..]] {
            let sp = TradingSpec { missing_score: rule, segment_starts: segments.to_vec(), ..spec() };
            let plain = trading::evaluate(&pred, &s.ret, &s.rows, &sp).unwrap();
            let xs = trading::evaluate(&pred, &s.ret, &s.rows, &TradingSpec { pnl_excess: PnlExcess::XsMean, ..sp.clone() }).unwrap();
            assert_eq!(plain.xs_excess_net, None);
            assert_eq!(xs.net_pnl.to_bits(), plain.net_pnl.to_bits(), "{rule:?}: the reported net stays absolute");
            assert_eq!(xs.json().replace(&format!(",\"xs_excess_net\":{}", rcsr::devlog::num(xs.xs_excess_net.unwrap())), ""), plain.json());
            let want = trading::evaluate(&pred, &demeaned, &s.rows, &sp).unwrap().net_pnl;
            let got = xs.xs_excess_net.unwrap();
            assert!((got - want).abs() <= 1e-12, "{rule:?} {segments:?}: {got} vs {want}");
            assert!((got - xs.net_pnl).abs() > 1e-6, "{rule:?}: the timestamp mean moves the net");
        }
    }
    let t = slice(&[150, 150, 120, 1]);
    let pred = prediction(&t);
    for segments in [&[][..], &starts[..]] {
        let book = FitnessBook::with_segments(&t, &xs_spec(), 900.0, 1, 1.0, segments, None).unwrap();
        let stats = trading::evaluate(&pred, &t.ret, &t.rows, &TradingSpec { segment_starts: segments.to_vec(), ..xs_spec() }).unwrap();
        let net = book.net_of(&book.totals(&pred));
        assert!((net - stats.xs_excess_net.unwrap()).abs() <= 1e-12, "{segments:?}: fitness net {net} vs {:?}", stats.xs_excess_net);
    }
}

#[test]
fn xs_mean_composes_with_the_mask_and_interval_segments() {
    let s = slice(&[300, 280, 260]);
    let pred = prediction(&s);
    let n = s.ret.len();
    let plain = FitnessBook::new(&s, &xs_spec(), 900.0).unwrap();
    let all_on = masked(&s, &xs_spec(), 1, &vec![true; n]).unwrap();
    assert_eq!(plain.hash(), all_on.hash());
    assert_eq!(plain.regret(&pred).to_bits(), all_on.regret(&pred).to_bits());
    // Under a mask the timestamp mean reads the rows with the gate on; rows outside earn nothing.
    let on: Vec<bool> = (0..n).map(|i| (i / 37 + s.rows.group[i] as usize) % 3 != 0).collect();
    let book = masked(&s, &xs_spec(), 1, &on).unwrap();
    let m = hand_means(&s, |j| on[j]);
    for i in 0..n {
        let want = if on[i] { s.ret[i] - m[i] } else { 0.0 };
        assert!((book.excess()[i] - want).abs() <= 1e-15, "row {i}: {} vs {want}", book.excess()[i]);
    }
    assert_ne!(book.hash(), masked(&s, &spec(), 1, &on).unwrap().hash());
    // Interval segments cut by time, so the timestamp means stay and the bound
    // and the net are the sums over the separately built segments.
    let cut = 1.6e9 + 900.0 * 140.0;
    let seg = FitnessBook::with_segments(&s, &xs_spec(), 900.0, 1, 1.0, &[1.6e9, cut], None).unwrap();
    let (mut bound, mut net) = (0.0, 0.0);
    for before in [true, false] {
        let keep: Vec<usize> = (0..n).filter(|&i| (s.rows.time[i] < cut) == before).collect();
        let part = TradingSlice {
            rows: TradingRows {
                time: keep.iter().map(|&i| s.rows.time[i]).collect(),
                group: keep.iter().map(|&i| s.rows.group[i]).collect(),
                funding: Vec::new(),
            },
            ret: keep.iter().map(|&i| s.ret[i]).collect(),
        };
        let b = FitnessBook::new(&part, &xs_spec(), 900.0).unwrap();
        let p: Vec<f64> = keep.iter().map(|&i| pred[i]).collect();
        bound += b.bound();
        net += b.net_of(&b.totals(&p));
    }
    assert!((seg.bound() - bound).abs() <= 1e-12, "bound {} vs {bound}", seg.bound());
    assert!((seg.net_of(&seg.totals(&pred)) - net).abs() <= 1e-12);
}

// -- CLI --

const PANEL_SYMBOLS: usize = 4;
const PANEL_STEPS: usize = 400;

/// Four symbols on 400 shared timestamps, rows in time order. `r = c_t + 0.01 x1
/// + noise`: `c_t` is a market path that rises, then falls; `x2` is a market
/// feature (`c_t` scaled, the same for every symbol at a timestamp); `x1` is a
/// symbol feature.
fn write_panel_csv(dir: &std::path::Path) -> String {
    let mut text = String::from("t,sym,x1,x2,r,y\n");
    for t in 0..PANEL_STEPS {
        let c = if t < PANEL_STEPS / 2 { 0.003 } else { -0.003 } * (0.5 + frac(t as f64 * 0.414_213_562_373_095));
        for k in 0..PANEL_SYMBOLS {
            let f = (t * PANEL_SYMBOLS + k) as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let r = c + 0.01 * x1 + 0.0005 * noise;
            text.push_str(&format!("{},S{},{},{},{},{}\n", T0 + 900 * t as i64, k, x1, c / 0.003, r, r));
        }
    }
    let path = dir.join("panel.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// The spec of `write_spec` with the entry at the label time, so the test block
/// starts at the split.
fn write_panel_spec(dir: &std::path::Path) -> String {
    let text = std::fs::read_to_string(write_spec(dir)).unwrap().replace("\"entry_offset_seconds\":900", "\"entry_offset_seconds\":0");
    let path = dir.join("panel-spec.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn run_panel(data: &str, spec: &str, extra: &[&str]) -> Output {
    run(data, spec, &[&["--group-by", "sym"][..], extra].concat())
}

#[test]
fn xs_mean_reports_absolute_net() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let out = run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "xs-mean"]);
    let j = json(&out);
    assert_eq!(at(&j, &["search_fitness", "excess"]).as_str(), Some("xs_mean"));
    assert_eq!(at(&j, &["objective", "spec", "pnl_excess"]).as_str(), Some("xs_mean"));
    for slot in ["validation", "test"] {
        let b = at(&j, &["objective", slot]);
        let (net, xs) = (num(b, &["net_pnl"]), num(b, &["xs_excess_net"]));
        assert!(net.is_finite() && xs.is_finite() && net != xs, "{slot}: net {net} xs {xs}");
    }
    let run_path = dir.join("run.json");
    std::fs::write(&run_path, &out.stdout).unwrap();
    let run = run_path.to_string_lossy().into_owned();
    let step = |k: usize| (T0 + 900 * k as i64).to_string();
    let w = format!("v,{},{},{},{}", step(0), step(299), step(300), step(PANEL_STEPS));
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let wf = rcsr_cmd::rcsr()
        .args(["walkforward", "--data", &data, "--time", "t", "--group-by", "sym", "--from-run", &run, "--window", &w, "--out", &wf_path])
        .output()
        .unwrap();
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let wf_json = Json::parse(std::fs::read_to_string(&wf_path).unwrap().trim()).unwrap();
    let m = match at(&wf_json, &["windows"]) {
        Json::Arr(a) => at(&a[0], &["metrics"]).clone(),
        _ => panic!("windows is not an array"),
    };
    let test = at(&j, &["objective", "test"]);
    for key in ["rows", "trades", "entries", "total_cost", "exposure"] {
        assert!((num(&m, &[key]) - num(test, &[key])).abs() <= 1e-12, "{key}: walkforward {} objective {}", num(&m, &[key]), num(test, &[key]));
    }
    let wf_net = num(&m, &["mean_return"]) * num(&m, &["periods"]) * PANEL_SYMBOLS as f64;
    assert!((wf_net - num(test, &["net_pnl"])).abs() <= 1e-12, "walkforward net {wf_net} objective {}", num(test, &["net_pnl"]));
}

#[test]
fn explicit_none_is_the_default_run() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let plain = run_panel(&data, &spec, &["--fitness", "pnl"]);
    let none = run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "none"]);
    let (a, b) = (json(&plain), json(&none));
    assert_eq!(members(&a), members(&b));
    assert_eq!(at(&a, &["search_fitness"]).get("book_hash"), at(&b, &["search_fitness"]).get("book_hash"));
    assert_eq!(without_digits(&plain), without_digits(&none));
    assert!(at(&b, &["search_fitness"]).get("excess").is_none());
    assert!(at(&b, &["objective", "spec"]).get("pnl_excess").is_none());
    assert!(at(&b, &["objective", "test"]).get("xs_excess_net").is_none());
    let (la, lb) = (dir.join("plain.jsonl"), dir.join("none.jsonl"));
    let logged = |log: &std::path::Path, extra: &[&str]| run_logged(&data, &spec, log, &[&["--group-by", "sym"][..], extra].concat());
    assert_eq!(code(&logged(&la, &["--fitness", "pnl"])), 0);
    assert_eq!(code(&logged(&lb, &["--fitness", "pnl", "--pnl-excess", "none"])), 0);
    assert_eq!(archive_hash(&la), archive_hash(&lb));
    let xs = json(&run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "xs-mean"]));
    assert_ne!(at(&a, &["search_fitness"]).get("book_hash"), at(&xs, &["search_fitness"]).get("book_hash"));
}

#[test]
fn xs_mean_flags_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    usage_error(&run_panel(&data, &spec, &["--pnl-excess", "xs-mean"]), "--pnl-excess needs --fitness pnl");
    usage_error(&run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "beta"]), "--pnl-excess must be none or xs-mean");
    usage_error(
        &run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "xs-mean", "--cash-return", "0.0001"]),
        "--pnl-excess xs-mean needs --cash-return 0",
    );
    // One symbol: every timestamp has one row, so no row has a relative return.
    let single = write_csv(&dir, 900);
    usage_error(&run(&single, &write_spec(&dir), &["--fitness", "pnl", "--pnl-excess", "xs-mean"]), "no profitable long/flat path");
}

/// #1232: `rcsr switch` and `rcsr compose` take the fitness flags; without
/// their inputs they exit 2 for the missing input, not for the flag.
#[test]
fn compose_and_switch_accept_pnl_excess() {
    for (sub, missing) in [("switch", "--a <expr|flat> is required"), ("compose", "--gates <menu.json> is required")] {
        for flags in [&["--pnl-excess", "xs-mean"][..], &["--pnl-stat", "mean-semi"], &["--pnl-stat", "cvar", "--pnl-cvar-q", "0.5"]] {
            let o = rcsr_cmd::rcsr().arg(sub).args(flags).output().unwrap();
            usage_error(&o, missing);
            assert!(!stderr(&o).contains(&format!("{}:", flags[0])), "{sub} {flags:?}: {}", stderr(&o));
        }
    }
}

#[test]
fn resume_needs_the_same_excess() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let cp = dir.join("xs.rcsr").to_string_lossy().into_owned();
    let saved = ["--fitness", "pnl", "--pnl-excess", "xs-mean"];
    let first = run_panel(&data, &spec, &[&saved[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let straight = json(&run_panel(&data, &spec, &[&saved[..], &["--generations", "20"]].concat()));
    let resumed = json(&run_panel(&data, &spec, &[&saved[..], &["--generations", "20", "--resume", &cp]].concat()));
    assert_eq!(members(&straight), members(&resumed));
    let resume = |extra: &[&str]| run_panel(&data, &spec, &[&["--fitness", "pnl", "--generations", "20", "--resume", &cp][..], extra].concat());
    usage_error(&resume(&[]), "fitness");
    usage_error(&resume(&["--pnl-excess", "none"]), "fitness");
}

/// An interval split (two training segments, a validation and a test interval)
/// with a regime mask: the xs-mean fitness, the masked books and `xs_excess_net`.
#[test]
fn xs_mean_runs_with_a_split_table_and_a_mask() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let step = |k: i64| T0 + 900 * k;
    let table = dir.join("splits.csv");
    std::fs::write(
        &table,
        format!(
            "split_id,role,start,end\na,train,{},{}\na,validation,{},{}\na,train,{},{}\na,test,{},{}\n",
            step(0),
            step(150),
            step(150),
            step(200),
            step(200),
            step(300),
            step(300),
            step(400)
        ),
    )
    .unwrap();
    let table = table.to_string_lossy().into_owned();
    let out = rcsr_cmd::rcsr()
        .args([
            "--data", &data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "sym", "--split-table", &table,
            "--split-id", "a", "--target-spec", &spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "10", "--pop",
            "32", "--metric", "squared", "--json", "--quiet", "--objective", "pnl", "--cost-bps", "5", "--fitness", "pnl",
            "--pnl-excess", "xs-mean", "--pnl-mask", "x1 + 0.3",
        ])
        .output()
        .unwrap();
    let j = json(&out);
    assert_eq!(at(&j, &["split", "mode"]).as_str(), Some("temporal-intervals"));
    assert_eq!(at(&j, &["search_fitness", "excess"]).as_str(), Some("xs_mean"));
    assert!(num(&j, &["search_fitness", "mask", "off_rows"]) > 0.0);
    for slot in ["validation", "test"] {
        let b = at(&j, &["objective", slot]);
        assert!(num(b, &["xs_excess_net"]).is_finite(), "{slot}");
        assert!(num(b, &["mask", "on_share"]) < 1.0, "{slot}");
    }
}

// ---- #1246: one masked sd definition ----

const MASKED_SD_ARGS: [&str; 10] = ["--fitness", "pnl", "--pnl-mask", "g", "--band", "0.5", "--band-units", "sd", "--min-holding", "2"];

/// The gate-on training rows of the `g` column of `write_gate_csv`, as `(x1, x2)` and the scores of `exact` on them.
fn masked_train_scores(j: &Json, data: &str, exact: &str) -> Vec<f64> {
    let tree = rcsr::parse::parse(exact, &["x1".to_string(), "x2".to_string()]).unwrap();
    let (lo, hi) = time_range(j, "train_time");
    let text = std::fs::read_to_string(data).unwrap();
    text.lines()
        .skip(1)
        .map(|l| l.split(',').map(|c| c.parse::<f64>().unwrap()).collect::<Vec<f64>>())
        .filter(|v| v[0] >= lo && v[0] <= hi && v[5] > 0.0)
        .map(|v| rcsr::expr::eval(&tree, &[v[1], v[2]]))
        .collect()
}

#[test]
fn masked_sd_run_reports_the_training_band() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let j = json(&run(&data, &spec, &MASKED_SD_ARGS));
    let mut seen = 0;
    for m in pareto(&j) {
        let v = at(m, &["validation_book"]);
        let Some(sd) = v.get("band_sd").and_then(Json::as_f64) else { continue };
        let on = masked_train_scores(&j, &data, at(m, &["expression_exact"]).as_str().unwrap());
        let want = trading::score_sd(&on).sd.unwrap();
        assert!(close(sd, want), "{} rows {} sd {sd} want {want}", at(m, &["expression_exact"]).as_str().unwrap(), on.len());
        assert!(close(num(v, &["band_resolved"]), 0.5 * want));
        seen += 1;
    }
    assert!(seen >= 2, "{seen} members with an sd");
    let exact = at(&j, &["objective", "finalist", "expression_exact"]).as_str().unwrap();
    let on = masked_train_scores(&j, &data, exact);
    let sd = trading::score_sd(&on);
    assert_eq!(num(&j, &["objective", "spec", "band_sd_rows"]), on.len() as f64);
    let band = num(&j, &["objective", "spec", "band_resolved"]).to_bits();
    assert_eq!(num(&j, &["objective", "spec", "band_sd_rows"]), sd.rows as f64);
    assert!(close(f64::from_bits(band), 0.5 * sd.sd.unwrap()));
    let member = pareto(&j).iter().find(|m| at(m, &["expression_exact"]).as_str() == Some(exact)).expect("the finalist is on the front");
    assert_eq!(num(at(member, &["validation_book"]), &["band_resolved"]).to_bits(), band);
}

#[test]
fn walkforward_from_masked_sd_run_uses_the_training_band() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let out = run(&data, &spec, &MASKED_SD_ARGS);
    let j = json(&out);
    let run_path = dir.join("run.json").to_string_lossy().into_owned();
    std::fs::write(&run_path, &out.stdout).unwrap();
    let (tlo, thi) = time_range(&j, "train_time");
    let (vlo, vhi) = time_range(&j, "validation_time");
    let window = format!("v,{tlo},{thi},{vlo},{vhi}");
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let wf = rcsr_cmd::rcsr()
        .args(["walkforward", "--data", &data, "--time", "t", "--from-run", &run_path, "--window", &window, "--allow-in-sample-windows", "--out", &wf_path])
        .output()
        .unwrap();
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let report = Json::parse(std::fs::read_to_string(&wf_path).unwrap().trim()).unwrap();
    let exact = at(&j, &["objective", "finalist", "expression_exact"]).as_str().unwrap();
    let want = 0.5 * trading::score_sd(&masked_train_scores(&j, &data, exact)).sd.unwrap();
    let Json::Arr(windows) = at(&report, &["windows"]) else { panic!("windows") };
    assert_eq!(num(&windows[0], &["band", "band_width"]).to_bits(), num(&j, &["objective", "spec", "band_resolved"]).to_bits());
    assert!(close(num(&windows[0], &["band", "band_width"]), want));
}

/// The book hash of a masked sd-unit book on `slice(&[400, 400])` before #1246.
const MASKED_SD_HASH_BEFORE: u64 = 0xc99ea16df2e72a3a;
const MASKED_SCORE_HASH: u64 = 0xc4f6750e164e983d;
const UNMASKED_SD_HASH: u64 = 0xc12c5650f2d4ac73;
const MASKED_SD_CHECKPOINT_HASH_BEFORE: u64 = 0xe2ff12ae5c721c6d;

#[test]
fn book_hash_unmasked_sd_unchanged() {
    let s = slice(&[400, 400]);
    let book = FitnessBook::new(&s, &band_spec(0.5, trading::BandUnits::Sd, 1), 900.0).unwrap();
    assert_eq!(book.hash(), UNMASKED_SD_HASH, "{:#x}", book.hash());
}

#[test]
fn book_hash_masked_score_unchanged() {
    let s = slice(&[400, 400]);
    let on = alternate_mask(&s, 50);
    let book = masked(&s, &band_spec(0.5, trading::BandUnits::Score, 1), 2, &on).unwrap();
    assert_eq!(book.hash(), MASKED_SCORE_HASH, "{:#x}", book.hash());
}

#[test]
fn masked_sd_book_hash_names_the_masked_definition() {
    let s = slice(&[400, 400]);
    let on = alternate_mask(&s, 50);
    let book = masked(&s, &band_spec(0.5, trading::BandUnits::Sd, 1), 2, &on).unwrap();
    assert_ne!(book.hash(), MASKED_SD_HASH_BEFORE, "{:#x}", book.hash());
}

#[test]
fn masked_sd_checkpoint_from_before_the_fix_refuses_resume() {
    let (_scratch, dir) = scratch();
    let data = write_gate_csv(&dir);
    let spec = write_spec(&dir);
    let cp = dir.join("masked-sd.rcsr").to_string_lossy().into_owned();
    let first = run(&data, &spec, &[&MASKED_SD_ARGS[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let hash = saved_book_hash(&cp);
    assert_ne!(hash, MASKED_SD_CHECKPOINT_HASH_BEFORE, "{hash:#x}");
    let before = unfnv_u64(hash, 1);
    assert_eq!(before, MASKED_SD_CHECKPOINT_HASH_BEFORE, "{hash:#x} {before:#x}");
    let resume = |path: &str| run(&data, &spec, &[&MASKED_SD_ARGS[..], &["--generations", "20", "--resume", path]].concat());
    let same = resume(&cp);
    assert_eq!(code(&same), 0, "{}", stderr(&same));
    let text = std::fs::read_to_string(&cp).unwrap().replace(&format!("long_pnl:{hash:016x}"), &format!("long_pnl:{before:016x}"));
    let block: String = text.lines().filter(|l| l.starts_with("config ")).map(|l| format!("{l}\n")).collect();
    let config_hash = format!("config_hash {:016x}", rcsr::checkpoint::hash_config_text(&block));
    let old: String = text.lines().map(|l| if l.starts_with("config_hash ") { format!("{config_hash}\n") } else { format!("{l}\n") }).collect();
    let old_cp = dir.join("before.rcsr");
    std::fs::write(&old_cp, old).unwrap();
    usage_error(&resume(&old_cp.to_string_lossy()), "fitness");
}

#[test]
fn masked_sd_band_is_one_definition_across_books() {
    let s = slice(&[400, 400]);
    let on = alternate_mask(&s, 50);
    let scores: Vec<f64> = (0..s.ret.len()).map(|i| if on[i] { if i % 2 == 0 { 5.0 } else { -5.0 } } else { 0.0 }).collect();
    let sp = band_spec(0.5, trading::BandUnits::Sd, 1);
    let book = masked(&s, &sp, 2, &on).unwrap();
    let off_bits: Vec<bool> = book.flags().iter().map(|f| f & trading::FITNESS_MASK_OFF != 0).collect();
    assert_eq!(off_bits, on.iter().map(|o| !o).collect::<Vec<bool>>());
    let member = BandSpec::from_training_masked(BandUnits::Sd, 0.5, &scores, &s.rows.time, book.flags());
    let on_scores: Vec<f64> = scores.iter().zip(&on).filter(|(_, &o)| o).map(|(&p, _)| p).collect();
    let want = trading::score_sd(&on_scores);
    assert_eq!(member.sd.map(f64::to_bits), want.sd.map(f64::to_bits));
    assert_eq!(member.sd_rows, Some(on_scores.len()));
    assert_eq!(book.resolve_band(&scores).to_bits(), member.resolved.to_bits());
    assert_eq!(member.resolved.to_bits(), (0.5 * want.sd.unwrap()).to_bits());
    let last_on = (0..on.len()).rev().find(|&i| on[i]).unwrap();
    assert_eq!(member.sd_max_time, Some(s.rows.time[last_on]));
    let all = BandSpec::from_training(BandUnits::Sd, 0.5, &scores, &s.rows.time);
    assert_ne!(all.resolved.to_bits(), member.resolved.to_bits());
}

#[test]
fn masked_sd_edge_cases_give_no_band() {
    let time: Vec<f64> = (0..6).map(|i| i as f64).collect();
    let band = |scores: &[f64], flags: &[u8]| BandSpec::from_training_masked(BandUnits::Sd, 1.0, scores, &time, flags);
    let off = trading::FITNESS_MASK_OFF;
    let all_off = band(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[off; 6]);
    assert_eq!((all_off.sd, all_off.resolved, all_off.sd_rows, all_off.sd_max_time), (None, 0.0, Some(0), None));
    let one_on = band(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[off, off, 0, off, off, off]);
    assert_eq!((one_on.sd, one_on.resolved, one_on.sd_rows), (None, 0.0, Some(1)));
    let constant = band(&[9.0, 1.0, 4.0, 4.0, 4.0, 2.0], &[off, off, 0, 0, 0, off]);
    assert_eq!((constant.sd, constant.resolved, constant.sd_rows), (None, 0.0, Some(3)));
    let score = BandSpec::from_training_masked(BandUnits::Score, 0.3, &[1.0], &time, &[0]);
    assert_eq!(score, BandSpec::score(0.3));
}

/// The interpreter and the test recomputation may differ by a rounding in a score.
fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * a.abs().max(b.abs())
}

// ---- #1268: the training mask follows rcsr-gate-state/1 ----

use rcsr::regime_gate::{self, Barrier, BarrierKind, GateSpec};

const HOUR: i64 = 3600;
const DAY: i64 = 86_400;
const STRIDED_DAYS: usize = 160;
const STRIDED_ARGS: [&str; 10] = ["--group-by", "s", "--fill", "drop", "--fitness", "pnl", "--pnl-mask", "g", "--pnl-mask-dwell", "168"];

/// A gate that changes sign every 10 days, a day later for symbol b.
fn ten_day_gate(h: i64, j: usize) -> f64 {
    if ((h + 24 * j as i64) / 240) % 2 == 0 { 1.0 } else { -1.0 }
}

/// A gate that changes sign every 3 days, a day later for symbol b.
fn three_day_gate(h: i64, j: usize) -> f64 {
    if ((h + 24 * j as i64) / 72) % 2 == 0 { 1.0 } else { -1.0 }
}

/// Hourly rows of symbols a and b over `days` days with the gate `g = gate(hour, symbol)`. The label
/// `y` and the return `r` are finite at hour 0 of each day only (a stride-24 label), so `--fill drop`
/// keeps one row per symbol and day.
fn strided_text(days: usize, gate: fn(i64, usize) -> f64) -> String {
    strided_text_every(days, gate, 24)
}

/// [`strided_text`] with the label and the return finite at every `stride`-th hour.
fn strided_text_every(days: usize, gate: fn(i64, usize) -> f64, stride: i64) -> String {
    let mut text = String::from("t,s,x1,x2,g,r,y\n");
    for h in 0..(days * 24) as i64 {
        for (j, sym) in ["a", "b"].iter().enumerate() {
            let f = (2 * h + j as i64) as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            let r = 0.01 * x1 + 0.0005 * noise;
            let (r, y) = if h % stride == 0 { (r.to_string(), (r + 0.05 * x2).to_string()) } else { (String::new(), String::new()) };
            text.push_str(&format!("{},{},{},{},{},{},{}\n", T0 + HOUR * h, sym, x1, x2, gate(h, j), r, y));
        }
    }
    text
}

fn write_strided_csv(dir: &std::path::Path, gate: fn(i64, usize) -> f64) -> String {
    let path = dir.join("strided.csv");
    std::fs::write(&path, strided_text(STRIDED_DAYS, gate)).unwrap();
    path.to_string_lossy().into_owned()
}

/// A one-day forward-return label on `y` with return column `r`.
fn write_daily_spec(dir: &std::path::Path) -> String {
    write_horizon_spec(dir, DAY)
}

/// A forward-return label on `y` with return column `r` and a horizon of `seconds`.
fn write_horizon_spec(dir: &std::path::Path, seconds: i64) -> String {
    let text = format!(
        "{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":{seconds},\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}]}}"
    );
    let path = dir.join(format!("h{seconds}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

/// The strided fixture prepared like a `--fitness pnl` run with `--fill drop` and a temporal split.
fn strided_prepared(text: &str, gate: &GateSpec) -> (rcsr::data::Table, rcsr::data::Prepared) {
    use rcsr::data::{FillMethod, PrepSpec, SplitMode, SplitSpec, TradingColumns};
    let table = rcsr::data::parse_table(text).unwrap();
    let cols = TradingColumns { return_column: Some("r".to_string()), log_returns: true, train: true, mask: Some(gate.clone()), ..Default::default() };
    let spec = SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        group_col: Some(1),
        test_frac: 0.25,
        validation_frac: 0.25,
        trading: Some(cols),
        canonical_train_order: true,
        ..SplitSpec::default()
    };
    let prep = PrepSpec { fill: FillMethod::Drop, ..PrepSpec::default() };
    let vars = ["x1".to_string(), "x2".to_string()];
    let p = rcsr::data::prepare_full_with(&table, &vars, "y", &spec, &mut rcsr::rng::Mt::new(1), &prep).unwrap();
    (table, p)
}

/// The file row of every slice row, matched on group and time.
fn file_rows(table: &rcsr::data::Table, rows: &TradingRows) -> Vec<usize> {
    let time = rcsr::data::time_keys(table, 0).to_f64();
    let group = rcsr::data::group_ids(table, 1).0;
    let row_at: std::collections::HashMap<(u64, u64), usize> = (0..time.len()).map(|r| ((group[r] as u64, time[r].to_bits()), r)).collect();
    (0..rows.time.len()).map(|i| row_at[&(rows.group[i], rows.time[i].to_bits())]).collect()
}

/// The `rcsr-gate-state/1` states of `gate` on `rows`: the kernel over every row of the file, not
/// reading rows from `barrier_from` on (the selection barrier of a temporal split).
fn evaluation_states(table: &rcsr::data::Table, gate: &GateSpec, barrier_from: Option<f64>, rows: &TradingRows) -> Vec<bool> {
    let time = rcsr::data::time_keys(table, 0).to_f64();
    let group: Vec<u64> = rcsr::data::group_ids(table, 1).0.into_iter().map(|g| g as u64).collect();
    let g = table.col(table.index_of("g").unwrap()).to_vec();
    let barriers: Vec<Barrier> = barrier_from.map(|start| Barrier { start, end: f64::INFINITY, kind: BarrierKind::Test }).into_iter().collect();
    regime_gate::prefix_states(gate, &group, &time, &[g], &barriers, &file_rows(table, rows)).unwrap()
}

/// The kernel over the slice rows alone, per group: the training rule before #1268.
fn slice_kernel_states(table: &rcsr::data::Table, gate: &GateSpec, rows: &TradingRows) -> Vec<bool> {
    let g = table.col(table.index_of("g").unwrap());
    let values: Vec<f64> = file_rows(table, rows).into_iter().map(|r| g[r]).collect();
    regime_gate::gate_states(gate, &rows.group, &rows.time, &[values]).unwrap()
}

/// `search_fitness.mask.off_rows`; the block is absent when every training row is on.
fn off_rows(run: &Json) -> f64 {
    at(run, &["search_fitness"]).get("mask").map_or(0.0, |m| num(m, &["off_rows"]))
}

/// `(group, time)` of the first row of every maximal on-run of a group.
fn on_run_starts(rows: &TradingRows, on: &[bool]) -> Vec<(u64, f64)> {
    (0..on.len()).filter(|&i| on[i] && (i == 0 || rows.group[i] != rows.group[i - 1] || !on[i - 1])).map(|i| (rows.group[i], rows.time[i])).collect()
}

#[test]
fn training_mask_equals_the_evaluation_states_on_a_strided_label() {
    let gate = GateSpec::parse("g", 0.0, 168).unwrap();
    let (table, p) = strided_prepared(&strided_text(STRIDED_DAYS, ten_day_gate), &gate);
    let split = p.trading.as_ref().expect("trading split");
    let masks = split.masks.as_ref().expect("masks");
    let rows = &split.train.as_ref().expect("training rows").rows;
    assert!(rows.time.windows(2).zip(rows.group.windows(2)).all(|(t, g)| g[0] != g[1] || t[1] - t[0] == DAY as f64), "one training row per symbol and day");
    let test_start = split.test.rows.time.iter().copied().fold(f64::INFINITY, f64::min);
    let want = evaluation_states(&table, &gate, Some(test_start), rows);
    assert!(want.iter().any(|&o| o) && want.iter().any(|&o| !o));
    assert_ne!(slice_kernel_states(&table, &gate, rows), want, "the fixture separates the row sets: 168 training rows are 168 days");
    assert_eq!(masks.train.as_deref(), Some(&want[..]), "training rows");
    let validation = &split.validation.as_ref().expect("validation rows").rows;
    assert_eq!(masks.validation.as_deref(), Some(&evaluation_states(&table, &gate, Some(test_start), validation)[..]), "validation rows");
    assert_eq!(masks.test, evaluation_states(&table, &gate, None, &split.test.rows), "test rows");
}

#[test]
fn training_on_runs_match_the_evaluation_rule() {
    let gate = GateSpec::parse("g", 0.0, 168).unwrap();
    let (table, p) = strided_prepared(&strided_text(STRIDED_DAYS, three_day_gate), &gate);
    let split = p.trading.as_ref().expect("trading split");
    let rows = &split.train.as_ref().expect("training rows").rows;
    let test_start = split.test.rows.time.iter().copied().fold(f64::INFINITY, f64::min);
    let want = on_run_starts(rows, &evaluation_states(&table, &gate, Some(test_start), rows));
    for g in 0..2u64 {
        assert!(want.iter().filter(|s| s.0 == g).count() >= 2, "group {g}: {want:?}");
    }
    assert_ne!(on_run_starts(rows, &slice_kernel_states(&table, &gate, rows)), want, "the fixture separates the row sets");
    let train = split.masks.as_ref().and_then(|m| m.train.as_deref()).expect("training mask");
    assert_eq!(on_run_starts(rows, train), want);
}

/// Every row of the strided CSV as `(time, symbol, labelled, state)`: the state from the kernel over
/// every row of the file per symbol (no barrier lies before the test block).
fn strided_row_states(path: &str, dwell: usize) -> Vec<(f64, u64, bool, bool)> {
    let text = std::fs::read_to_string(path).unwrap();
    let rows: Vec<Vec<&str>> = text.lines().skip(1).map(|l| l.split(',').collect()).collect();
    let group: Vec<u64> = rows.iter().map(|r| u64::from(r[1] == "b")).collect();
    let gate: Vec<f64> = rows.iter().map(|r| r[4].parse().unwrap()).collect();
    let states = regime_gate::hysteresis_states(&gate, &group, 0.0, dwell);
    rows.iter().zip(&group).zip(&states).map(|((r, &g), &s)| (r[0].parse().unwrap(), g, !r[6].is_empty(), s)).collect()
}

#[test]
fn strided_training_states_agree_with_validation_and_walkforward() {
    let (_scratch, dir) = scratch();
    let data = write_strided_csv(&dir, three_day_gate);
    let spec = write_daily_spec(&dir);
    let out = run(&data, &spec, &STRIDED_ARGS);
    let j = json(&out);
    let run_path = dir.join("strided.run.json").to_string_lossy().into_owned();
    std::fs::write(&run_path, &out.stdout).unwrap();
    let rows = strided_row_states(&data, 168);
    let labelled = |lo: f64, hi: f64| -> Vec<(f64, u64, bool, bool)> { rows.iter().copied().filter(|r| r.2 && r.0 >= lo && r.0 <= hi).collect() };
    let (tlo, thi) = time_range(&j, "train_time");
    let train = labelled(tlo, thi);
    assert_eq!(train.len() as f64, num(&j, &["train_rows"]));
    let off = train.iter().filter(|r| !r.3).count();
    assert!(off > 0 && off < train.len(), "{off} of {}", train.len());
    assert_eq!(off_rows(&j), off as f64, "training off rows");
    let (vlo, vhi) = time_range(&j, "validation_time");
    let validation = labelled(vlo, vhi);
    let on = validation.iter().filter(|r| r.3).count();
    let vm = at(&j, &["objective", "validation", "mask"]);
    assert!((num(vm, &["on_share"]) - on as f64 / validation.len() as f64).abs() <= 1e-12, "validation on share");
    let runs: usize = (0..2u64)
        .map(|g| {
            let on: Vec<bool> = validation.iter().filter(|r| r.1 == g).map(|r| r.3).collect();
            (0..on.len()).filter(|&i| on[i] && (i == 0 || !on[i - 1])).count()
        })
        .sum();
    assert_eq!(num(vm, &["runs"]), runs as f64, "validation on-runs");
    // An in-sample walk-forward window over the second half of the training days prices the rows the
    // training mask holds on.
    let mid = tlo + (DAY * ((thi - tlo) as i64 / DAY / 2)) as f64;
    let window = format!("w,{},{},{},{}", tlo, mid - (2 * DAY) as f64, mid, thi + 1.0);
    let wf = rcsr_cmd::rcsr()
        .args(["walkforward", "--data", &data, "--time", "t", "--group-by", "s", "--from-run", &run_path, "--window", &window, "--allow-in-sample-windows", "--json"])
        .output()
        .unwrap();
    let report = json(&wf);
    let Json::Arr(windows) = at(&report, &["windows"]) else { panic!("windows") };
    let priced = labelled(mid, thi).iter().filter(|r| r.3).count();
    assert!(priced > 0);
    assert_eq!(num(&windows[0], &["metrics", "rows"]), priced as f64, "walk-forward rows on");
}

/// The book hash of `STRIDED_ARGS` with `--generations 10` on the three-day strided fixture under the
/// training rule before #1268: the kernel over the training rows alone.
const STRIDED_MASK_HASH_BEFORE: u64 = 0xe370_75dd_e50d_5673;

/// A copy of checkpoint `cp` at `out` whose book hash is `to`, with its config hash recomputed.
fn with_book_hash(cp: &str, to: u64, out: &std::path::Path) {
    let from = saved_book_hash(cp);
    let text = std::fs::read_to_string(cp).unwrap().replace(&format!("long_pnl:{from:016x}"), &format!("long_pnl:{to:016x}"));
    let block: String = text.lines().filter(|l| l.starts_with("config ")).map(|l| format!("{l}\n")).collect();
    let config_hash = format!("config_hash {:016x}", rcsr::checkpoint::hash_config_text(&block));
    let edited: String = text.lines().map(|l| if l.starts_with("config_hash ") { format!("{config_hash}\n") } else { format!("{l}\n") }).collect();
    std::fs::write(out, edited).unwrap();
}

#[test]
fn a_masked_checkpoint_of_the_training_row_rule_refuses_resume() {
    let (_scratch, dir) = scratch();
    let data = write_strided_csv(&dir, three_day_gate);
    let spec = write_daily_spec(&dir);
    let cp = dir.join("strided.rcsr").to_string_lossy().into_owned();
    let first = run(&data, &spec, &[&STRIDED_ARGS[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let hash = saved_book_hash(&cp);
    assert_ne!(hash, STRIDED_MASK_HASH_BEFORE, "{hash:#x}: the training mask still reads the training rows alone");
    let resume = |path: &str| run(&data, &spec, &[&STRIDED_ARGS[..], &["--generations", "20", "--resume", path]].concat());
    let same = resume(&cp);
    assert_eq!(code(&same), 0, "{}", stderr(&same));
    let old = dir.join("before.rcsr");
    with_book_hash(&cp, STRIDED_MASK_HASH_BEFORE, &old);
    usage_error(&resume(&old.to_string_lossy()), "--pnl-mask");
}

// ---- #1278: the masked book skips unlabelled rows ----

use rcsr::lockbox::{self, Lockbox};

const BAND_SD: [&str; 6] = ["--band", "0.5", "--band-units", "sd", "--min-holding", "2"];
const DENSE_DAYS: usize = 60;

/// The strided fixture of [`three_day_gate`] over `days` days with the label every `stride` hours.
fn write_strided_every(dir: &std::path::Path, name: &str, days: usize, stride: i64) -> String {
    let path = dir.join(name);
    std::fs::write(&path, strided_text_every(days, three_day_gate, stride)).unwrap();
    path.to_string_lossy().into_owned()
}

/// The windows whose test blocks are the validation and the test rows of run `j` on the strided
/// fixture. The test rows are the last `test_rows / 2` labelled stamps, the same for both symbols.
fn strided_block_windows(j: &Json, days: usize, stride: i64, test_rows: f64) -> [(&'static str, String); 2] {
    let (vlo, vhi) = time_range(j, "validation_time");
    let (vlo, vhi) = (vlo as i64, vhi as i64);
    let stamps = (days as i64 * 24 + stride - 1) / stride;
    let start = T0 + HOUR * stride * (stamps - test_rows as i64 / 2);
    let end = T0 + DAY * days as i64;
    // A training range that ends two label strides (at least two days) before the test block.
    let gap = (2 * HOUR * stride).max(2 * DAY);
    [("validation", format!("v,{T0},{},{vlo},{}", vlo - gap, vhi + 1)), ("test", format!("t,{T0},{},{start},{end}", start - gap))]
}

/// Window 0 of `walkforward --from-run run` on `data` with `window`, `extra` added.
fn walk_window(data: &str, run: &str, window: &str, extra: &[&str]) -> Json {
    let mut args = vec!["walkforward", "--data", data, "--time", "t", "--group-by", "s", "--from-run", run, "--window", window, "--json"];
    args.extend_from_slice(extra);
    let report = json(&rcsr_cmd::rcsr().args(&args).output().unwrap());
    match at(&report, &["windows"]) {
        Json::Arr(a) => a[0].clone(),
        _ => panic!("windows is not an array"),
    }
}

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

/// The net of a walk-forward window of one symbol: one priced row per period.
fn symbol_net(w: &Json) -> f64 {
    let m = at(w, &["metrics"]);
    if num(m, &["rows"]) == 0.0 {
        0.0
    } else {
        num(m, &["mean_return"]) * num(m, &["periods"])
    }
}

/// A JSON value in the engine's text form (`devlog::num` keeps every bit).
fn json_text(j: &Json) -> String {
    match j {
        Json::Null => "null".to_string(),
        Json::Bool(b) => b.to_string(),
        Json::Num(v) => rcsr::devlog::num(*v),
        Json::Str(v) => rcsr::devlog::text(v),
        Json::Arr(a) => format!("[{}]", a.iter().map(json_text).collect::<Vec<_>>().join(",")),
        Json::Obj(o) => format!("{{{}}}", o.iter().map(|(k, v)| format!("{}:{}", rcsr::devlog::text(k), json_text(v))).collect::<Vec<_>>().join(",")),
    }
}

/// Walk-forward of the validation and the test block of run `j` (`run` on `data`) gives the run's
/// books: the rows, trades, entries and cost of the two-symbol window, and the net as the sum of
/// the windows of each symbol alone (a period return is the mean over the rows priced at its stamp,
/// so the two-symbol net is not a multiple of it when the symbols' gates differ).
fn assert_blocks_reproduced(tag: &str, j: &Json, data: &str, run: &str, days: usize, stride: i64, test_rows: f64) {
    let one = [symbol_csv(data, "a"), symbol_csv(data, "b")];
    for (slot, window) in strided_block_windows(j, days, stride, test_rows) {
        let want = at(j, &["objective", slot]);
        let m = at(&walk_window(data, run, &window, &["--allow-in-sample-windows"]), &["metrics"]).clone();
        for key in ["rows", "trades", "entries"] {
            assert_eq!(num(&m, &[key]), num(want, &[key]), "{tag} {slot} {key}");
        }
        assert!((num(&m, &["total_cost"]) - num(want, &["total_cost"])).abs() <= 1e-12, "{tag} {slot} total_cost");
        let net: f64 = one.iter().map(|d| symbol_net(&walk_window(d, run, &window, &["--allow-in-sample-windows"]))).sum();
        assert!((net - num(want, &["net_pnl"])).abs() <= 1e-12, "{tag} {slot}: walkforward net {net} vs objective {}", num(want, &["net_pnl"]));
    }
}

#[test]
fn strided_masked_walkforward_reproduces_objective_test() {
    let (_scratch, dir) = scratch();
    let band: Vec<&str> = [&STRIDED_ARGS[..], &BAND_SD[..]].concat();
    for (stride, days) in [(24i64, STRIDED_DAYS), (72, 2 * STRIDED_DAYS)] {
        // A label horizon equal to the stride: one period per labelled row.
        let spec = write_horizon_spec(&dir, HOUR * stride);
        let data = write_strided_every(&dir, &format!("strided{stride}.csv"), days, stride);
        for (book, args) in [("unmasked", &STRIDED_ARGS[..6]), ("threshold", &STRIDED_ARGS[..]), ("band", &band[..])] {
            let tag = format!("stride {stride} {book}");
            let out = run(&data, &spec, args);
            let j = json(&out);
            if book != "unmasked" {
                assert!(num(&j, &["objective", "test", "mask", "runs"]) >= 2.0, "{tag}: the mask must cut several runs");
            }
            assert!(num(&j, &["objective", "test", "trades"]) > 0.0, "{tag}: the book must trade");
            let run_path = dir.join(format!("strided{stride}-{book}.run.json")).to_string_lossy().into_owned();
            std::fs::write(&run_path, &out.stdout).unwrap();
            assert_blocks_reproduced(&tag, &j, &data, &run_path, days, stride, num(&j, &["test_rows"]));
        }
    }
}

/// #1278: rows whose fit label is missing and whose book return is finite stay in the book
/// (`--book-target`, #948). The fit label `y` is missing on every fifth labelled day; the book label
/// `yb` shares the return column `r`, so walk-forward prices the rows the search prices.
#[test]
fn strided_masked_book_keeps_rows_without_a_fit_label() {
    let (_scratch, dir) = scratch();
    let mut text = String::from("t,s,x1,x2,g,r,y,yb\n");
    for line in strided_text(STRIDED_DAYS, three_day_gate).lines().skip(1) {
        let c: Vec<&str> = line.split(',').collect();
        let day = (c[0].parse::<i64>().unwrap() - T0) / DAY;
        let y = if day % 5 == 2 { "" } else { c[6] };
        text.push_str(&format!("{},{},{},{},{},{},{},{}\n", c[0], c[1], c[2], c[3], c[4], c[5], y, c[6]));
    }
    let data = dir.join("book.csv").to_string_lossy().into_owned();
    std::fs::write(&data, text).unwrap();
    let entry = |column: &str| {
        format!(
            "{{\"column\":\"{column}\",\"name\":\"{column}\",\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\
             \"return_units\":\"log\",\"horizon_seconds\":86400,\"entry_offset_seconds\":0,\"delay_seconds\":0,\"entry_price\":\"open\",\
             \"exit_price\":\"close\",\"offset\":0.0,\"scale\":{{\"kind\":\"constant\",\"value\":1.0}},\"clip\":null,\"units\":\"log_return\",\
             \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}}"
        )
    };
    let spec = dir.join("book.json").to_string_lossy().into_owned();
    std::fs::write(&spec, format!("{{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{},{}]}}", entry("y"), entry("yb"))).unwrap();
    let mut book_rows = 0.0;
    for (book, mask) in [("unmasked", &[][..]), ("masked", &STRIDED_ARGS[6..])] {
        let args: Vec<&str> = [&STRIDED_ARGS[..6], mask, &["--book-target", "yb"][..]].concat();
        let out = run(&data, &spec, &args);
        let j = json(&out);
        if book == "unmasked" {
            book_rows = num(&j, &["objective", "test", "rows"]);
            assert!(book_rows > num(&j, &["test_rows"]), "the test book holds rows without a fit label");
        }
        let run_path = dir.join(format!("book-{book}.run.json")).to_string_lossy().into_owned();
        std::fs::write(&run_path, &out.stdout).unwrap();
        assert_blocks_reproduced(book, &j, &data, &run_path, STRIDED_DAYS, 24, book_rows);
    }
}

/// #1278: on hourly labels (every row labelled) the books are unchanged: `objective.validation`,
/// `objective.test` and the walk-forward metrics of both blocks of the unmasked, the masked
/// threshold and the masked band book of two symbols, pinned to the bytes of the base build.
#[test]
fn dense_masked_books_are_pinned() {
    let (_scratch, dir) = scratch();
    let spec = write_horizon_spec(&dir, HOUR);
    let data = write_strided_every(&dir, "dense.csv", DENSE_DAYS, 1);
    let band: Vec<&str> = [&STRIDED_ARGS[..], &BAND_SD[..]].concat();
    let mut got: Vec<String> = Vec::new();
    for (book, args) in [("unmasked", &STRIDED_ARGS[..6]), ("threshold", &STRIDED_ARGS[..]), ("band", &band[..])] {
        let out = run(&data, &spec, args);
        let j = json(&out);
        let run_path = dir.join(format!("dense-{book}.run.json")).to_string_lossy().into_owned();
        std::fs::write(&run_path, &out.stdout).unwrap();
        for (slot, window) in strided_block_windows(&j, DENSE_DAYS, 1, num(&j, &["test_rows"])) {
            let w = walk_window(&data, &run_path, &window, &["--allow-in-sample-windows"]);
            for (part, text) in [("objective", json_text(at(&j, &["objective", slot]))), ("walkforward", json_text(at(&w, &["metrics"])))] {
                got.push(format!("{book}.{slot}.{part} {}", rcsr::manifest::sha256_hex(text.as_bytes())));
            }
        }
    }
    let all = rcsr::manifest::sha256_hex(got.join("\n").as_bytes());
    assert_eq!(all, PINNED_DENSE_BOOKS, "{}", got.join("\n"));
}

const PINNED_DENSE_BOOKS: &str = "4ab830e2ea9b23584cca27e12f9a366c37e3582c7d4c51a46bd6024e50a9459b";

/// #1278: the sealed read prices the masked book of a strided label like the search: a synthetic
/// lock over the run's test block, a development window on the validation rows, a final seal and
/// one locked read of the test block.
#[test]
fn sealed_read_prices_the_strided_masked_book() {
    let (_scratch, dir) = scratch();
    let spec = write_daily_spec(&dir);
    let data = write_strided_csv(&dir, three_day_gate);
    let out = run(&data, &spec, &STRIDED_ARGS);
    let j = json(&out);
    let run_path = dir.join("sealed.run.json").to_string_lossy().into_owned();
    std::fs::write(&run_path, &out.stdout).unwrap();
    let [(_, validation_window), (_, test_window)] = strided_block_windows(&j, STRIDED_DAYS, 24, num(&j, &["test_rows"]));
    let field = |w: &str, k: usize| -> i64 { w.split(',').nth(k).unwrap().parse().unwrap() };
    let (vlo, start) = (field(&validation_window, 3), field(&test_window, 3));
    let lock_end = T0 + DAY * STRIDED_DAYS as i64;
    let locked = write_strided_every(&dir, "locked.csv", STRIDED_DAYS, 24);
    lockbox::write_sidecar(&locked, &Lockbox { start, end: lock_end }, "test").unwrap();
    let ledger = dir.join("ledger.jsonl").to_string_lossy().into_owned();
    let sealed = dir.join("sealed.json").to_string_lossy().into_owned();
    let dev = format!("dev,{T0},{},{vlo},{}", vlo - 2 * DAY, vlo + 10 * DAY);
    walk_window(&locked, &run_path, &dev, &["--ledger", &ledger, "--allow-in-sample-windows"]);
    let seal = rcsr_cmd::rcsr().args(["protocol", "seal", "--final", "--ledger", &ledger, "--out", &sealed]).output().unwrap();
    assert_eq!(code(&seal), 0, "{}", stderr(&seal));
    let read = walk_window(&locked, &run_path, &test_window, &["--ledger", &ledger, "--allow-in-sample-windows", "--locked-test", "--sealed", &sealed]);
    let open = walk_window(&data, &run_path, &test_window, &["--allow-in-sample-windows"]);
    let m = at(&read, &["metrics"]);
    assert_eq!(json_text(m), json_text(at(&open, &["metrics"])), "the sealed read prices the open book");
    let want = at(&j, &["objective", "test"]);
    for key in ["rows", "trades", "entries"] {
        assert_eq!(num(m, &[key]), num(want, &[key]), "{key}");
    }
    assert!((num(m, &["total_cost"]) - num(want, &["total_cost"])).abs() <= 1e-12);
}

/// #1278: a row the masked book cannot price (no finite return: an unlabelled row of a strided
/// label) neither starts nor ends an on-run, whatever its gate state: `masked_slice` over every row
/// prices the book of `masked_slice` over the priced rows alone.
#[test]
fn masked_slice_skips_rows_without_a_return() {
    let mut seed = 0x1278u64;
    let mut uniform = move || {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let spec = TradingSpec { cost_bps: 5.0, threshold: 0.0, min_trades: 0, ..TradingSpec::default() };
    let mut inside = 0usize;
    for case in 0..200 {
        let groups = 1 + case % 3;
        let mut full = TradingSlice { rows: TradingRows::default(), ret: Vec::new() };
        let (mut on, mut pred) = (Vec::new(), Vec::new());
        for k in 0..40 {
            for g in 0..groups {
                full.rows.time.push(1.6e9 + 3600.0 * k as f64);
                full.rows.group.push(g as u64);
                full.ret.push(if uniform() < 0.4 { f64::NAN } else { 0.01 * (uniform() - 0.45) });
                on.push(uniform() < 0.7);
                pred.push(uniform() - 0.4);
            }
        }
        let keep: Vec<usize> = (0..full.ret.len()).filter(|&i| full.ret[i].is_finite()).collect();
        for g in 0..groups as u64 {
            let rows: Vec<usize> = (0..full.ret.len()).filter(|&i| full.rows.group[i] == g).collect();
            inside += rows.windows(3).filter(|w| on[w[0]] && !on[w[1]] && on[w[2]] && full.ret[w[0]].is_finite() && !full.ret[w[1]].is_finite() && full.ret[w[2]].is_finite()).count();
        }
        let pick = |v: &[f64]| -> Vec<f64> { keep.iter().map(|&i| v[i]).collect() };
        let sub = TradingSlice {
            rows: TradingRows { time: pick(&full.rows.time), group: keep.iter().map(|&i| full.rows.group[i]).collect(), funding: Vec::new() },
            ret: pick(&full.ret),
        };
        let sub_on: Vec<bool> = keep.iter().map(|&i| on[i]).collect();
        let a_book = regime_gate::masked_slice(&full, &on);
        let b_book = regime_gate::masked_slice(&sub, &sub_on);
        let a = trading::evaluate(&pred, &a_book.ret, &a_book.rows, &spec);
        let b = trading::evaluate(&pick(&pred), &b_book.ret, &b_book.rows, &spec);
        match (a, b) {
            (Ok(a), Ok(b)) => {
                assert_eq!(a.net_pnl.to_bits(), b.net_pnl.to_bits(), "case {case}: net {} vs {}", a.net_pnl, b.net_pnl);
                assert_eq!((a.rows, a.trades, a.entries), (b.rows, b.trades, b.entries), "case {case}");
            }
            (Err(_), Err(_)) => {}
            (a, b) => panic!("case {case}: {:?} vs {:?}", a.map(|s| s.net_pnl), b.map(|s| s.net_pnl)),
        }
    }
    assert!(inside > 0, "the cases put an off row without a return inside an on-run");
}

// ---- #1273: --cash-return train-mean ----

const TRAIN_MEAN: [&str; 4] = ["--fitness", "pnl", "--cash-return", "train-mean"];

/// Simple return of every `write_panel_csv` row with a time in `[lo, hi]`.
fn panel_returns(data: &str, lo: f64, hi: f64) -> Vec<f64> {
    std::fs::read_to_string(data)
        .unwrap()
        .lines()
        .skip(1)
        .map(|l| l.split(',').collect::<Vec<_>>())
        .filter(|c| (lo..=hi).contains(&c[0].parse::<f64>().unwrap()))
        .map(|c| c[4].parse::<f64>().unwrap().exp_m1())
        .collect()
}

fn cash_of(j: &Json) -> f64 {
    assert_eq!(at(j, &["search_fitness", "cash_return_source"]).as_str(), Some("train_mean"));
    num(j, &["search_fitness", "cash_return"])
}

/// Reports stay at cash 0: the spec, the cash books and the walk-forward book.
fn assert_reports_at_cash_zero(j: &Json) {
    assert_eq!(num(j, &["objective", "spec", "cash_return"]), 0.0);
    for slot in ["validation", "test"] {
        let cash = at(at(j, &["objective", slot]), &["cash"]);
        assert_eq!(num(cash, &["net_pnl"]), 0.0, "{slot} cash book");
    }
}

#[test]
fn cash_return_train_mean_is_the_training_mean() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let j = json(&run_panel(&data, &spec, &TRAIN_MEAN));
    let (lo, hi) = time_range(&j, "train_time");
    let rows = panel_returns(&data, lo, hi);
    assert_eq!(num(&j, &["search_fitness", "rows"]), rows.len() as f64);
    let want = rows.iter().sum::<f64>() / rows.len() as f64;
    let got = cash_of(&j);
    assert!(want != 0.0 && (got - want).abs() <= 1e-15, "m {got} hand mean {want}");
    assert_reports_at_cash_zero(&j);
    let plain = json(&run_panel(&data, &spec, &["--fitness", "pnl"]));
    assert!(at(&plain, &["search_fitness"]).get("cash_return").is_none());
    assert!(at(&plain, &["search_fitness"]).get("cash_return_source").is_none());
}

#[test]
fn cash_return_train_mean_prices_like_the_constant_in_the_fitness() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let mean = json(&run_panel(&data, &spec, &TRAIN_MEAN));
    let m = cash_of(&mean);
    let constant = json(&run_panel(&data, &spec, &["--fitness", "pnl", "--cash-return", &format!("{m:e}")]));
    for key in ["rows", "groups", "bound", "cost_per_side", "threshold", "min_trades"] {
        assert_eq!(at(&mean, &["search_fitness", key]), at(&constant, &["search_fitness", key]), "{key}");
    }
    assert_eq!(members(&mean), members(&constant));
    assert_ne!(at(&mean, &["search_fitness", "book_hash"]), at(&constant, &["search_fitness", "book_hash"]));
    assert_eq!(num(&constant, &["objective", "spec", "cash_return"]), m);
    assert_reports_at_cash_zero(&mean);
}

#[test]
fn cash_return_train_mean_reads_training_rows_only() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let step = |k: i64| T0 + 900 * k;
    let table = dir.join("splits.csv");
    std::fs::write(
        &table,
        format!(
            "split_id,role,start,end\na,train,{},{}\na,validation,{},{}\na,train,{},{}\na,test,{},{}\n",
            step(0),
            step(150),
            step(150),
            step(200),
            step(200),
            step(300),
            step(300),
            step(400)
        ),
    )
    .unwrap();
    let table = table.to_string_lossy().into_owned();
    // Same file with every validation and test return replaced.
    let edited = dir.join("edited.csv");
    let text: String = std::fs::read_to_string(&data)
        .unwrap()
        .lines()
        .enumerate()
        .map(|(i, l)| {
            let c: Vec<&str> = l.split(',').collect();
            let t = if i == 0 { 0 } else { (c[0].parse::<i64>().unwrap() - T0) / 900 };
            if i > 0 && ((150..200).contains(&t) || t >= 300) {
                format!("{},{},{},{},{},{}\n", c[0], c[1], c[2], c[3], 0.02 + 0.001 * (t % 7) as f64, 0.02)
            } else {
                format!("{l}\n")
            }
        })
        .collect();
    std::fs::write(&edited, text).unwrap();
    let edited = edited.to_string_lossy().into_owned();
    let go = |d: &str| {
        let out = rcsr_cmd::rcsr()
            .args([
                "--data", d, "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "sym", "--split-table", &table,
                "--split-id", "a", "--target-spec", &spec, "--cpu", "--no-dev-log", "--seed", "3", "--generations", "10", "--pop",
                "32", "--metric", "squared", "--json", "--quiet", "--objective", "pnl", "--cost-bps", "5", "--fitness", "pnl",
                "--cash-return", "train-mean",
            ])
            .output()
            .unwrap();
        json(&out)
    };
    let (a, b) = (go(&data), go(&edited));
    assert_ne!(cash_of(&a), 0.0);
    assert_eq!(cash_of(&a).to_bits(), cash_of(&b).to_bits());
    assert_eq!(at(&a, &["search_fitness", "book_hash"]), at(&b, &["search_fitness", "book_hash"]));
    assert_ne!(num(at(&a, &["objective", "test"]), &["net_pnl"]), num(at(&b, &["objective", "test"]), &["net_pnl"]));
}

#[test]
fn cash_return_train_mean_keeps_reports_at_cash_zero() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let out = run_panel(&data, &spec, &TRAIN_MEAN);
    let j = json(&out);
    assert_reports_at_cash_zero(&j);
    let run_path = dir.join("run.json");
    std::fs::write(&run_path, &out.stdout).unwrap();
    let run = run_path.to_string_lossy().into_owned();
    let step = |k: usize| (T0 + 900 * k as i64).to_string();
    let w = format!("v,{},{},{},{}", step(0), step(299), step(300), step(PANEL_STEPS));
    let wf_path = dir.join("wf.json").to_string_lossy().into_owned();
    let wf = rcsr_cmd::rcsr()
        .args(["walkforward", "--data", &data, "--time", "t", "--group-by", "sym", "--from-run", &run, "--window", &w, "--out", &wf_path])
        .output()
        .unwrap();
    assert_eq!(code(&wf), 0, "{}", stderr(&wf));
    let wf_json = Json::parse(std::fs::read_to_string(&wf_path).unwrap().trim()).unwrap();
    let m = match at(&wf_json, &["windows"]) {
        Json::Arr(a) => at(&a[0], &["metrics"]).clone(),
        _ => panic!("windows is not an array"),
    };
    let test = at(&j, &["objective", "test"]);
    let wf_net = num(&m, &["mean_return"]) * num(&m, &["periods"]) * PANEL_SYMBOLS as f64;
    assert!((wf_net - num(test, &["net_pnl"])).abs() <= 1e-12, "walkforward net {wf_net} objective {}", num(test, &["net_pnl"]));
}

#[test]
fn cash_return_train_mean_flags_name_their_reason() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    usage_error(&run_panel(&data, &spec, &["--cash-return", "train-mean"]), "--cash-return train-mean needs --fitness pnl");
    usage_error(
        &run_panel(&data, &spec, &["--fitness", "pnl", "--pnl-excess", "xs-mean", "--cash-return", "train-mean"]),
        "--pnl-excess xs-mean needs --cash-return 0",
    );
    usage_error(&run_panel(&data, &spec, &["--fitness", "pnl", "--cash-return", "mean"]), "--cash-return");
}

#[test]
fn resume_needs_the_same_cash_mode() {
    let (_scratch, dir) = scratch();
    let data = write_panel_csv(&dir);
    let spec = write_panel_spec(&dir);
    let cp = dir.join("cash.rcsr").to_string_lossy().into_owned();
    let first = run_panel(&data, &spec, &[&TRAIN_MEAN[..], &["--generations", "10", "--save", &cp]].concat());
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    let m = cash_of(&json(&first));
    let straight = json(&run_panel(&data, &spec, &[&TRAIN_MEAN[..], &["--generations", "20"]].concat()));
    let resumed = json(&run_panel(&data, &spec, &[&TRAIN_MEAN[..], &["--generations", "20", "--resume", &cp]].concat()));
    assert_eq!(members(&straight), members(&resumed));
    let resume = |extra: &[&str]| run_panel(&data, &spec, &[&["--fitness", "pnl", "--generations", "20", "--resume", &cp][..], extra].concat());
    usage_error(&resume(&[]), "fitness");
    usage_error(&resume(&["--cash-return", &format!("{m:e}")]), "fitness");
}
