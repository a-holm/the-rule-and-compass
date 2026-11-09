//! `--fitness pnl`: the exact long/flat regret book, the search wiring and the
//! CLI refusals (#480).

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

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
    assert_eq!(
        String::from_utf8_lossy(&plain.stdout).replace(|c: char| c.is_ascii_digit() || c == '.', ""),
        String::from_utf8_lossy(&one.stdout).replace(|c: char| c.is_ascii_digit() || c == '.', "")
    );
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
    usage_error(&run(&data, &spec, &["--fitness", "pnl", "--pnl-blocks", "9"]), "--pnl-blocks must be");
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
    String::from_utf8_lossy(&o.stdout).replace(|c: char| c.is_ascii_digit() || c == '.', "")
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
