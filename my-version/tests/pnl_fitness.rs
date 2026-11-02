//! `--fitness pnl`: the exact long/flat regret book, the search wiring and the
//! CLI refusals (#480).

use rcsr::data::TradingSlice;
use rcsr::devlog::Json;
use rcsr::trading::{self, FitnessBook, TradingRows, TradingSpec};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

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
    let holding = TradingSpec {
        min_holding: 2,
        ..spec()
    };
    assert!(FitnessBook::new(&s, &holding, 900.0).is_err());
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

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-pnl-fitness-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `r = 0.01 x1 + noise` is the realised return; `y = r + 0.05 x2` is the
/// regression target, so the squared error prefers `x2` and the P&L `x1`.
fn write_csv(dir: &std::path::Path, step: i64) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
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
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().expect("run rcsr")
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
    let dir = scratch();
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
    let dir = scratch();
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
    let dir = scratch();
    let data = write_csv(&dir, 900);
    let spec = write_spec(&dir);

    let no_objective = Command::new(env!("CARGO_BIN_EXE_rcsr"))
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
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--min-holding", "2"]),
        "--min-holding",
    );
    usage_error(
        &run(&data, &spec, &["--fitness", "pnl", "--fit-samples", "50"]),
        "--fit-samples",
    );
    usage_error(&run(&data, &spec, &["--fitness", "other"]), "--fitness must be");

    let overlapping = write_csv(&dir, 300);
    usage_error(&run(&overlapping, &spec, &["--fitness", "pnl"]), "overlapping returns");
}

#[test]
fn resume_needs_the_same_book() {
    let dir = scratch();
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
    let dir = scratch();
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
    let dir = scratch();
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
    let dir = scratch();
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
    let dir = scratch();
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
    let dir = scratch();
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
