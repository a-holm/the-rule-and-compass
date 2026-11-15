//! #1177: `rcsr trials --table`, the multi-metric candidate table.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use rcsr::rng::Mt;
use rcsr::trials_table::{average_ranks, column_ranks, friedman, pareto_fronts};
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 3600;

fn run(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn stdout_json(o: &Output) -> Json {
    assert_eq!(code(o), 0, "{}", stderr(o));
    Json::parse(String::from_utf8_lossy(&o.stdout).trim()).expect("stdout is JSON")
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}"));
    }
    cur
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number: {:?}", at(j, path)))
}

fn items(j: &Json) -> &[Json] {
    match j {
        Json::Arr(v) => v,
        other => panic!("not an array: {other:?}"),
    }
}

fn scratch(tag: &str) -> (Scratch, PathBuf) {
    let s = Scratch::new(tag);
    let p = s.path().to_path_buf();
    (s, p)
}

fn gauss(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Gaussian noise with its sample mean removed, so a planted mean is exact.
fn centred_noise(n: usize, seed: u32, sd: f64) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let v: Vec<f64> = (0..n).map(|_| sd * gauss(&mut rng)).collect();
    let m = v.iter().sum::<f64>() / n as f64;
    v.iter().map(|x| x - m).collect()
}

fn grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (T0 + STEP * i as i64) as f64).collect()
}

fn arr(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

fn mean(x: &[f64]) -> f64 {
    x.iter().sum::<f64>() / x.len() as f64
}

fn sd(x: &[f64]) -> f64 {
    let m = mean(x);
    (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (x.len() as f64 - 1.0)).sqrt()
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let (ma, mb) = (mean(a), mean(b));
    let cov = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (a.len() as f64 - 1.0);
    cov / (sd(a) * sd(b))
}

/// One synthetic walk-forward report with the #1175 and #1176 series.
struct Fx {
    net: Vec<f64>,
    gross: Option<Vec<f64>>,
    exposure: Option<Vec<f64>>,
    bh_net: Option<Vec<f64>>,
    entry_times: Option<u32>,
    trades: Option<u32>,
    /// Test windows `[start, end)` in seconds.
    windows: Vec<(f64, f64)>,
    /// The raw `protocol` object, when the report carries one.
    protocol: Option<String>,
    ppy: f64,
    time: Vec<f64>,
}

impl Fx {
    fn new(net: Vec<f64>) -> Fx {
        let n = net.len();
        let time = grid(n);
        let end = T0 as f64 + (STEP * n as i64) as f64;
        Fx {
            net,
            gross: None,
            exposure: None,
            bh_net: None,
            entry_times: None,
            trades: None,
            windows: vec![(T0 as f64, end)],
            protocol: None,
            ppy: 8760.0,
            time,
        }
    }
}

fn write_fx(dir: &Path, name: &str, fx: &Fx) -> String {
    let windows: Vec<String> = fx
        .windows
        .iter()
        .enumerate()
        .map(|(k, (s, e))| format!("{{\"label\":\"w{k}\",\"train_start\":0,\"train_end\":1,\"test_start\":{s},\"test_end\":{e}}}"))
        .collect();
    let mut pooled = format!(
        "\"periods\":{},\"entries\":10,\"active_rows\":100,\"period_net\":[{}],\"period_time\":[{}]",
        fx.net.len(),
        arr(&fx.net),
        arr(&fx.time)
    );
    let opt = |key: &str, v: &Option<Vec<f64>>| v.as_ref().map_or(String::new(), |v| format!(",\"{key}\":[{}]", arr(v)));
    pooled.push_str(&opt("period_gross", &fx.gross));
    pooled.push_str(&opt("period_exposure", &fx.exposure));
    pooled.push_str(&opt("period_bh_net", &fx.bh_net));
    if let Some(e) = fx.entry_times {
        pooled.push_str(&format!(",\"entry_times\":{e}"));
    }
    if let Some(t) = fx.trades {
        pooled.push_str(&format!(",\"trades\":{t}"));
    }
    let text = format!(
        "{{\"mode\":\"walkforward\",\"expression\":\"{name}\",\"window_definition\":[{}],\
\"purge\":{{\"horizon_seconds\":3600,\"embargo_seconds\":3600}},\
\"cost_model\":{{\"per_side_bps\":10,\"round_trip_bps\":20}},\
\"cost\":{{\"per_side_bps\":10,\"source\":\"flag\",\"return_units\":\"log\"}},\
\"policy\":{{\"name\":\"threshold\",\"band_width\":0,\"min_hold\":1}},\
\"threshold\":{{\"quantile\":0.5,\"fixed\":null}},\
\"annualization\":{{\"periods_per_year\":{},\"method\":\"explicit\"}},{}\
\"pooled\":{{{}}}}}",
        windows.join(","),
        fx.ppy,
        fx.protocol.as_ref().map_or(String::new(), |p| format!("\"protocol\":{p},")),
        pooled
    );
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_text(dir: &Path, name: &str, text: &str) -> String {
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn row<'a>(j: &'a Json, i: usize) -> &'a Json {
    &items(at(j, &["rows"]))[i]
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Six series over 12 periods and two test windows, with costs, exposure and buy-and-hold net.
fn six_series(dir: &Path) -> (Vec<String>, Vec<Fx>) {
    let n = 12usize;
    let mut paths = Vec::new();
    let mut fxs = Vec::new();
    for i in 0..6usize {
        let fi = i as f64;
        let net: Vec<f64> = (0..n).map(|t| 0.01 * (1.3 * t as f64 + fi).sin() + 0.002 * fi - 0.003).collect();
        let cost: Vec<f64> = (0..n).map(|t| 0.0005 * (1.0 + ((t + i) % 3) as f64)).collect();
        let mut fx = Fx::new(net.clone());
        fx.ppy = 12.0;
        fx.gross = Some(net.iter().zip(&cost).map(|(a, c)| a + c).collect());
        fx.exposure = Some((0..n).map(|t| 0.25 + 0.05 * ((t * i) % 4) as f64).collect());
        fx.bh_net = Some((0..n).map(|t| 0.008 * (0.7 * t as f64 + 2.0 * fi).cos()).collect());
        fx.entry_times = Some(20 + i as u32);
        fx.trades = Some(30 + i as u32);
        let mid = T0 as f64 + (STEP * 6) as f64;
        fx.windows = vec![(T0 as f64, mid), (mid, T0 as f64 + (STEP * 12) as f64)];
        paths.push(write_fx(dir, &format!("s{i}"), &fx));
        fxs.push(fx);
    }
    (paths, fxs)
}

#[test]
fn table_metrics_match_hand_values() {
    let (_s, dir) = scratch("table-metrics");
    let (paths, fxs) = six_series(&dir);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--table", "--draws", "50", "--block-len", "2", "--pbo-blocks", "4"]);
    let j = stdout_json(&run(&args));
    assert_eq!(at(&j, &["schema"]).as_str(), Some("rcsr-trials-table/1"));
    assert_eq!(num(&j, &["trials", "nominal"]), 6.0);
    for (i, fx) in fxs.iter().enumerate() {
        let ppy = fx.ppy;
        let net = &fx.net;
        let gross = fx.gross.as_ref().unwrap();
        let cost: Vec<f64> = gross.iter().zip(net).map(|(g, n)| g - n).collect();
        let e = fx.exposure.as_ref().unwrap();
        let bh = fx.bh_net.as_ref().unwrap();
        let net10 = mean(net) * ppy;
        let net20 = mean(&net.iter().zip(&cost).map(|(n, c)| n - c).collect::<Vec<_>>()) * ppy;
        let sharpe = mean(net) / sd(net) * ppy.sqrt();
        let downside = (net.iter().map(|r| r.min(0.0).powi(2)).sum::<f64>() / net.len() as f64).sqrt();
        let sortino = mean(net) / downside * ppy.sqrt();
        let (mut equity, mut peak, mut mdd) = (1.0f64, 1.0f64, 0.0f64);
        for r in net {
            equity *= 1.0 + r;
            peak = peak.max(equity);
            mdd = mdd.min(equity / peak - 1.0);
        }
        let calmar = net10 / -mdd;
        let alpha = mean(&(0..net.len()).map(|t| net[t] - e[t] * bh[t]).collect::<Vec<_>>()) * ppy;
        let groups_pos = [0..6usize, 6..12].iter().filter(|r| net[r.clone()].iter().sum::<f64>() > 0.0).count();
        let m = at(row(&j, i), &["metrics"]);
        for (name, want) in [
            ("net10", net10),
            ("net20", net20),
            ("sharpe", sharpe),
            ("sortino", sortino),
            ("calmar", calmar),
            ("max_drawdown", mdd),
            ("alpha_bh", alpha),
            ("entry_times", 20.0 + i as f64),
            ("exposure", mean(e)),
            ("groups", 2.0),
            ("groups_pos", groups_pos as f64),
        ] {
            let got = num(m, &[name]);
            assert!((got - want).abs() <= 1e-12, "row {i} {name}: table {got} hand {want}");
        }
    }
}

// ---------------------------------------------------------------------------
// Pareto front, average rank, Friedman
// ---------------------------------------------------------------------------

#[test]
fn pareto_front_is_exact() {
    let v = vec![
        vec![3.0, 1.0, 2.0],
        vec![2.0, 2.0, 2.0],
        vec![1.0, 3.0, 1.0],
        vec![2.0, 1.0, 1.0],
        vec![1.0, 1.0, 1.0],
        vec![2.0, 2.0, 1.0],
        vec![3.0, 1.0, 2.0],
        vec![0.0, 0.0, f64::NAN],
    ];
    let fronts = pareto_fronts(&v);
    assert_eq!(fronts, vec![1, 1, 1, 3, 4, 2, 1, 5]);
    let first: Vec<usize> = (0..v.len()).filter(|&i| fronts[i] == 1).collect();
    assert_eq!(first, vec![0, 1, 2, 6], "planted first front");
}

#[test]
fn average_rank_matches_hand_table() {
    let v = vec![vec![0.3, 1.0, 5.0], vec![0.1, 2.0, 5.0], vec![0.2, 3.0, f64::NAN], vec![0.2, 4.0, 1.0]];
    assert_eq!(column_ranks(&[0.3, 0.1, 0.2, 0.2]), vec![1.0, 4.0, 2.5, 2.5]);
    let (ranks, avg) = average_ranks(&v);
    let hand = [[1.0, 4.0, 1.5], [4.0, 3.0, 1.5], [2.5, 2.0, 4.0], [2.5, 1.0, 3.0]];
    for (i, h) in hand.iter().enumerate() {
        assert_eq!(ranks[i], h.to_vec(), "row {i}");
        assert!((avg[i] - h.iter().sum::<f64>() / 3.0).abs() <= 1e-15, "row {i} average {}", avg[i]);
    }
    let f = friedman(&ranks);
    assert!((f.chi2 - 0.8).abs() <= 1e-12, "chi2 {}", f.chi2);
    assert_eq!(f.df, 3);
    assert!((f.p - rcsr::diagnostics::chi2_sf(0.8, 3)).abs() <= 1e-12, "p {}", f.p);
}

// ---------------------------------------------------------------------------
// Gates, rank and the diversity walk through the CLI
// ---------------------------------------------------------------------------

/// Five series with exact means 0.0020 > 0.0018 > 0.0016 > 0.0014 > 0.0012; the second is
/// 0.8 x the first's noise plus 0.6 x its own.
fn five_series(dir: &Path) -> (Vec<String>, Vec<Vec<f64>>) {
    let n = 2000;
    let base = centred_noise(n, 11, 0.01);
    let own = centred_noise(n, 12, 0.01);
    let means = [0.0020, 0.0018, 0.0016, 0.0014, 0.0012];
    let mut nets: Vec<Vec<f64>> = Vec::new();
    for (k, m) in means.iter().enumerate() {
        let noise: Vec<f64> = match k {
            0 => base.clone(),
            1 => base.iter().zip(&own).map(|(a, b)| 0.8 * a + 0.6 * b).collect(),
            _ => centred_noise(n, 20 + k as u32, 0.01),
        };
        nets.push(noise.iter().map(|x| x + m).collect());
    }
    let paths = nets.iter().enumerate().map(|(k, v)| write_fx(dir, &format!("d{k}"), &Fx::new(v.clone()))).collect();
    (paths, nets)
}

#[test]
fn diversity_cap_skips_near_duplicates() {
    let (_s, dir) = scratch("table-diversity");
    let (paths, nets) = five_series(&dir);
    let rho = corr(&nets[0], &nets[1]);
    assert!((0.75..0.85).contains(&rho), "planted correlation {rho}");
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--table", "--rank", "net10", "--pick", "3", "--diversity-rho", "0.7", "--draws", "50"]);
    let j = stdout_json(&run(&args));
    let picks: Vec<usize> = items(at(&j, &["picks"])).iter().map(|p| num(p, &["row"]) as usize).collect();
    assert_eq!(picks, vec![0, 2, 3], "{j:?}");
    let skipped = at(row(&j, 1), &["skipped"]);
    assert_eq!(num(skipped, &["row"]), 0.0);
    assert!(num(skipped, &["rho"]) >= 0.7);
    assert!(matches!(at(row(&j, 4), &["pick"]), Json::Null));
    // The walk is the rank order.
    for (i, want) in [1.0, 2.0, 3.0, 4.0, 5.0].iter().enumerate() {
        assert_eq!(num(row(&j, i), &["avg_rank"]), *want, "row {i}");
    }
}

#[test]
fn gates_list_every_failure_and_fronts_rank_the_passers() {
    let (_s, dir) = scratch("table-gates");
    let (paths, _) = five_series(&dir);
    let rule = write_text(
        &dir,
        "rule.json",
        "{\"schema\":\"rcsr-trials-rule/1\",\"cost_multiples\":[1],\
\"gates\":{\"net10\":[\">\",13],\"sharpe\":[\"between\",0,100]},\
\"pareto\":{\"net10\":\"max\",\"sharpe\":\"max\"},\"rank\":[\"net10\",\"sharpe\"],\"pick\":2,\"diversity_rho\":0.95}",
    );
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--table", "--rule", &rule, "--draws", "50"]);
    let j = stdout_json(&run(&args));
    let text = std::fs::read(&rule).unwrap();
    assert_eq!(at(&j, &["rule", "sha256"]).as_str(), Some(rcsr::manifest::sha256_hex(&text).as_str()));
    // net10 = 8760 x mean: 17.52, 15.768, 14.016, 12.264, 10.512; the last two fail the net gate.
    let passes: Vec<bool> = (0..5).map(|i| at(row(&j, i), &["gates", "pass"]) == &Json::Bool(true)).collect();
    assert_eq!(passes, vec![true, true, true, false, false]);
    let failed = items(at(row(&j, 4), &["gates", "failed"]));
    assert_eq!(failed.len(), 1);
    assert_eq!(at(&failed[0], &["metric"]).as_str(), Some("net10"));
    assert_eq!(num(&j, &["gate_passers"]), 3.0);
    assert_eq!(num(row(&j, 0), &["front"]), 1.0);
    assert!(matches!(at(row(&j, 3), &["front"]), Json::Null));
    assert!(num(&j, &["rank", "friedman", "chi2"]).is_finite());
    assert_eq!(items(at(&j, &["picks"])).len(), 2);
    // --rule holds the whole rule.
    let mut bad = args.clone();
    bad.extend(["--pick", "1"]);
    let o = run(&bad);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--rule"), "{}", stderr(&o));
}

// ---------------------------------------------------------------------------
// Deflation
// ---------------------------------------------------------------------------

const DN: usize = 400;

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn write_data(dir: &Path) -> String {
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..DN {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let e = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * e;
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r, r));
    }
    write_text(dir, "data.csv", &text)
}

fn walkforward(dir: &Path, data: &str, name: &str, extra: &[&str]) -> String {
    let out = dir.join(format!("{name}.json")).to_string_lossy().into_owned();
    let w = format!("v,{},{},{},{}", T0, T0 + 900 * 309, T0 + 900 * 310, T0 + 900 * 400);
    let mut args = vec![
        "walkforward", "--data", data, "--time", "t", "--target", "y", "--expr", "x1", "--vars", "x1,x2",
        "--window", &w, "--threshold-quantile", "0.5", "--periods-per-year", "35040", "--out", &out,
    ];
    args.extend_from_slice(extra);
    let o = run(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    out
}

fn read_json(path: &str) -> Json {
    Json::parse(std::fs::read_to_string(path).unwrap().trim()).unwrap()
}

#[test]
fn table_dsr_matches_walkforward() {
    let (_s, dir) = scratch("table-dsr");
    let data = write_data(&dir);
    for n in [1usize, 5] {
        let trials = n.to_string();
        let wf = walkforward(&dir, &data, &format!("wf{n}"), &["--bootstrap", "200", "--block-len", "12", "--trials", &trials]);
        let want = num(&read_json(&wf), &["inference", "multiple_testing", "dsr_at_n_eff", "dsr"]);
        assert!(want.is_finite());
        let prior = (n - 1).to_string();
        let j = stdout_json(&run(&[
            "trials", "--wf", &wf, "--table", "--pick", "1", "--prior-trials", &prior, "--block-len", "12", "--draws", "200",
        ]));
        let pick = &items(at(&j, &["picks"]))[0];
        assert_eq!(num(pick, &["dsr_nominal", "n_eff"]), n as f64);
        let got = num(pick, &["dsr_nominal", "value"]);
        assert!((got - want).abs() <= 1e-12, "N {n}: table {got} walkforward {want}");
    }
}

fn four_noisy(dir: &Path) -> Vec<String> {
    (0..4u32)
        .map(|k| {
            let net: Vec<f64> = centred_noise(2000, 40 + k, 0.01).iter().map(|x| x + 0.0004 - 0.0001 * k as f64).collect();
            write_fx(dir, &format!("n{k}"), &Fx::new(net))
        })
        .collect()
}

#[test]
fn prior_trials_change_only_nominal_dsr() {
    let (_s, dir) = scratch("table-prior");
    let paths = four_noisy(&dir);
    let table = |extra: &[&str]| {
        let mut args = vec!["trials", "--wf"];
        args.extend(paths.iter().map(String::as_str));
        args.extend(["--table", "--rank", "sharpe", "--pick", "1", "--draws", "200"]);
        args.extend_from_slice(extra);
        stdout_json(&run(&args))
    };
    let plain = table(&[]);
    let prior = table(&["--prior-trials", "51"]);
    let pick = |j: &Json| items(at(j, &["picks"]))[0].clone();
    let (a, b) = (pick(&plain), pick(&prior));
    assert_eq!(num(&a, &["dsr", "value"]), num(&b, &["dsr", "value"]));
    assert_eq!(num(&a, &["dsr_10x", "value"]), num(&b, &["dsr_10x", "value"]));
    assert_eq!(num(&a, &["dsr_nominal", "n_eff"]), 4.0);
    assert_eq!(num(&b, &["dsr_nominal", "n_eff"]), 55.0);
    assert!(num(&b, &["dsr_nominal", "value"]) < num(&a, &["dsr_nominal", "value"]), "{a:?} {b:?}");
    assert_eq!(num(&prior, &["trials", "nominal"]), 4.0);
    assert_eq!(num(&prior, &["trials", "prior_trials"]), 51.0);
}

#[test]
fn stepm_finds_planted_edges() {
    let n = 500;
    let mut exact = 0;
    for draw in 0..100u32 {
        let mut rng = Mt::new(777 + draw);
        let mut d: Vec<Vec<f64>> = (0..10).map(|_| (0..n).map(|_| gauss(&mut rng)).collect()).collect();
        for _ in 0..2 {
            d.push((0..n).map(|_| 0.35 + gauss(&mut rng)).collect());
        }
        let s = rcsr::inference::stepm(&d, 5, 300, 9000 + draw, 0.05).unwrap();
        let want: Vec<bool> = (0..12).map(|k| k >= 10).collect();
        if s.rejected == want {
            exact += 1;
        }
    }
    assert!(exact >= 90, "StepM rejected exactly the planted edges in {exact} of 100 draws");
}

// ---------------------------------------------------------------------------
// Refusals
// ---------------------------------------------------------------------------

#[test]
fn table_gate_needs_its_field() {
    let (_s, dir) = scratch("table-field");
    let a = write_fx(&dir, "plain_a", &Fx::new(centred_noise(200, 1, 0.01)));
    let b = write_fx(&dir, "plain_b", &Fx::new(centred_noise(200, 2, 0.01)));
    let gates = write_text(&dir, "gates.json", "{\"alpha_bh\":[\">\",0]}");
    let o = run(&["trials", "--wf", &a, &b, "--table", "--gates", &gates, "--draws", "20"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("alpha_bh") && e.contains("period_bh_net") && e.contains(&a), "{e}");
    // A stratum label missing from the calendar, a null quantile without a null set and an unknown metric.
    let cal = write_text(&dir, "cal.csv", &format!("label,start,end\nbear,{},{}\n", T0, T0 + 3600 * 100));
    let gates = write_text(&dir, "bull.json", "{\"bull_net\":[\">\",0]}");
    let o = run(&["trials", "--wf", &a, &b, "--table", "--gates", &gates, "--regimes", &cal, "--draws", "20"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("'bull'"), "{}", stderr(&o));
    let o = run(&["trials", "--wf", &a, &b, "--table", "--rank", "null_q", "--draws", "20"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--null-wf"), "{}", stderr(&o));
    let o = run(&["trials", "--wf", &a, &b, "--table", "--rank", "net30", "--draws", "20"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("unknown metric 'net30'"), "{}", stderr(&o));
    // Table flags need --table.
    let o = run(&["trials", "--wf", &a, &b, "--pick", "2"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--pick needs --table"), "{}", stderr(&o));
}

#[test]
fn table_refuses_locked_reports() {
    let (_s, dir) = scratch("table-locked");
    let n = 200;
    let clean = write_fx(&dir, "clean", &Fx::new(centred_noise(n, 3, 0.01)));
    let start = T0 + 3600 * 150;
    let end = T0 + 3600 * 400;
    let mut read = Fx::new(centred_noise(n, 4, 0.01));
    read.protocol = Some("{\"lockbox\":{\"rows\":0,\"evaluated\":false},\"test_read\":true}".to_string());
    let read = write_fx(&dir, "sealed_read", &read);
    let mut rows = Fx::new(centred_noise(n, 5, 0.01));
    rows.protocol = Some(format!(
        "{{\"lockbox\":{{\"declared\":{{\"start\":{start},\"end\":{end}}},\"rows_in_scope\":0,\"evaluated\":false}},\"test_read\":false}}"
    ));
    let rows = write_fx(&dir, "locked_rows", &rows);
    for bad in [&read, &rows] {
        let o = run(&["trials", "--wf", &clean, bad, "--table", "--draws", "20"]);
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        assert!(stderr(&o).contains(bad.as_str()) && stderr(&o).contains("locked"), "{}", stderr(&o));
    }
    let other = write_fx(&dir, "other", &Fx::new(centred_noise(n, 6, 0.01)));
    let ok = run(&["trials", "--wf", &clean, &other, "--table", "--draws", "20"]);
    assert_eq!(code(&ok), 0, "{}", stderr(&ok));
    let range = format!("{start},{end}");
    let o = run(&["trials", "--wf", &clean, &other, "--table", "--locked-test", &range, "--draws", "20"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains(clean.as_str()) && stderr(&o).contains("locked"), "{}", stderr(&o));
    // The campaign mode reads a sealed read's strata as before.
    let o = run(&["trials", "--wf", &clean, &read, "--draws", "20"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

// ---------------------------------------------------------------------------
// Members, calendar, determinism and the unchanged default output
// ---------------------------------------------------------------------------

/// #1176: a `walkforward --front --all-members` report on a 3-symbol panel and a front of 3 members.
fn members_report(dir: &Path) -> String {
    let mut csv = String::from("ts,sym,x,target\n");
    for i in 0..300i64 {
        for (s, sym) in ["A", "B", "C"].iter().enumerate() {
            let s = s as i64;
            let x = ((i * 7 + s * 3) % 11) as f64 - 5.0;
            let y = (((i * 5 + s * 7) % 13) as f64 - 6.0) / 4096.0;
            csv.push_str(&format!("{},{},{},{}\n", i * 3600, sym, x, y));
        }
    }
    let data = write_text(dir, "panel.csv", &csv);
    let front = write_text(
        dir,
        "run.json",
        r#"{"tool":"rcsr","variables":["x"],"target_spec":{"return_column":"target"},
        "split":{"train_time":[0,356400],"validation_time":[360000,536400],"required_purge_seconds":3600,"purge_seconds":3600},
        "objective":{"spec":{"threshold":0,"per_side_bps":10}},
        "pareto":[{"expression":"x"},{"expression":"0.5 - x"},{"expression":"sqrt(x) - 1"}]}"#,
    );
    let out = dir.join("wf_members.json");
    let o = run(&[
        "walkforward", "--data", &data, "--time", "ts", "--group-by", "sym", "--horizon", "3600", "--periods-per-year", "8760",
        "--front", &front, "--all-members", "--period-grid", "target", "--window", "w,0,540000,543600,1080000", "--json", "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    out.to_string_lossy().into_owned()
}

fn series_of(m: &Json, key: &str) -> Vec<f64> {
    items(at(m, &["series", key])).iter().map(|v| v.as_f64().unwrap()).collect()
}

#[test]
fn table_reads_members_and_calendar() {
    let (_s, dir) = scratch("table-members");
    let report = members_report(&dir);
    let cal = write_text(&dir, "cal.csv", "label,start,end\nbull,0,800000\nbear,800000,1100000\n");
    let rule = write_text(
        &dir,
        "rule.json",
        "{\"schema\":\"rcsr-trials-rule/1\",\"gates\":{\"entry_times\":[\">=\",0]},\"pareto\":{\"net20\":\"max\",\"bull_net\":\"max\"},\
\"rank\":[\"net20\",\"sharpe\",\"bear_excess\"],\"pick\":2,\"diversity_rho\":0.99}",
    );
    let md = dir.join("table.md");
    let j = stdout_json(&run(&[
        "trials", "--wf", &report, "--members", "all", "--table", "--regimes", &cal, "--rule", &rule, "--draws", "50", "--markdown",
        md.to_str().unwrap(),
    ]));
    let r = read_json(&report);
    let members = items(at(&r, &["members"]));
    assert_eq!(items(at(&j, &["rows"])).len(), 3);
    let ppy = 8760.0;
    for (k, m) in members.iter().enumerate() {
        let net = series_of(m, "period_net");
        let cost = series_of(m, "period_cost");
        let time = series_of(m, "period_time");
        let bh = series_of(m, "period_bh_net");
        let x = row(&j, k);
        assert_eq!(at(x, &["label"]).as_str().unwrap(), format!("wf_members#{}", k + 1));
        let net20 = mean(&net.iter().zip(&cost).map(|(a, c)| a - c).collect::<Vec<_>>()) * ppy;
        assert!((num(x, &["metrics", "net20"]) - net20).abs() <= 1e-12);
        let bull: Vec<f64> = (0..net.len()).filter(|&t| time[t] < 800000.0).map(|t| net[t]).collect();
        let bear: Vec<usize> = (0..net.len()).filter(|&t| time[t] >= 800000.0).collect();
        assert!((num(x, &["metrics", "bull_net"]) - mean(&bull) * ppy).abs() <= 1e-12);
        let excess = mean(&bear.iter().map(|&t| net[t] - bh[t]).collect::<Vec<_>>()) * ppy;
        assert!((num(x, &["metrics", "bear_excess"]) - excess).abs() <= 1e-12);
        let labels: Vec<&str> = items(at(x, &["strata"])).iter().map(|s| at(s, &["label"]).as_str().unwrap()).collect();
        assert_eq!(labels, vec!["bull", "bear"]);
        assert_eq!(num(x, &["metrics", "entry_times"]), num(m, &["pooled", "entry_times"]));
    }
    assert_eq!(num(&j, &["trials", "nominal"]), 3.0);
    assert!(num(&j, &["trials", "n_eff"]) >= 1.0);
    let text = std::fs::read_to_string(&md).unwrap();
    assert!(text.contains("| wf_members#1 |") && text.contains("Rule sha256"), "{text}");
}

#[test]
fn table_is_byte_identical_across_runs() {
    let (_s, dir) = scratch("table-determinism");
    let (paths, _) = five_series(&dir);
    let once = |tag: &str| {
        let out = dir.join(format!("{tag}.json"));
        let md = dir.join(format!("{tag}.md"));
        let mut args = vec!["trials", "--wf"];
        args.extend(paths.iter().map(String::as_str));
        args.extend([
            "--table", "--pareto", "net10:max,sharpe:max", "--rank", "net10,sortino", "--pick", "3", "--diversity-rho", "0.7",
            "--draws", "100", "--out", out.to_str().unwrap(), "--markdown", md.to_str().unwrap(),
        ]);
        let o = run(&args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        (std::fs::read(&out).unwrap(), std::fs::read(&md).unwrap())
    };
    let (a, b) = (once("a"), once("b"));
    assert_eq!(a.0, b.0, "JSON differs between two runs");
    assert_eq!(a.1, b.1, "Markdown differs between two runs");
}

/// The output of `rcsr trials` without `--table` on fixed synthetic reports, pinned to the
/// SHA-256 of the integration base before #1177.
#[test]
fn trials_default_output_is_unchanged() {
    let (_s, dir) = scratch("table-golden");
    let (paths, _) = five_series(&dir);
    let time = grid(2000);
    let cash = vec![0.0; 2000];
    let hold: Vec<f64> = centred_noise(2000, 99, 0.01).iter().map(|x| x + 0.0003).collect();
    let ladder = write_text(
        &dir,
        "ladder.json",
        &format!(
            "{{\"schema\":\"rcsr-ladder/1\",\"series\":{{\"period_time\":[{}],\"period_net\":{{\"cash\":[{}],\"buy_and_hold\":[{}]}}}}}}",
            arr(&time),
            arr(&cash),
            arr(&hold)
        ),
    );
    let cal = write_text(&dir, "cal.csv", &format!("label,start,end\nbull,{},{}\nbear,{},{}\n", T0, T0 + 3600 * 900, T0 + 3600 * 900, T0 + 3600 * 1800));
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--baseline", &ladder, "--regimes", &cal, "--draws", "200", "--pbo-blocks", "4", "--cluster-rho", "0.7"]);
    let campaign = run(&args);
    assert_eq!(code(&campaign), 0, "{}", stderr(&campaign));
    let compare = run(&["trials", "--compare", &paths[0], &paths[1], "--draws", "200"]);
    assert_eq!(code(&compare), 0, "{}", stderr(&compare));
    let mut bytes = campaign.stdout.clone();
    bytes.extend_from_slice(&compare.stdout);
    assert_eq!(rcsr::manifest::sha256_hex(&bytes), "PINNED_FROM_BASE");
}
