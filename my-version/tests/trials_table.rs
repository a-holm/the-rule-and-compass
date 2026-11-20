//! #1177: `rcsr trials --table`, the multi-metric candidate table.

#[path = "common/cpcv_series.rs"]
mod cpcv_series;
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
    /// The raw `selection` object, when the report carries one.
    selection: Option<String>,
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
            selection: None,
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
\"annualization\":{{\"periods_per_year\":{},\"method\":\"explicit\"}},{}{}\
\"pooled\":{{{}}}}}",
        windows.join(","),
        fx.ppy,
        fx.protocol.as_ref().map_or(String::new(), |p| format!("\"protocol\":{p},")),
        fx.selection.as_ref().map_or(String::new(), |s| format!("\"selection\":{s},")),
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
        let groups_pos = [0..6usize, 6..12].into_iter().filter(|r| net[r.clone()].iter().sum::<f64>() > 0.0).count();
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
    assert_eq!(at(skipped, &["reason"]).as_str(), Some("diversity"));
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
    assert_eq!(rcsr::manifest::sha256_hex(&bytes), "21cde625be2b0f3b294574b3a3bd59184770d9b1411823a72b4ecd2e436314d9");
}

// ---------------------------------------------------------------------------
// CPCV path rows (#1236)
// ---------------------------------------------------------------------------

/// The paths of a CPCV(4, 2) assembly of synthetic split reports with every series, the
/// split reports and the scheme.
fn cpcv_paths(dir: &Path) -> (Vec<String>, Vec<Json>, rcsr::cpcv::Scheme) {
    let (table, scheme) = cpcv_series::write_table(dir);
    let reports = cpcv_series::write_reports(dir, "wf", &scheme, &|_| cpcv_series::ALL);
    let out = dir.join("paths");
    let (o, sm) = (out.to_string_lossy().into_owned(), dir.join("summary.json").to_string_lossy().into_owned());
    let mut args = vec!["trials", "--cpcv-assemble"];
    args.extend(reports.iter().map(String::as_str));
    args.extend_from_slice(&["--splits", &table, "--out-dir", &o, "--summary", &sm]);
    let r = run(&args);
    assert_eq!(code(&r), 0, "{}", stderr(&r));
    let paths = (1..=scheme.paths()).map(|j| out.join(format!("path_{j}.json")).to_string_lossy().into_owned()).collect();
    (paths, reports.iter().map(|p| read_json(p)).collect(), scheme)
}

fn bear_cell(x: &Json) -> &Json {
    items(at(x, &["strata"])).iter().find(|c| at(c, &["label"]).as_str() == Some("bear")).expect("a bear stratum")
}

#[test]
fn cpcv_path_table_metrics_match_group_series() {
    let (_s, dir) = scratch("table-cpcv");
    let (paths, wf, scheme) = cpcv_paths(&dir);
    let cal = cpcv_series::write_calendar(&dir);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--table", "--regimes", &cal, "--draws", "50"]);
    let j = stdout_json(&run(&args));
    assert_eq!(items(at(&j, &["rows"])).len(), 3);
    let ppy = 8760.0;
    let bear_from = cpcv_series::day("2021-01-07") as f64;
    let assign = rcsr::cpcv::assignment(&scheme.test_sets, scheme.groups.len());
    for k in 0..paths.len() {
        let v = |key: &str| cpcv_series::path_values(&wf, &scheme, k, key);
        let (net, cost, exposure, bh, time) = (v("period_net"), v("period_cost"), v("period_exposure"), v("period_bh_net"), v("period_time"));
        let ann = |idx: &[usize], f: &dyn Fn(usize) -> f64| idx.iter().map(|&t| f(t)).sum::<f64>() / idx.len() as f64 * ppy;
        let all: Vec<usize> = (0..net.len()).collect();
        let bull: Vec<usize> = all.iter().copied().filter(|&t| time[t] < bear_from).collect();
        let bear: Vec<usize> = all.iter().copied().filter(|&t| time[t] >= bear_from).collect();
        assert!(!bull.is_empty() && !bear.is_empty());
        let groups_pos = (0..scheme.groups.len())
            .filter(|&g| cpcv_series::group_values(&wf[assign[g][k]], &scheme.groups[g], "period_net").iter().sum::<f64>() > 0.0)
            .count();
        let want = [
            ("net10", ann(&all, &|t| net[t])),
            ("net20", ann(&all, &|t| net[t] - cost[t])),
            ("exposure", mean(&exposure)),
            ("entry_times", cpcv_series::path_entry_times(&wf, &scheme, k)),
            ("alpha_bh", ann(&all, &|t| net[t] - exposure[t] * bh[t])),
            ("groups_pos", groups_pos as f64),
            ("bull_net", ann(&bull, &|t| net[t])),
            ("bear_excess", ann(&bear, &|t| net[t]) - ann(&bear, &|t| bh[t])),
        ];
        let x = row(&j, k);
        for (name, w) in want {
            let got = num(x, &["metrics", name]);
            assert!((got - w).abs() <= 1e-12, "path {} {name}: table {got} group series {w}", k + 1);
        }
        let w = ann(&bear, &|t| net[t] - exposure[t] * bh[t]);
        let got = num(bear_cell(x), &["alpha_bh"]);
        assert!((got - w).abs() <= 1e-12, "path {} alpha_bh:bear: table {got} group series {w}", k + 1);
    }
}

#[test]
fn pilot_rule_runs_on_cpcv_paths() {
    let (_s, dir) = scratch("table-cpcv-pilot");
    let (paths, _, _) = cpcv_paths(&dir);
    let cal = cpcv_series::write_calendar(&dir);
    let rule = write_text(&dir, "rule.json", cpcv_series::PILOT_RULE);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend(["--table", "--rule", &rule, "--regimes", &cal, "--draws", "50"]);
    let j = stdout_json(&run(&args));
    assert_eq!(items(at(&j, &["rows"])).len(), 3);
    for i in 0..3 {
        let x = row(&j, i);
        assert!(items(at(x, &["unavailable"])).is_empty(), "row {i}: {:?}", at(x, &["unavailable"]));
        for m in cpcv_series::PILOT_METRICS {
            assert!(num(x, &["metrics", m]).is_finite(), "row {i}: {m}");
        }
        assert!(num(bear_cell(x), &["alpha_bh"]).is_finite(), "row {i}: alpha_bh:bear");
    }
}

// ---------------------------------------------------------------------------
// Frozen sealed-read thresholds and the composite cap (#1237)
// ---------------------------------------------------------------------------

/// 2020-01-01T00:00:00Z.
const F0: i64 = 1_577_836_800;
const DAY: i64 = 86_400;
const F_NET1: [f64; 6] = [0.01, -0.02, 0.03, 0.00, -0.01, 0.02];
const F_NET2: [f64; 6] = [0.02, 0.01, -0.03, 0.01, 0.00, -0.02];
const SHA_S1: &str = "1111111111111111111111111111111111111111111111111111111111111111";
const SHA_S2: &str = "2222222222222222222222222222222222222222222222222222222222222222";
const SHA_S3: &str = "3333333333333333333333333333333333333333333333333333333333333333";
const SCHEME: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const SPLITS: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

/// One path report of fixture F: CPCV(3, 2) with groups G1 = days 0-2, G2 = days 2-4 and
/// G3 = days 4-6 from 2020-01-01, splits s1 = (G1, G2), s2 = (G1, G3), s3 = (G2, G3), six
/// daily periods; path 1 takes s1, s1, s2 and path 2 takes s2, s3, s3.
#[derive(Clone)]
struct FPath {
    index: usize,
    net: Vec<f64>,
    cost: Option<Vec<f64>>,
    bps: f64,
    splits_sha: &'static str,
    test_groups: usize,
    segments: [(&'static str, &'static str); 3],
}

fn f_path(index: usize) -> FPath {
    let (net, segments) = if index == 1 {
        (F_NET1, [("s1", SHA_S1), ("s1", SHA_S1), ("s2", SHA_S2)])
    } else {
        (F_NET2, [("s2", SHA_S2), ("s3", SHA_S3), ("s3", SHA_S3)])
    };
    FPath { index, net: net.to_vec(), cost: None, bps: 10.0, splits_sha: SPLITS, test_groups: 2, segments }
}

fn write_f_path(dir: &Path, name: &str, p: &FPath) -> String {
    let segs: Vec<String> = (0..3)
        .map(|g| {
            let (split, sha) = p.segments[g];
            let (a, b) = (F0 + 2 * DAY * g as i64, F0 + 2 * DAY * (g as i64 + 1));
            format!(
                "{{\"group\":\"G{}\",\"start\":{a},\"end\":{b},\"split\":\"{split}\",\"report\":\"wf_{split}.json\",\"report_sha256\":\"{sha}\",\
\"expression\":[\"x\"],\"periods\":2,\"net\":{}}}",
                g + 1,
                p.net[2 * g] + p.net[2 * g + 1]
            )
        })
        .collect();
    let time: Vec<f64> = (0..6).map(|d| (F0 + DAY * d) as f64).collect();
    let cost = p.cost.as_ref().map_or(String::new(), |c| format!(",\"period_cost\":[{}]", arr(c)));
    let text = format!(
        "{{\"tool\":\"rcsr\",\"mode\":\"cpcv_path\",\"schema\":\"rcsr-cpcv-path/1\",\"window_definition\":\"{SCHEME}\",\
\"purge\":{{\"horizon_seconds\":86400,\"embargo_seconds\":0}},\"cost_model\":{{\"per_side_bps\":{bps},\"round_trip_bps\":{rt}}},\
\"cost\":{{\"per_side_bps\":{bps},\"source\":\"flag\"}},\
\"annualization\":{{\"periods_per_year\":365,\"method\":\"explicit\",\"source\":\"explicit\",\"period_seconds\":86400}},\
\"threshold\":{{\"quantile\":0.5,\"fixed\":0}},\"policy\":{{\"name\":\"threshold\",\"band_width\":0,\"min_hold\":1}},\
\"book\":{{\"missing_score\":\"flat\"}},\"pooled_seam\":{{\"reset_flat\":true,\"seam_cost\":0,\"seam_trades\":0}},\
\"cpcv\":{{\"scheme_sha256\":\"{SCHEME}\",\"splits_sha256\":\"{splits}\",\"groups\":3,\"test_groups\":{k},\"splits\":3,\"paths\":2,\
\"path\":{index},\"seam_rule\":\"flat\",\"segments\":[{segs}]}},\
\"pooled\":{{\"periods\":6,\"periods_per_year\":365,\"period_net\":[{net}],\"period_time\":[{time}]{cost}}}}}\n",
        bps = p.bps,
        rt = 2.0 * p.bps,
        splits = p.splits_sha,
        k = p.test_groups,
        index = p.index,
        segs = segs.join(","),
        net = arr(&p.net),
        time = arr(&time),
    );
    write_text(dir, name, &text)
}

/// The two path files of fixture F as written by `write_f_path`.
fn f_paths(dir: &Path, tag: &str, edit: &dyn Fn(&mut FPath)) -> Vec<String> {
    [1usize, 2]
        .iter()
        .map(|&k| {
            let mut p = f_path(k);
            edit(&mut p);
            write_f_path(dir, &format!("{tag}_path_{k}.json"), &p)
        })
        .collect()
}

fn f_rule(cpcv: (usize, usize), s4_percentile: u32, floor: f64) -> String {
    format!(
        "{{\"schema\":\"rcsr-trials-rule/1\",\"pick\":1,\"freeze\":{{\"cpcv\":[{},{}],\"cost_bps\":10,\
\"s4\":{{\"window_days\":3,\"percentile\":{s4_percentile}}},\"p1\":{{\"window_days\":2,\"percentile\":10}},\
\"drawdown_cap\":{{\"window_days\":3,\"percentile\":90,\"floor\":{floor}}}}}}}",
        cpcv.0, cpcv.1
    )
}

/// Two candidate rows on an hourly grid from 2020-09-13 (outside fixture F's days).
fn freeze_rows(dir: &Path) -> Vec<String> {
    (0..2u32).map(|k| write_fx(dir, &format!("row{k}"), &Fx::new(centred_noise(200, 60 + k, 0.01)))).collect()
}

fn write_map(dir: &Path, name: &str, rows: &[(&str, &[String], Option<&str>)]) -> String {
    let list: Vec<String> = rows
        .iter()
        .map(|(row, paths, run)| {
            let files: Vec<String> = paths.iter().map(|p| format!("\"{p}\"")).collect();
            let run = run.map_or(String::new(), |r| format!(",\"run\":\"{r}\""));
            format!("{{\"row\":\"{row}\",\"configuration\":\"F\",\"paths\":[{}]{run}}}", files.join(","))
        })
        .collect();
    write_text(dir, name, &format!("{{\"schema\":\"rcsr-trials-freeze-map/1\",\"rows\":[{}]}}", list.join(",")))
}

fn freeze_run(rows: &[String], rule: &str, map: &str, extra: &[&str]) -> Output {
    let mut args = vec!["trials", "--wf"];
    args.extend(rows.iter().map(String::as_str));
    args.extend(["--table", "--rule", rule, "--freeze-map", map, "--draws", "20"]);
    args.extend_from_slice(extra);
    run(&args)
}

/// The `freeze` object of the first pick of a table over fixture F.
fn f_freeze(tag: &str, rule_text: &str, edit: &dyn Fn(&mut FPath)) -> Json {
    let (_s, dir) = scratch(tag);
    let rows = freeze_rows(&dir);
    let paths = f_paths(&dir, "f", edit);
    let rule = write_text(&dir, "rule.json", rule_text);
    let map = write_map(&dir, "map.json", &[(rows[0].as_str(), &paths[..], None)]);
    let j = stdout_json(&freeze_run(&rows, &rule, &map, &[]));
    let picks = items(at(&j, &["picks"]));
    assert_eq!(picks.len(), 1);
    assert_eq!(num(&picks[0], &["row"]), 0.0);
    at(&picks[0], &["freeze"]).clone()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12, "{what}: {got} vs {want}");
}

#[test]
fn nearest_rank_matches_hand_ranks() {
    let want: [(usize, [usize; 5]); 3] = [(1, [1, 1, 1, 1, 1]), (8, [1, 1, 3, 8, 8]), (10, [1, 1, 3, 9, 10])];
    for (n, ranks) in want {
        for (p, k) in [1u32, 10, 30, 90, 99].iter().zip(ranks) {
            assert_eq!(rcsr::inference::nearest_rank(n, *p), k, "n {n} P {p}");
        }
    }
}

#[test]
fn freeze_window_enumerator_matches_hand_windows() {
    let days = |d: &[i64]| d.iter().map(|x| (F0 + DAY * x) as f64).collect::<Vec<f64>>();
    let t = days(&[0, 1, 2, 3, 4, 5]);
    let end = (F0 + 6 * DAY) as f64;
    let d = DAY as f64;
    assert_eq!(rcsr::trials_freeze::windows(&t, end, 3.0 * d), vec![(0, 3), (1, 4), (2, 5), (3, 6)]);
    assert_eq!(rcsr::trials_freeze::windows(&t, end, 2.0 * d), vec![(0, 2), (1, 3), (2, 4), (3, 5), (4, 6)]);
    assert!(rcsr::trials_freeze::windows(&t, end, 7.0 * d).is_empty());
    // A gap: a window holds the periods of its calendar span only.
    assert_eq!(rcsr::trials_freeze::windows(&days(&[0, 1, 4, 5]), end, 3.0 * d), vec![(0, 2), (1, 2)]);
}

#[test]
fn drawdown_tree_matches_sequential_push() {
    let mut rng = Mt::new(77);
    let mut x: Vec<f64> = (0..300).map(|_| 0.02 * gauss(&mut rng)).collect();
    x[150] = -1.0;
    let tree = rcsr::trials_freeze::DrawdownTree::new(&x);
    let mut ranges: Vec<(usize, usize)> = vec![(0, 300), (0, 1), (299, 300), (10, 10)];
    for a in (0..300).step_by(7) {
        ranges.extend((a + 1..=300).step_by(5).map(|b| (a, b)));
    }
    for (a, b) in ranges {
        let mut d = rcsr::risk::Drawdown::identity();
        for &v in &x[a..b] {
            d.push(v);
        }
        let q = tree.query(a, b);
        assert_eq!(q.n, d.n, "[{a}, {b})");
        if b > a {
            close(q.max_drawdown(), d.max_drawdown(), &format!("[{a}, {b})"));
        }
    }
}

#[test]
fn freeze_windows_match_naive_loop() {
    let w_days = 30u32;
    let span = (DAY * w_days as i64) as f64;
    for seed in 0..3u32 {
        let mut rng = Mt::new(500 + seed);
        let mut time = Vec::new();
        let mut h = 0i64;
        while time.len() < 2000 {
            if !(seed == 1 && (700..760).contains(&h)) {
                time.push((T0 + STEP * h) as f64);
            }
            h += 1;
        }
        let x: Vec<f64> = (0..2000).map(|t| 0.01 * gauss(&mut rng) - if t % 97 == 0 { 0.05 } else { 0.0 }).collect();
        let end = time[1999] + STEP as f64;
        let got = rcsr::trials_freeze::rolling_windows(&time, &x, end, w_days);
        let mut want: Vec<(usize, usize, f64, f64)> = Vec::new();
        for i in 0..time.len() {
            if time[i] + span > end {
                continue;
            }
            let idx: Vec<usize> = (0..time.len()).filter(|&j| time[j] >= time[i] && time[j] < time[i] + span).collect();
            let m = idx.iter().map(|&j| x[j]).sum::<f64>() / idx.len() as f64;
            let (mut equity, mut peak, mut dd) = (1.0f64, 1.0f64, 0.0f64);
            for &j in &idx {
                equity *= 1.0 + x[j];
                peak = peak.max(equity);
                dd = dd.min(equity / peak - 1.0);
            }
            want.push((idx[0], idx[idx.len() - 1] + 1, m, -dd));
        }
        assert!(want.len() > 1000, "seed {seed}: {} windows", want.len());
        assert_eq!(got.len(), want.len(), "seed {seed}: window count");
        for (k, (g, w)) in got.iter().zip(&want).enumerate() {
            assert_eq!((g.start, g.end), (w.0, w.1), "seed {seed} window {k}");
            close(g.mean, w.2, &format!("seed {seed} window {k} mean"));
            close(g.drawdown, w.3, &format!("seed {seed} window {k} drawdown"));
        }
    }
}

#[test]
fn freeze_s4_floor_matches_hand_value() {
    for (p, rank, value) in [(30u32, 3.0, -0.01 / 3.0 * 365.0), (10, 1.0, -0.02 / 3.0 * 365.0)] {
        let f = f_freeze("freeze-s4", &f_rule((3, 2), p, 0.025), &|_| {});
        let s4 = at(&f, &["s4_floor"]);
        assert_eq!((num(s4, &["window_days"]), num(s4, &["percentile"])), (3.0, p as f64));
        assert_eq!((num(s4, &["windows"]), num(s4, &["rank"])), (8.0, rank), "P{p}");
        close(num(s4, &["value"]), value, &format!("S4 P{p}"));
    }
    close(-0.01 / 3.0 * 365.0, -1.2166666666666667, "hand value P30");
    close(-0.02 / 3.0 * 365.0, -2.4333333333333333, "hand value P10");
}

#[test]
fn freeze_p1_floor_matches_hand_value() {
    let f = f_freeze("freeze-p1", &f_rule((3, 2), 30, 0.025), &|_| {});
    let p1 = at(&f, &["p1_floor"]);
    assert_eq!((num(p1, &["windows"]), num(p1, &["rank"])), (10.0, 1.0));
    close(num(p1, &["value"]), -3.65, "P1 floor");
}

#[test]
fn freeze_drawdown_cap_matches_hand_value() {
    for (floor, value) in [(0.025, 0.03), (0.20, 0.20)] {
        let f = f_freeze("freeze-d9", &f_rule((3, 2), 30, floor), &|_| {});
        let cap = at(&f, &["drawdown_cap"]);
        assert_eq!((num(cap, &["windows"]), num(cap, &["rank"])), (8.0, 8.0));
        close(num(cap, &["percentile_value"]), 0.03, "D9 percentile");
        assert_eq!(num(cap, &["floor"]), floor);
        close(num(cap, &["value"]), value, &format!("D9 cap at floor {floor}"));
    }
}

#[test]
fn freeze_reprices_paths_to_cost_bps() {
    let rule = f_rule((3, 2), 30, 0.025);
    let base = f_freeze("freeze-cost-base", &rule, &|_| {});
    let repriced = f_freeze("freeze-cost-5bp", &rule, &|p| {
        p.bps = 5.0;
        p.net = p.net.iter().map(|x| x + 0.001).collect();
        p.cost = Some(vec![0.001; 6]);
    });
    for (key, field) in [("s4_floor", "value"), ("p1_floor", "value"), ("drawdown_cap", "percentile_value"), ("drawdown_cap", "value")] {
        close(num(&repriced, &[key, field]), num(&base, &[key, field]), &format!("{key}.{field}"));
    }
    assert_eq!(num(&repriced, &["cost_bps"]), 10.0);
    // A path priced at another cost without period_cost cannot be repriced.
    let (_s, dir) = scratch("freeze-cost-missing");
    let rows = freeze_rows(&dir);
    let paths = f_paths(&dir, "f", &|p| p.bps = 5.0);
    let rule = write_text(&dir, "rule.json", &rule);
    let map = write_map(&dir, "map.json", &[(rows[0].as_str(), &paths[..], None)]);
    let o = freeze_run(&rows, &rule, &map, &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.period_cost") && stderr(&o).contains(&paths[0]), "{}", stderr(&o));
}

#[test]
fn freeze_refuses_bad_population() {
    let (_s, dir) = scratch("freeze-population");
    let rows = freeze_rows(&dir);
    let good = f_paths(&dir, "good", &|_| {});
    let rule = write_text(&dir, "rule.json", &f_rule((3, 2), 30, 0.025));
    let refused = |name: &str, rule: &str, paths: &[String], extra: &[&str], needles: &[&str]| {
        let map = write_map(&dir, &format!("{name}.json"), &[(rows[0].as_str(), paths, None)]);
        let o = freeze_run(&rows, rule, &map, extra);
        assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
        for n in needles {
            assert!(stderr(&o).contains(n), "{name}: '{n}' not in {}", stderr(&o));
        }
    };
    refused("missing", &rule, &good[..1], &[], &["cpcv.path 2"]);
    let twice = f_paths(&dir, "twice", &|p| p.index = 1);
    refused("twice", &rule, &[good[0].clone(), twice[1].clone()], &[], &["cpcv.path", &twice[1]]);
    let other = f_paths(&dir, "other", &|p| p.splits_sha = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc");
    refused("tables", &rule, &[good[0].clone(), other[1].clone()], &[], &["cpcv.splits_sha256", &other[1]]);
    let shas = f_paths(&dir, "shas", &|p| {
        if p.index == 2 {
            p.segments[0] = ("s2", SHA_S3);
        }
    });
    refused("report", &rule, &[good[0].clone(), shas[1].clone()], &[], &["report_sha256", "s2"]);
    let k1 = write_text(&dir, "rule_k1.json", &f_rule((3, 1), 30, 0.025));
    refused("test_groups", &k1, &good, &[], &["cpcv.test_groups", "freeze.cpcv"]);
    refused("locked", &rule, &good, &["--locked-test", "2020-01-05,2020-01-06"], &[&good[0], "locked"]);
    let map = write_map(&dir, "ok.json", &[(rows[0].as_str(), &good[..], None)]);
    let o = freeze_run(&rows, &rule, &map, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[test]
fn freeze_needs_map_and_rule() {
    let (_s, dir) = scratch("freeze-map");
    let rows = freeze_rows(&dir);
    let paths = f_paths(&dir, "f", &|_| {});
    let rule = write_text(&dir, "rule.json", &f_rule((3, 2), 30, 0.025));
    let other = write_map(&dir, "other.json", &[(rows[1].as_str(), &paths[..], None)]);
    let o = freeze_run(&rows, &rule, &other, &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("has no entry in --freeze-map") && stderr(&o).contains("row0"), "{}", stderr(&o));
    let mut args = vec!["trials", "--wf"];
    args.extend(rows.iter().map(String::as_str));
    let o = run(&[&args[..], &["--table", "--rule", &rule, "--draws", "20"][..]].concat());
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("the rule has freeze: give --freeze-map"), "{}", stderr(&o));
    let o = run(&[&args[..], &["--table", "--pick", "1", "--freeze-map", &other, "--draws", "20"][..]].concat());
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--freeze-map needs a rule with freeze"), "{}", stderr(&o));
}

#[test]
fn freeze_output_carries_rule_and_map_sha() {
    let (_s, dir) = scratch("freeze-sha");
    let rows = freeze_rows(&dir);
    let paths = f_paths(&dir, "f", &|_| {});
    let rule = write_text(&dir, "rule.json", &f_rule((3, 2), 30, 0.025));
    let run_json = write_text(&dir, "final.run.json", "{\"tool\":\"rcsr\",\"pareto\":[{\"expression\":\"x\",\"expression_exact\":\"x\"}]}\n");
    let map = write_map(&dir, "map.json", &[(rows[0].as_str(), &paths[..], Some(run_json.as_str()))]);
    let j = stdout_json(&freeze_run(&rows, &rule, &map, &[]));
    let sha = |p: &str| rcsr::manifest::sha256_hex(&std::fs::read(p).unwrap());
    let top = at(&j, &["freeze"]);
    assert_eq!(at(top, &["schema"]).as_str(), Some("rcsr-trials-freeze/1"));
    assert_eq!(at(top, &["rule_sha256"]).as_str(), at(&j, &["rule", "sha256"]).as_str());
    assert_eq!(at(top, &["rule_sha256"]).as_str(), Some(sha(&rule).as_str()));
    assert_eq!(at(top, &["map"]).as_str(), Some(map.as_str()));
    assert_eq!(at(top, &["map_sha256"]).as_str(), Some(sha(&map).as_str()));
    let f = at(&items(at(&j, &["picks"]))[0], &["freeze"]);
    assert_eq!(at(f, &["schema"]).as_str(), Some("rcsr-trials-freeze/1"));
    assert_eq!(at(f, &["configuration"]).as_str(), Some("F"));
    assert_eq!((num(f, &["cost_bps"]), num(f, &["periods_per_year"]), num(f, &["period_seconds"])), (10.0, 365.0, 86400.0));
    let pop = at(f, &["population"]);
    assert_eq!((num(pop, &["groups"]), num(pop, &["test_groups"]), num(pop, &["paths"])), (3.0, 2.0, 2.0));
    assert_eq!((at(pop, &["splits_sha256"]).as_str(), at(pop, &["scheme_sha256"]).as_str()), (Some(SPLITS), Some(SCHEME)));
    let files = items(at(pop, &["files"]));
    assert_eq!(files.len(), 2);
    for (k, file) in files.iter().enumerate() {
        assert_eq!(at(file, &["file"]).as_str(), Some(paths[k].as_str()));
        assert_eq!(at(file, &["sha256"]).as_str(), Some(sha(&paths[k]).as_str()));
        assert_eq!((num(file, &["path"]), num(file, &["periods"])), ((k + 1) as f64, 6.0));
    }
    assert_eq!(at(f, &["run"]).as_str(), Some(run_json.as_str()));
    assert_eq!(at(f, &["run_sha256"]).as_str(), Some(sha(&run_json).as_str()));
    // Without a run in the map the pick has no run hash.
    let bare = write_map(&dir, "bare.json", &[(rows[0].as_str(), &paths[..], None)]);
    let j = stdout_json(&freeze_run(&rows, &rule, &bare, &[]));
    assert!(matches!(at(&items(at(&j, &["picks"]))[0], &["freeze", "run_sha256"]), Json::Null));
}

#[test]
fn max_composites_caps_composite_picks() {
    let (_s, dir) = scratch("max-composites");
    let marker = |t: &str| format!("{{\"composite\":{{\"source\":\"rcsr compose\",\"type\":\"{t}\",\"form\":\"if(gate, f, cash)\",\"gate\":\"g\"}}}}");
    let rows: Vec<String> = [("A", None), ("B", Some("C1")), ("C", Some("C2")), ("D", None)]
        .iter()
        .enumerate()
        .map(|(k, (name, kind))| {
            let mut fx = Fx::new(centred_noise(400, 80 + k as u32, 0.01));
            fx.selection = kind.map(marker);
            write_fx(&dir, name, &fx)
        })
        .collect();
    let rule = write_text(&dir, "rule.json", "{\"schema\":\"rcsr-trials-rule/1\",\"pick\":3,\"diversity_rho\":1,\"max_composites\":1}");
    let mut args = vec!["trials", "--wf"];
    args.extend(rows.iter().map(String::as_str));
    args.extend(["--table", "--rule", &rule, "--draws", "20"]);
    let j = stdout_json(&run(&args));
    let picks: Vec<usize> = items(at(&j, &["picks"])).iter().map(|p| num(p, &["row"]) as usize).collect();
    assert_eq!(picks, vec![0, 1, 3], "{j:?}");
    let skipped = at(row(&j, 2), &["skipped"]);
    assert_eq!(at(skipped, &["reason"]).as_str(), Some("max_composites"));
    assert_eq!(num(skipped, &["limit"]), 1.0);
    assert!(matches!(at(row(&j, 0), &["composite"]), Json::Null));
    assert_eq!(at(row(&j, 1), &["composite", "type"]).as_str(), Some("C1"));
    assert_eq!(at(row(&j, 2), &["composite", "source"]).as_str(), Some("rcsr compose"));
    assert_eq!(num(&j, &["rule", "max_composites"]), 1.0);
    // Without the cap the walk picks the first three rows.
    let plain = write_text(&dir, "plain.json", "{\"schema\":\"rcsr-trials-rule/1\",\"pick\":3,\"diversity_rho\":1}");
    let mut args = vec!["trials", "--wf"];
    args.extend(rows.iter().map(String::as_str));
    args.extend(["--table", "--rule", &plain, "--draws", "20"]);
    let j = stdout_json(&run(&args));
    let picks: Vec<usize> = items(at(&j, &["picks"])).iter().map(|p| num(p, &["row"]) as usize).collect();
    assert_eq!(picks, vec![0, 1, 2]);
}

/// A flag-built rule without `freeze` and `max_composites`, pinned to the rule hash of the
/// integration base before #1237.
#[test]
fn rule_sha_unchanged_without_new_keys() {
    let (_s, dir) = scratch("rule-sha");
    let (paths, _) = five_series(&dir);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(String::as_str));
    args.extend([
        "--table", "--pareto", "net10:max,sharpe:max", "--rank", "net10,sortino", "--pick", "3", "--diversity-rho", "0.7",
        "--min-front", "2", "--cost-multiples", "1,3", "--draws", "20",
    ]);
    let j = stdout_json(&run(&args));
    let rule = at(&j, &["rule"]);
    assert!(rule.get("freeze").is_none() && rule.get("max_composites").is_none(), "{rule:?}");
    assert_eq!(at(rule, &["sha256"]).as_str(), Some(PINNED_FLAG_RULE_SHA), "{rule:?}");
}

const PINNED_FLAG_RULE_SHA: &str = "db5dabcc350c18d4c719a404617b67a92a75f53431c5e0123bd013e6d51b4b90";

#[test]
fn freeze_rule_refuses_bad_keys() {
    let (_s, dir) = scratch("freeze-rule-keys");
    let rows = freeze_rows(&dir);
    let paths = f_paths(&dir, "f", &|_| {});
    let map = write_map(&dir, "map.json", &[(rows[0].as_str(), &paths[..], None)]);
    let good = f_rule((3, 2), 30, 0.025);
    for (k, (from, to, key)) in [
        ("\"percentile\":30", "\"percentile\":0", "freeze.s4.percentile"),
        ("\"percentile\":30", "\"percentile\":100", "freeze.s4.percentile"),
        ("\"p1\":{\"window_days\":2", "\"p1\":{\"window_days\":0", "freeze.p1.window_days"),
        ("\"floor\":0.025", "\"floor\":1", "freeze.drawdown_cap.floor"),
        ("\"floor\":0.025", "\"floor\":-0.1", "freeze.drawdown_cap.floor"),
        ("\"cpcv\":[3,2]", "\"cpcv\":[3,3]", "freeze.cpcv"),
        ("\"cost_bps\":10", "\"cost_bps\":10,\"extra\":1", "freeze.extra"),
    ]
    .iter()
    .enumerate()
    {
        assert!(good.contains(from), "{from}");
        let rule = write_text(&dir, &format!("bad{k}.json"), &good.replacen(from, to, 1));
        let o = freeze_run(&rows, &rule, &map, &[]);
        assert_eq!(code(&o), 2, "{key}: {}", stderr(&o));
        assert!(stderr(&o).contains(key), "{key}: {}", stderr(&o));
    }
}

// ---------------------------------------------------------------------------
// Sibling horizon tables (#1269)
// ---------------------------------------------------------------------------

/// `rows` walk-forward reports `<tag><k>.json` over `n` hourly periods with exact means
/// `lift + 0.0002 k`; with `twin` the second series is 0.98 x the first's noise plus 0.2 x
/// its own, so the two share one cluster at `--cluster-rho 0.9`.
fn family_reports(dir: &Path, tag: &str, rows: usize, n: usize, seed: u32, lift: f64, twin: bool) -> Vec<String> {
    let first = centred_noise(n, seed, 0.01);
    (0..rows)
        .map(|k| {
            let noise: Vec<f64> = match k {
                0 => first.clone(),
                1 if twin => first.iter().zip(centred_noise(n, seed + 50, 0.01)).map(|(a, b)| 0.98 * a + 0.2 * b).collect(),
                _ => centred_noise(n, seed + k as u32, 0.01),
            };
            let net: Vec<f64> = noise.iter().map(|x| x + lift + 0.0002 * k as f64).collect();
            write_fx(dir, &format!("{tag}{k}"), &Fx::new(net))
        })
        .collect()
}

/// `rcsr trials --table` over `reports` that picks every row, written to `out`.
fn family_run(reports: &[String], out: &str, extra: &[&str]) -> Output {
    let picks = reports.len().to_string();
    let mut args = vec!["trials", "--wf"];
    args.extend(reports.iter().map(String::as_str));
    args.extend(["--table", "--pick", &picks, "--diversity-rho", "1", "--draws", "200", "--out", out]);
    args.extend_from_slice(extra);
    run(&args)
}

/// Fixture A, B, C: three tables of 6, 5 and 4 rows (B and C written to `b.json` and
/// `c.json`); A's second row is the twin of its first, so N_eff is 5, 5 and 4.
fn abc(dir: &Path) -> (Vec<String>, String, String) {
    let a = family_reports(dir, "a", 6, 400, 100, 0.0002, true);
    let b = family_reports(dir, "b", 5, 400, 200, 0.0001, false);
    let c = family_reports(dir, "c", 4, 300, 300, 0.0001, false);
    let (tb, tc) = (dir.join("b.json").to_string_lossy().into_owned(), dir.join("c.json").to_string_lossy().into_owned());
    for (reports, out) in [(&b, &tb), (&c, &tc)] {
        let o = family_run(reports, out, &[]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
    (a, tb, tc)
}

/// `E[max_N]` of Bailey and López de Prado (2014) with cross-trial Sharpe variance `var`.
fn hand_expected_max(n: f64, var: f64) -> f64 {
    let q = rcsr::inference::normal_quantile;
    let g = 0.577_215_664_901_532_9;
    var.sqrt() * ((1.0 - g) * q(1.0 - 1.0 / n) + g * q(1.0 - 1.0 / (n * std::f64::consts::E))).max(0.0)
}

/// The DSR of a pick's `dsr` object at `n` trials, from its Sharpe, T, skew, kurtosis and `var_sr`.
fn hand_dsr(d: &Json, n: f64) -> f64 {
    let (sr, t, skew, kurt, var) = (num(d, &["sharpe_per_period"]), num(d, &["t"]), num(d, &["skew"]), num(d, &["kurtosis"]), num(d, &["var_sr"]));
    let denom = 1.0 - skew * sr + (kurt - 1.0) / 4.0 * sr * sr;
    rcsr::inference::normal_cdf((sr - hand_expected_max(n, var)) * (t - 1.0).sqrt() / denom.sqrt())
}

fn file_sha(path: &str) -> String {
    rcsr::manifest::sha256_hex(&std::fs::read(path).unwrap())
}

#[test]
fn family_tables_add_their_n_eff_to_the_dsr() {
    let (_s, dir) = scratch("family-neff");
    let (a, tb, tc) = abc(&dir);
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let (ta, tf) = (p("a.json"), p("a_family.json"));
    for (out, extra) in [(&ta, vec![]), (&tf, vec!["--family-table", tb.as_str(), tc.as_str()])] {
        let o = family_run(&a, out, &extra);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
    let (ja, jb, jc, jf) = (read_json(&ta), read_json(&tb), read_json(&tc), read_json(&tf));
    let n_eff = |j: &Json| num(j, &["trials", "n_eff"]);
    assert_eq!((n_eff(&ja), n_eff(&jb), n_eff(&jc)), (5.0, 5.0, 4.0), "planted clusters");
    assert_eq!(n_eff(&jf), n_eff(&ja), "trials.n_eff stays the table's own");
    let total = n_eff(&ja) + n_eff(&jb) + n_eff(&jc);
    let family = at(&jf, &["trials", "family"]);
    assert_eq!(num(family, &["n_eff_total"]), total);
    assert_eq!(num(family, &["nominal_total"]), 6.0 + 5.0 + 4.0);
    let tables = items(at(family, &["tables"]));
    assert_eq!(tables.len(), 2);
    for (t, (path, j)) in tables.iter().zip([(&tb, &jb), (&tc, &jc)]) {
        assert_eq!(at(t, &["path"]).as_str(), Some(path.as_str()));
        assert_eq!(at(t, &["sha256"]).as_str(), Some(file_sha(path).as_str()));
        assert_eq!(num(t, &["n_eff"]), n_eff(j));
        assert_eq!(num(t, &["nominal"]), num(j, &["trials", "nominal"]));
        assert_eq!(num(t, &["spa_p"]), num(j, &["spa", "p_consistent"]));
    }
    let mut shas = vec![file_sha(&tb), file_sha(&tc)];
    shas.sort();
    let joined: String = shas.iter().map(|s| format!("{s}\n")).collect();
    assert_eq!(at(family, &["sha256"]).as_str(), Some(rcsr::manifest::sha256_hex(joined.as_bytes()).as_str()));
    let (picks_a, picks_f) = (items(at(&ja, &["picks"])), items(at(&jf, &["picks"])));
    assert_eq!((picks_a.len(), picks_f.len()), (6, 6), "every row is picked");
    for (k, (pa, pf)) in picks_a.iter().zip(picks_f).enumerate() {
        let (da, df) = (at(pa, &["dsr"]), at(pf, &["dsr"]));
        assert_eq!(num(da, &["n_eff"]), 5.0, "pick {k}");
        close(num(da, &["value"]), hand_dsr(da, 5.0), &format!("pick {k} own DSR"));
        assert_eq!(num(df, &["n_eff"]), total, "pick {k}");
        for key in ["sharpe_per_period", "t", "skew", "kurtosis", "var_sr"] {
            assert_eq!(num(df, &[key]).to_bits(), num(da, &[key]).to_bits(), "pick {k} {key}: the family adds trials, not variance");
        }
        close(num(df, &["value"]), hand_dsr(df, total), &format!("pick {k} family DSR"));
        assert!(num(df, &["value"]) < num(da, &["value"]), "pick {k}: {} vs {}", num(df, &["value"]), num(da, &["value"]));
        assert_eq!(num(pf, &["dsr_10x", "n_eff"]), 10.0 * total);
        close(num(pf, &["dsr_10x", "value"]), hand_dsr(at(pf, &["dsr_10x"]), 10.0 * total), &format!("pick {k} DSR 10 N_eff"));
        assert_eq!(num(pf, &["dsr_nominal", "n_eff"]), 15.0);
        close(num(pf, &["dsr_nominal", "value"]), hand_dsr(at(pf, &["dsr_nominal"]), 15.0), &format!("pick {k} nominal DSR"));
        let sharpe = num(pf, &["sharpe_annual"]);
        close(num(pf, &["min_btl_years"]), (hand_expected_max(total, 1.0) / sharpe).powi(2), &format!("pick {k} MinBTL"));
        assert_eq!(num(pf, &["psr"]), num(pa, &["psr"]), "pick {k}");
    }
}

#[test]
fn family_spa_p_is_bonferroni_over_the_tables() {
    let (_s, dir) = scratch("family-spa");
    let (a, tb, tc) = abc(&dir);
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let (ta, tf, md) = (p("a.json"), p("a_family.json"), p("a_family.md"));
    let o = family_run(&a, &ta, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = family_run(&a, &tf, &["--family-table", &tb, &tc, "--markdown", &md]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let (ja, jf) = (read_json(&ta), read_json(&tf));
    assert!(at(&ja, &["spa"]).get("p_family").is_none() && at(&ja, &["trials"]).get("family").is_none(), "no family block without the flag");
    let p_own = num(&jf, &["spa", "p_consistent"]);
    assert_eq!(p_own, num(&ja, &["spa", "p_consistent"]), "the family leaves the SPA of the table");
    assert!(p_own > 0.0 && 3.0 * p_own < 1.0, "fixture A tests the uncapped branch: p {p_own}");
    assert_eq!(num(&jf, &["spa", "p_family"]).to_bits(), (3.0 * p_own).min(1.0).to_bits());
    let text = std::fs::read_to_string(&md).unwrap();
    assert!(text.contains("SPA family p") && text.contains("cross-trial Sharpe variance"), "{text}");
    // A table of losing rows: 3 p is above 1 and the family p is 1.
    let losers = family_reports(&dir, "n", 4, 400, 400, -0.001, false);
    let tn = p("n_family.json");
    let o = family_run(&losers, &tn, &["--family-table", &tb, &tc]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let jn = read_json(&tn);
    let p_losers = num(&jn, &["spa", "p_consistent"]);
    assert!(3.0 * p_losers > 1.0, "losing rows: p {p_losers}");
    assert_eq!(num(&jn, &["spa", "p_family"]), 1.0);
}

/// A minimal sibling table: the fields the family reads and nothing else, with one report
/// named after the table that resolves nowhere.
fn sibling(dir: &Path, name: &str, n_eff: u32, nominal: u32, spa_p: f64) -> String {
    write_text(
        dir,
        name,
        &format!(
            "{{\"schema\":\"rcsr-trials-table/1\",\"inputs\":{{\"reports\":[\"missing-{name}\"]}},\"trials\":{{\"nominal\":{nominal},\"n_eff\":{n_eff}}},\
\"spa\":{{\"benchmark\":\"cash\",\"p_consistent\":{spa_p}}}}}\n"
        ),
    )
}

fn refused_family(reports: &[String], out: &str, extra: &[&str], needles: &[&str]) {
    let o = family_run(reports, out, extra);
    assert_eq!(code(&o), 2, "{extra:?}: {}", stderr(&o));
    for n in needles {
        assert!(stderr(&o).contains(n), "{extra:?}: '{n}' not in {}", stderr(&o));
    }
    assert!(!Path::new(out).exists() || extra.contains(&out), "{extra:?}: a refused table writes nothing");
}

#[test]
fn family_table_refuses_two_tables_with_one_sha256() {
    let (_s, dir) = scratch("family-twice");
    let a = family_reports(&dir, "a", 3, 200, 500, 0.0002, false);
    let b = sibling(&dir, "b.json", 5, 5, 0.2);
    let copy = write_text(&dir, "b_copy.json", &std::fs::read_to_string(&b).unwrap());
    let out = dir.join("out.json").to_string_lossy().into_owned();
    refused_family(&a, &out, &["--family-table", &b, &copy], &[&copy, &b, "sha256"]);
    refused_family(&a, &out, &["--family-table", &b, "--family-table", &b], &[&b, "sha256"]);
    let c = sibling(&dir, "c.json", 4, 4, 0.3);
    let o = family_run(&a, &out, &["--family-table", &b, "--family-table", &c]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(items(at(&read_json(&out), &["trials", "family", "tables"])).len(), 2, "the flag repeats");
}

#[test]
fn family_table_refuses_the_out_file() {
    let (_s, dir) = scratch("family-out");
    let a = family_reports(&dir, "a", 3, 200, 510, 0.0002, false);
    let b = sibling(&dir, "b.json", 5, 5, 0.2);
    let before = std::fs::read(&b).unwrap();
    let o = family_run(&a, &b, &["--family-table", &b]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains(&b) && stderr(&o).contains("--out"), "{}", stderr(&o));
    let respelled = dir.join(".").join("b.json").to_string_lossy().into_owned();
    let o = family_run(&a, &respelled, &["--family-table", &b]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains(&b) && stderr(&o).contains("--out"), "{}", stderr(&o));
    assert_eq!(std::fs::read(&b).unwrap(), before, "the family table is not overwritten");
}

#[test]
fn family_table_refuses_missing_fields() {
    let (_s, dir) = scratch("family-fields");
    let a = family_reports(&dir, "a", 3, 200, 520, 0.0002, false);
    let full = std::fs::read_to_string(sibling(&dir, "full.json", 5, 5, 0.2)).unwrap();
    let out = dir.join("out.json").to_string_lossy().into_owned();
    for (k, path) in [&["trials", "n_eff"][..], &["trials", "nominal"], &["spa", "p_consistent"], &["inputs", "reports"]].iter().enumerate() {
        let text = rcsr::verdict::remove_member(&full, path).unwrap();
        assert_ne!(text, full, "{path:?}");
        let bad = write_text(&dir, &format!("bad{k}.json"), &text);
        refused_family(&a, &out, &["--family-table", &bad], &[&bad, &path.join(".")]);
    }
    let skipped = write_text(&dir, "skipped.json", &full.replace("{\"benchmark\":\"cash\",\"p_consistent\":0.2}", "{\"skipped\":\"no varying series\"}"));
    refused_family(&a, &out, &["--family-table", &skipped], &[&skipped, "spa.p_consistent"]);
    let report = a[0].clone();
    refused_family(&a, &out, &["--family-table", &report], &[&report, "rcsr-trials-table/1"]);
}

#[test]
fn family_table_refuses_locked_data() {
    let (_s, dir) = scratch("family-locked");
    let a = family_reports(&dir, "a", 3, 200, 530, 0.0002, false);
    let full = std::fs::read_to_string(sibling(&dir, "full.json", 5, 5, 0.2)).unwrap();
    let out = dir.join("out.json").to_string_lossy().into_owned();
    let marked = write_text(&dir, "marked.json", &full.replacen("{\"schema\"", "{\"protocol\":{\"test_read\":true},\"schema\"", 1));
    refused_family(&a, &out, &["--family-table", &marked], &[&marked, "locked"]);
    let mut read = Fx::new(centred_noise(200, 531, 0.01));
    read.protocol = Some("{\"lockbox\":{\"rows\":0,\"evaluated\":false},\"test_read\":true}".to_string());
    let read = write_fx(&dir, "sealed_read", &read);
    refused_family(&a, &out, &["--family-table", &read], &[&read, "locked"]);
}

#[test]
fn family_table_refuses_shared_reports() {
    let (_s, dir) = scratch("family-shared");
    let p = |name: &str| dir.join(name).to_string_lossy().into_owned();
    let a = family_reports(&dir, "a", 3, 200, 540, 0.0002, false);
    let b = family_reports(&dir, "b", 4, 200, 550, 0.0001, false);
    let out = p("out.json");
    // An earlier table of the same rows under another file name, its reports spelled through `.`.
    let respelled: Vec<String> = a.iter().map(|r| dir.join(".").join(Path::new(r).file_name().unwrap()).to_string_lossy().into_owned()).collect();
    let earlier = p("earlier.json");
    let o = family_run(&respelled, &earlier, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    refused_family(&a, &out, &["--family-table", &earlier], &[&earlier, "this table", &respelled[0]]);
    // Two siblings that share the report b1.
    let (tb1, tb2, tb3) = (p("tb1.json"), p("tb2.json"), p("tb3.json"));
    for (reports, table) in [(&b[..2], &tb1), (&b[1..3], &tb2), (&b[2..], &tb3)] {
        let o = family_run(reports, table, &[]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
    refused_family(&a, &out, &["--family-table", &tb1, &tb2], &[&tb2, &tb1, &b[1]]);
    let o = family_run(&a, &out, &["--family-table", &tb1, &tb3]);
    assert_eq!(code(&o), 0, "disjoint reports: {}", stderr(&o));
}

/// The JSON and Markdown of `rcsr trials --table` without `--family-table` on two fixtures,
/// with relative paths, pinned to the SHA-256 of the integration base before #1269.
#[test]
fn table_without_family_is_unchanged() {
    let (_s, dir) = scratch("family-golden");
    let in_dir = |args: &[&str]| rcsr_cmd::rcsr().current_dir(&dir).args(args).output().expect("run rcsr");
    let (d, _) = five_series(&dir);
    let (s, _) = six_series(&dir);
    let names = |paths: &[String]| paths.iter().map(|p| Path::new(p).file_name().unwrap().to_string_lossy().into_owned()).collect::<Vec<_>>();
    let (d, s) = (names(&d), names(&s));
    write_text(&dir, "cal.csv", &format!("label,start,end\nbull,{},{}\nbear,{},{}\n", T0, T0 + 3600 * 6, T0 + 3600 * 6, T0 + 3600 * 12));
    write_text(
        &dir,
        "rule.json",
        "{\"schema\":\"rcsr-trials-rule/1\",\"gates\":{\"net10\":[\">\",-100]},\"rank\":[\"net20\",\"sharpe\",\"bear_excess\"],\"pick\":2,\"diversity_rho\":0.95}",
    );
    let fixtures: [(&[String], Vec<&str>, &str); 2] = [
        (&d, vec!["--pareto", "net10:max,sharpe:max", "--rank", "net10,sortino", "--pick", "3", "--diversity-rho", "0.7", "--draws", "100"], "t1"),
        (&s, vec!["--rule", "rule.json", "--regimes", "cal.csv", "--prior-trials", "3", "--pbo-blocks", "4", "--block-len", "2", "--draws", "50"], "t2"),
    ];
    let mut got = Vec::new();
    for (paths, extra, tag) in fixtures {
        let (out, md) = (format!("{tag}.json"), format!("{tag}.md"));
        let mut args = vec!["trials", "--wf"];
        args.extend(paths.iter().map(String::as_str));
        args.push("--table");
        args.extend(extra);
        args.extend(["--out", &out, "--markdown", &md]);
        let o = in_dir(&args);
        assert_eq!(code(&o), 0, "{tag}: {}", stderr(&o));
        let mut bytes = std::fs::read(dir.join(&out)).unwrap();
        bytes.extend(std::fs::read(dir.join(&md)).unwrap());
        got.push(rcsr::manifest::sha256_hex(&bytes));
    }
    assert_eq!(got, [PINNED_TABLE_T1, PINNED_TABLE_T2]);
}

const PINNED_TABLE_T1: &str = "48f3193890355a469c4b4ac30987aa0ef5a96496fbac2a882a27492ad491e8ae";
const PINNED_TABLE_T2: &str = "7e24c7106fdbe3ce17387bacf90f674721ca8e813ab5c43332bc69840b764f68";

// ---------------------------------------------------------------------------
// Baselines from walk-forward reports (#1271)
// ---------------------------------------------------------------------------

const BN: usize = 400;

/// Three candidate reports and two baseline reports on one grid and one cost; the first baseline is the best rung.
struct BaselineFx {
    candidates: Vec<String>,
    nets: Vec<Vec<f64>>,
    baselines: Vec<String>,
    base_nets: Vec<Vec<f64>>,
    bh: Vec<f64>,
}

fn baseline_fx(dir: &Path, edit: &dyn Fn(&mut Fx)) -> BaselineFx {
    let nets: Vec<Vec<f64>> = [0.0010, 0.0007, 0.0004]
        .iter()
        .enumerate()
        .map(|(k, m)| centred_noise(BN, 40 + k as u32, 0.01).iter().map(|x| x + m).collect())
        .collect();
    let candidates = nets.iter().enumerate().map(|(k, v)| write_fx(dir, &format!("c{k}"), &Fx::new(v.clone()))).collect();
    let bh: Vec<f64> = centred_noise(BN, 50, 0.01).iter().map(|x| x + 0.0002).collect();
    let base_nets: Vec<Vec<f64>> = [0.0006, -0.0003]
        .iter()
        .enumerate()
        .map(|(k, m)| centred_noise(BN, 60 + k as u32, 0.01).iter().map(|x| x + m).collect())
        .collect();
    let mut baselines = Vec::new();
    for (k, v) in base_nets.iter().enumerate() {
        let sub = dir.join(format!("base{k}"));
        std::fs::create_dir_all(&sub).unwrap();
        let mut fx = Fx::new(v.clone());
        fx.bh_net = Some(bh.clone());
        edit(&mut fx);
        baselines.push(write_fx(&sub, "path_1", &fx));
    }
    BaselineFx { candidates, nets, baselines, base_nets, bh }
}

fn baseline_run(fx: &BaselineFx, baselines: &[String], extra: &[&str]) -> Output {
    let mut args = vec!["trials", "--wf"];
    args.extend(fx.candidates.iter().map(String::as_str));
    args.extend(["--table", "--draws", "200", "--block-len", "4", "--pbo-blocks", "4", "--cluster-rho", "0.7", "--gates", "net10>-100", "--baseline-wf"]);
    args.extend(baselines.iter().map(String::as_str));
    args.extend(extra);
    run(&args)
}

#[test]
fn baseline_wf_takes_the_best_rung() {
    let (_s, dir) = scratch("baseline-wf-best");
    let fx = baseline_fx(&dir, &|_| {});
    let j = stdout_json(&baseline_run(&fx, &fx.baselines, &["--rank", "baseline_excess"]));
    let rungs = [vec![0.0; BN], fx.bh.clone(), fx.base_nets[0].clone(), fx.base_nets[1].clone()];
    let best = (0..rungs.len()).max_by(|&a, &b| mean(&rungs[a]).total_cmp(&mean(&rungs[b]))).unwrap();
    assert_eq!(best, 2, "fixture: the first baseline report is the best rung");
    assert_eq!(at(&j, &["spa", "benchmark"]).as_str(), Some("base0/path_1"));
    assert_eq!(at(&j, &["stepm", "benchmark"]).as_str(), Some("base0/path_1"));
    assert_eq!(at(&j, &["inputs", "baseline", "source"]).as_str(), Some("wf"));
    assert_eq!(at(&j, &["inputs", "baseline", "best"]).as_str(), Some("base0/path_1"));
    let names: Vec<&str> = items(at(&j, &["inputs", "baseline", "rungs"])).iter().map(|r| at(r, &["name"]).as_str().unwrap()).collect();
    assert_eq!(names, ["cash", "buy_and_hold", "base0/path_1", "base1/path_1"]);
    let sha = at(&j, &["inputs", "baseline", "rungs"]);
    for (r, p) in items(sha)[2..].iter().zip(&fx.baselines) {
        assert_eq!(at(r, &["sha256"]).as_str().unwrap(), rcsr::manifest::sha256_hex(&std::fs::read(p).unwrap()));
    }
    for (i, net) in fx.nets.iter().enumerate() {
        let want = (mean(net) - mean(&rungs[best])) * 8760.0;
        let got = num(at(row(&j, i), &["metrics"]), &["baseline_excess"]);
        assert!((got - want).abs() <= 1e-12, "row {i}: table {got} hand {want}");
    }
    let diffs: Vec<Vec<f64>> = fx.nets.iter().map(|c| c.iter().zip(&rungs[best]).map(|(a, b)| a - b).collect()).collect();
    let spa = rcsr::inference::spa_test(&diffs, 4, 200, 20_260_101).unwrap();
    assert_eq!(num(&j, &["spa", "p_consistent"]), spa.p_consistent);
}

#[test]
fn baseline_wf_buy_and_hold_and_cash_can_win() {
    let (_s, dir) = scratch("baseline-wf-builtin");
    let fx = baseline_fx(&dir, &|f| {
        f.net = f.net.iter().map(|x| x - 0.01).collect();
        f.bh_net = f.bh_net.as_ref().map(|b| b.iter().map(|x| x + 0.005).collect());
    });
    let j = stdout_json(&baseline_run(&fx, &fx.baselines, &[]));
    assert_eq!(at(&j, &["spa", "benchmark"]).as_str(), Some("buy_and_hold"));
}

#[test]
fn baseline_wf_refuses_another_grid() {
    let (_s, dir) = scratch("baseline-wf-grid");
    let fx = baseline_fx(&dir, &|_| {});
    let sub = dir.join("shifted");
    std::fs::create_dir_all(&sub).unwrap();
    let mut other = Fx::new(fx.base_nets[0].clone());
    other.bh_net = Some(fx.bh.clone());
    other.time = other.time.iter().map(|t| t + 60.0).collect();
    let bad = write_fx(&sub, "path_1", &other);
    let o = baseline_run(&fx, &[bad.clone()], &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("pooled.period_time") && e.contains(&bad), "{e}");
}

#[test]
fn baseline_wf_refuses_another_cost() {
    let (_s, dir) = scratch("baseline-wf-cost");
    let fx = baseline_fx(&dir, &|_| {});
    let text = std::fs::read_to_string(&fx.baselines[0]).unwrap().replace("\"cost_model\":{\"per_side_bps\":10,", "\"cost_model\":{\"per_side_bps\":15,");
    let bad = write_text(&dir, "dear.json", &text);
    let o = baseline_run(&fx, &[bad.clone()], &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("cost_model.per_side_bps") && e.contains(&bad), "{e}");
}

#[test]
fn baseline_wf_accepts_another_minimum_hold() {
    let (_s, dir) = scratch("baseline-wf-hold");
    let fx = baseline_fx(&dir, &|_| {});
    let text = std::fs::read_to_string(&fx.baselines[0]).unwrap().replace("\"min_hold\":1", "\"min_hold\":6");
    assert_ne!(text, std::fs::read_to_string(&fx.baselines[0]).unwrap());
    let held = write_text(&dir, "held.json", &text);
    let j = stdout_json(&baseline_run(&fx, &[held], &[]));
    assert_eq!(at(&j, &["spa", "benchmark"]).as_str(), Some("held"));
}

#[test]
fn baseline_wf_refuses_ladder_baseline_locked_and_name_clashes() {
    let (_s, dir) = scratch("baseline-wf-refusals");
    let fx = baseline_fx(&dir, &|_| {});
    let o = baseline_run(&fx, &fx.baselines, &["--baseline", "ladder.json"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--baseline-wf excludes --baseline"), "{}", stderr(&o));
    let text = std::fs::read_to_string(&fx.baselines[0]).unwrap().replace("\"mode\":\"walkforward\",", "\"mode\":\"walkforward\",\"protocol\":{\"test_read\":true},");
    let locked = write_text(&dir, "locked.json", &text);
    let o = baseline_run(&fx, &[locked.clone()], &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains(&locked) && stderr(&o).contains("locked"), "{}", stderr(&o));
    let a = write_text(&dir, "cash.json", &std::fs::read_to_string(&fx.baselines[0]).unwrap());
    let o = baseline_run(&fx, &[a], &[]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("'cash' is taken"), "{}", stderr(&o));
}

#[test]
fn baseline_wf_needs_the_table() {
    let (_s, dir) = scratch("baseline-wf-notable");
    let fx = baseline_fx(&dir, &|_| {});
    let o = run(&["trials", "--wf", &fx.candidates[0], &fx.candidates[1], "--baseline-wf", &fx.baselines[0]]);
    assert_eq!(code(&o), 2);
    assert!(stderr(&o).contains("--baseline-wf needs --table"), "{}", stderr(&o));
}

/// The table output without `--baseline-wf`, pinned to the SHA-256 of the integration base before #1271.
#[test]
fn table_output_without_baseline_wf_is_unchanged() {
    let (_s, dir) = scratch("table-baseline-golden");
    let fx = baseline_fx(&dir, &|_| {});
    let ladder = write_text(
        &dir,
        "ladder.json",
        &format!(
            "{{\"schema\":\"rcsr-ladder/1\",\"series\":{{\"period_time\":[{}],\"period_net\":{{\"cash\":[{}],\"buy_and_hold\":[{}]}}}}}}",
            arr(&grid(BN)),
            arr(&vec![0.0; BN]),
            arr(&fx.bh)
        ),
    );
    let mut bytes = Vec::new();
    for baseline in [None, Some(ladder.as_str())] {
        let mut args = vec!["trials", "--wf"];
        args.extend(fx.candidates.iter().map(String::as_str));
        args.extend(["--table", "--draws", "200", "--block-len", "4", "--pbo-blocks", "4", "--cluster-rho", "0.7", "--rank", "net10,sharpe"]);
        if let Some(l) = baseline {
            args.extend(["--baseline", l]);
        }
        let o = run(&args);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
        bytes.extend_from_slice(&o.stdout);
    }
    assert_eq!(rcsr::manifest::sha256_hex(&bytes), "PIN");
}
