//! #926: `rcsr trials` campaign accounting and paired comparison.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use rcsr::inference;
use rcsr::rng::Mt;
use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const PPY: f64 = 8760.0;
const N: usize = 8760;
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
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} is not a number"))
}

fn gauss(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn noise(n: usize, seed: u32, sd: f64) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    (0..n).map(|_| sd * gauss(&mut rng)).collect()
}

fn grid(n: usize) -> Vec<f64> {
    (0..n).map(|i| (T0 + STEP * i as i64) as f64).collect()
}

struct Spec {
    cost_bps: f64,
    ppy: f64,
    window: &'static str,
    time: Vec<f64>,
    net_text: Option<String>,
    quantile: Option<f64>,
    fixed: Option<f64>,
    policy: &'static str,
    fit: Option<(u32, u32, f64)>,
    /// #1203: `book.missing_score`; `None` writes no `book` object.
    book: Option<&'static str>,
    /// #1208: `annualization.period_seconds`; `None` writes no member (a report of an older build).
    period_seconds: Option<f64>,
    /// #1328: `pooled.grid.rule`; `None` writes no `grid` object (the score grid).
    grid_rule: Option<&'static str>,
}

impl Spec {
    fn new(n: usize) -> Spec {
        Spec {
            cost_bps: 10.0,
            ppy: PPY,
            window: "w1",
            time: grid(n),
            net_text: None,
            quantile: Some(0.5),
            fixed: None,
            policy: "threshold",
            fit: None,
            book: None,
            period_seconds: None,
            grid_rule: None,
        }
    }
}

fn arr(v: &[f64]) -> String {
    v.iter().map(|x| format!("{x}")).collect::<Vec<_>>().join(",")
}

fn write_report(dir: &Path, name: &str, net: &[f64], s: &Spec) -> String {
    let net_text = s.net_text.clone().unwrap_or_else(|| arr(net));
    let text = format!(
        "{{\"mode\":\"walkforward\",\"window_definition\":[{{\"label\":\"{w}\",\"test_start\":1}}],\
\"purge\":{{\"horizon_seconds\":3600,\"embargo_seconds\":3600}},\
\"cost_model\":{{\"per_side_bps\":{c},\"round_trip_bps\":{rt}}},\
\"cost\":{{\"per_side_bps\":{c},\"source\":\"flag\",\"return_units\":\"log\"}},\
\"policy\":{{\"name\":\"{policy}\",\"band_width\":0,\"min_hold\":1}},\
\"threshold\":{{\"quantile\":{quantile},\"fixed\":{fixed}{fit}}},\
\"annualization\":{{\"periods_per_year\":{ppy},\"method\":\"explicit\"{period_seconds}}},\
{book}\
\"pooled\":{{\"periods\":{n},\"entries\":10,\"active_rows\":100,\"period_net\":[{net_text}],\"period_time\":[{time}]{grid}}}}}",
        w = s.window,
        policy = s.policy,
        quantile = s.quantile.map_or("null".to_string(), |q| q.to_string()),
        fixed = s.fixed.map_or("null".to_string(), |q| q.to_string()),
        book = s.book.map_or(String::new(), |r| format!("\"book\":{{\"missing_score\":\"{r}\"}},")),
        fit = s.fit.map_or(String::new(), |(e, b, l)| format!(
            ",\"fit\":\"fitness\",\"fit_method\":\"text\",\"fit_sources\":[\"fixed\"],\"cut_min_entries\":{e},\"cut_blocks\":{b},\"cut_lambda\":{l}"
        )),
        c = s.cost_bps,
        rt = s.cost_bps * 2.0,
        ppy = s.ppy,
        period_seconds = s.period_seconds.map_or(String::new(), |p| format!(",\"period_seconds\":{p}")),
        n = net.len(),
        time = arr(&s.time),
        grid = s.grid_rule.map_or(String::new(), |r| format!(",\"grid\":{{\"rule\":\"{r}\"}}")),
    );
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_ladder(dir: &Path, name: &str, time: &[f64], rungs: &[(&str, Vec<f64>)]) -> String {
    let nets: Vec<String> = rungs.iter().map(|(n, v)| format!("\"{n}\":[{}]", arr(v))).collect();
    let text = format!(
        "{{\"schema\":\"rcsr-ladder/1\",\"series\":{{\"period_time\":[{}],\"period_net\":{{{}}}}}}}",
        arr(time),
        nets.join(",")
    );
    let path = dir.join(format!("{name}.json"));
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn trials_dir() -> (Scratch, std::path::PathBuf) {
    let s = Scratch::new("trials");
    let p = s.path().to_path_buf();
    (s, p)
}

// ---------------------------------------------------------------------------
// Campaign
// ---------------------------------------------------------------------------

#[test]
fn trials_cluster_counts() {
    let (_s, dir) = trials_dir();
    let spec = Spec::new(N);
    let same = noise(N, 1, 0.01);
    let paths: Vec<String> = (0..20).map(|i| write_report(&dir, &format!("same{i}"), &same, &spec)).collect();
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(|p| p.as_str()));
    args.extend(["--draws", "50", "--block-len", "8"]);
    let j = stdout_json(&run(&args));
    assert_eq!(num(&j, &["series"]), 20.0);
    assert_eq!(num(&j, &["distinct_books"]), 1.0);
    assert_eq!(num(&j, &["clusters"]), 1.0);

    let paths: Vec<String> = (0..20)
        .map(|i| write_report(&dir, &format!("ind{i}"), &noise(N, 100 + i as u32, 0.01), &spec))
        .collect();
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(|p| p.as_str()));
    args.extend(["--draws", "50", "--block-len", "8", "--cluster-rho", "0.9"]);
    let j = stdout_json(&run(&args));
    assert_eq!(num(&j, &["distinct_books"]), 20.0);
    assert_eq!(num(&j, &["clusters"]), 20.0);
    assert!(num(&j, &["participation_ratio"]) > 19.0);
}

#[test]
fn trials_constant_series_is_one_book_outside_clustering() {
    let (_s, dir) = trials_dir();
    let spec = Spec::new(N);
    let flat = vec![0.0; N];
    let a = write_report(&dir, "flat1", &flat, &spec);
    let b = write_report(&dir, "flat2", &flat, &spec);
    let c = write_report(&dir, "live", &noise(N, 3, 0.01), &spec);
    let j = stdout_json(&run(&["trials", "--wf", &a, &b, &c, "--draws", "50", "--block-len", "8"]));
    assert_eq!(num(&j, &["distinct_books"]), 2.0);
    assert_eq!(num(&j, &["constant_books"]), 1.0);
    assert_eq!(num(&j, &["clusters"]), 2.0);
    assert!(at(&j, &["spa", "skipped"]).as_str().is_some());
    assert!(at(&j, &["pbo", "skipped"]).as_str().is_some());
}

#[test]
fn trials_spa_matches_inference() {
    let (_s, dir) = trials_dir();
    let spec = Spec::new(N);
    let series: Vec<Vec<f64>> = (0..4).map(|i| noise(N, 20 + i, 0.01)).collect();
    let paths: Vec<String> = series.iter().enumerate().map(|(i, s)| write_report(&dir, &format!("m{i}"), s, &spec)).collect();
    let bh: Vec<f64> = noise(N, 77, 0.01).iter().map(|v| v + 0.0002).collect();
    let ladder = write_ladder(&dir, "ladder", &spec.time, &[("cash", vec![0.0; N]), ("buy_and_hold", bh.clone())]);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(|p| p.as_str()));
    args.extend(["--baseline", &ladder, "--block-len", "10", "--draws", "300", "--seed", "5"]);
    let j = stdout_json(&run(&args));
    assert_eq!(at(&j, &["spa", "benchmark"]).as_str(), Some("buy_and_hold"));
    let diffs: Vec<Vec<f64>> = series.iter().map(|s| s.iter().zip(&bh).map(|(a, b)| a - b).collect()).collect();
    let want = inference::spa_test(&diffs, 10, 300, 5).unwrap();
    assert_eq!(num(&j, &["spa", "p_consistent"]), want.p_consistent);
    assert_eq!(num(&j, &["spa", "p_lower"]), want.p_lower);
    assert_eq!(num(&j, &["spa", "p_upper"]), want.p_upper);
    assert_eq!(num(&j, &["spa", "models"]), 4.0);
    assert!(num(&j, &["pbo", "value"]).is_finite());
}

#[test]
fn trials_refuse_mismatched_reports() {
    let (_s, dir) = trials_dir();
    let base = Spec::new(N);
    let x = noise(N, 1, 0.01);
    let a = write_report(&dir, "a", &x, &base);
    let cases: Vec<(&str, Spec, &str)> = vec![
        ("cost", Spec { cost_bps: 20.0, ..Spec::new(N) }, "cost_model.per_side_bps"),
        ("window", Spec { window: "w2", ..Spec::new(N) }, "window_definition"),
        ("ppy", Spec { ppy: 365.0, ..Spec::new(N) }, "annualization.periods_per_year"),
        ("grid", Spec { time: grid(N).iter().map(|t| t + 1.0).collect(), ..Spec::new(N) }, "pooled.period_time"),
    ];
    for (name, spec, field) in cases {
        let b = write_report(&dir, name, &noise(N, 2, 0.01), &spec);
        let o = run(&["trials", "--wf", &a, &b, "--draws", "50"]);
        assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
        assert!(stderr(&o).contains(field), "{name}: {}", stderr(&o));
    }
    let bad = Spec { net_text: Some(String::new()), ..Spec::new(N) };
    let no_series = write_report(&dir, "short", &x, &bad);
    let o = run(&["trials", "--wf", &a, &no_series]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.period_net"), "{}", stderr(&o));
}

// ---------------------------------------------------------------------------
// Walkforward and ladder series, end to end
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
    let path = dir.join("data.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn window_arg() -> String {
    format!("v,{},{},{},{}", T0, T0 + 900 * 309, T0 + 900 * 310, T0 + 900 * 400)
}

fn write_windows(dir: &Path) -> String {
    let text = format!(
        "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,v,{},{},{},{},{}\n",
        T0,
        T0 + 900 * 150,
        T0 + 900 * 100,
        T0 + 900 * 310,
        T0 + 900 * 400
    );
    let path = dir.join("windows.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn walkforward(dir: &Path, data: &str, expr: &str, name: &str, extra: &[&str]) -> String {
    let out = dir.join(format!("{name}.json")).to_string_lossy().into_owned();
    let w = window_arg();
    let mut args = vec![
        "walkforward", "--data", data, "--time", "t", "--target", "y", "--expr", expr, "--vars", "x1,x2",
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
fn trials_single_series_matches_walkforward() {
    let (_s, dir) = trials_dir();
    let data = write_data(&dir);
    let wf = walkforward(&dir, &data, "x1", "wf", &["--bootstrap", "200", "--block-len", "12", "--trials", "1"]);
    let want = num(&read_json(&wf), &["inference", "multiple_testing", "dsr_at_n_eff", "dsr"]);
    assert!(want.is_finite());
    let j = stdout_json(&run(&["trials", "--wf", &wf, "--block-len", "12", "--draws", "200"]));
    let got = num(&j, &["best", "dsr", "value"]);
    assert!((got - want).abs() <= 1e-12, "trials {got} walkforward {want}");
    assert_eq!(num(&j, &["clusters"]), 1.0);
}

#[test]
fn trials_end_to_end_with_ladder_baseline() {
    let (_s, dir) = trials_dir();
    let data = write_data(&dir);
    let windows = write_windows(&dir);
    let a = walkforward(&dir, &data, "x1", "wf-x1", &[]);
    let b = walkforward(&dir, &data, "x2", "wf-x2", &[]);
    let ladder = dir.join("ladder.json").to_string_lossy().into_owned();
    let o = run(&[
        "ladder", "--data", &data, "--target", "y", "--time", "t", "--windows", &windows, "--features", "x1,x2",
        "--purge-seconds", "900", "--draws", "50", "--gbm-trees", "10", "--emit-series", "--out", &ladder,
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let l = read_json(&ladder);
    let periods = num(&l, &["periods"]) as usize;
    let times = match at(&l, &["series", "period_time"]) {
        Json::Arr(a) => a.len(),
        _ => panic!("period_time"),
    };
    assert_eq!(times, periods);
    let cash = match at(&l, &["series", "period_net", "cash"]) {
        Json::Arr(a) => a.iter().all(|v| v.as_f64() == Some(0.0)) && a.len() == periods,
        _ => false,
    };
    assert!(cash);
    assert!(l.get("series").and_then(|s| s.get("period_net")).and_then(|p| p.get("buy_and_hold")).is_some());
    let j = stdout_json(&run(&["trials", "--wf", &a, &b, "--baseline", &ladder, "--draws", "100", "--pbo-blocks", "4"]));
    assert_eq!(num(&j, &["series"]), 2.0);
    assert!(at(&j, &["spa", "benchmark"]).as_str().is_some());
    assert!(num(&j, &["spa", "p_consistent"]).is_finite());
}

#[test]
fn trials_six_reports_process_every_series() {
    let (_s, dir) = trials_dir();
    let (j, _) = six_reports(&dir);
    assert_eq!(num(&j, &["series"]), 6.0);
    assert_eq!(num(&j, &["periods"]), N as f64);
    assert_eq!(num(&j, &["distinct_books"]), 6.0);
    assert_eq!(num(&j, &["spa", "models"]), 6.0);
    assert_eq!(num(&j, &["spa", "n"]), N as f64);
}

#[test]
#[ignore = "timing measurement"]
fn trials_six_reports_timing() {
    let (_s, dir) = trials_dir();
    let (_, secs) = six_reports(&dir);
    println!("six reports: {secs} s");
}

fn six_reports(dir: &Path) -> (Json, f64) {
    let spec = Spec::new(N);
    let paths: Vec<String> = (0..6).map(|i| write_report(dir, &format!("s{i}"), &noise(N, 40 + i as u32, 0.01), &spec)).collect();
    let ladder = write_ladder(dir, "ladder", &spec.time, &[("cash", vec![0.0; N])]);
    let mut args = vec!["trials", "--wf"];
    args.extend(paths.iter().map(|p| p.as_str()));
    args.extend(["--baseline", &ladder]);
    let t = std::time::Instant::now();
    let j = stdout_json(&run(&args));
    (j, t.elapsed().as_secs_f64())
}

// ---------------------------------------------------------------------------
// Compare
// ---------------------------------------------------------------------------

fn compare(dir: &Path, a: &[&[f64]], b: &[&[f64]], extra: &[&str]) -> Output {
    let spec = Spec::new(a[0].len());
    let side = |tag: &str, v: &[&[f64]]| -> String {
        v.iter()
            .enumerate()
            .map(|(i, s)| write_report(dir, &format!("{tag}{i}"), s, &spec))
            .collect::<Vec<_>>()
            .join(",")
    };
    let (pa, pb) = (side("a", a), side("b", b));
    let mut args = vec!["trials", "--compare", pa.as_str(), pb.as_str()];
    args.extend_from_slice(extra);
    run(&args)
}

#[test]
fn compare_identical_series_gives_zero_diff() {
    let (_s, dir) = trials_dir();
    let x = noise(N, 1, 0.01);
    let j = stdout_json(&compare(&dir, &[&x], &[&x], &["--block-len", "21", "--draws", "1000"]));
    assert_eq!(num(&j, &["mean_diff"]), 0.0);
    assert!(num(&j, &["p_two_sided"]) > 0.4);
    assert_eq!(at(&j, &["bootstrap"]).as_str(), Some("stationary"));
}

#[test]
fn compare_detects_planted_shift() {
    let (_s, dir) = trials_dir();
    let b = noise(N, 1, 0.01);
    let e = noise(N, 2, 0.002);
    let a: Vec<f64> = b.iter().zip(&e).map(|(x, n)| x + 0.001 + n).collect();
    let j = stdout_json(&compare(&dir, &[&a], &[&b], &["--block-len", "21", "--draws", "10000", "--seed", "7"]));
    assert!(num(&j, &["p_period"]) <= 0.01, "{}", num(&j, &["p_period"]));
    let (lo, hi) = match at(&j, &["ci"]) {
        Json::Arr(v) => (v[0].as_f64().unwrap(), v[1].as_f64().unwrap()),
        _ => panic!("ci"),
    };
    assert!(lo > 0.0 && hi > lo, "ci [{lo}, {hi}]");
    assert!((num(&j, &["mean_diff"]) - 0.001).abs() < 2e-4);
    assert!((num(&j, &["mean_diff_annual"]) - num(&j, &["mean_diff"]) * PPY).abs() < 1e-9);
}

#[test]
fn compare_paired_ci_narrower_than_unpaired() {
    let (_s, dir) = trials_dir();
    let (z1, z2) = (noise(N, 11, 0.01), noise(N, 12, 0.01));
    let rho = 0.95f64;
    let a = z1.clone();
    let b: Vec<f64> = z1.iter().zip(&z2).map(|(p, q)| rho * p + (1.0 - rho * rho).sqrt() * q + 0.0002).collect();
    let j = stdout_json(&compare(&dir, &[&a], &[&b], &["--block-len", "21", "--draws", "4000", "--seed", "3"]));
    let (lo, hi) = match at(&j, &["ci"]) {
        Json::Arr(v) => (v[0].as_f64().unwrap(), v[1].as_f64().unwrap()),
        _ => panic!("ci"),
    };
    let paired = (hi - lo) / 2.0;
    let da = inference::stationary_bootstrap_draws(&a, 21, 4000, 1);
    let db = inference::stationary_bootstrap_draws(&b, 21, 4000, 2);
    let diff: Vec<f64> = da.means.iter().zip(&db.means).map(|(p, q)| p - q).collect();
    let iv = inference::percentile_interval(&diff, 0.95);
    let unpaired = (iv.hi - iv.lo) / 2.0;
    assert!(paired < 0.5 * unpaired, "paired {paired} unpaired {unpaired}");
}

#[test]
fn compare_circular_matches_inference() {
    let (_s, dir) = trials_dir();
    let b = noise(N, 1, 0.01);
    let a: Vec<f64> = b.iter().zip(noise(N, 2, 0.003)).map(|(x, n)| x + 0.0003 + n).collect();
    let j = stdout_json(&compare(&dir, &[&a], &[&b], &["--bootstrap", "circular", "--block-len", "12", "--draws", "500", "--seed", "9"]));
    let d: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x - y).collect();
    let draws = inference::bootstrap_draws(&d, 12, 500, 9);
    assert_eq!(num(&j, &["p_period"]), inference::bootstrap_p_mean_positive(&d, &draws));
    assert_eq!(at(&j, &["bootstrap"]).as_str(), Some("circular"));
}

#[test]
fn compare_is_deterministic() {
    let (_s, dir) = trials_dir();
    let b = noise(N, 1, 0.01);
    let a: Vec<f64> = b.iter().zip(noise(N, 2, 0.003)).map(|(x, n)| x + 0.0003 + n).collect();
    let args = ["--block-len", "21", "--draws", "2000", "--seed", "4"];
    let first = compare(&dir, &[&a], &[&b], &args);
    let second = compare(&dir, &[&a], &[&b], &args);
    assert_eq!(code(&first), 0, "{}", stderr(&first));
    assert_eq!(first.stdout, second.stdout);
}

fn seed_sets() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let a = (0..3).map(|i| noise(N, 50 + i, 0.01).iter().map(|v| v + 0.0004 * (i as f64 + 1.0)).collect()).collect();
    let b = (0..3).map(|i| noise(N, 60 + i, 0.01).iter().map(|v| v + 0.0001 * (i as f64 + 1.0)).collect()).collect();
    (a, b)
}

fn annual(s: &[f64]) -> f64 {
    inference::mean(s) * PPY
}

#[test]
fn compare_seed_z_matches_formula() {
    let (_s, dir) = trials_dir();
    let (a, b) = seed_sets();
    let ar: Vec<&[f64]> = a.iter().map(|v| v.as_slice()).collect();
    let br: Vec<&[f64]> = b.iter().map(|v| v.as_slice()).collect();
    let j = stdout_json(&compare(&dir, &ar, &br, &["--block-len", "21", "--draws", "200"]));
    let ma: Vec<f64> = a.iter().map(|s| annual(s)).collect();
    let mb: Vec<f64> = b.iter().map(|s| annual(s)).collect();
    let delta = inference::mean(&ma) - inference::mean(&mb);
    let var = |m: &[f64]| inference::sample_sd(m).powi(2) * (m.len() as f64 - 1.0);
    let s = ((var(&ma) + var(&mb)) / 4.0).sqrt();
    let z = delta / (s * (1.0f64 / 3.0 + 1.0 / 3.0).sqrt());
    assert!((num(&j, &["seeds", "z"]) - z).abs() <= 1e-12 * z.abs().max(1.0), "{} vs {z}", num(&j, &["seeds", "z"]));
    assert!((num(&j, &["seeds", "s_seed"]) - s).abs() <= 1e-12 * s);
    assert_eq!(num(&j, &["seeds", "df"]), 4.0);
    assert!((num(&j, &["seeds", "delta"]) - delta).abs() <= 1e-12);

    let j = stdout_json(&compare(&dir, &ar, &br, &["--block-len", "21", "--draws", "200", "--s-seed", "0.5"]));
    let z = delta / (0.5 * (1.0f64 / 3.0 + 1.0 / 3.0).sqrt());
    assert!((num(&j, &["seeds", "z"]) - z).abs() <= 1e-12 * z.abs().max(1.0));
    assert_eq!(num(&j, &["seeds", "s_seed"]), 0.5);
}

#[test]
fn compare_seed_z_needs_two_seeds() {
    let (_s, dir) = trials_dir();
    let (a, b) = seed_sets();
    let j = stdout_json(&compare(&dir, &[&a[0]], &[&b[0]], &["--block-len", "21", "--draws", "200"]));
    assert!(matches!(at(&j, &["seeds", "z"]), Json::Null));
    assert!(at(&j, &["seeds", "z_reason"]).as_str().is_some());
    let j = stdout_json(&compare(&dir, &[&a[0]], &[&b[0]], &["--block-len", "21", "--draws", "200", "--s-seed", "0.25"]));
    let delta = annual(&a[0]) - annual(&b[0]);
    let z = delta / (0.25 * 2.0f64.sqrt());
    assert!((num(&j, &["seeds", "z"]) - z).abs() <= 1e-12 * z.abs().max(1.0));
}

#[test]
fn compare_refuses_mismatched_reports() {
    let (_s, dir) = trials_dir();
    let x = noise(N, 1, 0.01);
    let y = noise(N, 2, 0.01);
    let a = write_report(&dir, "a", &x, &Spec::new(N));
    let cases: Vec<(&str, Spec, &str)> = vec![
        ("cost", Spec { cost_bps: 20.0, ..Spec::new(N) }, "cost_model.per_side_bps"),
        ("ppy", Spec { ppy: 365.0, ..Spec::new(N) }, "annualization.periods_per_year"),
        ("grid", Spec { time: grid(N).iter().map(|t| t + 1.0).collect(), ..Spec::new(N) }, "pooled.period_time"),
    ];
    for (name, spec, field) in cases {
        let b = write_report(&dir, name, &y, &spec);
        let o = run(&["trials", "--compare", &a, &b]);
        assert_eq!(code(&o), 2, "{name}: {}", stderr(&o));
        assert!(stderr(&o).contains(field), "{name}: {}", stderr(&o));
    }
    let mut text = arr(&y);
    text.replace_range(0..text.find(',').unwrap(), "null");
    let nan = write_report(&dir, "nan", &y, &Spec { net_text: Some(text), ..Spec::new(N) });
    let o = run(&["trials", "--compare", &a, &nan]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.period_net"), "{}", stderr(&o));
    let o = run(&["trials", "--compare", &a, &a]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("compare"), "{}", stderr(&o));
    let o = run(&["trials", "--compare", &a, &a, "--wf", &a]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
}

#[test]
fn compare_short_series_is_flagged() {
    let (_s, dir) = trials_dir();
    let x = noise(50, 1, 0.01);
    let y = noise(50, 2, 0.01);
    let j = stdout_json(&compare(&dir, &[&x], &[&y], &["--block-len", "21", "--draws", "100"]));
    assert!(matches!(at(&j, &["short_series"]), Json::Bool(true)));
}

#[test]
fn compare_two_sets_of_three_count_the_work() {
    let (_s, dir) = trials_dir();
    let (j, _) = big_compare(&dir);
    assert_eq!(num(&j, &["draws"]), 10000.0);
    assert_eq!(num(&j, &["n"]), N as f64);
    assert_eq!(num(&j, &["seeds", "n_a"]), 3.0);
    assert_eq!(num(&j, &["seeds", "n_b"]), 3.0);
}

#[test]
#[ignore = "timing measurement"]
fn compare_two_sets_of_three_timing() {
    let (_s, dir) = trials_dir();
    let (_, secs) = big_compare(&dir);
    println!("compare 3+3 reports, 10000 draws: {secs} s");
}

fn big_compare(dir: &Path) -> (Json, f64) {
    let (a, b) = seed_sets();
    let ar: Vec<&[f64]> = a.iter().map(|v| v.as_slice()).collect();
    let br: Vec<&[f64]> = b.iter().map(|v| v.as_slice()).collect();
    let t = std::time::Instant::now();
    let o = compare(dir, &ar, &br, &["--block-len", "21", "--draws", "10000", "--seed", "7"]);
    let secs = t.elapsed().as_secs_f64();
    (stdout_json(&o), secs)
}

// ---------------------------------------------------------------------------
// #1163: the fitted-cut rule is part of the threshold rule
// ---------------------------------------------------------------------------

fn strs(j: &Json) -> Vec<&str> {
    match j {
        Json::Arr(v) => v.iter().map(|x| x.as_str().unwrap()).collect(),
        other => panic!("not an array: {other:?}"),
    }
}

fn strs_of<'a>(j: &'a Json, key: &str) -> Vec<&'a str> {
    match j {
        Json::Arr(v) => v.iter().map(|x| at(x, &[key]).as_str().unwrap()).collect(),
        other => panic!("not an array: {other:?}"),
    }
}

fn fixed_cut(n: usize) -> Spec {
    Spec { quantile: None, fixed: Some(0.5), ..Spec::new(n) }
}

fn fitted_cut(n: usize, e: u32, b: u32, l: f64) -> Spec {
    Spec { fit: Some((e, b, l)), ..fixed_cut(n) }
}

fn arm_pair(dir: &Path) -> (String, String) {
    let fitted = write_report(dir, "fitted", &noise(N, 1, 0.01), &fitted_cut(N, 30, 4, 1.0));
    let fixed = write_report(dir, "fixed", &noise(N, 2, 0.01), &fixed_cut(N));
    (fitted, fixed)
}

fn assert_refused(o: &Output, field: &str, hint: bool) {
    assert_eq!(code(o), 2, "{}", stderr(o));
    let e = stderr(o);
    assert!(e.contains(&format!("'{field}'")), "{e}");
    assert_eq!(e.contains("--threshold-arms"), hint, "{e}");
}

#[test]
fn trials_refuse_a_fitted_cut_mixed_with_a_fixed_cut() {
    let (_s, dir) = trials_dir();
    let (fitted, fixed) = arm_pair(&dir);
    let o = run(&["trials", "--wf", &fitted, &fixed, "--draws", "50"]);
    assert_refused(&o, "threshold.fit", true);
    let o = run(&["trials", "--wf", &fixed, &fitted, "--draws", "50"]);
    assert_refused(&o, "threshold.fit", true);
}

#[test]
fn compare_refuses_a_fitted_cut_mixed_with_a_fixed_cut() {
    let (_s, dir) = trials_dir();
    let (fitted, fixed) = arm_pair(&dir);
    let o = run(&["trials", "--compare", &fitted, &fixed, "--draws", "50"]);
    assert_refused(&o, "threshold.fit", true);
}

#[test]
fn trials_refuse_fitted_cuts_with_other_values() {
    let (_s, dir) = trials_dir();
    let base = write_report(&dir, "base", &noise(N, 1, 0.01), &fitted_cut(N, 30, 4, 1.0));
    let cases = [
        ("entries", fitted_cut(N, 5, 4, 1.0), "threshold.cut_min_entries"),
        ("blocks", fitted_cut(N, 30, 1, 1.0), "threshold.cut_blocks"),
        ("lambda", fitted_cut(N, 30, 4, 2.0), "threshold.cut_lambda"),
    ];
    for (name, spec, field) in cases {
        let other = write_report(&dir, name, &noise(N, 2, 0.01), &spec);
        assert_refused(&run(&["trials", "--wf", &base, &other, "--draws", "50"]), field, true);
        assert_refused(&run(&["trials", "--compare", &base, &other, "--draws", "50"]), field, true);
    }
    let same = write_report(&dir, "same", &noise(N, 3, 0.01), &fitted_cut(N, 30, 4, 1.0));
    let j = stdout_json(&run(&["trials", "--wf", &base, &same, "--draws", "50", "--block-len", "8"]));
    assert!(j.get("threshold_arms").is_none());
}

#[test]
fn trials_threshold_arms_accepts_the_fit_fields_only() {
    let (_s, dir) = trials_dir();
    let (fitted, fixed) = arm_pair(&dir);
    let j = stdout_json(&run(&["trials", "--wf", &fitted, &fixed, "--threshold-arms", "--draws", "50", "--block-len", "8"]));
    assert!(matches!(at(&j, &["threshold_arms"]), Json::Bool(true)));
    assert_eq!(at(&j, &["series"]).as_f64(), Some(2.0));
    let rules = strs_of(at(&j, &["members"]), "threshold_rule");
    assert_eq!(rules, ["fitness(min_entries=30,blocks=4,lambda=1)", "fixed"]);

    let j = stdout_json(&run(&["trials", "--compare", &fitted, &fixed, "--threshold-arms", "--draws", "50", "--block-len", "8"]));
    assert!(matches!(at(&j, &["threshold_arms"]), Json::Bool(true)));
    assert_eq!(strs(at(&j, &["threshold_rule_a"])), ["fitness(min_entries=30,blocks=4,lambda=1)"]);
    assert_eq!(strs(at(&j, &["threshold_rule_b"])), ["fixed"]);

    let quantile = write_report(&dir, "quantile", &noise(N, 3, 0.01), &Spec::new(N));
    let q = stdout_json(&run(&["trials", "--wf", &quantile, "--threshold-arms", "--draws", "50", "--block-len", "8"]));
    assert_eq!(strs_of(at(&q, &["members"]), "threshold_rule"), ["quantile"]);

    let other_cut = write_report(&dir, "cut", &noise(N, 4, 0.01), &Spec { fixed: Some(0.25), ..fixed_cut(N) });
    let other_policy = write_report(&dir, "policy", &noise(N, 5, 0.01), &Spec { policy: "band", ..fixed_cut(N) });
    for (other, field) in [(&other_cut, "threshold.fixed"), (&other_policy, "policy.name")] {
        assert_refused(&run(&["trials", "--wf", &fitted, other, "--threshold-arms", "--draws", "50"]), field, false);
        assert_refused(&run(&["trials", "--compare", &fixed, other, "--threshold-arms", "--draws", "50"]), field, false);
    }
}

#[test]
fn trials_without_the_flag_keep_the_old_reports_matching() {
    let (_s, dir) = trials_dir();
    let a = write_report(&dir, "old_a", &noise(N, 1, 0.01), &Spec::new(N));
    let b = write_report(&dir, "old_b", &noise(N, 2, 0.01), &Spec::new(N));
    let j = stdout_json(&run(&["trials", "--wf", &a, &b, "--draws", "50", "--block-len", "8"]));
    assert!(j.get("threshold_arms").is_none());
    assert!(!String::from_utf8_lossy(&run(&["trials", "--wf", &a, &b, "--draws", "50", "--block-len", "8"]).stdout).contains("threshold_rule"));
}

/// #1199: one walk-forward report of two symbols, hourly over 600 hours; the gate `g` is off
/// for both symbols on hours 300..380 of the test window.
fn grid_report(dir: &Path, expr: &str, gate: bool, grid: rcsr::walkforward::PeriodGrid) -> String {
    use rcsr::walkforward::{self, EvalSpec, Policy, ProtocolOptions, Window};
    let mut text = String::from("t,sym,x1,x2,g,y\n");
    for i in 0..600usize {
        for (k, sym) in ["A", "B"].iter().enumerate() {
            let f = i as f64 + 0.37 * k as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let y = 0.01 * (frac(f * 0.324_717_957_244_746) - 0.5) + 0.002 * x1;
            let g = if (300..380).contains(&i) { -1 } else { 1 };
            text.push_str(&format!("{},{sym},{x1},{x2},{g},{y}\n", T0 + STEP * i as i64));
        }
    }
    let table = rcsr::data::parse_table(&text).unwrap();
    let at = |h: i64| (T0 + STEP * h) as f64;
    let spec = EvalSpec {
        expression: expr.to_string(),
        vars: vec!["x1".to_string(), "x2".to_string()],
        target: "y".to_string(),
        time: "t".to_string(),
        group: Some("sym".to_string()),
        windows: vec![Window { label: "w".to_string(), train_start: at(0), train_end: at(200), test_start: at(200), test_end: at(600) }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.7,
        threshold_fixed: Some(0.0),
        validation_frac: 0.0,
        periods_per_year: Some(PPY),
        candidates: Vec::new(),
        band_select: false,
        pool_reset_flat: true,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 0,
        policy: Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    };
    let opts = ProtocolOptions {
        period_grid: grid,
        mask: gate.then(|| rcsr::regime_gate::GateSpec::parse("g", 0.0, 1).unwrap()),
        ..ProtocolOptions::default()
    };
    let r = walkforward::run_table_with(&table, &spec, "test", &opts).unwrap();
    let name = format!("{}-{}-{}.json", expr, if gate { "gated" } else { "plain" }, grid.as_str());
    let path = dir.join(name);
    std::fs::write(&path, r.json()).unwrap();
    path.to_string_lossy().into_owned()
}

#[test]
fn trials_common_grid_across_configurations() {
    use rcsr::walkforward::PeriodGrid::{Score, Target};
    let (_s, dir) = trials_dir();
    let target: Vec<String> = [("x1", false), ("x2", false), ("x1", true), ("x2", true)]
        .iter()
        .map(|&(e, gate)| grid_report(&dir, e, gate, Target))
        .collect();
    let mut args = vec!["trials", "--wf"];
    args.extend(target.iter().map(String::as_str));
    args.extend_from_slice(&["--draws", "100", "--pbo-blocks", "4"]);
    let j = stdout_json(&run(&args));
    assert_eq!(num(&j, &["series"]), 4.0);
    assert_eq!(num(&j, &["periods"]), 400.0, "every hour of the test window, gate on or off");
    let pbo = num(&j, &["pbo", "value"]);
    assert!((0.0..=1.0).contains(&pbo), "{pbo}");
    // Today's grid: the gated book has no period while the gate is off.
    let gated_score = grid_report(&dir, "x1", true, Score);
    let plain_score = grid_report(&dir, "x1", false, Score);
    let o = run(&["trials", "--wf", &plain_score, &gated_score, "--draws", "50"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.period_time"), "{}", stderr(&o));
    // A score report and a target report never mix.
    let o = run(&["trials", "--wf", &target[0], &plain_score, "--draws", "50"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.grid.rule"), "{}", stderr(&o));
}

#[test]
fn trials_refuse_mixed_missing_score_rules() {
    let (_s, dir) = trials_dir();
    let rule = |name: &str, seed: u32, book: Option<&'static str>| write_report(&dir, name, &noise(N, seed, 0.01), &Spec { book, ..Spec::new(N) });
    let flat = rule("flat", 1, Some("flat"));
    let hold = rule("hold", 2, Some("hold"));
    let carry = rule("carry", 3, None);
    for other in [&hold, &carry] {
        for mode in ["--wf", "--compare"] {
            let o = run(&["trials", mode, &flat, other, "--draws", "50"]);
            assert_eq!(code(&o), 2, "{mode}: {}", stderr(&o));
            assert!(stderr(&o).contains("'book.missing_score'") && stderr(&o).contains("--missing-score"), "{mode}: {}", stderr(&o));
        }
    }
    let same = rule("flat2", 4, Some("flat"));
    let o = run(&["trials", "--wf", &flat, &same, "--draws", "50", "--block-len", "8"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

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
    let data = dir.join("panel.csv");
    std::fs::write(&data, csv).unwrap();
    let front = dir.join("run.json");
    std::fs::write(
        &front,
        r#"{"tool":"rcsr","variables":["x"],"target_spec":{"return_column":"target"},
        "split":{"train_time":[0,356400],"validation_time":[360000,536400],"required_purge_seconds":3600,"purge_seconds":3600},
        "objective":{"spec":{"threshold":0,"per_side_bps":10}},
        "pareto":[{"expression":"x"},{"expression":"0.5 - x"},{"expression":"sqrt(x) - 1"}]}"#,
    )
    .unwrap();
    let out = dir.join("wf_members.json");
    let o = run(&[
        "walkforward", "--data", data.to_str().unwrap(), "--time", "ts", "--group-by", "sym", "--horizon", "3600",
        "--periods-per-year", "8760", "--front", front.to_str().unwrap(), "--all-members", "--period-grid", "target",
        "--window", "w,0,540000,543600,1080000", "--json", "--out", out.to_str().unwrap(),
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    out.to_string_lossy().into_owned()
}

#[test]
fn trials_reads_members() {
    let (_s, dir) = trials_dir();
    let report = members_report(&dir);
    let all = stdout_json(&run(&["trials", "--wf", &report, "--members", "all", "--draws", "50"]));
    assert_eq!(num(&all, &["series"]), 3.0);
    let labels: Vec<String> = match at(&all, &["members"]) {
        Json::Arr(m) => m.iter().map(|e| at(e, &["label"]).as_str().unwrap().to_string()).collect(),
        other => panic!("{other:?}"),
    };
    assert_eq!(labels, ["wf_members#1", "wf_members#2", "wf_members#3"]);
    for args in [vec![], vec!["--members", "selected"]] {
        let one = stdout_json(&run(&[&["trials", "--wf", report.as_str(), "--draws", "50"][..], &args].concat()));
        assert_eq!(num(&one, &["series"]), 1.0, "{args:?}");
    }
    // A report without members, an unknown mode and --compare are refused.
    let plain = write_report(&dir, "plain", &noise(N, 1, 0.01), &Spec::new(N));
    let o = run(&["trials", "--wf", &plain, "--members", "all", "--draws", "50"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("'members'") && stderr(&o).contains("--all-members"), "{}", stderr(&o));
    let o = run(&["trials", "--wf", &report, "--members", "some", "--draws", "50"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--members must be all or selected"), "{}", stderr(&o));
    let o = run(&["trials", "--compare", &report, &report, "--members", "all", "--draws", "50"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--members does not apply to --compare"), "{}", stderr(&o));
}

// ---------------------------------------------------------------------------
// #1208: periods_per_year against the series
// ---------------------------------------------------------------------------

const DAY: i64 = 86_400;

fn stepped(n: usize, step: i64) -> Vec<f64> {
    (0..n).map(|i| (T0 + step * i as i64) as f64).collect()
}

fn daily(ppy: f64, period_seconds: Option<f64>) -> Spec {
    Spec { ppy, period_seconds, time: stepped(30, DAY), ..Spec::new(30) }
}

fn trials_one(path: &str) -> Output {
    run(&["trials", "--wf", path, "--draws", "50", "--block-len", "2"])
}

#[test]
fn trials_refuses_ppy_contradicting_period_time() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-x", &noise(30, 1, 0.01), &daily(8760.0, None));
    let o = trials_one(&p);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("annualization.periods_per_year") && e.contains("8760") && e.contains("365") && e.contains("wf-x.json"), "{e}");
    let o = run(&["trials", "--wf", &p, "--table", "--draws", "50", "--block-len", "2"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("annualization.periods_per_year"), "{}", stderr(&o));
}

#[test]
fn trials_accepts_consistent_daily_ppy() {
    let (_s, dir) = trials_dir();
    let net = noise(30, 1, 0.01);
    let p = write_report(&dir, "wf-x", &net, &daily(365.0, None));
    let j = stdout_json(&trials_one(&p));
    let got = num(&j, &["best", "sharpe_annual"]);
    let want = inference::sharpe_per_period(&net) * 365f64.sqrt();
    assert!((got - want).abs() < 1e-9, "{got} vs {want}");
}

#[test]
fn trials_weekly_ppy_checked() {
    let (_s, dir) = trials_dir();
    let weekly = |ppy| Spec { ppy, time: stepped(20, 7 * DAY), ..Spec::new(20) };
    let ok = write_report(&dir, "ok", &noise(20, 1, 0.01), &weekly(365.0 / 7.0));
    let o = trials_one(&ok);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let bad = write_report(&dir, "bad", &noise(20, 1, 0.01), &weekly(8760.0));
    let o = trials_one(&bad);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("annualization.periods_per_year"), "{}", stderr(&o));
}

#[test]
fn trials_hourly_output_unchanged() {
    let (_s, dir) = trials_dir();
    let net = noise(100, 1, 0.01);
    let p = write_report(&dir, "wf-h", &net, &Spec::new(100));
    let o = trials_one(&p);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(stderr(&o), "");
    let j = stdout_json(&o);
    assert_eq!(num(&j, &["periods_per_year"]), 8760.0);
    let got = num(&j, &["best", "sharpe_annual"]);
    assert_eq!(got.to_bits(), (inference::sharpe_per_period(&net) * 8760f64.sqrt()).to_bits());
}

#[test]
fn trials_refuses_ppy_contradicting_period_seconds() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-x", &noise(30, 1, 0.01), &daily(365.0, Some(3600.0)));
    let o = trials_one(&p);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("annualization.period_seconds") && e.contains("365") && e.contains("8760"), "{e}");
}

#[test]
fn trials_accepts_report_without_period_seconds() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-x", &noise(30, 1, 0.01), &daily(365.0, None));
    let o = trials_one(&p);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let q = write_report(&dir, "wf-y", &noise(30, 1, 0.01), &daily(365.0, Some(86400.0)));
    let o = trials_one(&q);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[test]
fn trials_short_series_note() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-s", &noise(2, 1, 0.01), &Spec { time: stepped(2, DAY), ..Spec::new(2) });
    let o = trials_one(&p);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(stderr(&o).contains("too few periods"), "{}", stderr(&o));
}

#[test]
fn trials_refuses_nonpositive_ppy() {
    let (_s, dir) = trials_dir();
    for ppy in [0.0, -365.0] {
        let p = write_report(&dir, "wf-x", &noise(30, 1, 0.01), &daily(ppy, None));
        let o = trials_one(&p);
        assert_eq!(code(&o), 2, "{ppy}: {}", stderr(&o));
        assert!(stderr(&o).contains("annualization.periods_per_year"), "{}", stderr(&o));
    }
}

// #1328: a score-grid series thinned at random keeps its native period_seconds

fn thinned(n: usize, step: i64) -> Vec<f64> {
    let mut t = Vec::new();
    let mut i = 0usize;
    while t.len() < n {
        let u = (i as f64 * 0.682_327_803_828_019_3 + 0.123).fract();
        if u < 0.3 {
            t.push((T0 + step * i as i64) as f64);
        }
        i += 1;
    }
    t
}

fn sparse(ppy: f64, grid_rule: Option<&'static str>) -> Spec {
    Spec { ppy, period_seconds: Some(3600.0), time: thinned(60, STEP), grid_rule, ..Spec::new(60) }
}

#[test]
fn trials_score_grid_sparse_time_accepted() {
    let (_s, dir) = trials_dir();
    let t = sparse(8760.0, None).time;
    let mut d: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    d.sort_by(|a, b| a.total_cmp(b));
    assert!(d[d.len() / 2] > 1.5 * STEP as f64, "median step {}", d[d.len() / 2]);
    let p = write_report(&dir, "wf-x", &noise(60, 1, 0.01), &sparse(8760.0, None));
    let o = trials_one(&p);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[test]
fn trials_score_grid_sparse_time_wrong_period_seconds_refused() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-x", &noise(60, 1, 0.01), &sparse(365.0, None));
    let o = trials_one(&p);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("annualization.period_seconds"), "{}", stderr(&o));
}

#[test]
fn trials_period_grid_step_still_checked() {
    let (_s, dir) = trials_dir();
    let p = write_report(&dir, "wf-x", &noise(60, 1, 0.01), &sparse(8760.0, Some("target")));
    let o = trials_one(&p);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("pooled.period_time"), "{}", stderr(&o));
}
