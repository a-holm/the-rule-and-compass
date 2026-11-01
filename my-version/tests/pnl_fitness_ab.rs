//! #483: `--fitness pnl` against `--fitness metric` on the synthetic panel, with a
//! walk-forward of each run's front. Ignored: it is a measurement.
//!
//! `cargo test --release --features cuda --test pnl_fitness_ab -- --ignored --nocapture`
//!
//! Environment: `RCSR_AB_DIR` (work directory), `RCSR_AB_SEEDS` (default `11,12,13`),
//! `RCSR_AB_BACKENDS` (default `cpu,gpu`), `RCSR_AB_SYMBOLS` and `RCSR_AB_BARS`
//! (default 16 and 20000), `RCSR_AB_GENERATIONS` (default 30).

use std::path::{Path, PathBuf};
use std::process::Command;

use rcsr::devlog::Json;

#[path = "common/synth_panel.rs"]
mod synth_panel;

const VARS: &str = "r1,r3,r6,r12,r24,r48,vol20,vol60,ma20,ma50,rsi14,range1,clv1,body1,don20,vz96";
const FEATURES: &str = "returns,ma,vol,rsi,range,candle,donchian,vz";
const COST_BPS: &str = "5";
const WINDOWS: usize = 4;
const HOUR: i64 = 3600;

fn env_or(key: &str, default: &str) -> String {
    std::env::var(key).ok().filter(|v| !v.is_empty()).unwrap_or_else(|| default.to_string())
}

fn rcsr(args: &[String]) -> (i32, String, String) {
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

fn must(args: &[String]) -> String {
    let (code, out, err) = rcsr(args);
    assert_eq!(code, 0, "rcsr {} failed: {}", args.join(" "), err);
    out
}

fn s(v: &[&str]) -> Vec<String> {
    v.iter().map(|x| x.to_string()).collect()
}

fn num(j: &Json, path: &[&str]) -> f64 {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing {}", path.join(".")));
    }
    cur.as_f64().unwrap_or(f64::NAN)
}

fn opt_num(j: &Json, path: &[&str]) -> f64 {
    let mut cur = j;
    for k in path {
        match cur.get(k) {
            Some(n) => cur = n,
            None => return f64::NAN,
        }
    }
    cur.as_f64().unwrap_or(f64::NAN)
}

fn prepare(dir: &Path, symbols: usize, bars: usize) -> (PathBuf, PathBuf) {
    let feat = dir.join("feat.rcsb");
    let spec = dir.join("feat.targets.json");
    if feat.exists() && spec.exists() {
        return (feat, spec);
    }
    let csv = dir.join("panel.csv");
    std::fs::write(&csv, synth_panel::synth_panel_csv(symbols, bars, 7)).unwrap();
    let panel = dir.join("panel.rcsb");
    must(&s(&["import", csv.to_str().unwrap(), panel.to_str().unwrap(), "--time", "ts", "--quiet"]));
    let mut a = s(&["enrich", "--data", panel.to_str().unwrap(), "--out", feat.to_str().unwrap()]);
    a.extend(s(&["--features", FEATURES, "--label", "long:4,name=h4,fee=5,stride=4", "--cpu"]));
    must(&a);
    std::fs::remove_file(&csv).ok();
    (feat, spec)
}

struct Arm {
    backend: &'static str,
    fitness: &'static str,
    seed: u64,
}

struct ArmResult {
    val_net: f64,
    test_net: f64,
    test_sharpe: f64,
    flips: f64,
    exposure: f64,
    front: usize,
    search_s: f64,
    sec_per_gen: f64,
    eval_s: f64,
    fit_s: f64,
    device_frac: f64,
    wf_net: f64,
    wf_min_window: f64,
    wf_trades: f64,
    wf_exposure: f64,
    wf_sharpe: f64,
    wf_dsr: f64,
    wf_se: f64,
}

fn run_arm(dir: &Path, feat: &Path, spec: &Path, arm: &Arm, generations: usize, windows: &[String]) -> ArmResult {
    let tag = format!("{}-{}-{}", arm.backend, arm.fitness, arm.seed);
    let run = dir.join(format!("{tag}.run.json"));
    let dev = dir.join(format!("{tag}.dev.jsonl"));
    let wf_out = dir.join(format!("{tag}.wf.json"));
    let mut a = s(&["--data", feat.to_str().unwrap(), "--target", "lbl_h4_y", "--target-spec", spec.to_str().unwrap()]);
    a.extend(s(&["--vars", VARS, "--time", "ts", "--group-by", "symbol", "--split", "temporal"]));
    a.extend(s(&["--validation-frac", "0.2", "--test-frac", "0.2", "--objective", "pnl", "--metric", "squared"]));
    a.extend(s(&["--cost-bps", COST_BPS, "--fill", "drop", "--no-project", "--pop", "64", "--islands", "4"]));
    a.extend(s(&["--patience", "0", "--fitness", arm.fitness, "--seed", &arm.seed.to_string()]));
    a.extend(s(&["--generations", &generations.to_string()]));
    a.extend(s(&[if arm.backend == "gpu" { "--gpu" } else { "--cpu" }]));
    a.extend(s(&["--json-out", run.to_str().unwrap(), "--dev-log", dev.to_str().unwrap(), "--dev-log-every", "1", "--json"]));
    must(&a);

    let r = Json::parse(&std::fs::read_to_string(&run).unwrap()).unwrap();
    let front = match r.get("pareto") {
        Some(Json::Arr(v)) => v.len(),
        _ => 0,
    };
    let events: Vec<Json> = std::fs::read_to_string(&dev).unwrap().lines().filter_map(Json::parse).collect();
    let end = events
        .iter()
        .find(|e| e.get("event").and_then(Json::as_str) == Some("run_end"))
        .expect("dev log has no run_end");
    let phase = |name: &str| -> f64 {
        events
            .iter()
            .filter(|e| e.get("event").and_then(Json::as_str) == Some("generation"))
            .map(|e| opt_num(e, &["phases", name]))
            .sum()
    };
    let search_s = num(end, &["search"]);

    let mut w = s(&["walkforward", "--data", feat.to_str().unwrap(), "--front", run.to_str().unwrap()]);
    w.extend(s(&["--time", "ts", "--group-by", "symbol"]));
    w.extend(s(&["--threshold-fixed", "0", "--validation-frac", "0.2", "--bootstrap", "1000", "--block-len", "24"]));
    if front > 1 {
        w.push("--select-band".to_string());
    }
    w.extend(s(&["--json", "--out", wf_out.to_str().unwrap()]));
    for win in windows {
        w.extend(s(&["--window", win]));
    }
    must(&w);
    let wf = Json::parse(&std::fs::read_to_string(&wf_out).unwrap()).unwrap();
    let window_returns: Vec<f64> = match wf.get("windows") {
        Some(Json::Arr(v)) => v.iter().map(|w| opt_num(w, &["metrics", "total_return"])).collect(),
        _ => Vec::new(),
    };

    ArmResult {
        val_net: num(&r, &["objective", "validation", "net_pnl"]),
        test_net: num(&r, &["objective", "test", "net_pnl"]),
        test_sharpe: num(&r, &["objective", "test", "sharpe"]),
        flips: num(&r, &["objective", "test", "trades"]),
        exposure: num(&r, &["objective", "test", "exposure"]),
        front,
        search_s,
        sec_per_gen: search_s / generations as f64,
        eval_s: phase("eval_cpu"),
        fit_s: phase("fit_cpu"),
        device_frac: opt_num(end, &["device_fraction_of_run"]),
        wf_net: opt_num(&wf, &["pooled", "total_return"]),
        wf_min_window: window_returns.iter().cloned().fold(f64::INFINITY, f64::min),
        wf_trades: opt_num(&wf, &["pooled", "trades"]),
        wf_exposure: opt_num(&wf, &["pooled", "exposure"]),
        wf_sharpe: opt_num(&wf, &["pooled", "sharpe"]),
        wf_dsr: opt_num(&wf, &["inference", "multiple_testing", "dsr_at_n_eff", "dsr"]),
        wf_se: opt_num(&wf, &["inference", "bootstrap", "mean_ci", "se"]) * opt_num(&wf, &["pooled", "periods"]),
    }
}

#[test]
#[ignore]
fn pnl_fitness_against_metric_fitness() {
    let symbols: usize = env_or("RCSR_AB_SYMBOLS", "16").parse().unwrap();
    let bars: usize = env_or("RCSR_AB_BARS", "20000").parse().unwrap();
    let generations: usize = env_or("RCSR_AB_GENERATIONS", "30").parse().unwrap();
    let seeds: Vec<u64> = env_or("RCSR_AB_SEEDS", "11,12,13").split(',').map(|x| x.parse().unwrap()).collect();
    let backends: Vec<String> = env_or("RCSR_AB_BACKENDS", "cpu,gpu").split(',').map(String::from).collect();
    let dir = PathBuf::from(env_or(
        "RCSR_AB_DIR",
        &std::env::temp_dir().join(format!("rcsr-ab-{symbols}x{bars}")).to_string_lossy(),
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let (feat, spec) = prepare(&dir, symbols, bars);

    // The search holds out the last 20 % as test rows; the windows split them in
    // WINDOWS equal parts after the boundary purge of the 4-bar label.
    let t0 = synth_panel::SYNTH_T0;
    let test_first = t0 + HOUR * (bars as i64 * 8 / 10);
    let first = test_first + HOUR * 6;
    let last = t0 + HOUR * (bars as i64 - 6);
    let step = (last - first) / WINDOWS as i64;
    let windows: Vec<String> = (0..WINDOWS as i64)
        .map(|k| format!("w{},{},{},{},{}", k + 1, t0, first + k * step - 5 * HOUR, first + k * step, first + (k + 1) * step - HOUR))
        .collect();

    println!("backend fitness seed | val_net test_net test_sharpe test_trades test_exposure front | search_s s/gen eval_cpu_s fit_cpu_s dev_frac | wf_net wf_min_win wf_trades wf_exposure wf_sharpe wf_dsr wf_se");
    let mut rows: Vec<(&str, &str, u64, ArmResult)> = Vec::new();
    for backend in &backends {
        let backend: &'static str = if backend == "gpu" { "gpu" } else { "cpu" };
        for &seed in &seeds {
            for fitness in ["metric", "pnl"] {
                let r = run_arm(&dir, &feat, &spec, &Arm { backend, fitness, seed }, generations, &windows);
                println!(
                    "{backend} {fitness} {seed} | {:.4} {:.4} {:.3} {} {:.3} {} | {:.1} {:.3} {:.1} {:.1} {:.3} | {:.4} {:.4} {} {:.3} {:.3} {:.3} {:.4}",
                    r.val_net, r.test_net, r.test_sharpe, r.flips, r.exposure, r.front, r.search_s, r.sec_per_gen,
                    r.eval_s, r.fit_s, r.device_frac, r.wf_net, r.wf_min_window, r.wf_trades, r.wf_exposure, r.wf_sharpe, r.wf_dsr, r.wf_se
                );
                rows.push((backend, fitness, seed, r));
            }
        }
    }

    // Pre-registered rule (#483): recommend pnl when the seed-mean walk-forward net
    // beats metric by more than one block-bootstrap standard error and the mean
    // trades are at most 1.5x. The standard error is the root of the two arms'
    // mean squared bootstrap errors of the pooled return.
    for backend in &backends {
        let mean = |fitness: &str, f: &dyn Fn(&ArmResult) -> f64| {
            let v: Vec<f64> = rows.iter().filter(|r| r.0 == backend && r.1 == fitness).map(|r| f(&r.3)).collect();
            v.iter().sum::<f64>() / v.len() as f64
        };
        let rms = |fitness: &str| {
            let v: Vec<f64> = rows.iter().filter(|r| r.0 == backend && r.1 == fitness).map(|r| r.3.wf_se * r.3.wf_se).collect();
            (v.iter().sum::<f64>() / v.len() as f64).sqrt()
        };
        let (m, p) = (mean("metric", &|r| r.wf_net), mean("pnl", &|r| r.wf_net));
        let se = (rms("metric").powi(2) + rms("pnl").powi(2)).sqrt();
        let (tm, tp) = (mean("metric", &|r| r.wf_trades), mean("pnl", &|r| r.wf_trades));
        let go = p - m > se && tp <= 1.5 * tm;
        println!(
            "decision {backend}: wf_net metric {m:.4} pnl {p:.4} diff {:.4} se {se:.4}; trades metric {tm:.0} pnl {tp:.0} ratio {:.2}; recommend {}",
            p - m,
            tp / tm,
            if go { "--fitness pnl" } else { "--fitness metric" }
        );
    }
}
