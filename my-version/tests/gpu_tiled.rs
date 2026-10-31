//! #293: tiled device residency. The feature columns stream from the host
//! through two device tiles; scores, searches and reports do not change.
//!
//! Device tests skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::Mutex;

use rcsr::data::Sample;
use rcsr::dispatch::{Route, Stage};
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::fit::FitStrategy;
use rcsr::genome::{GenomeMeta, GenomePool};
use rcsr::gpu::{compile_program, GpuEvaluator, Population, ScoreBatch};
use rcsr::memory::Residency;
use rcsr::metrics::{Metric, ScoreObjective, ScoreStatus};
use rcsr::search::{self, Config, Outcome};

/// Serialises the tests that read `RCSR_TEST_GPU_*` or run a whole search.
static ENV: Mutex<()> = Mutex::new(());

const CARRY_BYTES: u64 = 256 * 16 + 16;

const KINDS: [ScoreObjective; 15] = [
    ScoreObjective::Squared,
    ScoreObjective::Absolute,
    ScoreObjective::LogSquashed,
    ScoreObjective::Signed,
    ScoreObjective::Maximum,
    ScoreObjective::Correlation,
    ScoreObjective::CorrelationSigned,
    ScoreObjective::Aic,
    ScoreObjective::Aicc,
    ScoreObjective::Bic,
    ScoreObjective::LogLoss,
    ScoreObjective::Brier,
    ScoreObjective::HingeSigmoid,
    ScoreObjective::HingeOrdinary,
    ScoreObjective::HingeSquared,
];

fn has_device() -> bool {
    rcsr::gpu::probe().available()
}

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0 >> 11
    }
    fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64
    }
}

struct Data {
    n: usize,
    cols: Vec<f32>,
    y: Vec<f32>,
    sign: Vec<f64>,
}

/// Two columns; rows 10, 12, 266 and the last row carry `|y| = 7`, so the
/// maximum ties inside and across lanes.
fn data(n: usize) -> Data {
    let mut rng = Lcg(0x5eed_0293 ^ n as u64);
    let mut cols = vec![0.0f32; 2 * n];
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let a = rng.unit() * 4.0 - 2.0;
        let b = rng.unit() * 2.0 - 1.0;
        cols[i] = a as f32;
        cols[n + i] = b as f32;
        y[i] = (0.7 * a + 0.3 * (2.0 * b).sin() + 0.2 * (rng.unit() - 0.5)) as f32;
    }
    for (k, &row) in [10usize, 12, 266, n - 1].iter().enumerate() {
        y[row] = if k % 2 == 0 { 7.0 } else { -7.0 };
    }
    let sign = y.iter().map(|&v| if v > 0.1 { 1.0 } else { -1.0 }).collect();
    Data { n, cols, y, sign }
}

fn trees() -> Vec<Node> {
    let affine = node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(0.8), input(0)]), constant(0.15)],
    );
    let sine = node(
        Op::Add,
        vec![
            node(
                Op::Sin,
                vec![node(Op::Add, vec![node(Op::Mul, vec![constant(1.1), input(0)]), constant(0.1)])],
            ),
            node(Op::Mul, vec![constant(0.2), input(1)]),
        ],
    );
    vec![
        affine,
        sine,
        constant(0.0),
        constant(0.5),
        constant(f64::NAN),
        node(Op::Mul, vec![input(0), input(1)]),
        node(Op::Tanh, vec![node(Op::Mul, vec![constant(2.0), input(1)])]),
        node(Op::Sub, vec![node(Op::Square, vec![input(0)]), input(1)]),
    ]
}

/// `count` distinct programs built from the fixture trees.
fn many_trees(count: usize) -> Vec<Node> {
    let base = trees();
    (0..count)
        .map(|i| {
            node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![constant(0.001 * (i + 1) as f64), input(i % 2)]),
                    base[i % base.len()].clone(),
                ],
            )
        })
        .collect()
}

fn prepare(ev: &mut GpuEvaluator, d: &Data, weights: &[f64]) {
    assert!(ev.set_fit_sign(&d.sign));
    if !weights.is_empty() {
        assert!(ev.set_score_weights(weights));
    }
}

fn resident(d: &Data, weights: &[f64]) -> GpuEvaluator {
    let mut ev = GpuEvaluator::new(&d.cols, d.n, 2, &d.y).expect("resident evaluator");
    prepare(&mut ev, d, weights);
    ev
}

fn tiled(d: &Data, weights: &[f64], tile_rows: usize, partials_cap: Option<u64>) -> GpuEvaluator {
    let mut ev = GpuEvaluator::new_tiled(d.cols.clone(), d.n, 2, &d.y, tile_rows, None, partials_cap)
        .expect("tiled evaluator");
    assert_eq!(ev.residency(), Residency::Tiled);
    prepare(&mut ev, d, weights);
    ev
}

fn assert_same(label: &str, got: &ScoreBatch, want: &ScoreBatch) {
    assert!(got.ms >= 0.0, "{label}: launch failed");
    assert!(want.ms >= 0.0, "{label}: resident launch failed");
    assert_eq!(got.values.len(), want.values.len(), "{label}");
    for k in 0..want.values.len() {
        assert_eq!(got.status[k], want.status[k], "{label} slot {k} status");
        assert_eq!(
            got.values[k].to_bits(),
            want.values[k].to_bits(),
            "{label} slot {k}: {} vs {}",
            got.values[k],
            want.values[k]
        );
    }
}

fn pool_of(trees: &[Node]) -> GenomePool {
    let programs: Vec<_> = trees
        .iter()
        .map(|t| compile_program(t).expect("a native program"))
        .collect();
    let width = programs.iter().map(|p| p.words().len()).max().unwrap();
    let mut pool = GenomePool::new(trees.len(), width);
    pool.begin_generation();
    for p in &programs {
        assert!(pool.push(p.words(), GenomeMeta::default()));
    }
    pool
}

fn score_both(
    label: &str,
    res: &mut GpuEvaluator,
    til: &mut GpuEvaluator,
    trees: &[Node],
    kinds: &[ScoreObjective],
) {
    let pop = Population::from_trees(trees.iter());
    let (prog, offsets) = pop.programs();
    assert_eq!(pop.len(), trees.len());
    let pool = pool_of(trees);
    let pcount = vec![2i32; trees.len()];
    for &kind in kinds {
        let want = res.score_population(&pop, kind, &pcount);
        let got = til.score_population(&pop, kind, &pcount);
        assert_same(&format!("{label} population {}", kind.name()), &got, &want);
        let _ = (prog, offsets);
        let want = res.score_pool(&pool, kind, &pcount);
        let got = til.score_pool(&pool, kind, &pcount);
        assert_same(&format!("{label} pool {}", kind.name()), &got, &want);
    }
}

#[test]
fn tile_rows_do_not_change_scores() {
    if !has_device() {
        return;
    }
    let n = 70_001;
    let d = data(n);
    let zero: Vec<f64> = (0..n)
        .map(|i| if i % 3 == 0 { 0.0 } else { 1.0 + (i % 4) as f64 * 0.25 })
        .collect();
    let trees = trees();
    for (case, weights) in [("unweighted", Vec::new()), ("zero weights", zero)] {
        let mut res = resident(&d, &weights);
        for tile_rows in [512usize, 1_024, 8_192] {
            let mut til = tiled(&d, &weights, tile_rows, None);
            score_both(&format!("{case} T={tile_rows}"), &mut res, &mut til, &trees, &KINDS);
            let report = til.arena_report();
            assert_eq!(report.tile_rows, tile_rows as u64);
            assert!(report.tiles >= (n / tile_rows) as u64);
        }
    }
    let nan = score_nan_candidate(&d);
    assert!(nan, "the NaN candidate is unusable on both paths");
}

fn score_nan_candidate(d: &Data) -> bool {
    let mut res = resident(d, &[]);
    let mut til = tiled(d, &[], 512, None);
    let t = trees();
    let pop = Population::from_trees(t.iter());
    let pcount = vec![2i32; t.len()];
    let a = res.score_population(&pop, ScoreObjective::Squared, &pcount);
    let b = til.score_population(&pop, ScoreObjective::Squared, &pcount);
    a.status[4] == ScoreStatus::NonFinite && b.status[4] == ScoreStatus::NonFinite
}

#[test]
fn waves_do_not_change_scores_tiled() {
    if !has_device() {
        return;
    }
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    let n = 20_001;
    let d = data(n);
    let candidates = many_trees(1_000);
    let mut res = resident(&d, &[]);
    let mut til = tiled(&d, &[], 1_024, Some(64 * CARRY_BYTES));
    let kinds = [
        ScoreObjective::Squared,
        ScoreObjective::Correlation,
        ScoreObjective::Maximum,
        ScoreObjective::LogLoss,
    ];
    score_both("W=64", &mut res, &mut til, &candidates, &kinds);
    let report = til.arena_report();
    let per_batch = report.waves / (2 * kinds.len() as u64);
    assert_eq!(per_batch, 16, "1,000 candidates in waves of 64: {report:?}");

    // The same cap through the plan hook of the search.
    let train = sample(40_000, 3, 7);
    let test = sample(500, 3, 8);
    let mut cfg = search_config(Metric::Squared);
    cfg.islands = 4;
    cfg.pop = 250;
    cfg.generations = 3;
    let base = search::run(&cfg, &train, &test);
    std::env::set_var("RCSR_TEST_GPU_PARTIALS_CAP_BYTES", (64 * CARRY_BYTES).to_string());
    std::env::set_var("RCSR_TEST_GPU_TILE_ROWS", "1024");
    let plan = search::device_plan_for(&cfg, &train).expect("a plan");
    cfg.gpu_memory = Some(tiled_budget(&cfg, &train));
    let capped = search::run(&cfg, &train, &test);
    std::env::remove_var("RCSR_TEST_GPU_PARTIALS_CAP_BYTES");
    std::env::remove_var("RCSR_TEST_GPU_TILE_ROWS");
    assert_eq!(plan.mode, Residency::Resident);
    assert_eq!(capped.backend, rcsr::backend::Kind::Gpu, "{:?}", capped.backend_reason);
    let mp = capped.gpu.as_ref().and_then(|g| g.memory_plan.as_ref()).expect("plan");
    assert_eq!(mp.mode, Residency::Tiled);
    assert_eq!(mp.wave_candidates, 64);
    assert_eq!(mp.waves, 16);
    assert_eq!(archive_hash(&capped), archive_hash(&base));
}

#[test]
fn tile_staging_is_bounded() {
    if !has_device() {
        return;
    }
    let d = data(70_001);
    let mut til = tiled(&d, &[], 512, None);
    let candidates = many_trees(64);
    let pool = pool_of(&candidates);
    let pcount = vec![2i32; candidates.len()];
    let first = til.score_pool(&pool, ScoreObjective::Squared, &pcount);
    assert!(first.ms >= 0.0);
    let cold = til.arena_report();
    let cold_ledger = til.alloc_report();
    assert!(cold.tiles >= 64, "at least 64 tiles in a batch: {cold:?}");
    assert_eq!(cold.tile_pinned_bytes, 2 * 512 * 2 * 4);
    for _ in 0..100 {
        let batch = til.score_pool(&pool, ScoreObjective::Squared, &pcount);
        assert!(batch.ms >= 0.0);
    }
    let warm = til.arena_report();
    let ledger = til.alloc_report();
    assert_eq!(warm.tile_pinned_bytes, cold.tile_pinned_bytes);
    assert_eq!(warm.steady_alloc_calls, 0, "{warm:?}");
    assert_eq!(ledger.requests, cold_ledger.requests, "no new device request");
    assert_eq!(ledger.live_bytes, cold_ledger.live_bytes);
    assert!(warm.tile_overlapped_uploads > 0, "{warm:?}");
    assert_eq!(warm.tile_uploads, warm.tiles);
}

fn sample(n: usize, vars: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let mut row = vec![t];
        let mut acc = 0.0;
        for k in 1..vars {
            let s = ((i as u32 + k as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
            row.push(s * 2.0 - 1.0);
            acc += s * (k as f64) * 0.1;
        }
        y.push(1.7 * t.sin() + 0.4 * t + acc - 0.2);
        x.push(row);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn search_config(metric: Metric) -> Config {
    Config {
        generations: 8,
        islands: 4,
        pop: 64,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed: 17,
        fit_strategy: FitStrategy::Exact,
        metric,
        gpu: true,
        gpu_notes: false,
        ..Default::default()
    }
}

/// A `--gpu-memory` value one byte under the resident plan, which makes the
/// plan tiled.
fn tiled_budget(cfg: &Config, train: &Sample) -> u64 {
    let mut open = cfg.clone();
    open.gpu_memory = None;
    let plan = search::device_plan_for(&open, train).expect("a plan");
    assert_eq!(plan.mode, Residency::Resident);
    plan.required - 1
}

fn archive_hash(out: &Outcome) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    let names: Vec<String> = (0..8).map(|i| format!("x{i}")).collect();
    for s in &out.archive.items {
        eat(&s.error.to_bits().to_le_bytes());
        eat(&(s.complexity as u64).to_le_bytes());
        eat(rcsr::expr::to_string(&s.tree, &names).as_bytes());
    }
    for e in &out.test_errors {
        eat(&e.to_bits().to_le_bytes());
    }
    h
}

fn eval_route(out: &Outcome) -> (Route, String, u64) {
    let e = out
        .dispatch
        .ledger
        .entry(Stage::Eval)
        .expect("an eval ledger entry");
    (e.route, e.reason.name().to_string(), e.candidates)
}

fn compare_search(tile_rows: Option<&str>) {
    let train = sample(60_000, 6, 5);
    let test = sample(4_000, 6, 6);
    let mut cfg = search_config(Metric::Squared);
    let want = search::run(&cfg, &train, &test);
    let want_plan = want.gpu.as_ref().and_then(|g| g.memory_plan.as_ref()).expect("resident plan");
    assert_eq!(want_plan.mode, Residency::Resident);

    cfg.gpu_memory = Some(tiled_budget(&cfg, &train));
    match tile_rows {
        Some(t) => std::env::set_var("RCSR_TEST_GPU_TILE_ROWS", t),
        None => std::env::remove_var("RCSR_TEST_GPU_TILE_ROWS"),
    }
    let got = search::run(&cfg, &train, &test);
    std::env::remove_var("RCSR_TEST_GPU_TILE_ROWS");
    let plan = got.gpu.as_ref().and_then(|g| g.memory_plan.as_ref()).expect("tiled plan");
    assert_eq!(plan.mode, Residency::Tiled, "{plan:?}");
    if let Some(t) = tile_rows {
        assert_eq!(plan.tile_rows, t.parse::<usize>().unwrap());
    }
    assert_eq!(got.backend, want.backend);
    assert_eq!(got.backend, rcsr::backend::Kind::Gpu);
    assert_eq!(archive_hash(&got), archive_hash(&want));
    let (route, reason, _) = eval_route(&got);
    let (want_route, want_reason, _) = eval_route(&want);
    assert_eq!(route, Route::Device, "tiled eval route: {reason}");
    assert_eq!((route, reason), (want_route, want_reason));
}

#[test]
fn tiled_search_matches_resident() {
    if !has_device() {
        return;
    }
    let _guard = ENV.lock().unwrap_or_else(|e| e.into_inner());
    compare_search(None);
    compare_search(Some("512"));
}

fn tmp_csv(rows: usize) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-gpu-tiled-{}-{}", std::process::id(), rows));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("sine.csv");
    let mut text = String::from("x,y\n");
    for i in 0..rows {
        let x = i as f64 * 0.001;
        text.push_str(&format!("{:.6},{:.9}\n", x, x.sin()));
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn rcsr(csv: &PathBuf, extra: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", csv.to_str().unwrap(), "--target", "y", "--vars", "x"])
        .args(["--generations", "2", "--seed", "1"])
        .args(extra)
        .output()
        .expect("run rcsr")
}

fn number_after(text: &str, key: &str) -> u64 {
    let at = text.find(key).unwrap_or_else(|| panic!("no {key} in {text}"));
    text[at + key.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap()
}

/// Bytes of the label sign and weight vectors a native objective charges and
/// the exact statistic scorers do not: 12 B per training row.
const CLI_TRAIN_ROWS: u64 = 75_000;

/// `--gpu-memory` one byte under the resident rows of the run, as a number.
fn cli_tiled_budget_bytes(csv: &PathBuf) -> u64 {
    let out = rcsr(csv, &["--gpu", "--json", "--quiet"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("\"memory_plan\":{\"mode\":\"resident\""), "{stdout}");
    let plan = &stdout[stdout.find("\"memory_plan\"").unwrap()..];
    number_after(plan, "\"required_bytes\":") - 1
}

fn cli_tiled_budget(csv: &PathBuf) -> String {
    cli_tiled_budget_bytes(csv).to_string()
}

#[test]
fn cli_tiled_run_and_explicit_device_fit() {
    if !has_device() {
        return;
    }
    let csv = tmp_csv(100_000);
    let budget = cli_tiled_budget(&csv);

    let out = rcsr(&csv, &["--gpu", "--json", "--quiet", "--gpu-memory", budget.as_str()]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("\"memory_plan\":{\"mode\":\"tiled\""), "{stdout}");
    assert!(stdout.contains("\"backend\":\"gpu\""), "{stdout}");
    let plan = &stdout[stdout.find("\"memory_plan\"").unwrap()..];
    assert!(plan.contains("\"fit_reason\":\"device_memory\""), "{plan}");
    assert!(plan.contains("\"refusal\":null"), "{plan}");
    assert!(number_after(plan, "\"tile_rows\":") > 0);

    let out = rcsr(
        &csv,
        &["--gpu", "--fit-strategy", "gpu", "--gpu-memory", budget.as_str()],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("refused the fit stage"), "{stderr}");
    assert!(stderr.contains("tiled residency has no device fit"), "{stderr}");
}

#[test]
fn cli_tiled_rank_metric_runs_on_the_cpu_with_device_memory() {
    if !has_device() {
        return;
    }
    let csv = tmp_csv(100_000);
    // The exact statistic scorers carry no label sign, so their resident rows
    // are 12 B per training row smaller than the native objective's.
    let budget = (cli_tiled_budget_bytes(&csv) - 12 * CLI_TRAIN_ROWS).to_string();
    let out = rcsr(
        &csv,
        &[
            "--gpu",
            "--metric",
            "rank_correlation",
            "--json",
            "--quiet",
            "--gpu-memory",
            budget.as_str(),
        ],
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("\"backend\":\"cpu\""), "{stdout}");
    assert!(stdout.contains("\"backend_reason\":\"device_memory"), "{stdout}");
    assert!(
        stdout.contains("\"backend_reason\":\"device_memory: the exact order/rank scorer"),
        "{stdout}"
    );
    assert!(stdout.contains("tiles of"), "{stdout}");
}

/// Paired timing for the PR: 1,000,000 rows x 20 vars, 400 candidates,
/// resident against tiled with T = 8,192. Run with `--ignored --nocapture`.
#[test]
#[ignore]
fn tiled_timing_record() {
    if !has_device() {
        return;
    }
    let n = 1_000_000usize;
    let v = 20usize;
    let mut rng = Lcg(0x71ed);
    let mut cols = vec![0.0f32; n * v];
    for c in cols.iter_mut() {
        *c = (rng.unit() * 4.0 - 2.0) as f32;
    }
    let y: Vec<f32> = (0..n)
        .map(|i| 0.7 * cols[i] + 0.3 * cols[n + i].sin() + 0.1 * cols[2 * n + i])
        .collect();
    let candidates: Vec<Node> = (0..400)
        .map(|i| {
            node(
                Op::Add,
                vec![
                    node(Op::Mul, vec![constant(0.01 * (i + 1) as f64), input(i % v)]),
                    node(Op::Sin, vec![input((i * 7 + 3) % v)]),
                ],
            )
        })
        .collect();
    let pool = pool_of(&candidates);
    let pcount = vec![2i32; candidates.len()];
    let mut res = GpuEvaluator::new(&cols, n, v, &y).expect("resident");
    let mut til = GpuEvaluator::new_tiled(cols.clone(), n, v, &y, 8_192, None, None).expect("tiled");
    let kind = ScoreObjective::Squared;
    let run = |ev: &mut GpuEvaluator| {
        let t = std::time::Instant::now();
        let b = ev.score_pool(&pool, kind, &pcount);
        assert!(b.ms >= 0.0);
        (t.elapsed().as_secs_f64(), b)
    };
    let (_, a) = run(&mut res);
    let (_, b) = run(&mut til);
    assert_same("timing", &b, &a);
    let (mut r, mut t) = (Vec::new(), Vec::new());
    for _ in 0..7 {
        r.push(run(&mut res).0);
        t.push(run(&mut til).0);
    }
    r.sort_by(|a, b| a.partial_cmp(b).unwrap());
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let report = til.arena_report();
    eprintln!(
        "timing 1,000,000 x 20, P = 400: resident median {:.4} s (min {:.4}), tiled T=8192 median {:.4} s (min {:.4}), ratio {:.2}; tiles {}, uploads {}, overlapped {}",
        r[3], r[0], t[3], t[0], t[3] / r[3], report.tiles, report.tile_uploads, report.tile_overlapped_uploads
    );
}
