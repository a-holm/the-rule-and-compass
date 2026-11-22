//! `cpu_bench` — measure the CPU evaluator, the constant fitter, the metric fit arms and thread scaling.
//!
//! Compares, on the same deterministic data and trees:
//!
//! 1. the existing scalar tree walk ([`rcsr::expr::eval`] per row, row-major
//!    storage, which is what the search uses today);
//! 2. the SoA evaluator's portable scalar loop;
//! 3. the SoA evaluator's runtime-selected AVX kernel (same code as the scalar
//!    loop when the CPU has no AVX).
//!
//! Every result is also checked against the scalar walk, and the maximum
//! relative difference is printed. Times are microseconds per full evaluation
//! of the dataset, best of five runs, with the repetition count chosen to
//! keep each measurement around 50 ms. The `fit`, `metric` and `scale` modes
//! are described in `--help` and `docs/optimisation.md`.
//!
//! ```text
//! cargo run --release --bin cpu_bench
//! cargo run --release --bin cpu_bench -- --mode all --tsv
//! cargo run --release --bin cpu_bench -- --help
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use rcsr::eval_soa;
use rcsr::expr::{self, Node, Op};
use rcsr::fit;
use rcsr::metrics::{self, Metric};

const NVARS: usize = 5;

fn x(i: usize) -> Node {
    expr::input(i)
}

fn c(v: f64) -> Node {
    expr::constant(v)
}

fn unary(op: Op, kid: Node) -> Node {
    expr::node(op, vec![kid])
}

fn binary(op: Op, a: Node, b: Node) -> Node {
    expr::node(op, vec![a, b])
}

/// Four deterministic trees with 5, 9, 15 and 25 nodes, mixing arithmetic,
/// transcendentals, division and the guarded operations.
fn trees() -> Vec<(&'static str, Node)> {
    // 5 nodes: 1.7*x0 + x1
    let five = binary(Op::Add, binary(Op::Mul, x(0), c(1.7)), x(1));

    // 9 nodes: sin(x0*x1 + 0.3) + cos(x2)
    let nine = binary(
        Op::Add,
        unary(
            Op::Sin,
            binary(Op::Add, binary(Op::Mul, x(0), x(1)), c(0.3)),
        ),
        unary(Op::Cos, x(2)),
    );

    // 15 nodes: sin(x0*x1 + sqrt(abs(x2))) / (x1^2 + cos(x0))
    let fifteen = binary(
        Op::Div,
        unary(
            Op::Sin,
            binary(
                Op::Add,
                binary(Op::Mul, x(0), x(1)),
                unary(Op::Sqrt, unary(Op::Abs, x(2))),
            ),
        ),
        binary(
            Op::Add,
            binary(Op::Pow, x(1), c(2.0)),
            unary(Op::Cos, x(0)),
        ),
    );

    // 25 nodes: the 15-node tree plus tanh(logistic(-x3)) * gauss(x4 + 0.7)
    let twenty_five = binary(
        Op::Add,
        fifteen.clone(),
        binary(
            Op::Mul,
            unary(Op::Tanh, unary(Op::Logistic, unary(Op::Neg, x(3)))),
            unary(Op::Gauss, binary(Op::Add, x(4), c(0.7))),
        ),
    );

    vec![
        ("lin5", five),
        ("trig9", nine),
        ("mix15", fifteen),
        ("mix25", twenty_five),
        ("arith15", named_tree(ARITH15)),
        ("trans13", named_tree(TRANS13)),
    ]
}

/// Build both layouts from the same values: SoA columns for the new evaluator
/// and row vectors for the existing walk.
fn dataset(rows: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
    let mut cols = vec![vec![0.0; rows]; NVARS];
    for (v, col) in cols.iter_mut().enumerate() {
        for (r, cell) in col.iter_mut().enumerate() {
            // Cheap deterministic spread in [-2, 2).
            let h = (r as u64)
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add((v as u64 + 1).wrapping_mul(1_442_695_040_888_963_407));
            let u = ((h >> 33) as f64) / ((1u64 << 31) as f64);
            *cell = u * 4.0 - 2.0;
        }
    }
    let mut aos = vec![vec![0.0; NVARS]; rows];
    for (r, row) in aos.iter_mut().enumerate() {
        for (v, cell) in row.iter_mut().enumerate() {
            *cell = cols[v][r];
        }
    }
    (cols, aos)
}

/// Time one full evaluation, best of five, with an adaptive repetition count.
fn measure<F: FnMut()>(target: Duration, mut f: F) -> Duration {
    f(); // warm up
    let start = Instant::now();
    f();
    let one = start.elapsed();
    let reps = (target.as_nanos() / one.as_nanos().max(1)).clamp(1, 50_000) as u32;
    let mut best = Duration::MAX;
    for _ in 0..5 {
        let start = Instant::now();
        for _ in 0..reps {
            f();
        }
        best = best.min(start.elapsed() / reps);
    }
    best
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Relative difference used for the agreement check. Both `NaN` counts as
/// agreement; one `NaN` and a number does not. Equal infinities agree.
fn rel_diff(a: f64, b: f64) -> f64 {
    if a == b {
        return 0.0;
    }
    if a.is_nan() && b.is_nan() {
        return 0.0;
    }
    if a.is_nan() || b.is_nan() {
        return f64::INFINITY;
    }
    (a - b).abs() / (1.0 + a.abs().max(b.abs()))
}

const LEGACY_ROWS: &[usize] = &[1_000, 20_000, 200_000];
const NEW_ROWS: &[usize] = &[20_000, 150_000, 1_000_000];
const METRIC_ROWS: &[usize] = &[20_000, 150_000];
const DEFAULT_THREADS: &[usize] = &[1, 2, 4, 8];
const DEFAULT_MIN_SECONDS: f64 = 0.3;
const DEFAULT_FIT_ITERS: usize = 8;
const FIT_VARS: usize = 10;
const SEED: u64 = 0x9E37_79B9_7F4A_7C15;

const HELP: &str = "cpu_bench — CPU evaluator, constant fitter, metric fit and thread-scaling benchmark

USAGE:
  cpu_bench [--mode eval|fit|metric|scale|all] [--rows LIST] [--threads LIST]
            [--min-seconds S] [--fit-iters N] [--fit-irls MODE] [--tsv] [--help]

MODES:
  eval    scalar walk vs SoA scalar vs SoA AVX on the 5/9/15/25-node trees
          (default; rows 1000,20000,200000)
  fit     fit_weighted on arith15 and trans13: one-pass Mrow/s, fit wall,
          constants hash (rows 20000,150000,1000000)
  metric  fit_weighted_metric for squared, absolute, median, interquartile and
          rank_correlation on arith15, plus score_weighted time for median and
          squared (rows 20000,150000)
  scale   T scoped threads share the data; aggregate fits/s and Mrow/s and the
          speed-up over T = 1 (rows 20000,150000,1000000)
  all     eval, fit, metric and scale in turn

OPTIONS:
  --rows LIST        comma-separated row counts, overrides every mode's default
  --threads LIST     thread counts for scale (default 1,2,4,8), capped at
                     available_parallelism
  --min-seconds S    minimum time per cell (default 0.3 for fit, metric, scale;
                     eval keeps its 50 ms repetition batches unless set)
  --fit-iters N      iterations of the metric fits (default 8)
  --fit-irls MODE    schedule of the absolute fit in metric mode: nested (default),
                     reweighted or reweighted-final (#611)
  --tsv              tab-separated cells with one header row per table
  --help             this text

Fit cells report an FNV-1a 64 hash over the fitted constants' bits and the
returned score bits. The hash is the bit-identity check for fitter changes.";

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Eval,
    Fit,
    Metric,
    Scale,
    All,
}

struct Opts {
    mode: Mode,
    rows: Option<Vec<usize>>,
    threads: Vec<usize>,
    min_seconds: Option<f64>,
    fit_iters: usize,
    fit_irls: fit::FitIrls,
    tsv: bool,
}

fn usage_error(msg: &str) -> ! {
    eprintln!("cpu_bench: {msg} (try --help)");
    std::process::exit(2);
}

fn parse_list(flag: &str, text: &str) -> Vec<usize> {
    let mut out = Vec::new();
    for part in text.split(',') {
        match part.trim().parse::<usize>() {
            Ok(v) if v > 0 => out.push(v),
            _ => usage_error(&format!("{flag} needs positive integers, got '{part}'")),
        }
    }
    if out.is_empty() {
        usage_error(&format!("{flag} needs at least one value"));
    }
    out
}

fn parse_args(args: &[String]) -> Opts {
    let mut o = Opts {
        mode: Mode::Eval,
        rows: None,
        threads: DEFAULT_THREADS.to_vec(),
        min_seconds: None,
        fit_iters: DEFAULT_FIT_ITERS,
        fit_irls: fit::FitIrls::Nested,
        tsv: false,
    };
    let mut i = 0;
    while i < args.len() {
        let flag = args[i].as_str();
        let mut value = || -> &str {
            i += 1;
            match args.get(i) {
                Some(v) => v.as_str(),
                None => usage_error(&format!("{flag} needs a value")),
            }
        };
        match flag {
            "--mode" => {
                o.mode = match value() {
                    "eval" => Mode::Eval,
                    "fit" => Mode::Fit,
                    "metric" => Mode::Metric,
                    "scale" => Mode::Scale,
                    "all" => Mode::All,
                    other => usage_error(&format!("unknown mode '{other}'")),
                }
            }
            "--rows" => o.rows = Some(parse_list("--rows", value())),
            "--threads" => o.threads = parse_list("--threads", value()),
            "--min-seconds" => {
                o.min_seconds = match value().parse::<f64>() {
                    Ok(v) if v >= 0.0 && v.is_finite() => Some(v),
                    _ => usage_error("--min-seconds needs a non-negative number"),
                }
            }
            "--fit-iters" => {
                o.fit_iters = match value().parse::<usize>() {
                    Ok(v) => v,
                    Err(_) => usage_error("--fit-iters needs a non-negative integer"),
                }
            }
            "--fit-irls" => {
                o.fit_irls = match fit::FitIrls::parse(value()) {
                    Some(s) => s,
                    None => usage_error("--fit-irls needs nested, reweighted or reweighted-final"),
                }
            }
            "--tsv" => o.tsv = true,
            bad => usage_error(&format!("unknown argument '{bad}'")),
        }
        i += 1;
    }
    o
}

/// Table printer: aligned text, or tab-separated with `--tsv`.
struct Out {
    tsv: bool,
    widths: Vec<usize>,
}

impl Out {
    fn section(&self, title: &str) {
        println!();
        println!("# {title}");
    }

    fn header(&mut self, cols: &[&str]) {
        self.widths = cols
            .iter()
            .enumerate()
            .map(|(i, c)| c.len().max(if i == 0 { 16 } else { 10 }))
            .collect();
        self.line(&cols.iter().map(|c| c.to_string()).collect::<Vec<_>>());
        if !self.tsv {
            println!("{}", "-".repeat(self.widths.iter().map(|w| w + 1).sum::<usize>()));
        }
    }

    fn line(&self, cells: &[String]) {
        if self.tsv {
            println!("{}", cells.join("\t"));
        } else {
            let mut s = String::new();
            for (c, w) in cells.iter().zip(&self.widths) {
                s.push_str(&format!("{c:>w$} "));
            }
            println!("{}", s.trim_end());
        }
    }
}

/// Smallest time of repeated runs: at least `min_runs`, then until
/// `min_seconds` of timed work has accumulated. One untimed warm-up first.
fn best_of<F: FnMut()>(mut f: F, min_seconds: f64, min_runs: usize, warm: bool) -> f64 {
    if warm {
        f();
    }
    let start = Instant::now();
    let mut best = f64::INFINITY;
    let mut runs = 0;
    while runs < min_runs || (start.elapsed().as_secs_f64() < min_seconds && runs < 200) {
        let t = Instant::now();
        f();
        best = best.min(t.elapsed().as_secs_f64());
        runs += 1;
    }
    best
}

/// Deterministic fit fixture: 10 variables uniform in [-2, 2),
/// `y = 1.5 sin(x0 x1) + 0.7 x2^2 - 0.3 exp(0.2 x3) + noise`, noise in +-0.05.
struct FitData {
    x: Vec<Vec<f64>>,
    cols: Vec<Vec<f64>>,
    y: Vec<f64>,
}

fn fit_data(rows: usize) -> FitData {
    let mut s = SEED;
    let mut next = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        ((s >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(rows);
    let mut y = Vec::with_capacity(rows);
    for _ in 0..rows {
        let r: Vec<f64> = (0..FIT_VARS).map(|_| next() * 4.0 - 2.0).collect();
        let noise = (next() - 0.5) * 0.1;
        y.push(
            1.5 * (r[0] * r[1]).sin() + 0.7 * r[2] * r[2] - 0.3 * (0.2 * r[3]).exp() + noise,
        );
        x.push(r);
    }
    let cols = eval_soa::columns(&x);
    FitData { x, cols, y }
}

fn named_tree(src: &str) -> Node {
    let vars: Vec<String> = (0..FIT_VARS).map(|i| format!("x{i}")).collect();
    match rcsr::parse::parse(src, &vars) {
        Ok(t) => t,
        Err(_) => {
            eprintln!("cpu_bench: fixed tree '{src}' did not parse");
            std::process::exit(1);
        }
    }
}

const ARITH15: &str = "(1.1*x0 + 0.4*x1*x2) / (1.3 + x3*x3) - 0.2*x4";
const TRANS13: &str = "1.5*sin(x0*x1) + 0.7*x2*x2 - 0.3*exp(0.2*x3)";

fn constants(n: &Node, out: &mut Vec<f64>) {
    if n.op == Op::Const {
        out.push(n.val);
    }
    for k in &n.kids {
        constants(k, out);
    }
}

/// FNV-1a 64 over the fitted constants' bits then the returned score's bits.
fn fit_hash(tree: &Node, score: f64) -> u64 {
    let mut c = Vec::new();
    constants(tree, &mut c);
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for v in c.iter().copied().chain(std::iter::once(score)) {
        for b in v.to_bits().to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn mrows(rows: usize, secs: f64) -> f64 {
    rows as f64 / secs.max(f64::MIN_POSITIVE) / 1e6
}

fn bits_equal(a: &[f64], b: &[f64]) -> bool {
    a.iter()
        .zip(b)
        .all(|(p, q)| p.to_bits() == q.to_bits() || (p.is_nan() && q.is_nan()))
}

fn run_eval(o: &Opts, out: &mut Out) {
    let rows_list: Vec<usize> = o.rows.clone().unwrap_or_else(|| LEGACY_ROWS.to_vec());
    let target = o
        .min_seconds
        .map(Duration::from_secs_f64)
        .unwrap_or(Duration::from_millis(50));
    let trees = trees();
    let (mut d_scalar, mut d_avx, mut d_tile) = (0.0f64, 0.0f64, 0.0f64);
    let mut all_equal = true;

    if !o.tsv {
        out.section("eval: scalar tree walk vs structure-of-arrays CPU evaluators");
        println!(
            "AVX kernels: {} (4-row width {}, tile {} rows)",
            if eval_soa::avx_available() {
                "available"
            } else {
                "not available, scalar SoA used"
            },
            eval_soa::SIMD_WIDTH,
            eval_soa::TILE
        );
        println!();
        println!(
            "{:>8} {:>5} {:>8} {:>11} {:>11} {:>11} {:>11} {:>7} {:>7} {:>7} {:>8}",
            "tree", "nodes", "rows", "walk us", "soa us", "avx4 us", "tile us", "soa x", "avx4 x", "tile x",
            "tile/avx4"
        );
        println!("{}", "-".repeat(109));
    } else {
        out.section("eval");
        out.header(&[
            "tree",
            "tree_nodes",
            "rows",
            "walk_us",
            "soa_scalar_us",
            "soa_avx_us",
            "tile_us",
            "walk_mrows",
            "soa_scalar_mrows",
            "soa_avx_mrows",
            "tile_mrows",
            "tile_over_avx",
            "bitwise_equal",
        ]);
    }

    for (name, tree) in &trees {
        let program = match eval_soa::compile(tree) {
            Some(p) => p,
            None => {
                eprintln!("cpu_bench: a benchmark tree did not compile");
                std::process::exit(1);
            }
        };
        let nodes = expr::complexity(tree);
        for &rows in &rows_list {
            let (cols, aos) = dataset(rows);
            let mut walk_out = vec![0.0; rows];
            let mut soa_out = vec![0.0; rows];
            let mut avx_out = vec![0.0; rows];
            let mut tile_out = vec![0.0; rows];
            let mut scratch = eval_soa::Scratch::new();

            let walk = measure(target, || {
                for (r, row) in aos.iter().enumerate() {
                    walk_out[r] = expr::eval(tree, row);
                }
                black_box(&walk_out);
            });
            let soa = measure(target, || {
                eval_soa::eval_batch_scalar(&program, &cols, rows, &mut soa_out);
                black_box(&soa_out);
            });
            let avx = measure(target, || {
                eval_soa::eval_batch_rows4(&program, &cols, rows, &mut avx_out);
                black_box(&avx_out);
            });
            let tile = measure(target, || {
                eval_soa::eval_batch_into(&program, &cols, rows, &mut tile_out, &mut scratch);
                black_box(&tile_out);
            });

            for (r, row) in aos.iter().enumerate() {
                walk_out[r] = expr::eval(tree, row);
            }
            eval_soa::eval_batch_scalar(&program, &cols, rows, &mut soa_out);
            eval_soa::eval_batch_rows4(&program, &cols, rows, &mut avx_out);
            eval_soa::eval_batch_into(&program, &cols, rows, &mut tile_out, &mut scratch);
            for r in 0..rows {
                d_scalar = d_scalar.max(rel_diff(walk_out[r], soa_out[r]));
                d_avx = d_avx.max(rel_diff(walk_out[r], avx_out[r]));
                d_tile = d_tile.max(rel_diff(walk_out[r], tile_out[r]));
            }
            let equal = bits_equal(&walk_out, &soa_out)
                && bits_equal(&walk_out, &avx_out)
                && bits_equal(&walk_out, &tile_out);
            all_equal &= equal;

            let (walk_us, soa_us, avx_us, tile_us) = (us(walk), us(soa), us(avx), us(tile));
            let tile_over_avx = avx_us / tile_us.max(f64::MIN_POSITIVE);
            if o.tsv {
                out.line(&[
                    name.to_string(),
                    nodes.to_string(),
                    rows.to_string(),
                    format!("{walk_us:.2}"),
                    format!("{soa_us:.2}"),
                    format!("{avx_us:.2}"),
                    format!("{tile_us:.2}"),
                    format!("{:.2}", mrows(rows, walk.as_secs_f64())),
                    format!("{:.2}", mrows(rows, soa.as_secs_f64())),
                    format!("{:.2}", mrows(rows, avx.as_secs_f64())),
                    format!("{:.2}", mrows(rows, tile.as_secs_f64())),
                    format!("{tile_over_avx:.2}"),
                    equal.to_string(),
                ]);
            } else {
                println!(
                    "{:>8} {:>5} {:>8} {:>11.2} {:>11.2} {:>11.2} {:>11.2} {:>7.2} {:>7.2} {:>7.2} {:>8.2}",
                    name,
                    nodes,
                    rows,
                    walk_us,
                    soa_us,
                    avx_us,
                    tile_us,
                    walk_us / soa_us.max(f64::MIN_POSITIVE),
                    walk_us / avx_us.max(f64::MIN_POSITIVE),
                    walk_us / tile_us.max(f64::MIN_POSITIVE),
                    tile_over_avx
                );
            }
        }
    }

    if !o.tsv {
        println!();
        println!("max relative difference vs the scalar walk:");
        println!("  SoA scalar : {d_scalar:.3e}");
        println!("  SoA AVX 4  : {d_avx:.3e}");
        println!("  SoA tile   : {d_tile:.3e}");
        println!("all outputs bit-identical to the walk: {all_equal}");
        println!();
        println!("Times are one full evaluation of the dataset; lower is better.");
    }
}

fn run_fit(o: &Opts, out: &mut Out) {
    let rows_list = o.rows.clone().unwrap_or_else(|| NEW_ROWS.to_vec());
    let min_s = o.min_seconds.unwrap_or(DEFAULT_MIN_SECONDS);
    out.section("fit: fit_weighted on the fixed trees (single thread)");
    out.header(&[
        "tree",
        "rows",
        "consts",
        "pass_mrows",
        "fit_iters",
        "fit_s",
        "fits_per_s",
        "hash",
    ]);
    for (name, src) in [("arith15", ARITH15), ("trans13", TRANS13)] {
        let tree = named_tree(src);
        for &rows in &rows_list {
            let d = fit_data(rows);
            let pass = best_of(
                || {
                    let mut t = tree.clone();
                    black_box(fit::fit_weighted(&mut t, &d.x, &d.y, &[], 0));
                },
                min_s,
                3,
                true,
            );
            let mut hashed = tree.clone();
            let score = fit::fit_weighted(&mut hashed, &d.x, &d.y, &[], DEFAULT_FIT_ITERS);
            let hash = fit_hash(&hashed, score);
            let wall = best_of(
                || {
                    let mut t = tree.clone();
                    black_box(fit::fit_weighted(&mut t, &d.x, &d.y, &[], DEFAULT_FIT_ITERS));
                },
                min_s,
                3,
                false,
            );
            out.line(&[
                name.to_string(),
                rows.to_string(),
                expr::count_constants(&tree).to_string(),
                format!("{:.2}", mrows(rows, pass)),
                DEFAULT_FIT_ITERS.to_string(),
                format!("{wall:.4}"),
                format!("{:.3}", 1.0 / wall.max(f64::MIN_POSITIVE)),
                format!("{hash:016x}"),
            ]);
        }
    }
}

fn run_metric(o: &Opts, out: &mut Out) {
    let rows_list = o.rows.clone().unwrap_or_else(|| METRIC_ROWS.to_vec());
    let min_s = o.min_seconds.unwrap_or(DEFAULT_MIN_SECONDS);
    let metrics_list = [
        ("squared", Metric::Squared),
        ("absolute", Metric::Absolute),
        ("median", Metric::Median),
        ("interquartile", Metric::Interquartile),
        ("rank_correlation", Metric::RankCorrelation),
    ];
    let tree = named_tree(ARITH15);
    out.section("metric: fit_weighted_metric on arith15 (single thread)");
    out.header(&["metric", "rows", "fit_iters", "fit_s", "hash"]);
    for &rows in &rows_list {
        let d = fit_data(rows);
        for (name, m) in metrics_list {
            let mut hashed = tree.clone();
            let score = fit::with_fit_irls(o.fit_irls, || {
                fit::fit_weighted_metric(&mut hashed, &d.x, &d.y, &[], o.fit_iters, m)
            });
            let hash = fit_hash(&hashed, score);
            let wall = best_of(
                || {
                    let mut t = tree.clone();
                    black_box(fit::with_fit_irls(o.fit_irls, || {
                        fit::fit_weighted_metric(&mut t, &d.x, &d.y, &[], o.fit_iters, m)
                    }));
                },
                min_s,
                3,
                false,
            );
            out.line(&[
                name.to_string(),
                rows.to_string(),
                o.fit_iters.to_string(),
                format!("{wall:.4}"),
                format!("{hash:016x}"),
            ]);
        }
    }
    out.section("metric: score_weighted on a fixed prediction vector");
    out.header(&["scorer", "rows", "score_ms"]);
    for &rows in &rows_list {
        let d = fit_data(rows);
        let program = match eval_soa::compile(&tree) {
            Some(p) => p,
            None => {
                eprintln!("cpu_bench: a benchmark tree did not compile");
                std::process::exit(1);
            }
        };
        let mut pred = vec![0.0; rows];
        eval_soa::eval_batch(&program, &d.cols, rows, &mut pred);
        for (name, m) in [("median", Metric::Median), ("squared", Metric::Squared)] {
            let t = best_of(
                || {
                    black_box(metrics::score_weighted(&pred, &d.y, &[], m, 4));
                },
                min_s,
                3,
                true,
            );
            out.line(&[name.to_string(), rows.to_string(), format!("{:.4}", t * 1e3)]);
        }
    }
}

fn run_scale(o: &Opts, out: &mut Out, cpus: usize) {
    let rows_list = o.rows.clone().unwrap_or_else(|| NEW_ROWS.to_vec());
    let min_s = o.min_seconds.unwrap_or(DEFAULT_MIN_SECONDS);
    let fit_tree = named_tree(ARITH15);
    let eval_tree = named_tree(TRANS13);
    let program = match eval_soa::compile(&eval_tree) {
        Some(p) => p,
        None => {
            eprintln!("cpu_bench: a benchmark tree did not compile");
            std::process::exit(1);
        }
    };
    let mut threads: Vec<usize> = Vec::new();
    for &t in &o.threads {
        if t > cpus {
            eprintln!("cpu_bench: skipping T={t}, available_parallelism is {cpus}");
        } else {
            threads.push(t);
        }
    }
    out.section("scale: T scoped threads share the data; arith15 fits and trans13 evaluations");
    out.header(&[
        "rows",
        "threads",
        "fit_jobs",
        "fits_per_s",
        "fit_speedup",
        "eval_jobs",
        "eval_mrows",
        "eval_speedup",
    ]);
    for &rows in &rows_list {
        let d = fit_data(rows);
        let fit_jobs = (2_000_000 / rows).max(8);
        let eval_jobs = (50_000_000 / rows).max(8);
        let mut base: Option<(f64, f64)> = None;
        for &t in &threads {
            let fit_per = fit_jobs.div_ceil(t);
            let fit_wall = best_of(
                || {
                    std::thread::scope(|s| {
                        for _ in 0..t {
                            s.spawn(|| {
                                for _ in 0..fit_per {
                                    let mut tree = fit_tree.clone();
                                    black_box(fit::fit_weighted(
                                        &mut tree,
                                        &d.x,
                                        &d.y,
                                        &[],
                                        DEFAULT_FIT_ITERS,
                                    ));
                                }
                            });
                        }
                    });
                },
                min_s,
                1,
                false,
            );
            let eval_per = eval_jobs.div_ceil(t);
            let eval_wall = best_of(
                || {
                    std::thread::scope(|s| {
                        for _ in 0..t {
                            s.spawn(|| {
                                let mut o = vec![0.0; rows];
                                for _ in 0..eval_per {
                                    eval_soa::eval_batch(&program, &d.cols, rows, &mut o);
                                    black_box(&o);
                                }
                            });
                        }
                    });
                },
                min_s,
                1,
                false,
            );
            let fits_per_s = (fit_per * t) as f64 / fit_wall.max(f64::MIN_POSITIVE);
            let eval_rate = mrows(eval_per * t * rows, eval_wall);
            let (bf, be) = *base.get_or_insert((fits_per_s, eval_rate));
            out.line(&[
                rows.to_string(),
                t.to_string(),
                (fit_per * t).to_string(),
                format!("{fits_per_s:.3}"),
                format!("{:.2}", fits_per_s / bf),
                (eval_per * t).to_string(),
                format!("{eval_rate:.1}"),
                format!("{:.2}", eval_rate / be),
            ]);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{HELP}");
        return;
    }
    let opts = parse_args(&args);
    let cpus = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let mode_name = match opts.mode {
        Mode::Eval => "eval",
        Mode::Fit => "fit",
        Mode::Metric => "metric",
        Mode::Scale => "scale",
        Mode::All => "all",
    };
    println!(
        "# cpu_bench mode={mode_name} available_parallelism={cpus} avx={}",
        if eval_soa::avx_available() { "yes" } else { "no" }
    );
    let mut out = Out {
        tsv: opts.tsv,
        widths: Vec::new(),
    };
    if matches!(opts.mode, Mode::Eval | Mode::All) {
        run_eval(&opts, &mut out);
    }
    if matches!(opts.mode, Mode::Fit | Mode::All) {
        run_fit(&opts, &mut out);
    }
    if matches!(opts.mode, Mode::Metric | Mode::All) {
        run_metric(&opts, &mut out);
    }
    if matches!(opts.mode, Mode::Scale | Mode::All) {
        run_scale(&opts, &mut out, cpus);
    }
}
