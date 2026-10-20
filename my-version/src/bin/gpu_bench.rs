//! GPU evaluation benchmark: `rcsr-gpu-bench --data <file> --target y [--vars x]`.
//!
//! Builds a batch of random expressions, scores the whole batch on the GPU in a
//! single launch, then scores the same batch on the CPU, and reports the
//! speed-up and the agreement between the two. Requires `--features cuda`.

use rcsr::data;
use rcsr::eval_soa;
use rcsr::expr::{self, complexity, random_tree, Op};
use rcsr::gpu::{cpu_sse, GpuEvaluator, Population};
use rcsr::rng::Mt;
fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut path = "examples/sine.csv".to_string();
    let mut target = "y".to_string();
    let mut vars: Vec<String> = vec!["x".to_string()];
    let mut batch = 4000usize;
    let mut depth = 4usize;
    let mut seed = 1u32;
    let mut ops: Vec<Op> = Op::default_set();
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--ops" => {
                i += 1;
                if let Some(s) = args.get(i) {
                    ops = Op::parse_list(s);
                }
            }
            "--data" => {
                i += 1;
                path = args.get(i).cloned().unwrap_or(path);
            }
            "--target" => {
                i += 1;
                target = args.get(i).cloned().unwrap_or(target);
            }
            "--vars" => {
                i += 1;
                vars = args
                    .get(i)
                    .map(|s| s.split(',').map(|x| x.trim().to_string()).collect())
                    .unwrap_or(vars);
            }
            "--batch" => {
                i += 1;
                batch = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(batch);
            }
            "--depth" => {
                i += 1;
                depth = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(depth);
            }
            "--seed" => {
                i += 1;
                seed = args.get(i).and_then(|s| s.parse().ok()).unwrap_or(seed);
            }
            "-h" | "--help" => {
                println!(
                    "rcsr-gpu-bench [--data file] [--target y] [--vars a,b] [--batch N] [--depth D] [--seed S]"
                );
                return;
            }
            _ => {}
        }
        i += 1;
    }

    let table = match data::read_table(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    };
    let mut rng = Mt::new(seed);
    let (train, _test, _m, _s) = match data::prepare(&table, &vars, &target, 0.0, &mut rng) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(3);
        }
    };
    let n = train.len();
    let nv = vars.len();

    // Column-major (SoA) copies for the kernel.
    let mut cols = vec![0.0f32; nv * n];
    for (r, row) in train.x.iter().enumerate() {
        for v in 0..nv {
            cols[v * n + r] = row[v] as f32;
        }
    }
    let y32: Vec<f32> = train.y.iter().map(|v| *v as f32).collect();

    // Build a batch of random expressions and compile them into one population.
    let mut population = Population::new();
    let mut trees = Vec::new();
    while trees.len() < batch {
        let t = expr::simplify(&random_tree(&mut rng, nv, &ops, depth));
        if complexity(&t) > 40 {
            continue;
        }
        if !population.add(&t) {
            continue;
        }
        trees.push(t);
    }
    let bytes_missing = population.unsupported();

    let device = GpuEvaluator::device_name();
    let mut ev = match GpuEvaluator::new(&cols, n, nv, &y32) {
        Some(e) => e,
        None => {
            eprintln!("error: could not create a GPU context");
            std::process::exit(4);
        }
    };

    // Warm up, then time the GPU batch (averaged over several runs to hide the
    // single-launch overhead). `eval_population` scores every program in the
    // population in one call and one kernel launch. #118: the reported time is
    // the full evaluator path (alloc/upload/launch/sync/download), not the
    // kernel event alone.
    let _ = ev.eval_population(&population);
    let reps = 25;
    let before = ev.stats();
    let t_gpu = std::time::Instant::now();
    let mut gpu_err = Vec::new();
    for _ in 0..reps {
        let (e, _ms) = ev.eval_population(&population);
        gpu_err = e;
    }
    let wall = t_gpu.elapsed().as_secs_f64();
    let gpu_stats = ev.stats().since(&before);
    let ms = wall * 1000.0 / reps as f64;

    // CPU reference: same batch, f64 tree walk.
    let t0 = std::time::Instant::now();
    let mut cpu_err = Vec::with_capacity(trees.len());
    for t in &trees {
        cpu_err.push(cpu_sse(t, &cols, n, nv, &y32));
    }
    let cpu_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // #27/#118: the gate is the engine's own SoA/AVX path, not the interpreted
    // scalar walk. Compile once and time the same batch over all rows. Trees
    // the SoA evaluator cannot compile are counted, not silently included.
    let mut cols64: Vec<Vec<f64>> = vec![Vec::with_capacity(n); nv];
    for r in 0..n {
        for v in 0..nv {
            cols64[v].push(cols[v * n + r] as f64);
        }
    }
    let programs_soa: Vec<_> = trees.iter().map(eval_soa::compile).collect();
    let soa_n = programs_soa.iter().filter(|p| p.is_some()).count();
    let mut soa_out = vec![0.0; n];
    let soa_ms = if soa_n == 0 {
        0.0
    } else {
        let start = std::time::Instant::now();
        for p in programs_soa.iter().flatten() {
            eval_soa::eval_batch(p, &cols64, n, &mut soa_out);
            std::hint::black_box(&soa_out);
        }
        start.elapsed().as_secs_f64() * 1000.0
    };

    // Agreement check (f32 GPU vs f64 CPU), over pairs that both produced an
    // ordinary finite result. f32 has ~7 significant digits and a narrower
    // exponent range, so we report the median and the 99th percentile rather
    // than a single worst case.
    let mut rel: Vec<f64> = Vec::with_capacity(gpu_err.len());
    let mut penalised = 0usize;
    for (g, c) in gpu_err.iter().zip(&cpu_err) {
        let gf = (*g as f64).is_finite() && *g < 1e29;
        let cf = c.is_finite() && *c < 1e29;
        if !gf || !cf {
            penalised += 1;
            continue;
        }
        rel.push(((*g as f64) - c).abs() / (1.0 + c.abs()));
    }
    rel.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = rel.get(rel.len() / 2).copied().unwrap_or(0.0);
    let p99 = rel.get(rel.len() * 99 / 100).copied().unwrap_or(0.0);
    let within = rel.iter().filter(|d| **d < 1e-3).count();

    let row_evals = (trees.len() * n) as f64;
    let evals = trees.len() as f64;
    println!("device           : {}", device);
    println!(
        "memory           : {}",
        if GpuEvaluator::unified_memory() {
            "unified (cudaMallocManaged)"
        } else {
            "device (cudaMalloc)"
        }
    );
    println!("dataset          : {} ({} rows x {} vars)", path, n, nv);
    println!(
        "expressions      : {} (depth {}, avg complexity {:.1})",
        trees.len(),
        depth,
        trees.iter().map(|t| complexity(t) as f64).sum::<f64>() / trees.len() as f64
    );
    println!(
        "population       : {} programs, {} code words, one kernel launch per evaluation",
        population.len(),
        population.programs().0.len()
    );
    if bytes_missing > 0 {
        println!("skipped (no GPU): {} (branchy `if` nodes)", bytes_missing);
    }
    println!();
    println!(
        "GPU batch        : {:>8.3} ms   ({:>10.0} candidate evals/s, {:>10.3e} row-evals/s, full path incl. alloc/sync/copy)",
        ms,
        gpu_stats.evals_per_s(),
        gpu_stats.row_evals_per_s()
    );
    println!(
        "CPU batch (f64)  : {:>8.3} ms   ({:>10.0} candidates/s, {:>10.3e} row-evals/s, interpreted tree walk)",
        cpu_ms,
        evals / (cpu_ms / 1000.0).max(1e-12),
        row_evals / (cpu_ms / 1000.0).max(1e-12)
    );
    println!(
        "CPU batch (SoA)  : {:>8.3} ms   ({:>10.0} candidates/s, {:>10.3e} row-evals/s, AVX SoA path, {}/{} compiled)",
        soa_ms,
        if soa_ms > 0.0 {
            soa_n as f64 / (soa_ms / 1000.0)
        } else {
            0.0
        },
        if soa_ms > 0.0 {
            (soa_n * n) as f64 / (soa_ms / 1000.0)
        } else {
            0.0
        },
        soa_n,
        trees.len()
    );
    println!(
        "speed-up         : {:>8.1}x vs interpreted walk | {:>8.1}x vs SoA/AVX",
        cpu_ms / ms.max(1e-9),
        soa_ms / ms.max(1e-9)
    );
    let host = gpu_stats.host_s;
    println!(
        "accounting       : kernel={:.3}ms sync={:.3}ms alloc={:.3}ms h2d={:.3}ms d2h={:.3}ms launch={:.3}ms host={:.3}ms | wall={:.3}ms",
        gpu_stats.kernel_s * 1e3,
        gpu_stats.sync_s * 1e3,
        gpu_stats.alloc_s * 1e3,
        gpu_stats.h2d_s * 1e3,
        gpu_stats.d2h_s * 1e3,
        gpu_stats.launch_s * 1e3,
        host * 1e3,
        gpu_stats.wall_s * 1e3,
    );
    println!(
        "decomposition    : C windows {:>6.2}% + host {:>5.2}% = {:>6.2}% of wall (residual {:+.2}%); H2D {} B, D2H {} B per {} batches",
        100.0 * gpu_stats.window_s() / gpu_stats.wall_s.max(1e-12),
        100.0 * gpu_stats.host_s / gpu_stats.wall_s.max(1e-12),
        100.0 * gpu_stats.accounted_s() / gpu_stats.wall_s.max(1e-12),
        100.0 * gpu_stats.residual_fraction(),
        gpu_stats.h2d_bytes,
        gpu_stats.d2h_bytes,
        gpu_stats.batches
    );
    println!(
        "duty / occupancy : kernel-time duty {:.1}% (kernel/wall); SM occupancy requires ncu (not sampled, never fabricated)",
        100.0 * gpu_stats.duty_cycle()
    );
    let best = gpu_err
        .iter()
        .filter(|v| v.is_finite() && **v > 0.0)
        .fold(f64::INFINITY, |a, b| a.min(*b as f64));
    let mean_y = y32.iter().map(|&y| y as f64).sum::<f64>() / y32.len().max(1) as f64;
    let baseline = y32
        .iter()
        .map(|y| (*y as f64 - mean_y) * (*y as f64 - mean_y))
        .sum::<f64>();
    match gpu_stats.quality_per_gpu_hour(best, baseline) {
        Some(q) => println!(
            "quality/GPU-hour : {:.3} decades vs constant-mean baseline per measured full-path GPU-hour",
            q
        ),
        None => println!("quality/GPU-hour : n/a (no finite positive SSE or no GPU wall)"),
    }
    println!(
        "agreement        : median {:.1e}, p99 {:.1e} rel. error; {}/{} within 1e-3 ({} hard failures)",
        med,
        p99,
        within,
        rel.len(),
        penalised
    );
    println!(
        "provenance       : duty from the harness ledger; for CUPTI/nsys/ncu commands see docs/gpu-backend.md"
    );
}
