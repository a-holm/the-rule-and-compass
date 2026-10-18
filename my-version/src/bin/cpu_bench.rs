//! `cpu_bench` — measure the structure-of-arrays CPU evaluator.
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
//! keep each measurement around 50 ms.
//!
//! ```text
//! cargo run --release --bin cpu_bench
//! cargo run --release --bin cpu_bench -- --help
//! ```

use std::hint::black_box;
use std::time::{Duration, Instant};

use rcsr::eval_soa;
use rcsr::expr::{self, Node, Op};

const ROWS: &[usize] = &[1_000, 20_000, 200_000];
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
fn trees() -> Vec<Node> {
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

    vec![five, nine, fifteen, twenty_five]
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
fn measure<F: FnMut()>(mut f: F) -> Duration {
    f(); // warm up
    let start = Instant::now();
    f();
    let one = start.elapsed();
    let target = Duration::from_millis(50);
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

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        println!(
            "cpu_bench {} — scalar tree walk vs SoA CPU evaluator\n\n\
             USAGE:\n  cpu_bench [--help]\n\n\
             Measures trees of 5, 9, 15 and 25 nodes over 1 000, 20 000 and\n\
             200 000 rows. Reports microseconds per evaluation and speed-up\n\
             versus the scalar walk, plus the maximum relative difference.",
            env!("CARGO_PKG_VERSION")
        );
        return;
    }
    if let Some(bad) = args.first() {
        eprintln!("cpu_bench: unknown argument '{bad}' (try --help)");
        std::process::exit(2);
    }

    println!("cpu_bench — scalar tree walk vs structure-of-arrays CPU evaluator");
    println!(
        "AVX kernel: {} (SIMD width {})",
        if eval_soa::avx_available() {
            "available"
        } else {
            "not available, scalar SoA used"
        },
        eval_soa::SIMD_WIDTH
    );
    println!();

    let trees = trees();
    let mut max_scalar_vs_soa = 0.0f64;
    let mut max_scalar_vs_avx = 0.0f64;
    let mut max_soa_vs_avx = 0.0f64;

    println!(
        "{:>5} {:>8} {:>12} {:>12} {:>12} {:>8} {:>8}",
        "nodes", "rows", "walk us", "soa us", "avx us", "soa x", "avx x"
    );
    println!("{}", "-".repeat(72));

    for tree in &trees {
        let program = match eval_soa::compile(tree) {
            Some(p) => p,
            None => {
                eprintln!("cpu_bench: a benchmark tree did not compile");
                std::process::exit(1);
            }
        };
        let nodes = expr::complexity(tree);
        for &rows in ROWS {
            let (cols, aos) = dataset(rows);
            let mut walk_out = vec![0.0; rows];
            let mut soa_out = vec![0.0; rows];
            let mut avx_out = vec![0.0; rows];

            let walk = measure(|| {
                for (r, row) in aos.iter().enumerate() {
                    walk_out[r] = expr::eval(tree, row);
                }
                black_box(&walk_out);
            });
            let soa = measure(|| {
                eval_soa::eval_batch_scalar(&program, &cols, rows, &mut soa_out);
                black_box(&soa_out);
            });
            let avx = measure(|| {
                eval_soa::eval_batch(&program, &cols, rows, &mut avx_out);
                black_box(&avx_out);
            });

            // One clean evaluation per path for the agreement check.
            for (r, row) in aos.iter().enumerate() {
                walk_out[r] = expr::eval(tree, row);
            }
            eval_soa::eval_batch_scalar(&program, &cols, rows, &mut soa_out);
            eval_soa::eval_batch(&program, &cols, rows, &mut avx_out);

            for r in 0..rows {
                max_scalar_vs_soa =
                    max_scalar_vs_soa.max(rel_diff(walk_out[r], soa_out[r]));
                max_scalar_vs_avx =
                    max_scalar_vs_avx.max(rel_diff(walk_out[r], avx_out[r]));
                max_soa_vs_avx = max_soa_vs_avx.max(rel_diff(soa_out[r], avx_out[r]));
            }

            let (walk_us, soa_us, avx_us) = (us(walk), us(soa), us(avx));
            println!(
                "{:>5} {:>8} {:>12.2} {:>12.2} {:>12.2} {:>8.2} {:>8.2}",
                nodes,
                rows,
                walk_us,
                soa_us,
                avx_us,
                walk_us / soa_us.max(f64::MIN_POSITIVE),
                walk_us / avx_us.max(f64::MIN_POSITIVE)
            );
        }
    }

    println!();
    println!("max relative difference vs the scalar walk:");
    println!("  SoA scalar : {max_scalar_vs_soa:.3e}");
    println!("  SoA AVX    : {max_scalar_vs_avx:.3e}");
    println!("  AVX vs SoA : {max_soa_vs_avx:.3e}");
    println!();
    println!("Times are one full evaluation of the dataset; lower is better.");
}
