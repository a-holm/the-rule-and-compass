//! #288: the radix select path of the exact order-statistic scorers.
//!
//! Every statistic the select path computes (median, maximum) must be the same
//! bits as the segmented-sort path and as the CPU metric for the same `f64`
//! prediction vector, whatever the launch partition, wave split or batch
//! composition. The scoring interquartile mean is pinned bit for bit on the sort
//! path. Device-dependent tests skip cleanly on a machine without CUDA.

use super::*;
use crate::expr::{constant, input, node};
use crate::fit::{gpu_fit_plan_reason, GpuFitPlan};
use crate::metrics::{score_weighted, Metric};

struct Fixture {
    cols32: Vec<f32>,
    cols64: Vec<f64>,
    n: usize,
    nv: usize,
    y: Vec<f64>,
    x: Vec<Vec<f64>>,
}

/// Two columns (a ramp and a hashed noise column) and a target with exact
/// repeats (ties in `|r|`), gross outliers and exact zeros.
fn fixture(n: usize) -> Fixture {
    let nv = 2;
    let mut cols64 = vec![0.0f64; nv * n];
    let mut x = Vec::with_capacity(n);
    let mut y: Vec<f64> = Vec::with_capacity(n);
    for i in 0..n {
        let t = -2.0 + 4.0 * i as f64 / (n.max(2) - 1) as f64;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ 0x9e37_79b9) as f64 / u32::MAX as f64;
        let v = 1.4 * (0.9 * t).sin() + 0.3 * t + 0.5 * s;
        let v = if i % 5 == 4 { y[i - 1] } else { v };
        let v = if i % 11 == 7 { 30.0 } else { v };
        let v = if i % 13 == 6 { 0.0 } else { v };
        x.push(vec![t, s]);
        y.push(v);
        cols64[i] = t;
        cols64[n + i] = s;
    }
    custom_cols(cols64, nv, y)
}

/// A fixture from column-major `f64` columns and a target.
fn custom_cols(cols64: Vec<f64>, nv: usize, y: Vec<f64>) -> Fixture {
    let n = y.len();
    let x = (0..n).map(|i| (0..nv).map(|v| cols64[v * n + i]).collect()).collect();
    Fixture {
        cols32: cols64.iter().map(|&v| finite_f32(v)).collect(),
        cols64,
        n,
        nv,
        y,
        x,
    }
}

/// The f32 rows are only the legacy evaluator's copy; the order statistics
/// read the exact `f64` twin. Keep the f32 copy finite.
fn finite_f32(v: f64) -> f32 {
    let f = v as f32;
    if f.is_finite() {
        f
    } else {
        0.0
    }
}

fn one_column(x: &[f64], y: &[f64]) -> Fixture {
    custom_cols(x.to_vec(), 1, y.to_vec())
}

fn evaluator(f: &Fixture) -> Option<GpuEvaluator> {
    if !probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let y32: Vec<f32> = f.y.iter().map(|&v| finite_f32(v)).collect();
    let mut ev = GpuEvaluator::new(&f.cols32, f.n, f.nv, &y32).expect("a probed device opens");
    assert!(ev.set_fit_data(&f.cols64, &f.y, &[]), "f64 fit data upload");
    Some(ev)
}

fn plan(t: &Node) -> GpuFitPlan {
    gpu_fit_plan_reason(t).expect("an eligible plan")
}

/// `c0·x0 + c1`: arithmetic only, so the device's f64 forward sweep and
/// `expr::eval` agree to the last bit.
fn affine() -> Node {
    node(
        Op::Add,
        vec![node(Op::Mul, vec![constant(0.7), input(0)]), constant(0.1)],
    )
}

/// `c0·x0`.
fn scaled() -> Node {
    node(Op::Mul, vec![constant(1.0), input(0)])
}

fn with_constants(t: &Node, c: &[f64]) -> Node {
    let mut out = t.clone();
    plan(t).apply(&mut out, c);
    out
}

fn metric_of(code: OrderCode) -> Metric {
    match code {
        OrderCode::Median => Metric::Median,
        OrderCode::Maximum => Metric::Maximum,
        OrderCode::InterquartileExact | OrderCode::InterquartileTree => Metric::Interquartile,
    }
}

/// The configured CPU metric of tree `t` with constants `c`.
fn cpu(f: &Fixture, t: &Node, c: &[f64], code: OrderCode) -> f64 {
    let tree = with_constants(t, c);
    let pred: Vec<f64> = f.x.iter().map(|r| crate::expr::eval(&tree, r)).collect();
    score_weighted(&pred, &f.y, &[], metric_of(code), crate::expr::count_constants(t))
}

/// One candidate, its trials `cs`, one statistic on one path.
fn run(
    ev: &mut GpuEvaluator,
    p: &GpuFitPlan,
    cs: &[Vec<f64>],
    code: OrderCode,
    path: OrderPath,
    blocks: i32,
) -> Vec<Option<f64>> {
    ev.order_code_trials(&[p], &[cs.to_vec()], code, path, GpuEvaluator::MEDIAN_WAVE_BYTES, blocks)
        .expect("the device answered")
        .remove(0)
}

fn bits(v: &[Option<f64>]) -> Vec<Option<u64>> {
    v.iter().map(|x| x.map(f64::to_bits)).collect()
}

/// The CPU reference in the device's labelling: a non-finite score is `None`.
fn cpu_bits(f: &Fixture, t: &Node, cs: &[Vec<f64>], code: OrderCode) -> Vec<Option<u64>> {
    cs.iter()
        .map(|c| {
            let v = cpu(f, t, c, code);
            v.is_finite().then(|| v.to_bits())
        })
        .collect()
}

/// Every (statistic, path) pair this module pins: the select path computes
/// the median and the maximum; the scoring interquartile mean is sort-only.
const PAIRS: [(OrderCode, OrderPath); 5] = [
    (OrderCode::Median, OrderPath::Sort),
    (OrderCode::Median, OrderPath::Select),
    (OrderCode::Maximum, OrderPath::Sort),
    (OrderCode::Maximum, OrderPath::Select),
    (OrderCode::InterquartileExact, OrderPath::Sort),
];

fn assert_exact(f: &Fixture, ev: &mut GpuEvaluator, t: &Node, cs: &[Vec<f64>], what: &str) {
    let p = plan(t);
    for (code, path) in PAIRS {
        let got = bits(&run(ev, &p, cs, code, path, 0));
        let want = cpu_bits(f, t, cs, code);
        assert_eq!(got, want, "{what}: {code:?} on {path:?} differs from the CPU");
    }
}

fn trial_constants() -> Vec<Vec<f64>> {
    vec![
        vec![0.7, 0.1],
        vec![0.0, 0.0],
        vec![1.0, 0.5],
        vec![-1.0, 2.0],
        vec![12.0, -7.0],
        vec![0.3, -30.0],
    ]
}

/// Median and maximum on both paths, and the scoring interquartile mean, are
/// the CPU metric bit for bit at every small size (odd, even, `n < 4`, one
/// row) and around the 2 048-bin and tile boundaries.
#[test]
fn both_paths_match_the_cpu_bit_for_bit_at_every_small_size() {
    let sizes = (1..=40usize).chain([63, 64, 65, 127, 128, 129, 255, 256, 257, 1000, 2049, 4095, 4096, 4097]);
    for n in sizes {
        let f = fixture(n);
        let Some(mut ev) = evaluator(&f) else { return };
        assert_exact(&f, &mut ev, &affine(), &trial_constants(), &format!("n={n}"));
    }
}

/// The selected value is a function of the key multiset only: one block per
/// segment, several, and the production shape give the same bits, and they are
/// the sort path's and the CPU's. The window mean crosses many staging tiles.
#[test]
fn the_select_value_does_not_depend_on_the_block_partition() {
    for n in [70_001usize, 70_000] {
        let f = fixture(n);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let cs = trial_constants();
        for code in [OrderCode::Median, OrderCode::Maximum] {
            let want = cpu_bits(&f, &t, &cs, code);
            assert_eq!(bits(&run(&mut ev, &p, &cs, code, OrderPath::Sort, 0)), want, "n={n} {code:?} sort");
            for blocks in [0, 1, 2, 3, 7, 64] {
                let got = bits(&run(&mut ev, &p, &cs, code, OrderPath::Select, blocks));
                assert_eq!(got, want, "n={n} {code:?} select with {blocks} blocks per segment");
            }
        }
        let iqr = run(&mut ev, &p, &cs, OrderCode::InterquartileExact, OrderPath::Sort, 0);
        assert_eq!(bits(&iqr), cpu_bits(&f, &t, &cs, OrderCode::InterquartileExact), "n={n} interquartile");
    }
}

/// Batch composition, wave split and repetition cannot change a selected
/// value: a candidate alone, as one trial among three, as the second set of a
/// mixed batch, one segment per wave, and again.
#[test]
fn select_is_independent_of_batch_composition_and_waves() {
    let f = fixture(5_003);
    let Some(mut ev) = evaluator(&f) else { return };
    let t = affine();
    let p = plan(&t);
    let other = scaled();
    let q = plan(&other);
    let c = vec![0.8, -0.2];
    for code in [OrderCode::Median, OrderCode::Maximum] {
        let sel = OrderPath::Select;
        let wave = GpuEvaluator::MEDIAN_WAVE_BYTES;
        let alone = run(&mut ev, &p, &[c.clone()], code, sel, 0)[0].expect("finite");
        let triple = run(&mut ev, &p, &[vec![2.0, 2.0], c.clone(), vec![0.0, 1.0]], code, sel, 0)[1]
            .expect("finite");
        let mixed = ev
            .order_code_trials(&[&q, &p], &[vec![vec![-0.5]], vec![c.clone()]], code, sel, wave, 0)
            .expect("mixed batch")[1][0]
            .expect("finite");
        let one_per_wave = ev
            .order_code_trials(&[&q, &p], &[vec![vec![-0.5]], vec![c.clone()]], code, sel, 1, 0)
            .expect("one segment per wave")[1][0]
            .expect("finite");
        let again = run(&mut ev, &p, &[c.clone()], code, sel, 0)[0].expect("finite");
        let want = cpu(&f, &t, &c, code);
        for (label, v) in [
            ("alone", alone),
            ("triple", triple),
            ("mixed", mixed),
            ("one per wave", one_per_wave),
            ("again", again),
        ] {
            assert_eq!(v.to_bits(), want.to_bits(), "{code:?} {label}: {v} vs CPU {want}");
        }
    }
}

/// Ties, signed zeros, subnormal and huge residuals select exactly: an
/// entirely tied multiset at every small size, `-0.0` residuals that must read
/// as `+0.0`, subnormals, and magnitudes up to `f64::MAX`.
#[test]
fn ties_signed_zeros_and_extreme_residuals_are_exact() {
    let t = scaled();
    for n in 1..=9usize {
        let f = one_column(&vec![0.0; n], &vec![1.0; n]);
        let Some(mut ev) = evaluator(&f) else { return };
        assert_exact(&f, &mut ev, &t, &[vec![1.0], vec![-3.0]], &format!("all tied n={n}"));
    }
    // pred = -x: rows with x = 0 give r = -0.0 - 0.0 = -0.0.
    for x in [vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0], vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 3.0, 0.0]] {
        let f = one_column(&x, &vec![0.0; x.len()]);
        let Some(mut ev) = evaluator(&f) else { return };
        let p = plan(&t);
        let got = run(&mut ev, &p, &[vec![-1.0]], OrderCode::Median, OrderPath::Select, 0)[0];
        assert_eq!(got.map(f64::to_bits), Some(0.0f64.to_bits()), "a -0.0 residual is +0.0");
        assert_exact(&f, &mut ev, &t, &[vec![-1.0], vec![1.0]], &format!("signed zero n={}", x.len()));
    }
    let subnormal = [5e-324, 1e-310, 2.5e-320, 1e-300, 0.0, 4e-315, 7e-323];
    let huge = [1e300, -1e300, 7.0, 1e-300, -3.5, f64::MAX, -f64::MAX];
    for y in [subnormal.to_vec(), huge.to_vec()] {
        let f = one_column(&vec![1.0; y.len()], &y);
        let Some(mut ev) = evaluator(&f) else { return };
        assert_exact(&f, &mut ev, &t, &[vec![0.0], vec![-0.0]], &format!("extreme {y:?}"));
    }
}

/// A non-finite prediction is refused on both paths, whether every row or a
/// single row is non-finite; the CPU scores both `+inf`.
#[test]
fn a_non_finite_prediction_is_refused_on_both_paths() {
    let ln = node(Op::Ln, vec![node(Op::Mul, vec![constant(1.0), input(0)])]);
    let all_bad: Vec<f64> = (0..300).map(|i| -1.0 - i as f64).collect();
    let mut one_bad: Vec<f64> = (0..301).map(|i| 1.0 + i as f64).collect();
    one_bad[157] = -2.0;
    for x in [all_bad, one_bad] {
        let y: Vec<f64> = (0..x.len()).map(|i| (i % 7) as f64).collect();
        let f = one_column(&x, &y);
        let Some(mut ev) = evaluator(&f) else { return };
        let p = plan(&ln);
        for (code, path) in PAIRS {
            let got = run(&mut ev, &p, &[vec![1.0]], code, path, 0);
            assert_eq!(got, vec![None], "{code:?} on {path:?} scored a non-finite prediction");
            assert_eq!(cpu(&f, &ln, &[1.0], code), f64::INFINITY);
        }
    }
}

/// The measured crossover picks the path, and the sort path answers whenever
/// the select call fails as a whole (here: a statistic it does not compute).
#[test]
fn the_crossover_picks_the_path_and_the_sort_path_is_the_fallback() {
    let at = ORDER_SELECT_MIN_ROWS;
    for code in [OrderCode::Median, OrderCode::Maximum] {
        assert_eq!(order_stat_path(code, at - 1), OrderPath::Sort);
        assert_eq!(order_stat_path(code, at), OrderPath::Select);
    }
    for code in [OrderCode::InterquartileExact, OrderCode::InterquartileTree] {
        assert!(!code.selectable());
        assert_eq!(order_stat_path(code, at), OrderPath::Sort);
        assert_eq!(order_stat_path(code, usize::MAX), OrderPath::Sort);
    }
    for n in [at - 1, at] {
        let f = fixture(n);
        let Some(mut ev) = evaluator(&f) else { return };
        let t = affine();
        let p = plan(&t);
        let cs = trial_constants();
        let trials = vec![cs.clone()];
        let wave = GpuEvaluator::MEDIAN_WAVE_BYTES;
        for code in [OrderCode::Median, OrderCode::Maximum] {
            let chosen = ev.order_code_score(&[&p], &trials, code, wave).expect("dispatched")[0].clone();
            let sort = run(&mut ev, &p, &cs, code, OrderPath::Sort, 0);
            let select = run(&mut ev, &p, &cs, code, OrderPath::Select, 0);
            assert_eq!(bits(&chosen), bits(&sort), "n={n} {code:?}");
            assert_eq!(bits(&select), bits(&sort), "n={n} {code:?}");
            assert_eq!(bits(&sort), cpu_bits(&f, &t, &cs, code), "n={n} {code:?}");
        }
        let code = OrderCode::InterquartileExact;
        assert!(
            ev.order_code_trials(&[&p], &trials, code, OrderPath::Select, wave, 0).is_none(),
            "the select entry must refuse a window mean"
        );
        let fell_back = ev
            .order_code_score_on(&[&p], &trials, code, OrderPath::Select, wave)
            .expect("the sort path answered")[0]
            .clone();
        assert_eq!(bits(&fell_back), cpu_bits(&f, &t, &cs, code), "n={n} fallback");
    }
}

/// Paired measurement for the crossover: sort path vs select path vs the
/// 8-worker CPU scorer, one affine candidate shape, `P` candidates of `N` rows.
/// `RCSR_BENCH_NS` / `RCSR_BENCH_PS` override the comma-separated grids.
#[test]
#[ignore]
fn bench_order_select_crossover() {
    let grid = |key: &str, default: &[usize]| -> Vec<usize> {
        std::env::var(key)
            .ok()
            .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
            .unwrap_or_else(|| default.to_vec())
    };
    let ns = grid("RCSR_BENCH_NS", &[4096, 16_384, 65_536, 262_144, 1_048_576]);
    let ps = grid("RCSR_BENCH_PS", &[8, 128]);
    let reps = 3;
    let t = affine();
    let p = plan(&t);
    eprintln!("stat n p path seconds s_per_row_cand");
    for &n in &ns {
        let f = fixture(n);
        let Some(mut ev) = evaluator(&f) else { return };
        for &np in &ps {
            let constants: Vec<Vec<f64>> =
                (0..np).map(|k| vec![0.7 * (1.0 + 0.001 * k as f64), 0.1]).collect();
            let refs: Vec<&GpuFitPlan> = (0..np).map(|_| &p).collect();
            let trials: Vec<Vec<Vec<f64>>> = constants.iter().map(|c| vec![c.clone()]).collect();
            let wave = GpuEvaluator::MEDIAN_WAVE_BYTES;
            let report = |stat: &str, path: &str, s: f64| {
                eprintln!("{stat} {n} {np} {path} {s:.6} {:.3e}", s / (n as f64 * np as f64));
            };
            let mut device: Vec<(OrderCode, OrderPath, &str)> = vec![
                (OrderCode::Median, OrderPath::Sort, "sort"),
                (OrderCode::Median, OrderPath::Select, "select"),
                (OrderCode::Maximum, OrderPath::Sort, "sort"),
                (OrderCode::Maximum, OrderPath::Select, "select"),
                (OrderCode::InterquartileTree, OrderPath::Sort, "sort_tree"),
                (OrderCode::InterquartileExact, OrderPath::Sort, "sort_exact"),
            ];
            for (code, path, label) in device.drain(..) {
                let mut best = f64::INFINITY;
                let mut first: Option<Vec<Vec<Option<f64>>>> = None;
                for rep in 0..=reps {
                    let start = std::time::Instant::now();
                    let out = ev
                        .order_code_trials(&refs, &trials, code, path, wave, 0)
                        .expect("the device scored the batch");
                    let s = start.elapsed().as_secs_f64();
                    if rep > 0 {
                        best = best.min(s);
                    }
                    first.get_or_insert(out);
                }
                assert_eq!(first.expect("ran").len(), np);
                report(&format!("{:?}", metric_of(code)).to_lowercase(), label, best);
            }
            let mut best = f64::INFINITY;
            for rep in 0..=reps {
                let start = std::time::Instant::now();
                let out = ev.max_score_trials(&refs, &trials).expect("the scan ran");
                let s = start.elapsed().as_secs_f64();
                if rep > 0 {
                    best = best.min(s);
                }
                assert_eq!(out.len(), np);
            }
            report("maximum", "scan", best);
            for metric in [Metric::Median, Metric::Maximum, Metric::Interquartile] {
                let start = std::time::Instant::now();
                let workers = 8usize;
                let chunk = np.div_ceil(workers);
                std::thread::scope(|scope| {
                    for part in constants.chunks(chunk) {
                        let f = &f;
                        let t = &t;
                        scope.spawn(move || {
                            for c in part {
                                let tree = with_constants(t, c);
                                let pred: Vec<f64> =
                                    f.x.iter().map(|r| crate::expr::eval(&tree, r)).collect();
                                std::hint::black_box(score_weighted(&pred, &f.y, &[], metric, 2));
                            }
                        });
                    }
                });
                let s = start.elapsed().as_secs_f64();
                report(&format!("{metric:?}").to_lowercase(), "cpu8", s);
            }
        }
    }
}
