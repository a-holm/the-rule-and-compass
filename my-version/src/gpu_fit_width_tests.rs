//! #977: a candidate's fit result must not depend on its batch-mates.
//!
//! The fit kernel has two instantiations, chosen by the largest constant count
//! of the call. A fixed pool of trees with at most 16 and with 17 to 20
//! constants is fitted in several batch splits, from cold and from warm
//! starts, for every objective the overlap pipeline can fit and every polish
//! mode. Each tree's constants, loss and status must have the same bits in
//! every split as in a call of its own. Device tests skip cleanly on a machine
//! without CUDA.

use super::*;
use crate::expr::{constant, input, node, Node, Op};
use crate::fit::{self, DeviceLoss, FitPolish, GpuFitPlan};

/// Not a multiple of either block width, so the last lanes of both
/// instantiations see fewer rows than the first.
const ROWS: usize = 777;
const VARS: usize = 20;
const ITERS: usize = 8;
const SMALL: [usize; 4] = [3, 6, 10, 16];
const LARGE: [usize; 3] = [17, 18, 20];

/// Twenty orthogonal columns `sin((k + 1) t)` over one period, a target that
/// uses all of them plus a saturating term, and an outlier on every ninth row
/// so the robust objectives move away from the least-squares answer.
fn rows() -> (Vec<f64>, Vec<f64>) {
    let mut cols = vec![0.0f64; ROWS * VARS];
    let mut y = Vec::with_capacity(ROWS);
    for i in 0..ROWS {
        let t = std::f64::consts::TAU * i as f64 / (ROWS - 1) as f64;
        let mut v = 0.0;
        for k in 0..VARS {
            let a = ((k + 1) as f64 * t).sin();
            cols[k * ROWS + i] = a;
            v += (1.0 + 0.31 * k as f64) * a;
        }
        v += 0.8 * (1.3 * cols[i]).tanh();
        v += (((i * 37) % 11) as f64 - 5.0) * 2e-3;
        if i % 9 == 4 {
            v += 2.5;
        }
        y.push(v);
    }
    (cols, y)
}

/// `a * (x0 + Σ c_k x_k + tanh(c_t x0)) + b`: the outer pair is the affine
/// gauge of the correlation arms, the inner x0 coefficient fixes the scale.
fn gauged(p: usize) -> Node {
    let mut inner = input(0);
    for k in 1..p - 2 {
        let term = node(Op::Mul, vec![constant(0.2 + 0.05 * k as f64), input(k)]);
        inner = node(Op::Add, vec![inner, term]);
    }
    let sat = node(Op::Tanh, vec![node(Op::Mul, vec![constant(0.7), input(0)])]);
    inner = node(Op::Add, vec![inner, sat]);
    node(Op::Add, vec![node(Op::Mul, vec![constant(1.1), inner]), constant(-0.1)])
}

/// `Σ c_k x_k` over the first `p` columns.
fn linear(p: usize) -> Node {
    let mut t = node(Op::Mul, vec![constant(0.25), input(0)]);
    for k in 1..p {
        let term = node(Op::Mul, vec![constant(0.25 + 0.07 * k as f64), input(k)]);
        t = node(Op::Add, vec![t, term]);
    }
    t
}

fn pool() -> Vec<Node> {
    let mut out = Vec::new();
    for &p in SMALL.iter().chain(LARGE.iter()) {
        out.push(gauged(p));
        out.push(linear(p));
    }
    out
}

/// Batches of pool indices. Every split fits every tree at least once.
fn splits(nconst: &[usize]) -> Vec<(&'static str, Vec<Vec<usize>>)> {
    let all: Vec<usize> = (0..nconst.len()).collect();
    let small: Vec<usize> = all.iter().copied().filter(|&k| nconst[k] <= 16).collect();
    let large: Vec<usize> = all.iter().copied().filter(|&k| nconst[k] > 16).collect();
    let pairs = small
        .iter()
        .rev()
        .enumerate()
        .map(|(i, &s)| vec![s, large[i % large.len()]])
        .collect();
    vec![
        ("alone", all.iter().map(|&k| vec![k]).collect()),
        ("one call", vec![all.clone()]),
        ("by width", vec![small.clone(), large.clone()]),
        ("reversed pairs", pairs),
    ]
}

#[derive(Clone, Copy, Debug)]
struct Arm {
    loss: DeviceLoss,
    polish: FitPolish,
}

fn arms() -> Vec<Arm> {
    let mut out = Vec::new();
    for loss in [DeviceLoss::Squared, DeviceLoss::Absolute, DeviceLoss::LogSquashed, DeviceLoss::Slope] {
        for polish in [FitPolish::F64, FitPolish::F32, FitPolish::Off] {
            out.push(Arm { loss, polish });
        }
    }
    for loss in [DeviceLoss::Correlation, DeviceLoss::CorrelationSigned] {
        out.push(Arm { loss, polish: FitPolish::F64 });
    }
    out
}

fn fit(ev: &mut GpuEvaluator, arm: Arm, plans: &[&GpuFitPlan]) -> Vec<GpuFitOutput> {
    ev.set_fit_polish(arm.polish);
    let (out, ms) = if arm.loss == DeviceLoss::Slope {
        ev.slope_fit_batch(plans, ITERS)
    } else {
        ev.fit_batch(plans, ITERS, arm.loss)
    };
    assert!(ms >= 0.0 && out.len() == plans.len(), "{arm:?}: the launch was refused");
    out
}

/// `(constant bits, loss bits, status)` of every tree in one split, one entry
/// per occurrence.
type Bits = (Vec<u64>, u64, i32);

fn bits(r: &GpuFitOutput) -> Bits {
    (r.c.iter().map(|v| v.to_bits()).collect(), r.sse.to_bits(), r.status)
}

fn run_split(
    ev: &mut GpuEvaluator,
    arm: Arm,
    plans: &[GpuFitPlan],
    batches: &[Vec<usize>],
) -> Vec<(usize, Bits)> {
    let mut out = Vec::new();
    for batch in batches {
        let refs: Vec<&GpuFitPlan> = batch.iter().map(|&k| &plans[k]).collect();
        for (&k, r) in batch.iter().zip(fit(ev, arm, &refs)) {
            out.push((k, bits(&r)));
        }
    }
    out
}

fn digest(rows: &[(usize, Bits)]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    let mut mix = |x: u64| {
        h ^= x;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for (k, (c, sse, status)) in rows {
        mix(*k as u64);
        c.iter().for_each(|&w| mix(w));
        mix(*sse);
        mix(*status as u64);
    }
    h
}

/// #977: every tree of the pool gets the bits of its own call in every split.
/// The digests of the single-tree calls are printed so two builds can be
/// compared on the trees whose calls had no batch-mates.
#[test]
fn fit_bits_independent_of_batch_split() {
    let (cols64, y) = rows();
    let cols32: Vec<f32> = cols64.iter().map(|&v| v as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut ev = match GpuEvaluator::new(&cols32, ROWS, VARS, &y32) {
        Some(ev) => ev,
        None => {
            crate::backend::skip_without_device("skipping GPU test: no usable CUDA device");
            return;
        }
    };
    assert!(ev.set_fit_data(&cols64, &y, &[]), "f64 fit rows uploaded");
    let cold: Vec<Node> = pool();
    let nconst: Vec<usize> = cold.iter().map(crate::expr::count_constants).collect();
    assert!(nconst.iter().any(|&p| p <= 16) && nconst.iter().any(|&p| (17..=20).contains(&p)));
    let splits = splits(&nconst);
    let mut mismatches: Vec<String> = Vec::new();
    let mut compared = 0usize;
    let mut kinds = [0usize; 3];
    for arm in arms() {
        let mut trees = cold.clone();
        for phase in ["cold", "warm"] {
            let plans: Vec<GpuFitPlan> =
                trees.iter().map(|t| fit::gpu_fit_plan(t).expect("pool plan")).collect();
            let reference = run_split(&mut ev, arm, &plans, &splits[0].1);
            eprintln!(
                "#977 digest {:?}/{} {phase} alone small {:016x} large {:016x}",
                arm.loss,
                arm.polish.name(),
                digest(&reference.iter().filter(|r| nconst[r.0] <= 16).cloned().collect::<Vec<_>>()),
                digest(&reference.iter().filter(|r| nconst[r.0] > 16).cloned().collect::<Vec<_>>()),
            );
            for (name, batches) in &splits[1..] {
                for (k, got) in run_split(&mut ev, arm, &plans, batches) {
                    compared += 1;
                    let want = &reference[k].1;
                    let fields = [got.0 != want.0, got.1 != want.1, got.2 != want.2];
                    if fields.iter().any(|&f| f) {
                        for (i, &f) in fields.iter().enumerate() {
                            kinds[i] += f as usize;
                        }
                        mismatches.push(format!(
                            "{:?}/{} {phase} {name}: tree {k} (p={}) constants {} loss {} status {}",
                            arm.loss,
                            arm.polish.name(),
                            nconst[k],
                            if fields[0] { "differ" } else { "equal" },
                            if fields[1] { "differs" } else { "equal" },
                            if fields[2] { format!("{} vs {}", got.2, want.2) } else { "equal".into() },
                        ));
                    }
                }
            }
            // The warm pass refits each tree from its own single-call answer.
            for (k, (c, _, _)) in reference.iter() {
                let c: Vec<f64> = c.iter().map(|&w| f64::from_bits(w)).collect();
                if c.iter().all(|v| v.is_finite()) {
                    plans[*k].apply(&mut trees[*k], &c);
                }
            }
        }
    }
    for line in mismatches.iter().take(24) {
        eprintln!("  {line}");
    }
    assert!(
        mismatches.is_empty(),
        "{} of {compared} fits differ from the tree's own call (constants {}, loss {}, status {})",
        mismatches.len(),
        kinds[0],
        kinds[1],
        kinds[2]
    );
}
