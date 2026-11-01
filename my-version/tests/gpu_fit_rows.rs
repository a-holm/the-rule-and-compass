//! `--fit-rows` on the device: constants are fitted on a fixed subsample
//! resident next to the full-row evaluator, every row is still scored, and the
//! reported errors stay the exact CPU numbers on all rows.
//!
//! Device-dependent tests skip cleanly on a machine without CUDA.

#[cfg(feature = "cuda")]
use rcsr::checkpoint::Backend;
use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

const ROWS: usize = 80_000;
const FIT_ROWS: usize = 20_000;

fn sample(n: usize, seed: u32, weighted: bool) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut w = Vec::new();
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2 + (((i * 37) % 11) as f64 - 5.0) * 1e-3);
        if weighted {
            w.push(0.5 + ((i * 7) % 5) as f64 * 0.25);
        }
    }
    Sample {
        x,
        y,
        w,
        ys: Vec::new(),
        pairs: None,
    }
}

fn config(seed: u32, fit_rows: usize) -> Config {
    Config {
        generations: 5,
        islands: 1,
        pop: 24,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Squared,
        gpu: true,
        fit_rows,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

#[cfg(feature = "cuda")]
fn describe(out: &rcsr::search::Outcome) -> Vec<(u64, usize, String)> {
    out.archive
        .items
        .iter()
        .map(|s| (s.error.to_bits(), s.complexity, format!("{:?}", s.tree)))
        .collect()
}

#[cfg(feature = "cuda")]
#[test]
fn the_device_admits_a_fit_subsample_and_keeps_scoring_every_row() {
    let train = sample(ROWS, 5, false);
    assert_eq!(
        search::effective_backend(&config(3, FIT_ROWS), &train),
        Backend::Gpu
    );
    let cpu_fit = Config {
        fit_strategy: FitStrategy::Exact,
        ..config(3, FIT_ROWS)
    };
    assert_eq!(search::effective_backend(&cpu_fit, &train), Backend::Cpu);
    let reason = search::backend_blocker_reason(&cpu_fit, &train).expect("a labelled refusal");
    assert!(reason.contains("device fit arm"), "{reason}");
    // A request at or above the sample size is a full fit and needs nothing.
    assert_eq!(
        search::effective_backend(&cpu_fit_rows_full(&train), &train),
        Backend::Gpu
    );
}

#[cfg(feature = "cuda")]
fn cpu_fit_rows_full(train: &Sample) -> Config {
    config(3, train.len())
}

#[cfg(feature = "cuda")]
#[test]
fn a_fit_subsample_defers_fits_to_the_device_and_scores_all_rows() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(ROWS, 5, false);
    let test = sample(2_000, 6, false);
    let out = search::run(&config(11, FIT_ROWS), &train, &test);
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert!(
        g.fit_gpu > 0,
        "the device fit ran: attempts {} gpu {} fallback {:?} dispatch {:?}",
        g.fit_attempts,
        g.fit_gpu,
        g.fit_fallback_by_cause,
        out.dispatch.manifest_json()
    );
    assert_eq!(g.fit_rows, FIT_ROWS, "the fit twin holds the subsample rows");
    assert_eq!(g.score_rows, ROWS, "the evaluator scores every row");
    // Two variables plus the target, f64.
    assert_eq!(g.fit_twin_bytes, (FIT_ROWS * 3 * 8) as u64);
    assert!(g.fit_subsample_hash.is_some());
    assert_eq!(out.fit_rows, FIT_ROWS);
    // The device scored all rows: its accounting covers the full row count.
    assert!(g.accounting.batches > 0);
    assert_eq!(g.accounting.rows, ROWS as u64);
    assert_eq!(g.accounting.row_evals, g.accounting.programs * ROWS as u64);
}

#[cfg(feature = "cuda")]
#[test]
fn the_fit_twin_is_charged_at_the_subsample_rows_with_weights() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(ROWS, 5, true);
    let test = sample(2_000, 6, true);
    let out = search::run(&config(11, FIT_ROWS), &train, &test);
    let g = out.gpu.as_ref().expect("a GPU report");
    assert_eq!(g.fit_rows, FIT_ROWS);
    assert_eq!(g.fit_twin_bytes, (FIT_ROWS * 3 * 8 + FIT_ROWS * 8) as u64);
    let plan = g.memory_plan.as_ref().expect("a memory plan");
    assert_eq!(plan.workload.fit_rows, FIT_ROWS);
    assert_eq!(plan.terms.fit_twin, (FIT_ROWS * 3 * 12 + FIT_ROWS * 8) as u64);
}

#[cfg(feature = "cuda")]
#[test]
fn a_fit_subsample_run_is_deterministic_and_reports_exact_cpu_errors() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(ROWS, 7, false);
    let test = sample(2_000, 8, false);
    let cfg = config(13, FIT_ROWS);
    let a = search::run(&cfg, &train, &test);
    let b = search::run(&cfg, &train, &test);
    assert_eq!(a.backend, rcsr::backend::Kind::Gpu);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(a.test_errors, b.test_errors);
    assert_eq!(
        a.gpu.as_ref().unwrap().fit_subsample_hash,
        b.gpu.as_ref().unwrap().fit_subsample_hash
    );
    for s in &a.archive.items {
        let cpu = search::eval_error(&s.tree, &train, Metric::Squared);
        assert_eq!(
            s.error.to_bits(),
            cpu.to_bits(),
            "archive row is not the exact CPU score on all rows: {} vs {}",
            s.error,
            cpu
        );
    }
}

#[test]
fn the_manifest_records_fit_rows_and_the_subsample_hash() {
    let train = sample(ROWS, 5, false);
    let cfg = config(11, FIT_ROWS);
    let record = search::fit_subsample_record(&cfg, &train).expect("a subsample");
    assert_eq!(record.0, FIT_ROWS);
    let again = search::fit_subsample_record(&cfg, &train).expect("a subsample");
    assert_eq!(record, again, "the draw is deterministic");
    let other = search::fit_subsample_record(&config(12, FIT_ROWS), &train).unwrap();
    assert_ne!(record.1, other.1, "a different seed draws different rows");
    assert!(search::fit_subsample_record(&config(11, 0), &train).is_none());
    assert!(search::fit_subsample_record(&config(11, ROWS), &train).is_none());

    let fp = rcsr::checkpoint::fingerprint(&train);
    let mut m = rcsr::manifest::collect(
        "test", "-", fp.clone(), fp, &cfg, "gpu", None, None, None, None, None,
    );
    m.fit_subsample = Some(record);
    let json = m.json(None, &[]);
    assert!(
        json.contains(&format!(
            "\"fit_subsample\":{{\"fit_rows\":{},\"hash\":\"{:016x}\"",
            FIT_ROWS, record.1
        )),
        "{json}"
    );
    m.fit_subsample = None;
    assert!(!m.json(None, &[]).contains("fit_subsample"));
}
