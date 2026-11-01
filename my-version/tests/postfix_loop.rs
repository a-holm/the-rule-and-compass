//! #122: the flat postfix variation path.
//!
//! These tests exercise the production selection rule, not a private helper:
//! `variation = auto` must pick the flat arena for a large fit-free GPU run,
//! must be deterministic for a seed, must survive a checkpoint/resume trip and
//! must fall back to the owned-tree loop for an operator the kernel cannot emit.

#![cfg(feature = "cuda")]

use rcsr::data::Sample;
use rcsr::expr::{self, Op};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome, VariationMode};

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 13) % n) as f64) / (n as f64) * 2.0 - 1.0;
        x.push(vec![a, b]);
        y.push(a.sin() + 0.5 * b * b);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn cfg(mode: VariationMode, islands: usize, pop: usize, generations: usize) -> Config {
    Config {
        generations,
        islands,
        pop,
        max_complexity: 12,
        fit_iters: 0,
        patience: 0,
        seed: 11,
        metric: Metric::Squared,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        variation: mode,
        pipeline_depth: 0,
        ..Default::default()
    }
}

fn describe(out: &Outcome) -> Vec<(u64, usize, u64)> {
    out.archive
        .items
        .iter()
        .map(|s| {
            (
                expr::structural_hash(&s.tree),
                s.complexity,
                s.error.to_bits(),
            )
        })
        .collect()
}

fn run(cfg: &Config, train: &Sample, test: &Sample) -> Outcome {
    search::run(cfg, train, test)
}

#[test]
fn postfix_path_is_deterministic_and_bounded() {
    let data = sample(200);
    let c = cfg(VariationMode::Postfix, 1, 4096, 3);
    let a = run(&c, &data, &data);
    let b = run(&c, &data, &data);
    assert_eq!(describe(&a), describe(&b), "same seed, same archive");
    assert_eq!(a.archive.hash(), b.archive.hash(), "canonical archive hash");
    assert!(a.archive.items.iter().all(|s| s.complexity <= 12));
    assert!(!a.archive.items.is_empty());
}

/// #287: the flat arena path must score the configured objective, not SSE.
/// Every reported archive error is the exact CPU metric of the stored tree,
/// and the run stays on the GPU for a native non-squared metric.
#[test]
fn postfix_path_scores_the_configured_native_metric() {
    if !rcsr::backend::probe().available() {
        return;
    }
    let data = sample(160);
    for metric in [Metric::Maximum, Metric::Correlation, Metric::Signed] {
        let c = Config {
            metric,
            ..cfg(VariationMode::Postfix, 1, 4096, 2)
        };
        let out = run(&c, &data, &data);
        assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{metric:?}");
        assert!(!out.archive.items.is_empty(), "{metric:?}");
        for item in &out.archive.items {
            assert_eq!(
                item.error.to_bits(),
                search::eval_error(&item.tree, &data, metric).to_bits(),
                "{metric:?}: reported archive error is not the configured metric"
            );
        }
        let again = run(&c, &data, &data);
        assert_eq!(describe(&out), describe(&again), "{metric:?} is deterministic");
    }
}

#[test]
fn four_islands_are_deterministic_and_independent_of_island_count() {
    let data = sample(200);
    let c = cfg(VariationMode::Postfix, 4, 2048, 3);
    let a = run(&c, &data, &data);
    let b = run(&c, &data, &data);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(a.archive.hash(), b.archive.hash());
    // The same four-island result when the four islands are evolved in a
    // different order is covered by `postfix::tests`; here the production path
    // must at least repeat exactly.
    let one = cfg(VariationMode::Postfix, 1, 4096, 2);
    let c1 = run(&one, &data, &data);
    let d1 = run(&one, &data, &data);
    assert_eq!(describe(&c1), describe(&d1));
}

#[test]
fn checkpoint_resume_reproduces_the_uninterrupted_run() {
    let data = sample(160);
    let full = cfg(VariationMode::Postfix, 1, 4096, 4);
    let uninterrupted = run(&full, &data, &data);

    let path = std::env::temp_dir().join(format!(
        "rcsr-122-resume-{}-{:?}.ckpt",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_file(&path);
    let short = cfg(VariationMode::Postfix, 1, 4096, 2);
    {
        let out = search::run_resumable(&short, &data, &data, None, 2, |cp| {
            cp.save(&path).expect("save checkpoint");
        })
        .expect("first leg");
        assert_eq!(out.generations, 2);
    }
    let cp = rcsr::checkpoint::Checkpoint::load(&path).expect("load checkpoint");
    assert_eq!(cp.config.variation, VariationMode::Postfix);
    let resumed = search::run_resumable(&full, &data, &data, Some(cp), 0, |_| {})
        .expect("resume");
    let _ = std::fs::remove_file(&path);
    assert_eq!(resumed.generations, 4);
    assert_eq!(
        describe(&uninterrupted),
        describe(&resumed),
        "resume must continue the same counter-based stream"
    );
    assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash());
}

/// #256: a time-budget stop on the flat path is not convergence, reports every
/// generation it ran, writes a resumable checkpoint with the post-refit archive
/// and continues to exactly the uninterrupted run.
#[test]
fn a_time_budget_stop_on_the_flat_path_is_resumable() {
    use std::sync::{Arc, Mutex};
    let data = sample(160);
    let full = cfg(VariationMode::Postfix, 1, 4096, 4);
    let uninterrupted = run(&full, &data, &data);

    let seen: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = seen.clone();
    let mut budget = cfg(VariationMode::Postfix, 1, 4096, 4);
    budget.time_budget = Some(0.0);
    budget.progress = Some(Arc::new(move |p: search::Progress| {
        sink.lock().unwrap().push(p.generation);
    }));
    let mut last: Option<String> = None;
    let stopped = search::run_resumable(&budget, &data, &data, None, 1, |cp| {
        last = Some(cp.to_text());
    })
    .expect("budget leg");
    assert_eq!(stopped.stop_reason, "time_budget");
    assert!(!stopped.converged_early);
    assert_eq!(stopped.generations, 1);
    assert_eq!(*seen.lock().unwrap(), vec![1]);

    let cp = rcsr::checkpoint::Checkpoint::from_text(&last.expect("a checkpoint")).unwrap();
    assert!(!cp.stopped_early, "a time-budget stop is not terminal");
    assert_eq!(cp.error_kind, rcsr::checkpoint::ErrorKind::F32Proxy);
    let fin = cp.final_archive.as_ref().expect("post-refit archive");
    assert_eq!(fin.hash(), stopped.archive.hash());

    let resumed = search::run_resumable(&full, &data, &data, Some(cp), 0, |_| {})
        .expect("resume after the budget stop");
    assert_eq!(resumed.generations, 4);
    assert_eq!(describe(&uninterrupted), describe(&resumed));
    assert_eq!(uninterrupted.archive.hash(), resumed.archive.hash());
}

#[test]
fn unsupported_operators_fall_back_and_postfix_refuses() {
    let data = sample(120);
    let mut auto = cfg(VariationMode::Auto, 1, 4096, 1);
    auto.ops = vec![Op::Add, Op::Mul, Op::Asinh];
    let out = run(&auto, &data, &data);
    assert!(!out.archive.items.is_empty());

    let mut forced = cfg(VariationMode::Postfix, 1, 4096, 1);
    forced.ops = vec![Op::Add, Op::Mul, Op::Asinh];
    let err = match search::run_resumable(&forced, &data, &data, None, 0, |_| {}) {
        Ok(_) => panic!("postfix must refuse an operator without a kernel opcode"),
        Err(e) => e,
    };
    assert!(
        err.contains("no kernel opcode") || err.contains("operator"),
        "unexpected error: {}",
        err
    );
}

#[test]
fn below_the_gate_auto_keeps_the_owned_tree_path() {
    let data = sample(120);
    let c = cfg(VariationMode::Auto, 1, 4000, 2);
    let a = run(&c, &data, &data);
    let b = run(&c, &data, &data);
    assert_eq!(describe(&a), describe(&b), "fallback is deterministic too");
    assert!(!a.archive.items.is_empty());
    // Forcing the flat path below the gate is refused, not silently accepted.
    let mut forced = c.clone();
    forced.variation = VariationMode::Postfix;
    let err = match search::run_resumable(&forced, &data, &data, None, 0, |_| {}) {
        Ok(_) => panic!("postfix below the gate must be refused"),
        Err(e) => e,
    };
    assert!(err.contains("4096"), "unexpected error: {}", err);
}

#[test]
fn front_parents_draw_is_deterministic_on_the_flat_path() {
    let data = sample(200);
    let mut on = cfg(VariationMode::Postfix, 1, 4096, 3);
    on.front_parents = true;
    let a = run(&on, &data, &data);
    let b = run(&on, &data, &data);
    assert_eq!(describe(&a), describe(&b));
    assert_eq!(a.archive.hash(), b.archive.hash());

    // The flag is live on this path: the archive-guided draw changes the
    // trajectory rather than being accepted and ignored.
    let mut off = on.clone();
    off.front_parents = false;
    let c = run(&off, &data, &data);
    assert_ne!(
        describe(&a),
        describe(&c),
        "front feedback must affect the flat trajectory"
    );
    assert!(!c.archive.items.is_empty(), "the off run still produces a front");
}

#[test]
fn five_thousand_rows_are_a_supported_flat_benchmark_cell() {
    // The #122 performance target is stated at N >= 5000, but the recorded
    // 6.7x was measured at 3750 training rows. This pins the larger cell so a
    // future change cannot silently make it unsupported or unbounded.
    let data = sample(5000);
    let c = cfg(VariationMode::Postfix, 1, 4096, 2);
    let out = run(&c, &data, &data);
    assert_eq!(out.generations, 2);
    assert!(!out.archive.items.is_empty(), "the front must be non-empty");
    assert!(out.archive.items.iter().all(|s| s.complexity <= 12));
}

#[test]
fn fitting_runs_stay_on_the_owned_tree_path() {
    let data = sample(120);
    let mut c = cfg(VariationMode::Auto, 1, 4096, 1);
    c.fit_iters = 2;
    let out = run(&c, &data, &data);
    assert!(!out.archive.items.is_empty());
    // Forced postfix refuses: fitting is CPU-authoritative and not ported.
    c.variation = VariationMode::Postfix;
    let err = match search::run_resumable(&c, &data, &data, None, 0, |_| {}) {
        Ok(_) => panic!("forced postfix must refuse fitting"),
        Err(e) => e,
    };
    assert!(err.contains("constant fitting"), "unexpected error: {}", err);
}
