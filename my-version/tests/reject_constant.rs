//! #477: `reject_constant_offspring` keeps at most one input-free tree per
//! island, leaves the default trajectory alone and does not depend on the
//! thread count.

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::expr::row_dependent;
use rcsr::search::{self, Config};

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b * a + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn config(reject: bool, threads: usize) -> Config {
    Config {
        generations: 30,
        islands: 1,
        pop: 60,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed: 5,
        threads,
        gpu: false,
        reject_constant_offspring: reject,
        ..Default::default()
    }
}

/// Input-free trees per island after every completed generation.
fn constants_per_generation(cfg: &Config) -> Vec<usize> {
    let (train, test) = (sample(200), sample(50));
    let mut counts = Vec::new();
    search::run_resumable(cfg, &train, &test, None, 1, |cp| {
        if cp.generations >= 1 {
            let worst = cp
                .islands
                .iter()
                .map(|isl| isl.population.iter().filter(|i| !row_dependent(&i.tree)).count())
                .max()
                .unwrap();
            counts.push(worst);
        }
    })
    .expect("run");
    counts
}

fn final_text(cfg: &Config) -> String {
    let (train, test) = (sample(200), sample(50));
    let mut text = None;
    search::run_resumable(cfg, &train, &test, None, 1, |cp| text = Some(cp.to_text()))
        .expect("run");
    text.expect("a checkpoint")
}

#[test]
fn an_island_holds_at_most_one_constant_individual_with_the_flag() {
    let counts = constants_per_generation(&config(true, 1));
    assert!(counts.len() >= 30, "one checkpoint per generation, got {}", counts.len());
    assert!(counts.iter().all(|&c| c <= 1), "constants per generation: {counts:?}");
}

#[test]
fn without_the_flag_islands_hold_several_constant_individuals() {
    let counts = constants_per_generation(&config(false, 1));
    assert!(counts.iter().any(|&c| c > 1), "constants per generation: {counts:?}");
}

#[test]
fn the_default_archive_hash_is_pinned() {
    let (train, test) = (sample(200), sample(50));
    let out = search::run(&config(false, 1), &train, &test);
    assert_eq!(out.archive.hash(), 1243343133610422360, "pinned archive hash of the default trajectory");
}

#[test]
fn the_flag_result_is_identical_across_thread_counts() {
    let base = final_text(&config(true, 1));
    for threads in [2, 8] {
        assert!(base == final_text(&config(true, threads)), "threads {threads} changed the state");
    }
}

#[test]
fn the_setting_survives_a_checkpoint_round_trip() {
    let text = final_text(&config(true, 1));
    assert!(text.contains("config reject_constant_offspring true"));
    let cp = Checkpoint::from_text(&text).expect("parse");
    assert!(cp.config.reject_constant_offspring);
    let off = final_text(&config(false, 1));
    assert!(!off.contains("reject_constant_offspring"), "an off run writes the old checkpoint");
}

#[test]
fn qd_immigrants_and_restarts_honour_the_flag() {
    let qd = Config { qd_bins: 8, qd_policy: true, restart_interval: 3, generations: 40, ..config(true, 1) };
    let counts = constants_per_generation(&qd);
    assert!(counts.len() >= 40, "one checkpoint per generation, got {}", counts.len());
    assert!(counts.iter().all(|&c| c <= 1), "constants per generation with QD: {counts:?}");
    let off = Config { reject_constant_offspring: false, ..qd };
    let counts = constants_per_generation(&off);
    assert!(counts.iter().any(|&c| c > 1), "control without the flag: {counts:?}");
}
