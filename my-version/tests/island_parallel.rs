//! #463: fitting and scoring offspring on a worker pool must not change the
//! search. Every thread count gives the same archive, populations and RNG
//! states.

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
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

fn config(islands: usize, pop: usize, generations: usize, threads: usize) -> Config {
    Config {
        generations,
        islands,
        pop,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed: 5,
        threads,
        ..Default::default()
    }
}

/// The final checkpoint text: archive, per-island error bits and RNG states.
fn final_state(cfg: &Config, train: &Sample, test: &Sample) -> String {
    let mut text: Option<String> = None;
    search::run_resumable(cfg, train, test, None, 1, |cp| text = Some(cp.to_text()))
        .expect("run");
    text.expect("a checkpoint")
}

#[test]
fn offspring_are_identical_for_one_and_many_workers() {
    let train = sample(400);
    let test = sample(100);
    for islands in [1, 4] {
        let base = final_state(&config(islands, 80, 5, 1), &train, &test);
        for threads in [2, 8] {
            let got = final_state(&config(islands, 80, 5, threads), &train, &test);
            assert!(base == got, "islands {islands}: threads {threads} changed the state");
        }
    }
}

#[test]
fn parallel_fit_is_bit_identical_on_a_heavy_table() {
    // Above the row threshold a single candidate already gets its own worker.
    let train = sample(3000);
    let test = sample(100);
    let base = final_state(&config(1, 12, 3, 1), &train, &test);
    let got = final_state(&config(1, 12, 3, 8), &train, &test);
    assert!(base == got, "threads 8 changed the state on 3000 rows");
}

#[test]
fn parallel_fit_is_bit_identical_with_the_qd_policy() {
    let train = sample(400);
    let test = sample(100);
    let qd = |threads| Config {
        qd_bins: 8,
        qd_policy: true,
        ..config(2, 80, 5, threads)
    };
    let base = final_state(&qd(1), &train, &test);
    let got = final_state(&qd(8), &train, &test);
    assert!(base == got, "threads 8 changed the state under the QD policy");
}

#[test]
fn a_checkpoint_from_eight_threads_resumes_on_one_to_the_same_archive() {
    let train = sample(400);
    let test = sample(100);
    let full = search::run(&config(2, 60, 10, 1), &train, &test);

    let text = final_state(&config(2, 60, 5, 8), &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse");
    let resumed =
        search::run_resumable(&config(2, 60, 10, 1), &train, &test, Some(saved), 0, |_| {})
            .expect("resume");

    let bits = |o: &search::Outcome| -> Vec<(u64, u64)> {
        o.archive
            .items
            .iter()
            .map(|s| (s.error.to_bits(), rcsr::expr::structural_hash(&s.tree)))
            .collect()
    };
    assert_eq!(bits(&full), bits(&resumed));
}
