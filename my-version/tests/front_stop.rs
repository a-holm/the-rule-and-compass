//! Front-stability stop and scale-free patience (issue #464).

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::search::{self, Config};

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-2;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn config(generations: usize) -> Config {
    Config {
        generations,
        islands: 2,
        pop: 30,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 5,
        gpu: false,
        stop_front: Some(3),
        patience_tol: Some(1e-2),
        ..Default::default()
    }
}

fn checkpoint_text(cfg: &Config, train: &Sample, test: &Sample) -> String {
    let mut text: Option<String> = None;
    search::run_resumable(cfg, train, test, None, 0, |cp| text = Some(cp.to_text())).unwrap();
    text.expect("a checkpoint")
}

#[test]
fn front_stop_ends_a_noisy_run_early() {
    let (train, test) = (sample(120), sample(40));
    let out = search::run(&config(120), &train, &test);
    assert_eq!(out.stop_reason, "front_stable");
    assert!(out.generations < 120, "ran {}", out.generations);
    let off = Config { stop_front: None, patience_tol: None, ..config(out.generations) };
    let base = search::run(&off, &train, &test);
    assert_eq!(base.stop_reason, "generations");
    let a: Vec<u64> = out.archive.items.iter().map(|s| s.error.to_bits()).collect();
    let b: Vec<u64> = base.archive.items.iter().map(|s| s.error.to_bits()).collect();
    assert_eq!(a, b, "the stop does not change the trajectory up to the stop");
}

#[test]
fn front_stop_state_survives_a_checkpoint() {
    let (train, test) = (sample(120), sample(40));
    let full = config(120);
    let uninterrupted = search::run(&full, &train, &test);
    assert_eq!(uninterrupted.stop_reason, "front_stable");

    let first = config(uninterrupted.generations - 2);
    let text = checkpoint_text(&first, &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse");
    assert!(saved.front_stable > 0, "the counter is part of the checkpoint");
    assert!(text.contains("config stop_front 3"));
    assert_eq!(Checkpoint::from_text(&saved.to_text()).unwrap().to_text(), text);

    let resumed = search::run_resumable(&full, &train, &test, Some(saved), 0, |_| {}).unwrap();
    assert_eq!(resumed.generations, uninterrupted.generations);
    assert_eq!(resumed.stop_reason, "front_stable");
    let a: Vec<u64> = uninterrupted.archive.items.iter().map(|s| s.error.to_bits()).collect();
    let b: Vec<u64> = resumed.archive.items.iter().map(|s| s.error.to_bits()).collect();
    assert_eq!(a, b);
}

#[test]
fn defaults_write_no_new_checkpoint_records() {
    let (train, test) = (sample(60), sample(20));
    let cfg = Config { stop_front: None, patience_tol: None, ..config(3) };
    let text = checkpoint_text(&cfg, &train, &test);
    assert!(!text.contains("patience_tol") && !text.contains("stop_front"));
    assert!(!text.contains("front_stable"));
}

#[test]
fn a_resume_with_a_different_stop_setting_is_refused() {
    let (train, test) = (sample(60), sample(20));
    let text = checkpoint_text(&config(3), &train, &test);
    let saved = Checkpoint::from_text(&text).expect("parse");
    let other = Config { stop_front: Some(4), ..config(6) };
    let err = match search::run_resumable(&other, &train, &test, Some(saved), 0, |_| {}) {
        Ok(_) => panic!("expected a refusal"),
        Err(e) => e,
    };
    assert!(err.contains("stop_front"), "{}", err);
}
