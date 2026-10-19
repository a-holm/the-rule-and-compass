//! Convergence statistics over the Pareto front (issue #18).
//!
//! The definitions are exercised on hand-built front histories with known
//! answers, then end-to-end on one easy and one hard problem. The last test
//! pins the flag gating: `--progress` never changes the default (text) output.

use rcsr::data::Sample;
use rcsr::expr::{self, input, node, Op};
use rcsr::metrics::Metric;
use rcsr::pareto::{Archive, Sol};
use rcsr::progress_stats::{
    Convergence, ProgressStats, CONVERGED_THRESHOLD, DEFAULT_WINDOW,
};
use rcsr::search::{self, Config};
use std::process::Command;

fn sol(tree: expr::Node) -> Sol {
    Sol {
        tree,
        error: 1.0,
        complexity: 1,
        cost: 1.0,
    }
}

fn front(trees: Vec<expr::Node>) -> Archive {
    Archive {
        items: trees.into_iter().map(sol).collect(),
        cap: 100,
    }
}

fn x() -> expr::Node {
    input(0)
}

fn sin_x() -> expr::Node {
    node(Op::Sin, vec![input(0)])
}

fn cos_x() -> expr::Node {
    node(Op::Cos, vec![input(0)])
}

fn exp_x() -> expr::Node {
    node(Op::Exp, vec![input(0)])
}

#[test]
fn confidence_is_high_when_the_best_error_is_flat() {
    let mut stats = ProgressStats::new(2);
    let f = front(vec![x()]);

    // Generation 1 has no span yet: confidence is deliberately 0, not 1.
    assert_eq!(stats.observe(1, 100.0, &f).confidence, 0.0);

    // 100 -> 50 over the next generation is a huge relative improvement.
    let improving = stats.observe(2, 50.0, &f).confidence;
    assert!(improving < 1e-10, "improving confidence = {improving}");

    // From generation 3 the reference is generation 1 (window 2): still
    // 100 -> 50, so still no confidence.
    assert!(stats.observe(3, 50.0, &f).confidence < 1e-10);

    // Once the window contains only 50 -> 50, confidence is exactly 1.
    let flat = stats.observe(4, 50.0, &f).confidence;
    assert!((flat - 1.0).abs() < 1e-12, "flat confidence = {flat}");

    // A fresh tracker gives the same answer for a small improvement mapped
    // through the documented scale: r = 0.001 -> e^-1.
    let mut tiny = ProgressStats::new(1);
    tiny.observe(1, 1.0, &f);
    let c = tiny.observe(2, 0.998, &f).confidence;
    assert!((c - (-1.0f64).exp()).abs() < 1e-6, "tiny confidence = {c}");

    // A solved front (at or below the search's 1e-24 floor) is converged even
    // if the last window contained the jump to the exact answer.
    let mut solved = ProgressStats::new(2);
    solved.observe(1, 1.0, &f);
    let c = solved.observe(2, 1e-30, &f).confidence;
    assert_eq!(c, 1.0, "solved confidence = {c}");
}

#[test]
fn stability_is_the_mean_jaccard_overlap_of_structural_forms() {
    let mut stats = ProgressStats::new(1);
    let both = front(vec![x(), sin_x()]);

    // One snapshot tells us nothing yet.
    assert_eq!(stats.observe(1, 1.0, &both).stability, 0.0);
    // The same two forms one generation later: perfect survival.
    assert!((stats.observe(2, 1.0, &both).stability - 1.0).abs() < 1e-12);
    // A completely different front: no survival.
    let other = front(vec![cos_x(), exp_x()]);
    assert_eq!(stats.observe(3, 1.0, &other).stability, 0.0);
    // And back again: the window (w = 1) only looks at the last comparison.
    assert!((stats.observe(4, 1.0, &other).stability - 1.0).abs() < 1e-12);

    // Partial overlap: |{x, sin} ∩ {x, cos}| / |{x, sin, cos}| = 1/3.
    let mut partial = ProgressStats::new(1);
    partial.observe(1, 1.0, &front(vec![x(), sin_x()]));
    let s = partial
        .observe(2, 1.0, &front(vec![x(), cos_x()]))
        .stability;
    assert!((s - 1.0 / 3.0).abs() < 1e-12, "partial stability = {s}");
}

#[test]
fn maturity_is_the_normalised_age_of_the_oldest_surviving_form() {
    let mut stats = ProgressStats::new(3);
    let one = front(vec![x()]);

    assert_eq!(stats.observe(1, 1.0, &one).maturity, 0.0);
    assert!((stats.observe(5, 1.0, &one).maturity - 0.8).abs() < 1e-12);

    // Replacing the only form restarts the clock at the current generation.
    assert_eq!(stats.observe(6, 1.0, &front(vec![sin_x()])).maturity, 0.0);
    // ... and then ages from generation 6.
    let m = stats.observe(7, 1.0, &front(vec![sin_x()])).maturity;
    assert!((m - 1.0 / 7.0).abs() < 1e-12, "maturity = {m}");

    // An empty front has nothing to mature.
    assert_eq!(stats.observe(8, 1.0, &front(vec![])).maturity, 0.0);
}

#[test]
fn percent_converged_uses_the_documented_weights_and_threshold() {
    let c = Convergence::from_parts(0.5, 1.0, 0.5);
    // 0.5*0.5 + 0.3*1.0 + 0.2*0.5 = 0.65
    assert!((c.percent_converged - 0.65).abs() < 1e-12);
    assert!(!c.is_converged());

    let all = Convergence::from_parts(1.0, 1.0, 1.0);
    assert!((all.percent_converged - 1.0).abs() < 1e-12);
    assert!(all.is_converged());

    let none = Convergence::from_parts(0.0, 0.0, 0.0);
    assert_eq!(none.percent_converged, 0.0);
    assert!(!none.is_converged());

    // Values outside 0..1 are clamped, so the headline stays bounded.
    let clamped = Convergence::from_parts(-3.0, 4.0, f64::NAN);
    assert_eq!(clamped.confidence, 0.0);
    assert_eq!(clamped.stability, 1.0);
    assert_eq!(clamped.maturity, 0.0);
    assert!((clamped.percent_converged - 0.3).abs() < 1e-12);

    assert_eq!(DEFAULT_WINDOW, 20);
    assert_eq!(CONVERGED_THRESHOLD, 0.75);
}

/// Centre and scale the inputs the way `rcsr`'s default preparation does.
/// The search itself expects this; the target stays in its original units.
fn standardize_x(s: &Sample) -> Sample {
    let n = s.len() as f64;
    let d = s.x.first().map(|r| r.len()).unwrap_or(0);
    let mut means = vec![0.0; d];
    for row in &s.x {
        for (j, v) in row.iter().enumerate() {
            means[j] += *v;
        }
    }
    for m in &mut means {
        *m /= n;
    }
    let mut stds = vec![0.0; d];
    for row in &s.x {
        for (j, v) in row.iter().enumerate() {
            stds[j] += (*v - means[j]).powi(2);
        }
    }
    for s in &mut stds {
        *s = (*s / n).sqrt();
        if *s < 1e-12 {
            *s = 1.0;
        }
    }
    Sample {
        x: s.x
            .iter()
            .map(|row| {
                row.iter()
                    .enumerate()
                    .map(|(j, v)| (v - means[j]) / stds[j])
                    .collect()
            })
            .collect(),
        y: s.y.clone(),
        w: s.w.clone(),
        ys: Vec::new(),
    }
}

/// `y = 2x + 1` on a small grid.
fn easy_sample() -> Sample {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..40 {
        let a = -1.0 + 2.0 * i as f64 / 39.0;
        x.push(vec![a]);
        y.push(2.0 * a + 1.0);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
}

/// Noise plus a fast oscillation: no short closed form, and the front keeps
/// moving. The noise is deterministic so the test is reproducible.
fn hard_sample() -> Sample {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..80 {
        let a = -3.0 + 6.0 * i as f64 / 79.0;
        let noise = (((i * 37 + 11) % 17) as f64 - 8.0) / 10.0;
        x.push(vec![a]);
        y.push((2.0 * a).sin() + 0.4 * (13.0 * a).sin() + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
}

fn config(seed: u32, max_complexity: usize) -> Config {
    Config {
        generations: 150,
        islands: 1,
        pop: 120,
        max_complexity,
        seed,
        metric: Metric::Squared,
        ..Config::default()
    }
}

#[test]
fn an_easy_problem_reaches_the_converged_range() {
    let train = standardize_x(&easy_sample());
    let test = train.clone();
    let outcome = search::run(&config(1, 30), &train, &test);

    let c = outcome.convergence;
    assert!(
        c.is_converged(),
        "easy convergence = {:?} (best error {})",
        c,
        outcome
            .archive
            .sorted()
            .first()
            .map(|s| s.error)
            .unwrap_or(f64::NAN)
    );
    assert!(c.confidence > 0.9, "easy confidence = {}", c.confidence);
    // Structural stability is *not* high here, and that is correct: once every
    // mutation changes the tree (#77), a solved problem keeps discovering
    // alternative forms with the same error, so the front churns. Convergence
    // is asserted through `is_converged`/confidence; see issue #84.
    assert!(c.stability > 0.1, "easy stability = {}", c.stability);
    // An exact solve can stop after two generations (the search's own solved
    // floor), so maturity is only 0.5 — confidence and stability carry this
    // case, which is why confidence treats a solved front as 1.
    assert!(c.maturity > 0.4, "easy maturity = {}", c.maturity);
}

#[test]
fn a_hard_problem_does_not_reach_the_converged_range() {
    let train = standardize_x(&hard_sample());
    let test = train.clone();
    // A *short* budget: convergence measures whether the front is still moving,
    // and a long run on this fixture eventually stalls (stability ~1.0) — which
    // is convergence of the search, however poor the error. The test needs a
    // run that is still improving, so it uses a fraction of the budget.
    let mut cfg = config(2, 80);
    cfg.generations = 25;
    let outcome = search::run(&cfg, &train, &test);

    let c = outcome.convergence;
    assert!(
        !c.is_converged(),
        "hard convergence = {:?} must stay below {}",
        c,
        CONVERGED_THRESHOLD
    );
    // The hard run keeps improving, so confidence stays low even though some
    // old forms survive.
    assert!(c.confidence < 0.75, "hard confidence = {}", c.confidence);
}

#[test]
fn progress_is_flag_gated_and_does_not_change_default_output() {
    let path = std::env::temp_dir().join(format!(
        "rcsr-convergence-{}-{}.csv",
        std::process::id(),
        std::thread::current().name().unwrap_or("main")
    ));
    let mut csv = String::from("x,y\n");
    for i in 0..24 {
        let a = -1.0 + 2.0 * i as f64 / 23.0;
        csv.push_str(&format!("{:.8},{:.8}\n", a, 3.0 * a - 1.0));
    }
    std::fs::write(&path, csv).expect("write fixture");

    let run = |extra: &[&str]| {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr"));
        cmd.arg("--data")
            .arg(&path)
            .args(["--target", "y", "--vars", "x"])
            .args(["--generations", "8", "--pop", "40", "--islands", "1"])
            .args(["--seed", "4", "--metric", "squared"])
            .args(extra);
        cmd.output().expect("run rcsr")
    };

    let plain = run(&[]);
    let with_progress = run(&["--progress"]);
    assert!(plain.status.success(), "plain run failed");
    assert!(with_progress.status.success(), "progress run failed");

    // The default (text) output is the Pareto rows on stdout, and is identical
    // with and without --progress.
    assert_eq!(
        plain.stdout, with_progress.stdout,
        "default stdout changed when --progress was on"
    );
    let plain_text = String::from_utf8_lossy(&plain.stdout);
    assert!(
        !plain_text.contains("convergence") && !plain_text.contains("\"event\":\"progress\""),
        "default output leaked progress/convergence"
    );

    // --progress emits the four numbers, nested under "convergence".
    let progress_text = String::from_utf8_lossy(&with_progress.stderr);
    assert!(
        progress_text.contains("\"event\":\"progress\""),
        "no progress event: {progress_text}"
    );
    assert!(
        progress_text.contains("\"convergence\":{\"confidence\":"),
        "progress event is missing convergence: {progress_text}"
    );
    for key in ["confidence", "stability", "maturity", "percent_converged"] {
        assert!(
            progress_text.contains(&format!("\"{}\":", key)),
            "progress event is missing {key}"
        );
    }

    // The final --json object always carries the same convergence object, even
    // without --progress (machine-readable output is opt-in already).
    let json = run(&["--json"]);
    assert!(json.status.success(), "json run failed");
    let json_text = String::from_utf8_lossy(&json.stdout);
    assert!(
        json_text.contains("\"convergence\":{\"confidence\":"),
        "final JSON is missing convergence: {json_text}"
    );
    for key in ["confidence", "stability", "maturity", "percent_converged"] {
        assert!(
            json_text.contains(&format!("\"{}\":", key)),
            "final JSON is missing {key}"
        );
    }

    // NDJSON: every progress event and the done event carry the numbers.
    let ndjson = run(&["--ndjson", "--progress"]);
    assert!(ndjson.status.success(), "ndjson run failed");
    let ndjson_text = String::from_utf8_lossy(&ndjson.stdout);
    let progress_lines: Vec<&str> = ndjson_text
        .lines()
        .filter(|l| l.contains("\"event\":\"progress\""))
        .collect();
    assert!(!progress_lines.is_empty(), "no NDJSON progress events");
    for line in &progress_lines {
        assert!(
            line.contains("\"convergence\":{\"confidence\":"),
            "NDJSON progress event is missing convergence: {line}"
        );
    }
    let done = ndjson_text
        .lines()
        .find(|l| l.contains("\"event\":\"done\""))
        .expect("NDJSON done event");
    assert!(
        done.contains("\"convergence\":{\"confidence\":"),
        "NDJSON done event is missing convergence: {done}"
    );

    let _ = std::fs::remove_file(&path);
}
