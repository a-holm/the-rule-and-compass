//! Integration tests for the known-truth benchmark harness.
//!
//! These drive the same public API the `rcsr-bench` binary uses, so they
//! exercise the suite from outside the crate.

use rcsr::bench::{self, SuiteConfig};

#[test]
fn suite_has_at_least_twenty_known_truth_problems() {
    assert!(
        bench::known_truth().len() >= 20,
        "the suite must contain at least 20 problems"
    );
}

#[test]
fn tiny_suite_runs_end_to_end_and_reports_a_recovery_rate() {
    let cfg = SuiteConfig {
        noise: vec![0.0, 0.1],
        generations: 30,
        islands: 1,
        pop: 80,
        seed: 1,
        tiny: true,
        ..Default::default()
    };
    let report = bench::run_suite(&cfg);

    // One run per (tiny problem, noise level), and a poolable summary.
    assert!(report.problems >= 6, "problems = {}", report.problems);
    assert_eq!(report.results.len(), report.problems * cfg.noise.len());
    assert_eq!(report.summary.runs, report.results.len());
    assert!(
        (0.0..=1.0).contains(&report.summary.rate),
        "rate = {}",
        report.summary.rate
    );
    assert_eq!(report.by_noise.len(), cfg.noise.len());

    // On noiseless data the easiest law must be found.
    assert!(
        report
            .results
            .iter()
            .any(|r| r.problem == "line" && r.noise == 0.0 && r.recovered),
        "the line problem was not recovered on noiseless data"
    );

    // Both renderings carry the summary and the criterion.
    let text = bench::report_text(&report, &cfg);
    assert!(text.contains("== summary =="));
    assert!(text.contains("recovered"));
    let json = bench::report_json(&report, &cfg);
    assert!(json.contains("\"summary\""));
    assert!(json.contains("\"median_complexity\""));
    assert!(json.contains("\"recovery_threshold\""));
}
