//! #69: the k-NN null baseline is exact below its cap and bounded above it.

use rcsr::data::Sample;
use rcsr::diagnostics::{Baselines, KNN_REFERENCE_CAP};
use rcsr::metrics::Metric;

/// Deterministic synthetic sample with `n` rows and three inputs.
fn sample(n: usize, offset: u64) -> Sample {
    let mut state = 0x9e37_79b9_7f4a_7c15u64 ^ offset;
    let mut next = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64) / ((1u64 << 53) as f64)
    };
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let a = next() * 4.0 - 2.0;
        let b = next() * 4.0 - 2.0;
        let c = next() * 4.0 - 2.0;
        x.push(vec![a, b, c]);
        y.push(1.5 * a - 0.5 * b + 0.25 * c + next() * 0.1);
    }
    Sample::new(x, y)
}

fn assert_bit_identical(a: &Baselines, b: &Baselines) {
    for (name, x, y) in [
        ("mean.train", a.mean.train_error, b.mean.train_error),
        ("mean.test", a.mean.test_error, b.mean.test_error),
        ("linear.train", a.linear.train_error, b.linear.train_error),
        ("linear.test", a.linear.test_error, b.linear.test_error),
        ("knn.train", a.knn.train_error, b.knn.train_error),
        ("knn.test", a.knn.test_error, b.knn.test_error),
    ] {
        assert_eq!(
            x.to_bits(),
            y.to_bits(),
            "{name} changed: {x:?} vs {y:?}"
        );
    }
}

#[test]
fn small_data_below_the_cap_is_bit_identical() {
    let train = sample(60, 1);
    let test = vec![sample(20, 2)];

    // `0` is the pre-#69 exact path; 60 is a cap at the training size.
    let exact = Baselines::fit_sampled(&train, &test, Metric::Absolute, 0, 7);
    let capped = Baselines::fit_sampled(&train, &test, Metric::Absolute, 60, 7);

    assert_bit_identical(&exact, &capped);
    assert_eq!(capped.knn_reference_rows, 60);
    assert!(!capped.knn_capped);
    assert!(!capped.knn_json_extra().contains("\"capped\":true"));
}

#[test]
fn reference_set_is_capped_at_20k_and_40k() {
    let cap = 256;
    let a = sample(20_000, 3);
    let b = sample(40_000, 4);

    let small = Baselines::fit_sampled(&a, &[], Metric::Absolute, cap, 11);
    let large = Baselines::fit_sampled(&b, &[], Metric::Absolute, cap, 11);

    assert!(small.knn_capped && large.knn_capped);
    assert_eq!(small.knn_reference_rows, cap);
    assert_eq!(large.knn_reference_rows, cap);
    // The cost is governed by the cap, not by n: both report the same set size.
    assert_eq!(small.knn_reference_rows, large.knn_reference_rows);
    assert!(small.knn.train_error.is_finite());
    assert!(large.knn.train_error.is_finite());

    let json = large.knn_json_extra();
    assert!(json.contains("\"reference_rows\":256"), "{json}");
    assert!(json.contains("\"capped\":true"), "{json}");
}

#[test]
fn zero_samples_reproduces_the_exact_path_above_the_default_cap() {
    let n = 1200;
    assert!(n > 256, "the point is to be above a small cap");
    let train = sample(n, 5);

    let exact = Baselines::fit_sampled(&train, &[], Metric::Absolute, 0, 13);
    let explicit = Baselines::fit_sampled(&train, &[], Metric::Absolute, n, 13);

    assert_bit_identical(&exact, &explicit);
    assert_eq!(exact.knn_reference_rows, n);
    assert!(!exact.knn_capped);
}

#[test]
fn default_cap_is_a_few_thousand() {
    // A change here is a documented behaviour change, not an accident.
    assert_eq!(KNN_REFERENCE_CAP, 4096);
}
