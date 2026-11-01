//! #69: the k-NN null baseline keeps an exact reference set and, when query
//! rows are sampled, stays an estimate of the same row-average loss.

use rcsr::data::Sample;
use rcsr::diagnostics::{
    knn_predict, Baselines, DEFAULT_BASELINE_SAMPLES, KNN_KD_MAX_DIM,
};
use rcsr::metrics::{self, Metric};

/// Deterministic synthetic sample with `n` rows and `dims` inputs.
fn sample(n: usize, dims: usize, offset: u64) -> Sample {
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
        let row: Vec<f64> = (0..dims).map(|_| next() * 4.0 - 2.0).collect();
        let target: f64 = row.iter().enumerate().map(|(j, v)| (j as f64 + 1.0) * v).sum();
        x.push(row);
        y.push(target + next() * 0.1);
    }
    Sample::new(x, y)
}

/// The pre-#69 exact algorithm: full reference, full queries, `knn_predict`.
fn brute_force_error(train: &Sample, query: &Sample, metric: Metric) -> f64 {
    let k = 5.min(train.len().max(1));
    let pred: Vec<f64> = query.x.iter().map(|x| knn_predict(train, x, k)).collect();
    metrics::score(&pred, &query.y, metric)
}

fn assert_bits(a: f64, b: f64, what: &str) {
    assert_eq!(a.to_bits(), b.to_bits(), "{what}: {a:?} vs {b:?}");
}

#[test]
fn kd_tree_and_brute_force_are_bit_identical_to_knn_predict() {
    // d = 1/3/6/8: k-d tree path. d = 9: brute-force path.
    for dims in [1, 3, 6, KNN_KD_MAX_DIM, KNN_KD_MAX_DIM + 1] {
        let train = sample(600, dims, 1);
        let query = sample(120, dims, 2);
        let b = Baselines::fit_sampled(&train, &[query.clone()], Metric::Absolute, 0, 7);
        let expected_train = brute_force_error(&train, &train, Metric::Absolute);
        let expected_test = brute_force_error(&train, &query, Metric::Absolute);

        assert_bits(b.knn.train_error, expected_train, "train");
        assert_bits(b.knn.test_error, expected_test, "test");
        assert_eq!(b.knn_reference_rows, train.len());
        assert_eq!(b.knn_train_query_rows, train.len());
        assert_eq!(b.knn_test_query_rows, query.len());
        assert!(!b.knn_capped);
        assert!(b.knn_train_query_sem.is_nan());
        assert!(b.knn_test_query_sem.is_nan());
    }
}

#[test]
fn duplicate_points_and_exact_ties_match_brute_force() {
    // Coordinates from {0, 1, 2} produce many exact distance ties; targets
    // differ so a wrong tie-break changes the mean.
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..400usize {
        x.push(vec![(i % 3) as f64, ((i / 3) % 3) as f64, 0.0]);
        y.push((i % 7) as f64);
    }
    let train = Sample::new(x, y);
    let query = sample(50, 3, 99);
    let b = Baselines::fit_sampled(&train, &[query.clone()], Metric::Absolute, 0, 5);
    assert_bits(
        b.knn.train_error,
        brute_force_error(&train, &train, Metric::Absolute),
        "train ties",
    );
    assert_bits(
        b.knn.test_error,
        brute_force_error(&train, &query, Metric::Absolute),
        "test ties",
    );
}

#[test]
fn reference_set_stays_exact_at_20k_and_40k() {
    let cap = 512;
    let a = sample(20_000, 3, 3);
    let b = sample(40_000, 3, 4);

    let small = Baselines::fit_sampled(&a, &[], Metric::Absolute, cap, 11);
    let large = Baselines::fit_sampled(&b, &[], Metric::Absolute, cap, 11);

    // The reference set is never subsampled; only queries are.
    assert_eq!(small.knn_reference_rows, 20_000);
    assert_eq!(large.knn_reference_rows, 40_000);
    // Cost is governed by the query cap, not by n.
    assert_eq!(small.knn_train_query_rows, cap);
    assert_eq!(large.knn_train_query_rows, cap);
    assert_eq!(small.knn_test_query_rows, 0);
    assert_eq!(large.knn_test_query_rows, 0);
    assert!(small.knn_capped && large.knn_capped);
    assert!(small.knn_train_query_sem.is_finite());
    assert!(large.knn_train_query_sem.is_finite());
    assert!(small.knn.train_error.is_finite());
    assert!(large.knn.train_error.is_finite());
}

#[test]
fn zero_samples_means_no_caps_at_all() {
    let train = sample(1200, 4, 5);
    let fold = sample(300, 4, 6);

    let no_caps = Baselines::fit_sampled(&train, &[fold.clone()], Metric::Absolute, 0, 13);
    let plain = Baselines::fit(&train, &[fold.clone()], Metric::Absolute);

    assert_bits(no_caps.knn.train_error, plain.knn.train_error, "train");
    assert_bits(no_caps.knn.test_error, plain.knn.test_error, "test");
    assert_eq!(no_caps.knn_reference_rows, train.len());
    assert_eq!(no_caps.knn_train_query_rows, train.len());
    assert_eq!(no_caps.knn_test_query_rows, fold.len());
    assert!(!no_caps.knn_capped);
    assert!(no_caps.knn_train_query_sem.is_nan());
    assert!(no_caps.knn_test_query_sem.is_nan());
}

/// #132: the CLI default is bounded on a large training set but exact when the
/// panel is no larger than the cap. `0` still means every row.
#[test]
fn default_query_cap_is_bounded_large_and_exact_small() {
    let large = sample(5_000, 2, 41);
    let b = Baselines::fit_sampled(
        &large,
        &[],
        Metric::Absolute,
        DEFAULT_BASELINE_SAMPLES,
        7,
    );
    assert_eq!(b.knn_reference_rows, large.len());
    assert_eq!(b.knn_train_query_rows, DEFAULT_BASELINE_SAMPLES);
    assert!(b.knn_capped);
    assert!(b.knn_train_query_sem.is_finite() && b.knn_train_query_sem > 0.0);
    assert!(b.knn_seconds >= 0.0);

    let small = sample(100, 2, 42);
    let e = Baselines::fit_sampled(
        &small,
        &[],
        Metric::Absolute,
        DEFAULT_BASELINE_SAMPLES,
        7,
    );
    assert_eq!(e.knn_train_query_rows, small.len());
    assert!(!e.knn_capped);
    assert!(e.knn_train_query_sem.is_nan());

    let exact = Baselines::fit_sampled(&large, &[], Metric::Absolute, 0, 7);
    assert_eq!(exact.knn_train_query_rows, large.len());
    assert!(!exact.knn_capped);
}

#[test]
fn json_reports_split_query_rows_and_sems() {
    let train = sample(20_000, 3, 8);
    let sampled = Baselines::fit_sampled(&train, &[], Metric::Absolute, 256, 17);
    let json = sampled.knn_json_extra();
    assert!(json.contains("\"reference_rows\":20000"), "{json}");
    assert!(json.contains("\"train_query_rows\":256"), "{json}");
    assert!(json.contains("\"test_query_rows\":0"), "{json}");
    assert!(json.contains("\"capped\":true"), "{json}");
    assert!(json.contains("\"seconds\":"), "{json}");
    assert!(!json.contains("\"train_query_sem\":null"), "{json}");

    let exact = Baselines::fit_sampled(&train, &[], Metric::Absolute, 0, 17);
    let exact_json = exact.knn_json_extra();
    assert!(exact_json.contains("\"train_query_rows\":20000"), "{exact_json}");
    assert!(exact_json.contains("\"train_query_sem\":null"), "{exact_json}");
    assert!(exact_json.contains("\"capped\":false"), "{exact_json}");
}

/// #497: above the threshold every metric samples its queries and reports the
/// sample size and seed; the default is exact up to the threshold.
#[test]
fn every_metric_samples_queries_with_a_reported_size_and_seed() {
    let train = sample(1000, 3, 31);
    for m in [
        Metric::Absolute,
        Metric::Squared,
        Metric::LogSquashed,
        Metric::LogLoss,
        Metric::Brier,
        Metric::Hinge,
        Metric::HingeOrdinary,
        Metric::HingeSquared,
        Metric::Auc,
        Metric::Median,
        Metric::Correlation,
        Metric::Aic,
    ] {
        let b = Baselines::fit_sampled(&train, &[], m, 100, 3);
        assert!(b.knn_capped, "{m:?}");
        assert_eq!(b.knn_train_query_rows, 100, "{m:?}");
        assert_eq!(b.knn_sample_size, 100, "{m:?}");
        assert_eq!(b.knn_seed, 3, "{m:?}");
        let json = b.knn_json_extra();
        assert!(json.contains("\"sample_size\":100"), "{json}");
        assert!(json.contains("\"sample_seed\":3"), "{json}");
    }
    let exact = Baselines::fit_sampled(&train, &[], Metric::Median, 0, 3);
    assert!(!exact.knn_capped);
    assert_eq!(exact.knn_train_query_rows, train.len());
    assert_eq!(exact.knn_sample_size, 0);
}

#[test]
fn default_sample_is_exact_up_to_the_threshold_and_sampled_above() {
    use rcsr::diagnostics::{resolve_baseline_samples, KNN_EXACT_MAX_ROWS};
    assert_eq!(resolve_baseline_samples(None, KNN_EXACT_MAX_ROWS), 0);
    assert_eq!(
        resolve_baseline_samples(None, KNN_EXACT_MAX_ROWS + 1),
        DEFAULT_BASELINE_SAMPLES
    );
    assert_eq!(resolve_baseline_samples(Some(64), 10), 64);
    assert_eq!(resolve_baseline_samples(Some(0), KNN_EXACT_MAX_ROWS + 1), 0);
}
