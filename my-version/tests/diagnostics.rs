//! #69: the k-NN null baseline keeps an exact reference set and, when query
//! rows are sampled, stays an unbiased estimate of the same error.

use rcsr::data::Sample;
use rcsr::diagnostics::{knn_predict, Baselines, KNN_KD_MAX_DIM};
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
    // 3/6 inputs: k-d tree path. 9 inputs: brute-force path.
    for dims in [3, 6, KNN_KD_MAX_DIM + 1] {
        let train = sample(600, dims, 1);
        let query = sample(120, dims, 2);
        let b = Baselines::fit_sampled(&train, &[query.clone()], Metric::Absolute, 0, 7);
        let expected_train = brute_force_error(&train, &train, Metric::Absolute);
        let expected_test = brute_force_error(&train, &query, Metric::Absolute);

        assert_bits(b.knn.train_error, expected_train, "train");
        assert_bits(b.knn.test_error, expected_test, "test");
        assert_eq!(b.knn_reference_rows, train.len());
        assert_eq!(b.knn_query_rows, train.len() + query.len());
        assert!(!b.knn_capped);
        assert!(b.knn_query_sem.is_nan());
    }
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
    assert_eq!(small.knn_query_rows, cap);
    assert_eq!(large.knn_query_rows, cap);
    assert!(small.knn_capped && large.knn_capped);
    assert!(small.knn_query_sem.is_finite());
    assert!(large.knn_query_sem.is_finite());
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
    assert_eq!(no_caps.knn_query_rows, train.len() + fold.len());
    assert!(!no_caps.knn_capped);
    assert!(no_caps.knn_query_sem.is_nan());
}

#[test]
fn json_reports_reference_query_rows_and_sem() {
    let train = sample(20_000, 3, 8);
    let sampled = Baselines::fit_sampled(&train, &[], Metric::Absolute, 256, 17);
    let json = sampled.knn_json_extra();
    assert!(json.contains("\"reference_rows\":20000"), "{json}");
    assert!(json.contains("\"query_rows\":256"), "{json}");
    assert!(json.contains("\"capped\":true"), "{json}");
    assert!(!json.contains("\"query_sem\":null"), "{json}");

    let exact = Baselines::fit_sampled(&train, &[], Metric::Absolute, 0, 17);
    let exact_json = exact.knn_json_extra();
    assert!(exact_json.contains("\"query_rows\":20000"), "{exact_json}");
    assert!(exact_json.contains("\"query_sem\":null"), "{exact_json}");
    assert!(exact_json.contains("\"capped\":false"), "{exact_json}");
}
