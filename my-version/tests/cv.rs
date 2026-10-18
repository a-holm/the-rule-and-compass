//! Cross-validation tests (issue #19).
//!
//! Two things are checked: that the fold construction is what it claims to be
//! (partition, forward chaining, embargo, train-only normalisation), and that
//! cross-validation actually exposes overfitting on a known-truth problem.

use rcsr::expr::{self, Node, Op};
use rcsr::metrics::Metric;
use rcsr::{data, fit, search};

/// A table of `x = 0, 1, ...` with `y = x`.
fn ramp(n: usize) -> data::Table {
    let mut text = String::from("x,y\n");
    for i in 0..n {
        text.push_str(&format!("{},{}\n", i, i));
    }
    data::parse_table(&text).expect("table")
}

fn xvars() -> Vec<String> {
    vec!["x".to_string()]
}

#[test]
fn modes_parse_and_reject_junk() {
    assert_eq!(data::CvMode::parse("k-fold").unwrap(), data::CvMode::KFold);
    assert_eq!(data::CvMode::parse("KFold").unwrap(), data::CvMode::KFold);
    assert_eq!(
        data::CvMode::parse(" rolling ").unwrap(),
        data::CvMode::Rolling
    );
    assert_eq!(data::CvMode::parse("purged").unwrap(), data::CvMode::Purged);
    assert!(data::CvMode::parse("bogus").is_err());
}

#[test]
fn kfold_partitions_and_normalises_on_train_only() {
    let t = ramp(20);
    let folds = data::cv_folds(&t, &xvars(), "y", data::CvMode::KFold, 4, 0, 1).unwrap();
    assert_eq!(folds.len(), 4);
    let total_test: usize = folds.iter().map(|f| f.test.len()).sum();
    assert_eq!(total_test, 20, "the test folds must partition the rows");
    for f in &folds {
        assert_eq!(f.train.len() + f.test.len(), 20);
        // Normalisation statistics come from the fold's training rows alone.
        let m = f.train.x.iter().map(|r| r[0]).sum::<f64>() / f.train.len() as f64;
        assert!(m.abs() < 1e-9, "fold train mean should be 0, got {m}");
    }
}

#[test]
fn rolling_folds_expand_forwards() {
    let t = ramp(20);
    let folds = data::cv_folds(&t, &xvars(), "y", data::CvMode::Rolling, 4, 0, 1).unwrap();
    assert!(!folds.is_empty());
    for w in folds.windows(2) {
        assert!(
            w[0].train.len() <= w[1].train.len(),
            "the rolling training window must not shrink"
        );
    }
    // Every fold trains on the past and tests on the future. Compare in original
    // units, since each fold is normalised on its own training rows.
    let raw_x = |s: &data::Sample, f: &data::CvFold| -> Vec<f64> {
        let mut v: Vec<f64> = s.x.iter().map(|r| r[0] * f.stds[0] + f.means[0]).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    for f in &folds {
        let train = raw_x(&f.train, f);
        let test = raw_x(&f.test, f);
        assert!(
            train.last().unwrap() < test.first().unwrap(),
            "a rolling fold must train on the past only"
        );
    }
}

#[test]
fn purged_embargo_removes_training_rows() {
    let t = ramp(50);
    let plain = data::cv_folds(&t, &xvars(), "y", data::CvMode::KFold, 5, 0, 1).unwrap();
    let purged = data::cv_folds(&t, &xvars(), "y", data::CvMode::Purged, 5, 3, 1).unwrap();
    let plain_train: usize = plain.iter().map(|f| f.train.len()).sum();
    let purged_train: usize = purged.iter().map(|f| f.train.len()).sum();
    assert!(
        purged_train < plain_train,
        "the embargo must remove training rows ({purged_train} vs {plain_train})"
    );
    assert_eq!(
        purged.iter().map(|f| f.test.len()).sum::<usize>(),
        50,
        "the embargo must not touch the test blocks"
    );
}

#[test]
fn too_few_rows_is_an_error_not_a_panic() {
    let t = ramp(3);
    assert!(data::cv_folds(&t, &xvars(), "y", data::CvMode::KFold, 5, 0, 1).is_err());
}

/// `c0 + c1*x + ... + cdeg*x^deg` with free constants, built by repeated
/// multiplication so the exponent is not itself a fitted parameter.
fn poly(deg: usize) -> Node {
    fn powx(x: &Node, j: usize) -> Node {
        let mut acc = x.clone();
        for _ in 1..j {
            acc = expr::node(Op::Mul, vec![acc, x.clone()]);
        }
        acc
    }
    let x = expr::input(0);
    let mut acc = expr::constant(0.0);
    for j in 1..=deg {
        acc = expr::node(
            Op::Add,
            vec![acc, expr::node(Op::Mul, vec![expr::constant(0.0), powx(&x, j)])],
        );
    }
    acc
}

/// The point of cross-validation: a flexible model that looks excellent on its
/// training rows does worse than a simple one on rows it has never seen. If CV
/// could not show this it would be useless as an overfitting check.
#[test]
fn cross_validation_exposes_overfitting() {
    // y = 3x + 1 + noise on x in [-1, 1].
    let mut text = String::from("x,y\n");
    let mut s: u64 = 0x2545_F491_4F6C_DD1D;
    let mut jitter = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f64 / (1u64 << 23) as f64) - 1.0
    };
    let n = 30;
    for i in 0..n {
        let x = -1.0 + 2.0 * i as f64 / (n - 1) as f64;
        text.push_str(&format!("{:.6},{:.6}\n", x, 3.0 * x + 1.0 + jitter()));
    }
    let t = data::parse_table(&text).unwrap();
    let folds = data::cv_folds(&t, &xvars(), "y", data::CvMode::KFold, 5, 0, 11).unwrap();

    // Mean train and test error of a degree-`deg` polynomial, fitted on each
    // fold's training rows only.
    let score = |deg: usize| -> (f64, f64) {
        let mut tr = 0.0;
        let mut te = 0.0;
        for f in &folds {
            let mut tree = poly(deg);
            fit::fit(&mut tree, &f.train.x, &f.train.y, 500);
            tr += search::eval_error(&tree, &f.train, Metric::Squared);
            te += search::eval_error(&tree, &f.test, Metric::Squared);
        }
        (tr / folds.len() as f64, te / folds.len() as f64)
    };

    let (lin_train, lin_test) = score(1);
    let (flex_train, flex_test) = score(8);
    assert!(
        flex_train < lin_train,
        "the flexible model fits training data better: {flex_train} vs {lin_train}"
    );
    assert!(
        flex_test > lin_test * 2.0,
        "cross-validation punishes the flexible model: {flex_test} vs {lin_test}"
    );
}
