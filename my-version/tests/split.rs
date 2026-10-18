//! Train/test splitting for i.i.d., time-series and grouped (panel) data.

use rcsr::data::{self, SplitMode, SplitSpec};
use rcsr::rng::Mt;

fn spec(mode: SplitMode) -> SplitSpec {
    SplitSpec {
        mode,
        test_frac: 0.3,
        ..Default::default()
    }
}

/// Rows `1..=n`, target = the row index, one input column `x`.
fn series_csv(n: usize) -> data::Table {
    let mut s = String::from("t,x,y\n");
    for i in 1..=n {
        s.push_str(&format!("{},{},{}\n", i, i, i));
    }
    data::parse_table(&s).unwrap()
}

#[test]
fn label_columns_are_kept_as_text() {
    let t = data::parse_table("date,stock,close,y\n2024-01-01,AAPL,1.5,1\n2024-01-02,MSFT,2.5,2\n")
        .unwrap();
    assert_eq!(t.len(), 2);
    assert!(t.numeric[2] && t.numeric[3]);
    assert!(!t.numeric[0] && !t.numeric[1], "date/stock must not be numeric");
    assert!(t.rows[0][0].is_nan());
    assert_eq!(t.raw[0][1], "AAPL");
    assert_eq!(t.distinct_count(1), 2);
}

#[test]
fn temporal_split_is_chronological_and_honours_the_gap() {
    let t = series_csv(10);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0); // column `t`
    s.gap = 2;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    let train_y = &p.train.y;
    let test_y = &p.test.y;
    assert_eq!(train_y, &vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
    // Rows 8 and 9 are the embargo, so only row 10 is tested.
    assert_eq!(test_y, &vec![10.0]);
    let max_train = train_y.iter().cloned().fold(f64::MIN, f64::max);
    let min_test = test_y.iter().cloned().fold(f64::MAX, f64::min);
    assert!(max_train < min_test, "train must precede test in time");
    assert_eq!(p.report.gap, 2);
    assert_eq!(p.report.mode, "temporal");
}

#[test]
fn normalisation_statistics_come_from_training_rows_only() {
    let t = series_csv(10);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0);
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    // Training rows are 1..=7, whose mean is 4. Using all ten rows would give
    // 5.5 — that leak is exactly what this test guards against.
    assert!((p.means[0] - 4.0).abs() < 1e-12, "mean {:?}", p.means);
    // And the training inputs must be z-scored with those statistics.
    let m: f64 = p.train.x.iter().map(|r| r[0]).sum::<f64>() / p.train.x.len() as f64;
    assert!(m.abs() < 1e-12);
}

#[test]
fn group_split_never_puts_a_group_on_both_sides() {
    // Three stocks, four rows each; the target encodes the stock.
    let mut s = String::from("stock,x,y\n");
    for (g, name) in ["AAPL", "MSFT", "NVDA"].iter().enumerate() {
        for i in 0..4 {
            s.push_str(&format!("{},{},{}\n", name, i, g));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let mut sp = SplitSpec {
        mode: SplitMode::Group,
        group_col: Some(0),
        test_frac: 0.34,
        ..Default::default()
    };
    sp.folds = 1;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &sp, &mut rng).unwrap();

    let train_groups: Vec<f64> = p.train.y.clone();
    let test_groups: Vec<f64> = p.test.y.clone();
    assert!(!test_groups.is_empty() && !train_groups.is_empty());
    for g in &test_groups {
        assert!(
            !train_groups.contains(g),
            "group {} appears on both sides",
            g
        );
    }
    assert_eq!(p.report.groups_total, 3);
    assert_eq!(p.report.groups_train + p.report.groups_test, 3);
    assert_eq!(p.report.groups_test, 1, "one whole stock should be held out");
}

#[test]
fn temporal_group_tests_every_series_and_uses_whole_series_history() {
    let mut s = String::from("t,stock,x,y\n");
    for (g, name) in ["A", "B"].iter().enumerate() {
        for i in 1..=10 {
            s.push_str(&format!("{},{},{},{}\n", i, name, i, g));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let sp = SplitSpec {
        mode: SplitMode::TemporalGroup,
        time_col: Some(0),
        group_col: Some(1),
        test_frac: 0.3,
        ..Default::default()
    };
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &sp, &mut rng).unwrap();

    // Every group contributes to both sides.
    let mut train_groups: Vec<f64> = p.train.y.clone();
    train_groups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    train_groups.dedup();
    let mut test_groups: Vec<f64> = p.test.y.clone();
    test_groups.sort_by(|a, b| a.partial_cmp(b).unwrap());
    test_groups.dedup();
    assert_eq!(train_groups, vec![0.0, 1.0]);
    assert_eq!(test_groups, vec![0.0, 1.0]);
    // 3 of each series is tested (rows 8..=10), the rest trained.
    assert_eq!(p.test.len(), 6);
    assert_eq!(p.train.len(), 14);
}

#[test]
fn folds_partition_the_test_region() {
    let t = series_csv(12);
    let mut s = spec(SplitMode::Temporal);
    s.time_col = Some(0);
    s.folds = 3;
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &s, &mut rng).unwrap();

    assert_eq!(p.fold_samples.len(), 3);
    let total: usize = p.fold_samples.iter().map(|f| f.len()).sum();
    assert_eq!(total, p.test.len());
    assert_eq!(total, 4); // 30% of 12, rounded
    for f in &p.fold_samples {
        assert!(!f.is_empty());
    }
    // Blocks are contiguous and non-overlapping in time, in order.
    let bounds: Vec<(f64, f64)> = p
        .fold_samples
        .iter()
        .map(|f| {
            (
                f.y.iter().cloned().fold(f64::MAX, f64::min),
                f.y.iter().cloned().fold(f64::MIN, f64::max),
            )
        })
        .collect();
    for w in bounds.windows(2) {
        assert!(w[0].1 < w[1].0, "folds must not overlap: {:?}", bounds);
    }
}

#[test]
fn auto_detection_finds_time_and_group_columns() {
    let mut s = String::from("date,stock,x,y\n");
    for name in ["A", "B"] {
        for d in 1..=6 {
            s.push_str(&format!("2024-01-0{},{},{},{}\n", d, name, d, d));
        }
    }
    let t = data::parse_table(&s).unwrap();
    let (mode, tc, gc, _notes) = data::suggest_split(&t, 3, &[2]);
    assert_eq!(mode, SplitMode::TemporalGroup);
    assert_eq!(t.names[tc.unwrap()], "date");
    assert_eq!(t.names[gc.unwrap()], "stock");

    // Exchangeable rows stay random.
    let t2 = data::parse_table("a,b\n0.1,1\n0.7,2\n0.3,3\n0.9,4\n0.2,5\n").unwrap();
    let (m2, _, _, _) = data::suggest_split(&t2, 1, &[0]);
    assert_eq!(m2, SplitMode::Random);
}

#[test]
fn dates_parse_and_order_correctly() {
    let a = data::parse_time("2024-01-31").unwrap();
    let b = data::parse_time("2024-02-01").unwrap();
    let c = data::parse_time("2024-01-31T09:30:00").unwrap();
    assert!(a < b && a < c && c - a == 9.0 * 3600.0 + 30.0 * 60.0);
    assert_eq!(data::parse_time("2024/02/01"), Some(b));
    assert_eq!(data::parse_time("42.5"), Some(42.5));
    assert!(data::parse_time("not-a-date").is_none());
}
