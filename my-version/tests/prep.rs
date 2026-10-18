//! Data preparation: missing values, outliers, smoothing, normalisation and
//! row filters (issue #2). Every default must reproduce the plain mean/std run.

use rcsr::data::{
    self, Center, FillMethod, Filter, PrepSpec, Scale, SplitMode, SplitSpec,
};
use rcsr::expr;
use rcsr::metrics::Metric;
use rcsr::rng::Mt;
use rcsr::search::{run, Config};
use std::io::Write;
use std::process::{Command, Stdio};

fn temporal(test_frac: f64) -> SplitSpec {
    SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        test_frac,
        ..Default::default()
    }
}

fn prep(fill: FillMethod) -> PrepSpec {
    PrepSpec {
        fill,
        ..Default::default()
    }
}

fn prepare<'a>(
    t: &data::Table,
    sp: &SplitSpec,
    pr: &PrepSpec,
) -> data::Prepared {
    let mut rng = Mt::new(1);
    data::prepare_full_with(t, &["x".into()], "y", sp, &mut rng, pr).unwrap()
}

/// Missing markers are identified at load time and kept as `NaN` in the numeric
/// view, so a column with gaps is still usable.
#[test]
fn loader_accepts_missing_markers() {
    let t = data::parse_table("x,y\n1,2\nNA,3\n,4\nnull,5\nNaN,6\n").unwrap();
    assert!(t.numeric[0] && t.numeric[1], "a gap must not make a column text");
    assert_eq!(t.rows[0][0], 1.0);
    for r in 1..5 {
        assert!(t.rows[r][0].is_nan(), "row {r} should be missing");
    }
    assert!(data::is_missing_token("Na"));
    assert!(data::is_missing_token("NULL"));
    assert!(data::is_missing_token("  "));
    assert!(!data::is_missing_token("abc"));
}

/// `drop` removes exactly the rows with a gap, from both sides of the split.
#[test]
fn drop_removes_exactly_the_missing_rows() {
    // Rows 0..9, target = 10 + row. `x` is missing in train row 1 and test row
    // 8; those rows are the only ones that may disappear.
    let mut csv = String::from("t,x,y\n");
    for i in 0..10 {
        let x = if i == 1 || i == 8 {
            "NA".to_string()
        } else {
            i.to_string()
        };
        csv.push_str(&format!("{},{},{}\n", i, x, 10 + i));
    }
    let t = data::parse_table(&csv).unwrap();
    let p = prepare(&t, &temporal(0.3), &prep(FillMethod::Drop));

    // 30% of 10 = 3 test rows (7, 8, 9); rows 0..6 train. Row 1 is a training
    // row and row 8 a test row, so exactly those two are gone.
    assert_eq!(p.prep.rows_dropped_missing, 2);
    assert_eq!(p.prep.fill, "drop");
    assert_eq!(p.train.y, vec![10.0, 12.0, 13.0, 14.0, 15.0, 16.0]);
    assert_eq!(p.test.y, vec![17.0, 19.0]);
}

/// Mean/median filling uses the training statistic; a wild test value must not
/// move it. The filled training statistic normalises to exactly zero.
#[test]
fn mean_and_median_fill_ignore_the_test_side() {
    let mut csv = String::from("t,x,y\n");
    for i in 0..10 {
        let x = match i {
            8 => "NA".to_string(),
            9 => "1000000".to_string(), // wild test-side value
            _ => (i + 1).to_string(),   // train 1..8
        };
        let y = if i == 8 { "NA".to_string() } else { i.to_string() };
        csv.push_str(&format!("{},{},{}\n", i, x, y));
    }
    let t = data::parse_table(&csv).unwrap();

    for method in [FillMethod::Mean, FillMethod::Median] {
        let p = prepare(&t, &temporal(0.2), &prep(method));
        // Train x = 1..8: mean = median = 4.5. The test outlier is present in
        // the table but must not enter either statistic.
        assert!(
            (p.means[0] - 4.5).abs() < 1e-12,
            "{:?}: mean leaked from the test side: {:?}",
            method,
            p.means
        );
        // Test row 8 (first test row) was filled with 4.5; normalised = 0.
        assert!(p.test.x[0][0].abs() < 1e-12, "{:?}: fill {:?}", method, p.test.x[0]);
        assert_eq!(p.prep.rows_filled, 1);
        assert_eq!(p.prep.rows_dropped_missing, 0);
    }
}

/// Outlier bounds are training-only, hand-computed for this table.
#[test]
fn outlier_bounds_are_training_only_and_hand_computed() {
    // Train x = [0,1,2,3,4,5,6,100]; sorted quantiles (type 7):
    //   median = 3.5, q1 = 1.75, q3 = 5.25, IQR = 3.5.
    // With k = 1 the bounds are [0, 7], so only the 100 is dropped. The test
    // values 3.5 and 4.5 stay.
    let mut csv = String::from("t,x,y\n");
    for i in 0..10 {
        let x = match i {
            7 => 100.0,
            8 => 3.5,
            9 => 4.5,
            _ => i as f64,
        };
        csv.push_str(&format!("{},{},{}\n", i, x, 0));
    }
    let t = data::parse_table(&csv).unwrap();
    let pr = PrepSpec {
        outlier_iqr: Some(1.0),
        center: Center::None,
        scale: Scale::None,
        ..Default::default()
    };
    let p = prepare(&t, &temporal(0.2), &pr);
    assert_eq!(p.prep.rows_dropped_outlier, 1);
    assert_eq!(p.train.len(), 7);
    assert_eq!(p.test.len(), 2);
    assert!(p.test.x.iter().all(|r| r[0] == 3.5 || r[0] == 4.5));

    // A wider fence removes nothing.
    let pr = PrepSpec {
        outlier_iqr: Some(30.0),
        center: Center::None,
        scale: Scale::None,
        ..Default::default()
    };
    let p = prepare(&t, &temporal(0.2), &pr);
    assert_eq!(p.prep.rows_dropped_outlier, 0);
}

/// A 3-wide centred moving average with shrinking edges, in original units
/// (`center none`, `scale none`).
#[test]
fn smoothing_uses_a_centred_window_and_shrinks_at_the_edges() {
    // n=6 at 30%: train rows 0..3, test rows 4..5.
    let mut csv = String::from("t,x,y\n");
    for i in 0..6 {
        csv.push_str(&format!("{},{},{}\n", i, i + 1, i));
    }
    let t = data::parse_table(&csv).unwrap();
    let pr = PrepSpec {
        smooth: 3,
        center: Center::None,
        scale: Scale::None,
        ..Default::default()
    };
    let p = prepare(&t, &temporal(0.3), &pr);

    // Train: [1,2,3,4] -> [1.5, 2, 3, 3.5]; test: [5,6] -> [5.5, 5.5].
    let train: Vec<f64> = p.train.x.iter().map(|r| r[0]).collect();
    assert_eq!(train, vec![1.5, 2.0, 3.0, 3.5]);
    let test: Vec<f64> = p.test.x.iter().map(|r| r[0]).collect();
    assert_eq!(test, vec![5.5, 5.5]);
    // Two used columns (x and the target) x six rows.
    assert_eq!(p.prep.smoothed, 12);
}

/// Each centre/scale combination has the expected training statistic, and the
/// normalised value is always the exact affine transform of the original.
#[test]
fn every_center_and_scale_combination_denormalises_exactly() {
    // Train rows are x = 1..4; test rows 5..6.
    let mut csv = String::from("t,x,y\n");
    for i in 0..6 {
        csv.push_str(&format!("{},{},{}\n", i, i + 1, i));
    }
    let t = data::parse_table(&csv).unwrap();
    let std = 1.25f64.sqrt();

    let centers = [
        (Center::Mean, 2.5),
        (Center::Median, 2.5),
        (Center::Min, 1.0),
        (Center::Max, 4.0),
        (Center::None, 0.0),
    ];
    let scales = [
        (Scale::Std, std),
        (Scale::Iqr, 1.5),
        (Scale::Range, 3.0),
        (Scale::None, 1.0),
    ];

    for (center, expected_center) in centers {
        for (scale, expected_scale) in scales {
            let pr = PrepSpec {
                center,
                scale,
                ..Default::default()
            };
            let p = prepare(&t, &temporal(0.3), &pr);
            assert!(
                (p.means[0] - expected_center).abs() < 1e-12,
                "center {:?}: {:?}",
                center,
                p.means
            );
            assert!(
                (p.stds[0] - expected_scale).abs() < 1e-12,
                "scale {:?}: {:?}",
                scale,
                p.stds
            );
            // Original = normalised * scale + centre, for every training row.
            for (row, &raw) in [1.0, 2.0, 3.0, 4.0].iter().enumerate() {
                let z = p.train.x[row][0];
                let back = z * p.stds[0] + p.means[0];
                assert!((back - raw).abs() < 1e-12, "{:?}/{:?}: {} != {}", center, scale, back, raw);
            }
        }
    }
}

/// The default is exactly the previous mean/std behaviour (and no cleaning).
#[test]
fn defaults_reproduce_mean_std_and_change_nothing() {
    let mut csv = String::from("t,x,y\n");
    for i in 0..6 {
        csv.push_str(&format!("{},{},{}\n", i, i + 1, i));
    }
    let t = data::parse_table(&csv).unwrap();
    let mut rng = Mt::new(1);
    let p = data::prepare_full(&t, &["x".into()], "y", &temporal(0.3), &mut rng).unwrap();

    assert_eq!(p.means[0], 2.5);
    assert_eq!(p.stds[0], 1.25f64.sqrt());
    assert_eq!(p.prep.fill, "none");
    assert_eq!(p.prep.center, "mean");
    assert_eq!(p.prep.scale, "std");
    assert_eq!(p.prep.rows_filled, 0);
    assert_eq!(p.prep.rows_dropped_missing, 0);
    assert_eq!(p.prep.rows_dropped_outlier, 0);
    assert_eq!(p.prep.smoothed, 0);
    assert_eq!(p.prep.filtered, 0);
    assert!(!p.prep.active());
    // Mean of the normalised training inputs is zero, as before.
    let m: f64 = p.train.x.iter().map(|r| r[0]).sum::<f64>() / p.train.len() as f64;
    assert!(m.abs() < 1e-12);
}

/// A comparison filter keeps exactly the rows where it is true, including a
/// compound expression.
#[test]
fn filter_keeps_exactly_the_matching_rows() {
    let mut csv = String::from("x,y\n");
    let rows = [
        (0.1, 9.0),
        (0.2, 8.0),
        (0.3, 7.0),
        (0.4, 6.0),
        (0.6, 1.0),
        (0.7, 2.0),
        (0.8, 3.0),
        (0.9, 4.0),
    ];
    for (x, y) in rows {
        csv.push_str(&format!("{},{}\n", x, y));
    }
    let t = data::parse_table(&csv).unwrap();

    let vars = vec!["x".to_string(), "y".to_string()];
    for (src, predicate) in [
        ("x > 0.5", (|x: f64, _y: f64| x > 0.5) as fn(f64, f64) -> bool),
        ("x > 0.2 and y < 8", |x: f64, y: f64| x > 0.2 && y < 8.0),
        ("x > 0.8 or y > 8", |x: f64, y: f64| x > 0.8 || y > 8.0),
    ] {
        let tree = expr::parse(src, &vars).unwrap();
        let pr = PrepSpec {
            filter: Some(Filter {
                tree,
                vars: vars.clone(),
                expression: src.to_string(),
            }),
            center: Center::None,
            scale: Scale::None,
            ..Default::default()
        };
        let sp = SplitSpec {
            mode: SplitMode::Random,
            test_frac: 0.25,
            ..Default::default()
        };
        let p = prepare(&t, &sp, &pr);
        let expected = rows.iter().filter(|(x, y)| predicate(*x, *y)).count();
        assert_eq!(p.prep.filtered, rows.len() - expected, "{src}");
        assert_eq!(p.train.len() + p.test.len(), expected, "{src}");
        let kept = p
            .train
            .x
            .iter()
            .zip(&p.train.y)
            .chain(p.test.x.iter().zip(&p.test.y));
        for (r, y) in kept {
            assert!(predicate(r[0], *y), "{src}: kept a non-matching row {:?}", (r, y));
        }
    }
}

/// End-to-end: a handful of gaps filled by the median still recover y = 2x+1.
/// Gaps are placed in *both* x and y on the same rows; because the law is
/// affine and monotone, the median of y is 2*median(x)+1, so the filled pairs
/// stay exactly on the law.
#[test]
fn median_fill_still_recovers_the_law() {
    let mut csv = String::from("x,y\n");
    let mut n = 0usize;
    for i in -20..=20 {
        let x = i as f64 / 10.0;
        let y = 2.0 * x + 1.0;
        n += 1;
        if matches!(n, 5 | 13 | 21 | 33) {
            csv.push_str("NA,NA\n");
        } else {
            csv.push_str(&format!("{},{}\n", x, y));
        }
    }
    let t = data::parse_table(&csv).unwrap();
    let pr = prep(FillMethod::Median);
    let mut rng = Mt::new(7);
    let sp = SplitSpec {
        mode: SplitMode::Random,
        test_frac: 0.25,
        ..Default::default()
    };
    let p = data::prepare_full_with(&t, &["x".into()], "y", &sp, &mut rng, &pr).unwrap();
    assert_eq!(p.prep.rows_filled, 4);
    assert_eq!(p.prep.rows_dropped_missing, 0);

    let mut cfg = Config::default();
    cfg.generations = 300;
    cfg.pop = 200;
    cfg.islands = 2;
    cfg.seed = 7;
    cfg.metric = Metric::Squared;
    cfg.ops = vec![
        rcsr::expr::Op::Add,
        rcsr::expr::Op::Sub,
        rcsr::expr::Op::Mul,
        rcsr::expr::Op::Div,
    ];

    let out = run(&cfg, &p.train, &p.test);
    let best = out
        .test_errors
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    assert!(best < 1e-6, "best test error after median fill = {}", best);
}

/// A bad filter is a usage error (exit 2), and the message carries the parser's
/// 1-based column.
#[test]
fn bad_filter_is_a_usage_error_with_the_column() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", "-", "--target", "y", "--filter", "x > "])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let stdin = child.stdin.as_mut().unwrap();
        stdin.write_all(b"x,y\n1,2\n2,3\n4,5\n").unwrap();
    }
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("column 5"), "stderr: {err}");
    assert!(err.contains("unexpected end of input"), "stderr: {err}");
}

/// A gap left in a used column must be a loud, actionable error -- never a
/// silent empty front with `inf` baselines and exit code 0.
#[test]
fn a_gap_without_a_fill_method_is_an_error() {
    let t = data::parse_table("x,y\n1,3\n2,\n3,7\n4,9\n").unwrap();
    let split = SplitSpec::default();
    let vars = ["x".to_string()];
    let err = data::prepare_full_with(
        &t,
        &vars,
        "y",
        &split,
        &mut Mt::new(0),
        &PrepSpec::default(),
    )
    .err()
    .expect("a gap must not be silently accepted");
    assert!(err.contains("1 missing cell"), "err: {err}");
    assert!(err.contains("y: 1"), "the message must name the column: {err}");
    assert!(err.contains("--fill"), "the message must name the fix: {err}");

    // The same table is fine once a method is chosen.
    let prep = PrepSpec {
        fill: FillMethod::Drop,
        ..Default::default()
    };
    let p = data::prepare_full_with(&t, &vars, "y", &split, &mut Mt::new(0), &prep).unwrap();
    assert_eq!(p.prep.rows_dropped_missing, 1);
}
