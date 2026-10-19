//! Boolean and categorical variables (issue #5).
//!
//! A boolean input is 0/1 and **must not be standardised**: the evaluator's
//! truthiness rule (`> 0` is true) and the `== 1` comparison depend on the
//! literal values. A categorical input is expanded into one-hot boolean
//! indicators named `column=value`, with sorted levels so the encoding does not
//! depend on row order.

use rcsr::data::{self, DType, FillMethod, PrepSpec, SplitMode, SplitSpec};
use rcsr::expr::{self, Op};
use rcsr::metrics::Metric;
use rcsr::rng::Mt;
use rcsr::search::{run, Config};
use std::io::Write;
use std::process::{Command, Output, Stdio};

fn temporal() -> SplitSpec {
    SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        test_frac: 0.25,
        ..Default::default()
    }
}

fn run_cli(args: &[&str], stdin: &str) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .as_mut()
        .unwrap()
        .write_all(stdin.as_bytes())
        .unwrap();
    child.wait_with_output().unwrap()
}

/// Validation names the column, the row and the offending value. Missing cells
/// are not value errors: `--fill` owns that decision.
#[test]
fn boolean_values_other_than_0_and_1_are_rejected() {
    let t = data::parse_table("flag,other,y\n0,1,0\n1,2,1\n2,3,0\n").unwrap();
    let e = data::apply_dtypes(
        &t,
        &["flag".into(), "other".into()],
        &[("flag".into(), DType::Boolean)],
        16,
    )
    .err()
    .expect("2 is not a boolean value");
    assert!(!e.usage, "bad data is an input error, not a usage error");
    assert!(e.message.contains("boolean column 'flag'"), "{}", e.message);
    assert!(e.message.contains("row 3"), "{}", e.message);
    assert!(e.message.contains("'2'"), "{}", e.message);

    // Fractional and negative values are rejected too.
    for bad in ["0.5", "-1"] {
        let csv = format!("flag,y\n0,0\n{},1\n", bad);
        let t = data::parse_table(&csv).unwrap();
        let e = data::apply_dtypes(&t, &["flag".into()], &[("flag".into(), DType::Boolean)], 16)
            .err()
            .expect("fractional and negative values are rejected");
        assert!(e.message.contains(bad), "{}", e.message);
    }

    // 0/1 written as decimals are the same numbers and are accepted.
    let t = data::parse_table("flag,y\n0.0,0\n1.0,1\n").unwrap();
    assert!(data::apply_dtypes(&t, &["flag".into()], &[("flag".into(), DType::Boolean)], 16).is_ok());
}

/// A boolean column passes through preparation untouched: mean 0 / std 1, and
/// the evaluator sees exactly the 0/1 from the file. Continuous columns in the
/// same table keep the usual mean/std standardisation.
#[test]
fn boolean_columns_are_not_standardised() {
    let mut csv = String::from("t,flag,x,y\n");
    for i in 0..8 {
        csv.push_str(&format!("{},{},{},{}\n", i, i % 2, i, i % 2));
    }
    let t = data::parse_table(&csv).unwrap();
    let typed = data::apply_dtypes(
        &t,
        &["flag".into(), "x".into()],
        &[("flag".into(), DType::Boolean)],
        16,
    )
    .unwrap();
    assert_eq!(typed.dtypes, vec![DType::Boolean, DType::Continuous]);

    let prep = PrepSpec {
        dtypes: typed.dtypes.clone(),
        ..Default::default()
    };
    let p = data::prepare_full_with(
        &typed.table,
        &typed.vars,
        "y",
        &temporal(),
        &mut Mt::new(1),
        &prep,
    )
    .unwrap();

    // The bookkeeping is the affine identity, so a printed formula is in
    // original units and normalisation cannot move the truth value.
    assert_eq!(p.means[0], 0.0);
    assert_eq!(p.stds[0], 1.0);
    let printed = expr::simplify(&expr::denormalize(&expr::input(0), &p.means, &p.stds));
    assert_eq!(expr::to_string(&printed, &typed.vars), "flag");

    // Temporal split of 8 rows at 25%: train rows 0..6, flags 0,1,0,1,0,1.
    let flags: Vec<f64> = p.train.x.iter().map(|r| r[0]).collect();
    assert_eq!(flags, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    assert!(p.train.x.iter().all(|r| r[0] == 0.0 || r[0] == 1.0));

    // The continuous column is still standardised (train x = 0..5).
    assert!((p.means[1] - 2.5).abs() < 1e-12);
    assert!((p.stds[1] - (35.0f64 / 12.0).sqrt()).abs() < 1e-12);

    // Truthiness reaches the evaluator unchanged: if(flag, 7, -7).
    let branch = expr::node(
        Op::If,
        vec![expr::input(0), expr::constant(7.0), expr::constant(-7.0)],
    );
    for (row, &flag) in p.train.x.iter().zip(&flags) {
        let expected = if flag > 0.0 { 7.0 } else { -7.0 };
        assert_eq!(expr::eval(&branch, row), expected);
    }
}

/// One-hot expansion: levels are the distinct values in the whole table, sorted,
/// independent of row order, and each indicator is exactly 0/1.
#[test]
fn categorical_expands_to_sorted_one_hot_indicators() {
    let t = data::parse_table(
        "symbol,x,y\nB,1,0\nA,2,1\nB,3,0\nC,4,1\nA,5,0\n",
    )
    .unwrap();
    let typed = data::apply_dtypes(
        &t,
        &["symbol".into(), "x".into()],
        &[("symbol".into(), DType::Categorical)],
        16,
    )
    .unwrap();

    assert_eq!(typed.vars, vec!["symbol=A", "symbol=B", "symbol=C", "x"]);
    assert_eq!(
        typed.dtypes,
        vec![DType::Boolean, DType::Boolean, DType::Boolean, DType::Continuous]
    );
    assert_eq!(typed.variables[0].source.as_deref(), Some("symbol"));
    assert_eq!(typed.variables[3].source, None);

    // Indicator columns are appended after the original table, in level order.
    let at = |row: usize, name: &str| -> f64 {
        let j = typed.table.index_of(name).unwrap();
        typed.table.value(row, j)
    };
    assert_eq!(
        (0..5)
            .map(|r| at(r, "symbol=A"))
            .collect::<Vec<_>>(),
        vec![0.0, 1.0, 0.0, 0.0, 1.0]
    );
    assert_eq!(
        (0..5)
            .map(|r| at(r, "symbol=B"))
            .collect::<Vec<_>>(),
        vec![1.0, 0.0, 1.0, 0.0, 0.0]
    );
    assert_eq!(
        (0..5)
            .map(|r| at(r, "symbol=C"))
            .collect::<Vec<_>>(),
        vec![0.0, 0.0, 0.0, 1.0, 0.0]
    );
    // Every indicator is a boolean input and no original category survives.
    assert!(typed
        .vars
        .iter()
        .all(|v| v == "x" || v.starts_with("symbol=")));

    // Determinism: shuffling the rows changes neither the names nor their order.
    let shuffled = data::parse_table(
        "symbol,x,y\nA,5,0\nC,4,1\nB,1,0\nA,2,1\nB,3,0\n",
    )
    .unwrap();
    let again = data::apply_dtypes(
        &shuffled,
        &["symbol".into(), "x".into()],
        &[("symbol".into(), DType::Categorical)],
        16,
    )
    .unwrap();
    assert_eq!(again.vars, typed.vars);
}

/// Numeric levels sort by value, not by their text (`1, 2, 10`), and missing
/// cells get their own final `column=<missing>` level.
#[test]
fn numeric_levels_and_missing_use_a_stable_order() {
    let t = data::parse_table("k,y\n10,0\n2,1\n1,0\n,1\n").unwrap();
    let typed = data::apply_dtypes(&t, &["k".into()], &[("k".into(), DType::Categorical)], 16)
        .unwrap();
    assert_eq!(typed.vars, vec!["k=1", "k=2", "k=10", "k=<missing>"]);
    assert_eq!(typed.dtypes, vec![DType::Boolean; 4]);
    let last = typed.table.index_of("k=<missing>").unwrap();
    assert_eq!(
        typed.table.col(last).to_vec(),
        vec![0.0, 0.0, 0.0, 1.0]
    );
}

/// More levels than `--max-levels` is a usage error naming the column and the
/// count, with a hint to raise the limit. The CLI exits 2.
#[test]
fn max_levels_exceeded_is_a_usage_error() {
    let t = data::parse_table("symbol,y\nA,0\nB,1\nC,0\n").unwrap();
    let e = data::apply_dtypes(&t, &["symbol".into()], &[("symbol".into(), DType::Categorical)], 2)
        .err()
        .expect("three levels exceed the limit of two");
    assert!(e.usage, "the level limit is a usage error");
    assert!(e.message.contains("categorical column 'symbol'"), "{}", e.message);
    assert!(e.message.contains("3 distinct values"), "{}", e.message);
    assert!(e.message.contains("--max-levels"), "{}", e.message);

    let mut csv = String::from("symbol,y\n");
    for i in 0..5 {
        csv.push_str(&format!("S{},{}\n", i, i % 2));
    }
    let out = run_cli(
        &[
            "--data", "-", "--target", "y", "--dtype", "symbol=categorical",
            "--max-levels", "2", "--generations", "1", "--pop", "5",
        ],
        &csv,
    );
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("symbol"), "stderr: {err}");
    assert!(err.contains("--max-levels 2"), "stderr: {err}");
}

/// Unknown kinds and unknown columns are usage errors (exit 2).
#[test]
fn unknown_dtype_or_column_is_a_usage_error() {
    let csv = "flag,y\n0,0\n1,1\n";

    let out = run_cli(
        &["--data", "-", "--target", "y", "--dtype", "flag=ordinal"],
        csv,
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unknown --dtype kind 'ordinal'"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let out = run_cli(
        &["--data", "-", "--target", "y", "--dtype", "nope=boolean"],
        csv,
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("unknown --dtype column 'nope'"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    // A declared datatype may not silently add a column to an explicit --vars.
    let out = run_cli(
        &[
            "--data", "-", "--target", "y", "--vars", "flag",
            "--dtype", "nope=boolean",
        ],
        "flag,nope,y\n0,0,0\n1,1,1\n",
    );
    assert_eq!(out.status.code(), Some(2));
    assert!(
        String::from_utf8_lossy(&out.stderr).contains("column 'nope' is not an input variable"),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

/// Mean/interpolate fill can average truth values into a fraction; that would
/// silently change the meaning of `and`/`not`/`if`, so it is refused.
#[test]
fn a_fill_that_breaks_boolean_semantics_is_rejected() {
    let t = data::parse_table("t,flag,y\n0,0,0\n1,NA,1\n2,1,0\n3,1,1\n4,0,0\n").unwrap();
    let typed = data::apply_dtypes(&t, &["flag".into()], &[("flag".into(), DType::Boolean)], 16)
        .unwrap();
    let prep = PrepSpec {
        fill: FillMethod::Mean,
        dtypes: typed.dtypes.clone(),
        ..Default::default()
    };
    let e = data::prepare_full_with(
        &typed.table,
        &typed.vars,
        "y",
        &temporal(),
        &mut Mt::new(1),
        &prep,
    )
    .err()
    .expect("mean fill must not create a fractional boolean");
    assert!(e.contains("boolean column 'flag'"), "{e}");
    assert!(e.contains("--fill zero|drop|previous"), "{e}");

    // A zero fill keeps the column boolean.
    let prep = PrepSpec {
        fill: FillMethod::Zero,
        dtypes: typed.dtypes.clone(),
        ..Default::default()
    };
    let p = data::prepare_full_with(
        &typed.table,
        &typed.vars,
        "y",
        &temporal(),
        &mut Mt::new(1),
        &prep,
    )
    .unwrap();
    assert!(p.train.x.iter().all(|r| r[0] == 0.0 || r[0] == 1.0));
}

/// Acceptance target: recover `x1 AND NOT x2` from 0/1 inputs. Boolean inputs
/// reach the evaluator as raw truth values, so `and`/`not` can represent the
/// law exactly; the recovered tree reproduces the truth table.
#[test]
fn recovers_x1_and_not_x2() {
    let mut csv = String::from("x1,x2,y\n");
    for _ in 0..10 {
        for (a, b) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            let y = if a == 1 && b == 0 { 1 } else { 0 };
            csv.push_str(&format!("{},{},{}\n", a, b, y));
        }
    }
    let t = data::parse_table(&csv).unwrap();
    let typed = data::apply_dtypes(
        &t,
        &["x1".into(), "x2".into()],
        &[
            ("x1".into(), DType::Boolean),
            ("x2".into(), DType::Boolean),
        ],
        16,
    )
    .unwrap();
    let prep = PrepSpec {
        dtypes: typed.dtypes.clone(),
        ..Default::default()
    };
    let split = SplitSpec {
        test_frac: 0.25,
        ..Default::default()
    };
    let p = data::prepare_full_with(
        &typed.table,
        &typed.vars,
        "y",
        &split,
        &mut Mt::new(1),
        &prep,
    )
    .unwrap();
    assert!(p.train.x.iter().all(|r| r.iter().all(|v| *v == 0.0 || *v == 1.0)));

    let mut cfg = Config::default();
    cfg.generations = 80;
    cfg.pop = 300;
    cfg.islands = 2;
    cfg.seed = 1;
    cfg.metric = Metric::Absolute;
    cfg.ops = vec![Op::And, Op::Not];

    let out = run(&cfg, &p.train, &p.test);
    let best = out
        .archive
        .sorted()
        .into_iter()
        .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal))
        .expect("the search returned no candidates");
    assert!(
        best.error < 1e-12,
        "best training error = {:?} for {}",
        best.error,
        expr::to_string(&best.tree, &typed.vars)
    );

    // The recovered law computes the same truth table as x1 AND NOT x2.
    let expected = [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0)];
    for (a, b, y) in expected {
        let got = expr::eval(&best.tree, &[a, b]);
        assert_eq!(got, y, "x1={a}, x2={b}");
    }

    // The true law itself scores zero on the prepared table (the check that does
    // not depend on which equivalent form the search happened to find).
    let truth = expr::node(
        Op::And,
        vec![
            expr::input(0),
            expr::node(Op::Not, vec![expr::input(1)]),
        ],
    );
    assert!(rcsr::search::eval_error(&truth, &p.train, Metric::Absolute) < 1e-12);
}

/// The text report names every input's datatype, and `--json` carries the same
/// information as `variables_detail`.
#[test]
fn report_shows_variable_datatypes() {
    let csv = "flag,y\n0,0\n1,1\n0,1\n1,0\n0,0\n1,1\n";

    let out = run_cli(
        &[
            "--data", "-", "--target", "y", "--dtype", "flag=boolean",
            "--generations", "1", "--pop", "20", "--seed", "1",
        ],
        csv,
    );
    assert_eq!(out.status.code(), Some(0));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("# variables: flag (boolean)"),
        "stderr: {err}"
    );

    let out = run_cli(
        &[
            "--data", "-", "--target", "y", "--dtype", "flag=boolean",
            "--generations", "1", "--pop", "20", "--seed", "1", "--json",
        ],
        csv,
    );
    assert_eq!(out.status.code(), Some(0));
    let json = String::from_utf8_lossy(&out.stdout);
    assert!(
        json.contains("\"variables_detail\":[{\"name\":\"flag\",\"dtype\":\"boolean\",\"source\":null}]"),
        "stdout: {json}"
    );
}
