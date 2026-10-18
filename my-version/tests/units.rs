//! Dimensional-consistency constraints (issue #25).
//!
//! Covers the unit parser, the bottom-up inference rules, the hard rejection
//! under `--require-dimensions`, an accepted dimensionally correct law, and
//! the regression that the flag machinery leaves a units-free run unchanged.

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config, Outcome};
use rcsr::units::{infer, Inconsistency, Unit, Units};

fn dims(e: [i32; 7]) -> Unit {
    Unit::from_exponents(e)
}

fn m() -> Unit {
    dims([1, 0, 0, 0, 0, 0, 0])
}

fn s() -> Unit {
    dims([0, 1, 0, 0, 0, 0, 0])
}

fn kg() -> Unit {
    dims([0, 0, 1, 0, 0, 0, 0])
}

fn sample(x: Vec<Vec<f64>>, y: Vec<f64>) -> Sample {
    Sample { x, y, w: Vec::new() }
}

fn describe(out: &Outcome) -> Vec<(u64, u64, u64)> {
    out.archive
        .sorted()
        .iter()
        .map(|s| {
            (
                s.error.to_bits(),
                s.cost.to_bits(),
                rcsr::expr::structural_hash(&s.tree),
            )
        })
        .collect()
}

// ---- parser ---------------------------------------------------------------

#[test]
fn parses_base_dimensions_products_and_integer_powers() {
    assert_eq!(Unit::parse("m").unwrap(), m());
    assert_eq!(Unit::parse("s").unwrap(), s());
    assert_eq!(Unit::parse("kg").unwrap(), kg());
    assert_eq!(Unit::parse("m/s^2").unwrap(), dims([1, -2, 0, 0, 0, 0, 0]));
    assert_eq!(
        Unit::parse("kg*m^2/s^2").unwrap(),
        dims([2, -2, 1, 0, 0, 0, 0]),
        "the SI unit of energy"
    );
    assert_eq!(
        Unit::parse("(m/s)^2").unwrap(),
        dims([2, -2, 0, 0, 0, 0, 0])
    );
    assert_eq!(Unit::parse("m^-2").unwrap(), dims([-2, 0, 0, 0, 0, 0, 0]));
    assert_eq!(Unit::parse("m^0").unwrap(), Unit::DIMENSIONLESS);
    assert_eq!(Unit::parse("m/m").unwrap(), Unit::DIMENSIONLESS);
}

#[test]
fn the_empty_unit_is_one() {
    assert_eq!(Unit::parse("1").unwrap(), Unit::DIMENSIONLESS);
    assert_eq!(Unit::parse("1").unwrap().to_string(), "1");
    assert_eq!(Unit::parse("m/m").unwrap().to_string(), "1");
    assert_eq!(Unit::parse("1/s").unwrap().to_string(), "1/s");
    // Canonical round-trip through the checkpoint encoding.
    for src in ["m/s^2", "kg*m^2/s^2", "1", "1/s", "mol/(A*K)"] {
        let u = Unit::parse(src).unwrap();
        assert_eq!(Unit::parse(&u.to_string()).unwrap(), u, "{src}");
    }
}

#[test]
fn the_parser_rejects_malformed_units() {
    for bad in ["", "bogus", "m^", "m^x", "m^x^2", "(m/s", "m//s", "m s"] {
        assert!(
            Unit::parse(bad).is_err(),
            "'{bad}' must not parse as a unit"
        );
    }
}

// ---- inference ------------------------------------------------------------

#[test]
fn products_and_quotients_combine_exponents() {
    let vars = vec![m(), s(), kg()];
    let product = node(Op::Mul, vec![input(0), input(1)]);
    assert_eq!(infer(&product, &vars).unwrap(), m().mul(s()));

    let quotient = node(Op::Div, vec![input(0), input(1)]);
    assert_eq!(infer(&quotient, &vars).unwrap(), m().div(s()));
    assert_eq!(infer(&quotient, &vars).unwrap().to_string(), "m/s");
}

#[test]
fn addition_requires_equal_dimensions() {
    let vars = vec![m(), s(), kg()];
    let good = node(Op::Add, vec![input(0), node(Op::Mul, vec![input(0), constant(2.0)])]);
    assert_eq!(infer(&good, &vars).unwrap(), m());

    let bad = node(Op::Add, vec![input(0), input(1)]);
    assert!(matches!(
        infer(&bad, &vars),
        Err(Inconsistency::Mismatch { op: Op::Add, .. })
    ));
}

#[test]
fn transcendental_functions_need_a_pure_number() {
    let vars = vec![m(), Unit::DIMENSIONLESS];
    for op in [
        Op::Exp,
        Op::Ln,
        Op::Sin,
        Op::Cos,
        Op::Tanh,
        Op::Logistic,
        Op::Erf,
        Op::Floor,
        Op::Sign,
        Op::Step,
    ] {
        let tree = node(op, vec![input(0)]);
        assert!(
            matches!(
                infer(&tree, &vars),
                Err(Inconsistency::NotDimensionless { op: got, .. }) if got == op
            ),
            "{op:?} accepted a dimensioned argument"
        );
        let pure = node(op, vec![input(1)]);
        assert_eq!(infer(&pure, &vars).unwrap(), Unit::DIMENSIONLESS);
    }
}

#[test]
fn powers_need_a_dimensionless_exponent_and_an_integer_base_power() {
    let vars = vec![m(), Unit::DIMENSIONLESS];
    let square = node(Op::Pow, vec![input(0), constant(2.0)]);
    assert_eq!(infer(&square, &vars).unwrap(), m().pow(2));

    let inverse = node(Op::Pow, vec![input(0), constant(-1.0)]);
    assert_eq!(infer(&inverse, &vars).unwrap(), Unit::DIMENSIONLESS.div(m()));

    let dimensionless_base = node(Op::Pow, vec![input(1), constant(2.0)]);
    assert_eq!(
        infer(&dimensionless_base, &vars).unwrap(),
        Unit::DIMENSIONLESS
    );

    let fractional = node(Op::Pow, vec![input(0), constant(0.5)]);
    assert!(matches!(
        infer(&fractional, &vars),
        Err(Inconsistency::NonIntegerPower { .. })
    ));

    let variable_exponent = node(Op::Pow, vec![input(0), input(1)]);
    assert!(matches!(
        infer(&variable_exponent, &vars),
        Err(Inconsistency::NonIntegerPower { .. })
    ));

    let dimensioned_exponent = node(Op::Pow, vec![input(1), input(0)]);
    assert!(matches!(
        infer(&dimensioned_exponent, &vars),
        Err(Inconsistency::NotDimensionless { op: Op::Pow, .. })
    ));
}

#[test]
fn roots_and_branches_follow_the_algebra() {
    let vars = vec![m(), s()];
    let sqrt_square = node(
        Op::Sqrt,
        vec![node(Op::Pow, vec![input(0), constant(2.0)])],
    );
    assert_eq!(infer(&sqrt_square, &vars).unwrap(), m());
    assert!(matches!(
        infer(&node(Op::Sqrt, vec![input(0)]), &vars),
        Err(Inconsistency::NonIntegerRoot { .. })
    ));

    let condition = node(Op::Greater, vec![input(0), input(0)]);
    assert_eq!(infer(&condition, &vars).unwrap(), Unit::DIMENSIONLESS);

    let good = node(Op::If, vec![condition.clone(), input(0), input(0)]);
    assert_eq!(infer(&good, &vars).unwrap(), m());

    let bad = node(Op::If, vec![condition, input(0), input(1)]);
    assert!(matches!(
        infer(&bad, &vars),
        Err(Inconsistency::Mismatch { op: Op::If, .. })
    ));

    let dim_condition = node(Op::If, vec![input(0), input(0), input(0)]);
    assert!(matches!(
        infer(&dim_condition, &vars),
        Err(Inconsistency::NotDimensionless { op: Op::If, .. })
    ));
}

#[test]
fn undeclared_inputs_are_dimensionless() {
    let vars = vec![m()];
    let tree = node(Op::Add, vec![input(0), input(1)]);
    assert!(matches!(
        infer(&tree, &vars),
        Err(Inconsistency::Mismatch { .. })
    ));
    // input(1) alone has no declaration, so it is dimensionless.
    assert_eq!(infer(&input(1), &vars).unwrap(), Unit::DIMENSIONLESS);
}

// ---- search integration ---------------------------------------------------

fn base_config() -> Config {
    Config {
        generations: 0,
        islands: 1,
        pop: 1,
        fit_iters: 0,
        patience: 0,
        metric: Metric::Absolute,
        seed: 7,
        ..Default::default()
    }
}

/// `y = x` is numerically perfect, but the target is declared in seconds and
/// `x` in metres: the law is dimensionally wrong.
fn wrong_units() -> Units {
    Units::new(vec![m()], s())
}

#[test]
fn a_numerically_good_but_dimensionally_wrong_seed_is_rejected() {
    let train = sample(
        vec![vec![1.0], vec![2.0], vec![3.0]],
        vec![1.0, 2.0, 3.0],
    );
    let test = train.clone();
    let law = input(0);

    // Without the hard filter the same tree fits perfectly.
    let mut off = base_config();
    off.units = Some(wrong_units());
    off.require_dimensions = false;
    let accepted = search::run_seeded(&off, &train, &test, std::slice::from_ref(&law));
    assert!(accepted.seeds[0].error < 1e-12, "numerically perfect");
    assert_eq!(accepted.archive.items.len(), 1);

    // With --require-dimensions it is rejected with infinite error and never
    // reaches the archive.
    let mut on = base_config();
    on.units = Some(wrong_units());
    on.require_dimensions = true;
    let rejected = search::run_seeded(&on, &train, &test, std::slice::from_ref(&law));
    assert!(
        rejected.seeds[0].error.is_infinite(),
        "rejected candidate must carry an infinite error, got {}",
        rejected.seeds[0].error
    );
    assert!(
        rejected.archive.items.is_empty(),
        "an infinite-error candidate must not enter the Pareto front"
    );
}

#[test]
fn a_dimensionally_correct_law_is_accepted() {
    // y = x / t with units m / s.
    let train = sample(
        vec![vec![2.0, 1.0], vec![6.0, 2.0], vec![8.0, 4.0]],
        vec![2.0, 3.0, 2.0],
    );
    let test = train.clone();
    let law = node(Op::Div, vec![input(0), input(1)]);

    let mut cfg = base_config();
    cfg.units = Some(Units::new(vec![m(), s()], m().div(s())));
    cfg.require_dimensions = true;
    let out = search::run_seeded(&cfg, &train, &test, &[law]);
    assert_eq!(out.seeds[0].error, 0.0);
    assert_eq!(out.archive.items.len(), 1);
}

#[test]
fn the_full_data_rebuild_does_not_resurrect_a_rejected_candidate() {
    let train = sample(
        vec![vec![1.0], vec![2.0], vec![3.0]],
        vec![1.0, 2.0, 3.0],
    );
    let test = train.clone();
    let mut cfg = base_config();
    cfg.units = Some(wrong_units());
    cfg.require_dimensions = true;
    // Subsampling triggers the final full-data archive rebuild, which must
    // apply the filter too rather than re-scoring and admitting the seed.
    cfg.predictor_samples = Some(2);
    let out = search::run_seeded(&cfg, &train, &test, &[input(0)]);
    assert!(out.seeds[0].error.is_infinite());
    assert!(out.archive.items.is_empty());
}

#[test]
fn units_survive_a_checkpoint_round_trip() {
    let train = sample(vec![vec![2.0, 1.0]], vec![2.0]);
    let test = train.clone();
    let units = Units::new(vec![m(), s()], m().div(s()));
    let mut cfg = base_config();
    cfg.generations = 1;
    cfg.units = Some(units.clone());
    cfg.require_dimensions = true;

    let mut text: Option<String> = None;
    search::run_resumable(&cfg, &train, &test, None, 0, |cp| {
        text = Some(cp.to_text());
    })
    .expect("run");
    let cp = Checkpoint::from_text(text.as_deref().expect("checkpoint")).expect("parse");
    assert_eq!(cp.config.units, Some(units));
    assert!(cp.config.require_dimensions);
}

#[test]
fn without_required_dimensions_the_result_is_bit_identical() {
    let n = 60;
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 13) % n) as f64) / (n as f64) * 3.0 - 1.0;
        x.push(vec![a, b]);
        y.push(2.0 * a - b + 0.5);
    }
    let train = sample(x, y);
    let test = train.clone();

    let mut plain = base_config();
    plain.generations = 8;
    plain.pop = 24;
    plain.init_depth = 2;
    plain.ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Neg, Op::Sin, Op::Exp];
    plain.seed = 11;

    let mut declared = plain.clone();
    // The declaration is deliberately wrong: without --require-dimensions it
    // must not influence the search at all.
    declared.units = Some(wrong_units());
    declared.require_dimensions = false;

    let a = search::run(&plain, &train, &test);
    let b = search::run(&declared, &train, &test);
    assert_eq!(
        describe(&a),
        describe(&b),
        "declaring units without requiring them must not change the result"
    );
    assert_eq!(a.generations, b.generations);
}

// ---- CLI wiring -----------------------------------------------------------

fn rcsr(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn cli_rejects_a_malformed_unit() {
    let out = rcsr(&[
        "--data", "-", "--target", "y", "--units", "x=bogus", "--generations", "0",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("unknown unit 'bogus'"), "stderr: {stderr}");
}

#[test]
fn cli_requires_units_for_require_dimensions() {
    let out = rcsr(&[
        "--data", "-", "--target", "y", "--require-dimensions", "--generations", "0",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("--require-dimensions needs --units"),
        "stderr: {stderr}"
    );
}

/// The names resolve against the real variable list, and one comma-separated
/// declaration reaches the summary.
#[test]
fn cli_resolves_units_and_reports_them() {
    let table = "x,t,y\n1,1,1\n2,2,1\n4,2,2\n";
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", "-",
            "--target", "y",
            "--vars", "x,t",
            "--units", "x=m,t=s,y=m/s",
            "--generations", "0",
            "--pop", "1",
            "--fit-iters", "0",
            "--test-frac", "0",
            "--seed-expr", "x/t",
        ])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write as _;
            child
                .stdin
                .as_mut()
                .expect("stdin")
                .write_all(table.as_bytes())?;
            child.wait_with_output()
        })
        .expect("run rcsr");
    assert_eq!(out.status.code(), Some(0));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("units: target 'y' is m/s"), "stderr: {stderr}");
    assert!(
        stderr.contains("not filtering; pass --require-dimensions"),
        "declaring without requiring must be reported: {stderr}"
    );
}

/// A hand-built tree with no inputs lets the CLI test the filter without a
/// dataset search: this pins the reported seeded error, not the summary text.
#[test]
fn units_module_exposes_the_target_predicate() {
    let units = Units::new(vec![m()], s());
    assert!(!units.accepts(&input(0)), "m is not s");
    assert!(
        !units.accepts(&node(Op::Div, vec![input(0), constant(2.0)])),
        "m is still not s"
    );
    assert!(
        !units.accepts(&Node {
            op: Op::Const,
            val: 1.0,
            kids: Vec::new(),
        }),
        "a constant is dimensionless, not s"
    );
}
