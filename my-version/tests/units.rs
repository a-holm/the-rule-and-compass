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
use rcsr::units::{constant_dimensions, infer, Inconsistency, Unit, Units};

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
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
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
fn any_unknown_name_becomes_its_own_base_dimension() {
    let usd = Unit::parse("USD").unwrap();
    let shares = Unit::parse("shares").unwrap();
    assert_ne!(usd, shares);
    assert_eq!(Unit::parse("USD*USD/USD").unwrap(), usd);

    // The point of dynamic dimensions: tabular quantities are not forced into
    // the seven SI names, so `price + volume` is caught.
    let vars = vec![usd, shares];
    let bad = node(Op::Add, vec![input(0), input(1)]);
    assert!(matches!(
        infer(&bad, &vars),
        Err(Inconsistency::Mismatch { op: Op::Add, .. })
    ));

    // Dimensions only: `m*m*m` and `m^3` are the same dimension, with no unit
    // identity or scale factor in between.
    assert_eq!(Unit::parse("m*m*m").unwrap(), Unit::parse("m^3").unwrap());
    assert_eq!(Unit::parse("m/m").unwrap(), Unit::DIMENSIONLESS);
}

#[test]
fn the_parser_rejects_malformed_syntax() {
    for bad in ["", "m^", "m^x", "m^x^2", "(m/s", "m//s", "m s", "m^^2"] {
        assert!(
            Unit::parse(bad).is_err(),
            "'{bad}' must not parse as a dimension expression"
        );
    }
}

#[test]
fn constant_dimensions_are_derived_from_the_fitted_tree() {
    let vars = vec![m(), s()];

    // y = a*x + b with x in m and y in m: a is dimensionless, b is m.
    let linear = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(2.0), input(0)]),
            constant(1.0),
        ],
    );
    assert_eq!(
        constant_dimensions(&linear, &vars, &m()),
        vec![Some(Unit::DIMENSIONLESS), Some(m())]
    );

    // y = a*x/t with x in m, t in s, y in m/s: a is dimensionless.
    let speed = node(
        Op::Div,
        vec![
            node(Op::Mul, vec![constant(2.0), input(0)]),
            input(1),
        ],
    );
    assert_eq!(
        constant_dimensions(&speed, &vars, &m().div(s())),
        vec![Some(Unit::DIMENSIONLESS)]
    );

    // y = a*sin(x) + b with a dimensionless x and y in m: both constants are m.
    let wave = node(
        Op::Add,
        vec![
            node(
                Op::Mul,
                vec![constant(2.0), node(Op::Sin, vec![input(0)])],
            ),
            constant(1.0),
        ],
    );
    assert_eq!(
        constant_dimensions(&wave, &[Unit::DIMENSIONLESS], &m()),
        vec![Some(m()), Some(m())]
    );

    // An unsatisfiable case: `c^3 = m` needs `c = m^(1/3)`, which is not an
    // integer exponent vector, so the filter rejects it.
    let unknown = node(Op::Pow, vec![constant(2.0), constant(3.0)]);
    assert!(!Units::new(Vec::new(), m()).accepts(&unknown));

    // sqrt(c*x) with x in m and y in m is satisfiable: c is an area (m^2),
    // so the root is a length. sqrt(x) alone is not (m^(1/2)).
    let root_law = node(
        Op::Sqrt,
        vec![node(Op::Mul, vec![constant(4.0), input(0)])],
    );
    assert!(Units::new(vec![m()], m()).accepts(&root_law));
    assert_eq!(
        constant_dimensions(&root_law, &[m()], &m()),
        vec![Some(m())]
    );
    assert!(!Units::new(vec![m()], m()).accepts(&node(Op::Sqrt, vec![input(0)])));
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
fn layer2_and_layer3_units_alone_accepts_require_dimensions_rejects() {
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
fn a_fitted_intercept_absorbs_the_target_dimension() {
    // y = 2*x + 1 with x in m and y in m. The intercept's dimension is m;
    // a strict "constants are dimensionless" pass would wrongly reject this.
    let train = sample(
        vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]],
        vec![3.0, 5.0, 7.0, 9.0],
    );
    let test = train.clone();
    let law = node(
        Op::Add,
        vec![
            node(Op::Mul, vec![constant(2.0), input(0)]),
            constant(1.0),
        ],
    );
    let mut cfg = base_config();
    cfg.units = Some(Units::new(vec![m()], m()));
    cfg.require_dimensions = true;
    let out = search::run_seeded(&cfg, &train, &test, std::slice::from_ref(&law));
    assert_eq!(out.seeds[0].error, 0.0);
    assert_eq!(out.archive.items.len(), 1);
    assert_eq!(
        constant_dimensions(&law, &[m()], &m()),
        vec![Some(Unit::DIMENSIONLESS), Some(m())]
    );
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

// ---- layer 1: no --units --------------------------------------------------



/// Mask the nondeterministic timings: `elapsed_seconds` (the old field) and the
/// `timing` phase object added in #134, so the rest of the byte stream can be
/// compared against the recorded golden.
fn mask_elapsed(src: &str) -> String {
    let src = mask_timing(src);
    let key = "\"elapsed_seconds\":";
    let mut out = String::with_capacity(src.len());
    let mut rest = &src[..];
    while let Some(i) = rest.find(key) {
        out.push_str(&rest[..i + key.len()]);
        out.push_str("ELAPSED");
        let after = &rest[i + key.len()..];
        let end = after.find(|c| c == ',' || c == '}').unwrap_or(after.len());
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

fn mask_timing(src: &str) -> String {
    let key = "\"timing\":";
    let mut out = String::with_capacity(src.len());
    let mut rest = &src[..];
    while let Some(i) = rest.find(key) {
        out.push_str(&rest[..i + key.len()]);
        out.push_str("TIMING");
        let after = &rest[i + key.len()..];
        // The golden already contains the placeholder, not an object.
        if after.starts_with("TIMING") {
            rest = &after["TIMING".len()..];
            continue;
        }
        let mut depth = 0usize;
        let mut end = after.len();
        for (n, c) in after.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = n + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        rest = &after[end..];
    }
    out.push_str(rest);
    out
}

/// Layer 1: a units-free run is byte-identical to the recorded golden,
/// including the absence of any units field. The text golden is exact; the
/// JSON/NDJSON goldens are snapshots of this engine's output (regenerated when
/// another feature deliberately adds a field) and mask only `elapsed_seconds`,
/// which is nondeterministic by nature. The units-specific assertions - no
/// `constant_dimensions` field and no filtering - are independent of the
/// snapshot.
#[test]
fn layer1_no_units_output_is_byte_identical_to_pre_change() {
    let table = "x,y\n1,2\n2,4\n3,6\n4,8\n";
    let base = [
        "--data", "-", "--target", "y", "--vars", "x", "--seed-expr", "x",
        "--generations", "0", "--pop", "1", "--fit-iters", "0", "--test-frac",
        "0", "--cpu", "--quiet",
    ];

    let out = rcsr_stdin(&base, table);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(
        out.stdout,
        include_str!("fixtures/units_layer1_text.stdout").as_bytes(),
        "text stdout changed"
    );
    assert_eq!(
        out.stderr,
        include_str!("fixtures/units_layer1_text.stderr").as_bytes(),
        "text stderr changed"
    );

    let mut args = base.to_vec();
    args.push("--json");
    let out = rcsr_stdin(&args, table);
    assert_eq!(out.status.code(), Some(0));
    assert!(out.stderr.is_empty());
    let got = mask_elapsed(std::str::from_utf8(&out.stdout).expect("utf-8"));
    let want = mask_elapsed(include_str!("fixtures/units_layer1_json.stdout"));
    assert_eq!(got, want, "--json changed");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("constant_dimensions"),
        "units-free JSON must not gain a constant_dimensions field"
    );

    let mut args = base.to_vec();
    args.push("--ndjson");
    let out = rcsr_stdin(&args, table);
    assert_eq!(out.status.code(), Some(0));
    assert!(out.stderr.is_empty());
    let got = mask_elapsed(std::str::from_utf8(&out.stdout).expect("utf-8"));
    let want = mask_elapsed(include_str!("fixtures/units_layer1_ndjson.stdout"));
    assert_eq!(got, want, "--ndjson changed");
    assert!(
        !String::from_utf8_lossy(&out.stdout).contains("constant_dimensions"),
        "units-free NDJSON must not gain a constant_dimensions field"
    );
}

/// Layer 1: a units-free checkpoint keeps the old text format exactly; the
/// optional unit fields are absent.
#[test]
fn layer1_units_free_checkpoint_has_no_units_fields() {
    let train = sample(vec![vec![1.0], vec![2.0], vec![3.0]], vec![1.0, 2.0, 3.0]);
    let test = train.clone();
    let mut cfg = base_config();
    cfg.generations = 1;
    let mut text = None;
    search::run_resumable(&cfg, &train, &test, None, 1, |cp| {
        text = Some(cp.to_text());
    })
    .expect("run");
    let text = text.expect("checkpoint");
    assert!(!text.contains("config units"), "unexpected units field");
    assert!(
        !text.contains("require_dimensions"),
        "unexpected require_dimensions field"
    );
    Checkpoint::from_text(&text).expect("old-format checkpoint still parses");
}

// ---- CLI wiring -----------------------------------------------------------


fn rcsr(args: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

fn rcsr_stdin(args: &[&str], table: &str) -> std::process::Output {
    let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("run rcsr");
    use std::io::Write as _;
    child
        .stdin
        .as_mut()
        .expect("stdin")
        .write_all(table.as_bytes())
        .expect("write stdin");
    child.wait_with_output().expect("wait for rcsr")
}

#[test]
fn cli_unknown_names_are_separate_dimensions() {
    let table = "price,volume,y\n10,5,10\n20,3,20\n";
    let out = rcsr_stdin(
        &[
            "--data", "-", "--target", "y", "--vars", "price,volume",
            "--units", "price=USD,volume=shares,y=USD",
            "--require-dimensions", "--seed-expr", "price + volume",
            "--generations", "0", "--pop", "1", "--fit-iters", "0",
            "--test-frac", "0",
        ],
        table,
    );
    assert_eq!(out.status.code(), Some(4), "no finite solution expected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("train_error=inf"),
        "price + volume must be rejected: {stderr}"
    );
}

#[test]
fn cli_rejects_malformed_unit_syntax() {
    let out = rcsr(&[
        "--data", "-", "--target", "y", "--units", "x=m^", "--generations", "0",
    ]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("expected an integer exponent"),
        "stderr: {stderr}"
    );
}

#[test]
fn layer3_require_dimensions_without_units_is_usage_error() {
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

/// Layer 2: `--units` alone reports the constants and keeps a candidate that
/// Layer 3 rejects. `x + 1` is numerically perfect, but `x` is metres and the
/// target is seconds; the additive mismatch cannot be absorbed by a constant.
#[test]
fn layer2_units_alone_reports_and_does_not_reject() {
    let table = "x,y\n1,2\n2,3\n3,4\n4,5\n";
    let run = |extra: &[&str]| {
        let mut args = vec![
            "--data", "-", "--target", "y", "--vars", "x", "--units", "x=m,y=s",
            "--seed-expr", "x + 1", "--generations", "0", "--pop", "1",
            "--fit-iters", "0", "--test-frac", "0", "--center", "none",
            "--scale", "none",
        ];
        args.extend_from_slice(extra);
        let mut child = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args(&args)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("run rcsr");
        use std::io::Write as _;
        child
            .stdin
            .as_mut()
            .expect("stdin")
            .write_all(table.as_bytes())
            .expect("write stdin");
        child.wait_with_output().expect("wait for rcsr")
    };

    // Layer 2: accepted, with a constant-dimension note.
    let out = run(&[]);
    assert_eq!(out.status.code(), Some(0), "units alone must not reject");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("constant dimensions (rank 1): 1:m"),
        "stderr: {stderr}"
    );
    assert!(
        stderr.contains(
            "dimensional note: 1 of 1 archived solution(s) would be rejected"
        ),
        "Layer 2 must note the candidate Layer 3 would reject: {stderr}"
    );

    let out = run(&["--json"]);
    assert_eq!(out.status.code(), Some(0));
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("\"constant_dimensions\":[{\"value\":1,\"dimension\":\"m\"}]"),
        "stdout: {stdout}"
    );

    // Layer 3: the same declaration plus the filter rejects it.
    let out = run(&["--require-dimensions"]);
    assert_eq!(out.status.code(), Some(4), "no finite solution expected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("seed rejected"),
        "the inconsistent seed must be rejected: {stderr}"
    );
    assert!(
        stderr.contains("pareto members: 0"),
        "rejected candidate must not enter the archive: {stderr}"
    );
}

/// Layer 3: a column named `m` is still dimensionless unless declared. The
/// seed fits the numbers exactly, but `x + m` mixes metres with a pure number.
#[test]
fn layer3_undeclared_columns_are_not_inferred_from_their_names() {
    let table = "x,m,y\n1,10,11\n2,20,22\n3,30,33\n";
    let out = rcsr_stdin(
        &[
            "--data", "-", "--target", "y", "--vars", "x,m",
            "--units", "x=m,y=m", "--require-dimensions",
            "--seed-expr", "x + m", "--generations", "0", "--pop", "1",
            "--fit-iters", "0", "--test-frac", "0",
        ],
        table,
    );
    assert_eq!(out.status.code(), Some(4), "no finite solution expected");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("seed rejected"),
        "the undeclared 'm' column must stay dimensionless: {stderr}"
    );
}

/// A hand-built tree with no inputs lets the CLI test the filter without a
/// dataset search: this pins the reported seeded error, not the summary text.
#[test]
fn units_module_exposes_the_target_predicate() {
    let units = Units::new(vec![m()], s());
    assert!(!units.accepts(&input(0)), "m is not s");
    assert!(
        !units.accepts(&node(Op::Div, vec![input(0), input(0)])),
        "m/m is dimensionless, not s"
    );
    // A fitted constant can absorb the difference: x/c with c = m/s is s.
    assert!(units.accepts(&node(
        Op::Div,
        vec![input(0), constant(2.0)]
    )));
    // A bare constant can be assigned the target's dimension.
    assert!(units.accepts(&Node {
        op: Op::Const,
        val: 1.0,
        kids: Vec::new(),
        block: None,
    }));
}
