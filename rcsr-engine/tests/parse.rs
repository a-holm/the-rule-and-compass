//! Integration tests for the expression parser (issue #11) and its wiring.
//!
//! These exercise the public surface: `expr::parse`, the alias table shared
//! with `--ops`, and round-tripping of printed expressions.

use rcsr::expr::{self, Op};

fn vars() -> Vec<String> {
    vec!["x".to_string(), "y".to_string(), "z".to_string()]
}

/// Every primitive name `schema` advertises must be callable in an expression.
#[test]
fn every_primitive_name_parses_as_a_call() {
    for name in Op::names() {
        let op = Op::from_name(name).unwrap_or_else(|| panic!("from_name({name})"));
        let args = std::iter::repeat("x")
            .take(op.arity())
            .collect::<Vec<_>>()
            .join(", ");
        let src = format!("{name}({args})");
        let tree = expr::parse(&src, &vars()).unwrap_or_else(|e| panic!("{src}: {e}"));
        assert_eq!(tree.op, op, "{src} parsed to the wrong op");
    }
}

/// `--ops` parsing and the expression parser share one name table, so they
/// cannot drift; this pins the behaviour they had before the refactor.
#[test]
fn parse_list_agrees_with_from_name() {
    assert_eq!(
        Op::parse_list("sin,cos,min,max"),
        vec![Op::Sin, Op::Cos, Op::Min, Op::Max]
    );
    assert_eq!(Op::parse_list("sin bogus cos"), vec![Op::Sin, Op::Cos]);
    assert_eq!(Op::parse_list("log,sig,-"), vec![Op::Ln, Op::Logistic, Op::Neg]);
    // An all-unknown list falls back to the default operator set.
    assert_eq!(Op::parse_list("bogus"), Op::default_set());
    assert_eq!(Op::parse_list(""), Op::default_set());
}

#[test]
fn errors_report_a_column_within_the_input() {
    let src = "2 * (x + )";
    let e = expr::parse(src, &vars()).unwrap_err();
    assert!(e.column >= 1 && e.column <= src.len(), "{e}");
    let e = expr::parse("nope", &vars()).unwrap_err();
    assert!(e.message.contains("unknown variable"), "{e}");
}

/// Expressions in the exact printed form must parse back to the same tree.
#[test]
fn printed_forms_round_trip() {
    let v = vec!["x".to_string()];
    for s in ["(0.5556 * (0.0342 + x))", "if((x - 1), x, 0)", "sin((x ^ 2))"] {
        let t = expr::parse(s, &v).unwrap_or_else(|e| panic!("{s}: {e}"));
        let printed = expr::to_string(&t, &v);
        let again = expr::parse(&printed, &v).unwrap_or_else(|e| panic!("{printed}: {e}"));
        assert!(expr::same(&t, &again), "{s} did not round-trip via {printed}");
    }
}
