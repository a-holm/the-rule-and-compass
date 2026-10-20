//! #110: time-series hazards must be explicit. Smoothing is trailing by
//! default, interpolation is refused on time-ordered data, an ignored embargo
//! is warned about, time units/duplicates/monotonicity are reported, and `&&`
//! is accepted by the selector parser.

use rcsr::data::{self, FillMethod, PrepSpec, SplitSpec, SplitMode};
use rcsr::expr;
use rcsr::parse;
use rcsr::rng::Mt;

fn temporal(time_col: Option<usize>) -> SplitSpec {
    SplitSpec {
        mode: SplitMode::Temporal,
        time_col,
        test_frac: 0.3,
        ..Default::default()
    }
}

fn prep() -> PrepSpec {
    PrepSpec::default()
}

#[test]
fn interpolation_is_refused_on_time_ordered_data_and_can_be_allowed() {
    let t = data::parse_table("t,x,y\n0,1,1\n1,,2\n2,3,3\n3,,4\n4,5,5\n").unwrap();
    let err =
        data::prepare_full_with(&t, &["x".into()], "y", &temporal(Some(0)), &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Interpolate,
                ..prep()
            })
            .err()
            .expect("interpolating a time series must be refused");
    assert!(err.contains("interpolate") && err.contains("look-ahead"), "{err}");
    assert!(err.contains("--allow-interpolate") && err.contains("previous"), "{err}");

    // The same table with an explicit opt-in still runs.
    let p = data::prepare_full_with(
        &t,
        &["x".into()],
        "y",
        &temporal(Some(0)),
        &mut Mt::new(1),
        &PrepSpec {
            fill: FillMethod::Interpolate,
            allow_interpolate: true,
            ..prep()
        },
    )
    .unwrap();
    assert_eq!(p.prep.fill, "interpolate");
    assert!(p.prep.rows_filled >= 1, "{:?}", p.prep.rows_filled);
}

#[test]
fn a_selector_split_warns_that_the_embargo_is_ignored() {
    let t = data::parse_table("t,x,y\n0,1,1\n1,2,2\n2,3,3\n3,4,4\n").unwrap();
    let sp = SplitSpec {
        gap: 2,
        train_select: Some(data::Filter {
            tree: parse::parse("t < 2", &["t".into(), "x".into()]).unwrap(),
            vars: vec!["t".into(), "x".into()],
            expression: "t < 2".into(),
        }),
        ..Default::default()
    };
    let p = data::prepare_full_with(&t, &["x".into()], "y", &sp, &mut Mt::new(1), &prep()).unwrap();
    let notes = p.report.notes.join("\n");
    assert!(notes.contains("embargo gap does not apply"), "{notes}");
}

#[test]
fn time_unit_duplicates_and_monotonicity_appear_in_the_preparation_block() {
    // Milliseconds; one instant repeats and one step goes backwards.
    let t = data::parse_table(
        "t,x,y\n\
         1673343780000,1,1\n\
         1673343780000,2,2\n\
         1673343840000,3,3\n\
         1673343780000,4,4\n\
         1673343900000,5,5\n",
    )
    .unwrap();
    let sp = temporal(Some(0));
    let p = data::prepare_full_with(&t, &["x".into()], "y", &sp, &mut Mt::new(1), &prep()).unwrap();
    assert_eq!(p.prep.time_unit, "ms");
    assert_eq!(p.prep.time_rows, 5);
    assert_eq!(p.prep.time_duplicates, 2, "two later rows repeat the first instant");
    assert_eq!(p.prep.time_non_monotonic, 1);
    assert_eq!(p.prep.time_converted, 5);

    let lines = p.prep.text_lines(None, 0).join("\n");
    assert!(lines.contains("time unit=ms"), "{lines}");
    assert!(lines.contains("duplicate time instant"), "{lines}");
    assert!(lines.contains("not monotonic"), "{lines}");

    let json = p.prep.json();
    assert!(json.contains("\"time_unit\":\"ms\""), "{json}");
    assert!(json.contains("\"time_duplicates\":2"), "{json}");
    assert!(json.contains("\"time_non_monotonic\":1"), "{json}");
    assert!(json.contains("\"time_converted\":5"), "{json}");
}

#[test]
fn a_plain_seconds_time_column_reports_unit_s_and_no_conversion() {
    let t = data::parse_table("t,x,y\n0,1,1\n1,2,2\n2,3,3\n").unwrap();
    let p = data::prepare_full_with(&t, &["x".into()], "y", &temporal(Some(0)), &mut Mt::new(1), &prep())
        .unwrap();
    assert_eq!(p.prep.time_unit, "s");
    assert_eq!(p.prep.time_converted, 0);
    assert_eq!(p.prep.time_duplicates, 0);
}

#[test]
fn the_selector_parser_accepts_and_and_the_symbol() {
    let vars = vec!["x".to_string(), "y".to_string()];
    let a = parse::parse("x > 1 and y < 2", &vars).unwrap();
    let b = parse::parse("x > 1 && y < 2", &vars).unwrap();
    let c = parse::parse("x > 1 || y < 2", &vars).unwrap();
    assert_eq!(expr::to_string(&a, &vars), expr::to_string(&b, &vars));
    assert!(expr::to_string(&c, &vars).contains("or"));
}
