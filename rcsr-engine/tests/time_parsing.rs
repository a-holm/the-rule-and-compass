//! #130: strict ISO 8601 offsets, an explicit date-only policy and numeric
//! unit detection (s/ms/us/ns), with mixed formats refused.

use rcsr::data::{self, DateOnlyPolicy, TimeSpec, TimeUnit};

fn spec() -> TimeSpec {
    TimeSpec::default()
}

fn table(csv: &str) -> data::Table {
    data::parse_table(csv).unwrap()
}

#[test]
fn iso_offsets_are_honoured_and_z_means_the_same_instant() {
    let local = data::parse_time("2024-01-02T09:30:00-05:00").unwrap();
    let utc = data::parse_time("2024-01-02T14:30:00Z").unwrap();
    assert_eq!(local, 1_704_205_800.0);
    assert_eq!(local, utc);
    // A positive offset converts the other way.
    assert_eq!(
        data::parse_time("2024-01-02T20:30:00+06:00").unwrap(),
        local
    );
    // The old bug: the wall clock alone (interpreted as UTC) is four hours off.
    assert_ne!(data::parse_time("2024-01-02T09:30:00").unwrap(), local);
}

#[test]
fn date_only_default_is_an_error_and_policies_are_explicit() {
    let t = table("t,y\n2024-01-02,1\n2024-01-03,2\n");
    let err = data::time_keys_with(&t, 0, &spec()).unwrap_err();
    assert!(err.contains("date-only") && err.contains("--date-only"), "{err}");

    let utc = data::time_keys_with(
        &t,
        0,
        &TimeSpec {
            date_only: DateOnlyPolicy::UtcMidnight,
            ..spec()
        },
    )
    .unwrap();
    assert_eq!(utc.0, vec![1_704_153_600.0, 1_704_240_000.0]);

    let close = data::time_keys_with(
        &t,
        0,
        &TimeSpec {
            date_only: DateOnlyPolicy::SessionClose,
            ..spec()
        },
    )
    .unwrap();
    assert_eq!(close.0[0] - utc.0[0], 21.0 * 3600.0);
    assert!(close.1.notes.iter().any(|n| n.contains("session-close")));
}

#[test]
fn numeric_ms_us_ns_are_detected_and_scaled_to_seconds() {
    let cases = [
        ("t,y\n1700000000,1\n1700000001,2\n", TimeUnit::Seconds, 1.0),
        ("t,y\n1700000000000,1\n1700000001000,2\n", TimeUnit::Millis, 1e-3),
        ("t,y\n1700000000000000,1\n1700000001000000,2\n", TimeUnit::Micros, 1e-6),
        (
            "t,y\n1700000000000000000,1\n1700000001000000000,2\n",
            TimeUnit::Nanos,
            1e-9,
        ),
    ];
    for (csv, unit, factor) in cases {
        let t = table(csv);
        let (keys, rep) = data::time_keys_with(&t, 0, &spec()).unwrap();
        assert_eq!(rep.unit, unit.name(), "{csv}");
        assert!((keys[0] - 1_700_000_000.0).abs() < 1e-6, "{csv}: {keys:?}");
        assert!((keys[1] - keys[0] - 1.0).abs() < 1e-6, "{csv}: {keys:?}");
        if unit != TimeUnit::Seconds {
            assert_eq!(rep.scaled_rows, 2);
            assert!(rep.notes.iter().any(|n| n.contains("scaled to seconds")));
        }
        let _ = factor;
    }
}

#[test]
fn a_column_that_mixes_units_or_formats_is_refused_with_a_row() {
    // Seconds and milliseconds in one column: no single unit fits.
    let mixed_units = table("t,y\n1700000000,1\n1700000000000,2\n");
    let err = data::time_keys_with(&mixed_units, 0, &spec()).unwrap_err();
    assert!(err.contains("fits no single epoch unit") && err.contains("--time-unit"), "{err}");

    // Explicit --time-unit resolves a column whose values look small.
    let (keys, rep) = data::time_keys_with(
        &mixed_units,
        0,
        &TimeSpec {
            unit: Some(TimeUnit::Millis),
            ..spec()
        },
    )
    .unwrap();
    assert_eq!(rep.unit, "ms");
    assert!((keys[0] - 1_700_000.0).abs() < 1e-6);

    // Unpadded and padded dates sort wrongly when treated as text; refuse.
    let mixed_dates = table("t,y\n2024-1-2T00:00:00,1\n2024-01-03T00:00:00,2\n");
    let err = data::time_keys_with(&mixed_dates, 0, &spec()).unwrap_err();
    assert!(err.contains("mixed time formats") && err.contains("data row 2"), "{err}");
}

#[test]
fn mixed_offsets_need_an_explicit_tz_and_are_then_reported() {
    let t = table("t,y\n2024-01-02T09:30:00-05:00,1\n2024-01-03T09:30:00,2\n");
    let err = data::time_keys_with(&t, 0, &spec()).unwrap_err();
    assert!(err.contains("mixes offset-bearing") && err.contains("--tz"), "{err}");

    let (keys, rep) = data::time_keys_with(
        &t,
        0,
        &TimeSpec {
            tz: Some(0),
            ..spec()
        },
    )
    .unwrap();
    assert_eq!(rep.offset_rows, 1);
    assert_eq!(keys[0], 1_704_205_800.0);
    assert_eq!(keys[1], 1_704_274_200.0);
    assert!(rep.notes.iter().any(|n| n.contains("converted")));
}

#[test]
fn dst_ambiguous_local_times_follow_the_offset_not_a_guess() {
    // 2024-11-03 01:30 occurs twice in New York. Two offsets, two instants.
    let edt = data::parse_time("2024-11-03T01:30:00-04:00").unwrap();
    let est = data::parse_time("2024-11-03T01:30:00-05:00").unwrap();
    assert_eq!(est - edt, 3600.0);
    // An offset-less value uses --tz when given; a named zone is refused
    // rather than guessed from a missing tz database.
    assert_eq!(
        data::parse_time("2024-11-03T01:30:00-05:00").unwrap(),
        1_730_615_400.0
    );
    assert!(TimeSpec::parse_tz("America/New_York")
        .unwrap_err()
        .contains("fixed offset"));
    assert_eq!(TimeSpec::parse_tz("-05:00").unwrap(), -5 * 3600);
    assert_eq!(TimeSpec::parse_tz("+0530").unwrap(), 5 * 3600 + 30 * 60);
}

#[test]
fn plain_epoch_seconds_are_bit_identical() {
    let t = table("t,y\n1700000000,1\n1700000001,2\n1700000002,3\n");
    let (keys, rep) = data::time_keys_with(&t, 0, &spec()).unwrap();
    assert_eq!(keys, vec![1_700_000_000.0, 1_700_000_001.0, 1_700_000_002.0]);
    assert_eq!(rep.unit, "s");
    assert_eq!(rep.scaled_rows, 0);
}

#[test]
fn offset_less_iso_is_reported_as_utc_not_assumed_silently() {
    let t = table("t,y\n2024-01-02T09:30:00,1\n2024-01-03T09:30:00,2\n");
    let (keys, rep) = data::time_keys_with(&t, 0, &spec()).unwrap();
    assert_eq!(keys[0], 1_704_187_800.0); // 09:30 UTC
    assert!(rep
        .notes
        .iter()
        .any(|n| n.contains("interpreted as UTC")));
    assert!(rep.text_lines().iter().any(|l| l.contains("format=iso8601")));
    assert!(rep.json().contains("\"offset_rows\":0"));
}
