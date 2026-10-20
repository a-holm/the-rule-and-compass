//! #125: OHLCV types, causal resampling and multi-timeframe joins.

use rcsr::ohlcv::{
    join_causal, last_closed_as_of, lookahead_violations, resample, resample_as_of, visibility,
    Bar, Timeframe, Visibility,
};

const START: i64 = 1_704_067_200; // 2024-01-01T00:00:00Z

/// Two days of synthetic 1m bars. Minute `i` has open=i, high=i+1, low=i-0.5,
/// close=i+0.75, volume=1, quote=2, trades=1.
fn minutes(days: i64) -> Vec<Bar> {
    (0..days * 24 * 60)
        .map(|i| {
            let f = i as f64;
            Bar::new(START + i * 60, f, f + 1.0, f - 0.5, f + 0.75, 1.0)
                .with_flow(2.0, 1.0)
        })
        .collect()
}

#[test]
fn one_minute_resamples_to_known_hourly_and_daily_aggregates() {
    let bars = minutes(2);
    let hourly = resample(&bars, Timeframe::H1).unwrap();
    assert_eq!(hourly.len(), 48);
    let h0 = hourly[0];
    assert_eq!(h0.time, START);
    assert_eq!(h0.open, 0.0);
    assert_eq!(h0.high, 60.0);
    assert_eq!(h0.low, -0.5);
    assert_eq!(h0.close, 59.75);
    assert_eq!(h0.volume, 60.0);
    assert_eq!(h0.quote_volume, 120.0);
    assert_eq!(h0.trades, 60.0);
    assert!(h0.complete && h0.traded);
    assert_eq!(h0.vwap(), Some(2.0));

    let daily = resample(&bars, Timeframe::D1).unwrap();
    assert_eq!(daily.len(), 2);
    assert_eq!(daily[0].time, START);
    assert_eq!(daily[0].open, 0.0);
    assert_eq!(daily[0].high, 1440.0);
    assert_eq!(daily[0].low, -0.5);
    assert_eq!(daily[0].close, 1439.75);
    assert_eq!(daily[0].volume, 1440.0);
    assert_eq!(daily[1].open, 1440.0);
    assert_eq!(daily[1].close, 2879.75);
}

#[test]
fn a_15m_decision_at_10_00_cannot_see_the_current_daily_close() {
    let bars = minutes(2);
    let base = resample(&bars, Timeframe::M15).unwrap();
    let daily = resample(&bars, Timeframe::D1).unwrap();
    let joined = join_causal(&base, &daily, Timeframe::D1).unwrap();

    // Day 1, 10:00: the daily bucket has not closed -> no visible daily bar.
    let day1_10 = base
        .iter()
        .position(|b| b.time == START + 10 * 3600)
        .unwrap();
    assert!(joined[day1_10].is_none());
    assert!(lookahead_violations(&base, &joined, Timeframe::D1).is_empty());

    // Day 2, 10:00: the last closed daily bar is day 1, not day 2.
    let day2_10 = base
        .iter()
        .position(|b| b.time == START + 86_400 + 10 * 3600)
        .unwrap();
    let seen = joined[day2_10].expect("day 1 has closed");
    assert_eq!(seen.time, START);
    assert_eq!(seen.close, 1439.75, "must be day 1's close");

    // A deliberate future column (the current daily bar) is what the test
    // would catch: day 1 10:00 would then see day 1's full close.
    let mut future = vec![None; base.len()];
    for (i, b) in base.iter().enumerate() {
        future[i] = daily.iter().rev().find(|d| d.time <= b.time).copied();
    }
    assert_eq!(future[day1_10].unwrap().close, 1439.75);
    assert!(
        !lookahead_violations(&base, &future, Timeframe::D1).is_empty(),
        "a shifted/future daily column must be reported"
    );
    assert_eq!(visibility(&daily[0], Timeframe::D1, START + 3600), Visibility::Open);
    assert_eq!(visibility(&daily[0], Timeframe::D1, START + 86_400), Visibility::Visible);
}

#[test]
fn an_open_bucket_is_flagged_and_never_joined() {
    let bars = minutes(1);
    let hourly = resample_as_of(&bars, Timeframe::H1, START + 1800).unwrap();
    assert_eq!(hourly.len(), 1);
    assert!(!hourly[0].complete, "the 00:00 hour has not closed at 00:30");
    assert!(last_closed_as_of(&hourly, Timeframe::H1, START + 1800).is_none());
    assert_eq!(
        last_closed_as_of(&hourly, Timeframe::H1, START + 3600).unwrap().time,
        START
    );
}

#[test]
fn gaps_are_preserved_and_zero_volume_bars_are_flagged_not_dropped() {
    // Minute 0 and minute 60 only: the 15m buckets in between must not appear.
    let sparse = vec![
        Bar::new(START, 1.0, 1.0, 1.0, 1.0, 0.0),
        Bar::new(START + 3600, 2.0, 2.0, 2.0, 2.0, 0.0),
    ];
    let out = resample(&sparse, Timeframe::M15).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].time, START);
    assert_eq!(out[1].time, START + 3600);
    assert!(out.iter().all(|b| !b.traded), "zero volume is kept and flagged");
}

#[test]
fn unsorted_duplicate_or_unaligned_input_is_refused() {
    let mut bars = minutes(1);
    bars.swap(0, 1);
    assert!(resample(&bars, Timeframe::H1).unwrap_err().contains("strictly increasing"));

    let base = resample(&minutes(1), Timeframe::M15).unwrap();
    let bad = vec![Bar::new(START + 1, 0.0, 0.0, 0.0, 0.0, 0.0)];
    let err = join_causal(&base, &bad, Timeframe::H1).unwrap_err();
    assert!(err.contains("not aligned"), "{err}");
}

#[test]
fn timeframe_names_round_trip() {
    for tf in [
        Timeframe::M1,
        Timeframe::M3,
        Timeframe::M5,
        Timeframe::M15,
        Timeframe::M30,
        Timeframe::H1,
        Timeframe::H2,
        Timeframe::H4,
        Timeframe::D1,
        Timeframe::W1,
    ] {
        assert_eq!(Timeframe::parse(tf.name()).unwrap(), tf);
    }
    assert!(Timeframe::parse("7m").is_err());
    assert_eq!(Timeframe::M15.bucket_start(START + 899), START);
    assert_eq!(Timeframe::M15.bucket_start(START + 900), START + 900);
}
