//! #125: OHLCV layer — resampling, calendars, actions, splices, MTF, gaps.

use rcsr::features::{compute, expand, forward_label, LabelSpec};
use rcsr::ohlcv::{
    adjust, apply_migrations, gap_buckets, join_causal, join_column_causal, label_gap_mask,
    last_closed_as_of, lookahead_violations, resample, resample_as_of, resample_calendar, splice,
    splice_ratio, visibility, window_gap_mask, ActionKind, Bar, Calendar, CorporateAction,
    Migration, Resampler, Timeframe, Visibility,
};
use std::collections::BTreeMap;

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
    assert_eq!(daily[0].open, 0.0);
    assert_eq!(daily[0].close, 1439.75);
    assert_eq!(daily[0].volume, 1440.0);

    let q = resample(&bars, Timeframe::M15).unwrap();
    assert_eq!(q.len(), 192);
    assert_eq!(q[0].open, 0.0);
    assert_eq!(q[0].close, 14.75);
    assert_eq!(q[0].volume, 15.0);
}

#[test]
fn streaming_resampler_matches_the_batch_result() {
    let bars = minutes(2);
    let mut rs = Resampler::new(Timeframe::M15);
    let mut streamed = Vec::new();
    for b in bars {
        if let Some(s) = rs.push(b) {
            streamed.push(s);
        }
    }
    if let Some(s) = rs.finish() {
        streamed.push(s);
    }
    assert_eq!(streamed, resample(&minutes(2), Timeframe::M15).unwrap());
}

#[test]
fn an_open_bucket_is_incomplete_and_never_forward_filled() {
    let bars = minutes(1);
    let as_of = START + 90 * 60; // 01:30
    let hourly = resample_as_of(&bars[..90], Timeframe::H1, as_of).unwrap();
    assert_eq!(hourly.len(), 2);
    assert!(hourly[0].complete);
    assert!(!hourly[1].complete, "the 01:00 bucket is still open");
    assert_eq!(hourly[1].volume, 30.0);

    // A hole leaves a hole: no synthetic bucket appears.
    let mut with_gap = bars.clone();
    with_gap.drain(60..120);
    let h = resample(&with_gap, Timeframe::H1).unwrap();
    assert_eq!(h.len(), 23);
    assert_eq!(h[0].time, START);
    assert_eq!(h[0].volume, 60.0);
    assert_eq!(h[1].time, START + 2 * 3600, "the 01:00 bucket is absent");
    assert_eq!(gap_buckets(&with_gap, Timeframe::H1), 1);
}

#[test]
fn zero_volume_bars_are_kept_and_flagged() {
    let bars = vec![
        Bar::new(START, 10.0, 11.0, 9.0, 10.5, 0.0),
        Bar::new(START + 60, 10.5, 10.5, 10.5, 10.5, 0.0),
        Bar::new(START + 120, 10.5, 12.0, 10.0, 11.0, 3.0),
    ];
    let h = resample(&bars, Timeframe::H1).unwrap();
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].volume, 3.0);
    assert!(h[0].traded);
    let one = resample(&bars[..2], Timeframe::H1).unwrap();
    assert_eq!(one.len(), 1);
    assert!(!one[0].traded, "stale bars stay in the sample");
}

#[test]
fn mtf_join_cannot_see_the_current_daily_close() {
    let base = minutes(2);
    let hourly = resample(&base, Timeframe::H1).unwrap();
    let daily = resample(&base, Timeframe::D1).unwrap();

    // 10:00 on day one sees the 09:00-10:00 hour, which closed at 10:00.
    let decision = START + 10 * 3600;
    let seen = last_closed_as_of(&hourly, Timeframe::H1, decision).unwrap();
    assert_eq!(seen.time, START + 9 * 3600);
    assert_eq!(seen.close, 599.75);

    // It cannot see the daily bar, which closes at midnight.
    assert!(last_closed_as_of(&daily, Timeframe::D1, decision).is_none());
    let joined = join_causal(&base, &daily, Timeframe::D1).unwrap();
    let at_ten = joined.iter().position(|_| true).unwrap();
    assert!(joined[at_ten].is_none());
    // At the next midnight the previous day is visible.
    let after = last_closed_as_of(&daily, Timeframe::D1, START + 86_400).unwrap();
    assert_eq!(after.time, START);
    assert!(lookahead_violations(&base, &joined, Timeframe::D1).is_empty());
}

#[test]
fn a_deliberately_shifted_future_column_fails_the_join() {
    let base = minutes(2);
    let daily = resample(&base, Timeframe::D1).unwrap();
    let joined = join_causal(&base, &daily, Timeframe::D1).unwrap();
    // Shift the joined series one bucket forward: now every non-null row is
    // using a bar that had not closed at the decision instant.
    let mut future: Vec<Option<Bar>> = Vec::new();
    for slot in &joined {
        future.push(slot.map(|b| {
            let mut x = b;
            x.time += Timeframe::D1.seconds();
            x
        }));
    }
    let bad = lookahead_violations(&base, &future, Timeframe::D1);
    assert!(!bad.is_empty(), "the shifted column must be detected");
    assert_eq!(visibility(&future[bad[0]].unwrap(), Timeframe::D1, base[bad[0]].time), Visibility::Open);
    // The scalar join has the same property.
    let vals: Vec<f64> = daily.iter().map(|b| b.close).collect();
    let col = join_column_causal(&base, &daily, Timeframe::D1, &vals).unwrap();
    assert!(col[10 * 60].is_none());
}

#[test]
fn nyse_close_dst_and_half_days() {
    let cal = Calendar::Nyse;
    // 2024-01-02 is a normal EST session: 14:30-21:00 UTC.
    let s = cal.session_on(2024, 1, 2).unwrap();
    assert_eq!(s.open, 1_704_205_800); // 2024-01-02T14:30:00Z
    assert_eq!(s.close, 1_704_229_200); // 2024-01-02T21:00:00Z
    assert!(!s.half_day);

    // DST: 2024-03-08 closes at 21:00 UTC, 2024-03-11 at 20:00 UTC.
    let before = cal.session_on(2024, 3, 8).unwrap();
    let after = cal.session_on(2024, 3, 11).unwrap();
    assert_eq!(before.close - before.open, 6 * 3600 + 30 * 60);
    assert_eq!(after.close - after.open, 6 * 3600 + 30 * 60);
    assert_eq!(before.open, 1_709_908_200); // 14:30Z
    assert_eq!(after.open, 1_710_163_800); // 13:30Z
    assert!(cal.session_on(2024, 3, 10).is_none(), "Sunday");

    // Half-days: 2024-07-03 and 2024-11-29 close at 13:00 local.
    let july3 = cal.session_on(2024, 7, 3).unwrap();
    assert!(july3.half_day);
    assert_eq!(july3.close - july3.open, 3 * 3600 + 30 * 60);
    let nov29 = cal.session_on(2024, 11, 29).unwrap();
    assert!(nov29.half_day);
    assert_eq!(nov29.close - nov29.open, 3 * 3600 + 30 * 60);

    // Holidays.
    assert!(cal.session_on(2024, 1, 15).is_none(), "MLK");
    assert!(cal.session_on(2024, 3, 29).is_none(), "Good Friday");
    assert!(cal.session_on(2024, 12, 25).is_none(), "Christmas");
    assert!(cal.session_at(july3.open + 60).is_some());
    assert!(cal.session_at(july3.close).is_none(), "close is exclusive");
}

#[test]
fn crypto_24x7_has_no_session_breaks() {
    let cal = Calendar::Crypto247;
    for m in 0..(48 * 60) {
        assert!(cal.is_open(START + m * 60), "crypto must trade every minute");
    }
    let s = cal.session_at(START).unwrap();
    assert_eq!(s.close - s.open, 86_400);
    assert!(!s.half_day);
    // Daily resampling with the crypto calendar equals epoch buckets.
    let base = minutes(2);
    assert_eq!(
        resample_calendar(&base, Timeframe::D1, cal).unwrap(),
        resample(&base, Timeframe::D1).unwrap()
    );
}

#[test]
fn calendar_daily_buckets_open_at_the_session() {
    let cal = Calendar::Nyse;
    // 1m bars inside the 2024-01-02 RTH session.
    let open = cal.session_on(2024, 1, 2).unwrap().open;
    let bars: Vec<Bar> = (0..390)
        .map(|i| {
            let p = 100.0 + i as f64 * 0.1;
            Bar::new(open + i * 60, p, p + 0.2, p - 0.2, p + 0.1, 10.0).with_flow(p * 10.0, 1.0)
        })
        .collect();
    let daily = resample_calendar(&bars, Timeframe::D1, cal).unwrap();
    assert_eq!(daily.len(), 1);
    assert_eq!(daily[0].time, open);
    assert_eq!(daily[0].open, 100.0);
    assert_eq!(daily[0].close, bars.last().unwrap().close);
    assert!((daily[0].volume - 3900.0).abs() < 1e-9);

    // A daily bar cannot cross the session close.
    let bad = vec![Bar::new(open, 1.0, 1.0, 1.0, 1.0, 0.0)];
    assert!(resample_calendar(&bad, Timeframe::D1, cal).is_ok());
    let crossing = vec![Bar::new(open + 7 * 3600, 1.0, 1.0, 1.0, 1.0, 0.0)];
    assert!(
        resample_calendar(&crossing, Timeframe::D1, cal).is_err(),
        "a bar after the close is outside the session"
    );
}

#[test]
fn split_adjusts_prices_and_volume_and_keeps_raw() {
    let day = 86_400;
    let ex = civil::days(2024, 6, 10);
    let mut bars = Vec::new();
    for i in 0..6 {
        let t = (ex - 3 + i) * day;
        let p = if i < 3 { 100.0 } else { 50.0 };
        bars.push(Bar::new(t, p, p + 1.0, p - 1.0, p, 10.0).with_flow(p * 10.0, 1.0));
    }
    let split = CorporateAction {
        ex_date: ex,
        kind: ActionKind::Split { ratio: 2.0 },
    };
    let a = adjust(&[split], &bars).unwrap();
    assert_eq!(a.raw, bars, "raw prices are untouched");
    for i in 0..3 {
        assert!((a.adjusted[i].close - 50.0).abs() < 1e-12);
        assert!((a.adjusted[i].volume - 20.0).abs() < 1e-12);
        assert!((a.price_factor[i] - 0.5).abs() < 1e-12);
    }
    for i in 3..6 {
        assert!((a.adjusted[i].close - 50.0).abs() < 1e-12);
        assert!((a.adjusted[i].volume - 10.0).abs() < 1e-12);
    }
    // A 2-for-1 split leaves the adjusted series continuous.
    assert!((a.adjusted[2].close - a.adjusted[3].close).abs() < 1e-12);
}

#[test]
fn dividend_adjustment_uses_the_prior_close() {
    let day = 86_400;
    let ex = civil::days(2024, 6, 10);
    let bars: Vec<Bar> = (0..4)
        .map(|i| {
            let t = (ex - 2 + i) * day;
            let p = if i < 2 { 100.0 } else { 98.0 };
            Bar::new(t, p, p, p, p, 1.0)
        })
        .collect();
    let div = CorporateAction {
        ex_date: ex,
        kind: ActionKind::Dividend { cash: 2.0 },
    };
    let a = adjust(&[div], &bars).unwrap();
    assert!((a.adjusted[0].close - 98.0).abs() < 1e-12);
    assert_eq!(a.adjusted[2].close, 98.0);
    assert_eq!(a.adjusted[2].volume, 1.0, "cash dividends do not scale volume");
}

#[test]
fn splice_verifies_the_ratio_and_removes_overlap() {
    let day = 86_400;
    let ex = civil::days(2024, 6, 10);
    let old: Vec<Bar> = (0..4)
        .map(|i| {
            let t = (ex - 4 + i) * day;
            Bar::new(t, 100.0, 101.0, 99.0, 100.0, 10.0)
        })
        .collect();
    let new: Vec<Bar> = (0..3)
        .map(|i| {
            let t = (ex - 1 + i) * day;
            Bar::new(t, 1.0, 1.1, 0.9, 1.0, 1000.0)
        })
        .collect();
    assert!((splice_ratio(old.last().unwrap(), new.first().unwrap()) - 100.0).abs() < 1e-12);
    let s = splice(&old, &new, 100.0, 0.005).unwrap();
    // Old overlap at ex-1 is removed; 3 old + 3 new rows remain.
    assert_eq!(s.len(), 6);
    assert_eq!(s[0].close, 1.0);
    assert_eq!(s[0].volume, 1000.0);
    assert_eq!(s[3].time, new[0].time);
    assert_eq!(s[3].close, 1.0);
    assert!(s.windows(2).all(|w| w[0].time < w[1].time));

    // A wrong k is a hard failure, not a 99x fake return.
    let err = splice(&old, &new, 1.0, 0.005).unwrap_err();
    assert!(err.contains("differs from k=1.000000"), "{}", err);
    // TOMO's measured 1.0169 against a declared k=1 fails.
    let mut old2 = old.clone();
    old2.last_mut().unwrap().close = 1.0169;
    assert!(splice(&old2, &new, 1.0, 0.005).is_err());
}

#[test]
fn migrations_splice_all_legs_and_reject_a_missing_source() {
    let day = 86_400;
    let ex = civil::days(2024, 6, 10);
    let mut series = BTreeMap::new();
    series.insert(
        "OLD".to_string(),
        vec![Bar::new(ex - 2 * day, 100.0, 100.0, 100.0, 100.0, 10.0)],
    );
    series.insert(
        "NEW".to_string(),
        vec![Bar::new(ex, 1.0, 1.0, 1.0, 1.0, 1000.0)],
    );
    let out = apply_migrations(&series, &[Migration::new("OLD", "NEW", 100.0)]).unwrap();
    assert!(!out.contains_key("OLD"));
    assert_eq!(out["NEW"].len(), 2);
    assert_eq!(out["NEW"][0].close, 1.0);

    let bad = apply_migrations(&series, &[Migration::new("MISSING", "NEW", 1.0)]);
    assert!(bad.is_err());
}

#[test]
fn gap_masks_cover_the_lookback_and_the_label_window() {
    let mut bars = Vec::new();
    for i in 0..20 {
        let t = if i >= 10 { START + (i as i64 + 5) * 60 } else { START + i as i64 * 60 };
        let p = 100.0 + i as f64;
        bars.push(Bar::new(t, p, p, p, p, 1.0));
    }
    let times: Vec<i64> = bars.iter().map(|b| b.time).collect();
    let w = window_gap_mask(&times, Timeframe::M1, 5);
    assert!(!w[9], "the last row before the gap has a clean lookback");
    assert!(w[10], "the first row after the gap must be masked");
    let l = label_gap_mask(&times, Timeframe::M1, 2);
    assert!(l[9], "a label from row 9 spans the gap");
    assert!(!l[12]);

    let spec = LabelSpec {
        horizon: 2,
        ..Default::default()
    };
    let col = forward_label(&bars, Timeframe::M1, &spec);
    assert!(col.values[9].is_nan());
}

#[test]
fn features_have_a_warmup_and_masked_gap_rows() {
    let mut bars = minutes(2);
    bars[500].time += 60 * 30;
    let cols = compute(&bars, Timeframe::M1, &expand("ta:full,exotic:cheap,candle:geometry,calendar").unwrap()).unwrap();
    assert!(!cols.is_empty());
    for c in &cols {
        assert_eq!(c.values.len(), bars.len());
        assert!(c.values.iter().any(|v| v.is_finite()), "{} is all NaN", c.name);
        if c.warmup > 0 {
            assert!(c.values[0].is_nan(), "{} ignores its warm-up", c.name);
        }
    }
    // The ATR window spanning the gap is missing, not bridged.
    let atr = cols.iter().find(|c| c.name == "atr14").unwrap();
    assert!(atr.values[502].is_nan() || atr.values[505].is_nan());
}

// Small local helper so the test does not depend on a private module.
mod civil {
    pub fn days(y: i32, m: u32, d: u32) -> i64 {
        rcsr::ohlcv::days_from_civil(y, m, d)
    }
}
