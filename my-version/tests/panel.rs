//! #145: point-in-time membership, delisted coins, splices, costs and the
//! 2026H2 lockbox.

use rcsr::lockbox::Lockbox;
use rcsr::ohlcv::{self, Bar, Migration};
use rcsr::panel::{
    self, build_membership, build_panel, buy_and_hold, level_jumps, long_gaps,
    matched_exposure_random_long, net_returns, series_from_table, splice_canaries, CostModel,
    Membership, MembershipConfig,
};
use rcsr::walkforward::{self, EvalSpec, Window};
use std::collections::BTreeMap;

const DAY: i64 = 86_400;
/// 2020-01-01T00:00:00Z.
const T0: i64 = 1_577_836_800;

fn bars(days: usize, start_vol: f64, price: f64) -> Vec<Bar> {
    (0..days)
        .map(|i| {
            let p = price * (1.0 + 0.001 * i as f64);
            Bar::new(T0 + i as i64 * DAY, p, p * 1.01, p * 0.99, p, start_vol / p)
                .with_flow(start_vol, 10.0)
        })
        .collect()
}

fn series() -> BTreeMap<String, Vec<Bar>> {
    let mut s = BTreeMap::new();
    s.insert("A".to_string(), bars(180, 100_000.0, 10.0));
    s.insert("B".to_string(), bars(180, 200_000.0, 20.0));
    s.insert("C".to_string(), bars(120, 900_000.0, 5.0)); // delists at day 120
    s
}

fn cfg(top: usize) -> MembershipConfig {
    MembershipConfig {
        rebalance_days: 30,
        lookback_days: 30,
        min_history_days: 30,
        min_bars: 20,
        top,
        min_adv_usd: 0.0,
    }
}

#[test]
fn membership_is_point_in_time_and_ignores_a_future_volume_spike() {
    let mut plain = series();
    let mut spiked = series();
    // B becomes enormous ten days AFTER the first rebalance that can select it.
    let as_ofs = {
        let m = build_membership(&plain, &cfg(2)).unwrap();
        m.as_of
    };
    let target = as_ofs[2];
    for b in spiked.get_mut("B").unwrap() {
        if b.time > target && b.time < target + 10 * DAY {
            b.quote_volume = 1e15;
            b.volume = 1e15 / b.close;
        }
    }
    plain.insert("B".to_string(), bars(180, 200_000.0, 20.0));
    let m1 = build_membership(&plain, &cfg(2)).unwrap();
    let m2 = build_membership(&spiked, &cfg(2)).unwrap();
    // Every decision at or before the spike is identical.
    let early = |m: &Membership| -> Vec<(i64, String, bool)> {
        m.rows
            .iter()
            .filter(|r| r.as_of <= target)
            .map(|r| (r.as_of, r.symbol.clone(), r.member))
            .collect()
    };
    assert_eq!(early(&m1), early(&m2));
    // No decision reads a bar at/after its own as_of.
    for r in &m1.rows {
        let b = plain[&r.symbol]
            .iter()
            .filter(|b| b.time < r.as_of)
            .last()
            .unwrap();
        assert!(b.time < r.as_of);
    }
}

#[test]
fn delisted_coins_are_kept_and_removing_them_changes_the_universe() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let p = build_panel(&s, &m).unwrap();
    assert!(p.symbols.contains("C"), "C must enter while it is alive");
    assert!(p.delisted.contains("C"), "C stops and is reported delisted");
    let c_rows = p.rows.iter().filter(|r| r.symbol == "C").count();
    let expected = s["C"]
        .iter()
        .filter(|b| m.is_member("C", b.time))
        .count();
    assert_eq!(c_rows, expected, "every member C bar is kept, none invented");
    assert!(c_rows > 30);
    assert!(p.rows.iter().filter(|r| r.symbol == "C").all(|r| r.bar.time < T0 + 120 * DAY));
    assert!(p.counts_per_year().values().any(|(n, _)| *n >= 2));

    // The survivor-only variant has a different universe: the filter matters.
    let mut survivor = s.clone();
    survivor.remove("C");
    let m2 = build_membership(&survivor, &cfg(2)).unwrap();
    let p2 = build_panel(&survivor, &m2).unwrap();
    assert_ne!(p.rows.len(), p2.rows.len());
    assert!(!p2.symbols.contains("C"));
    assert!(p.delisted.len() > p2.delisted.len());
}

#[test]
fn no_rows_are_forward_filled_across_a_gap_or_a_listing() {
    let mut s = series();
    // Punch a 15-day hole in A.
    s.get_mut("A").unwrap().drain(100..115);
    let m = build_membership(&s, &cfg(3)).unwrap();
    let p = build_panel(&s, &m).unwrap();
    let a: Vec<i64> = p
        .rows
        .iter()
        .filter(|r| r.symbol == "A")
        .map(|r| r.bar.time)
        .collect();
    assert!(a.windows(2).all(|w| w[0] < w[1]));
    assert!(
        a.windows(2).any(|w| w[1] - w[0] > DAY),
        "the hole stays a hole"
    );
    let gapped = p.rows.iter().find(|r| r.symbol == "A" && r.gap_buckets > 0).unwrap();
    assert_eq!(gapped.gap_buckets, 15);
    assert!(long_gaps(&s, 7).iter().any(|(sym, _, n)| sym == "A" && *n == 15));
    // A feature window over the gap is masked rather than bridged.
    let times: Vec<i64> = s["A"].iter().map(|b| b.time).collect();
    let mask = ohlcv::window_gap_mask(&times, ohlcv::Timeframe::D1, 5);
    assert!(mask.iter().any(|x| *x));
}

#[test]
fn splice_canaries_pass_a_correct_k_and_fail_a_wrong_one() {
    let mut s = BTreeMap::new();
    s.insert(
        "OLD".to_string(),
        vec![Bar::new(T0, 100.0, 100.0, 100.0, 100.0, 10.0)],
    );
    s.insert(
        "NEW".to_string(),
        vec![Bar::new(T0 + DAY, 1.0, 1.0, 1.0, 1.0, 1000.0)],
    );
    let pass = splice_canaries(&s, &[Migration::new("OLD", "NEW", 100.0)]);
    assert!(pass[0].pass, "{:?}", pass[0]);
    let fail = splice_canaries(&s, &[Migration::new("OLD", "NEW", 1.0)]);
    assert!(!fail[0].pass);
    assert!(fail[0].reason.contains("tolerance"));

    // LUNA: an undeclared 20,000x gap is a level jump, not a return.
    let mut luna = BTreeMap::new();
    luna.insert(
        "LUNAUSDT".to_string(),
        vec![
            Bar::new(T0, 0.00005, 0.00005, 0.00005, 0.00005, 1.0),
            Bar::new(T0 + 18 * DAY, 1.0, 1.0, 1.0, 1.0, 1.0),
        ],
    );
    let jumps = level_jumps(&luna, 5.0);
    assert_eq!(jumps.len(), 1);
    assert!(jumps[0].ratio > 10_000.0);
    assert!(ohlcv::splice(
        &luna["LUNAUSDT"][..1],
        &luna["LUNAUSDT"][1..],
        1.0,
        0.005
    )
    .is_err());
}

#[test]
fn label_and_evaluator_costs_agree_and_the_final_exit_is_charged() {
    let cost = CostModel {
        fee_bps: 10.0,
        half_spread_bps: 0.0,
        impact_bps: 0.0,
    };
    // Next-open entry 100 -> close 110, gross log return.
    let gross = [(110.0f64 / 100.0).ln()];
    let positions = [1.0];
    let groups = [0u64];
    let (net, ledger) = net_returns(&gross, &positions, &groups, &cost, true);
    let expect = gross[0] - cost.label_cost();
    assert!((net[0] - expect).abs() < 1e-12, "net {} expect {}", net[0], expect);
    assert!((ledger.total_bps() - cost.round_trip_bps()).abs() < 1e-9);
    assert_eq!(ledger.entries, 1);
    assert_eq!(ledger.exits, 1);

    // Buy-and-hold pays the same entry and exit.
    let (bh, bh_ledger) = buy_and_hold(&gross, &groups, &cost);
    assert!((bh[0] - net[0]).abs() < 1e-12);
    assert!((bh_ledger.total_bps() - cost.round_trip_bps()).abs() < 1e-9);

    // A missing final exit would flatter the strategy by one side.
    let (free_exit, ledger_no_exit) = net_returns(&gross, &positions, &groups, &cost, false);
    assert!((free_exit[0] - (net[0] + cost.per_side_bps() / 10_000.0)).abs() < 1e-12);
    assert!((ledger_no_exit.total_bps() - cost.per_side_bps()).abs() < 1e-9);
}

#[test]
fn matched_exposure_random_long_keeps_the_same_exposure() {
    let positions = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0];
    let groups = [0u64; 6];
    let r = matched_exposure_random_long(&positions, &groups, 42);
    let long = |v: &[f64]| v.iter().filter(|x| **x > 0.0).count();
    assert_eq!(long(&positions), long(&r));
    assert_ne!(r, positions, "the benchmark is a permutation, not the strategy");
    // Determinism.
    assert_eq!(r, matched_exposure_random_long(&positions, &groups, 42));
}

#[test]
fn every_runner_refuses_2026h2_unless_overridden() {
    let l = Lockbox::crypto_2026h2();
    let mut spec = EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "y".to_string(),
        time: "ts".to_string(),
        group: None,
        windows: vec![Window {
            label: "2026H2".to_string(),
            train_start: l.start as f64 - 100.0,
            train_end: l.start as f64 - 10.0,
            test_start: l.start as f64,
            test_end: l.start as f64 + 86_400.0,
        }],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.7,
        validation_frac: 0.0,
        periods_per_year: None,
        allow_lockbox: false,
    };
    // A synthetic table is enough: validation refuses before reading rows.
    let t = rcsr::data::parse_table("ts,x,y\n0,0,0\n1,1,1\n2,2,2\n3,3,3\n").unwrap();
    let err = walkforward::run_table(&t, &spec, "test").unwrap_err();
    assert!(err.contains("lockbox"), "{}", err);
    spec.allow_lockbox = true;
    let _ = walkforward::run_table(&t, &spec, "test");
}

#[test]
fn the_panel_store_and_manifest_record_counts_delisted_and_checksums() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let p = build_panel(&s, &m).unwrap();
    let out = std::env::temp_dir().join(format!("rcsr-panel-test-{}.rcsb", std::process::id()));
    let mut input = std::env::temp_dir().join(format!("rcsr-panel-in-{}.csv", std::process::id()));
    // Write a canonical daily CSV as the provenance input.
    let mut csv = String::from("ts,symbol,open,high,low,close,volume,quote_volume\n");
    for (sym, bs) in &s {
        for b in bs {
            csv.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                b.time, sym, b.open, b.high, b.low, b.close, b.volume, b.quote_volume
            ));
        }
    }
    std::fs::write(&input, csv).unwrap();
    let out_str = out.to_string_lossy().to_string();
    let in_str = input.to_string_lossy().to_string();
    let manifest_path =
        panel::write_panel_panel(&out_str, &p, &m, &[in_str.as_str()], 65_536).unwrap();
    assert!(out.exists());
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(text.contains("\"sha256\""));
    assert!(text.contains("\"delisted\""));
    assert!(text.contains("\"C\""));
    assert!(text.contains("\"symbols_per_year\""));
    assert!(text.contains("\"rebalances\""));
    let _ = std::fs::remove_file(out);
    let _ = std::fs::remove_file(manifest_path);
    let _ = std::fs::remove_file(input);
}

#[test]
fn a_panel_table_round_trips_through_the_series_reader() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let p = build_panel(&s, &m).unwrap();
    let t = panel::panel_table(&p);
    let back = series_from_table(&t).unwrap();
    assert_eq!(back.len(), p.symbols.len());
    for (sym, bs) in &back {
        assert_eq!(bs.len(), p.rows.iter().filter(|r| &r.symbol == sym).count());
    }
}
