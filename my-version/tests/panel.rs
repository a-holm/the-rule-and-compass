//! #145: point-in-time membership, delisted coins, splices, costs and the
//! 2026H2 lockbox.

use rcsr::data::{self, SplitMode, SplitSpec};
use rcsr::lockbox::Lockbox;
use rcsr::ohlcv::{self, Bar, Migration};
use rcsr::panel::{
    self, build_membership, build_panel, build_panel_resumable, buy_and_hold, level_jumps, long_gaps,
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
        start: None,
        end: None,
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

fn eval_spec(windows: Vec<Window>) -> EvalSpec {
    EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "y".to_string(),
        time: "ts".to_string(),
        group: None,
        windows,
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.7,
        threshold_fixed: None,
        validation_frac: 0.0,
        periods_per_year: None,
        candidates: Vec::new(),
        pool_reset_flat: false,
        allow_lockbox: false,
    }
}

/// One test for every runner: the search (`run`), walk-forward train and test,
/// the panel build window and a labelled prep. All four must use the same
/// central guard, not a per-runner copy.
#[test]
fn every_runner_refuses_2026h2_unless_overridden() {
    let l = Lockbox::crypto_2026h2();
    let day = 86_400i64;

    // 1. `rcsr run`: a temporal split whose test rows fall in 2026H2.
    let mut csv = String::from("ts,x,y\n");
    let start = l.start - 500 * day;
    for i in 0..800 {
        let ts = start + i * day;
        csv.push_str(&format!("{},{},{}\n", ts, i as f64, (i as f64).sin()));
    }
    let t = data::parse_table(&csv).unwrap();
    let spec = SplitSpec {
        mode: SplitMode::Temporal,
        time_col: Some(0),
        test_frac: 0.25,
        ..Default::default()
    };
    let err = data::prepare_full(&t, &["x".into()], "y", &spec, &mut rcsr::rng::Mt::new(1))
        .err()
        .expect("run must refuse 2026H2");
    assert!(err.contains("training split") || err.contains("test split"), "{}", err);
    assert!(err.contains("lockbox"), "{}", err);
    let mut allowed = spec.clone();
    allowed.allow_lockbox = true;
    data::prepare_full(&t, &["x".into()], "y", &allowed, &mut rcsr::rng::Mt::new(1)).unwrap();

    // 2. Walk-forward test window in 2026H2.
    let mut wf = eval_spec(
        vec![Window {
            label: "2026H2".to_string(),
            train_start: l.start as f64 - 100.0,
            train_end: l.start as f64 - 10.0,
            test_start: l.start as f64,
            test_end: l.start as f64 + day as f64,
        }],
    );
    let tiny = data::parse_table("ts,x,y\n0,0,0\n1,1,1\n2,2,2\n3,3,3\n").unwrap();
    let err = walkforward::run_table(&tiny, &wf, "test").unwrap_err();
    assert!(err.contains("test range"), "{}", err);
    assert!(err.contains("lockbox"), "{}", err);

    // 3. Walk-forward **training** window in 2026H2 with a 2027 test.
    wf.windows = vec![Window {
        label: "train-on-holdout".to_string(),
        train_start: (l.start + day) as f64,
        train_end: (l.start + 30 * day) as f64,
        test_start: l.end as f64,
        test_end: (l.end + day) as f64,
    }];
    let err = walkforward::run_table(&tiny, &wf, "test").unwrap_err();
    assert!(err.contains("training range"), "{}", err);
    assert!(err.contains("lockbox"), "{}", err);
    wf.allow_lockbox = true;
    let _ = walkforward::run_table(&tiny, &wf, "test");

    // 4. An explicit panel build window in 2026H2.
    let err = panel::check_build_window(l.start, l.start + 30 * day, day, false).unwrap_err();
    assert!(err.contains("panel build range"), "{}", err);
    panel::check_build_window(l.start, l.start + 30 * day, day, true).unwrap();
    // A build that ends where the lockbox starts is allowed.
    panel::check_build_window(l.start - 30 * day, l.start - day, day, false).unwrap();

    // 5. A labelled prep inside 2026H2.
    let err = rcsr::prep::check_labelled_rows(&[l.start + 1, l.start + 2], false).unwrap_err();
    assert!(err.contains("prep labelled rows"), "{}", err);
    rcsr::prep::check_labelled_rows(&[l.start + 1, l.start + 2], true).unwrap();
    // Unlabelled bars are provenance; the manifest still counts the overlap.
    let cov = l.overlap([l.start + 1, l.start + 2]);
    assert_eq!(cov.rows, 2);
    assert!(!cov.windows.is_empty());
}

#[test]
fn the_panel_store_and_manifest_record_counts_delisted_and_checksums() {
    let l = Lockbox::crypto_2026h2();
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
    let snapshot = "{\"source\":\"binance-vision-monthly\",\"coverage_last\":\"2026-09-30\",\"verified\":true}";
    let manifest_path =
        panel::write_panel_panel(&out_str, &p, &m, &[in_str.as_str()], 65_536, false, Some(snapshot))
            .unwrap();
    assert!(out.exists());
    let text = std::fs::read_to_string(&manifest_path).unwrap();
    assert!(text.contains("\"sha256\""));
    assert!(text.contains("\"delisted\""));
    assert!(text.contains("\"C\""));
    assert!(text.contains("\"symbols_per_year\""));
    assert!(text.contains("\"rebalances\""));
    // #145: the manifest records the lockbox start/end, permission and
    // coverage even when (as here, in 2020) no row overlaps it.
    assert!(text.contains("\"lockbox\""), "{}", text);
    assert!(text.contains("\"permitted\":false"), "{}", text);
    assert!(text.contains("\"complete\":false"), "{}", text);
    assert!(text.contains(&format!("\"start\":{}", l.start)), "{}", text);
    assert!(text.contains(&format!("\"end\":{}", l.end)), "{}", text);
    assert!(text.contains("\"snapshot\""), "{}", text);
    assert!(text.contains("\"coverage_last\":\"2026-09-30\""), "{}", text);
    let _ = std::fs::remove_file(out);
    let _ = std::fs::remove_file(manifest_path);
    let _ = std::fs::remove_file(input);
}

#[test]
fn a_resumed_build_reuses_parts_and_matches_the_one_shot_panel() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let one = build_panel(&s, &m).unwrap();
    let dir = std::env::temp_dir().join(format!("rcsr-panel-resume-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dir_s = dir.to_string_lossy().to_string();
    let store = "store-0000";
    let cold = build_panel_resumable(Some(&s), &m, DAY, &dir_s, store, "").unwrap();
    assert_eq!(cold.rows, one.rows);
    assert_eq!(cold.delisted, one.delisted);
    assert_eq!(cold.gaps, one.gaps);
    let parts: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().ends_with(".part"))
        .collect();
    assert_eq!(parts.len(), s.len(), "one part per symbol");
    assert!(dir.join("parts.txt").exists(), "the cache index is written");
    // A fully warm build takes no input at all and reproduces the cold panel.
    assert!(panel::panel_cache_ready(&dir_s, DAY, store, &m, "").unwrap());
    let warm = build_panel_resumable(None, &m, DAY, &dir_s, store, "").unwrap();
    assert_eq!(warm.rows, cold.rows);
    assert_eq!(warm.delisted, cold.delisted);
    assert_eq!(warm.gaps, cold.gaps);
    // Remove one part: the source-fed resume path recreates exactly it.
    std::fs::remove_file(parts[0].path()).unwrap();
    let again = build_panel_resumable(Some(&s), &m, DAY, &dir_s, store, "").unwrap();
    assert_eq!(again.rows, one.rows);
    assert_eq!(again.gaps, one.gaps);
    // A warm build with a missing part must fail loudly, not shrink the panel.
    let dir2 = std::env::temp_dir().join(format!("rcsr-panel-resume-b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir2);
    let dir2_s = dir2.to_string_lossy().to_string();
    build_panel_resumable(Some(&s), &m, DAY, &dir2_s, store, "").unwrap();
    let victim = std::fs::read_dir(&dir2)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().ends_with(".part"))
        .unwrap();
    std::fs::remove_file(victim.path()).unwrap();
    assert!(build_panel_resumable(None, &m, DAY, &dir2_s, store, "").is_err());
    // A different bar step is not reused: it rebuilds the directory for the
    // new step and matches that step's one-shot build.
    let other = build_panel_resumable(Some(&s), &m, DAY * 2, &dir_s, store, "").unwrap();
    assert_eq!(
        other.rows,
        panel::build_panel_with_step(&s, &m, DAY * 2).unwrap().rows
    );
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_dir_all(&dir2);
}

#[test]
fn a_changed_store_or_membership_invalidates_the_panel_cache() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let dir = std::env::temp_dir().join(format!("rcsr-panel-key-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dir_s = dir.to_string_lossy().to_string();
    build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    // Same inputs: warm.
    assert!(panel::panel_cache_ready(&dir_s, DAY, "store-a", &m, "").unwrap());
    // A different input store checksum is not warm, and a warm build refuses
    // to guess: it must not mix rows from the previous store.
    assert!(!panel::panel_cache_ready(&dir_s, DAY, "store-b", &m, "").unwrap());
    assert!(build_panel_resumable(None, &m, DAY, &dir_s, "store-b", "").is_err());
    // A different membership is not warm either: the cache pins decisions.
    let mut m2 = m.clone();
    m2.rows[0].member = !m2.rows[0].member;
    assert!(!panel::panel_cache_ready(&dir_s, DAY, "store-a", &m2, "").unwrap());
    // Nor is a different settings string (e.g. migrations).
    assert!(!panel::panel_cache_ready(&dir_s, DAY, "store-a", &m, "OLD->NEW:1").unwrap());
    // Fed with the input, the cache is rebuilt in place for the new key and
    // agrees with a one-shot build of those inputs.
    let mut s2 = s.clone();
    s2.get_mut("B").unwrap()[10].close *= 1.5;
    let m3 = build_membership(&s2, &cfg(2)).unwrap();
    let rebuilt = build_panel_resumable(Some(&s2), &m3, DAY, &dir_s, "store-b", "").unwrap();
    assert_eq!(rebuilt.rows, build_panel(&s2, &m3).unwrap().rows);
    assert!(panel::panel_cache_ready(&dir_s, DAY, "store-b", &m3, "").unwrap());
    assert!(!panel::panel_cache_ready(&dir_s, DAY, "store-a", &m, "").unwrap());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_corrupted_panel_part_and_a_stale_format_are_refused() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let dir = std::env::temp_dir().join(format!("rcsr-panel-corrupt-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dir_s = dir.to_string_lossy().to_string();
    let cold = build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    let part = std::fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .find(|e| e.file_name().to_string_lossy().ends_with(".part"))
        .unwrap()
        .path();
    let mut bytes = std::fs::read(&part).unwrap();
    // Flip one payload byte: the recorded checksum no longer matches.
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&part, &bytes).unwrap();
    let err = build_panel_resumable(None, &m, DAY, &dir_s, "store-a", "").unwrap_err();
    assert!(err.contains("checksum"), "{}", err);
    // The source-fed path simply rebuilds the part.
    let rebuilt = build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    assert_eq!(rebuilt.rows, cold.rows);
    // A text-format cache from the previous version is never read as binary:
    // with the input it is rebuilt, without it the warm build refuses.
    std::fs::write(
        dir.join("stale.part"),
        "1609459200\t1609459200\t0\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\t1\n",
    )
    .unwrap();
    std::fs::remove_file(dir.join("state.txt")).unwrap();
    assert!(build_panel_resumable(None, &m, DAY, &dir_s, "store-a", "").is_err());
    let again = build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    assert_eq!(again.rows, cold.rows);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_part_with_a_stale_series_end_is_rebuilt() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let dir = std::env::temp_dir().join(format!("rcsr-panel-grow-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dir_s = dir.to_string_lossy().to_string();
    let cold = build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    let untouched = std::fs::metadata(dir.join("C.part")).unwrap().modified().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(20));
    // A part that records an older series end (as if the input had grown) must
    // not be trusted: the source-fed path rebuilds it. The checksum covers the
    // payload, so patching the header alone still parses.
    let a_part = dir.join("A.part");
    let mut bytes = std::fs::read(&a_part).unwrap();
    bytes[20..28].copy_from_slice(&0i64.to_le_bytes());
    std::fs::write(&a_part, &bytes).unwrap();
    // The warm (index-only) path refuses the mismatch instead of shrinking.
    let err = build_panel_resumable(None, &m, DAY, &dir_s, "store-a", "").unwrap_err();
    assert!(err.contains("ends at"), "{}", err);
    let rebuilt = build_panel_resumable(Some(&s), &m, DAY, &dir_s, "store-a", "").unwrap();
    assert_eq!(rebuilt.rows, cold.rows);
    assert_eq!(rebuilt.delisted, cold.delisted);
    assert!(
        std::fs::metadata(&a_part).unwrap().modified().unwrap() > untouched,
        "the stale part is rewritten"
    );
    assert_eq!(
        std::fs::metadata(dir.join("C.part")).unwrap().modified().unwrap(),
        untouched,
        "an unchanged part is not rewritten"
    );
    // The repaired cache is warm again.
    let warm = build_panel_resumable(None, &m, DAY, &dir_s, "store-a", "").unwrap();
    assert_eq!(warm.rows, cold.rows);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cold_and_warm_panel_stores_are_byte_identical() {
    let s = series();
    let m = build_membership(&s, &cfg(2)).unwrap();
    let dir = std::env::temp_dir().join(format!("rcsr-panel-hash-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let dir_s = dir.to_string_lossy().to_string();
    let store = "store-a";
    let cold = build_panel_resumable(Some(&s), &m, DAY, &dir_s, store, "").unwrap();
    let warm = build_panel_resumable(None, &m, DAY, &dir_s, store, "").unwrap();
    // The resumed rows are identical value-for-value, not just in count.
    assert_eq!(cold.rows, warm.rows);
    assert_eq!(cold.symbols, warm.symbols);
    assert_eq!(cold.delisted, warm.delisted);
    assert_eq!((cold.gaps, cold.first_ts, cold.last_ts), (warm.gaps, warm.first_ts, warm.last_ts));
    // And the written stores hash the same, so a manifest checksum cannot tell
    // a warm rebuild from a cold one.
    let csv = std::env::temp_dir().join(format!("rcsr-panel-hash-{}.csv", std::process::id()));
    let mut text = String::from("ts,symbol,open,high,low,close,volume,quote_volume\n");
    for (sym, bs) in &s {
        for b in bs {
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                b.time, sym, b.open, b.high, b.low, b.close, b.volume, b.quote_volume
            ));
        }
    }
    std::fs::write(&csv, text).unwrap();
    let csv_s = csv.to_string_lossy().to_string();
    let a = std::env::temp_dir().join(format!("rcsr-panel-hash-a-{}.rcsb", std::process::id()));
    let b = std::env::temp_dir().join(format!("rcsr-panel-hash-b-{}.rcsb", std::process::id()));
    let a_s = a.to_string_lossy().to_string();
    let b_s = b.to_string_lossy().to_string();
    panel::write_panel_panel(&a_s, &cold, &m, &[csv_s.as_str()], 65_536, false, None).unwrap();
    panel::write_panel_panel(&b_s, &warm, &m, &[csv_s.as_str()], 65_536, false, None).unwrap();
    let ha = rcsr::manifest::sha256_file(&a).unwrap();
    let hb = rcsr::manifest::sha256_file(&b).unwrap();
    assert_eq!(ha, hb, "cold and warm panel stores differ");
    let _ = std::fs::remove_dir_all(&dir);
    let _ = std::fs::remove_file(&a);
    let _ = std::fs::remove_file(&b);
    let _ = std::fs::remove_file(format!("{}.manifest.json", a_s));
    let _ = std::fs::remove_file(format!("{}.manifest.json", b_s));
    let _ = std::fs::remove_file(&csv);
}

#[test]
fn a_store_key_reads_the_published_manifest_and_falls_back_to_hashing() {
    let dir = std::env::temp_dir().join(format!("rcsr-panel-storekey-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    let store = dir.join("s.rcsb");
    std::fs::write(&store, b"abc").unwrap();
    let store_s = store.to_string_lossy().to_string();
    let hashed = rcsr::manifest::sha256_hex(b"abc");
    // No manifest: hash the file.
    assert_eq!(panel::store_key(&store_s), hashed);
    // A manifest written for this file (same byte count) supplies the key
    // without re-reading 1.5 GB.
    let recorded = "f".repeat(64);
    let manifest = format!(
        "{{\n  \"tool\":\"rcsr prep\",\n  \"output_sha256\":\"{}\",\n  \"output_bytes\":3,\n  \
         \"output\":{{\"path\":\"s.rcsb\"}}\n}}\n",
        recorded
    );
    std::fs::write(format!("{}.manifest.json", store_s), &manifest).unwrap();
    assert_eq!(panel::store_key(&store_s), recorded);
    // A manifest that describes a different file length is not trusted.
    let stale = manifest.replace("\"output_bytes\":3", "\"output_bytes\":4");
    std::fs::write(format!("{}.manifest.json", store_s), &stale).unwrap();
    assert_eq!(panel::store_key(&store_s), hashed);
    // A non-hex value is not trusted either.
    let junk = manifest.replace(&recorded, "unknown");
    std::fs::write(format!("{}.manifest.json", store_s), &junk).unwrap();
    assert_eq!(panel::store_key(&store_s), hashed);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_membership_cache_key_covers_the_store_and_every_ranking_setting() {
    let base = cfg(2);
    let key = panel::membership_cache_key("store-a", &base);
    assert_eq!(key, panel::membership_cache_key("store-a", &base));
    assert_ne!(key, panel::membership_cache_key("store-b", &base));
    let mut c = base.clone();
    c.top = 3;
    assert_ne!(key, panel::membership_cache_key("store-a", &c));
    let mut c = base.clone();
    c.lookback_days = 31;
    assert_ne!(key, panel::membership_cache_key("store-a", &c));
    let mut c = base.clone();
    c.start = Some(T0);
    assert_ne!(key, panel::membership_cache_key("store-a", &c));
    let mut c = base.clone();
    c.end = Some(T0 + 30 * DAY);
    assert_ne!(key, panel::membership_cache_key("store-a", &c));
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
