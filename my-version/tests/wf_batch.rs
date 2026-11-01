//! #309: batch walk-forward P&L over `(candidate, window)`.
//!
//! Every cell is compared with the scalar walk-forward evaluator. With
//! `--features cuda` the batch runs on the device and the tests assert that
//! it did. Run the device tests under `compute-sanitizer --tool memcheck`
//! first.
//!
//! The measurement is ignored by default:
//!
//! ```sh
//! cargo test --release --features cuda --test wf_batch -- --ignored --nocapture bench_batch_vs_scalar
//! ```

use rcsr::baselines;
use rcsr::trading_gpu::{self, PnlParams, PnlTotals};
use rcsr::walkforward::{self, EvalSpec, ProtocolOptions, Window};
use rcsr::wf_batch::{
    self, Backend, BatchResult, BatchSpec, BatchWindow, Seam, CPU_BATCH, EXACT_CPU, GPU_PROXY,
};
use std::collections::HashMap;

const COST_SIDE: f64 = 0.001;

fn rel_close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * 1.0f64.max(a.abs()).max(b.abs())
}

fn assert_totals_close(a: &PnlTotals, b: &PnlTotals, what: &str) {
    assert_eq!(a.rows, b.rows, "{what}: rows");
    assert_eq!(a.long_rows, b.long_rows, "{what}: long_rows");
    assert_eq!(a.entries, b.entries, "{what}: entries");
    assert_eq!(a.exits, b.exits, "{what}: exits");
    assert_eq!(a.trades, b.trades, "{what}: trades");
    assert_eq!(a.hits, b.hits, "{what}: hits");
    assert_eq!(a.final_exits, b.final_exits, "{what}: final_exits");
    for (x, y, f) in [
        (a.gross, b.gross, "gross"),
        (a.cost, b.cost, "cost"),
        (a.turnover, b.turnover, "turnover"),
        (a.net, b.net, "net"),
    ] {
        assert!(rel_close(x, y, 1e-9), "{what}: {f} {x} vs {y}");
    }
}

fn lcg(s: &mut u64) -> f64 {
    *s = s
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*s >> 11) as f64) / ((1u64 << 53) as f64)
}

/// Windows with gaps in group membership, rows in time-major order.
fn fixture_windows(windows: usize, groups: usize, base_rows: usize) -> Vec<BatchWindow> {
    let mut seed = 20_260_309u64;
    let mut out = Vec::new();
    for w in 0..windows {
        let mut time = Vec::new();
        let mut group = Vec::new();
        let mut target = Vec::new();
        let max_rows = base_rows + 12;
        for i in 0..max_rows {
            for g in 0..groups {
                if (g + w) % 5 == 0 {
                    continue;
                }
                if i >= base_rows + (g * 7 + w * 3) % 13 {
                    continue;
                }
                time.push((w * 10_000 + i) as f64);
                group.push((g as u64) * 3 + 11);
                target.push((lcg(&mut seed) - 0.48) * 0.02);
            }
        }
        out.push(BatchWindow {
            label: format!("w{w}"),
            time,
            group,
            target,
            train_mean: if w % 2 == 0 { 0.0004 } else { -0.0003 },
        });
    }
    out
}

fn score(c: usize, w: usize, i: usize) -> f64 {
    ((i as f64) * 0.013 * (c as f64 + 1.0) + w as f64 * 0.7).sin()
        + 0.35 * (((i * 7919 + c * 31 + w * 17) % 101) as f64 / 50.0 - 1.0)
}

fn threshold(c: usize, w: usize, candidates: usize) -> f64 {
    0.25 * (c as f64 / candidates as f64) - 0.1 + 0.01 * w as f64
}

fn thresholds(candidates: usize, windows: usize) -> Vec<Vec<f64>> {
    (0..candidates)
        .map(|c| (0..windows).map(|w| threshold(c, w, candidates)).collect())
        .collect()
}

fn run_batch(
    windows: &[BatchWindow],
    candidates: usize,
    seam: Seam,
    backend: Backend,
    configure: impl FnOnce(&mut BatchSpec),
) -> BatchResult {
    let thr = thresholds(candidates, windows.len());
    let mut spec = BatchSpec::new(windows, candidates, &thr, COST_SIDE, seam);
    spec.backend = backend;
    configure(&mut spec);
    let mut fetch = |c: usize, w: usize| -> Vec<f64> {
        (0..windows[w].time.len()).map(|i| score(c, w, i)).collect()
    };
    wf_batch::run(&spec, &mut fetch).expect("batch run")
}

fn scalar_cells(
    windows: &[BatchWindow],
    candidates: usize,
    seam: Seam,
) -> Vec<PnlTotals> {
    let thr = thresholds(candidates, windows.len());
    let mut out = Vec::new();
    for c in 0..candidates {
        let mut previous: HashMap<u64, f64> = HashMap::new();
        for (w, win) in windows.iter().enumerate() {
            if seam == Seam::PerWindow {
                previous.clear();
            }
            let s: Vec<f64> = (0..win.time.len()).map(|i| score(c, w, i)).collect();
            out.push(walkforward::reference_cell(
                &win.time,
                &win.group,
                &win.target,
                &s,
                thr[c][w],
                COST_SIDE,
                &mut previous,
            ));
        }
    }
    out
}

fn expected_backend(r: &BatchResult) {
    #[cfg(feature = "cuda")]
    {
        assert!(wf_batch::device_available(), "the device tests need a CUDA device");
        assert_eq!(r.backend, "cuda", "the device kernel must have run");
        assert!(r.stats.device_tiles > 0);
        assert_eq!(r.stats.host_tiles, 0, "no tile may fall back silently");
        assert!(r.stats.device_bytes > 0);
        assert!(r.cells.iter().all(|c| c.provenance == GPU_PROXY));
    }
    #[cfg(not(feature = "cuda"))]
    {
        assert_eq!(r.backend, "cpu");
        assert!(r.cells.iter().all(|c| c.provenance == CPU_BATCH));
    }
}

fn check_against_scalar(seam: Seam, backend: Backend) -> BatchResult {
    let windows = fixture_windows(4, 12, 90);
    let candidates = 9;
    let r = run_batch(&windows, candidates, seam, backend, |_| {});
    let want = scalar_cells(&windows, candidates, seam);
    assert_eq!(r.cells.len(), want.len());
    for (k, cell) in r.cells.iter().enumerate() {
        assert_eq!((cell.candidate, cell.window), (k / windows.len(), k % windows.len()));
        assert_totals_close(
            &cell.totals,
            &want[k],
            &format!("{} c{} w{}", seam.as_str(), cell.candidate, cell.window),
        );
        assert!(cell.totals.trades > 0, "the fixture trades");
    }
    r
}

#[test]
fn batch_equals_scalar_walkforward_per_window() {
    let r = check_against_scalar(Seam::PerWindow, Backend::Auto);
    expected_backend(&r);
}

#[test]
fn batch_equals_scalar_walkforward_pooled() {
    let r = check_against_scalar(Seam::Pooled, Backend::Auto);
    expected_backend(&r);
}

#[test]
fn host_mirror_equals_scalar_walkforward_in_both_seams() {
    for seam in [Seam::PerWindow, Seam::Pooled] {
        let r = check_against_scalar(seam, Backend::Cpu);
        assert_eq!(r.backend, "cpu");
        assert!(r.cells.iter().all(|c| c.provenance == CPU_BATCH));
        assert!(r.fallback_reason.is_some());
    }
}

#[test]
fn pooled_carries_the_seam_and_per_window_resets_it() {
    let windows = fixture_windows(4, 12, 90);
    let per = run_batch(&windows, 9, Seam::PerWindow, Backend::Auto, |_| {});
    let pooled = run_batch(&windows, 9, Seam::Pooled, Backend::Auto, |_| {});
    let mut differs = 0;
    for c in 0..9 {
        // The first window starts flat in both seams.
        assert_totals_close(&per.cell(c, 0).totals, &pooled.cell(c, 0).totals, "first window");
        for w in 1..4 {
            let (a, b) = (&per.cell(c, w).totals, &pooled.cell(c, w).totals);
            assert_eq!(a.rows, b.rows);
            // A carried position pays no entry at the seam, so the pooled
            // block never has more entries than the flat-start block.
            assert!(b.entries <= a.entries, "c{c} w{w}");
            if a.entries != b.entries {
                differs += 1;
            }
        }
    }
    assert!(differs > 0, "the fixture must exercise a carried position");
}

#[test]
fn input_row_order_does_not_change_any_cell() {
    let windows = fixture_windows(3, 8, 60);
    let a = run_batch(&windows, 5, Seam::Pooled, Backend::Cpu, |_| {});
    // Reverse every window's rows and replay the same score per row.
    let reversed: Vec<BatchWindow> = windows
        .iter()
        .map(|w| BatchWindow {
            label: w.label.clone(),
            time: w.time.iter().rev().copied().collect(),
            group: w.group.iter().rev().copied().collect(),
            target: w.target.iter().rev().copied().collect(),
            train_mean: w.train_mean,
        })
        .collect();
    let thr = thresholds(5, 3);
    let spec = {
        let mut s = BatchSpec::new(&reversed, 5, &thr, COST_SIDE, Seam::Pooled);
        s.backend = Backend::Cpu;
        s
    };
    let mut fetch = |c: usize, w: usize| -> Vec<f64> {
        let n = windows[w].time.len();
        (0..n).map(|i| score(c, w, n - 1 - i)).collect()
    };
    let b = wf_batch::run(&spec, &mut fetch).unwrap();
    for (x, y) in a.cells.iter().zip(&b.cells) {
        assert_totals_close(&x.totals, &y.totals, "reversed input");
    }
}

/// Two symbols, 317 timestamps. Symbol A holds a long position in alternating
/// runs, symbol B holds the opposite position except on the last bar.
fn flatten_fixture() -> (Vec<bool>, Vec<bool>) {
    let t = 317usize;
    let flips = 176usize;
    let mut a = vec![false; t];
    let mut state = false;
    for i in 0..t {
        if ((i + 1) * flips) / t != (i * flips) / t {
            state = !state;
        }
        a[i] = state;
    }
    let mut b: Vec<bool> = a.iter().map(|v| !v).collect();
    b[t - 1] = !b[t - 1];
    (a, b)
}

#[test]
fn flattened_row_order_invents_no_trades_352_not_456() {
    let (a, b) = flatten_fixture();
    let t = a.len();
    // Time-major input, the order a flattened (timestamp, symbol) scan yields.
    let mut time = Vec::new();
    let mut group = Vec::new();
    let mut target = Vec::new();
    let mut scores = Vec::new();
    for i in 0..t {
        for (g, side) in [(0u64, &a), (1u64, &b)] {
            time.push(i as f64);
            group.push(g);
            target.push(0.001 * ((i + g as usize) % 4) as f64);
            scores.push(if side[i] { 1.0 } else { -1.0 });
        }
    }
    let window = BatchWindow {
        label: "w".to_string(),
        time: time.clone(),
        group: group.clone(),
        target: target.clone(),
        train_mean: 0.0,
    };
    let windows = vec![window];
    let thr = vec![vec![0.0]];
    for seam in [Seam::PerWindow, Seam::Pooled] {
        let mut spec = BatchSpec::new(&windows, 1, &thr, 0.0005, seam);
        spec.backend = Backend::Auto;
        let mut fetch = |_c: usize, _w: usize| scores.clone();
        let r = wf_batch::run(&spec, &mut fetch).unwrap();
        assert_eq!(r.cell(0, 0).totals.trades, 352, "grouped trades ({})", seam.as_str());
        assert_eq!(r.cell(0, 0).totals.entries + r.cell(0, 0).totals.exits
            - r.cell(0, 0).totals.final_exits, 352);
    }

    // The same rows read as one series in input order invent trades.
    let params = PnlParams {
        threshold: 0.0,
        per_side_bps: 5.0,
        ..PnlParams::default()
    };
    let one_group = vec![0u64; time.len()];
    let flat = trading_gpu::reduce(&scores, &target, &one_group, &params).unwrap();
    assert_eq!(flat.strategy.trades, 456, "flattened order invents trades");
    // The scalar evaluator agrees with the batch on the grouped book.
    let mut previous = HashMap::new();
    let reference = walkforward::reference_cell(&time, &group, &target, &scores, 0.0, 0.0005, &mut previous);
    assert_eq!(reference.trades, 352);
}

#[test]
fn non_finite_score_cells_use_the_scalar_evaluator() {
    let windows = fixture_windows(3, 8, 60);
    let candidates = 4usize;
    let thr = thresholds(candidates, windows.len());
    let spec = {
        let mut s = BatchSpec::new(&windows, candidates, &thr, COST_SIDE, Seam::Pooled);
        s.random_seed = Some(7);
        s
    };
    let poisoned = |c: usize, w: usize, i: usize| c == 2 && w == 1 && i % 17 == 3;
    let mut fetch = |c: usize, w: usize| -> Vec<f64> {
        (0..windows[w].time.len())
            .map(|i| if poisoned(c, w, i) { f64::NAN } else { score(c, w, i) })
            .collect()
    };
    let r = wf_batch::run(&spec, &mut fetch).unwrap();
    assert_eq!(r.stats.fallback_candidates, 1);
    let mut previous = HashMap::new();
    for (w, win) in windows.iter().enumerate() {
        let s: Vec<f64> = (0..win.time.len())
            .map(|i| if poisoned(2, w, i) { f64::NAN } else { score(2, w, i) })
            .collect();
        let want = walkforward::reference_cell(
            &win.time, &win.group, &win.target, &s, thr[2][w], COST_SIDE, &mut previous,
        );
        let got = r.cell(2, w);
        assert_eq!(got.provenance, EXACT_CPU);
        assert_totals_close(&got.totals, &want, &format!("fallback w{w}"));
        assert!(r.ladder[w].random[2].is_none());
        assert!(r.ladder[w].random[0].is_some());
    }
    for c in [0usize, 1, 3] {
        for w in 0..windows.len() {
            assert_ne!(r.cell(c, w).provenance, EXACT_CPU);
        }
    }
}

#[test]
fn dropped_target_rows_are_priced_like_the_evaluator() {
    let mut windows = fixture_windows(2, 6, 50);
    for w in windows.iter_mut() {
        for i in (0..w.target.len()).step_by(11) {
            w.target[i] = f64::NAN;
        }
    }
    let r = run_batch(&windows, 3, Seam::PerWindow, Backend::Auto, |_| {});
    expected_backend(&r);
    let thr = thresholds(3, 2);
    for c in 0..3 {
        for (w, win) in windows.iter().enumerate() {
            let s: Vec<f64> = (0..win.time.len()).map(|i| score(c, w, i)).collect();
            let want = walkforward::reference_cell(
                &win.time,
                &win.group,
                &win.target,
                &s,
                thr[c][w],
                COST_SIDE,
                &mut HashMap::new(),
            );
            assert_totals_close(&r.cell(c, w).totals, &want, &format!("c{c} w{w}"));
            let kept = win.target.iter().filter(|t| t.is_finite()).count();
            assert_eq!(r.cell(c, w).totals.rows, kept as u64);
        }
    }
}

#[test]
fn ladder_always_has_cash_and_buy_and_hold_and_the_random_control_is_matched() {
    let windows = fixture_windows(3, 8, 60);
    let candidates = 4;
    let r = run_batch(&windows, candidates, Seam::PerWindow, Backend::Auto, |s| {
        s.random_seed = Some(11);
    });
    let thr = thresholds(candidates, windows.len());
    assert_eq!(r.ladder.len(), windows.len());
    for (w, l) in r.ladder.iter().enumerate() {
        let win = &windows[w];
        let mut idx: Vec<usize> = (0..win.time.len()).collect();
        idx.sort_by(|&a, &b| {
            win.group[a]
                .cmp(&win.group[b])
                .then(win.time[a].total_cmp(&win.time[b]))
                .then(a.cmp(&b))
        });
        let rt: Vec<f64> = idx.iter().map(|&i| win.target[i]).collect();
        let g: Vec<u64> = idx.iter().map(|&i| win.group[i]).collect();
        assert_eq!(l.cash.net, 0.0);
        assert_eq!(l.cash.cost, 0.0);
        let groups = g.iter().collect::<std::collections::BTreeSet<_>>().len() as f64;
        let want_bh = rt.iter().sum::<f64>() - 2.0 * COST_SIDE * groups;
        assert!(rel_close(l.buy_and_hold.net, want_bh, 1e-12), "w{w} bh");
        assert_eq!(l.constant_long, baselines::constant_is_long(win.train_mean));
        let want_const = if l.constant_long { l.buy_and_hold } else { l.cash };
        assert_eq!(l.constant, want_const);
        for c in 0..candidates {
            let pos: Vec<f64> = idx
                .iter()
                .map(|&i| trading_gpu::long_position(score(c, w, i), thr[c][w]) as f64)
                .collect();
            let matched = baselines::matched_exposure_positions(&pos, &g, 11).unwrap();
            let params = PnlParams {
                per_side_bps: COST_SIDE * 10_000.0,
                ..PnlParams::default()
            };
            let want = baselines::position_totals(&matched, &rt, &g, &params).unwrap();
            let got = l.random[c].as_ref().expect("random control");
            assert_eq!(got.long_rows, r.cell(c, w).totals.long_rows, "exposure is matched");
            assert_eq!(*got, want);
        }
    }
    let json = r.json(&windows.iter().map(|w| w.label.clone()).collect::<Vec<_>>());
    assert!(json.contains("\"cash\"") && json.contains("\"buy_and_hold\""));
    assert!(json.contains(GPU_PROXY) || json.contains(CPU_BATCH));
}

#[test]
fn the_score_tile_is_declared_and_never_scales_with_the_population() {
    let windows = fixture_windows(3, 8, 60);
    let candidates = 200usize;
    let rows: usize = windows.iter().map(|w| w.time.len()).sum();
    let big = run_batch(&windows, candidates, Seam::Pooled, Backend::Auto, |_| {});
    let small = run_batch(&windows, candidates, Seam::Pooled, Backend::Auto, |s| {
        s.tile_bytes = 3 * rows * 8;
    });
    assert!(big.stats.tile_candidates <= wf_batch::MAX_BATCH_TILE);
    assert_eq!(small.stats.tile_candidates, 3);
    let full = (candidates * rows * 8) as u64;
    assert!(big.stats.host_tile_bytes < full / 2, "tile {} vs P x N {}", big.stats.host_tile_bytes, full);
    assert_eq!(small.stats.host_tile_bytes, (3 * rows * 8) as u64);
    for (x, y) in big.cells.iter().zip(&small.cells) {
        assert_eq!(x.totals.bits(), y.totals.bits(), "tile size must not change a bit");
        assert_eq!(x.totals.trades, y.totals.trades);
    }
}

#[test]
fn block_size_does_not_change_a_bit() {
    let windows = fixture_windows(3, 10, 70);
    let base = run_batch(&windows, 6, Seam::Pooled, Backend::Auto, |_| {});
    expected_backend(&base);
    for threads in [32usize, 64, 256, 512] {
        let r = run_batch(&windows, 6, Seam::Pooled, Backend::Auto, |s| s.threads = threads);
        for (x, y) in base.cells.iter().zip(&r.cells) {
            assert_eq!(x.totals.bits(), y.totals.bits(), "threads {threads}");
        }
    }
}

#[test]
fn invalid_input_is_refused() {
    let windows = fixture_windows(2, 4, 20);
    let thr = thresholds(2, 2);
    let mut fetch = |_c: usize, w: usize| vec![0.0; windows[w].time.len()];
    let bad = BatchSpec::new(&windows, 2, &thr[..1], COST_SIDE, Seam::Pooled);
    assert!(wf_batch::run(&bad, &mut fetch).is_err());
    let bad = BatchSpec::new(&windows, 2, &thr, f64::NAN, Seam::Pooled);
    assert!(wf_batch::run(&bad, &mut fetch).is_err());
    let mut short = |_c: usize, _w: usize| vec![0.0; 3];
    let ok = BatchSpec::new(&windows, 2, &thr, COST_SIDE, Seam::Pooled);
    assert!(wf_batch::run(&ok, &mut short).is_err());
}

// ---------------------------------------------------------------------------
// Report integration
// ---------------------------------------------------------------------------

fn report_table() -> rcsr::data::Table {
    let mut s = String::from("ts,sym,x,target\n");
    let mut seed = 5u64;
    for t in 0..160 {
        for g in ["A", "B", "C"] {
            let x = lcg(&mut seed);
            let y = (x - 0.5) * 0.01 + (lcg(&mut seed) - 0.5) * 0.01;
            s.push_str(&format!("{},{},{},{}\n", t, g, x, y));
        }
    }
    rcsr::data::parse_table(&s).unwrap()
}

fn report_spec() -> EvalSpec {
    let w = |l: &str, a: f64, b: f64, c: f64, d: f64| Window {
        label: l.to_string(),
        train_start: a,
        train_end: b,
        test_start: c,
        test_end: d,
    };
    EvalSpec {
        expression: "x".to_string(),
        vars: vec!["x".to_string()],
        target: "target".to_string(),
        time: "ts".to_string(),
        group: Some("sym".to_string()),
        windows: vec![w("w1", 0.0, 60.0, 60.0, 100.0), w("w2", 0.0, 100.0, 100.0, 140.0)],
        horizon: 0.0,
        embargo: None,
        cost_bps: 10.0,
        threshold_quantile: 0.6,
        threshold_fixed: None,
        validation_frac: 0.3,
        periods_per_year: Some(365.0),
        candidates: vec!["x".to_string(), "-x".to_string(), "x*x".to_string()],
        band_select: false,
        pool_reset_flat: false,
        allow_lockbox: false,
        device_pnl: false,
        control_seed: 3,
        policy: walkforward::Policy::Threshold,
        band_width: 0.0,
        min_hold: 1,
        gpu: false,
    }
}

#[test]
fn report_carries_the_batch_block_and_the_finalists_stay_exact() {
    let table = report_table();
    let spec = report_spec();
    let plain = walkforward::run_table(&table, &spec, "test").unwrap();
    assert!(plain.batch_pnl.is_none());
    assert!(plain.json().contains("\"batch_pnl\":null"));

    let opts = ProtocolOptions {
        batch_pnl: true,
        ..ProtocolOptions::default()
    };
    let r = walkforward::run_table_with(&table, &spec, "test", &opts).unwrap();
    let b = r.batch_pnl.as_ref().expect("batch block");
    assert_eq!(b.finalists.len(), 2);
    assert!(b.max_abs_net_delta() <= 1e-9, "delta {}", b.max_abs_net_delta());
    assert!(b.pooled.is_some());
    assert_eq!(b.per_window.cells.len(), 3 * 2);
    // The reported metrics are the scalar evaluator's, unchanged by the batch.
    for (x, y) in plain.windows.iter().zip(&r.windows) {
        assert_eq!(x.metrics.total_return.to_bits(), y.metrics.total_return.to_bits());
        assert_eq!(x.metrics.trades, y.metrics.trades);
    }
    for f in &b.finalists {
        assert!(f.provenance == GPU_PROXY || f.provenance == CPU_BATCH);
    }
    let json = r.json();
    assert!(json.contains("\"reported_values\":\"exact_cpu_f64\""));
    assert!(json.contains("\"batch_values\":\"gpu_proxy\""));
    assert!(json.contains("\"matched_exposure_random_long\""));
    #[cfg(feature = "cuda")]
    assert_eq!(b.per_window.backend, "cuda");
}

// ---------------------------------------------------------------------------
// Allocation failure paths (run under compute-sanitizer first)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
#[test]
fn a_refused_device_allocation_falls_back_to_the_host_without_a_dangling_pointer() {
    let windows = fixture_windows(2, 6, 40);
    let want = run_batch(&windows, 3, Seam::Pooled, Backend::Cpu, |_| {});
    // The context makes seven guarded allocations; refuse each in turn.
    for k in 1..=7u64 {
        rcsr::enrich_gpu::set_alloc_fault_for_tests(k);
        let r = run_batch(&windows, 3, Seam::Pooled, Backend::Auto, |_| {});
        rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
        assert_eq!(r.backend, "cpu", "allocation {k} refused");
        assert_eq!(r.fallback_reason.as_deref(), Some("device context allocation refused"));
        for (x, y) in r.cells.iter().zip(&want.cells) {
            assert_eq!(x.totals.bits(), y.totals.bits());
        }
    }
    let ok = run_batch(&windows, 3, Seam::Pooled, Backend::Auto, |_| {});
    expected_backend(&ok);
}

#[cfg(feature = "cuda")]
#[test]
fn an_armed_allocation_fault_never_reaches_another_thread() {
    use std::sync::mpsc::channel;
    let windows = fixture_windows(2, 6, 40);
    let want = run_batch(&windows, 3, Seam::Pooled, Backend::Cpu, |_| {});
    for _ in 0..5 {
        let (armed_tx, armed_rx) = channel::<()>();
        let (done_tx, done_rx) = channel::<()>();
        std::thread::scope(|s| {
            let (windows, want) = (&windows, &want);
            let armed = s.spawn(move || {
                rcsr::enrich_gpu::set_alloc_fault_for_tests(1);
                armed_tx.send(()).unwrap();
                done_rx.recv().unwrap();
                let r = run_batch(windows, 3, Seam::Pooled, Backend::Auto, |_| {});
                rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
                assert_eq!(r.backend, "cpu", "the armed thread must get its refusal");
                for (x, y) in r.cells.iter().zip(&want.cells) {
                    assert_eq!(x.totals.bits(), y.totals.bits());
                }
            });
            let other = s.spawn(move || {
                armed_rx.recv().unwrap();
                let r = run_batch(windows, 3, Seam::Pooled, Backend::Auto, |_| {});
                done_tx.send(()).unwrap();
                expected_backend(&r);
            });
            armed.join().unwrap();
            other.join().unwrap();
        });
    }
}

// ---------------------------------------------------------------------------
// Measurement
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_batch_vs_scalar() {
    use std::time::Instant;
    let windows = fixture_windows(8, 40, 1_500);
    let candidates = 64usize;
    let rows: usize = windows.iter().map(|w| w.time.len()).sum();
    let thr = thresholds(candidates, windows.len());

    let t0 = Instant::now();
    let mut scalar_net = 0.0f64;
    for c in 0..candidates {
        for (w, win) in windows.iter().enumerate() {
            let s: Vec<f64> = (0..win.time.len()).map(|i| score(c, w, i)).collect();
            let mut previous = HashMap::new();
            let t = walkforward::reference_cell(
                &win.time, &win.group, &win.target, &s, thr[c][w], COST_SIDE, &mut previous,
            );
            scalar_net += t.net;
        }
    }
    let scalar_s = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let r = run_batch(&windows, candidates, Seam::PerWindow, Backend::Auto, |_| {});
    let batch_s = t1.elapsed().as_secs_f64();
    let batch_net: f64 = r.cells.iter().map(|c| c.totals.net).sum();
    assert!(rel_close(scalar_net, batch_net, 1e-9), "{scalar_net} vs {batch_net}");
    expected_backend(&r);

    let speedup = scalar_s / batch_s;
    println!(
        "bench candidates={candidates} windows={} rows={rows} cells={} backend={}",
        windows.len(),
        r.cells.len(),
        r.backend
    );
    println!("scalar_s={scalar_s:.4} batch_s={batch_s:.4} speedup={speedup:.1}x");
    println!(
        "candidates_per_s={:.1} row_evals_per_s={:.3e} device_fraction={:.3} device_bytes={} host_tile_bytes={} tile={}",
        r.stats.candidates_per_s,
        r.stats.row_evals_per_s,
        r.stats.device_fraction,
        r.stats.device_bytes,
        r.stats.host_tile_bytes,
        r.stats.tile_candidates
    );
    println!("{}", r.stats.json());
    assert!(speedup >= 5.0, "speedup {speedup:.2}x is below 5x");
}
