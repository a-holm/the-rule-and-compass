//! `rcsr enrich` (issue #200): append-only feature enrichment, gap/causality
//! rules, determinism across chunk sizes and the CPU/GPU cost model.

use rcsr::data;
use rcsr::enrich::{self, GpuMode, OpClass, Options};
use rcsr::ohlcv::{self, Bar, Timeframe};
use rcsr::store::{self, FileStore};

fn csv_of(rows: &[(i64, &str, f64, f64, f64, f64, f64)]) -> String {
    let mut s = String::from("ts,symbol,open,high,low,close,volume,keep\n");
    for (i, (t, sym, o, h, l, c, v)) in rows.iter().enumerate() {
        s.push_str(&format!(
            "{},{},{},{},{},{},{},{}\n",
            t,
            sym,
            o,
            h,
            l,
            c,
            v,
            1000 + i
        ));
    }
    s
}

fn bars(symbol: &str, n: usize, start: i64, step: i64) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|i| {
            let p = 100.0 + (i as f64) * 0.5 + (i as f64 * 0.1).sin();
            (
                start + i as i64 * step,
                symbol.to_string(),
                p,
                p + 0.3,
                p - 0.4,
                p + 0.05,
                10.0 + i as f64,
            )
        })
        .collect()
}

/// A two-sided price series: a monotonic ramp would leave RSI/MFI undefined
/// because every close-to-close move is a gain.
fn wavy_rows(
    symbol: &str,
    n: usize,
    start: i64,
    step: i64,
    phase: f64,
) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|i| {
            let t = i as f64;
            let p = 100.0 + 5.0 * (t * 0.37 + phase).sin() + 0.02 * t;
            let o = p + 0.4 * (t * 0.11 + phase).cos();
            let c = p + 0.5 * (t * 0.7).sin();
            let h = o.max(c) + 0.8 + (i % 5) as f64 * 0.03;
            let l = o.min(c) - 0.7 - (i % 3) as f64 * 0.02;
            let v = 10.0 + (i % 7) as f64 + 0.01 * t;
            (start + i as i64 * step, symbol.to_string(), o, h, l, c, v)
        })
        .collect()
}


fn write_store(path: &str, rows: &[(i64, String, f64, f64, f64, f64, f64)], group_rows: usize) {
    let refs: Vec<(i64, &str, f64, f64, f64, f64, f64)> =
        rows.iter().map(|(t, s, o, h, l, c, v)| (*t, s.as_str(), *o, *h, *l, *c, *v)).collect();
    let text = csv_of(&refs);
    let t = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &t, group_rows, Some(0)).unwrap();
}

#[allow(clippy::too_many_arguments)]
fn write_flow_store(
    path: &str,
    rows: &[(i64, String, f64, f64, f64, f64, f64, f64, f64, f64, f64)],
    with_vwap: bool,
    group_rows: usize,
) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote");
    if with_vwap {
        text.push_str(",vwap");
    }
    text.push('\n');
    for (t, sym, o, h, l, c, v, qv, n, tb, tbq) in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{}",
            t, sym, o, h, l, c, v, qv, n, tb, tbq
        ));
        if with_vwap {
            let vw = if *v > 0.0 { *qv / *v } else { f64::NAN };
            text.push_str(&format!(",{}", vw));
        }
        text.push('\n');
    }
    let t = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &t, group_rows, Some(0)).unwrap();
}

/// Write a store that carries the full raw OHLCV/flow block from seven-column
/// bar tuples (quote volume and taker flow are derived deterministically).
fn write_store_with_flow(
    path: &str,
    rows: &[(i64, String, f64, f64, f64, f64, f64)],
    group_rows: usize,
) {
    let rich: Vec<(i64, String, f64, f64, f64, f64, f64, f64, f64, f64, f64)> = rows
        .iter()
        .map(|(t, s, o, h, l, c, v)| {
            let qv = *v * (*h + *l + *c) / 3.0;
            (*t, s.clone(), *o, *h, *l, *c, *v, qv, 100.0 + *v, *v * 0.5, *v * 0.5 * *c)
        })
        .collect();
    write_flow_store(path, &rich, false, group_rows);
}

/// Write a symbol-major PIT panel with an explicit `member_asof` column.
fn write_panel_store(
    path: &str,
    rows: &[(i64, String, f64, f64, f64, f64, f64, f64)],
    group_rows: usize,
) {
    let mut text =
        String::from("ts,symbol,open,high,low,close,volume,quote_volume,member_asof\n");
    for (t, sym, o, h, l, c, v, asof) in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{}\n",
            t,
            sym,
            o,
            h,
            l,
            c,
            v,
            v * c,
            asof
        ));
    }
    let t = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &t, group_rows, Some(0)).unwrap();
}

/// 21-symbol, symbol-major panel used by the cross-asset tests. Each symbol has
/// a distinct deterministic drift so market mean, breadth and dispersion are
/// non-degenerate; the last symbol can be marked pre-listing.
fn cross_panel_rows(
    n: usize,
    step: i64,
    member_asof: f64,
    pre_listing_symbol: Option<&str>,
    pre_listing_asof: f64,
) -> Vec<(i64, String, f64, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for k in 0..21usize {
        let sym = if k == 0 {
            "BTCUSDT".to_string()
        } else {
            format!("ALT{:02}USDT", k)
        };
        let asof = if pre_listing_symbol == Some(sym.as_str()) {
            pre_listing_asof
        } else {
            member_asof
        };
        let mu = 0.0002 * (k as f64 - 10.0);
        let phase = k as f64 * 0.41;
        for i in 0..n {
            let p = 100.0 * (mu * i as f64 + 0.03 * (i as f64 * 0.23 + phase).sin()).exp();
            let o = p * 0.999;
            let c = p;
            let h = p * 1.002;
            let l = p * 0.998;
            let v = 1000.0 + 10.0 * k as f64 + (i % 7) as f64;
            rows.push((i as i64 * step, sym.clone(), o, h, l, c, v, asof));
        }
    }
    rows
}

fn flow_rows(n: usize) -> Vec<(i64, String, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|i| {
            let p = 100.0 + (i as f64 * 0.17).sin() * 1.5 + i as f64 * 0.02;
            let o = p - 0.13;
            let c = p + 0.07;
            let h = p.max(o).max(c) + 0.21;
            let l = p.min(o).min(c) - 0.19;
            let v = 100.0 + i as f64 * 0.5;
            let qv = v * (p + 0.03) + (i % 7) as f64 * 0.11;
            let trades = 50.0 + i as f64 * 0.25;
            let tb = v * 0.61 + (i % 5) as f64 * 0.07;
            let tbq = tb * p;
            (i as i64 * 900, "AAA".to_string(), o, h, l, c, v, qv, trades, tb, tbq)
        })
        .collect()
}

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-enrich-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn opts(data: &str, out: &str, features: &str) -> Options {
    Options {
        data: data.to_string(),
        out: Some(out.to_string()),
        features: vec![features.to_string()],
        ..Default::default()
    }
}

/// The 21-symbol fixtures carry BTC but no separate ETH series; point the ETH
/// benchmark at BTC so the complete `cross` group can be exercised. The
/// dedicated benchmark test above covers a distinct ETH series.
fn cross_opts(data: &str, out: &str, features: &str) -> Options {
    let mut o = opts(data, out, features);
    o.eth_symbol = "BTCUSDT".to_string();
    o
}

fn column(path: &str, name: &str) -> Vec<f64> {
    let s = FileStore::open(path).unwrap();
    let c = s.header().column_index(name).unwrap();
    let mut out = Vec::new();
    for g in 0..s.n_groups() {
        out.extend(s.read_f64_group(g, c).unwrap());
    }
    out
}

#[test]
fn catalogue_expands_every_family_and_refuses_typos() {
    let returns = enrich::expand("returns").unwrap();
    assert_eq!(returns.len(), 120);
    assert_eq!(returns.first().unwrap(), "r1");
    assert_eq!(returns.last().unwrap(), "r120");
    for g in ["ma", "vol", "atr", "rsi", "bollinger", "vz", "range", "candle", "donchian", "calendar", "lags", "xs", "btc", "py15m"] {
        assert!(!enrich::expand(g).unwrap().is_empty(), "group {g}");
    }
    let all = enrich::expand("all").unwrap();
    for n in ["r1", "r120", "ma200", "vol120", "atr14", "rsi14", "bb20_z", "vz96", "range1", "clv1", "don96", "dow_sin", "lag_r1_10", "xs_rank_r1", "beta_btc_60"] {
        assert!(all.contains(&n.to_string()), "all is missing {n}");
    }
    assert_eq!(enrich::py15m_names().len(), 35);
    let flow = enrich::expand("flow").unwrap();
    assert_eq!(flow, enrich::flow_names());
    assert_eq!(flow.len(), 10);
    for n in [
        "vwap",
        "taker_ratio96",
        "taker_imb16",
        "trade_intensity96",
        "avg_trade_size96",
        "amihud96",
        "zero_ret_share96",
        "obv_z96",
        "cvd_proxy",
        "vwap_dev96",
    ] {
        assert!(flow.contains(&n.to_string()), "flow is missing {n}");
    }
    assert!(enrich::usage().contains("flow"));
    assert!(enrich::expand("nope_not_a_feature").is_err());
    assert!(enrich::expand("ta:core").unwrap().contains(&"ret1".to_string()));
}

#[test]
fn append_preserves_every_existing_column_and_metadata() {
    let dir = tmp_dir("append");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows = bars("AAA", 40, 0, 3600);
    rows.extend(bars("BBB", 40, 0, 3600));
    rows.extend(bars("CCC", 40, 0, 3600));
    write_store(&src, &rows, 7);
    let before = std::fs::read(&src).unwrap();
    let src_store = FileStore::open(&src).unwrap();
    let old_names = src_store.header().names();
    let old_levels = src_store.header().levels(src_store.header().column_index("symbol").unwrap()).unwrap().to_vec();
    let old_keep = column(&src, "keep");

    let report = enrich::build(&opts(&src, &out, "r1,ma20")).unwrap();
    assert_eq!(report.columns_in, old_names.len());
    assert_eq!(report.columns_out, old_names.len() + 2);
    assert_eq!(report.rows, 120);
    // The append writer hashes while writing; the digest must equal a plain
    // re-read of the output file.
    assert_eq!(
        report.output_sha256,
        rcsr::manifest::sha256_file(&out).unwrap(),
        "streamed output hash differs from a re-read"
    );

    let dst = FileStore::open(&out).unwrap();
    let new_names = dst.header().names();
    assert_eq!(&new_names[..old_names.len()], &old_names[..], "existing order changed");
    assert_eq!(&new_names[old_names.len()..], &["r1", "ma20"]);
    assert_eq!(
        dst.header().levels(dst.header().column_index("symbol").unwrap()).unwrap(),
        &old_levels[..],
        "dictionary levels changed"
    );
    let keep_col = dst.header().column_index("keep").unwrap();
    let mut keep = Vec::new();
    for g in 0..dst.n_groups() {
        keep.extend(dst.read_f64_group(g, keep_col).unwrap());
    }
    assert_eq!(keep, old_keep, "existing column values changed");
    // The input store is untouched.
    assert_eq!(std::fs::read(&src).unwrap(), before);
}

#[test]
fn duplicate_columns_need_replace_and_replace_is_stable() {
    let dir = tmp_dir("replace");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out1 = dir.join("out1.rcsb").to_string_lossy().into_owned();
    let out2 = dir.join("out2.rcsb").to_string_lossy().into_owned();
    write_store(&src, &bars("AAA", 200, 0, 3600), 64);
    let first = enrich::build(&opts(&src, &out1, "r1,r5")).unwrap();
    let h1 = first.features.iter().find(|f| f.name == "r1").unwrap().hash.clone();

    let mut dup = opts(&out1, &out2, "r1");
    let err = enrich::build(&dup).unwrap_err();
    assert!(err.contains("already exists"), "unexpected error: {err}");

    dup.replace = true;
    let second = enrich::build(&dup).unwrap();
    assert_eq!(second.columns_out, second.columns_in, "replace must not grow the schema");
    let h2 = second.features.iter().find(|f| f.name == "r1").unwrap().hash.clone();
    assert_eq!(h1, h2, "replacing an identical column must hash identically");
}

#[test]
fn feature_hashes_and_store_checksum_are_identical_across_chunk_sizes() {
    let dir = tmp_dir("chunks");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let a = dir.join("a.rcsb").to_string_lossy().into_owned();
    let b = dir.join("b.rcsb").to_string_lossy().into_owned();
    let mut rows = bars("AAA", 300, 0, 900);
    rows.extend(bars("BBB", 260, 0, 900));
    write_store(&src, &rows, 32);
    let features = "returns,ma,vol,rsi,bollinger,vz,range,candle,donchian,calendar,lags";
    let mut oa = opts(&src, &a, features);
    oa.chunk_rows = 7;
    let mut ob = opts(&src, &b, features);
    ob.chunk_rows = 100_000;
    let ra = enrich::build(&oa).unwrap();
    let rb = enrich::build(&ob).unwrap();
    let hashes = |r: &enrich::Report| r.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect::<Vec<_>>();
    assert_eq!(hashes(&ra), hashes(&rb));
    assert_eq!(ra.output_sha256, rb.output_sha256);
    assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
}

#[test]
fn gap_windows_are_nan_and_a_prefix_is_causal() {
    let dir = tmp_dir("gaps");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let short_store = dir.join("short.rcsb").to_string_lossy().into_owned();
    let short_out = dir.join("short_out.rcsb").to_string_lossy().into_owned();
    // 100 hourly bars with a three-bar hole after index 49.
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for i in 0..100i64 {
        let t = if i < 50 { i * 3600 } else { (i + 3) * 3600 };
        let p = 100.0 + i as f64 * 0.5;
        rows.push((t, "AAA".to_string(), p, p + 0.3, p - 0.4, p + 0.05, 10.0 + i as f64));
    }
    write_store(&src, &rows, 16);
    write_store(&short_store, &rows[..80], 16);
    let features = "r1,r5,ma20,vol20,atr14,rsi14";
    enrich::build(&opts(&src, &out, features)).unwrap();
    enrich::build(&opts(&short_store, &short_out, features)).unwrap();
    let r1 = column(&out, "r1");
    let r5 = column(&out, "r5");
    let ma20 = column(&out, "ma20");
    assert!(r1[50].is_nan(), "a return across a hole must be missing");
    assert!(r5[53].is_nan(), "r5 still spans the hole");
    assert!(r5[54].is_nan(), "r5 at 54 still reaches back across the hole");
    assert!(r5[55].is_finite(), "r5 is valid five contiguous bars after the hole");
    assert!(ma20[60].is_nan(), "ma20 spans the hole");
    assert!(ma20[72].is_finite(), "ma20 valid 20 contiguous bars after the hole");

    for name in ["r1", "ma20", "vol20", "atr14", "rsi14"] {
        let full = column(&out, name);
        let short = column(&short_out, name);
        for i in 0..80 {
            let (a, b) = (full[i], short[i]);
            assert_eq!(a.is_nan(), b.is_nan(), "{name} differs at {i}");
            if a.is_finite() {
                assert!((a - b).abs() <= 1e-12, "{name} at {i}: {a} vs {b}");
            }
        }
    }
}

#[test]
fn labels_enter_at_the_next_open_and_stay_out_of_features() {
    let dir = tmp_dir("label");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows = bars("AAA", 50, 0, 3600);
    write_store(&src, &rows, 8);
    let mut o = opts(&src, &out, "r1");
    o.labels = vec!["fwd:1,cost=0".to_string()];
    let report = enrich::build(&o).unwrap();
    let fwd = column(&out, "fwd1");
    assert_eq!(fwd.len(), 50);
    assert!(fwd[48].is_finite());
    assert!(fwd[49].is_nan(), "the last row has no next open");
    let entry = rows[1].2; // open[1]
    let exit = rows[1].5; // close[1]
    assert!((fwd[0] - (exit / entry).ln()).abs() < 1e-12);
    assert!(report.features.iter().any(|f| f.name == "fwd1" && f.label));
}

#[test]
fn cross_sectional_ranks_require_twenty_members_and_scale_to_minus_one_one() {
    // 20 members at each of two timestamps. At t=0 values 1..19 and a tie at
    // 19; at t=1 a reversed ramp. The rank scale is [-1, 1].
    let mut src = Vec::new();
    let mut times = Vec::new();
    let mut keys = Vec::new();
    for i in 0..20u32 {
        src.push(if i >= 18 { 19.0 } else { (i + 1) as f64 });
        times.push(0i64);
        keys.push(i);
    }
    for i in 0..20u32 {
        src.push((20 - i) as f64);
        times.push(1i64);
        keys.push(i);
    }
    let r = enrich::xs_rank(&src, &times, &keys, 40).unwrap();
    assert_eq!(r[0], -1.0);
    assert!((r[18] - (2.0 * 18.0 / 19.0 - 1.0)).abs() < 1e-12);
    assert_eq!(r[19], 1.0, "tie broken by symbol order");
    assert_eq!(r[20], 1.0);
    assert_eq!(r[39], -1.0);
    assert!(r.iter().all(|v| (-1.0..=1.0).contains(v)));
    // Repeating the call is bit-identical.
    assert_eq!(r, enrich::xs_rank(&src, &times, &keys, 40).unwrap());

    // A 19-member timestamp is NaN, never a rank over a partial cross-section.
    let short = enrich::xs_rank(&src[1..], &times[1..], &keys[1..], 39).unwrap();
    assert!(short[..19].iter().all(|v| v.is_nan()));
    assert!(short[19..].iter().all(|v| v.is_finite()));
}

#[test]
fn cost_model_keeps_small_stores_on_the_cpu_and_names_the_reason() {
    let mut counts = std::collections::BTreeMap::new();
    counts.insert("returns", (120usize, OpClass::Elementwise));
    let mut supported = std::collections::BTreeMap::new();
    supported.insert("returns", 120usize);
    let small = enrich::plan_groups(&counts, &supported, 10_000, GpuMode::Auto, true);
    assert_eq!(small[0].backend, "cpu");
    assert!(small[0].reason.contains("below"));
    assert_eq!(small[0].gpu_cols, 0);
    let large = enrich::plan_groups(&counts, &supported, 3_900_000, GpuMode::Auto, true);
    assert!(large[0].gpu_ms < large[0].cpu_ms, "the cost model should still see a GPU win");
    if enrich::gpu_kernels_compiled() {
        assert_eq!(large[0].backend, "gpu");
        assert_eq!(large[0].gpu_cols, 120);
        assert_eq!(large[0].cpu_cols, 0);
    } else {
        assert_eq!(large[0].backend, "cpu");
        assert!(large[0].reason.contains("without --features cuda"));
    }
    let off = enrich::plan_groups(&counts, &supported, 3_900_000, GpuMode::Off, true);
    assert_eq!(off[0].backend, "cpu");
    let none = enrich::plan_groups(&counts, &supported, 3_900_000, GpuMode::Auto, false);
    assert!(
        none[0].reason.to_ascii_lowercase().contains("cuda"),
        "reason: {}",
        none[0].reason
    );
    let unregistered = enrich::plan_groups(
        &counts,
        &std::collections::BTreeMap::new(),
        3_900_000,
        GpuMode::Auto,
        true,
    );
    assert_eq!(unregistered[0].backend, "cpu");
    assert!(unregistered[0].reason.contains("no exact GPU kernel"));
}

#[test]
fn cost_model_v2_charges_the_measured_write_and_a_slow_disk_flips_the_verdict() {
    let mut counts = std::collections::BTreeMap::new();
    counts.insert("returns", (120usize, OpClass::Elementwise));
    let mut supported = std::collections::BTreeMap::new();
    supported.insert("returns", 120usize);
    let rows = 3_900_000usize;
    let fast = enrich::plan_groups_with_write_bw(
        &counts,
        &supported,
        rows,
        GpuMode::Auto,
        true,
        enrich::GPU_WRITE_BYTES_PER_SECOND,
    );
    assert!(
        fast[0].gpu_ms < fast[0].cpu_ms,
        "measured 222 MB/s keeps returns on the GPU"
    );
    // A disk that writes 5 MB/s makes the same family's output write dominate.
    let slow = enrich::plan_groups_with_write_bw(
        &counts,
        &supported,
        rows,
        GpuMode::Auto,
        true,
        5_000_000.0,
    );
    assert!(
        slow[0].gpu_ms > fast[0].gpu_ms,
        "the write term must change the estimate"
    );
    assert_eq!(slow[0].backend, "cpu");
    if enrich::gpu_kernels_compiled() {
        assert!(slow[0].reason.contains("not below"));
    }
    // The default entry point still uses the measured filesystem constant.
    let default = enrich::plan_groups(&counts, &supported, rows, GpuMode::Auto, true);
    assert_eq!(default[0].gpu_ms, fast[0].gpu_ms);
}

#[test]
fn output_stage_is_bounded_at_64_mib_for_any_column_count() {
    let wide = rcsr::memory::enrich_output_stage_bytes(16_000_000, 358);
    assert_eq!(
        wide,
        rcsr::memory::GPU_OUTPUT_ARENA_BYTES + rcsr::memory::ENRICH_TEMP_WRITE_CHUNK_BYTES
    );
    let narrow = rcsr::memory::enrich_output_stage_bytes(1_000, 2);
    assert_eq!(
        narrow,
        1_000 * 2 * 8 + rcsr::memory::ENRICH_TEMP_WRITE_CHUNK_BYTES
    );
}

/// #300 measurement: calibrate `--write-bw` for the filesystem that will hold
/// the temp feature files and the output store. Ignored because it writes
/// scratch bytes; run it explicitly.
#[test]
#[ignore]
fn measure_write_bandwidth_on_a_filesystem() {
    let dir = std::env::var("RCSR_BW_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let bps = enrich::measure_write_bandwidth(&dir, 128 * 1024 * 1024).unwrap();
    eprintln!(
        "# write bandwidth {}: {:.1} MB/s",
        dir.display(),
        bps / 1e6
    );
    assert!(bps > 10_000_000.0, "implausibly slow durable write");
}

#[test]
fn dry_run_writes_nothing() {
    let dir = tmp_dir("dry");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &bars("AAA", 30, 0, 3600), 8);
    let mut o = opts(&src, &out, "r1");
    o.dry_run = true;
    let r = enrich::build(&o).unwrap();
    assert_eq!(r.rows, 30);
    assert!(!std::path::Path::new(&out).exists());
}

/// Parity with the Python `enrich_panel_15m.py` step on a committed BTC slice.
///
/// The fixture is 1,400 contiguous 15m BTCUSDT rows taken from the pilot's
/// `panel_15m_rich.csv`; the expected file holds the Python values for the last
/// 400 rows. The 1,000-row pre-roll lets the recursive EWMAs/Wilders forget the
/// 192 base rows that the Python panel dropped before its first output row.
/// Tolerance is pre-registered at 1e-6 absolute or relative, whichever is
/// larger. Two deliberate correctness fixes are documented in
/// [`enrich::py15m_names`]: gaps are not bridged, and the first true range
/// does not read the last close.
#[test]
fn py15m_matches_the_python_panel_within_tolerance() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/enrich");
    let input = base.join("btc_15m_ohlcv.csv");
    let expected_path = base.join("btc_15m_py15m_expected.csv");
    let dir = tmp_dir("py15m");
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();

    // The #273 market-state columns now require 20 point-in-time members, so
    // exercise the fixture on a 20-member slice whose members carry the same
    // BTC bars. The equal-weight market state is then numerically BTC, which
    // preserves the committed Python expectation for `mkt_*` while testing the
    // new producer. BTCUSDT is the last block so the compared tail is its rows.
    let fixture_text = std::fs::read_to_string(&input).unwrap();
    let mut fixture_lines = fixture_text.lines();
    let fixture_header = fixture_lines.next().unwrap();
    let fixture_body: Vec<&str> = fixture_lines.filter(|l| !l.trim().is_empty()).collect();
    let mut panel = String::from(fixture_header);
    panel.push('\n');
    let mut symbols: Vec<String> = (0..19).map(|k| format!("X{:02}USDT", k)).collect();
    symbols.push("BTCUSDT".to_string());
    for sym in &symbols {
        for line in &fixture_body {
            let mut fields: Vec<&str> = line.split(',').collect();
            fields[1] = sym;
            panel.push_str(&fields.join(","));
            panel.push('\n');
        }
    }
    let input = dir.join("panel20.csv");
    std::fs::write(&input, panel).unwrap();

    let text = std::fs::read_to_string(&expected_path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let mut expected: Vec<Vec<f64>> = Vec::new();
    let mut expected_ts: Vec<i64> = Vec::new();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        expected_ts.push(fields[0].parse().unwrap());
        expected.push(fields[1..].iter().map(|v| v.parse::<f64>().unwrap()).collect());
    }
    assert_eq!(expected.len(), 400);

    let report = enrich::build(&opts(
        &input.to_string_lossy(),
        &out,
        "py15m",
    ))
    .unwrap();
    assert_eq!(report.features.len(), 35);
    let times: Vec<i64> = column(&out, "ts").into_iter().map(|v| v as i64).collect();
    // The fixture's own `ts` column is the first base column; compare the
    // last 400 rows by timestamp.
    let start = times.len() - 400;
    assert_eq!(&times[start..], &expected_ts[..], "fixture row alignment");
    let mut worst = 0.0f64;
    let mut worst_name = "";
    for (j, name) in header.iter().enumerate().skip(1) {
        let got = column(&out, name);
        for (i, exp) in expected.iter().enumerate() {
            let a = got[start + i];
            let b = exp[j - 1];
            if a.is_nan() && b.is_nan() {
                continue;
            }
            let tol = 1e-6f64.max(1e-6 * b.abs());
            let d = (a - b).abs();
            if !(d <= tol) {
                panic!("{name} row {i}: got {a}, expected {b}, diff {d} > {tol}");
            }
            if d > worst {
                worst = d;
                worst_name = name;
            }
        }
    }
    eprintln!("py15m parity: worst absolute difference {worst:e} ({worst_name})");
}


/// The compact profile is the parity/regression subset of the default financial
/// feature set. It must emit exactly the 52 names in compact order.
#[test]
fn compact_profile_is_the_exact_52_name_order() {
    let names = enrich::expand("compact").unwrap();
    assert_eq!(names, enrich::compact_names());
    assert_eq!(names.len(), 52);
    assert_eq!(names[0], "r16");
    assert_eq!(names[51], "don96");
    assert_eq!(&names[43..47], &["btc_r96", "btc_vol96", "rel_btc96", "beta384"]);
}

/// Every compact name is registered `available:true`; `wasym1` is `derived`.
/// Every registered but unavailable default-set column names its dependency.
#[test]
fn default_financial_registry_is_complete_and_explicit() {
    let reg = enrich::default_registry();
    let mut seen = std::collections::BTreeSet::new();
    for e in &reg {
        assert!(seen.insert(e.name.clone()), "duplicate registry name {}", e.name);
        assert!(!e.formula.is_empty(), "{} has no formula", e.name);
        assert!(!e.family.is_empty(), "{} has no family", e.name);
        if !e.available {
            assert!(e.reason.as_deref().map(|r| !r.is_empty()).unwrap_or(false), "{} unavailable without reason", e.name);
        }
    }
    for name in enrich::COMPACT_NAMES {
        let e = reg.iter().find(|e| e.name == *name).unwrap_or_else(|| panic!("{name} missing from registry"));
        assert!(e.available, "{name} must be available in the compact profile");
        assert_eq!(e.timeframe, "15m");
        assert_eq!(e.derived, *name == "wasym1");
    }
    // All existing engine columns are registered available.
    for name in enrich::expand("all").unwrap() {
        let e = reg.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("{name} missing"));
        assert!(e.available, "{name} must be available");
    }
    for name in enrich::py15m_names() {
        let e = reg.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("{name} missing"));
        assert!(e.available, "{name} must be available");
    }
    // Complete OHLCV/flow family (#270): vwap plus the nine derived columns.
    for name in enrich::FLOW_NAMES {
        let e = reg
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("flow {name} missing from registry"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.family, "flow", "{name} family");
        assert_eq!(e.timeframe, "15m", "{name} timeframe");
        assert!(e.formula.contains("mean") || e.formula.contains("quote_volume") || e.formula.contains("OBV") || e.formula.contains("cumulative"), "{name} formula: {}", e.formula);
        let schema = rcsr::schema::json();
        assert!(schema.contains(&format!("\"name\":\"{name}\"")), "{name} absent from schema");
    }
    // Every raw OHLCV/flow field is registered with original units.
    for name in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "vwap",
    ] {
        let e = reg.iter().find(|e| e.name == name).unwrap_or_else(|| panic!("{name} missing"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.source, if name == "vwap" { "raw OHLCV/flow columns" } else { "input" }, "{name} source");
    }
    // All 36 multi-timeframe columns are registered and available.
    let mtf = enrich::mtf_names();
    assert_eq!(mtf.len(), 36);
    for name in &mtf {
        let e = reg
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("{name} missing from registry"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.family, "mtf");
        assert!(matches!(e.timeframe, "1h" | "4h" | "1d"));
        assert!(e.reason.is_none());
        assert_eq!(enrich::expand(name).unwrap(), vec![name.clone()]);
    }
    assert_eq!(enrich::expand("mtf").unwrap().len(), 36);

    // Unavailable families are explicit, never silently omitted.
    let unavailable: Vec<&enrich::RegistryEntry> = reg.iter().filter(|e| !e.available).collect();
    assert!(!unavailable.is_empty());
    assert!(unavailable.iter().any(|e| e.family == "derivatives"));
    // The complete #273 cross family is available with family=cross, and every
    // name expands to a producer.
    for name in enrich::cross_names() {
        let e = reg
            .iter()
            .find(|e| e.name == name)
            .unwrap_or_else(|| panic!("{name} missing from registry"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.family, "cross", "{name} family");
        assert_eq!(e.timeframe, "15m");
        assert!(!e.formula.is_empty());
        assert!(enrich::expand(&name).is_ok(), "{name} must expand");
    }
    assert_eq!(enrich::expand("cross").unwrap(), enrich::cross_names());
    // Issue #261: the candle/price-action family is complete and available.
    for name in enrich::DEFAULT_CANDLE_NAMES {
        let e = reg
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("candle {name} missing from registry"));
        assert!(e.available, "{name} must be available");
        assert_eq!(e.timeframe, "15m");
        assert!(!e.derived, "{name} must be materialised");
        if enrich::PRICE_ACTION_NAMES.contains(name) {
            assert!(
                e.source.contains("not a target construction column"),
                "{name} must record that the family is not a target construction column"
            );
        }
        assert!(enrich::expand(name).is_ok(), "{name} must expand");
    }
    for name in enrich::PRICE_ACTION_NAMES {
        let e = reg.iter().find(|e| e.name == *name).unwrap();
        assert!(e.available, "{name} must be available");
        assert_eq!(e.family, "price_action");
        assert!(!e.reason.as_deref().unwrap_or("").contains("not implemented"));
    }
    for name in enrich::DERIVATIVE_NAMES {
        let e = reg.iter().find(|e| e.name == *name).unwrap();
        assert!(!e.available);
        assert!(e.reason.as_deref().unwrap().contains("no local"));
    }
    assert!(enrich::expand("mtf_2h_ret1").unwrap_err().contains("unknown"));
    assert!(enrich::expand("funding_rate").unwrap_err().contains("unavailable"));
    // The schema prints the registry.
    assert!(rcsr::schema::json().contains("\"feature_registry\""));
    assert!(rcsr::schema::json().contains("\"wasym1\""));
}

/// `wasym1 = body1 - 2*clv1` on every non-degenerate bar, and the registry
/// marks it derived. The compact `H==L` convention is documented separately.
#[test]
fn wasym1_identity_holds_on_finite_range_bars() {
    let dir = tmp_dir("wasym");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for i in 0..240i64 {
        let base = 100.0 + (i as f64 * 0.37).sin() * 3.0;
        let o = base + (i as f64 * 0.11).cos() * 0.4;
        let c = base + (i as f64 * 0.19).sin() * 0.5;
        let h = o.max(c) + 0.2 + (i % 5) as f64 * 0.03;
        let l = o.min(c) - 0.25 - (i % 3) as f64 * 0.02;
        rows.push((i * 900, "AAA".to_string(), o, h, l, c, 10.0 + i as f64));
    }
    write_store(&src, &rows, 32);
    enrich::build(&opts(&src, &out, "clv1,body1,wasym1,lower_wick,don96")).unwrap();
    let clv = column(&out, "clv1");
    let body = column(&out, "body1");
    let wasym = column(&out, "wasym1");
    let mut n = 0;
    for i in 0..rows.len() {
        assert!(clv[i] >= -0.5 && clv[i] <= 0.5, "clv1 out of range at {i}");
        assert!(body[i] >= -1.0 && body[i] <= 1.0, "body1 out of range at {i}");
        if wasym[i].is_finite() && body[i].is_finite() && clv[i].is_finite() {
            let identity = body[i] - 2.0 * clv[i];
            assert!(
                (wasym[i] - identity).abs() <= 1e-12,
                "wasym1 identity at {i}: {} vs {}",
                wasym[i],
                identity
            );
            n += 1;
        }
    }
    assert!(n > 200, "identity was tested on too few bars ({n})");
}

/// A compact window that spans a hole is NaN; it is never bridged.
#[test]
fn compact_windows_are_gap_masked() {
    let dir = tmp_dir("compact_gap");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for i in 0..700i64 {
        let t = if i < 350 { i * 900 } else { (i + 3) * 900 };
        let p = 100.0 + i as f64 * 0.01 + (i as f64 * 0.3).sin();
        rows.push((t, "AAA".to_string(), p, p + 0.2, p - 0.2, p + 0.05, 10.0 + i as f64));
    }
    write_store(&src, &rows, 64);
    enrich::build(&opts(&src, &out, "r16,vol16,ma96,don96,acf1,ou_hl96,volrat")).unwrap();
    for name in ["r16", "vol16", "ma96", "don96", "acf1", "ou_hl96"] {
        let v = column(&out, name);
        for i in 350..360 {
            assert!(v[i].is_nan(), "{name} bridged the hole at row {i}");
        }
    }
}

/// Raw flow fields are required for the flow features; a missing column is a
/// hard error naming it, never a zero column.
#[test]
fn compact_flow_features_require_raw_trades_and_taker_columns() {
    let dir = tmp_dir("compact_raw");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &bars("AAA", 120, 0, 900), 16);
    let err = enrich::build(&opts(&src, &out, "trade_intensity96")).unwrap_err();
    assert!(err.contains("trades"), "unexpected error: {err}");
    let err = enrich::build(&opts(&src, &out, "taker_ratio96")).unwrap_err();
    assert!(err.contains("taker_buy_base"), "unexpected error: {err}");
}

/// Every derived flow formula on hand-built bars, to 1e-12 absolute-or-relative.
#[test]
fn flow_formulas_match_hand_built_bars() {
    let dir = tmp_dir("flow_formulas");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows = flow_rows(160);
    write_flow_store(&src, &rows, false, 32);
    enrich::build(&opts(&src, &out, "flow")).unwrap();

    let i = rows.len() - 1;
    let lo = i + 1 - 96;
    let close: Vec<f64> = rows.iter().map(|r| r.5).collect();
    let vol: Vec<f64> = rows.iter().map(|r| r.6).collect();
    let qv: Vec<f64> = rows.iter().map(|r| r.7).collect();
    let trades: Vec<f64> = rows.iter().map(|r| r.8).collect();
    let tb: Vec<f64> = rows.iter().map(|r| r.9).collect();
    let mean = |x: &[f64], a: usize, b: usize| -> f64 {
        x[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
    };
    let close_at = |name: &str| column(&out, name)[i];
    let check = |name: &str, got: f64, want: f64| {
        let a = (got - want).abs();
        let r = a / want.abs().max(1e-300);
        assert!(
            a <= 1e-12 || r <= 1e-12,
            "{name}: got {got}, expected {want}, abs {a:e} rel {r:e}"
        );
    };

    check("vwap", close_at("vwap"), qv[i] / vol[i]);
    check(
        "taker_ratio96",
        close_at("taker_ratio96"),
        mean(&tb, lo, i) / mean(&vol, lo, i),
    );
    let lo16 = i + 1 - 16;
    check(
        "taker_imb16",
        close_at("taker_imb16"),
        (2.0 * mean(&tb, lo16, i) - mean(&vol, lo16, i)) / mean(&vol, lo16, i),
    );
    check(
        "trade_intensity96",
        close_at("trade_intensity96"),
        (1.0 + mean(&trades, lo, i) / mean(&vol, lo, i)).ln(),
    );
    let per_trade: Vec<f64> = (0..=i).map(|j| qv[j] / trades[j].max(1.0)).collect();
    check(
        "avg_trade_size96",
        close_at("avg_trade_size96"),
        (1.0 + mean(&per_trade, lo, i)).ln(),
    );

    let mut lr = vec![f64::NAN; rows.len()];
    for j in 1..=i {
        lr[j] = (close[j] / close[j - 1]).ln();
    }
    let amihud: Vec<f64> = (0..=i).map(|j| lr[j].abs() / qv[j]).collect();
    check("amihud96", close_at("amihud96"), mean(&amihud, lo, i) * 1e6);
    let zero: Vec<f64> = (0..=i).map(|j| if lr[j] == 0.0 { 1.0 } else { 0.0 }).collect();
    check("zero_ret_share96", close_at("zero_ret_share96"), mean(&zero, lo, i));

    let mut obv = vec![0.0f64; rows.len()];
    for j in 1..=i {
        let sign = if close[j] > close[j - 1] { 1.0 } else if close[j] < close[j - 1] { -1.0 } else { 0.0 };
        obv[j] = obv[j - 1] + sign * vol[j];
    }
    let om = mean(&obv, lo, i);
    let osd = (obv[lo..=i].iter().map(|x| (x - om) * (x - om)).sum::<f64>() / 96.0).sqrt();
    check("obv_z96", close_at("obv_z96"), (obv[i] - om) / osd);

    let mut cvd = vec![0.0f64; rows.len()];
    for j in 0..=i {
        cvd[j] = (if j == 0 { 0.0 } else { cvd[j - 1] }) + (2.0 * tb[j] - vol[j]);
    }
    let cm = mean(&cvd, lo, i);
    let csd = (cvd[lo..=i].iter().map(|x| (x - cm) * (x - cm)).sum::<f64>() / 96.0).sqrt();
    check("cvd_proxy", close_at("cvd_proxy"), cvd[i] / csd);

    let vwap96 = mean(&qv, lo, i) / mean(&vol, lo, i);
    check("vwap_dev96", close_at("vwap_dev96"), (close[i] - vwap96) / vwap96);
}

/// Zero volume and zero trades yield NaN for the affected ratios and
/// illiquidity; `avg_trade_size96` keeps its explicit max(trades, 1).
#[test]
fn flow_zero_volume_and_zero_trades_are_nan_not_zero() {
    let dir = tmp_dir("flow_zero");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows = flow_rows(160);
    for r in &mut rows {
        r.6 = 0.0;
        r.7 = 0.0;
        r.8 = 0.0;
        r.9 = 0.0;
        r.10 = 0.0;
    }
    write_flow_store(&src, &rows, false, 32);
    enrich::build(&opts(&src, &out, "flow")).unwrap();
    let last = rows.len() - 1;
    for name in [
        "vwap",
        "vwap_dev96",
        "taker_ratio96",
        "taker_imb16",
        "trade_intensity96",
        "amihud96",
        "obv_z96",
        "cvd_proxy",
    ] {
        assert!(column(&out, name)[last].is_nan(), "{name} must be NaN at zero volume/trades");
    }

    let dir = tmp_dir("flow_zero_trades");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows = flow_rows(160);
    for r in &mut rows {
        r.8 = 0.0;
    }
    write_flow_store(&src, &rows, false, 32);
    enrich::build(&opts(&src, &out, "flow")).unwrap();
    let last = rows.len() - 1;
    assert!(column(&out, "trade_intensity96")[last].is_nan(), "zero trades must make trade_intensity96 NaN");
    assert!(
        column(&out, "avg_trade_size96")[last].is_finite(),
        "avg_trade_size96 uses max(trades,1) and stays finite"
    );
    assert!(column(&out, "taker_ratio96")[last].is_finite(), "taker ratio is unaffected by zero trades");
}

/// A flow window that spans a hole is NaN; raw vwap is still finite and
/// nothing is forward-filled.
#[test]
fn flow_windows_are_gap_masked() {
    let dir = tmp_dir("flow_gap");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows = flow_rows(300);
    for r in &mut rows {
        if (120..123).contains(&(r.0 / 900)) {
            r.0 += 3 * 900;
        } else if r.0 / 900 >= 123 {
            r.0 += 3 * 900;
        }
    }
    write_flow_store(&src, &rows, false, 32);
    enrich::build(&opts(&src, &out, "flow")).unwrap();
    let names = [
        "taker_ratio96",
        "trade_intensity96",
        "avg_trade_size96",
        "amihud96",
        "zero_ret_share96",
        "obv_z96",
        "cvd_proxy",
        "vwap_dev96",
    ];
    for name in names {
        let v = column(&out, name);
        for i in 120..=120 + 94 {
            assert!(v[i].is_nan(), "{name} bridged the gap at row {i}");
        }
    }
    let imb = column(&out, "taker_imb16");
    for i in 120..=120 + 14 {
        assert!(imb[i].is_nan(), "taker_imb16 bridged the gap at row {i}");
    }
    assert!(column(&out, "vwap_dev96")[215].is_finite(), "a full post-gap window must become finite again");
    let vwap = column(&out, "vwap");
    assert!(vwap.iter().all(|x| x.is_finite()), "raw vwap does not need a window");
}

/// The four flow names shared with the compact profile pass the compact
/// regression fixture at abs/rel <= 1e-12.
#[test]
fn flow_shared_names_match_the_compact_regression_fixture() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/enrich");
    let input = base.join("compact_15m_ohlcv.csv");
    let expected_path = base.join("compact_15m_expected.csv");
    let dir = tmp_dir("flow_parity");
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    enrich::build(&opts(&input.to_string_lossy(), &out, "flow")).unwrap();

    let text = std::fs::read_to_string(&expected_path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let wanted = ["taker_ratio96", "taker_imb16", "trade_intensity96", "vwap_dev96"];
    let start_ts = 1_680_271_200i64;
    let rows_per_symbol = 3600usize;
    let out_cols: Vec<Vec<f64>> = wanted.iter().map(|n| column(&out, n)).collect();
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let ts: i64 = fields[0].parse().unwrap();
        let sym = fields[1];
        let seg = if sym == "AAVEUSDT" { 0 } else { 1 };
        let idx = seg * rows_per_symbol + ((ts - start_ts) / 900) as usize;
        for (j, name) in wanted.iter().enumerate() {
            let col = header.iter().position(|h| h == name).unwrap();
            let expected = fields[col].parse::<f64>().unwrap();
            let got = out_cols[j][idx];
            match (got.is_nan(), expected.is_nan()) {
                (true, true) => continue,
                (true, false) => panic!("{name} at {sym}/{ts}: got NaN, expected {expected}"),
                (false, true) => panic!("{name} at {sym}/{ts}: got {got}, expected NaN"),
                (false, false) => {}
            }
            let a = (got - expected).abs();
            let r = a / expected.abs().max(1e-300);
            worst = worst.max(a.max(r));
            assert!(
                a <= 1e-12 || r <= 1e-12,
                "{name} at {sym}/{ts}: got {got}, expected {expected}, abs {a:e} rel {r:e}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 300 * wanted.len());
    eprintln!("flow compact parity: worst abs/rel {worst:e}");
}

/// A raw OHLCV/flow fixture round-trips every field exactly, including the
/// optional `vwap` column, while enrich appends derived columns.
#[test]
fn raw_ohlcv_flow_fields_round_trip_exactly() {
    let dir = tmp_dir("flow_roundtrip");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows = flow_rows(80);
    write_flow_store(&src, &rows, true, 16);
    let before = FileStore::open(&src).unwrap();
    let old_names = before.header().names();
    let old_dtypes = before.header().dtypes();
    let old: Vec<Vec<f64>> = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "vwap",
    ]
    .iter()
    .map(|n| column(&src, n))
    .collect();
    enrich::build(&opts(&src, &out, "taker_ratio96")).unwrap();
    let after = FileStore::open(&out).unwrap();
    assert_eq!(&after.header().names()[..old_names.len()], &old_names[..]);
    assert_eq!(&after.header().dtypes()[..old_dtypes.len()], &old_dtypes[..]);
    for (j, name) in [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trades",
        "taker_buy_base",
        "taker_buy_quote",
        "vwap",
    ]
    .iter()
    .enumerate()
    {
        let got = column(&out, name);
        assert_eq!(got.len(), old[j].len());
        for (i, (a, b)) in got.iter().zip(&old[j]).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "{name} changed at row {i}");
        }
    }
}

/// The flow family fails closed when a raw field is missing, naming both the
/// feature and the column; an unknown name is still a hard error.
#[test]
fn flow_family_requires_raw_fields_and_unknown_names_fail() {
    let dir = tmp_dir("flow_missing");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &bars("AAA", 120, 0, 900), 16);
    let cases = [
        ("vwap", "quote_volume"),
        ("trade_intensity96", "trades"),
        ("taker_ratio96", "taker_buy_base"),
        ("avg_trade_size96", "quote_volume"),
        ("amihud96", "quote_volume"),
        ("cvd_proxy", "taker_buy_base"),
    ];
    for (feature, column) in cases {
        let err = enrich::build(&opts(&src, &out, feature)).unwrap_err();
        assert!(err.contains(feature), "error for {feature} does not name the feature: {err}");
        assert!(err.contains(column), "error for {feature} does not name '{column}': {err}");
    }
    assert!(enrich::expand("flow_not_a_feature").is_err());
}

/// Parity against the independent Python compact builder on a contiguous 15 m
/// slice. The expected file holds the compact values for the last 300 hourly
/// rows of two symbols (AAVEUSDT and BTCUSDT). Rows before index 3000 are
/// pre-roll, so every 672/2880-bar window lies inside the fixture.
#[test]
fn compact_matches_the_python_builder_on_a_contiguous_slice() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/enrich");
    let input = base.join("compact_15m_ohlcv.csv");
    let expected_path = base.join("compact_15m_expected.csv");
    let dir = tmp_dir("compact_parity");
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let report = enrich::build(&opts(&input.to_string_lossy(), &out, "compact")).unwrap();
    assert_eq!(report.features.len(), 52);

    let text = std::fs::read_to_string(&expected_path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let start_ts = 1_680_271_200i64;
    let rows_per_symbol = 3600usize;
    let mut worst_abs = 0.0f64;
    let mut worst_rel = 0.0f64;
    let mut worst_name = "";
    let mut compared = 0usize;
    let out_cols: Vec<Vec<f64>> = header[2..].iter().map(|n| column(&out, n)).collect();
    let out_ts: Vec<i64> = column(&out, "ts").into_iter().map(|v| v as i64).collect();
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let ts: i64 = fields[0].parse().unwrap();
        let sym = fields[1];
        let seg = if sym == "AAVEUSDT" { 0 } else { 1 };
        let idx = seg * rows_per_symbol + ((ts - start_ts) / 900) as usize;
        assert_eq!(out_ts[idx], ts, "row alignment for {sym} at {ts}");
        for (j, name) in header.iter().enumerate().skip(2) {
            let expected = fields[j].parse::<f64>().unwrap();
            let got = out_cols[j - 2][idx];
            match (got.is_nan(), expected.is_nan()) {
                (true, true) => continue,
                (true, false) => panic!("{name} at {sym}/{ts}: got NaN, expected {expected}"),
                (false, true) => panic!("{name} at {sym}/{ts}: got {got}, expected NaN"),
                (false, false) => {}
            }
            let a = (got - expected).abs();
            let r = a / expected.abs().max(1e-300);
            if a > worst_abs {
                worst_abs = a;
                worst_name = name;
            }
            worst_rel = worst_rel.max(r);
            assert!(
                a <= 1e-10 || r <= 1e-10,
                "{name} at {sym}/{ts}: got {got}, expected {expected}, abs {a:e} rel {r:e}"
            );
            compared += 1;
        }
    }
    assert!(compared > 10_000, "parity compared too few values ({compared})");
    eprintln!(
        "compact parity: {compared} finite values, worst abs {worst_abs:e} ({worst_name}), worst rel {worst_rel:e}"
    );
}

#[test]
fn every_catalogue_column_is_causal_on_a_prefix() {
    let dir = tmp_dir("causal");
    let full = dir.join("full.rcsb").to_string_lossy().into_owned();
    let prefix = dir.join("prefix.rcsb").to_string_lossy().into_owned();
    let full_out = dir.join("full_out.rcsb").to_string_lossy().into_owned();
    let prefix_out = dir.join("prefix_out.rcsb").to_string_lossy().into_owned();
    let rows = bars("BTCUSDT", 320, 0, 3600);
    write_store(&full, &rows, 33);
    write_store(&prefix, &rows[..200], 33);
    enrich::build(&opts(&full, &full_out, "all")).unwrap();
    enrich::build(&opts(&prefix, &prefix_out, "all")).unwrap();
    let names = FileStore::open(&full_out).unwrap().header().names();
    for name in names.iter().skip(8) {
        let a = column(&full_out, name);
        let b = column(&prefix_out, name);
        for i in 0..200 {
            assert_eq!(a[i].is_nan(), b[i].is_nan(), "{name} NaN pattern differs at {i}");
            if a[i].is_finite() {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-12,
                    "{name} at {i}: full {} prefix {}",
                    a[i],
                    b[i]
                );
            }
        }
    }
}

#[test]
fn replace_overwrites_the_existing_values() {
    let dir = tmp_dir("replace_values");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    // A store that already carries a deliberately wrong r1 column.
    let rows = bars("AAA", 120, 0, 3600);
    let refs: Vec<(i64, &str, f64, f64, f64, f64, f64)> = rows
        .iter()
        .map(|(t, s, o, h, l, c, v)| (*t, s.as_str(), *o, *h, *l, *c, *v))
        .collect();
    let mut text = csv_of(&refs);
    text = text.replace(",keep\n", ",keep,r1\n");
    let mut csv2 = String::new();
    for (i, line) in text.lines().enumerate() {
        if i == 0 {
            csv2.push_str(line);
            csv2.push('\n');
        } else {
            csv2.push_str(line);
            csv2.push_str(",0\n");
        }
    }
    let t = data::parse_table_with_time(&csv2, false, Some("ts")).unwrap();
    store::write_store_with_time(&src, &t, 16, Some(0)).unwrap();
    assert!(column(&src, "r1").iter().all(|v| *v == 0.0));
    let mut o = opts(&src, &out, "r1");
    o.replace = true;
    let r = enrich::build(&o).unwrap();
    assert_eq!(r.columns_in, r.columns_out);
    let r1 = column(&out, "r1");
    assert!(r1[1].is_finite() && r1[1].abs() > 0.0, "r1 was not recomputed");
    assert_eq!(r1[0].is_nan(), true);
}

#[test]
fn time_major_panel_stores_are_refused_with_a_clear_message() {
    let dir = tmp_dir("timemajor");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    // Interleave two symbols by timestamp.
    let mut rows = Vec::new();
    for t in 0..20i64 {
        for sym in ["AAA", "BBB"] {
            let p = 100.0 + t as f64 + if sym == "AAA" { 0.0 } else { 10.0 };
            rows.push((t * 3600, sym.to_string(), p, p + 0.2, p - 0.2, p + 0.1, 1.0));
        }
    }
    write_store(&src, &rows, 4);
    let err = enrich::build(&opts(&src, &out, "r1")).unwrap_err();
    assert!(err.contains("time-major"), "unexpected error: {err}");
    assert!(!std::path::Path::new(&out).exists());
}

/// Every advertised name must appear in a real output store, with a per-column
/// hash that reproduces the bytes actually written, not only in the dry-run
/// catalogue. `all` has 171 columns, `py15m` 35, and their union 201 (five
/// names are shared).
#[test]
fn advertised_column_set_is_exactly_what_a_real_run_appends() {
    let dir = tmp_dir("advertised");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    // A two-sided series: zero-loss windows make RSI undefined (NaN) by the
    // Python definition, so a monotonic ramp cannot exercise `rsi6`.
    let wavy = |symbol: &str, phase: f64| -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
        (0..300usize)
            .map(|i| {
                let t = i as f64;
                let p = 100.0 + 5.0 * (t * 0.37 + phase).sin() + 0.02 * t;
                (
                    i as i64 * 3600,
                    symbol.to_string(),
                    p,
                    p + 1.0,
                    p - 1.0,
                    p + 0.5 * (t * 0.7).sin(),
                    10.0 + (i % 7) as f64,
                )
            })
            .collect()
    };
    let mut rows = wavy("AAA", 0.0);
    rows.extend(wavy("BBB", 1.3));
    rows.extend(wavy("BTCUSDT", 2.1));
    write_store_with_flow(&src, &rows, 64);
    let input_names = FileStore::open(&src).unwrap().header().names();

    for (spec, expected) in [("all", 171usize), ("py15m", 35), ("all,py15m", 201)] {
        let advertised = enrich::expand(spec).unwrap();
        assert_eq!(advertised.len(), expected, "{spec} catalogue size");
        let out = dir.join(format!("{}.rcsb", spec.replace(',', "-"))).to_string_lossy().into_owned();
        let dry_out = format!("{}.dry", out);

        let mut dry = opts(&src, &dry_out, spec);
        dry.dry_run = true;
        let dry_report = enrich::build(&dry).unwrap();
        let real = enrich::build(&opts(&src, &out, spec)).unwrap();

        // The dry run predicts exactly the schema the real run writes.
        assert_eq!(
            dry_report.columns_out,
            input_names.len() + advertised.len(),
            "{spec}: dry-run schema"
        );
        assert_eq!(dry_report.columns_out, real.columns_out, "{spec}: dry-run vs real schema");

        let store = FileStore::open(&out).unwrap();
        let names = store.header().names();
        assert_eq!(&names[..input_names.len()], &input_names[..], "{spec}: input order changed");
        let appended = &names[input_names.len()..];
        assert_eq!(appended.len(), advertised.len(), "{spec}: appended column count");
        for (i, name) in advertised.iter().enumerate() {
            assert_eq!(&appended[i], name, "{spec}: column {} is out of catalogue order", i);
            assert_eq!(store.header().column_index(name), Some(input_names.len() + i));
        }

        // Report order and hashes follow the file, and each hash is the digest
        // of that column's little-endian f64 bytes in the output store.
        assert_eq!(real.features.len(), advertised.len());
        for (f, name) in real.features.iter().zip(advertised.iter()) {
            assert_eq!(&f.name, name);
            let values = column(&out, name);
            let mut bytes = Vec::with_capacity(values.len() * 8);
            for v in &values {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            assert_eq!(
                f.hash,
                rcsr::manifest::sha256_hex(&bytes),
                "{spec}: reported hash for '{name}' is not the output column digest"
            );
            // Cross-sectional ranks deliberately require 20 PIT members; this
            // three-symbol smoke fixture cannot produce one. Their dedicated
            // 20-member tests below cover the finite path.
            if name.starts_with("xs_rank_") || matches!(name.as_str(), "mkt_r16" | "mkt_r96" | "mkt_vol96") {
                assert_eq!(f.nan_rows, real.rows, "{spec}: '{name}' must need 20 members");
            } else {
                assert!(
                    f.nan_rows < real.rows,
                    "{spec}: '{name}' is NaN on every row of the fixture"
                );
            }
        }
        assert_eq!(
            real.output_sha256,
            rcsr::manifest::sha256_file(&out).unwrap(),
            "{spec}: streamed output checksum"
        );
    }
}

// ---------------------------------------------------------------------------
// Issue #273: complete cross-asset, breadth and cross-sectional family
// ---------------------------------------------------------------------------

fn closeish(expected: f64, got: f64, tol: f64, what: &str) {
    assert!(
        (expected - got).abs() <= tol,
        "{what}: expected {expected}, got {got}, diff {}",
        (expected - got).abs()
    );
}

fn bench_price(symbol: &str, i: usize) -> f64 {
    let t = i as f64;
    match symbol {
        "BTCUSDT" => 100.0 * (0.0007 * t + 0.010 * (t * 0.21).sin()).exp(),
        "ETHUSDT" => 80.0 * (0.0005 * t + 0.020 * (t * 0.17).cos()).exp(),
        _ => 50.0 * (0.0010 * t + 0.030 * (t * 0.40).sin()).exp(),
    }
}

fn bench_rows(shift_btc: i64, btc_hole: bool) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for sym in ["BTCUSDT", "ETHUSDT", "AAA"] {
        for i in 0..200usize {
            if sym == "BTCUSDT" && btc_hole && i == 100 {
                continue;
            }
            let p = bench_price(sym, i);
            let shift = if sym == "BTCUSDT" { shift_btc } else { 0 };
            rows.push((
                i as i64 * 900 + shift,
                sym.to_string(),
                p * 0.999,
                p * 1.002,
                p * 0.998,
                p,
                100.0 + i as f64 + sym.len() as f64,
            ));
        }
    }
    rows
}

#[test]
fn btc_and_eth_joins_use_the_same_timestamp_and_a_shift_fails() {
    let dir = tmp_dir("cross-bench");
    let input = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let features = "btc_r1,btc_r16,btc_r20,btc_vol20,rel_btc_r16,beta_btc_60,eth_r16,rel_eth_r16,beta_eth_60";
    write_store(&input, &bench_rows(0, false), 64);
    enrich::build(&opts(&input, &out, features)).unwrap();

    let n = 200usize;
    let aaa = 2 * n + (n - 1);
    let btc16 = (bench_price("BTCUSDT", n - 1) / bench_price("BTCUSDT", n - 17)).ln();
    closeish(btc16, column(&out, "btc_r16")[aaa], 1e-12, "btc_r16");
    let eth16 = (bench_price("ETHUSDT", n - 1) / bench_price("ETHUSDT", n - 17)).ln();
    closeish(eth16, column(&out, "eth_r16")[aaa], 1e-12, "eth_r16");
    let own16 = (bench_price("AAA", n - 1) / bench_price("AAA", n - 17)).ln();
    closeish(own16 - btc16, column(&out, "rel_btc_r16")[aaa], 1e-12, "rel_btc_r16");
    closeish(own16 - eth16, column(&out, "rel_eth_r16")[aaa], 1e-12, "rel_eth_r16");
    let beta = column(&out, "beta_btc_60")[aaa];
    assert!(beta.is_finite() && beta >= -5.0 && beta <= 5.0, "beta {beta}");
    assert!(column(&out, "beta_eth_60")[aaa].is_finite());

    // A missing benchmark bucket is NaN, never 0 or a forward fill.
    let hole_in = dir.join("hole.rcsb").to_string_lossy().into_owned();
    let hole_out = dir.join("hole_out.rcsb").to_string_lossy().into_owned();
    write_store(&hole_in, &bench_rows(0, true), 64);
    enrich::build(&opts(&hole_in, &hole_out, "btc_r1,btc_r16")).unwrap();
    let btc_rows = n - 1;
    let aaa_hole_start = btc_rows + n;
    let r1 = column(&hole_out, "btc_r1");
    let r16 = column(&hole_out, "btc_r16");
    assert!(r1[aaa_hole_start + 100].is_nan(), "missing BTC bucket must be NaN");
    assert!(r16[aaa_hole_start + 116].is_nan(), "a window spanning the hole must be NaN");

    // One-bucket shift: the exact-timestamp join must change the value.
    let shift_in = dir.join("shift.rcsb").to_string_lossy().into_owned();
    let shift_out = dir.join("shift_out.rcsb").to_string_lossy().into_owned();
    write_store(&shift_in, &bench_rows(900, false), 64);
    enrich::build(&opts(&shift_in, &shift_out, "btc_r1,btc_r16")).unwrap();
    let shifted = column(&shift_out, "btc_r1")[aaa];
    let expected_shift = (bench_price("BTCUSDT", n - 2) / bench_price("BTCUSDT", n - 3)).ln();
    closeish(expected_shift, shifted, 1e-12, "one-bucket shift");
    assert!(
        (shifted - column(&out, "btc_r1")[aaa]).abs() > 1e-9,
        "a one-bucket shift must fail the benchmark parity"
    );

    // A requested ETH column without an ETH series fails closed, naming ETH.
    let no_eth = dir.join("no_eth.rcsb").to_string_lossy().into_owned();
    let no_eth_rows: Vec<_> = bench_rows(0, false)
        .into_iter()
        .filter(|r| r.1 != "ETHUSDT")
        .collect();
    write_store(&no_eth, &no_eth_rows, 64);
    let err = enrich::build(&opts(&no_eth, &out, "eth_r16")).unwrap_err();
    assert!(err.contains("ETHUSDT"), "unexpected error: {err}");
}

fn panel_close(symbol: &str, i: usize) -> f64 {
    let k = if symbol == "BTCUSDT" {
        0
    } else {
        symbol.trim_start_matches("ALT").trim_end_matches("USDT").parse::<usize>().unwrap()
    };
    let mu = 0.0002 * (k as f64 - 10.0);
    let phase = k as f64 * 0.41;
    100.0 * (mu * i as f64 + 0.03 * (i as f64 * 0.23 + phase).sin()).exp()
}

fn panel_ret(symbol: &str, i: usize, k: usize) -> f64 {
    (panel_close(symbol, i) / panel_close(symbol, i - k)).ln()
}

fn panel_vol96(symbol: &str, i: usize) -> f64 {
    let mut vals = Vec::new();
    for j in i + 1 - 96..=i {
        vals.push((panel_close(symbol, j) / panel_close(symbol, j - 1)).ln());
    }
    let mean = vals.iter().sum::<f64>() / vals.len() as f64;
    (vals.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / vals.len() as f64).sqrt()
}

fn symbol_index(k: usize, i: usize, n: usize) -> usize {
    k * n + i
}

#[test]
fn cross_family_is_exactly_the_registry_and_market_state_is_ex_self_pit() {
    let dir = tmp_dir("cross-market");
    let input = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let n = 800usize;
    let rows = cross_panel_rows(n, 900, 0.0, Some("ALT01USDT"), 1.0e12);
    write_panel_store(&input, &rows, 64);
    let report = enrich::build(&cross_opts(&input, &out, "cross")).unwrap();
    let expected = enrich::cross_names();
    assert_eq!(report.features.len(), expected.len());
    let store = FileStore::open(&out).unwrap();
    let names = store.header().names();
    let start = names.len() - expected.len();
    assert_eq!(&names[start..], &expected[..], "cross emission order");
    assert!(rcsr::schema::json().contains("\"family\":\"cross\""));
    for name in &expected {
        let vals = column(&out, name);
        assert!(
            vals.iter().any(|v| v.is_finite()),
            "cross column '{name}' is NaN on every row of the 21-symbol fixture"
        );
    }

    let i = n - 1;
    let valid: Vec<usize> = (0..21).filter(|k| *k != 1).collect();
    let r16: Vec<f64> = valid.iter().map(|k| panel_ret(&format!("ALT{:02}USDT", *k), i, 16)).collect();
    let mean = r16.iter().sum::<f64>() / r16.len() as f64;
    let breadth_expected = r16.iter().filter(|v| **v > 0.0).count() as f64 / r16.len() as f64;
    let dispersion_expected =
        (r16.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / r16.len() as f64).sqrt();

    let mkt = column(&out, "mkt_r16");
    let rel = column(&out, "rel_mkt16");
    let breadth = column(&out, "breadth");
    let dispersion = column(&out, "dispersion");
    let rank = column(&out, "xs_rank_r16");
    let mkt1 = column(&out, "mkt_ew_ret1");
    let corr = column(&out, "avg_corr");
    let share = column(&out, "btc_volume_share");

    let mut total_volume = 0.0;
    for k in &valid {
        total_volume += 1000.0 + 10.0 * *k as f64 + (i % 7) as f64;
    }
    let btc_volume = 1000.0 + (i % 7) as f64;
    let r1: Vec<f64> = valid.iter().map(|k| panel_ret(&format!("ALT{:02}USDT", *k), i, 1)).collect();
    let mean1 = r1.iter().sum::<f64>() / r1.len() as f64;
    let vol: Vec<f64> = valid.iter().map(|k| panel_vol96(&format!("ALT{:02}USDT", *k), i)).collect();
    let mean_vol = vol.iter().sum::<f64>() / vol.len() as f64;

    for (j, k) in valid.iter().enumerate() {
        let idx = symbol_index(*k, i, n);
        let sym = if *k == 0 { "BTCUSDT".to_string() } else { format!("ALT{:02}USDT", *k) };
        closeish(mean, mkt[idx], 1e-12, "mkt_r16");
        closeish(mean, column(&out, "mkt_ew_ret16")[idx], 1e-12, "mkt_ew_ret16 alias");
        closeish(breadth_expected, breadth[idx], 1e-12, "breadth");
        closeish(dispersion_expected, dispersion[idx], 1e-12, "dispersion");
        closeish(mean1, mkt1[idx], 1e-12, "mkt_ew_ret1");
        closeish(mean_vol, column(&out, "mkt_vol96")[idx], 1e-12, "mkt_vol96");
        closeish(btc_volume / total_volume, share[idx], 1e-12, "btc_volume_share");
        let ex_self = (r16.iter().sum::<f64>() - r16[j]) / (r16.len() - 1) as f64;
        closeish(r16[j] - ex_self, rel[idx], 1e-12, "rel_mkt16 ex-self");
        assert!(corr[idx].is_finite() && (-1.0..=1.0).contains(&corr[idx]), "avg_corr {}", corr[idx]);
        assert!((-1.0..=1.0).contains(&rank[idx]), "rank {}", rank[idx]);
        assert!(column(&out, "xs_rank_r1")[idx].is_finite());
        assert!(column(&out, "xs_rank_r20")[idx].is_finite());
        assert!(column(&out, "xs_rank_vol20")[idx].is_finite());
        assert!(column(&out, "xs_rank_amihud96")[idx].is_finite());
        assert!(column(&out, "cs_rank_r60")[idx].is_finite());
        assert!(column(&out, "rel_mkt96")[idx].is_finite());
        let _ = sym;
    }

    // The pre-listing symbol does not enter the market statistics, ranks or
    // the relative columns; its cross-sectional outputs are NaN.
    let pre = symbol_index(1, i, n);
    for name in [
        "mkt_r16",
        "breadth",
        "dispersion",
        "rel_mkt16",
        "xs_rank_r16",
        "xs_rank_r1",
        "avg_corr",
        "btc_volume_share",
        "mkt_vol96",
    ] {
        assert!(column(&out, name)[pre].is_nan(), "{name} used a pre-listing row");
    }

    // The rank is exactly the sorted position of r16, scaled to [-1, 1].
    let mut sorted: Vec<(f64, usize)> = valid
        .iter()
        .enumerate()
        .map(|(j, k)| (r16[j], *k))
        .collect();
    sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap().then(a.1.cmp(&b.1)));
    for (pos, (_, k)) in sorted.iter().enumerate() {
        let expected_rank = 2.0 * pos as f64 / (sorted.len() - 1) as f64 - 1.0;
        closeish(expected_rank, rank[symbol_index(*k, i, n)], 1e-12, "xs_rank_r16");
    }
}

#[test]
fn a_partial_cross_section_is_nan_and_a_prefix_is_causal() {
    let dir = tmp_dir("cross-partial");
    let input = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let n = 240usize;
    let mut rows = cross_panel_rows(n, 900, 0.0, None, 0.0);
    // Drop the last row of two symbols: only 19 members remain at t_last.
    for k in [3usize, 2] {
        let idx = symbol_index(k, n - 1, n);
        rows.remove(idx);
    }
    write_panel_store(&input, &rows, 64);
    let features = "mkt_r16,breadth,dispersion,rel_mkt16,avg_corr,xs_rank_r16";
    enrich::build(&cross_opts(&input, &out, features)).unwrap();
    let last = rows.len() - 1;
    for name in ["mkt_r16", "breadth", "dispersion", "rel_mkt16", "avg_corr", "xs_rank_r16"] {
        assert!(column(&out, name)[last].is_nan(), "{name} must be NaN at N=19");
    }

    // Prefix causality: every earlier value is unchanged by truncating later
    // rows, for every cross-family column.
    let full_in = dir.join("full.rcsb").to_string_lossy().into_owned();
    let full_out = dir.join("full_out.rcsb").to_string_lossy().into_owned();
    let pre_in = dir.join("pre.rcsb").to_string_lossy().into_owned();
    let pre_out = dir.join("pre_out.rcsb").to_string_lossy().into_owned();
    let full_rows = cross_panel_rows(120, 900, 0.0, None, 0.0);
    let prefix = 80usize;
    let mut pre_rows = Vec::new();
    for k in 0..21usize {
        pre_rows.extend_from_slice(&full_rows[symbol_index(k, 0, 120)..symbol_index(k, prefix, 120)]);
    }
    write_panel_store(&full_in, &full_rows, 32);
    write_panel_store(&pre_in, &pre_rows, 32);
    let names = enrich::cross_names();
    enrich::build(&cross_opts(&full_in, &full_out, "cross")).unwrap();
    enrich::build(&cross_opts(&pre_in, &pre_out, "cross")).unwrap();
    for name in &names {
        let a = column(&full_out, name);
        let b = column(&pre_out, name);
        for k in 0..21usize {
            for i in 0..prefix {
                let x = a[symbol_index(k, i, 120)];
                let y = b[symbol_index(k, i, prefix)];
                assert_eq!(x.is_nan(), y.is_nan(), "{name} NaN mask at k={k} i={i}");
                if x.is_finite() {
                    closeish(x, y, 1e-12, &format!("{name} causal k={k} i={i}"));
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-timeframe family (#272)
// ---------------------------------------------------------------------------

/// Build one base bar per higher bucket, so resampling is an identity and the
/// higher OHLC used by the feature test is exactly the fixture OHLC.
fn mtf_single_bar_fixture(step: i64, n: usize) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|k| {
            let o = 100.0 + k as f64 * 0.1 + (k as f64 * 0.7).sin() * 2.0;
            let c = o + (k as f64 * 0.31).sin() * 1.5;
            let h = o.max(c) + 0.6 + (k % 3) as f64 * 0.1;
            let l = o.min(c) - 0.5 - (k % 2) as f64 * 0.1;
            (k as i64 * step, "AAA".to_string(), o, h, l, c, 10.0 + k as f64)
        })
        .collect()
}

fn higher_arrays(rows: &[(i64, String, f64, f64, f64, f64, f64)]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    (
        rows.iter().map(|r| r.2).collect(),
        rows.iter().map(|r| r.3).collect(),
        rows.iter().map(|r| r.4).collect(),
        rows.iter().map(|r| r.5).collect(),
    )
}

fn mean(x: &[f64], lo: usize, hi: usize) -> f64 {
    x[lo..=hi].iter().sum::<f64>() / (hi - lo + 1) as f64
}

fn expected_leaf(
    leaf: &str,
    _o: &[f64],
    h: &[f64],
    l: &[f64],
    c: &[f64],
    i: usize,
) -> f64 {
    let ret = |k: usize| c[i] / c[i - k] - 1.0;
    match leaf {
        "ret1" => ret(1),
        "ret4" => ret(4),
        "ret24" => ret(24),
        "ma_ratio" => c[i] / mean(c, i - 19, i) - 1.0,
        "vol" => {
            let r: Vec<f64> = (i - 19..=i).map(|j| c[j] / c[j - 1] - 1.0).collect();
            let m = r.iter().sum::<f64>() / 20.0;
            (r.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / 20.0).sqrt()
        }
        "atr_pct" => {
            let mut tr = Vec::with_capacity(i + 1);
            for j in 0..=i {
                tr.push(if j == 0 {
                    h[j] - l[j]
                } else {
                    (h[j] - l[j])
                        .max((h[j] - c[j - 1]).abs())
                        .max((l[j] - c[j - 1]).abs())
                });
            }
            let seed = tr[..14].iter().sum::<f64>() / 14.0;
            let mut atr = seed;
            for x in tr.iter().skip(14) {
                atr += (x - atr) / 14.0;
            }
            atr / c[i]
        }
        "rsi" => {
            let mut gain = 0.0;
            let mut loss = 0.0;
            for j in 1..=14 {
                let d = c[j] - c[j - 1];
                gain += d.max(0.0);
                loss += (-d).max(0.0);
            }
            let (mut ag, mut al) = (gain / 14.0, loss / 14.0);
            for j in 15..=i {
                let d = c[j] - c[j - 1];
                ag = (ag * 13.0 + d.max(0.0)) / 14.0;
                al = (al * 13.0 + (-d).max(0.0)) / 14.0;
            }
            let rsi = if al > 0.0 { 100.0 - 100.0 / (1.0 + ag / al) } else { 100.0 };
            rsi / 50.0 - 1.0
        }
        "close_pos" => (c[i] - l[i]) / (h[i] - l[i]) - 0.5,
        "donchian_pos" => {
            let lo = l[i - 19..=i].iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let hi = h[i - 19..=i].iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (c[i] - lo) / (hi - lo) - 0.5
        }
        "range_pct" => (h[i] - l[i]) / c[i],
        "trend_slope" => mean(c, i - 19, i) / mean(c, i - 39, i - 20) - 1.0,
        other => panic!("no expected value for {other}"),
    }
}

#[test]
fn mtf_36_columns_match_hand_built_higher_bars() {
    // One base bar per higher bucket: at base row i the last closed bucket is
    // i-1, so the joined value is the feature evaluated at higher index i-1.
    for (tf_name, step) in [("1h", 3600i64), ("4h", 14_400), ("1d", 86_400)] {
        let rows = mtf_single_bar_fixture(step, 60);
        let (o, h, l, c) = higher_arrays(&rows);
        let dir = tmp_dir(&format!("mtf-unit-{tf_name}"));
        let src = dir.join("in.rcsb").to_string_lossy().into_owned();
        let out = dir.join("out.rcsb").to_string_lossy().into_owned();
        write_store(&src, &rows, 32);
        let report = enrich::build(&opts(&src, &out, "mtf")).unwrap();
        assert_eq!(report.features.len(), 36, "{tf_name}: 36 MTF columns");

        let base_i = 50usize;
        let higher_i = base_i - 1;
        for leaf in enrich::MTF_LEAVES {
            let name = format!("mtf_{tf_name}_{leaf}");
            let got = column(&out, &name);
            assert_eq!(got.len(), rows.len());
            // Warm-up rows are NaN.
            let warm = enrich::mtf_warmup(leaf);
            for j in 0..warm.min(base_i) {
                assert!(got[j].is_nan(), "{name}: row {j} must be warm-up NaN");
            }
            if *leaf == "bar_age" {
                // Base and higher bucket share a step: the bucket before the
                // decision closed exactly at the decision instant, age zero.
                assert_eq!(got[base_i], 0.0, "{name} at {tf_name}");
            } else {
                let want = expected_leaf(leaf, &o, &h, &l, &c, higher_i);
                assert!(
                    (got[base_i] - want).abs() <= 1e-12,
                    "{name} at row {base_i}: got {}, want {want}",
                    got[base_i]
                );
            }
        }
    }
}

#[test]
fn mtf_gap_is_nan_and_bar_age_reports_staleness() {
    let step = 900i64;
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for hour in 0..12i64 {
        if hour == 5 {
            continue; // an entire higher bucket is missing
        }
        for q in 0..4i64 {
            let t = hour * 3600 + q * step;
            let p = 100.0 + hour as f64 + q as f64 * 0.1;
            rows.push((t, "AAA".to_string(), p, p + 0.5, p - 0.4, p + 0.05, 10.0));
        }
    }
    let dir = tmp_dir("mtf-gap");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 32);
    enrich::build(&opts(&src, &out, "mtf_1h_ret1,mtf_1h_ret4,mtf_1h_close_pos,mtf_1h_bar_age")).unwrap();

    let times: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let idx = |t: i64| times.iter().position(|x| *x == t).unwrap();
    let ret1 = column(&out, "mtf_1h_ret1");
    let ret4 = column(&out, "mtf_1h_ret4");
    let pos = column(&out, "mtf_1h_close_pos");
    let age = column(&out, "mtf_1h_bar_age");

    // The first decision after the missing hour expects bucket 05 but the last
    // closed bucket is 04: value features must not bridge the gap.
    let i = idx(6 * 3600 + step);
    assert!(ret1[i].is_nan());
    assert!(ret4[i].is_nan());
    assert!(pos[i].is_nan());
    // Age is exact: bucket 04 closed at 05:00, the decision is 06:15.
    assert_eq!(age[i], (6 * 3600 + 900 - 5 * 3600) as f64);

    // Bucket 06 exists, so its bar-position features are valid, but the ret4
    // window (03..06) crosses the missing bucket 05 and stays NaN.
    let j = idx(7 * 3600);
    assert!(pos[j].is_finite());
    assert!(ret4[j].is_nan());
    assert_eq!(age[j], 0.0);

    // Once the window no longer spans the gap, ret4 becomes valid again.
    let k = idx(11 * 3600);
    assert!(ret4[k].is_finite(), "ret4 must recover after the gap leaves the window");
}

fn mtf_15m_rows(hours: usize) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for hour in 0..hours {
        for q in 0..4i64 {
            let t = hour as i64 * 3600 + q * 900;
            let p = 100.0 + hour as f64 * 0.4 + q as f64 * 0.15;
            rows.push((t, "AAA".to_string(), p, p + 0.4, p - 0.3, p + 0.05, 10.0 + q as f64));
        }
    }
    rows
}

#[test]
fn mtf_join_is_causal_and_a_future_shift_is_caught() {
    let rows = mtf_15m_rows(48);
    let bars: Vec<Bar> = rows
        .iter()
        .map(|r| Bar {
            time: r.0,
            open: r.2,
            high: r.3,
            low: r.4,
            close: r.5,
            volume: r.6,
            quote_volume: 0.0,
            trades: 0.0,
            taker_buy_base: 0.0,
            taker_buy_quote: 0.0,
            complete: true,
            traded: true,
        })
        .collect();
    let higher = ohlcv::resample(&bars, Timeframe::H1).unwrap();
    let joined = ohlcv::join_causal(&bars, &higher, Timeframe::H1).unwrap();
    assert!(ohlcv::lookahead_violations(&bars, &joined, Timeframe::H1).is_empty());

    // Moving the visible value one bucket earlier is a look-ahead leak: the
    // base row now sees a bar that has not closed yet.
    let mut shifted = joined.clone();
    for i in 0..shifted.len().saturating_sub(1) {
        shifted[i] = joined[i + 1];
    }
    assert!(
        !ohlcv::lookahead_violations(&bars, &shifted, Timeframe::H1).is_empty(),
        "a future-shifted join must be detected"
    );

    // The enrich column at a 10:15 decision uses the 09:00 hourly bar (closed
    // at 10:00), never the in-progress 10:00 bar.
    let dir = tmp_dir("mtf-causal");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 32);
    enrich::build(&opts(&src, &out, "mtf_1h_ret1,mtf_1h_bar_age")).unwrap();
    let ret1 = column(&out, "mtf_1h_ret1");
    let age = column(&out, "mtf_1h_bar_age");
    let i = rows.iter().position(|r| r.0 == 10 * 3600 + 900).unwrap();
    let want = higher[9].close / higher[8].close - 1.0; // 09:00 over 08:00
    assert!((ret1[i] - want).abs() <= 1e-12);
    assert_eq!(age[i], 900.0);
    let current = higher[10].close / higher[9].close - 1.0; // 10:00 bar, not closed
    assert!((ret1[i] - current).abs() > 1e-6);
}

#[test]
fn mtf_prefix_truncation_replay_changes_no_earlier_value() {
    let rows = mtf_15m_rows(96);
    let prefix_len = 200usize;
    let prefix_rows = &rows[..prefix_len];
    let dir = tmp_dir("mtf-prefix");
    let full_src = dir.join("full.rcsb").to_string_lossy().into_owned();
    let full_out = dir.join("full-out.rcsb").to_string_lossy().into_owned();
    let pre_src = dir.join("pre.rcsb").to_string_lossy().into_owned();
    let pre_out = dir.join("pre-out.rcsb").to_string_lossy().into_owned();
    write_store(&full_src, &rows, 32);
    write_store(&pre_src, prefix_rows, 32);
    enrich::build(&opts(&full_src, &full_out, "mtf")).unwrap();
    enrich::build(&opts(&pre_src, &pre_out, "mtf")).unwrap();

    for name in enrich::mtf_names() {
        let full = column(&full_out, &name);
        let pre = column(&pre_out, &name);
        assert_eq!(pre.len(), prefix_len);
        for i in 0..prefix_len {
            assert_eq!(
                full[i].to_bits(),
                pre[i].to_bits(),
                "{name} changed at row {i} under a prefix replay"
            );
        }
    }
}

#[test]
fn mtf_registry_and_schema_expose_all_36_inventory_rows() {
    let names = enrich::mtf_names();
    assert_eq!(names.len(), 36);
    for tf in enrich::MTF_TIMEFRAMES {
        assert_eq!(
            names.iter().filter(|n| n.starts_with(&format!("mtf_{tf}_"))).count(),
            12,
            "{tf} has twelve leaves"
        );
    }
    let schema = rcsr::schema::json();
    for name in &names {
        assert!(schema.contains(&format!("\"name\":\"{name}\"")), "schema misses {name}");
    }

    // The checked-in inventory JSON is the reviewable 36-row evidence artifact.
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../docs/mtf-inventory.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(text.matches("\"name\"").count(), 36, "inventory must have 36 rows");
    for name in &names {
        assert!(text.contains(&format!("\"name\": \"{name}\"")), "inventory misses {name}");
    }
}

#[test]
fn mtf_every_column_is_nan_when_its_higher_bucket_is_missing() {
    // Drop one complete higher bucket of each timeframe from a 15m base. The
    // first decision after the hole expects that bucket, so every value leaf
    // must be NaN; bar_age remains exact staleness.
    let cases = [
        ("1h", 31usize, 32usize, 32i64 * 3600),
        ("4h", 28usize, 32usize, 32i64 * 3600),
        ("1d", 48usize, 72usize, 72i64 * 3600),
    ];
    for (tf, drop_lo, drop_hi, decision) in cases {
        let mut rows = Vec::new();
        for hour in 0..144i64 {
            if (hour as usize) >= drop_lo && (hour as usize) < drop_hi {
                continue;
            }
            for q in 0..4i64 {
                let t = hour * 3600 + q * 900;
                let p = 100.0 + hour as f64 * 0.05 + q as f64 * 0.01;
                rows.push((t, "AAA".to_string(), p, p + 0.4, p - 0.3, p + 0.02, 10.0));
            }
        }
        let dir = tmp_dir(&format!("mtf-missing-{tf}"));
        let src = dir.join("in.rcsb").to_string_lossy().into_owned();
        let out = dir.join("out.rcsb").to_string_lossy().into_owned();
        write_store(&src, &rows, 32);
        enrich::build(&opts(&src, &out, "mtf")).unwrap();
        let times: Vec<i64> = rows.iter().map(|r| r.0).collect();
        let i = times.iter().position(|t| *t == decision).unwrap();
        for leaf in enrich::MTF_LEAVES {
            let name = format!("mtf_{tf}_{leaf}");
            let got = column(&out, &name);
            if *leaf == "bar_age" {
                // The last closed bucket ended exactly at the dropped bucket's
                // close, so staleness spans the missing bucket(s).
                let expected = (decision - (drop_lo as i64 * 3600)) as f64;
                assert_eq!(got[i], expected, "{name} bar_age after a gap");
            } else {
                assert!(got[i].is_nan(), "{name} must be NaN when its bucket is missing");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Candle and price-action family (#261)
// ---------------------------------------------------------------------------

/// Flat hand-built bars give exact, closed-form values for the continuous
/// geometry and structure block. ATR14 is 2.0 because every true range is 2.
#[test]
fn price_action_geometry_matches_flat_hand_bars() {
    let dir = tmp_dir("pa_flat");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = (0..120i64)
        .map(|i| (i * 900, "AAA".to_string(), 100.0, 101.0, 99.0, 100.5, 10.0))
        .collect();
    write_store(&src, &rows, 32);
    let spec = format!("{},batr", enrich::PRICE_ACTION_NAMES.join(","));
    enrich::build(&opts(&src, &out, &spec)).unwrap();

    let clv16 = column(&out, "clv16");
    let body16 = column(&out, "body16");
    let rng_atr = column(&out, "rng_atr");
    let prev = column(&out, "prev_range_pos");
    let inside = column(&out, "inside_s");
    let outside = column(&out, "outside_s");
    let engulf = column(&out, "engulf_s");
    let harami = column(&out, "harami_s");
    let run8 = column(&out, "run8");
    let nr4 = column(&out, "nr4");
    let trend = column(&out, "trend96");
    let batr = column(&out, "batr");

    for i in 0..rows.len() {
        assert!(rng_atr[i].is_finite(), "rng_atr warm-up at {i}");
        assert!((rng_atr[i] - 1.0).abs() <= 1e-12, "rng_atr at {i}: {}", rng_atr[i]);
        assert!((batr[i] - 0.25).abs() <= 1e-12, "batr at {i}: {}", batr[i]);
        if i >= 15 {
            assert!((clv16[i] - 0.25).abs() <= 1e-12, "clv16 at {i}: {}", clv16[i]);
            assert!((body16[i] - 0.25).abs() <= 1e-12, "body16 at {i}: {}", body16[i]);
        } else {
            assert!(clv16[i].is_nan(), "clv16 warm-up at {i}");
            assert!(body16[i].is_nan(), "body16 warm-up at {i}");
        }
        if i >= 1 {
            assert!((prev[i] - 0.25).abs() <= 1e-12, "prev_range_pos at {i}: {}", prev[i]);
            assert!((inside[i] - 1.0).abs() <= 1e-12, "inside_s at {i}: {}", inside[i]);
            assert!((outside[i] - 1.0).abs() <= 1e-12, "outside_s at {i}: {}", outside[i]);
            assert!(engulf[i].abs() <= 1e-12, "engulf_s at {i}: {}", engulf[i]);
            assert!(harami[i].abs() <= 1e-12, "harami_s at {i}: {}", harami[i]);
        }
        if i >= 1 {
            assert!(run8[i].abs() <= 1e-12, "run8 at {i}: {}", run8[i]);
        }
        if i >= 3 {
            assert!((nr4[i] - 1.0).abs() <= 1e-12, "nr4 at {i}: {}", nr4[i]);
        }
        if i >= 95 {
            assert!(trend[i].abs() <= 1e-12, "trend96 at {i}: {}", trend[i]);
        }
    }
}


/// A rising close series produces the signed clipped streak.
#[test]
fn run8_tracks_the_signed_close_streak_and_clips_at_eight() {
    let dir = tmp_dir("pa_run");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = (0..20i64)
        .map(|i| {
            let c = 100.0 + i as f64;
            (i * 900, "AAA".to_string(), c - 1.0, c + 0.5, c - 1.5, c, 10.0)
        })
        .collect();
    write_store(&src, &rows, 16);
    enrich::build(&opts(&src, &out, "run8")).unwrap();
    let run = column(&out, "run8");
    assert!(run[0].is_nan());
    for i in 1..8 {
        assert!((run[i] - i as f64 / 8.0).abs() <= 1e-12, "run8 at {i}: {}", run[i]);
    }
    for i in 8..20 {
        assert!((run[i] - 1.0).abs() <= 1e-12, "run8 clipped at {i}: {}", run[i]);
    }
}


/// Hand-built reversal shapes exercise the continuous inside/outside/engulf and
/// harami strengths, including their sign.
#[test]
fn price_action_strengths_are_continuous_and_signed() {
    let dir = tmp_dir("pa_shapes");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let bars: Vec<(f64, f64, f64, f64)> = vec![
        (100.0, 101.0, 99.0, 99.5),
        (99.2, 101.0, 99.0, 100.6),   // bullish engulfing
        (101.0, 101.5, 98.9, 99.1),   // bearish engulfing
        (99.8, 100.2, 99.4, 99.9),    // inside bar
        (99.5, 102.0, 98.0, 100.0),   // outside bar
        (101.0, 101.5, 98.5, 99.0),   // prior down body
        (99.6, 99.9, 99.3, 99.7),     // bullish harami
        (99.5, 101.0, 99.0, 100.5),   // prior up body
        (100.2, 100.5, 99.3, 99.9),   // bearish harami
    ];
    let rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = bars
        .iter()
        .enumerate()
        .map(|(i, (o, h, l, c))| (i as i64 * 900, "AAA".to_string(), *o, *h, *l, *c, 10.0))
        .collect();
    write_store(&src, &rows, 16);
    enrich::build(
        &opts(&src, &out, "inside_s,outside_s,engulf_s,harami_s"),
    )
    .unwrap();
    let inside = column(&out, "inside_s");
    let outside = column(&out, "outside_s");
    let engulf = column(&out, "engulf_s");
    let harami = column(&out, "harami_s");

    assert!(engulf[1] > 0.0, "bullish engulfing must be positive: {}", engulf[1]);
    assert!(engulf[2] < 0.0, "bearish engulfing must be negative: {}", engulf[2]);
    assert!(harami[6] > 0.0, "bullish harami must be positive: {}", harami[6]);
    assert!(harami[8] < 0.0, "bearish harami must be negative: {}", harami[8]);
    for v in engulf.iter().chain(harami.iter()).filter(|v| v.is_finite()) {
        assert!(v.abs() <= 1.0, "strength is not bounded: {v}");
    }
    assert!((inside[3] - 1.0).abs() <= 1e-12, "inside bar: {}", inside[3]);
    assert!(outside[3] < 1.0, "inside bar is not an outside bar: {}", outside[3]);
    assert!((outside[4] - 1.0).abs() <= 1e-12, "outside bar: {}", outside[4]);
    assert!(inside[4] < outside[4], "outside bar must score inside lower");
}


/// The three shape identities hold on random finite-range bars.
#[test]
fn price_action_identities_hold_on_random_bars() {
    let dir = tmp_dir("pa_identities");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    let mut seed = 0x1234_5678_9abc_def0u64;
    let mut next = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed as f64) / (u64::MAX as f64)
    };
    for i in 0..300i64 {
        let o = 50.0 + next() * 40.0;
        let c = 50.0 + next() * 40.0;
        let h = o.max(c) + 0.05 + next() * 2.0;
        let l = o.min(c) - 0.05 - next() * 2.0;
        rows.push((i * 900, "AAA".to_string(), o, h, l, c, 10.0));
    }
    write_store(&src, &rows, 32);
    enrich::build(&opts(&src, &out, "clv1,body1,wasym1")).unwrap();
    let clv = column(&out, "clv1");
    let body = column(&out, "body1");
    let wasym = column(&out, "wasym1");
    let mut checked = 0;
    for (i, (_, _, o, h, l, c, _)) in rows.iter().enumerate() {
        let r = h - l;
        if !(r > 0.0) {
            continue;
        }
        let u = h - o.max(*c);
        let d = o.min(*c) - l;
        let up = u / r;
        let down = d / r;
        assert!(
            (up - (0.5 - clv[i] + body[i].min(0.0))).abs() <= 1e-12,
            "U/R identity at {i}"
        );
        assert!(
            (down - (0.5 + clv[i] - body[i].max(0.0))).abs() <= 1e-12,
            "D/R identity at {i}"
        );
        assert!(
            (wasym[i] - (body[i] - 2.0 * clv[i])).abs() <= 1e-12,
            "wasym identity at {i}"
        );
        checked += 1;
    }
    assert!(checked > 250, "too few finite-range bars ({checked})");
}


/// A zero-range or zero-volume bar is NaN for strict geometry, never a fake 0.
/// The compact `batr` keeps its documented compact zero on `H == L`.
#[test]
fn price_action_zero_range_and_zero_volume_are_nan() {
    let dir = tmp_dir("pa_degenerate");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = (0..220i64)
        .map(|i| (i * 900, "AAA".to_string(), 100.0, 101.0, 99.0, 100.5, 10.0))
        .collect();
    rows[100] = (100 * 900, "AAA".to_string(), 100.0, 100.0, 100.0, 100.0, 10.0);
    rows[120].6 = 0.0;
    write_store(&src, &rows, 32);
    let spec = format!("{},batr", enrich::PRICE_ACTION_NAMES.join(","));
    enrich::build(&opts(&src, &out, &spec)).unwrap();

    for name in ["rng_atr", "clv16", "body16", "prev_range_pos", "inside_s", "outside_s", "engulf_s", "harami_s", "nr4", "trend96"] {
        let v = column(&out, name);
        assert!(v[100].is_nan(), "{name} must be NaN at a zero-range bar");
        assert!(v[120].is_nan(), "{name} must be NaN at a zero-volume bar");
    }
    let nr4 = column(&out, "nr4");
    assert!(nr4[101].is_nan() && nr4[102].is_nan() && nr4[103].is_nan());
    assert!(nr4[104].is_finite(), "nr4 must recover four bars after the hole");
    let trend = column(&out, "trend96");
    assert!(trend[215].is_nan(), "trend96 window still contains the zero-volume bar");
    assert!(trend[216].is_finite(), "trend96 must recover after a full window");
    let batr = column(&out, "batr");
    assert_eq!(batr[100], 0.0, "compact batr keeps the documented H==L zero");
}


/// Every two-bar feature is NaN across a missing bucket, and every 16/96-bar
/// window is NaN until the hole leaves it. Nothing bridges a gap.
#[test]
fn price_action_two_bar_and_window_gap_masked() {
    let dir = tmp_dir("pa_gap");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for i in 0..700i64 {
        let t = if i < 350 { i * 900 } else { (i + 3) * 900 };
        let p = 100.0 + i as f64 * 0.01 + (i as f64 * 0.3).sin();
        rows.push((t, "AAA".to_string(), p, p + 0.2, p - 0.2, p + 0.05, 10.0 + i as f64));
    }
    write_store(&src, &rows, 64);
    let spec = enrich::PRICE_ACTION_NAMES.join(",");
    enrich::build(&opts(&src, &out, &spec)).unwrap();

    for name in ["prev_range_pos", "inside_s", "outside_s", "engulf_s", "harami_s"] {
        let v = column(&out, name);
        assert!(v[350].is_nan(), "{name} bridged the hole at row 350");
    }
    for name in ["clv16", "body16"] {
        let v = column(&out, name);
        for i in 350..365 {
            assert!(v[i].is_nan(), "{name} bridged the hole at row {i}");
        }
        assert!(v[365].is_finite(), "{name} did not recover after the hole");
    }
    let trend = column(&out, "trend96");
    for i in 350..445 {
        assert!(trend[i].is_nan(), "trend96 bridged the hole at row {i}");
    }
    assert!(trend[445].is_finite(), "trend96 did not recover after the hole");
    let nr4 = column(&out, "nr4");
    for i in 350..353 {
        assert!(nr4[i].is_nan(), "nr4 bridged the hole at row {i}");
    }
    assert!(nr4[353].is_finite(), "nr4 did not recover after the hole");
}


/// Prefix causality: changing a later bar never changes an earlier row, and a
/// truncated store recomputes the same prefix.
#[test]
fn price_action_is_prefix_causal_under_a_shifted_future_bar() {
    let dir = tmp_dir("pa_causal");
    let full = dir.join("full.rcsb").to_string_lossy().into_owned();
    let cut = dir.join("cut.rcsb").to_string_lossy().into_owned();
    let shifted = dir.join("shifted.rcsb").to_string_lossy().into_owned();
    let out_full = dir.join("full_out.rcsb").to_string_lossy().into_owned();
    let out_cut = dir.join("cut_out.rcsb").to_string_lossy().into_owned();
    let out_shifted = dir.join("shifted_out.rcsb").to_string_lossy().into_owned();
    let rows = bars("AAA", 320, 0, 900);
    let mut future = rows.clone();
    future[250] = (
        future[250].0,
        future[250].1.clone(),
        future[250].2 + 37.0,
        future[250].3 + 40.0,
        future[250].4 - 5.0,
        future[250].5 + 21.0,
        999.0,
    );
    write_store(&full, &rows, 31);
    write_store(&cut, &rows[..250], 31);
    write_store(&shifted, &future, 31);
    let spec = format!("{},batr", enrich::PRICE_ACTION_NAMES.join(","));
    enrich::build(&opts(&full, &out_full, &spec)).unwrap();
    enrich::build(&opts(&cut, &out_cut, &spec)).unwrap();
    enrich::build(&opts(&shifted, &out_shifted, &spec)).unwrap();
    for name in enrich::PRICE_ACTION_NAMES.iter().chain(["batr"].iter()) {
        let a = column(&out_full, name);
        let b = column(&out_cut, name);
        let c = column(&out_shifted, name);
        for i in 0..250 {
            assert_eq!(a[i].is_nan(), b[i].is_nan(), "{name} prefix NaN differs at {i}");
            assert_eq!(a[i].is_nan(), c[i].is_nan(), "{name} future NaN differs at {i}");
            if a[i].is_finite() {
                assert!((a[i] - b[i]).abs() <= 1e-12, "{name} prefix differs at {i}");
                assert!((a[i] - c[i]).abs() <= 1e-12, "{name} future bar changed {i}");
            }
        }
    }
}


/// The ten compact candle names pass the independent Python fixture at
/// `abs/rel <= 1e-12`; `wasym1` is recomputed from `body1` and `clv1`.
#[test]
fn compact_candle_subset_passes_parity_at_1e12() {
    let base = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/enrich");
    let input = base.join("compact_15m_ohlcv.csv");
    let expected_path = base.join("compact_15m_expected.csv");
    let dir = tmp_dir("candle_parity");
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let names = [
        "clv1", "clv4", "body1", "batr", "wasym1", "lower_wick", "rng1", "volrat", "gapatr", "don96",
    ];
    enrich::build(&opts(&input.to_string_lossy(), &out, &names.join(","))).unwrap();

    let text = std::fs::read_to_string(&expected_path).unwrap();
    let mut lines = text.lines();
    let header: Vec<&str> = lines.next().unwrap().split(',').collect();
    let start_ts = 1_680_271_200i64;
    let rows_per_symbol = 3600usize;
    let out_cols: Vec<Vec<f64>> = names.iter().map(|n| column(&out, n)).collect();
    let mut compared = 0usize;
    let mut worst = 0.0f64;
    for line in lines {
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').collect();
        let ts: i64 = fields[0].parse().unwrap();
        let sym = fields[1];
        let seg = if sym == "AAVEUSDT" { 0 } else { 1 };
        let idx = seg * rows_per_symbol + ((ts - start_ts) / 900) as usize;
        for (j, name) in names.iter().enumerate() {
            let col = header.iter().position(|h| h == name).unwrap();
            let expected = fields[col].parse::<f64>().unwrap();
            let got = out_cols[j][idx];
            match (got.is_nan(), expected.is_nan()) {
                (true, true) => continue,
                (true, false) => panic!("{name} at {sym}/{ts}: got NaN, expected {expected}"),
                (false, true) => panic!("{name} at {sym}/{ts}: got {got}, expected NaN"),
                (false, false) => {}
            }
            let a = (got - expected).abs();
            let r = a / expected.abs().max(1e-300);
            worst = worst.max(a.max(r));
            assert!(
                a <= 1e-12 || r <= 1e-12,
                "{name} at {sym}/{ts}: got {got}, expected {expected}, abs {a:e} rel {r:e}"
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 300 * names.len());
    let clv = column(&out, "clv1");
    let body = column(&out, "body1");
    let wasym = column(&out, "wasym1");
    let mut identity = 0usize;
    for i in 0..clv.len() {
        if wasym[i].is_finite() && body[i].is_finite() && clv[i].is_finite() {
            assert!((wasym[i] - (body[i] - 2.0 * clv[i])).abs() <= 1e-12);
            identity += 1;
        }
    }
    assert!(identity > 0);
    eprintln!("candle compact parity: {compared} values, worst abs/rel {worst:e}");
}


/// The report lists warm-up, NaN count, prevalence and the first finite date
/// for every column, in text and JSON.
#[test]
fn enrich_report_lists_prevalence_and_first_finite_date() {
    let dir = tmp_dir("pa_report");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = (0..120i64)
        .map(|i| (i * 900, "AAA".to_string(), 100.0, 101.0, 99.0, 100.5, 10.0))
        .collect();
    write_store(&src, &rows, 32);
    let report = enrich::build(&opts(&src, &out, "clv16")).unwrap();
    let f = report.features.iter().find(|f| f.name == "clv16").unwrap();
    assert_eq!(f.warmup, 15);
    assert_eq!(f.nan_rows, 15);
    assert_eq!(f.first_finite_row, Some(15));
    assert_eq!(f.first_finite_time, Some(15 * 900));
    assert!((f.prevalence - 105.0 / 120.0).abs() <= 1e-12);
    let json = report.json();
    assert!(json.contains("\"prevalence\":0.875"), "json: {json}");
    assert!(json.contains("\"first_finite_iso\":\"1970-01-01T03:45:00Z\""), "json: {json}");
    let text = report.text_lines().join("\n");
    assert!(text.contains("# enrich-feature: name=clv16 warmup=15 nan_rows=15 prevalence=0.875000"));
    assert!(text.contains("first_finite_iso=1970-01-01T03:45:00Z"));
}



/// The complete engine feature-bucket inventory: every name in
/// `enrich::engine_names()` is registered `available:true` with a real family,
/// exact formula and warm-up, and the union command emits exactly those names,
/// once, through real producers.
#[test]
fn engine_inventory_is_complete_available_and_runs() {
    let dir = tmp_dir("engine_inventory");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let n = 3200usize;
    let mut rows = wavy_rows("AAA", n, 0, 900, 0.0);
    rows.extend(wavy_rows("BTCUSDT", n, 0, 900, 1.3));
    write_store_with_flow(&src, &rows, 128);

    let engine = enrich::engine_names();
    let catalogue = enrich::catalogue_names();
    assert_eq!(catalogue.len(), 216, "the #271 catalogue command count");
    assert_eq!(engine.len(), 254, "the full engine inventory incl. compact-only names");
    assert!(engine.len() > catalogue.len());
    assert_eq!(
        enrich::expand("all,py15m,ta:full,exotic:cheap,cross:btc,compact").unwrap(),
        engine,
        "the advertised command must resolve to the engine inventory"
    );

    let report = enrich::build(&opts(
        &src,
        &out,
        "all,py15m,ta:full,exotic:cheap,cross:btc,compact",
    ))
    .unwrap();
    assert_eq!(report.features.len(), engine.len(), "emitted vs registry count");
    let reg = enrich::default_registry();
    for name in &engine {
        let e = reg
            .iter()
            .find(|e| e.name == *name)
            .unwrap_or_else(|| panic!("{name} has no registry entry"));
        assert!(e.available, "{name} must be available:true");
        assert_eq!(e.timeframe, "15m", "{name} timeframe");
        assert_ne!(e.family, "other", "{name} has no engine family");
        assert_ne!(e.formula, "see docs/ohlcv.md", "{name} has no exact formula");
        assert!(e.warmup <= n, "{name} warm-up {} exceeds the fixture", e.warmup);
    }
    for f in &report.features {
        if f.name.starts_with("xs_rank_") || matches!(f.name.as_str(), "mkt_r16" | "mkt_r96" | "mkt_vol96") {
            // #273 market state and ranks require 20 point-in-time members;
            // this two-symbol engine inventory cannot produce one. The
            // dedicated 20-member family test covers the finite path.
            assert_eq!(f.nan_rows, report.rows, "{} must need 20 members", f.name);
        } else {
            assert!(f.nan_rows < report.rows, "{} is NaN on every row", f.name);
        }
    }
    let schema = rcsr::schema::json();
    assert!(
        schema.contains(&format!("\"engine_catalogue\":{{\"count\":{}", engine.len())),
        "schema does not record the engine inventory count"
    );
    for name in &engine {
        assert!(schema.contains(&format!("\"name\":\"{}\"", name)), "{name} absent from schema");
    }

    // Representative hand-computed formulas for every engine family. The
    // compact/py15m parity fixtures cover the full 52/35 sets separately.
    let close: Vec<f64> = rows.iter().take(n).map(|r| r.5).collect();
    let high: Vec<f64> = rows.iter().take(n).map(|r| r.3).collect();
    let low: Vec<f64> = rows.iter().take(n).map(|r| r.4).collect();
    let vol: Vec<f64> = rows.iter().take(n).map(|r| r.6).collect();
    let t: Vec<i64> = rows.iter().take(n).map(|r| r.0).collect();
    let mut lr = vec![f64::NAN; n];
    for i in 1..n {
        lr[i] = (close[i] / close[i - 1]).ln();
    }
    let i = n - 1;
    let mean = |x: &[f64]| x.iter().sum::<f64>() / x.len() as f64;
    let std_pop = |x: &[f64]| {
        let m = mean(x);
        (x.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / x.len() as f64).sqrt()
    };
    let check = |name: &str, want: f64| {
        let got = column(&out, name)[i];
        let a = (got - want).abs();
        let r = a / want.abs().max(1e-300);
        assert!(
            a <= 1e-10 || r <= 1e-10,
            "{name}: got {got}, expected {want}, abs {a:e} rel {r:e}"
        );
    };
    check("r1", (close[i] / close[i - 1]).ln());
    check("r5", (close[i] / close[i - 5]).ln());
    check("r120", (close[i] / close[i - 120]).ln());
    check("ma20", close[i] / mean(&close[i - 19..=i]) - 1.0);
    check("vol20", std_pop(&lr[i - 19..=i]));
    check("bb20_z", (close[i] - mean(&close[i - 19..=i])) / std_pop(&close[i - 19..=i]));
    check("bb20_bw", 4.0 * std_pop(&close[i - 19..=i]) / mean(&close[i - 19..=i]));
    let logv: Vec<f64> = vol.iter().map(|v| (1.0 + v).ln()).collect();
    check(
        "vz96",
        (logv[i] - mean(&logv[i - 95..=i])) / std_pop(&logv[i - 95..=i]),
    );
    let lo20 = low[i - 19..=i].iter().fold(f64::INFINITY, |a, b| a.min(*b));
    let hi20 = high[i - 19..=i].iter().fold(f64::NEG_INFINITY, |a, b| a.max(*b));
    check("don20", (close[i] - lo20) / (hi20 - lo20));
    check("lag_r1_3", lr[i - 3]);
    check("rmean5", mean(&lr[i - 4..=i]));
    check("rstd5", std_pop(&lr[i - 4..=i]));
    let m = mean(&lr[i - 19..=i]);
    let sd = std_pop(&lr[i - 19..=i]);
    check("skew20", lr[i - 19..=i].iter().map(|v| ((v - m) / sd).powi(3)).sum::<f64>() / 20.0);
    check(
        "kurt20",
        lr[i - 19..=i].iter().map(|v| ((v - m) / sd).powi(4)).sum::<f64>() / 20.0 - 3.0,
    );
    check("close_pos20", (close[i] - lo20) / (hi20 - lo20));
    check("range1", (high[i] - low[i]) / close[i]);
    let day_frac = t[i].rem_euclid(86_400) as f64 / 86_400.0;
    check("hour_sin", (2.0 * std::f64::consts::PI * day_frac).sin());
    let wd = rcsr::ohlcv::weekday(t[i].div_euclid(86_400)) as f64;
    check("dow_sin", (2.0 * std::f64::consts::PI * wd / 7.0).sin());
}

/// A name shared by several groups has exactly one canonical value, whichever
/// specification requested it.
#[test]
fn engine_shared_names_have_one_canonical_value() {
    let dir = tmp_dir("engine_aliases");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let mut rows = bars("AAA", 400, 0, 900);
    rows.extend(bars("BTCUSDT", 400, 0, 900));
    write_store_with_flow(&src, &rows, 64);

    let pairs = [
        ("rsi14", "compact,py15m,ta:full,rsi"),
        ("mfi14", "compact,py15m,ta:full"),
        ("cmf20", "compact,ta:full"),
        ("adx14", "compact,py15m,ta:full"),
        ("clv1", "compact,candle,all"),
        ("don96", "compact,donchian,all"),
        ("macd_hist", "compact,py15m"),
        ("vwap_dev96", "flow,py15m,compact"),
        ("obv_z96", "flow,py15m"),
        ("taker_ratio96", "compact,flow"),
        ("taker_imb16", "compact,flow"),
        ("trade_intensity96", "compact,flow"),
        ("bb20_bw", "bollinger,py15m"),
        ("natr14", "atr,compact,ta:full"),
    ];
    for (name, groups) in pairs {
        let a_out = dir.join(format!("{name}-solo.rcsb")).to_string_lossy().into_owned();
        let b_out = dir.join(format!("{name}-group.rcsb")).to_string_lossy().into_owned();
        enrich::build(&opts(&src, &a_out, name)).unwrap();
        enrich::build(&opts(&src, &b_out, groups)).unwrap();
        let a = column(&a_out, name);
        let b = column(&b_out, name);
        for i in 0..a.len() {
            assert_eq!(a[i].is_nan(), b[i].is_nan(), "{name} NaN mask differs at {i}");
            if a[i].is_finite() {
                let d = (a[i] - b[i]).abs();
                assert!(
                    d <= 1e-12 || d / a[i].abs().max(1e-300) <= 1e-12,
                    "{name} at {i}: solo {a:?} group {b:?}",
                    a = a[i],
                    b = b[i]
                );
            }
        }
    }
}

/// #125 engine names that are not compact also obey the same gap policy as the
/// rest of the catalogue: a return or window spanning a hole is NaN.
#[test]
fn engine_feature_groups_are_gap_masked() {
    let dir = tmp_dir("engine_gap");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let gap = 450i64;
    let mut rows: Vec<(i64, String, f64, f64, f64, f64, f64)> = Vec::new();
    for symbol in ["AAA", "BTCUSDT"] {
        let base = if symbol == "AAA" { 100.0 } else { 200.0 };
        for i in 0..900i64 {
            let t = if i < gap { i * 900 } else { (i + 5) * 900 };
            let p = base + (i as f64 * 0.23).sin() * 2.0 + i as f64 * 0.01;
            rows.push((t, symbol.to_string(), p, p + 0.3, p - 0.4, p + 0.1, 10.0 + i as f64));
        }
    }
    write_store_with_flow(&src, &rows, 64);
    let spec = "ret1,ret5,open_gap1,ma20_ratio,ma60_ratio,bb20_z,vol20_ratio,cmf20,mfi14,\
                vwap20_dev,rmean5,rstd5,skew20,kurt20,close_pos20,cross_ret1,cross_beta20,\
                atr14,rsi14,adx14,don20,lag_r1_3";
    enrich::build(&opts(&src, &out, spec)).unwrap();
    for segment in 0..2usize {
        let start = segment * 900 + gap as usize;
        let after = start + 70;
        for name in spec.split(',') {
            let v = column(&out, name);
            assert!(v[start].is_nan(), "{name} bridged the gap at row {start}");
            assert!(v[after].is_finite(), "{name} did not recover after the gap");
        }
    }
}

/// The whole engine inventory is causal on a prefix: truncating the store never
/// changes an earlier row.
#[test]
fn engine_inventory_is_causal_on_a_prefix() {
    let dir = tmp_dir("engine_causal");
    let full = dir.join("full.rcsb").to_string_lossy().into_owned();
    let prefix = dir.join("prefix.rcsb").to_string_lossy().into_owned();
    let full_out = dir.join("full_out.rcsb").to_string_lossy().into_owned();
    let prefix_out = dir.join("prefix_out.rcsb").to_string_lossy().into_owned();
    let n = 3400usize;
    let cut = 3200usize; // per-symbol prefix, long enough for the 2,880-bar window
    let mut rows = wavy_rows("AAA", n, 0, 900, 0.0);
    rows.extend(wavy_rows("BTCUSDT", n, 0, 900, 1.3));
    let mut short = wavy_rows("AAA", cut, 0, 900, 0.0);
    short.extend(wavy_rows("BTCUSDT", cut, 0, 900, 1.3));
    write_store_with_flow(&full, &rows, 128);
    write_store_with_flow(&prefix, &short, 128);
    let spec = "all,py15m,ta:full,exotic:cheap,cross:btc,compact";
    enrich::build(&opts(&full, &full_out, spec)).unwrap();
    enrich::build(&opts(&prefix, &prefix_out, spec)).unwrap();
    let input_names = FileStore::open(&full).unwrap().header().names();
    let names = FileStore::open(&full_out).unwrap().header().names();
    for name in names.iter().skip(input_names.len()) {
        let a = column(&full_out, name);
        let b = column(&prefix_out, name);
        for i in 0..cut {
            assert_eq!(a[i].is_nan(), b[i].is_nan(), "{name} NaN pattern differs at {i}");
            if a[i].is_finite() {
                assert!(
                    (a[i] - b[i]).abs() <= 1e-12,
                    "{name} at {i}: full {} prefix {}",
                    a[i],
                    b[i]
                );
            }
        }
    }
}

