//! `rcsr enrich` (issue #200): append-only feature enrichment, gap/causality
//! rules, determinism across chunk sizes and the CPU/GPU cost model.

use rcsr::data;
use rcsr::enrich::{self, GpuMode, OpClass, Options};
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

fn write_store(path: &str, rows: &[(i64, String, f64, f64, f64, f64, f64)], group_rows: usize) {
    let refs: Vec<(i64, &str, f64, f64, f64, f64, f64)> =
        rows.iter().map(|(t, s, o, h, l, c, v)| (*t, s.as_str(), *o, *h, *l, *c, *v)).collect();
    let text = csv_of(&refs);
    let t = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &t, group_rows, Some(0)).unwrap();
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
fn cross_sectional_ranks_are_stable_and_break_ties_by_symbol() {
    // t=0: AAA=1, BBB=2, CCC=2 (tie); t=1: AAA=3, BBB=2, CCC=1.
    let src = vec![1.0, 2.0, 2.0, 3.0, 2.0, 1.0];
    let times = vec![0i64, 0, 0, 1, 1, 1];
    let keys = vec![0u32, 1, 2, 0, 1, 2];
    let r = enrich::xs_rank(&src, &times, &keys, 6).unwrap();
    assert_eq!(r[0], 0.0);
    assert_eq!(r[1], 0.5, "tie broken by symbol order");
    assert_eq!(r[2], 1.0);
    assert_eq!(r[3], 1.0);
    assert_eq!(r[4], 0.5);
    assert_eq!(r[5], 0.0);
    // Repeating the call is bit-identical.
    assert_eq!(r, enrich::xs_rank(&src, &times, &keys, 6).unwrap());
}

#[test]
fn cost_model_keeps_small_stores_on_the_cpu_and_names_the_reason() {
    let mut counts = std::collections::BTreeMap::new();
    counts.insert("returns", (120usize, OpClass::Elementwise));
    let small = enrich::plan_groups(&counts, 10_000, GpuMode::Auto, true);
    assert_eq!(small[0].backend, "cpu");
    assert!(small[0].reason.contains("below"));
    let large = enrich::plan_groups(&counts, 3_900_000, GpuMode::Auto, true);
    assert_eq!(large[0].backend, "cpu");
    assert!(large[0].reason.contains("no exact GPU kernel"));
    assert!(large[0].gpu_ms < large[0].cpu_ms, "the cost model should still see a GPU win");
    let off = enrich::plan_groups(&counts, 3_900_000, GpuMode::Off, true);
    assert_eq!(off[0].backend, "cpu");
    let none = enrich::plan_groups(&counts, 3_900_000, GpuMode::Auto, false);
    assert!(none[0].reason.contains("no usable CUDA"));
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
