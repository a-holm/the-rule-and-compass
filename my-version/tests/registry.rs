//! #260: feature registry, schema sidecar, family-stratified views, the
//! group-by-group store writer, the time-major family append and the lag menu.

use rcsr::data::{self, LagSpec, TimeSpec};
use rcsr::enrich::{self, Options};
use rcsr::panel_append::{self, AppendOptions};
use rcsr::registry;
use rcsr::store::{self, ColumnPlan, DType, FileStore};
use std::process::Command;

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-registry-{}-{}-{}",
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

fn path(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn column(p: &str, name: &str) -> Vec<f64> {
    let s = FileStore::open(p).unwrap();
    let c = s.header().column_index(name).unwrap();
    let mut out = Vec::new();
    for g in 0..s.n_groups() {
        out.extend(s.read_f64_group(g, c).unwrap());
    }
    out
}

/// Registered derived feature names, in registry order.
fn feature_names() -> Vec<String> {
    enrich::default_registry()
        .into_iter()
        .filter(|e| e.available && e.source != "input")
        .map(|e| e.name)
        .collect()
}

/// A wide store whose column names are registry names and whose values are
/// deterministic noise. Only the names matter to the registry.
fn registry_store(dir: &std::path::Path, n_cols: usize, rows: usize, file: &str) -> (String, Vec<String>) {
    let names: Vec<String> = feature_names().into_iter().take(n_cols).collect();
    let mut text = String::from("ts,symbol,y");
    for n in &names {
        text.push(',');
        text.push_str(n);
    }
    text.push('\n');
    for r in 0..rows {
        let sym = if r % 2 == 0 { "AAA" } else { "BBB" };
        let ts = 1_700_000_000i64 + (r as i64 / 2) * 3600;
        text.push_str(&format!("{},{},{}", ts, sym, (r as f64 * 0.07).sin()));
        for c in 0..names.len() {
            text.push_str(&format!(",{}", ((r * 31 + c * 17) % 101) as f64 * 0.01 + c as f64 * 0.001));
        }
        text.push('\n');
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let p = path(dir, file);
    store::write_store_with_time(&p, &table, 64, Some(0)).unwrap();
    (p, names)
}

#[test]
fn the_default_set_is_registered_with_complete_metadata() {
    let reg = enrich::default_registry();
    let quality = ["gap", "cov15m", "flags", "traded", "dup_rows", "archive_unit"];
    assert!(quality.iter().all(|q| registry::lookup(q).is_some()));
    assert!(reg.len() + quality.len() >= 364, "registered columns: {}", reg.len() + quality.len());
    let available: Vec<_> = reg.iter().filter(|e| e.available).collect();
    assert!(reg.iter().all(|e| e.available || e.reason.is_some()), "an unavailable entry names its missing dependency");
    for e in &available {
        let r = registry::lookup(&e.name).unwrap_or_else(|| panic!("{} not resolvable", e.name));
        assert!(!r.family.is_empty() && !r.timeframe.is_empty(), "{}", e.name);
        assert!(!r.pit_rule.is_empty() && !r.unit.is_empty(), "{}", e.name);
        assert!(r.causal_offset <= 0, "{} reaches the future", e.name);
        if !r.input {
            assert!(!r.sources.is_empty(), "{} has no source columns", e.name);
            assert!(!r.transform.is_empty(), "{}", e.name);
        }
    }
    let lag = registry::lookup("r16_lag4").unwrap();
    assert_eq!(lag.family, "lag");
    assert_eq!(lag.causal_offset, -4);
    assert_eq!(lag.warmup, registry::lookup("r16").unwrap().warmup + 4);
    assert!(registry::lookup("not_a_feature").is_none());
}

#[test]
fn the_schema_sidecar_hashes_every_column_and_detects_a_change() {
    let dir = tmp_dir("schema");
    let (p, names) = registry_store(&dir, 30, 200, "panel.rcsb");
    // `y` is not a registered feature, so a store with it has no valid schema.
    let err = registry::write_schema(&p, "2").unwrap_err();
    assert!(err.contains("unknown column 'y'"), "{err}");

    let mut text = String::from("ts,symbol");
    for n in &names {
        text.push_str(&format!(",{}", n));
    }
    text.push('\n');
    for r in 0..120 {
        text.push_str(&format!("{},AAA", 1_700_000_000i64 + r as i64 * 3600));
        for c in 0..names.len() {
            let v = if r < 4 { "nan".to_string() } else { format!("{}", (r + c) as f64 * 0.5) };
            text.push_str(&format!(",{}", v));
        }
        text.push('\n');
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let q = path(&dir, "ok.rcsb");
    store::write_store_with_time(&q, &table, 50, Some(0)).unwrap();
    let schema = registry::write_schema(&q, "2").unwrap();
    let body = std::fs::read_to_string(&schema).unwrap();
    assert!(body.contains("\"panel_version\":\"2\""));
    assert!(body.contains("\"schema_version\":1"));
    assert!(body.contains("\"feature_registry_version\":1"));
    let hashes = registry::read_hashes(&schema).unwrap();
    assert_eq!(hashes.len(), names.len() + 2);
    for h in hashes.values() {
        assert_eq!(h.len(), 64);
    }
    // Availability starts at the first finite value (row 4).
    let first = body.lines().find(|l| l.starts_with(&format!("{{\"name\":\"{}\"", names[0]))).unwrap();
    assert!(first.contains(&format!("\"availability_start\":{}", 1_700_000_000i64 + 4 * 3600)), "{first}");
    assert!(registry::verify_schema(&q).unwrap().is_empty());

    // Change one stored value: only that column's hash differs.
    let mut changed = text.clone();
    changed = changed.replacen(&format!(",{}\n", (119) as f64 * 0.5 + 29.0 * 0.5), ",999\n", 1);
    let table2 = data::parse_table_with_time(&changed, false, Some("ts")).unwrap();
    store::write_store_with_time(&q, &table2, 50, Some(0)).unwrap();
    let bad = registry::verify_schema(&q).unwrap();
    assert_eq!(bad, vec![names[names.len() - 1].clone()], "{bad:?}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_unknown_vars_name_exits_nonzero_with_the_registry_name_and_family() {
    let dir = tmp_dir("unknown");
    let mut text = String::from("ts,symbol,y,r16,r96,vol20\n");
    for r in 0..300 {
        let sym = if r % 2 == 0 { "AAA" } else { "BBB" };
        text.push_str(&format!(
            "{},{},{},{},{},{}\n",
            1_700_000_000i64 + (r as i64 / 2) * 3600,
            sym,
            (r as f64 * 0.07).sin(),
            (r % 7) as f64,
            (r % 5) as f64,
            (r % 11) as f64
        ));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let p = path(&dir, "p.rcsb");
    store::write_store_with_time(&p, &table, 64, Some(0)).unwrap();
    // `y` is a target, so give the schema a target-free twin.
    let twin = path(&dir, "twin.rcsb");
    let t2 = data::parse_table_with_time(
        &text.replace(",y,", ",close,").replacen("ts,symbol,close,r16", "ts,symbol,close,r16", 1),
        false,
        Some("ts"),
    )
    .unwrap();
    store::write_store_with_time(&twin, &t2, 64, Some(0)).unwrap();
    registry::write_schema(&twin, "2").unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", &twin, "--target", "close", "--vars", "r16,vol21", "--time", "ts",
            "--group-by", "symbol", "--generations", "1", "--pop", "8", "--cpu",
        ])
        .output()
        .unwrap();
    assert!(!out.status.success());
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("unknown column 'vol21'"), "{err}");
    assert!(err.contains("registry name 'vol20'") || err.contains("registry name 'vol2"), "{err}");
    assert!(err.contains("family '"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn views_are_deterministic_stratified_and_traceable() {
    let names = feature_names();
    let a = registry::select_view(&names, 36, 7, 0).unwrap();
    let b = registry::select_view(&names, 36, 7, 0).unwrap();
    assert_eq!(a, b);
    assert!(a.columns.len() <= 48 && a.columns.len() >= 24, "{}", a.columns.len());
    for c in &a.columns {
        let r = registry::lookup(c).unwrap_or_else(|| panic!("{c} has no registry entry"));
        assert!(r.available && r.causal_offset <= 0);
    }
    assert_eq!(a.delivered, 36);
    let next = registry::select_view(&names, 36, 7, 1).unwrap();
    assert_ne!(a.id, next.id);
    assert_ne!(a.columns, next.columns);
    let other_seed = registry::select_view(&names, 36, 8, 0).unwrap();
    assert_ne!(a.id, other_seed.id);
    assert!(registry::select_view(&names, 10, 7, 0).is_err());
    assert!(registry::select_view(&names, 64, 7, 0).is_err());
    let research = registry::select_view(&names, 96, 7, 0).unwrap();
    assert!(research.columns.len() > 48);
}

#[test]
fn a_truncated_store_yields_the_same_view_so_no_future_column_enters() {
    let dir = tmp_dir("trunc");
    let (full, names) = registry_store(&dir, 200, 400, "full.rcsb");
    let (prefix, _) = registry_store(&dir, 200, 160, "prefix.rcsb");
    let cols = |p: &str| FileStore::open(p).unwrap().header().names();
    let a = registry::select_view(&cols(&full), 36, 3, 2).unwrap();
    let b = registry::select_view(&cols(&prefix), 36, 3, 2).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.columns.len(), a.columns.iter().filter(|c| names.contains(c)).count());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn project_on_a_200_column_store_records_the_view_in_the_manifest() {
    let dir = tmp_dir("project");
    let (p, _) = registry_store(&dir, 200, 256, "wide.rcsb");
    let run = |view: &str, manifest: &str| {
        let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "--data", &p, "--target", "y", "--project", "24", "--view", view, "--time", "ts",
                "--group-by", "symbol", "--generations", "1", "--pop", "8", "--seed", "5", "--cpu",
                "--json", "--run-manifest", manifest,
            ])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        std::fs::read_to_string(manifest).unwrap()
    };
    let m0 = run("0", &path(&dir, "m0.json"));
    let m0b = run("0", &path(&dir, "m0b.json"));
    let m1 = run("1", &path(&dir, "m1.json"));
    let view = |m: &str| m[m.find("\"view\":{").unwrap()..].lines().next().unwrap().to_string();
    assert_eq!(view(&m0), view(&m0b));
    assert_ne!(view(&m0), view(&m1));
    let id = rcsr::manifest::string_field(&view(&m0)["\"view\":".len()..], "view_id").unwrap();
    assert_eq!(id.len(), 12);
    let n = view(&m0).matches("\"r").count() + view(&m0).matches("\"ma").count();
    assert!(n > 0, "{}", view(&m0));
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Group-by-group writer
// ---------------------------------------------------------------------------

fn mixed_table() -> data::Table {
    let mut text = String::from("ts,symbol,a,b,flag\n");
    for r in 0..1000 {
        text.push_str(&format!(
            "{},{},{},{},{}\n",
            1_700_000_000i64 + r as i64 * 60,
            ["AAA", "BBB", "CCC"][r % 3],
            (r as f64 * 0.37).sin(),
            if r % 13 == 0 { "nan".to_string() } else { format!("{}", r as f64 / 7.0) },
            r % 2
        ));
    }
    data::parse_table_with_time(&text, false, Some("ts")).unwrap()
}

#[test]
fn the_grouped_writer_is_byte_identical_to_the_in_memory_path() {
    let dir = tmp_dir("writer");
    let table = mixed_table();
    let mem = path(&dir, "mem.rcsb");
    store::write_store_with_time(&mem, &table, 128, Some(0)).unwrap();

    let plans: Vec<ColumnPlan> = (0..table.n_cols())
        .map(|c| ColumnPlan {
            name: table.names[c].clone(),
            numeric: table.numeric[c],
            dtype: if c == 0 {
                DType::I64
            } else if !table.numeric[c] && table.dict(c).is_some() {
                DType::U32
            } else {
                DType::F64
            },
            levels: table.dict(c).map(|d| d.levels.clone()).unwrap_or_default(),
            has_codes: table.dict(c).is_some(),
        })
        .collect();
    let grouped = path(&dir, "grouped.rcsb");
    let mut calls = 0usize;
    store::write_store_grouped(&grouped, &plans, table.len(), 128, |lo, hi, out| {
        calls += 1;
        for (c, plan) in plans.iter().enumerate() {
            match plan.dtype {
                DType::F64 => out[c].f64s.extend_from_slice(&table.col(c)[lo..hi]),
                DType::I64 => out[c].i64s.extend_from_slice(&table.ints(c).unwrap()[lo..hi]),
                DType::U32 => {}
            }
            if let Some(d) = table.dict(c) {
                out[c].codes.extend_from_slice(&d.codes[lo..hi]);
            }
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(calls, 8);
    assert_eq!(std::fs::read(&mem).unwrap(), std::fs::read(&grouped).unwrap());

    // A producer that returns the wrong number of rows is refused.
    let err = store::write_store_grouped(&path(&dir, "bad.rcsb"), &plans[..1], 10, 5, |_, _, _| Ok(()))
        .unwrap_err();
    assert!(err.contains("producer returned 0 rows"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

fn vm_hwm_kib() -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find(|l| l.starts_with("VmHWM:"))
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|v| v.parse().ok())
        .unwrap()
}

const RSS_COLS: usize = 128;
const RSS_ROWS: usize = 1_200_000;
const RSS_GROUP: usize = 65_536;
const RSS_BOUND_MIB: u64 = 256;

/// Runs in a child process so the high-water mark covers only this work.
#[test]
fn rss_child() {
    let Ok(dir) = std::env::var("RCSR_REGISTRY_RSS_DIR") else {
        return;
    };
    let p = format!("{}/big.rcsb", dir);
    let mut plans = vec![ColumnPlan {
        name: "ts".to_string(),
        numeric: true,
        dtype: DType::I64,
        levels: Vec::new(),
        has_codes: false,
    }];
    for c in 0..RSS_COLS {
        plans.push(ColumnPlan {
            name: format!("c{}", c),
            numeric: true,
            dtype: DType::F64,
            levels: Vec::new(),
            has_codes: false,
        });
    }
    store::write_store_grouped(&p, &plans, RSS_ROWS, RSS_GROUP, |lo, hi, out| {
        out[0].i64s.extend((lo..hi).map(|r| r as i64));
        for c in 0..RSS_COLS {
            out[c + 1].f64s.extend((lo..hi).map(|r| (r * (c + 1)) as f64));
        }
        Ok(())
    })
    .unwrap();
    let file_mib = std::fs::metadata(&p).unwrap().len() >> 20;
    let written = vm_hwm_kib() >> 10;
    // Projected read of 12 of the 128 columns.
    let fs = FileStore::open(&p).unwrap();
    let names: Vec<String> = std::iter::once("ts".to_string())
        .chain((0..12).map(|c| format!("c{}", c * 10)))
        .collect();
    let proj = store::Projection::resolve_owned(&fs, &names).unwrap();
    let t = fs.to_table_projected(&proj).unwrap();
    assert_eq!(t.len(), RSS_ROWS);
    let full = {
        let c = fs.header().column_index("c10").unwrap();
        fs.read_f64_group(3, c).unwrap()
    };
    let tc = t.index_of("c10").unwrap();
    assert_eq!(&t.col(tc)[3 * RSS_GROUP..3 * RSS_GROUP + 5], &full[..5]);
    let peak = vm_hwm_kib() >> 10;
    println!("RSS file_mib={} after_write_mib={} peak_mib={}", file_mib, written, peak);
    assert!(file_mib > 4 * RSS_BOUND_MIB, "fixture too small to make the bound meaningful");
    assert!(written < RSS_BOUND_MIB, "writer peak {} MiB", written);
    assert!(peak < RSS_BOUND_MIB, "projected read peak {} MiB", peak);
}

#[test]
fn the_grouped_writer_and_projected_read_stay_far_below_the_table_size() {
    let dir = tmp_dir("rss");
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["rss_child", "--exact", "--nocapture", "--test-threads", "1"])
        .env("RCSR_REGISTRY_RSS_DIR", &dir)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{}\n{}", stdout, String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("RSS file_mib="), "{stdout}");
}

// ---------------------------------------------------------------------------
// Time-major family append
// ---------------------------------------------------------------------------

fn bar_rows(symbols: &[&str], n: usize) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for (k, s) in symbols.iter().enumerate() {
        for i in 0..n {
            let p = 100.0 + k as f64 * 3.0 + (i as f64 * 0.21 + k as f64).sin() * 2.0 + i as f64 * 0.05;
            rows.push((
                1_700_000_000 + i as i64 * 3600,
                s.to_string(),
                p,
                p + 0.4,
                p - 0.5,
                p + 0.1,
                10.0 + (i % 9) as f64,
            ));
        }
    }
    rows
}

fn write_rows(p: &str, rows: &[(i64, String, f64, f64, f64, f64, f64)], group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
    for (t, s, o, h, l, c, v) in rows {
        text.push_str(&format!("{},{},{},{},{},{},{}\n", t, s, o, h, l, c, v));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(p, &table, group_rows, Some(0)).unwrap();
}

fn time_major(rows: &[(i64, String, f64, f64, f64, f64, f64)]) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let mut v = rows.to_vec();
    v.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    v
}

fn enrich_family(dir: &std::path::Path, tag: &str, rows: &[(i64, String, f64, f64, f64, f64, f64)], features: &str) -> String {
    let src = path(dir, &format!("{}-base.rcsb", tag));
    let out = path(dir, &format!("{}-family.rcsb", tag));
    write_rows(&src, rows, 41);
    enrich::build(&Options {
        data: src,
        out: Some(out.clone()),
        features: vec![features.to_string()],
        ..Default::default()
    })
    .unwrap();
    out
}

fn append(dir: &std::path::Path, target: &str, family: &str, cols: &[&str], out: &str) -> panel_append::AppendReport {
    panel_append::append_family(&AppendOptions {
        target: target.to_string(),
        family: family.to_string(),
        columns: cols.iter().map(|s| s.to_string()).collect(),
        skip_existing: false,
        out: out.to_string(),
        panel_version: "2".to_string(),
        scratch: Some(path(dir, "scratch")),
        locked_test: rcsr::lockbox::Keep::Unset,
    })
    .unwrap()
}

#[test]
fn a_time_major_family_is_appended_without_changing_existing_columns() {
    let dir = tmp_dir("append");
    let symbols = ["AAA", "BBB", "CCC"];
    let sym_major = bar_rows(&symbols, 150);
    let panel_rows = time_major(&sym_major);
    let target = path(&dir, "panel.rcsb");
    write_rows(&target, &panel_rows, 37);
    let base_schema = registry::write_schema(&target, "2");
    // `symbol` rows hold dictionary codes; the base panel schema must exist.
    base_schema.unwrap();
    let before_hashes = registry::read_hashes(&registry::schema_path(&target)).unwrap();
    let family = enrich_family(&dir, "fam", &sym_major, "r1,r4,ma5,vol20");

    let out = path(&dir, "panel_plus.rcsb");
    let report = append(&dir, &target, &family, &["r4", "ma5", "vol20"], &out);
    assert_eq!(report.input_sha256_before, report.input_sha256_after);
    assert_eq!(report.unmatched_rows, 0);
    assert_eq!(report.output_sha256, rcsr::manifest::sha256_file(&out).unwrap());

    // Existing per-column hashes are unchanged and the schema covers the new ones.
    let after_hashes = registry::read_hashes(&report.schema).unwrap();
    for (name, h) in &before_hashes {
        assert_eq!(after_hashes.get(name), Some(h), "{name} changed");
    }
    for n in ["r4", "ma5", "vol20"] {
        assert!(after_hashes.contains_key(n), "{n} missing from the schema");
    }
    assert!(registry::verify_schema(&out).unwrap().is_empty());

    // Values equal the symbol-major family at the same (symbol, ts).
    let ts = column(&out, "ts");
    let sym = FileStore::open(&out).unwrap();
    let levels = sym.header().levels(sym.header().column_index("symbol").unwrap()).unwrap().to_vec();
    let mut codes = Vec::new();
    let sc = sym.header().column_index("symbol").unwrap();
    for g in 0..sym.n_groups() {
        codes.extend(sym.read_codes(g, sc).unwrap());
    }
    let fam_r4 = column(&family, "r4");
    let fam_ts = column(&family, "ts");
    let fam_sym_rows: Vec<&str> = sym_major.iter().map(|r| r.1.as_str()).collect();
    let got = column(&out, "r4");
    let mut checked = 0usize;
    for (i, v) in got.iter().enumerate() {
        let s = levels[codes[i] as usize].as_str();
        let j = (0..fam_ts.len())
            .find(|&j| fam_sym_rows[j] == s && fam_ts[j] == ts[i])
            .unwrap();
        assert_eq!(v.to_bits(), fam_r4[j].to_bits(), "row {i}");
        checked += 1;
    }
    assert_eq!(checked, panel_rows.len());

    // An unregistered or absent column is refused.
    let err = panel_append::append_family(&AppendOptions {
        target: target.clone(),
        family: family.clone(),
        columns: vec!["vol21".to_string()],
        skip_existing: false,
        out: path(&dir, "x.rcsb"),
        panel_version: "2".to_string(),
        scratch: Some(path(&dir, "scratch")),
        locked_test: rcsr::lockbox::Keep::Unset,
    })
    .unwrap_err();
    assert!(err.contains("unknown column 'vol21'") && err.contains("family"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_appended_family_is_causal_on_a_prefix() {
    let dir = tmp_dir("append-prefix");
    let symbols = ["AAA", "BBB"];
    let full_rows = bar_rows(&symbols, 160);
    let cut = 100usize;
    let prefix_rows: Vec<_> = full_rows.iter().filter(|r| r.0 < 1_700_000_000 + cut as i64 * 3600).cloned().collect();
    let mut outs = Vec::new();
    for (tag, rows) in [("full", &full_rows), ("prefix", &prefix_rows)] {
        let target = path(&dir, &format!("{tag}-panel.rcsb"));
        write_rows(&target, &time_major(rows), 29);
        let family = enrich_family(&dir, tag, rows, "r1,r4,ma5,vol20");
        let out = path(&dir, &format!("{tag}-out.rcsb"));
        append(&dir, &target, &family, &["r1", "r4", "ma5", "vol20"], &out);
        outs.push(out);
    }
    for name in ["r1", "r4", "ma5", "vol20"] {
        let a = column(&outs[0], name);
        let b = column(&outs[1], name);
        let ts_a = column(&outs[0], "ts");
        let ts_b = column(&outs[1], "ts");
        let mut matched = 0;
        for (i, tb) in ts_b.iter().enumerate() {
            // Rows are time-major with symbols in a fixed order, so the same
            // (ts, symbol) sits at the same ordinal within its timestamp.
            let j = i;
            assert_eq!(ts_a[j], *tb);
            assert_eq!(a[j].is_nan(), b[i].is_nan(), "{name} row {i}");
            if a[j].is_finite() {
                assert!((a[j] - b[i]).abs() <= 1e-12, "{name} row {i}");
            }
            matched += 1;
        }
        assert_eq!(matched, prefix_rows.len());
    }
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Prefix causality for every registered derived column
// ---------------------------------------------------------------------------

fn panel_rows21(n: usize) -> Vec<(i64, String, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for k in 0..21usize {
        let sym = if k == 0 { "BTCUSDT".to_string() } else { format!("ALT{:02}USDT", k) };
        let mu = 0.0002 * (k as f64 - 10.0);
        for i in 0..n {
            let p = 100.0 * (mu * i as f64 + 0.03 * (i as f64 * 0.23 + k as f64 * 0.41).sin()).exp();
            let o = p * 0.999;
            let v = 1000.0 + 10.0 * k as f64 + (i % 7) as f64;
            let qv = v * p;
            rows.push((
                i as i64 * 3600,
                sym.clone(),
                o,
                p * 1.002,
                p * 0.998,
                p,
                v,
                qv,
                50.0 + (i % 11) as f64,
                v * 0.55,
                v * 0.55 * p,
                0.0,
            ));
        }
    }
    rows
}

fn write_panel21(p: &str, rows: &[(i64, String, f64, f64, f64, f64, f64, f64, f64, f64, f64, f64)]) {
    let mut text = String::from(
        "ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote,member_asof\n",
    );
    for (t, s, o, h, l, c, v, qv, n, tb, tbq, a) in rows {
        text.push_str(&format!("{},{},{},{},{},{},{},{},{},{},{},{}\n", t, s, o, h, l, c, v, qv, n, tb, tbq, a));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(p, &table, 64, Some(0)).unwrap();
}

#[test]
fn every_registered_available_column_is_causal_on_a_prefix() {
    let dir = tmp_dir("registry-causal");
    let n = 700usize;
    let prefix = 520usize;
    let full_rows = panel_rows21(n);
    let mut pre_rows = Vec::new();
    for k in 0..21usize {
        pre_rows.extend_from_slice(&full_rows[k * n..k * n + prefix]);
    }
    let full_in = path(&dir, "full.rcsb");
    let pre_in = path(&dir, "pre.rcsb");
    write_panel21(&full_in, &full_rows);
    write_panel21(&pre_in, &pre_rows);
    let names: Vec<String> = feature_names()
        .into_iter()
        .filter(|n| !["ts", "symbol", "listing_id", "member_asof"].contains(&n.as_str()))
        .collect();
    let run = |input: &str, out: &str| {
        let mut o = Options {
            data: input.to_string(),
            out: Some(out.to_string()),
            features: vec![names.join(",")],
            ..Default::default()
        };
        o.eth_symbol = "BTCUSDT".to_string();
        enrich::build(&o).unwrap();
    };
    let full_out = path(&dir, "full_out.rcsb");
    let pre_out = path(&dir, "pre_out.rcsb");
    run(&full_in, &full_out);
    run(&pre_in, &pre_out);
    let mut compared = 0usize;
    for name in &names {
        let a = column(&full_out, name);
        let b = column(&pre_out, name);
        for k in 0..21usize {
            for i in 0..prefix {
                let x = a[k * n + i];
                let y = b[k * prefix + i];
                assert_eq!(x.is_nan(), y.is_nan(), "{name} NaN mask k={k} i={i}");
                if x.is_finite() {
                    assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0), "{name} k={k} i={i}: {x} vs {y}");
                    compared += 1;
                }
            }
        }
    }
    eprintln!("registered columns checked: {}, finite values compared: {}", names.len(), compared);
    assert!(compared > 100_000);
    let _ = std::fs::remove_dir_all(&dir);
}

// ---------------------------------------------------------------------------
// Lag menu
// ---------------------------------------------------------------------------

fn lag_table() -> data::Table {
    let mut text = String::from("ts,g,x,z\n");
    for gi in 0..2 {
        for i in 0..200 {
            text.push_str(&format!("{},g{},{},{}\n", 1_700_000_000i64 + i as i64 * 3600, gi, i as f64 + 1000.0 * gi as f64, (i * i) as f64));
        }
    }
    data::parse_table_with_time(&text, false, Some("ts")).unwrap()
}

fn lag_spec(lags: usize) -> LagSpec {
    LagSpec {
        lags,
        history_frac: 0.0,
        gap_break: 0.0,
        time: TimeSpec {
            date_only: data::DateOnlyPolicy::Error,
            unit: None,
            tz: None,
        },
    }
}

#[test]
fn the_lag_menu_adds_only_the_declared_delays_and_reports_nvars() {
    let t = lag_table();
    let vars = vec!["x".to_string(), "z".to_string()];
    let menu = registry::parse_lag_menu("default").unwrap();
    assert_eq!(menu, vec![1, 4, 24, 96]);
    let (out, names, rep) = data::add_lags_menu(&t, &vars, &lag_spec(0), &menu, Some(0), Some(1)).unwrap();
    assert_eq!(rep.nvars_before, 2);
    assert_eq!(rep.nvars_after, 10);
    assert_eq!(names.len(), 10);
    assert!(names.contains(&"x_lag24".to_string()) && !names.contains(&"x_lag2".to_string()));
    // 96 bars of history per series are dropped; each lag is the value k bars back.
    assert_eq!(out.len(), 2 * (200 - 96));
    let x = out.index_of("x").unwrap();
    for k in menu {
        let l = out.index_of(&format!("x_lag{}", k)).unwrap();
        for r in 0..out.len() {
            assert_eq!(out.value(r, l), out.value(r, x) - k as f64, "lag {k} row {r}");
        }
    }
    // The blanket --lags 96 would add 192 columns; the menu adds 8.
    let (_, blanket, rep96) = data::add_lags(&t, &vars, &lag_spec(96), Some(0), Some(1)).unwrap();
    assert_eq!(blanket.len(), 2 + 192);
    assert_eq!(rep96.nvars_after, 194);
    assert!(registry::parse_lag_menu("0,4").is_err());
    assert!(registry::parse_lag_menu("a").is_err());
    assert_eq!(registry::parse_lag_menu("24, 4,4,1").unwrap(), vec![1, 4, 24]);
}

#[test]
fn lags_and_lag_set_report_nvars_on_the_command_line() {
    let dir = tmp_dir("lagcli");
    let mut text = String::from("ts,symbol,y,x0,x1\n");
    for r in 0..600 {
        let sym = if r % 2 == 0 { "AAA" } else { "BBB" };
        text.push_str(&format!(
            "{},{},{},{},{}\n",
            1_700_000_000i64 + (r as i64 / 2) * 3600,
            sym,
            (r as f64 * 0.07).sin(),
            (r % 17) as f64,
            (r % 5) as f64
        ));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let p = path(&dir, "lags.rcsb");
    store::write_store_with_time(&p, &table, 64, Some(0)).unwrap();
    let run = |extra: &[&str]| {
        let mut args = vec![
            "--data", &p, "--target", "y", "--vars", "x0,x1", "--time", "ts", "--group-by", "symbol",
            "--generations", "1", "--pop", "8", "--cpu",
        ];
        args.extend_from_slice(extra);
        Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().unwrap()
    };
    let menu = run(&["--lag-set", "1,4,24,96"]);
    assert!(menu.status.success(), "{}", String::from_utf8_lossy(&menu.stderr));
    assert!(String::from_utf8_lossy(&menu.stderr).contains("nvars 2 -> 10"), "{}", String::from_utf8_lossy(&menu.stderr));
    let blanket = run(&["--lags", "4"]);
    assert!(blanket.status.success());
    assert!(String::from_utf8_lossy(&blanket.stderr).contains("nvars 2 -> 10"));
    let both = run(&["--lags", "4", "--lag-set", "1,4"]);
    assert!(!both.status.success());
    let _ = std::fs::remove_dir_all(&dir);
}

/// A projection may leave a column out, but it records the omission and the
/// reason, and the registry still lists the column.
#[test]
fn a_view_records_every_omitted_column_and_the_registry_still_lists_it() {
    let mut names = feature_names();
    names.extend(["ts", "close", "flags", "not_a_feature"].map(String::from));
    names.push("r16_lag4".to_string());
    let v = registry::select_view(&names, 36, 7, 0).unwrap();
    assert_eq!(v.columns.len() + v.omitted.len(), names.len(), "every candidate is chosen or omitted");
    let mut seen = std::collections::BTreeSet::new();
    for (name, family, reason) in &v.omitted {
        assert!(!v.columns.contains(name), "{name} both chosen and omitted");
        assert!(seen.insert(name.clone()), "{name} omitted twice");
        assert!(!reason.is_empty() && !family.is_empty());
        if name == "not_a_feature" {
            assert_eq!(family, "unregistered");
        } else {
            let r = registry::lookup(name).unwrap_or_else(|| panic!("{name} left the registry"));
            assert_eq!(&r.family, family);
        }
    }
    let reason_of = |n: &str| v.omitted.iter().find(|(x, _, _)| x == n).map(|(_, _, r)| r.clone());
    assert!(reason_of("close").unwrap().contains("raw input"));
    assert!(reason_of("flags").unwrap().contains("raw input"));
    assert!(reason_of("not_a_feature").unwrap().contains("not in the feature registry"));
    assert!(v.omitted.iter().any(|(_, _, r)| r.contains("not drawn by view 0")));
    let json = v.json();
    assert!(json.contains("\"omitted\":[{\"name\":"), "{json}");
    assert!(json.contains("\"reason\":"), "{json}");
}

fn construction(col: &str) -> Vec<(String, String)> {
    vec![(col.to_string(), "construction column of lbl_h1_y".to_string())]
}

#[test]
fn project_never_draws_construction_columns() {
    let names = feature_names();
    let scale = "vol96".to_string();
    assert!(names.contains(&scale));
    let excluded = construction(&scale);
    let mut drawn_without = false;
    for seed in 0..100u64 {
        for view in 0..4usize {
            let plain = registry::select_view(&names, 36, seed, view).unwrap();
            drawn_without |= plain.columns.contains(&scale);
            let v = registry::select_view_excluding(&names, 36, seed, view, &excluded).unwrap();
            assert!(!v.columns.contains(&scale), "seed {seed} view {view}");
            assert!(!v.families.iter().any(|(_, c)| c.contains(&scale)));
            let row = v.omitted.iter().find(|(n, _, _)| *n == scale).expect("listed as omitted");
            assert_eq!(row.2, "construction column of lbl_h1_y");
            assert_eq!(row.1, "vol");
        }
    }
    assert!(drawn_without, "the unfiltered sweep never draws the scale column");
}

#[test]
fn view_records_requested_and_delivered_sizes() {
    let names = feature_names();
    let mut seen = std::collections::BTreeSet::new();
    let mut sixteen: Vec<String> = Vec::new();
    for n in &names {
        if let Some(r) = registry::lookup(n) {
            if r.available && !r.input && r.causal_offset <= 0 {
                if seen.contains(&r.family) || seen.len() < 16 {
                    seen.insert(r.family);
                    sixteen.push(n.clone());
                }
            }
        }
    }
    assert_eq!(seen.len(), 16);
    let v = registry::select_view(&sixteen, 36, 11, 0).unwrap();
    assert_eq!(v.delivered, v.columns.len());
    assert_eq!(v.delivered, 36);
    let json = v.json();
    assert!(json.contains("\"requested\":36"), "{json}");
    assert!(json.contains(&format!("\"delivered\":{}", v.delivered)), "{json}");
    let full = registry::select_view(&names, 36, 11, 0).unwrap();
    assert_eq!(full.delivered, full.columns.len());
}

struct AppendFixture {
    dir: std::path::PathBuf,
    panel: String,
    family: String,
}

fn append_fixture(tag: &str, family_close_shift: Option<(usize, f64)>) -> AppendFixture {
    let dir = tmp_dir(tag);
    let symbols = ["AAA", "BBB"];
    let sym_major = bar_rows(&symbols, 60);
    let panel = path(&dir, "panel.rcsb");
    write_rows(&panel, &time_major(&sym_major), 37);
    registry::write_schema(&panel, "2").unwrap();
    let mut fam_rows = sym_major.clone();
    if let Some((row, d)) = family_close_shift {
        fam_rows[row].5 += d;
    }
    let family = enrich_family(&dir, "fam", &fam_rows, "r1,r4,ma5,vol20");
    AppendFixture { dir, panel, family }
}

fn options(f: &AppendFixture, cols: &[&str], skip_existing: bool, out: &str) -> AppendOptions {
    AppendOptions {
        target: f.panel.clone(),
        family: f.family.clone(),
        columns: cols.iter().map(|s| s.to_string()).collect(),
        skip_existing,
        out: path(&f.dir, out),
        panel_version: "2".to_string(),
        scratch: Some(path(&f.dir, "scratch")),
        locked_test: rcsr::lockbox::Keep::Unset,
    }
}

fn leftovers(dir: &std::path::Path) -> Vec<String> {
    let mut found = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                stack.push(p);
            } else if name.starts_with("rcsr-append-") || name.contains(".tmp-") {
                found.push(name);
            }
        }
    }
    found
}

#[test]
fn a_duplicate_name_fails_before_any_column_is_read() {
    let f = append_fixture("append-dup", None);
    let err = panel_append::append_family(&options(&f, &["close", "r4", "volume", "nope"], false, "out.rcsb")).unwrap_err();
    assert!(err.contains("column 'close' already exists in the panel"), "{err}");
    assert!(err.contains("column 'volume' already exists in the panel"), "{err}");
    assert!(err.contains("unknown column 'nope'") && err.contains("column 'nope' is not in the family store"), "{err}");
    assert!(!err.contains("--replace"), "{err}");
    assert!(!std::path::Path::new(&path(&f.dir, "scratch")).exists(), "scratch dir created before the name check");
    assert!(!std::path::Path::new(&path(&f.dir, "out.rcsb")).exists());

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["panel", "append", "--data", &f.panel, "--family", &f.family, "--columns", "close,r4,volume", "--out"])
        .arg(path(&f.dir, "cli.rcsb"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stderr);
    assert!(text.contains("'close'") && text.contains("'volume'"), "{text}");
    assert!(!std::path::Path::new(&path(&f.dir, "cli.rcsb")).exists());
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn failed_append_leaves_no_output() {
    let f = append_fixture("append-atomic", Some((7, 0.5)));
    let out = path(&f.dir, "out.rcsb");
    let err = panel_append::append_family(&options(&f, &["close", "r4"], true, "out.rcsb")).unwrap_err();
    assert!(err.contains("another value at row") && err.contains("'close'"), "{err}");
    assert!(!std::path::Path::new(&out).exists());
    assert_eq!(leftovers(&f.dir), Vec::<String>::new());

    // a failure inside the writer, after the scratch columns exist
    let mut o = options(&f, &["r4"], false, "out.rcsb");
    o.out = f.panel.clone();
    let err = panel_append::append_family(&o).unwrap_err();
    assert!(err.contains("in place"), "{err}");
    assert_eq!(leftovers(&f.dir), Vec::<String>::new());
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn the_store_appender_leaves_no_file_on_an_error() {
    let f = append_fixture("append-store", None);
    let source = FileStore::open(&f.panel).unwrap();
    let bad = std::path::PathBuf::from(path(&f.dir, "short.f64"));
    std::fs::write(&bad, [0u8; 8]).unwrap();
    let out = path(&f.dir, "out.rcsb");
    let extras = [store::AppendColumn { name: "x".to_string(), file: bad, replace: false }];
    assert!(store::write_store_appending_hashed(&out, &source, &extras).is_err());
    assert!(!std::path::Path::new(&out).exists());
    assert_eq!(leftovers(&f.dir), Vec::<String>::new());
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn skip_existing_drops_equal_columns_and_appends_the_rest() {
    let f = append_fixture("append-skip", None);
    let report = panel_append::append_family(&options(&f, &["close", "r4", "volume"], true, "out.rcsb")).unwrap();
    assert_eq!(report.skipped_existing, vec!["close".to_string(), "volume".to_string()]);
    assert_eq!(report.appended, vec!["r4".to_string()]);
    let out = FileStore::open(&path(&f.dir, "out.rcsb")).unwrap();
    assert_eq!(out.n_cols(), FileStore::open(&f.panel).unwrap().n_cols() + 1);
    assert!(leftovers(&f.dir).is_empty());
    let err = panel_append::append_family(&options(&f, &["close"], true, "out2.rcsb")).unwrap_err();
    assert!(err.contains("nothing to append"), "{err}");
    let _ = std::fs::remove_dir_all(&f.dir);
}

#[test]
fn all_resolves_to_the_family_columns_the_panel_lacks() {
    let f = append_fixture("append-all", None);
    let panel = FileStore::open(&f.panel).unwrap();
    let family = FileStore::open(&f.family).unwrap();
    let want: Vec<String> = family
        .header()
        .names()
        .iter()
        .filter(|n| panel.header().column_index(n).is_none())
        .cloned()
        .collect();
    assert!(want.contains(&"r4".to_string()));
    assert_eq!(panel_append::resolve_all(&panel, &family), want);

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["panel", "append", "--data", &f.panel, "--family", &f.family, "--columns", "@all", "--out"])
        .arg(path(&f.dir, "all.rcsb"))
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let got = FileStore::open(&path(&f.dir, "all.rcsb")).unwrap();
    for n in &want {
        assert!(got.header().column_index(n).is_some(), "{n}");
    }
    let _ = std::fs::remove_dir_all(&f.dir);
}

fn timeframe_store(dir: &std::path::Path, file: &str, steps: &[i64]) -> String {
    let base = feature_names().into_iter().find(|n| registry::lookup(n).is_some_and(|r| r.timeframe == "15m")).unwrap();
    let mtf = enrich::mtf_names().remove(0);
    let mut text = format!("ts,symbol,{},{},gap\n", base, mtf);
    let mut ts = 1_700_000_000i64;
    for (r, step) in steps.iter().enumerate() {
        ts += step;
        text.push_str(&format!("{},AAA,{},{},{}\n", ts, r as f64, r as f64 * 0.5, r % 3));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let p = path(dir, file);
    store::write_store_with_time(&p, &table, 50, Some(0)).unwrap();
    p
}

fn column_timeframes(body: &str) -> Vec<(String, String)> {
    body.lines()
        .filter(|l| l.starts_with("{\"name\":"))
        .map(|l| {
            let name = l.split("\"name\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
            let tf = l.split("\"timeframe\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
            (name, tf)
        })
        .collect()
}

#[test]
fn schema_timeframe_matches_hourly_store() {
    let dir = tmp_dir("schema_tf_hourly");
    let p = timeframe_store(&dir, "hourly.rcsb", &[3600; 120]);
    let body = std::fs::read_to_string(registry::write_schema(&p, "2").unwrap()).unwrap();
    assert!(body.contains("\"bar_seconds\":3600,"), "{body}");
    let tfs = column_timeframes(&body);
    let mtf = enrich::mtf_names().remove(0);
    for (name, tf) in &tfs {
        if *name == mtf {
            assert_eq!(tf, mtf.split('_').nth(1).unwrap(), "{name}");
        } else if name != "ts" && name != "symbol" {
            assert_eq!(tf, "1h", "{name}");
        }
    }
    assert!(!body.contains("\"timeframe\":\"15m\""), "{body}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn schema_timeframe_of_a_15m_store_is_15m() {
    let dir = tmp_dir("schema_tf_15m");
    let p = timeframe_store(&dir, "m15.rcsb", &[900; 120]);
    let body = std::fs::read_to_string(registry::write_schema(&p, "2").unwrap()).unwrap();
    assert!(body.contains("\"bar_seconds\":900,"), "{body}");
    let mtf = enrich::mtf_names().remove(0);
    for (name, tf) in column_timeframes(&body) {
        if name == "gap" {
            assert_eq!(tf, "15m");
        } else if name == mtf {
            assert_ne!(tf, "15m");
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn schema_timeframe_edge_cases_single_row_and_mixed_steps() {
    let dir = tmp_dir("schema_tf_edge");
    let one = timeframe_store(&dir, "one.rcsb", &[3600]);
    let body = std::fs::read_to_string(registry::write_schema(&one, "2").unwrap()).unwrap();
    assert!(body.contains("\"bar_seconds\":null,"), "{body}");
    assert!(column_timeframes(&body).iter().any(|(n, tf)| n == "gap" && tf == "unknown"), "{body}");

    let mut steps = vec![3600i64; 90];
    steps.extend([7200; 10]);
    let mixed = timeframe_store(&dir, "mixed.rcsb", &steps);
    let body = std::fs::read_to_string(registry::write_schema(&mixed, "2").unwrap()).unwrap();
    assert!(body.contains("\"bar_seconds\":3600,\"bar_seconds_modal_share\":0.8989"), "{body}");

    let odd = timeframe_store(&dir, "odd.rcsb", &[1234; 20]);
    let body = std::fs::read_to_string(registry::write_schema(&odd, "2").unwrap()).unwrap();
    assert!(column_timeframes(&body).iter().any(|(n, tf)| n == "gap" && tf == "1234s"), "{body}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn rewriting_an_old_sidecar_keeps_hashes_and_verifies() {
    let dir = tmp_dir("schema_tf_rewrite");
    let p = timeframe_store(&dir, "hourly.rcsb", &[3600; 120]);
    let schema = registry::write_schema(&p, "2").unwrap();
    let new = std::fs::read_to_string(&schema).unwrap();
    // The pre-change form: 15m labels, no bar_seconds in the header.
    let old = new.replace("\"bar_seconds\":3600,", "").replace("\"timeframe\":\"1h\"", "\"timeframe\":\"15m\"");
    assert_ne!(old, new);
    std::fs::write(&schema, &old).unwrap();
    assert!(registry::verify_schema(&p).unwrap().is_empty());
    let before = registry::read_hashes(&schema).unwrap();
    registry::write_schema(&p, "2").unwrap();
    assert_eq!(registry::read_hashes(&schema).unwrap(), before);
    assert_eq!(std::fs::read_to_string(&schema).unwrap(), new);
    assert!(registry::verify_schema(&p).unwrap().is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Every available derived column of the first `n` registry families.
fn family_columns(n: usize) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    let mut out: Vec<String> = Vec::new();
    for name in feature_names() {
        if let Some(r) = registry::lookup(&name) {
            if r.available && !r.input && r.causal_offset <= 0 && (seen.contains(&r.family) || seen.len() < n) {
                seen.insert(r.family);
                out.push(name);
            }
        }
    }
    assert_eq!(seen.len(), n);
    out
}

#[test]
fn project_delivers_the_requested_size() {
    let sixteen = family_columns(16);
    assert!(sixteen.len() >= 60, "{}", sixteen.len());
    for size in [33usize, 36, 40, 48] {
        let v = registry::select_view(&sixteen, size, 11, 0).unwrap();
        assert_eq!(v.delivered, size);
        assert_eq!(v.columns.len(), size);
        assert!(v.json().contains(&format!("\"requested\":{size},\"delivered\":{size}")));
    }
    let views: Vec<Vec<String>> = (0..4).map(|k| registry::select_view(&sixteen, 48, 11, k).unwrap().columns).collect();
    for i in 0..4 {
        for j in i + 1..4 {
            assert_ne!(views[i], views[j], "views {i} and {j} are equal");
        }
    }
}

#[test]
fn project_refuses_a_store_with_fewer_candidate_columns_than_requested() {
    let all = family_columns(16);
    let thirty: Vec<String> = all.into_iter().take(30).collect();
    let err = registry::select_view(&thirty, 48, 11, 0).unwrap_err();
    assert!(err.contains("30 candidate columns"), "{err}");
    assert!(registry::select_view(&thirty, 30, 11, 0).is_ok());
    let dir = tmp_dir("shortfall");
    let (p, _) = registry_store(&dir, 30, 64, "short.rcsb");
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", &p, "--target", "y", "--project", "48", "--time", "ts", "--group-by", "symbol", "--cpu"])
        .output()
        .unwrap();
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("30 candidate columns"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn view_ids_of_fully_delivered_sizes_are_pinned() {
    let sixteen = family_columns(16);
    let got: Vec<String> = [24usize, 28, 32]
        .iter()
        .flat_map(|&s| (0..2).map(move |k| (s, k)))
        .map(|(s, k)| registry::select_view(&sixteen, s, 11, k).unwrap().id)
        .collect();
    let golden: [&str; 6] = ["2abdb5320a08", "c498b629e472", "41f0170017d4", "8ed29caee6cb", "a32035702727", "558b4954cb4b"];
    assert_eq!(got, golden, "{got:?}");
}
