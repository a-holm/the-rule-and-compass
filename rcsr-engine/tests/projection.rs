//! #294: projected reads, identity and the no-silent-drop contract.

use rcsr::data;
use rcsr::memory::{EstimateOptions, StoreShape};
use rcsr::store::{self, DType, FileStore, ProjectedGroup, Projection};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn tmp_dir(tag: &str) -> (Scratch, std::path::PathBuf) {
    let scratch = Scratch::new(tag);
    let path = scratch.path().to_path_buf();
    (scratch, path)
}

/// A 364-column store: `ts`, `symbol` and 362 numeric columns. The first 20
/// numerics are the projection variables.
fn wide_store(dir: &std::path::Path, rows: usize) -> (String, Vec<String>) {
    let mut header = String::from("ts,symbol");
    for c in 0..362 {
        header.push_str(&format!(",v{}", c));
    }
    header.push('\n');
    let mut text = header;
    for r in 0..rows {
        let ts = 1_700_000_000i64 + r as i64 * 3600;
        let sym = if r % 2 == 0 { "AAA" } else { "BBB" };
        text.push_str(&format!("{},{}", ts, sym));
        for c in 0..362 {
            // Deterministic, finite, distinct per column and row.
            let v = (c as f64 + 1.0) * 0.25 + (r % 977) as f64 * 0.001;
            text.push_str(&format!(",{}", v));
        }
        text.push('\n');
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = dir.join("wide.rcsb").to_string_lossy().into_owned();
    store::write_store_with_time(&path, &table, 4096, Some(0)).unwrap();
    let vars: Vec<String> = (0..20).map(|c| format!("v{}", c)).collect();
    (path, vars)
}

fn projected_names(vars: &[String]) -> Vec<String> {
    let mut names = vec!["ts".to_string()];
    names.extend(vars.iter().cloned());
    names
}

#[test]
fn projection_identity_is_canonical_and_never_drops_a_name() {
    let (_scratch, dir) = tmp_dir("identity");
    let (path, vars) = wide_store(&dir, 32);
    let fs = FileStore::open(&path).unwrap();

    let a = Projection::resolve_owned(&fs, &projected_names(&vars)).unwrap();
    let mut reversed = projected_names(&vars);
    reversed.reverse();
    let b = Projection::resolve_owned(&fs, &reversed).unwrap();
    assert_eq!(a.hash(), b.hash(), "request order must not change the hash");
    assert_eq!(a.indices(), b.indices(), "indices are in header order");
    assert_eq!(a.names(), b.names());
    assert_eq!(a.len(), 21);
    assert_eq!(a.hash_hex().len(), 16);

    // Duplicates collapse because the same header column is read once.
    let mut dup = projected_names(&vars);
    dup.push("v3".to_string());
    let c = Projection::resolve_owned(&fs, &dup).unwrap();
    assert_eq!(c.hash(), a.hash());
    assert_eq!(c.len(), a.len());

    // A name that does not exist is a hard error that names it: no silent
    // truncation of the requested list.
    let mut missing = projected_names(&vars);
    missing.push("not_a_column".to_string());
    let e = Projection::resolve_owned(&fs, &missing).unwrap_err();
    assert!(e.contains("not_a_column"), "{e}");
    assert!(e.contains("refusing to drop it silently"), "{e}");

    // A text/dictionary column can be carried for the split, but the numeric
    // device form refuses it: no value block means no arithmetic.
    let with_symbol = Projection::resolve(&fs, &["ts", "symbol"]).unwrap();
    assert_eq!(with_symbol.len(), 2);
    assert_eq!(with_symbol.columns()[1].width, 0);
    let table = fs.to_table_projected(&with_symbol).unwrap();
    assert!(!table.numeric[1]);
    assert_eq!(table.dict(1).unwrap().codes.len(), fs.n_rows());
    let e = Projection::resolve_numeric(&fs, &["ts", "symbol"]).unwrap_err();
    assert!(e.contains("symbol"), "{e}");
    assert!(e.contains("dictionary"), "{e}");

    // Explicit index resolution is the same projection.
    let d = Projection::from_indices(&fs, &a.indices()).unwrap();
    assert_eq!(d.hash(), a.hash());
}

#[test]
fn projected_values_are_bit_identical_to_the_full_read() {
    let (_scratch, dir) = tmp_dir("values");
    let (path, vars) = wide_store(&dir, 9_000);
    let fs = FileStore::open(&path).unwrap();
    let proj = Projection::resolve_owned(&fs, &projected_names(&vars)).unwrap();

    // Reference: the existing per-column seam, one block at a time.
    let mut reference: Vec<Vec<f64>> = Vec::new();
    for c in proj.indices() {
        let mut col = Vec::new();
        for g in 0..fs.n_groups() {
            col.extend(fs.read_f64_group(g, c).unwrap());
        }
        reference.push(col);
    }

    // Projected group reads, reusing one buffer, in projection order.
    let mut group = ProjectedGroup::new();
    let mut seen: Vec<Vec<f64>> = vec![Vec::new(); proj.len()];
    let mut bytes = 0u64;
    let mut columns = 0usize;
    for g in 0..fs.n_groups() {
        let stats = fs.read_group_projected(g, &proj, &mut group).unwrap();
        bytes = bytes.saturating_add(stats.value_bytes);
        columns = columns.saturating_add(stats.columns);
        for c in 0..proj.len() {
            seen[c].extend_from_slice(group.col(c));
        }
    }
    for (c, (a, b)) in reference.iter().zip(seen.iter()).enumerate() {
        let ab: Vec<u64> = a.iter().map(|v| v.to_bits()).collect();
        let bb: Vec<u64> = b.iter().map(|v| v.to_bits()).collect();
        assert_eq!(ab, bb, "projected column {} differs", c);
    }
    // Only projected columns were read: every group contributes exactly
    // projection.len() * group_len value bytes.
    assert_eq!(columns, fs.n_groups() * proj.len());
    assert_eq!(
        bytes,
        fs.n_rows() as u64 * proj.len() as u64 * 8,
        "the reader must not touch unprojected blocks"
    );

    // The resident projected table has exactly the projection, in order, and
    // the same bits.
    let table = fs.to_table_projected(&proj).unwrap();
    assert_eq!(table.names, proj.names());
    assert_eq!(table.n_cols(), proj.len());
    for (c, col) in reference.iter().enumerate() {
        let got: Vec<u64> = table.col(c).iter().map(|v| v.to_bits()).collect();
        let want: Vec<u64> = col.iter().map(|v| v.to_bits()).collect();
        assert_eq!(got, want, "table column {} differs", c);
    }

    // The time column keeps its exact i64 copy even though its f64 view is
    // lossy in principle.
    assert_eq!(proj.columns()[0].dtype, DType::I64);
    let ints = table.ints(0).expect("the time column keeps i64");
    assert_eq!(ints[0], 1_700_000_000);
    assert_eq!(ints[ints.len() - 1], 1_700_000_000 + (fs.n_rows() as i64 - 1) * 3600);
}

#[test]
fn the_364_to_20_estimate_is_under_eight_gigabytes() {
    // Header-derived arithmetic for the issue's acceptance cell: 364 columns,
    // 20 variables + time, 4.78 M rows (the epic's measured panel), 65 536-row
    // groups. The unprojected image is ~14 GB; the projection is ~0.8 GB.
    let rows = 4_780_000usize;
    let columns = (0..364)
        .map(|c| rcsr::memory::ColumnShape {
            name: format!("c{}", c),
            value_width: 8,
            int_width: if c == 0 { 8 } else { 0 },
            code_width: if c == 1 { 4 } else { 0 },
            dictionary_bytes: if c == 1 { 256 } else { 0 },
        })
        .collect();
    let shape = StoreShape {
        rows,
        group_rows: 65_536,
        header_len: 1 << 20,
        columns,
    };
    let opts = EstimateOptions {
        used_cols: 20,
        scan_cols: 1,
        projection: Some((0..21).collect()),
        gpu: true,
        ..EstimateOptions::default()
    };
    let e = shape.estimate(&opts);
    assert_eq!(e.projected_cols, 21);
    assert!(e.storage_bytes < 4 * 1024 * 1024 * 1024, "storage {}", e.storage_bytes);
    assert!(e.ring_bytes <= 64 * 1024 * 1024, "ring {}", e.ring_bytes);
    assert!(e.total_bytes < 8 * 1024 * 1024 * 1024, "total {}", e.total_bytes);
    // The unprojected estimate is far larger, which is the point of the run.
    let full = shape.estimate(&EstimateOptions {
        gpu: true,
        ..EstimateOptions::default()
    });
    assert!(full.storage_bytes > 8 * 1024 * 1024 * 1024);
}

/// 6 groups of 4096 rows: `ts` (i64), `symbol` (dictionary), `a`, `b` (f64).
fn grouped_store(dir: &std::path::Path) -> String {
    let rows = 5 * 4096 + 777;
    let mut text = String::from("ts,symbol,a,b\n");
    for r in 0..rows {
        let sym = ["AAA", "BBB", "CCC"][r % 3];
        text.push_str(&format!(
            "{},{},{},{}\n",
            1_700_000_000i64 + r as i64 * 3600,
            sym,
            r as f64 * 0.125 - 40.0,
            (r % 977) as f64 / 7.0
        ));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = dir.join("grouped.rcsb").to_string_lossy().into_owned();
    store::write_store_with_time(&path, &table, 4096, Some(0)).unwrap();
    path
}

#[test]
fn projected_table_is_identical_for_one_and_four_readers() {
    let (_scratch, dir) = tmp_dir("readers");
    let path = grouped_store(&dir);
    let fs = FileStore::open(&path).unwrap();
    assert!(fs.n_groups() >= 5);
    let proj = Projection::resolve(&fs, &["ts", "symbol", "a", "b"]).unwrap();
    assert_eq!(proj.columns()[0].dtype, DType::I64);
    assert_eq!(proj.columns()[1].dtype, DType::U32);

    let one = fs.to_table_projected_with_readers(&proj, 1).unwrap();
    for readers in [4usize, 64] {
        let many = fs.to_table_projected_with_readers(&proj, readers).unwrap();
        assert_eq!(one.names, many.names);
        assert_eq!(one.numeric, many.numeric);
        assert_eq!(one.len(), many.len());
        for c in 0..one.n_cols() {
            let a: Vec<u64> = one.col(c).iter().map(|v| v.to_bits()).collect();
            let b: Vec<u64> = many.col(c).iter().map(|v| v.to_bits()).collect();
            assert_eq!(a, b, "values of column {} differ at {} readers", c, readers);
            assert_eq!(one.ints(c), many.ints(c), "ints of column {}", c);
            assert_eq!(one.dict(c), many.dict(c), "codes of column {}", c);
        }
    }
    assert!(one.ints(0).is_some());
    assert_eq!(one.dict(1).unwrap().codes.len(), fs.n_rows());
    assert_eq!(one.dict(1).unwrap().codes[4], 4 % 3);
}

#[test]
fn store_readers_parses_the_environment_value() {
    assert_eq!(store::parse_store_readers(None), 4);
    assert_eq!(store::parse_store_readers(Some("1")), 1);
    assert_eq!(store::parse_store_readers(Some("8")), 8);
    assert_eq!(store::parse_store_readers(Some("0")), 1);
    assert_eq!(store::parse_store_readers(Some("1000")), 64);
    assert_eq!(store::parse_store_readers(Some("many")), 4);
}

/// Reference load: the group-at-a-time public reads copied into the columns,
/// the way the loader worked before it read blocks straight into the table.
fn reference_columns(fs: &FileStore, proj: &Projection) -> Vec<data::Column> {
    let n = fs.n_rows();
    let rows = fs.group_rows();
    let mut values: Vec<Vec<f64>> = Vec::new();
    let mut ints: Vec<Option<Vec<i64>>> = Vec::new();
    let mut codes: Vec<Option<Vec<u32>>> = Vec::new();
    for col in proj.columns() {
        values.push(if col.width == 0 { Vec::new() } else { vec![f64::NAN; n] });
        ints.push((col.dtype == DType::I64).then(|| vec![i64::MIN; n]));
        codes.push((col.dtype == DType::U32).then(|| vec![0u32; n]));
    }
    let mut group = ProjectedGroup::new();
    for g in 0..fs.n_groups() {
        fs.read_group_projected(g, proj, &mut group).unwrap();
        let base = g * rows;
        let len = group.len();
        for (c, col) in proj.columns().iter().enumerate() {
            if col.width > 0 {
                values[c][base..base + len].copy_from_slice(group.col(c));
            }
            if let Some(dst) = ints[c].as_mut() {
                dst[base..base + len].copy_from_slice(&fs.read_i64(g, col.index).unwrap());
            }
            if let Some(dst) = codes[c].as_mut() {
                dst[base..base + len].copy_from_slice(&fs.read_codes(g, col.index).unwrap());
            }
        }
    }
    (0..proj.len())
        .map(|c| data::Column {
            values: std::mem::take(&mut values[c]),
            ints: ints[c].take(),
            dict: codes[c].take().map(|codes| data::DictColumn {
                codes,
                levels: fs
                    .header()
                    .levels(proj.columns()[c].index)
                    .map(|l| l.to_vec())
                    .unwrap_or_default(),
            }),
        })
        .collect()
}

/// 3 full groups of 7 rows and a last group of 3. Columns: `ts` (i64 with
/// missing cells), `x` (f64 with special values), `sym` (text dictionary),
/// `cat` (numeric f64 with a dictionary), `y` (f64).
fn all_dtypes_store(dir: &std::path::Path) -> String {
    use store::{ColumnPlan, GroupColumn};
    let n = 24usize;
    let plans = vec![
        ColumnPlan { name: "ts".into(), numeric: true, dtype: DType::I64, levels: vec![], has_codes: false },
        ColumnPlan { name: "x".into(), numeric: true, dtype: DType::F64, levels: vec![], has_codes: false },
        ColumnPlan {
            name: "sym".into(),
            numeric: false,
            dtype: DType::U32,
            levels: vec!["AAA".into(), "BBB".into(), "CCC".into()],
            has_codes: true,
        },
        ColumnPlan {
            name: "cat".into(),
            numeric: true,
            dtype: DType::F64,
            levels: vec!["1".into(), "2".into()],
            has_codes: true,
        },
        ColumnPlan { name: "y".into(), numeric: true, dtype: DType::F64, levels: vec![], has_codes: false },
    ];
    let special = [
        f64::NAN,
        -0.0,
        0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_dead_beef),
        f64::MIN_POSITIVE / 4.0,
    ];
    let path = dir.join("all.rcsb").to_string_lossy().into_owned();
    store::write_store_grouped(&path, &plans, n, 7, |lo, hi, out: &mut [GroupColumn]| {
        for r in lo..hi {
            out[0].i64s.push(if r % 5 == 4 { i64::MIN } else { 1_700_000_000 + r as i64 * 60 });
            out[1].f64s.push(special[r % special.len()]);
            out[2].codes.push((r % 3) as u32);
            out[3].f64s.push((r % 2) as f64 + 1.0);
            out[3].codes.push((r % 2) as u32);
            out[4].f64s.push(r as f64 * 0.37 - 4.0);
        }
        Ok(())
    })
    .unwrap();
    path
}

#[test]
fn direct_load_equals_the_group_read_reference_for_every_dtype_and_projection() {
    let (_scratch, dir) = tmp_dir("alldtypes");
    let path = all_dtypes_store(&dir);
    let fs = FileStore::open(&path).unwrap();
    assert_eq!(fs.n_groups(), 4);
    let subsets: [&[&str]; 9] = [
        &["ts", "x", "sym", "cat", "y"],
        &["ts"],
        &["x"],
        &["sym"],
        &["cat", "sym"],
        &["y", "ts", "sym"],
        &["x", "y"],
        &["sym", "ts", "cat"],
        &["y"],
    ];
    for names in subsets {
        let proj = Projection::resolve(&fs, names).unwrap();
        let want = reference_columns(&fs, &proj);
        let numeric: Vec<bool> = proj.columns().iter().map(|c| c.dtype != DType::U32).collect();
        for readers in [1usize, 2, 4, 64] {
            let got = fs.to_table_projected_with_readers(&proj, readers).unwrap();
            let label = format!("{:?} at {} readers", names, readers);
            assert_eq!(proj.names(), got.names, "{}", label);
            assert_eq!(numeric, got.numeric, "{}", label);
            assert_eq!(fs.n_rows(), got.len(), "{}", label);
            for (c, w) in want.iter().enumerate() {
                let a: Vec<u64> = w.values.iter().map(|v| v.to_bits()).collect();
                let b: Vec<u64> = got.col(c).iter().map(|v| v.to_bits()).collect();
                assert_eq!(a, b, "values of column {}: {}", c, label);
                assert_eq!(w.ints.as_deref(), got.ints(c), "ints of column {}: {}", c, label);
                assert_eq!(w.dict.as_ref(), got.dict(c), "codes of column {}: {}", c, label);
            }
        }
    }
    let proj = Projection::resolve(&fs, &["ts", "x"]).unwrap();
    let t = fs.to_table_projected_with_readers(&proj, 1).unwrap();
    assert!(t.col(0)[4].is_nan());
    assert_eq!(t.ints(0).unwrap()[4], i64::MIN);
    assert_eq!(t.col(1)[5].to_bits(), 0x7ff8_0000_dead_beef);
}
