//! #294: projected reads, identity and the no-silent-drop contract.

use rcsr::data;
use rcsr::memory::{EstimateOptions, StoreShape};
use rcsr::store::{self, DType, FileStore, ProjectedGroup, Projection};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-projection-{}-{}-{}",
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
    let dir = tmp_dir("identity");
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
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn projected_values_are_bit_identical_to_the_full_read() {
    let dir = tmp_dir("values");
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
    let _ = std::fs::remove_dir_all(&dir);
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
