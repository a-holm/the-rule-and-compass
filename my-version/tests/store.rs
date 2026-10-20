//! Versioned column store, memory guard and out-of-core backing (issue #71).
//!
//! The fixtures are tiny on purpose: the large-file measurement lives in the
//! issue report, not in the test suite.

use rcsr::data::{self, Table};
use rcsr::store::{self, FileStore, MAGIC};
use std::process::Command;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-store-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn full_table() -> Table {
    data::parse_table(
        "x,label,flag,y\n\
         1,A,1,2\n\
         2,B,0,4\n\
         3,A,1,6\n\
         4,C,0,8\n\
         5,A,1,10\n\
         6,B,0,12\n",
    )
    .unwrap()
}

#[test]
fn roundtrip_preserves_numbers_and_dictionaries() {
    let table = full_table();
    let dir = temp_dir("roundtrip");
    let path = dir.join("t.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 4).unwrap();
    // Deterministic: the same input and group size produce the same bytes.
    let again = dir.join("t2.rcsb");
    store::write_store(again.to_str().unwrap(), &table, 4).unwrap();
    assert_eq!(
        std::fs::read(&path).unwrap(),
        std::fs::read(&again).unwrap()
    );

    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    assert_eq!(fs.n_rows(), 6);
    assert_eq!(fs.n_cols(), 4);
    assert_eq!(fs.group_rows(), 4);
    assert_eq!(fs.n_groups(), 2);
    let back = fs.to_table().unwrap();

    assert_eq!(back.names, table.names);
    assert_eq!(back.numeric, table.numeric);
    // Numeric columns are bit-identical, including a low-cardinality numeric
    // column that keeps its dictionary.
    for c in [0usize, 2, 3] {
        let a: Vec<u64> = table.col(c).iter().map(|v| v.to_bits()).collect();
        let b: Vec<u64> = back.col(c).iter().map(|v| v.to_bits()).collect();
        assert_eq!(a, b, "column {}", c);
    }
    // Text column: values stay absent, codes and levels survive.
    assert!(back.col(1).is_empty());
    assert_eq!(back.dict(1).unwrap().levels, table.dict(1).unwrap().levels);
    assert_eq!(back.dict(1).unwrap().codes, table.dict(1).unwrap().codes);
    // The low-cardinality numeric dictionary survives too.
    assert_eq!(back.dict(2).unwrap().levels, table.dict(2).unwrap().levels);
    assert_eq!(back.dict(2).unwrap().codes, table.dict(2).unwrap().codes);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_readable_hand_built_store_has_the_expected_groups() {
    // Two columns, five rows, groups of two. Offsets are relative to the data
    // section; layout is per group, per column, so x/y alternate.
    let header = "{\"format\":\"rcsr-column-store\",\"version\":1,\"n_rows\":5,\
                  \"n_cols\":2,\"group_rows\":2,\"columns\":[\
                  {\"name\":\"x\",\"numeric\":true,\"levels\":[],\"has_codes\":false,\
                   \"offsets\":[0,32,64],\"code_offsets\":[]},\
                  {\"name\":\"y\",\"numeric\":true,\"levels\":[],\"has_codes\":false,\
                   \"offsets\":[16,48,72],\"code_offsets\":[]}]}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in [1.0f64, 2.0, 10.0, 20.0, 3.0, 4.0, 30.0, 40.0, 5.0, 50.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let dir = temp_dir("hand");
    let path = dir.join("hand.rcsb");
    std::fs::write(&path, &bytes).unwrap();

    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let mut buf = rcsr::column::GroupBuf::new();
    let g0 = fs.load_group(0, &mut buf).unwrap();
    assert_eq!(g0.col(0), &[1.0, 2.0]);
    assert_eq!(g0.col(1), &[10.0, 20.0]);
    let g1 = fs.load_group(1, &mut buf).unwrap();
    assert_eq!(g1.col(0), &[3.0, 4.0]);
    assert_eq!(g1.col(1), &[30.0, 40.0]);
    let g2 = fs.load_group(2, &mut buf).unwrap();
    assert_eq!(g2.len(), 1, "the last group is short, not padded");
    assert_eq!(g2.col(0), &[5.0]);
    assert_eq!(g2.col(1), &[50.0]);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn version_mismatch_is_a_clear_error() {
    let dir = temp_dir("version");
    let path = dir.join("v.rcsb");
    store::write_store(path.to_str().unwrap(), &full_table(), 4).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes[8..12].copy_from_slice(&2u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = FileStore::open(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("version 2"), "{err}");
    assert!(err.contains("version 1"), "{err}");
    assert!(err.contains("rcsr import"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn truncated_data_is_refused_not_read_past_the_end() {
    let dir = temp_dir("truncated");
    let path = dir.join("t.rcsb");
    store::write_store(path.to_str().unwrap(), &full_table(), 2).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    std::fs::write(&path, &bytes[..bytes.len() - 4]).unwrap();
    let err = FileStore::open(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("truncated"), "{err}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sampling_never_materialises_more_than_the_cap_and_keeps_rows_aligned() {
    let dir = temp_dir("sample");
    let path = dir.join("s.rcsb");
    let mut csv = String::from("x,y\n");
    for i in 0..1000 {
        csv.push_str(&format!("{},{}\n", i, i * 2));
    }
    let table = data::parse_table(&csv).unwrap();
    store::write_store(path.to_str().unwrap(), &table, 7).unwrap();
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let sample = fs.sample_table(100, 7).unwrap();
    assert!(sample.len() > 0 && sample.len() <= 130, "sampled {}", sample.len());
    for r in 0..sample.len() {
        assert_eq!(sample.value(r, 1), sample.value(r, 0) * 2.0);
    }
    // Order is preserved (the scan is row order), so the sample is usable for
    // a temporal range without a shuffle.
    let mut prev = f64::NEG_INFINITY;
    for r in 0..sample.len() {
        assert!(sample.value(r, 0) > prev);
        prev = sample.value(r, 0);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn import_then_search_is_byte_identical_to_the_csv_path() {
    let dir = temp_dir("exact");
    let store_path = dir.join("clean.rcsb");
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "import",
            csv,
            store_path.to_str().unwrap(),
            "--group-rows",
            "17",
            "--quiet",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "import failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let from_csv = dir.join("csv.tsv");
    let from_store = dir.join("store.tsv");
    let common = |data: &str, out: &std::path::Path| {
        let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "--data",
                data,
                "--target",
                "y",
                "--vars",
                "x",
                "--seed",
                "3",
                "--islands",
                "1",
                "--pop",
                "60",
                "--generations",
                "20",
                "--out",
                out.to_str().unwrap(),
                "--quiet",
            ])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "search failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    };
    common(csv, &from_csv);
    common(store_path.to_str().unwrap(), &from_store);
    let a = std::fs::read(&from_csv).unwrap();
    let b = std::fs::read(&from_store).unwrap();
    assert_eq!(a, b, "the imported store changed the search trajectory");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn csv_shape_counts_header_width_and_data_rows() {
    let text = "# note\n\nx,y,z\n1,2,3\n\n4,5,6\n";
    assert_eq!(
        data::csv_shape(text),
        data::CsvShape {
            rows: 2,
            cols: 3,
            bytes: text.len() as u64,
        }
    );
    // The delimiter follows the loader: tab, comma, else whitespace.
    assert_eq!((data::csv_shape("a\tb\n1\t2\n").rows, data::csv_shape("a\tb\n1\t2\n").cols), (1, 2));
    assert_eq!(
        (data::csv_shape("a b c\n1 2 3\n4 5 6\n").rows, data::csv_shape("a b c\n1 2 3\n4 5 6\n").cols),
        (2, 3)
    );
}

#[test]
fn memory_estimate_follows_shape_not_file_size() {
    // A wide table with a *small* file: every cell is one byte.
    let mut wide = (0..90).map(|j| format!("c{}", j)).collect::<Vec<_>>().join(",");
    let wide_row = std::iter::repeat("1").take(90).collect::<Vec<_>>().join(",");
    for _ in 0..200 {
        wide.push('\n');
        wide.push_str(&wide_row);
    }
    // A narrow table with a *larger* file: two very long numeric cells.
    let long = "1".repeat(200);
    let mut narrow = String::from("x,y");
    for _ in 0..200 {
        narrow.push('\n');
        narrow.push_str(&long);
        narrow.push(',');
        narrow.push_str(&long);
    }
    assert!(
        wide.len() < narrow.len(),
        "the fixture must be wider but smaller: {} vs {}",
        wide.len(),
        narrow.len()
    );
    let wide_shape = data::csv_shape(&wide);
    let narrow_shape = data::csv_shape(&narrow);
    let wide_est = data::estimate_csv_bytes(wide_shape, 0);
    let narrow_est = data::estimate_csv_bytes(narrow_shape, 0);
    assert!(
        wide_est > narrow_est,
        "the estimate must be driven by rows x columns, not bytes: {wide_est} vs {narrow_est}"
    );
    // The old guard was exactly 33 x file size; neither estimate is that.
    assert_ne!(wide_est, wide_shape.bytes * 33);
    assert_ne!(narrow_est, narrow_shape.bytes * 33);
    // A projected search costs less preparation than the whole header.
    assert!(data::estimate_csv_bytes(wide_shape, 7) < wide_est);
}

#[test]
fn a_wide_small_csv_is_refused_before_a_narrow_large_one() {
    let dir = temp_dir("shape-guard");
    let mut wide = (0..90).map(|j| format!("c{}", j)).collect::<Vec<_>>().join(",");
    let wide_row = std::iter::repeat("1").take(90).collect::<Vec<_>>().join(",");
    for _ in 0..200 {
        wide.push('\n');
        wide.push_str(&wide_row);
    }
    let mut narrow = String::from("x,y");
    let long = "1".repeat(200);
    for _ in 0..200 {
        narrow.push('\n');
        narrow.push_str(&long);
        narrow.push(',');
        narrow.push_str(&long);
    }
    let wide_path = dir.join("wide.csv");
    let narrow_path = dir.join("narrow.csv");
    std::fs::write(&wide_path, &wide).unwrap();
    std::fs::write(&narrow_path, &narrow).unwrap();
    assert!(wide.len() < narrow.len());

    // 300 kB is above the narrow table's estimate and below the wide one's.
    let budget = "300KB";
    let wide_run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            wide_path.to_str().unwrap(),
            "--target",
            "c0",
            "--max-memory",
            budget,
            "--generations",
            "1",
            "--pop",
            "8",
        ])
        .output()
        .unwrap();
    assert_eq!(wide_run.status.code(), Some(3));
    let err = String::from_utf8_lossy(&wide_run.stderr);
    assert!(err.contains("200 rows x 90 columns"), "{err}");
    assert!(err.contains("estimate basis"), "{err}");

    let narrow_run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            narrow_path.to_str().unwrap(),
            "--target",
            "y",
            "--max-memory",
            budget,
            "--generations",
            "1",
            "--pop",
            "8",
        ])
        .output()
        .unwrap();
    assert_ne!(
        narrow_run.status.code(),
        Some(3),
        "a narrow file with more bytes must pass the shape guard: {}",
        String::from_utf8_lossy(&narrow_run.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
}


#[test]
fn the_memory_guard_refuses_a_csv_cleanly_with_exit_3() {
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            csv,
            "--target",
            "y",
            "--max-memory",
            "1",
            "--generations",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    let err = String::from_utf8_lossy(&output.stderr);
    assert!(err.contains("memory budget"), "{err}");
    assert!(err.contains("--max-memory"), "{err}");
    assert!(err.contains("rcsr import"), "{err}");
    assert!(!err.contains("memory allocation"), "{err}");
    assert!(
        String::from_utf8_lossy(&output.stdout).is_empty(),
        "the guard must refuse before printing a result"
    );
}

#[test]
fn the_import_command_honours_the_same_guard() {
    let dir = temp_dir("import-guard");
    let csv = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let out = dir.join("x.rcsb");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "import",
            csv,
            out.to_str().unwrap(),
            "--max-memory",
            "1",
        ])
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&output.stderr).contains("memory budget"));
    assert!(!out.exists());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_store_is_detected_by_magic_even_without_the_extension() {
    let dir = temp_dir("magic");
    let path = dir.join("store.bin");
    store::write_store(path.to_str().unwrap(), &full_table(), 4).unwrap();
    assert!(store::is_store_path(path.to_str().unwrap()));
    assert!(!store::is_store_path(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/clean.csv"
    )));
    let _ = std::fs::remove_dir_all(&dir);
}
