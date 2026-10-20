//! Versioned column store, memory guard and out-of-core backing (issue #71).
//!
//! The fixtures are tiny on purpose: the large-file measurement lives in the
//! issue report, not in the test suite.

use rcsr::data::{self, Table};
use rcsr::store::{self, DType, FileStore, MAGIC};
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

fn nanosecond_table() -> Table {
    // Consecutive prints 1 ns apart around 1.7e18: binary64 rounds both to the
    // same f64, so this fixture fails unless the store keeps i64.
    data::parse_table(
        "ts,symbol,y\n\
         1700000000000000001,A,1\n\
         1700000000000000002,A,2\n\
         1700000000000000003,B,3\n\
         1700000000000000004,B,4\n\
         1700000000000000005,A,5\n",
    )
    .unwrap()
}

#[test]
fn nanosecond_timestamps_round_trip_exactly_and_stay_distinct() {
    let table = nanosecond_table();
    let exact: Vec<i64> = (1..=5).map(|i| 1_700_000_000_000_000_000 + i).collect();
    assert_eq!(table.ints(0).unwrap(), &exact[..]);

    let dir = temp_dir("ns");
    let path = dir.join("ns.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 2).unwrap();
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    assert_eq!(fs.header().dtypes(), vec![DType::I64, DType::F64, DType::F64]);
    let back = fs.to_table().unwrap();
    assert_eq!(back.ints(0).unwrap(), &exact[..]);

    // Distinct and ordered through the public time-key path.
    match data::time_keys(&back, 0) {
        data::TimeKeys::I64(keys) => {
            assert_eq!(keys, exact);
            assert!(keys.windows(2).all(|w| w[0] < w[1]));
        }
        data::TimeKeys::F64(_) => panic!("store lost the exact i64 time view"),
    }
    // The evaluator view is still f64 (not used for arithmetic here).
    assert_eq!(back.col(0).len(), 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn millisecond_and_microsecond_integers_are_exact_too() {
    for base in [1_700_000_000_000i64, 1_700_000_000_000_000i64] {
        let csv = format!(
            "ts,x\n{},{}\n{},{}\n",
            base,
            1,
            base + 1,
            2
        );
        let table = data::parse_table(&csv).unwrap();
        assert_eq!(table.ints(0).unwrap(), &[base, base + 1]);
        let dir = temp_dir("msus");
        let path = dir.join("t.rcsb");
        store::write_store(path.to_str().unwrap(), &table, 1).unwrap();
        let back = FileStore::open(path.to_str().unwrap())
            .unwrap()
            .to_table()
            .unwrap();
        assert_eq!(back.ints(0).unwrap(), &[base, base + 1]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}

#[test]
fn a_v1_store_without_dtype_fields_still_loads_as_f64() {
    // The original v1 header has no `dtype`: bytes must keep the old meaning.
    let header = "{\"format\":\"rcsr-column-store\",\"version\":1,\"n_rows\":3,\
                  \"n_cols\":2,\"group_rows\":2,\"columns\":[\
                  {\"name\":\"ts\",\"numeric\":true,\"levels\":[],\"has_codes\":false,\
                   \"offsets\":[0,32],\"code_offsets\":[]},\
                  {\"name\":\"y\",\"numeric\":true,\"levels\":[],\"has_codes\":false,\
                   \"offsets\":[16,40],\"code_offsets\":[]}]}";
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for v in [1.0f64, 2.0, 10.0, 20.0, 3.0, 30.0] {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    let dir = temp_dir("oldv1");
    let path = dir.join("old.rcsb");
    std::fs::write(&path, &bytes).unwrap();
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    assert_eq!(fs.header().dtypes(), vec![DType::F64, DType::F64]);
    let back = fs.to_table().unwrap();
    assert!(back.ints(0).is_none());
    assert_eq!(back.col(0), &[1.0, 2.0, 3.0]);
    assert!(matches!(
        data::time_keys(&back, 0),
        data::TimeKeys::F64(_)
    ));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn sorting_duplicate_instants_and_gap_detection_use_exact_i64() {
    // A duplicate instant, two prints 1 ns apart, and a one-hour gap.
    let base = 1_700_000_000_000_000_000i64;
    let csv = format!(
        "ts,x\n{base},1\n{base},2\n{},3\n{},4\n{},5\n",
        base + 1,
        base + 2,
        base + 3_600_000_000_000
    );
    let table = data::parse_table(&csv).unwrap();
    let lag = data::LagSpec {
        lags: 1,
        gap_break: 4.0,
        ..Default::default()
    };
    let (_out, _names, report) =
        data::add_lags(&table, &["x".into()], &lag, Some(0), None).unwrap();
    assert_eq!(report.breaks, 1, "the one-hour gap is a series boundary");
    assert_eq!(report.series, 2);

    // Duplicates keep distinct row identities and sort by row after the key.
    match data::time_keys(&table, 0) {
        data::TimeKeys::I64(keys) => {
            assert_eq!(keys[0], keys[1], "same instant stays duplicated");
            assert!(keys[1] < keys[2] && keys[2] < keys[3]);
        }
        data::TimeKeys::F64(_) => panic!("expected i64 keys"),
    }
}

#[test]
fn an_explicit_time_column_is_honoured_by_import() {
    let dir = temp_dir("time-col");
    let csv = dir.join("t.csv");
    std::fs::write(&csv, "stamp,x\n10,1\n20,2\n30,3\n").unwrap();
    let out = dir.join("t.rcsb");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "import",
            csv.to_str().unwrap(),
            out.to_str().unwrap(),
            "--time",
            "stamp",
            "--quiet",
        ])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "import failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let fs = FileStore::open(out.to_str().unwrap()).unwrap();
    assert_eq!(fs.header().dtypes(), vec![DType::I64, DType::F64]);
    assert_eq!(fs.to_table().unwrap().ints(0).unwrap(), &[10, 20, 30]);
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
