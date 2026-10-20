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
    bytes[8..12].copy_from_slice(&99u32.to_le_bytes());
    std::fs::write(&path, &bytes).unwrap();
    let err = FileStore::open(path.to_str().unwrap()).unwrap_err();
    assert!(err.contains("version 99"), "{err}");
    assert!(err.contains("versions 1 and 2"), "{err}");
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
fn v2_has_per_column_dtypes_and_round_trips_nanoseconds_and_dictionaries() {
    let table = nanosecond_table();
    let dir = temp_dir("v2-dtypes");
    let path = dir.join("t.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 2).unwrap();
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(
        u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
        store::VERSION
    );
    assert_eq!(store::VERSION, 2);
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    assert_eq!(
        fs.header().dtypes(),
        vec![DType::I64, DType::U32, DType::F64]
    );
    let back = fs.to_table().unwrap();
    let exact: Vec<i64> = (1..=5).map(|i| 1_700_000_000_000_000_000 + i).collect();
    assert_eq!(back.ints(0).unwrap(), &exact[..]);
    assert_eq!(back.dict(1).unwrap().levels, table.dict(1).unwrap().levels);
    assert_eq!(back.dict(1).unwrap().codes, table.dict(1).unwrap().codes);
    // A dictionary column has no evaluator values; its levels are the view and
    // a code never appears as a number.
    assert!(back.col(1).is_empty());
    assert!(back.value(0, 1).is_nan());
    assert_eq!(back.col(2), table.col(2));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_code_blocks_are_four_bytes_and_the_file_is_dictionary_heavy_smaller() {
    let mut csv = String::from("a,b,x\n");
    for i in 0..1000 {
        csv.push_str(&format!("L{},{},{}\n", i % 10, i % 7, i));
    }
    let table = data::parse_table(&csv).unwrap();
    let dir = temp_dir("v2-size");
    let path = dir.join("t.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 1000).unwrap();
    let size = std::fs::metadata(&path).unwrap().len() as usize;
    // The one-group data section is 2 x 4000 u32 codes for the text column,
    // 8000 bytes of b values + 4000 codes, and 8000 x values + 4000 codes:
    // 32,000 bytes, plus a header dominated by the 1000-level dictionary of x.
    // The v1 layout would add 8-byte value/code blocks throughout (~54 KiB).
    assert!(size < 36_000, "v2 store is {} bytes", size);
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let back = fs.to_table().unwrap();
    assert_eq!(back.dict(0).unwrap().codes, table.dict(0).unwrap().codes);
    assert_eq!(back.dict(1).unwrap().codes, table.dict(1).unwrap().codes);
    assert_eq!(back.col(2), table.col(2));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_hundred_symbol_daily_panel_migrates_from_v1_to_v2_identically() {
    // The shape of the 100-symbol daily panel: 100 symbols, 60 days, a text
    // dictionary plus numeric features. Written once as a v1-equivalent
    // hand-built store and once through the v2 writer; decoded rows agree.
    let mut csv = String::from("ts,symbol,open,r1,y1\n");
    for day in 0..60 {
        for s in 0..100 {
            csv.push_str(&format!(
                "{},{},{},{},{}\n",
                1619827200 + day * 86400,
                format!("SYM{:03}USDT", s),
                100.0 + s as f64 + day as f64 * 0.5,
                (s as f64 - 50.0) / 100.0,
                day as f64 / 60.0
            ));
        }
    }
    let table = data::parse_table(&csv).unwrap();
    assert_eq!(table.dict(1).unwrap().levels.len(), 100);

    let dir = temp_dir("daily100");
    let v2 = dir.join("v2.rcsb");
    store::write_store(v2.to_str().unwrap(), &table, 997).unwrap();
    let v1 = dir.join("v1.rcsb");
    write_v1_from_table(&v1, &table, 997);
    let a = FileStore::open(v1.to_str().unwrap()).unwrap();
    let b = FileStore::open(v2.to_str().unwrap()).unwrap();
    assert_eq!(a.header().version, 1);
    assert_eq!(b.header().version, 2);
    let ta = a.to_table().unwrap();
    let tb = b.to_table().unwrap();
    for c in [0usize, 2, 3, 4] {
        let bad = ta
            .col(c)
            .iter()
            .zip(tb.col(c))
            .position(|(a, b)| a.to_bits() != b.to_bits());
        if let Some(i) = bad {
            panic!(
                "column {} differs at row {}: v1={:?} (bits {:x}) v2={:?} (bits {:x})",
                c,
                i,
                ta.col(c)[i],
                ta.col(c)[i].to_bits(),
                tb.col(c)[i],
                tb.col(c)[i].to_bits()
            );
        }
    }
    assert_eq!(ta.dict(1).unwrap().levels, tb.dict(1).unwrap().levels);
    assert_eq!(ta.dict(1).unwrap().codes, tb.dict(1).unwrap().codes);
    assert!(std::fs::metadata(&v2).unwrap().len() < std::fs::metadata(&v1).unwrap().len());
    let _ = std::fs::remove_dir_all(&dir);
}

/// Write the historical v1 layout (all values and codes as f64) for migration
/// tests. This mirrors what `rcsr` wrote before #128.
fn write_v1_from_table(path: &std::path::Path, table: &Table, group_rows: usize) {
    let n = table.len();
    let n_cols = table.n_cols();
    let n_groups = (n + group_rows - 1) / group_rows;
    let mut cursor = 0u64;
    let mut metas: Vec<(Vec<u64>, Vec<u64>)> = (0..n_cols)
        .map(|_| (Vec::new(), Vec::new()))
        .collect();
    for g in 0..n_groups {
        let len = ((n - g * group_rows).min(group_rows)) as u64;
        for c in 0..n_cols {
            metas[c].0.push(cursor);
            cursor += len * 8;
            if table.dict(c).is_some() {
                metas[c].1.push(cursor);
                cursor += len * 8;
            }
        }
    }
    let mut json = format!(
        "{{\"format\":\"rcsr-column-store\",\"version\":1,\"n_rows\":{},\"n_cols\":{},\"group_rows\":{},\"columns\":[",
        n, n_cols, group_rows
    );
    for c in 0..n_cols {
        if c > 0 {
            json.push(',');
        }
        let levels = table
            .dict(c)
            .map(|d| {
                d.levels
                    .iter()
                    .map(|s| format!("\"{}\"", s))
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();
        json.push_str(&format!(
            "{{\"name\":\"{}\",\"numeric\":{},\"levels\":[{}],\"has_codes\":{},\"offsets\":[{}],\"code_offsets\":[{}]}}",
            table.names[c],
            table.numeric[c],
            levels,
            table.dict(c).is_some(),
            metas[c].0.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
            metas[c].1.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
        ));
    }
    json.push_str("]}");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&(json.len() as u32).to_le_bytes());
    bytes.extend_from_slice(json.as_bytes());
    for g in 0..n_groups {
        let lo = g * group_rows;
        let hi = (lo + group_rows).min(n);
        for c in 0..n_cols {
            for r in lo..hi {
                bytes.extend_from_slice(&table.value(r, c).to_le_bytes());
            }
            if let Some(d) = table.dict(c) {
                for r in lo..hi {
                    bytes.extend_from_slice(&(d.codes[r] as f64).to_le_bytes());
                }
            }
        }
    }
    std::fs::write(path, bytes).unwrap();
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
