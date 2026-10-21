//! Header-derived store memory estimator (issue #112).
//!
//! The old model was `rows x columns x 300 B`; on the 3,943,856 x 56 panel it
//! reported 61.71 GiB while the forced full load peaked at 6.49 GB. These tests
//! pin the replacement: resident dtypes, dictionaries, row groups, the
//! used-column projection and the explicit full-load/sample decision.

use rcsr::data::{self, Table};
use rcsr::memory::{EstimateOptions, StoreShape};
use rcsr::store::{self, FileStore, MAGIC};
use std::process::Command;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-estimator-{}-{}",
        tag,
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Historical v1 layout: every value and code block is f64 on disk.
fn write_v1_from_table(path: &std::path::Path, table: &Table, group_rows: usize) {
    let n = table.len();
    let n_cols = table.n_cols();
    let n_groups = (n + group_rows - 1) / group_rows;
    let mut cursor = 0u64;
    let mut metas: Vec<(Vec<u64>, Vec<u64>)> =
        (0..n_cols).map(|_| (Vec::new(), Vec::new())).collect();
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

fn table_with_text() -> Table {
    // More than `NUMERIC_DICT_MAX` distinct values, so x and y keep no
    // dictionary while `label` does. That separates a value block from a code
    // block and catches a model that charges every coded column a value.
    let mut csv = String::from("x,label,y\n");
    for i in 0..5_000 {
        csv.push_str(&format!("{},{},{}\n", i, if i % 2 == 0 { "A" } else { "B" }, i * 2));
    }
    data::parse_table(&csv).unwrap()
}

#[test]
fn v1_header_charges_resident_widths_not_the_disk_code_width() {
    let dir = temp_dir("v1");
    let path = dir.join("v1.rcsb");
    let table = table_with_text();
    // Three groups: 30 rows with group_rows = 7.
    write_v1_from_table(&path, &table, 7);
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let shape = fs.memory_shape();
    let n = table.len() as u64;
    assert_eq!(shape.rows, 5_000);
    assert_eq!(shape.n_groups(), 715);

    let x = &shape.columns[0];
    let label = &shape.columns[1];
    let y = &shape.columns[2];
    assert_eq!(x.value_width, 8);
    assert_eq!(x.code_width, 0);
    // A pure dictionary column has no resident value block, even though v1
    // wrote an f64 value block for it on disk.
    assert_eq!(label.value_width, 0);
    assert_eq!(label.code_width, 4);
    assert_eq!(y.value_width, 8);
    assert_eq!(y.code_width, 0);

    // Exact resident storage: 2 numeric values + one u32 code per row, plus
    // the two dictionary levels. The 8-byte on-disk code block is not charged.
    let level_bytes = label.dictionary_bytes;
    assert_eq!(shape.storage_bytes(), n * 8 + n * 4 + n * 8 + level_bytes);
    // The old model would have charged 5000 * 3 * 300 = 4,500,000 B.
    assert!(shape.storage_bytes() < n * 3 * 300);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn v2_header_charges_the_i64_copy_and_pure_u32_columns() {
    // Hand-built v2: one f64, one i64 (values + ints), one pure u32 dictionary.
    let n = 10usize;
    let group_rows = 4usize;
    let n_groups = 3usize;
    let mut cursor = 0u64;
    let mut offsets = [Vec::new(), Vec::new(), Vec::new()];
    let mut code_offsets = [Vec::new(), Vec::new(), Vec::new()];
    for g in 0..n_groups {
        let len = ((n - g * group_rows).min(group_rows)) as u64;
        for c in 0..3 {
            offsets[c].push(cursor);
            cursor += if c == 2 { 0 } else { len * 8 };
            if c == 2 {
                code_offsets[c].push(cursor);
                cursor += len * 4;
            }
        }
    }
    let header = format!(
        "{{\"format\":\"rcsr-column-store\",\"version\":2,\"n_rows\":{},\"n_cols\":3,\"group_rows\":{},\"columns\":[\
         {{\"name\":\"x\",\"numeric\":true,\"dtype\":\"f64\",\"levels\":[],\"has_codes\":false,\"offsets\":[{}],\"code_offsets\":[]}},\
         {{\"name\":\"ts\",\"numeric\":true,\"dtype\":\"i64\",\"levels\":[],\"has_codes\":false,\"offsets\":[{}],\"code_offsets\":[]}},\
         {{\"name\":\"sym\",\"numeric\":false,\"dtype\":\"u32\",\"levels\":[\"a\",\"b\"],\"has_codes\":true,\"offsets\":[{}],\"code_offsets\":[{}]}}]}}",
        n,
        group_rows,
        offsets[0].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        offsets[1].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        offsets[2].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(","),
        code_offsets[2].iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",")
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&2u32.to_le_bytes());
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(header.as_bytes());
    for g in 0..n_groups {
        let lo = g * 4;
        let hi = (lo + 4).min(n);
        for r in lo..hi {
            bytes.extend_from_slice(&(r as f64).to_le_bytes());
        }
        for r in lo..hi {
            bytes.extend_from_slice(&(1_600_000_000i64 + r as i64).to_le_bytes());
        }
        for r in lo..hi {
            bytes.extend_from_slice(&((r % 2) as u32).to_le_bytes());
        }
    }
    let dir = temp_dir("v2");
    let path = dir.join("v2.rcsb");
    std::fs::write(&path, bytes).unwrap();
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let shape = fs.memory_shape();
    assert_eq!(shape.columns[0].value_width, 8);
    assert_eq!(shape.columns[0].int_width, 0);
    assert_eq!(shape.columns[1].value_width, 8);
    assert_eq!(shape.columns[1].int_width, 8);
    assert_eq!(shape.columns[2].value_width, 0);
    assert_eq!(shape.columns[2].code_width, 4);
    assert_eq!(
        shape.storage_bytes(),
        n as u64 * (8 + 16 + 4) + shape.columns[2].dictionary_bytes
    );
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn row_groups_and_dictionaries_are_separate_terms() {
    let make = |group_rows: usize| StoreShape {
        rows: 1_000,
        group_rows,
        header_len: 0,
        columns: vec![rcsr::memory::ColumnShape {
            name: "g".into(),
            value_width: 0,
            int_width: 0,
            code_width: 4,
            dictionary_bytes: 1_000,
        }],
    };
    let small = make(7);
    let large = make(65_536);
    let opts = EstimateOptions::default();
    let a = small.estimate(&opts);
    let b = large.estimate(&opts);
    assert!(a.directory_bytes > b.directory_bytes);
    assert!(a.group_buffer_bytes < b.group_buffer_bytes);
    assert_eq!(a.storage_bytes, b.storage_bytes);
    assert!(a.directory_bytes >= 8 * small.n_groups() as u64);
}

fn panel_shape(rows: usize) -> StoreShape {
    let mut columns = Vec::new();
    for i in 0..51 {
        columns.push(rcsr::memory::ColumnShape {
            name: format!("v{}", i),
            value_width: 8,
            int_width: 0,
            code_width: 0,
            dictionary_bytes: 0,
        });
    }
    for i in 0..4 {
        columns.push(rcsr::memory::ColumnShape {
            name: format!("code{}", i),
            value_width: 8,
            int_width: 0,
            code_width: 4,
            dictionary_bytes: 1_500,
        });
    }
    columns.push(rcsr::memory::ColumnShape {
        name: "symbol".into(),
        value_width: 0,
        int_width: 0,
        code_width: 4,
        dictionary_bytes: 5_538,
    });
    StoreShape {
        rows,
        group_rows: 65_536,
        header_len: 46_343,
        columns,
    }
}

fn panel_options() -> EstimateOptions {
    EstimateOptions {
        used_cols: 53, // 52 variables + target
        scan_cols: 2,  // time + group
        gpu: true,
        gpu_fit: true,
        mutable_prep: false,
        simd: true,
    }
}

#[test]
fn the_3_9m_panel_estimates_near_the_measured_footprint() {
    let estimate = panel_shape(3_943_856).estimate(&panel_options());
    // 53 search cells + 2 scan cells; storage is 1.69 GiB, the measured search
    // plateau is 4.93 GiB and the forced full peak is 6.49 GB.
    assert!(
        estimate.total_bytes >= 10_000_000_000 && estimate.total_bytes <= 10_800_000_000,
        "estimate {} ({}) is not near the calibrated 10.28-10.54 GB",
        estimate.total_bytes,
        rcsr::memory::human(estimate.total_bytes)
    );
    assert!(estimate.total_bytes <= 2 * 6_491_676 * 1024);
    assert!(estimate.total_bytes >= 4_933_540 * 1024);
    // The 12 GB budget on the pilot host must choose the full load.
    assert!(estimate.total_bytes <= 12u64 * 1_000_000_000);
    assert!(!estimate.saturated);
    // The reported terms must add up exactly, with `fixed` not folded into
    // `search` a second time.
    let terms = estimate.storage_bytes
        + estimate.directory_bytes
        + estimate.group_buffer_bytes
        + estimate.search_bytes
        + estimate.fit_copy_bytes
        + estimate.fixed_bytes;
    assert_eq!(terms, estimate.base_bytes);
    assert_eq!(estimate.base_bytes + estimate.slack_bytes, estimate.total_bytes);
}

#[test]
fn the_15_9m_store_scales_with_cells_not_a_global_constant() {
    let small = panel_shape(3_943_856).estimate(&panel_options());
    let large = panel_shape(15_775_424).estimate(&panel_options());
    let ratio = large.total_bytes as f64 / small.total_bytes as f64;
    assert!((3.9..=4.1).contains(&ratio), "ratio {ratio}");
    // Four times the 6.49 GB peak with the issue's 2x ceiling.
    assert!(large.total_bytes <= 2 * 4 * 6_491_676 * 1024);
    assert!(large.total_bytes >= 4 * 4_933_540 * 1024);
}

#[test]
fn the_projection_changes_the_search_copy_not_the_storage() {
    let shape = panel_shape(100_000);
    let mut opts = panel_options();
    let full = shape.estimate(&opts);
    opts.used_cols = 5;
    opts.scan_cols = 0;
    opts.gpu = false;
    opts.gpu_fit = false;
    let projected = shape.estimate(&opts);
    assert_eq!(full.storage_bytes, projected.storage_bytes);
    assert_eq!(full.directory_bytes, projected.directory_bytes);
    assert!(projected.search_bytes < full.search_bytes);
    assert!(projected.total_bytes < full.total_bytes);
}

fn small_store(dir: &std::path::Path) -> std::path::PathBuf {
    let mut csv = String::from("x,y\n");
    for i in 0..2_000 {
        csv.push_str(&format!("{},{}\n", i, (i % 17) as f64 * 0.5));
    }
    let table = data::parse_table(&csv).unwrap();
    let path = dir.join("small.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 251).unwrap();
    path
}

#[test]
fn a_store_that_fits_loads_the_full_rows_not_a_sample() {
    let dir = temp_dir("full");
    let path = small_store(&dir);
    let fs = FileStore::open(path.to_str().unwrap()).unwrap();
    let opts = EstimateOptions {
        used_cols: 2,
        ..EstimateOptions::default()
    };
    let estimate = fs.estimate_resident(&opts);
    // `--cpu` pins the guard's model to the same options the estimate above
    // uses: a CUDA build defaults to the GPU backend, which legitimately adds
    // the f32 staging copy and would need a larger budget.
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--pop",
            "4",
            "--cpu",
            "--max-memory",
            &estimate.total_bytes.to_string(),
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {stderr}");
    assert!(!stderr.contains("out of core"), "{stderr}");
    assert!(!stderr.contains("sampling (explicit)"), "{stderr}");
    assert!(!stderr.contains("hash sample"), "{stderr}");
    assert!(stdout.contains("\"full_rows\":true"), "{stdout}");
    assert!(stdout.contains("\"sampled_rows\":null"), "{stdout}");
    assert!(stdout.contains("\"store_memory\":{"), "{stdout}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_over_budget_store_is_refused_never_silently_sampled() {
    let dir = temp_dir("refuse");
    let path = small_store(&dir);
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--max-memory",
            "1",
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("store header"), "{stderr}");
    assert!(stderr.contains("--sample-rows"), "{stderr}");
    assert!(stderr.contains("--force"), "{stderr}");
    assert!(!stderr.contains("hash sample"), "{stderr}");
    assert!(!stderr.contains("sampling ("), "{stderr}");
    assert!(String::from_utf8_lossy(&output.stdout).is_empty());
    let _ = std::fs::remove_dir_all(&dir);
}

/// The real pilot artifact is deliberately not committed. Run with
/// `RCSR_PANEL_STORE=/path/panel_hourly_compact.rcsb` and
/// `cargo test --release --test store_estimator -- --ignored`.
#[test]
#[ignore]
fn the_real_panel_store_loads_full_under_a_12gb_budget() {
    let path = match std::env::var("RCSR_PANEL_STORE") {
        Ok(p) => p,
        Err(_) => return,
    };
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            &path,
            "--target",
            "y60",
            "--time",
            "ts",
            "--group-by",
            "symbol",
            "--fill",
            "drop",
            "--generations",
            "1",
            "--pop",
            "4",
            "--fit-iters",
            "0",
            "--max-memory",
            "12GB",
            "--allow-lockbox",
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {stderr}");
    assert!(!stderr.contains("sampling"), "{stderr}");
    assert!(!stderr.contains("out of core"), "{stderr}");
    assert!(stdout.contains("\"full_rows\":true"), "{stdout}");
    assert!(stdout.contains("\"sampled_rows\":null"), "{stdout}");
    assert!(stdout.contains("\"rows\":3943856"), "{stdout}");
}

#[test]
fn a_sample_request_covering_every_row_cannot_bypass_the_budget() {
    // `--sample-rows N` with `N >= rows` is a full load. It must not become a
    // quieter way to ignore `--max-memory`.
    let dir = temp_dir("sample-full");
    let path = small_store(&dir);
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--pop",
            "4",
            "--max-memory",
            "1",
            "--sample-rows",
            "999999999",
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("covers every row"), "{stderr}");
    assert!(stderr.contains("--force"), "{stderr}");
    assert!(String::from_utf8_lossy(&output.stdout).is_empty());
    // A real sample that is still over budget is refused too.
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--pop",
            "4",
            "--max-memory",
            "1",
            "--sample-rows",
            "250",
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(3), "{stderr}");
    assert!(stderr.contains("would still exceed"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn force_and_sample_rows_are_rejected_as_contradictory() {
    let dir = temp_dir("force-sample");
    let path = small_store(&dir);
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--pop",
            "4",
            "--force",
            "--sample-rows",
            "250",
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("contradict"), "{stderr}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn explicit_sampling_is_recorded_as_full_rows_false() {
    let dir = temp_dir("sample");
    let path = small_store(&dir);
    let manifest = dir.join("run.manifest.json");
    let dev_log = dir.join("run.dev.ndjson");
    let output = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data",
            path.to_str().unwrap(),
            "--target",
            "y",
            "--vars",
            "x",
            "--generations",
            "1",
            "--pop",
            "4",
            "--sample-rows",
            "250",
            "--max-memory",
            "4GB",
            "--run-manifest",
            manifest.to_str().unwrap(),
            "--dev-log",
            dev_log.to_str().unwrap(),
            "--json",
        ])
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(output.status.success(), "stderr: {stderr}");
    assert!(stderr.contains("sampling (explicit)"), "{stderr}");
    assert!(stdout.contains("\"full_rows\":false"), "{stdout}");
    assert!(stdout.contains("\"sampled_rows\":250"), "{stdout}");
    let manifest_text = std::fs::read_to_string(&manifest).unwrap();
    assert!(manifest_text.contains("\"full_rows\":false"), "{manifest_text}");
    assert!(manifest_text.contains("\"sampled_rows\":250"), "{manifest_text}");
    let dev_text = std::fs::read_to_string(&dev_log).unwrap();
    assert!(dev_text.contains("\"full_rows\":false"), "{dev_text}");
    assert!(dev_text.contains("\"sampled_rows\":250"), "{dev_text}");
    let _ = std::fs::remove_dir_all(&dir);
}
