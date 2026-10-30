//! #294: pinned ring bound, copy/compute overlap and tile invariance.
//!
//! Device tests skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich_gpu::{DeviceColumn, EnrichGpu, ProjectedSource};
use rcsr::projected_ring::overlap_probe;
use rcsr::store::{self, FileStore, Projection};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-gpu-ring-{}-{}-{}",
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

/// Two contiguous symbols with a gap, enough groups to exercise slot reuse.
fn write_store(path: &str, rows_per_symbol: usize, group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,quote_volume\n");
    for (sym, start) in [("AAA", 0i64), ("BBB", 10_000_000)] {
        for i in 0..rows_per_symbol {
            let mut t = start + i as i64 * 3600;
            if i > rows_per_symbol / 2 {
                t += 3 * 3600;
            }
            let p = 100.0 + (i as f64 * 0.013).sin() * 7.0 + i as f64 * 0.0001;
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                t,
                sym,
                p,
                p + 0.3 + (i as f64 * 0.07).cos().abs() * 0.2,
                p - 0.4 - (i as f64 * 0.05).sin().abs() * 0.2,
                p + (i as f64 * 0.11).sin() * 0.1,
                10.0 + (i % 97) as f64,
                1000.0 + (i % 53) as f64
            ));
        }
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
}

struct Built {
    proj: Projection,
    sources: Vec<ProjectedSource>,
    seg_start: Vec<u32>,
    seg_len: Vec<u32>,
}

fn build(fs: &FileStore) -> Built {
    let names = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ];
    let proj = Projection::resolve(fs, &names).unwrap();
    let dests = [
        DeviceColumn::Time,
        DeviceColumn::Value(0),
        DeviceColumn::Value(1),
        DeviceColumn::Value(2),
        DeviceColumn::Value(3),
        DeviceColumn::Value(4),
        DeviceColumn::Value(5),
    ];
    let sources = proj
        .columns()
        .iter()
        .zip(dests)
        .map(|(c, d)| ProjectedSource {
            header_index: c.index,
            destination: d,
        })
        .collect();
    // The store is symbol-major with two contiguous runs.
    let rows = fs.n_rows();
    let half = rows / 2;
    Built {
        proj,
        sources,
        seg_start: vec![0, half as u32],
        seg_len: vec![half as u32, (rows - half) as u32],
    }
}

fn run(fs: &FileStore, b: &Built, slots: usize, names: &[String]) -> (Vec<Vec<u64>>, u64) {
    let mut gpu = EnrichGpu::from_projected(
        fs,
        b.proj.clone(),
        &b.sources,
        &b.seg_start,
        &b.seg_len,
        3600,
        names,
        slots,
    )
    .expect("projected transport");
    let ring = gpu.ring_bytes();
    let mut out = Vec::new();
    for n in names {
        let vals = gpu.take_precomputed(n).expect("precomputed tile");
        out.push(vals.iter().map(|v| v.to_bits()).collect());
    }
    (out, ring)
}

#[test]
fn ring_is_bounded_constant_and_tiles_are_invariant() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("bound");
    let path = dir.join("in.rcsb").to_string_lossy().into_owned();
    let path_small_groups = dir.join("in2.rcsb").to_string_lossy().into_owned();
    // 100 groups of 64 rows for the bound, and a 2-group layout for the
    // group-size invariance check.
    write_store(&path, 3_200, 64);
    write_store(&path_small_groups, 3_200, 4096);

    let fs = FileStore::open(&path).unwrap();
    assert_eq!(fs.n_groups(), 100);
    let b = build(&fs);
    let names = [
        "r1".to_string(),
        "r5".to_string(),
        "lag_r1_3".to_string(),
        "range1".to_string(),
        "clv1".to_string(),
        "hour_sin".to_string(),
    ];

    let (two, ring_two) = run(&fs, &b, 2, &names);
    let (three, ring_three) = run(&fs, &b, 3, &names);
    assert!(ring_two > 0 && ring_two <= 64 * 1024 * 1024);
    assert!(ring_three > 0 && ring_three <= 64 * 1024 * 1024);
    // The pin is a function of one group and the slot count, not of how many
    // groups were read: 100 groups do not grow it.
    assert_eq!(ring_two % 2, 0);
    assert_eq!(ring_three % 3, 0);
    assert_eq!(ring_three / 3, ring_two / 2);
    assert_eq!(two, three, "slot count changed the values");

    let fs2 = FileStore::open(&path_small_groups).unwrap();
    assert_eq!(fs2.n_groups(), 2);
    let b2 = build(&fs2);
    let (two_groups, _) = run(&fs2, &b2, 3, &names);
    assert_eq!(three, two_groups, "group size changed the values");

    // The tiled kernels must match the whole-column reference bit for bit.
    let mut whole = {
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for name in ["ts", "open", "high", "low", "close", "volume", "quote_volume"] {
            let c = fs.header().column_index(name).unwrap();
            let mut v = Vec::new();
            for g in 0..fs.n_groups() {
                v.extend(fs.read_f64_group(g, c).unwrap());
            }
            cols.push(v);
        }
        let times: Vec<i64> = cols[0].iter().map(|v| *v as i64).collect();
        EnrichGpu::new(
            &times, &cols[1], &cols[2], &cols[3], &cols[4], &cols[5], &cols[6],
            &b.seg_start, &b.seg_len, 3600,
        )
        .unwrap()
    };
    for (i, name) in names.iter().enumerate() {
        let expected = match name.as_str() {
            "r1" => whole.ret(1),
            "r5" => whole.ret(5),
            "lag_r1_3" => whole.lag(3),
            "range1" => whole.range(),
            "clv1" => whole.clv(),
            "hour_sin" => whole.calendar(2),
            _ => None,
        }
        .unwrap();
        let want: Vec<u64> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(two[i], want, "tiled '{}' differs from the whole kernel", name);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn copy_overlaps_compute_with_events_not_stream_sync() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let probe = overlap_probe(8 * 1024 * 1024, 8, 20_000_000).expect("overlap probe");
    assert!(probe.copies >= 8);
    assert!(!probe.failed);
    assert!(
        probe.overlapped,
        "copy {:.3} ms and compute {:.3} ms did not overlap",
        probe.copy_ms, probe.compute_ms
    );
    assert!(probe.copy_ms > 0.0 && probe.compute_ms > 0.0);
}
