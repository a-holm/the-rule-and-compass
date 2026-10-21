//! #116: disk-backed `StoreRows` over `FileStore::load_group_with`.
//!
//! The streaming fitter must consume a `.rcsb` store group by group and merge
//! the exact `f64` sufficient statistics, without `Sample` or a design matrix
//! becoming resident. These tests pin the exact row/weight/subsample mapping
//! and the bit-identical merge against the resident `SliceRows` source.

use rcsr::data;
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::fit::{fit_stream_with, FitStrategy, RowSource, SliceRows, StoreRows, StoreSpec};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};
use rcsr::store::{self, FileStore};
use std::sync::Arc;

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-stream-{}-{}", tag, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn model() -> Node {
    // a*sin(b*x) + c
    node(
        Op::Add,
        vec![
            node(
                Op::Mul,
                vec![
                    constant(0.5),
                    node(Op::Sin, vec![node(Op::Mul, vec![constant(0.5), input(0)])]),
                ],
            ),
            constant(0.0),
        ],
    )
}

fn fitted(tree: &Node) -> Vec<f64> {
    fn walk(n: &Node, out: &mut Vec<f64>) {
        if n.op == Op::Const {
            out.push(n.val);
        }
        for k in &n.kids {
            walk(k, out);
        }
        if let Some(b) = &n.block {
            walk(&b.body, out);
        }
    }
    let mut out = Vec::new();
    walk(tree, &mut out);
    out
}

fn generated_table(n: usize) -> data::Table {
    // Full-precision text so the store round-trip is exact, and a third
    // column with non-unit weights.
    let mut text = String::from("x,y,w\n");
    for i in 0..n {
        let x = -3.0 + 6.0 * i as f64 / (n - 1) as f64;
        let y = 2.0 * (0.9 * x).sin() + 0.3;
        let w = 1.0 + (i % 3) as f64 * 0.25;
        text.push_str(&format!("{x:.17},{y:.17},{w:.17}\n"));
    }
    data::parse_table(&text).expect("numeric table")
}

#[test]
fn store_rows_reproduce_the_resident_f64_fit_bit_for_bit() {
    let n = 5_000;
    let table = generated_table(n);
    let dir = temp_dir("exact");
    let path = dir.join("data.rcsb");
    // Groups deliberately do not divide the row count.
    store::write_store(path.to_str().unwrap(), &table, 997).unwrap();
    let fs = Arc::new(FileStore::open(path.to_str().unwrap()).unwrap());
    let spec = Arc::new(StoreSpec::new(fs, vec![0], 1).unwrap());
    let source = StoreRows::new(spec);
    assert_eq!(source.rows(), n);

    // Materialise the same rows only for the comparison; the source itself
    // never holds more than one group.
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    assert!(source.for_each_row(&mut |row, t, w| {
        x.push(row.to_vec());
        y.push((t, w));
        true
    }));
    for (i, row) in x.iter().enumerate() {
        assert_eq!(row[0].to_bits(), table.value(i, 0).to_bits(), "x row {i}");
        assert_eq!(y[i].0.to_bits(), table.value(i, 1).to_bits(), "y row {i}");
        assert_eq!(y[i].1, 1.0, "unweighted source row {i}");
    }

    let y: Vec<f64> = y.into_iter().map(|(t, _)| t).collect();
    let mut disk = model();
    let a = fit_stream_with(&mut disk, &source, 12).expect("disk stream fit");
    let mut resident = model();
    let b = fit_stream_with(
        &mut resident,
        &SliceRows { x: &x, y: &y, w: &[] },
        12,
    )
    .expect("resident stream fit");
    assert_eq!(a.to_bits(), b.to_bits(), "SSE must merge identically");
    let ca: Vec<u64> = fitted(&disk).iter().map(|v| v.to_bits()).collect();
    let cb: Vec<u64> = fitted(&resident).iter().map(|v| v.to_bits()).collect();
    assert_eq!(ca, cb, "constants must be bit-identical");
    assert!(a < 1e-20, "the generated law is exactly separable: {a}");
}

#[test]
fn store_rows_honour_weights_subsets_and_standardisation() {
    let n = 2_000;
    let table = generated_table(n);
    let dir = temp_dir("projection");
    let path = dir.join("data.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 256).unwrap();
    let fs = Arc::new(FileStore::open(path.to_str().unwrap()).unwrap());

    // Every third row, in raw store order; means/stds mimic preparation.
    let rows: Vec<usize> = (0..n).filter(|i| i % 3 == 0).collect();
    let means = vec![0.25];
    let stds = vec![1.5];
    let spec = Arc::new(
        StoreSpec::new(fs, vec![0], 1)
            .unwrap()
            .with_weight(2)
            .unwrap()
            .with_standardisation(means.clone(), stds.clone())
            .unwrap()
            .with_rows(rows.clone())
            .unwrap(),
    );
    let source = StoreRows::new(spec);
    assert_eq!(source.rows(), rows.len());

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws = Vec::new();
    assert!(source.for_each_row(&mut |row, y, w| {
        xs.push(row.to_vec());
        ys.push(y);
        ws.push(w);
        true
    }));
    for (j, &raw) in rows.iter().enumerate() {
        let expected = (table.value(raw, 0) - means[0]) / stds[0];
        assert_eq!(xs[j][0].to_bits(), expected.to_bits(), "x at row {raw}");
        assert_eq!(ys[j].to_bits(), table.value(raw, 1).to_bits(), "y at row {raw}");
        assert_eq!(ws[j], table.value(raw, 2), "w at row {raw}");
    }

    let mut disk = model();
    let a = fit_stream_with(&mut disk, &source, 8).expect("projected disk fit");
    let mut resident = model();
    let b = fit_stream_with(
        &mut resident,
        &SliceRows { x: &xs, y: &ys, w: &ws },
        8,
    )
    .expect("projected resident fit");
    assert_eq!(a.to_bits(), b.to_bits());
}

#[test]
fn non_finite_store_rows_reject_the_streaming_arm() {
    let table = data::parse_table("x,y\n0,1\n1,NaN\n2,3\n").unwrap();
    let dir = temp_dir("nonfinite");
    let path = dir.join("data.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 2).unwrap();
    let fs = Arc::new(FileStore::open(path.to_str().unwrap()).unwrap());
    let source = StoreRows::new(Arc::new(StoreSpec::new(fs, vec![0], 1).unwrap()));
    let mut tree = model();
    assert!(fit_stream_with(&mut tree, &source, 4).is_none());
}

/// #116: the search's `--fit-strategy stream` path reads the same training rows
/// from the resolved `StoreSpec` and produces the identical trajectory as the
/// resident `SliceRows` source: same archive errors (bit for bit), same trees.
#[test]
fn search_streams_from_the_store_with_the_same_trajectory() {
    let n = 240;
    let table = generated_table(n);
    let dir = temp_dir("search");
    let path = dir.join("data.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 31).unwrap();
    let fs = Arc::new(FileStore::open(path.to_str().unwrap()).unwrap());
    let spec = Arc::new(StoreSpec::new(fs, vec![0], 1).unwrap());
    let x: Vec<Vec<f64>> = (0..n).map(|i| vec![table.value(i, 0)]).collect();
    let y: Vec<f64> = (0..n).map(|i| table.value(i, 1)).collect();
    let train = data::Sample::new(x, y);

    let base = Config {
        generations: 3,
        pop: 24,
        islands: 1,
        seed: 7,
        metric: Metric::Squared,
        fit_strategy: FitStrategy::Stream,
        gpu: false,
        ..Config::default()
    };
    let resident = search::run(&base, &train, &train);
    let with_store = Config {
        fit_stream_source: Some(spec),
        ..base.clone()
    };
    let streamed = search::run(&with_store, &train, &train);
    assert_eq!(
        resident.archive.items.len(),
        streamed.archive.items.len(),
        "the archive must not change when rows come from disk"
    );
    for (a, b) in resident.archive.items.iter().zip(&streamed.archive.items) {
        assert_eq!(a.error.to_bits(), b.error.to_bits(), "archive error");
        assert_eq!(a.complexity, b.complexity);
        assert_eq!(
            rcsr::expr::to_string(&a.tree, &["x".to_string()]),
            rcsr::expr::to_string(&b.tree, &["x".to_string()])
        );
    }
}

/// #116: bounded out-of-core probe. It is ignored by default because it
/// materialises the fixture through `rcsr import`; run it only on a host with
/// the disk and RAM headroom. `RCSR_STREAM_PROBE_STORE` points at an existing
/// `.rcsb` with columns `x, y`; the test reports rows, groups and RSS.
#[test]
#[ignore]
fn probe_large_store_fit() {
    let Ok(path) = std::env::var("RCSR_STREAM_PROBE_STORE") else {
        eprintln!("probe skipped: set RCSR_STREAM_PROBE_STORE to a store with x,y");
        return;
    };
    let fs = Arc::new(FileStore::open(&path).expect("open probe store"));
    let n = fs.n_rows();
    let source = StoreRows::new(Arc::new(StoreSpec::new(fs, vec![0], 1).unwrap()));
    let mut tree = model();
    let start = std::time::Instant::now();
    let sse = fit_stream_with(&mut tree, &source, 8).expect("large stream fit");
    let seconds = start.elapsed().as_secs_f64();
    let rss = std::fs::read_to_string("/proc/self/status")
        .ok()
        .and_then(|s| {
            s.lines()
                .find(|l| l.starts_with("VmHWM:"))
                .map(|l| l.to_string())
        })
        .unwrap_or_default();
    eprintln!(
        "probe N={n} sse={sse:.6e} seconds={seconds:.3} constants={:?} {rss}",
        fitted(&tree)
    );
    assert!(sse.is_finite());
}
