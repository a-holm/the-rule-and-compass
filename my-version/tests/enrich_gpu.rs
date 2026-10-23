//! #200: GPU enrichment parity and dispatch.
//!
//! The CPU columns are the reference. These tests build a store above the cost
//! model's row floor, enrich it once with `--gpu off` and once with `--gpu on`,
//! and compare every column value plus the report's GPU accounting. Device tests
//! skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::store::{self, DType, FileStore};

fn bars(symbol: &str, n: usize, start: i64, step: i64, gap_at: Option<usize>) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|i| {
            let mut t = start + i as i64 * step;
            if let Some(g) = gap_at {
                if i >= g {
                    t += 3 * step;
                }
            }
            let p = 100.0 + (i as f64 * 0.013).sin() * 7.0 + i as f64 * 0.0001;
            (
                t,
                symbol.to_string(),
                p,
                p + 0.3 + (i as f64 * 0.07).cos().abs() * 0.2,
                p - 0.4 - (i as f64 * 0.05).sin().abs() * 0.2,
                p + (i as f64 * 0.11).sin() * 0.1,
                10.0 + (i % 97) as f64,
            )
        })
        .collect()
}

fn write_store(path: &str, rows: &[(i64, String, f64, f64, f64, f64, f64)], group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,keep\n");
    for (i, (t, s, o, h, l, c, v)) in rows.iter().enumerate() {
        text.push_str(&format!("{},{},{},{},{},{},{},{}\n", t, s, o, h, l, c, v, i));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
}

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-enrich-gpu-{}-{}-{}",
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

fn opts(data: &str, out: &str, features: &str, gpu: GpuMode) -> Options {
    Options {
        data: data.to_string(),
        out: Some(out.to_string()),
        features: vec![features.to_string()],
        labels: vec!["fwd:1h,cost=10bp,volnorm=20".to_string()],
        allow_lockbox: true,
        gpu,
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

fn close_enough(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() != b.is_nan() {
        return false;
    }
    if a == b {
        return true;
    }
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

const FEATURES: &str = "r1,r5,ma20,vol20,rsi6,rsi28,atr50,don20,vz96,bb20_bw,bb50_bw,\
    lag_r1_3,range1,clv1,hour_sin,dow_cos,ema20,macd,macd_sig,macd_hist,xs";

#[test]
fn gpu_columns_match_cpu_and_report_the_share() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("parity");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    let gpu_out2 = dir.join("gpu2.rcsb").to_string_lossy().into_owned();
    let mut rows = bars("AAA", 500_000, 0, 3600, Some(100));
    rows.extend(bars("BBB", 500_000, 0, 3600, Some(250_000)));
    write_store(&src, &rows, 65_536);

    let cpu = enrich::build(&opts(&src, &cpu_out, FEATURES, GpuMode::Off)).unwrap();
    let gpu = enrich::build(&opts(&src, &gpu_out, FEATURES, GpuMode::On)).unwrap();
    let gpu2 = enrich::build(&opts(&src, &gpu_out2, FEATURES, GpuMode::On)).unwrap();

    assert_eq!(cpu.gpu_columns, 0);
    assert_eq!(cpu.cpu_columns, cpu.features.len());
    assert!(gpu.gpu_columns > 0, "the cost model must dispatch a real kernel");
    assert_eq!(gpu.backend, "gpu");
    assert_eq!(
        gpu.gpu_columns,
        cpu.features.len(),
        "every column has a kernel; plans={:?} fallback={}",
        gpu.plans,
        gpu.gpu_fallback
    );
    assert_eq!(gpu.cpu_columns, 0);
    assert!((gpu.gpu_share - 1.0).abs() < 1e-12);
    assert_eq!(gpu.gpu_fallback, 0);
    assert!(gpu.gpu_kernel_seconds > 0.0);
    assert!(gpu.gpu_transfer_bytes > 0);
    assert!(gpu.json().contains("\"gpu\":{"), "JSON must carry the GPU share");
    assert!(
        gpu.json().contains("\"projection\":{\"hash\":"),
        "JSON must carry the projected read hash and column list: {}",
        gpu.json()
    );

    // Deterministic: a repeated GPU run writes the same values and hashes.
    let gpu_hashes: Vec<_> = gpu.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    let gpu2_hashes: Vec<_> = gpu2.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    assert_eq!(gpu_hashes, gpu2_hashes, "GPU enrichment must be deterministic");

    // Parity: NaN masks agree and finite values are within the pre-registered
    // 1e-12 relative tolerance (most sequential families are bit-identical).
    let mut worst = 0.0f64;
    for f in &cpu.features {
        let a = column(&cpu_out, &f.name);
        let b = column(&gpu_out, &f.name);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                close_enough(*x, *y),
                "column {} row {}: CPU {:?} vs GPU {:?}",
                f.name,
                i,
                x,
                y
            );
            if x.is_finite() && y.is_finite() {
                worst = worst.max((x - y).abs());
            }
        }
    }
    eprintln!("GPU/CPU parity max abs diff = {:.3e}", worst);
    eprintln!(
        "GPU share {:.3}, kernel {:.4}s, transfer {} bytes",
        gpu.gpu_share, gpu.gpu_kernel_seconds, gpu.gpu_transfer_bytes
    );
    eprintln!(
        "feature {:.3}s, d2h {:.3}s, gpu staging write {:.3}s, write_bw {:.0} B/s",
        gpu.feature_seconds, gpu.gpu_d2h_seconds, gpu.gpu_write_seconds, gpu.write_bw
    );
    assert!(gpu.gpu_d2h_seconds > 0.0, "D2H must be recorded separately");
    assert!(gpu.gpu_write_seconds > 0.0, "output staging write must be timed");
    assert!(gpu.write_bw > 0.0);

    // Same-column feature hashes are reported for both paths. The elementwise
    // and non-transcendental families are bit-identical; the report keeps the
    // actual hashes so a reviewer can diff them.
    let same = cpu
        .features
        .iter()
        .zip(gpu.features.iter())
        .filter(|(a, b)| a.name == b.name && a.hash == b.hash)
        .count();
    eprintln!("identical feature hashes: {}/{}", same, cpu.features.len());
}

#[test]
fn small_store_stays_on_the_cpu_with_a_reason() {
    let dir = tmp_dir("small");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_store(&src, &bars("AAA", 200, 0, 3600, None), 32);
    let r = enrich::build(&opts(&src, &out, "r1,ma20", GpuMode::On)).unwrap();
    assert_eq!(r.backend, "cpu");
    assert_eq!(r.gpu_columns, 0);
    let plan = r.plans.iter().find(|p| p.group == "returns").unwrap();
    assert_eq!(plan.backend, "cpu");
    assert!(plan.reason.contains("below"), "reason: {}", plan.reason);
}

/// Full-catalogue CPU/GPU parity on a large store.
///
/// Run the CLI twice on the same input store, once with `--gpu off` and once
/// with `--gpu auto` (or `on`), then point these variables at the two outputs:
///
/// ```sh
/// RCSR_ENRICH_PARITY_CPU=/data/cpu.rcsb \
/// RCSR_ENRICH_PARITY_GPU=/data/gpu.rcsb \
/// RCSR_ENRICH_PARITY_SPEC=all \
///   cargo test --release --features cuda --test enrich_gpu \
///   advertised_catalogue_matches_between_cpu_and_gpu_on_a_large_store -- --ignored --nocapture
/// ```
///
/// Every advertised column must exist, every NaN mask must agree and every
/// finite value must be within the pre-registered `1e-12` relative tolerance.
/// The test also prints how many columns are bit-identical; transcendentals
/// differ in the last bit because CUDA `log`/`sin`/`cos` are not the host libm.
#[test]
#[ignore = "needs a CPU/GPU output pair from the large-store harness (see docs/ohlcv.md)"]
fn advertised_catalogue_matches_between_cpu_and_gpu_on_a_large_store() {
    let cpu_path = match std::env::var("RCSR_ENRICH_PARITY_CPU") {
        Ok(v) => v,
        Err(_) => {
            eprintln!("skipping: RCSR_ENRICH_PARITY_CPU is not set");
            return;
        }
    };
    let gpu_path = std::env::var("RCSR_ENRICH_PARITY_GPU").expect("RCSR_ENRICH_PARITY_GPU");
    let spec = std::env::var("RCSR_ENRICH_PARITY_SPEC").unwrap_or_else(|_| "all".to_string());
    let cpu = FileStore::open(&cpu_path).unwrap();
    let gpu = FileStore::open(&gpu_path).unwrap();
    assert_eq!(cpu.n_rows(), gpu.n_rows(), "row count differs");
    let names = cpu.header().names();
    assert_eq!(names, gpu.header().names(), "schema differs between the two arms");

    let advertised = rcsr::enrich::expand(&spec).unwrap();
    for name in &advertised {
        assert!(names.iter().any(|n| n == name), "advertised column '{name}' is missing");
    }

    let mut compared = 0usize;
    let mut identical = 0usize;
    let mut worst: f64 = 0.0;
    for (ci, name) in names.iter().enumerate() {
        if name == "ts" || name == "symbol" {
            continue;
        }
        if cpu.header().dtypes()[ci] != DType::F64 || gpu.header().dtypes()[ci] != DType::F64 {
            continue;
        }
        let mut bit_same = true;
        let mut max_abs: f64 = 0.0;
        let mut check = 0usize;
        for g in 0..cpu.n_groups() {
            let a = cpu.read_f64_group(g, ci).unwrap();
            let b = gpu.read_f64_group(g, ci).unwrap();
            assert_eq!(a.len(), b.len());
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    x.is_finite() == y.is_finite(),
                    "column '{name}' group {g} row {i}: NaN mask differs (CPU {x:?}, GPU {y:?})"
                );
                if x.to_bits() != y.to_bits() {
                    bit_same = false;
                }
                if x.is_finite() && y.is_finite() {
                    check += 1;
                    if !close_enough(*x, *y) {
                        panic!(
                            "column '{name}' group {g} row {i}: CPU {x:?} vs GPU {y:?} beyond 1e-12"
                        );
                    }
                    max_abs = max_abs.max((x - y).abs());
                }
            }
        }
        assert!(check > 0, "column '{name}' has no finite values in the pair");
        compared += 1;
        if bit_same {
            identical += 1;
        }
        worst = worst.max(max_abs);
        if advertised.contains(name) {
            eprintln!(
                "column {name}: bit-identical={bit_same} max_abs_diff={max_abs:.3e}"
            );
        }
    }
    eprintln!(
        "compared {compared} numeric columns ({} advertised), bit-identical {identical}, \
         max abs diff {worst:.3e}",
        advertised.len()
    );
    assert!(
        compared >= advertised.len(),
        "compared only {compared} numeric columns for {} advertised names",
        advertised.len()
    );
}

/// #300: the batched output arena returns exactly the same values as the
/// per-column path, with one pinned chunked D2H per block and no per-column
/// download.
#[test]
fn batch_output_matches_per_column_downloads() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = 20_000usize;
    let step = 3600i64;
    let mut times = Vec::with_capacity(rows);
    let mut open = Vec::with_capacity(rows);
    let mut high = Vec::with_capacity(rows);
    let mut low = Vec::with_capacity(rows);
    let mut close = Vec::with_capacity(rows);
    let mut volume = Vec::with_capacity(rows);
    let mut qvolume = Vec::with_capacity(rows);
    for i in 0..rows {
        let p = 100.0 + (i as f64 * 0.013).sin() * 7.0;
        times.push(i as i64 * step + if i > 12_000 { 5 * step } else { 0 });
        open.push(p);
        high.push(p + 0.4);
        low.push(p - 0.3);
        close.push(p + (i as f64 * 0.11).sin() * 0.1);
        volume.push(10.0 + (i % 97) as f64);
        qvolume.push(1000.0 + (i % 89) as f64);
    }
    let starts = [0u32, 10_000u32];
    let lens = [10_000u32, 10_000u32];
    let mut gpu = rcsr::enrich_gpu::EnrichGpu::new(
        &times, &open, &high, &low, &close, &volume, &qvolume, &starts, &lens, step,
    )
    .expect("device context");
    assert!(
        gpu.batch_capacity() >= 3,
        "a 20k-row block should fit several columns in the 64 MiB arena"
    );

    // Reference: the existing one-column-at-a-time entry points.
    let want = vec![
        ("r5", gpu.ret(5).unwrap()),
        ("ma20", gpu.ma(20).unwrap()),
        ("vol20", gpu.vol(20).unwrap()),
        ("range1", gpu.range().unwrap()),
        ("clv1", gpu.clv().unwrap()),
        ("dow_sin", gpu.calendar(0).unwrap()),
        ("ema20", gpu.ema(20).unwrap()),
        ("rsi6", gpu.rsi_py(6).unwrap()),
        ("atr50", gpu.atr_py(50).unwrap()),
        ("don20", gpu.donchian(20).unwrap()),
        ("lag_r1_3", gpu.lag(3).unwrap()),
        ("vz96", gpu.vz(96).unwrap()),
        ("bb20_bw", gpu.bb_width(20).unwrap()),
    ];
    let (m, s, h) = gpu.macd().unwrap();

    use rcsr::enrich_gpu::BatchOp;
    let ops: Vec<BatchOp> = vec![
        BatchOp::ret(5),
        BatchOp::ma(20),
        BatchOp::vol(20),
        BatchOp::range(),
        BatchOp::clv(),
        BatchOp::calendar(0),
        BatchOp::ema(20),
        BatchOp::rsi(6),
        BatchOp::atr(50),
        BatchOp::donchian(20),
        BatchOp::lag(3),
        BatchOp::vz(96),
        BatchOp::bb_width(20),
        BatchOp::macd(),
    ];
    let before = gpu.stats();
    {
        let block = gpu.batch(&ops).expect("batched output block");
        assert_eq!(block.len(), want.len() + 3);
        for (i, (name, expected)) in want.iter().enumerate() {
            let got = block.column(i);
            assert_eq!(got.len(), expected.len(), "column {name} length");
            assert!(
                got.iter()
                    .zip(expected.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "column {name} differs"
            );
        }
        for (i, expected) in [m.as_slice(), s.as_slice(), h.as_slice()].iter().enumerate() {
            let got = block.column(want.len() + i);
            assert!(
                got.iter()
                    .zip(expected.iter())
                    .all(|(a, b)| a.to_bits() == b.to_bits()),
                "macd slot {i} differs"
            );
        }
    }
    let after = gpu.stats();
    let delta = after.since(&before);
    let expected_bytes = rows as u64 * 8 * (want.len() as u64 + 3);
    assert_eq!(delta.d2h_bytes, expected_bytes);
    assert!(delta.d2h_s > 0.0, "the pinned D2H window must be measured");
    assert!(delta.kernel_s > 0.0);
    eprintln!(
        "# batch block: {} columns, d2h {:.3} ms, kernel {:.3} ms",
        ops.iter().map(|o| o.slots()).sum::<usize>(),
        delta.d2h_s * 1e3,
        delta.kernel_s * 1e3
    );
}
