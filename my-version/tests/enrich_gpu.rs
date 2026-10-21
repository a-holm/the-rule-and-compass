//! #200: GPU enrichment parity and dispatch.
//!
//! The CPU columns are the reference. These tests build a store above the cost
//! model's row floor, enrich it once with `--gpu off` and once with `--gpu on`,
//! and compare every column value plus the report's GPU accounting. Device tests
//! skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::store::{self, FileStore};

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
    assert_eq!(gpu.gpu_columns, cpu.features.len(), "every column has a kernel");
    assert_eq!(gpu.cpu_columns, 0);
    assert!((gpu.gpu_share - 1.0).abs() < 1e-12);
    assert_eq!(gpu.gpu_fallback, 0);
    assert!(gpu.gpu_kernel_seconds > 0.0);
    assert!(gpu.gpu_transfer_bytes > 0);
    assert!(gpu.json().contains("\"gpu\":{"), "JSON must carry the GPU share");

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
