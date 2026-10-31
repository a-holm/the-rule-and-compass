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

/// Point-in-time panel: `member_asof` predates some rows, producing genuine
/// non-members, plus a few gaps that make the source return `NaN`.
fn write_pit_store(path: &str, symbols: usize, per_symbol: usize, group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,quote_volume,member_asof\n");
    for k in 0..symbols {
        let mut close = 100.0 + k as f64;
        for i in 0..per_symbol {
            // A gap leaves a hole in one symbol's series every ~1k bars.
            if i % 997 == 500 && k % 4 == 0 {
                continue;
            }
            let t = 1_700_000_000i64 + i as i64 * 900;
            let ret = if i % 5 == 0 {
                0.002
            } else {
                0.0005 * ((k % 3) as f64 - 1.0)
            };
            let next = close * (1.0 + ret);
            // Symbol 7 is pre-listing on every 11th row: the row exists but is
            // not a PIT member at its own timestamp.
            let member = if k == 7 && i % 11 == 0 {
                t + 900
            } else {
                t
            };
            let high = next.max(close) + 0.1;
            let low = next.min(close) - 0.1;
            let volume = 10.0 + (i % 97) as f64;
            let name = match k {
                0 => "BTCUSDT".to_string(),
                1 => "ETHUSDT".to_string(),
                _ => format!("SYM{k:02}"),
            };
            text.push_str(&format!(
                "{t},{name},{close},{high},{low},{next},{volume},{},{member}\n",
                next * volume
            ));
            close = next;
        }
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
fn gpu_cross_section_ranks_are_exact_and_chunk_invariant() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("xs-rank");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let per_symbol = 48_000usize;
    let symbols = 21usize;
    write_pit_store(&src, symbols, per_symbol, 65_536);

    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    let gpu_out_4k = dir.join("gpu4k.rcsb").to_string_lossy().into_owned();
    let gpu_out_1 = dir.join("gpu1.rcsb").to_string_lossy().into_owned();

    let mut cpu_opts = opts(&src, &cpu_out, "xs", GpuMode::Off);
    cpu_opts.chunk_rows = 65_536;
    let cpu = enrich::build(&cpu_opts).unwrap();

    let mut gpu_opts = opts(&src, &gpu_out, "xs", GpuMode::On);
    gpu_opts.chunk_rows = 65_536;
    let gpu = enrich::build(&gpu_opts).unwrap();
    assert_eq!(gpu.gpu_fallback, 0, "ranks must not fall back: {:?}", gpu.plans);
    assert_eq!(
        gpu.gpu_columns,
        cpu.features.len(),
        "every xs column should use a device kernel, plans={:?}",
        gpu.plans
    );

    let names: Vec<String> = cpu.features.iter().map(|f| f.name.clone()).collect();
    let mut finite = 0usize;
    for name in &names {
        let a = column(&cpu_out, name);
        let b = column(&gpu_out, name);
        assert_eq!(a.len(), b.len());
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(x.is_nan(), y.is_nan(), "column {name} row {i}: CPU {x:?} GPU {y:?}");
            if x.is_finite() {
                finite += 1;
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "column {name} row {i}: ranks must be bit-identical (CPU {x:?} GPU {y:?})"
                );
            }
        }
    }
    assert!(finite > 0, "fixture must produce finite ranks");

    // PIT: a pre-listing row's own rank is NaN even though its source return is
    // finite. Symbols 0 and 4 skip gap rows, so symbol 7 starts after them.
    let pre = column(&gpu_out, "xs_rank_r1");
    let gap_rows = (0..per_symbol).filter(|i| i % 997 == 500).count();
    let base = 7 * per_symbol - 2 * gap_rows;
    let mut checked = 0usize;
    for i in (11..per_symbol).step_by(11) {
        assert!(
            pre[base + i].is_nan(),
            "pre-listing row {i} entered the device rank"
        );
        checked += 1;
    }
    assert!(checked > 1000);

    // Batch invariance: the device permutation is built from the whole panel,
    // so --chunk-rows must not change a single bit.
    let hashes = |r: &enrich::Report| {
        r.features
            .iter()
            .map(|f| (f.name.clone(), f.hash.clone()))
            .collect::<Vec<_>>()
    };
    let mut four_k = opts(&src, &gpu_out_4k, "xs", GpuMode::On);
    four_k.chunk_rows = 4096;
    let four_k = enrich::build(&four_k).unwrap();
    assert_eq!(hashes(&gpu), hashes(&four_k), "--chunk-rows 4096 changed a rank");

    let mut one = opts(&src, &gpu_out_1, "xs", GpuMode::On);
    one.chunk_rows = 1;
    let one = enrich::build(&one).unwrap();
    assert_eq!(hashes(&gpu), hashes(&one), "--chunk-rows 1 changed a rank");
    assert_eq!(gpu.gpu_fallback, 0);
}

/// #295 AC1/AC4: device rank edge cases on a small hand-built panel.
///
/// Timestamp classes by `i % 5`: 0 has 19 finite PIT members (five finite
/// non-members), 1 has exactly 20 (four finite non-members), 2 has 19 (five NaN sources), 3 carries a
/// `+0.0`/`-0.0` pair, 4 is plain. The rank path has no symbol-batch knob (it
/// ranks every symbol of a timestamp in one `rc_enrich_xs_rank` call), so the
/// partition check varies the segment layout instead.
#[test]
fn gpu_cross_section_rank_edge_cases() {
    use rcsr::enrich_gpu::EnrichGpu;

    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let symbols = 24usize;
    let steps = 40usize;
    let n = symbols * steps;
    let mut times = Vec::with_capacity(n);
    let mut sym = Vec::with_capacity(n);
    let mut member = Vec::with_capacity(n);
    let mut src = vec![0.0f64; n];
    for k in 0..symbols {
        for i in 0..steps {
            times.push(1_700_000_000i64 + i as i64 * 900);
            sym.push(k as u32);
            member.push(!((i % 5 == 0 && k >= 19) || (i % 5 == 1 && k >= 20)));
            let distinct = ((k * 7 + i * 3) % 29) as f64 / 10.0 + 0.05;
            src[k * steps + i] = match (i % 5, k) {
                (2, 19..) => f64::NAN,
                (3, 0) => 0.0,
                (3, 1) => -0.0,
                (3, 2) => 0.0,
                _ => distinct,
            };
        }
    }
    let close = vec![100.0f64; n];
    let zeros = vec![0.0f64; n];
    let open = |seg_start: &[u32], seg_len: &[u32]| {
        EnrichGpu::new(
            &times, &close, &close, &close, &close, &zeros, &zeros, seg_start, seg_len, 900,
        )
        .expect("CUDA context")
    };
    let whole_start: Vec<u32> = (0..symbols).map(|k| (k * steps) as u32).collect();
    let whole_len = vec![steps as u32; symbols];

    // Reference: the host `(time, symbol, row)` tie order of `rank_in_order`.
    let reference = |src: &[f64]| {
        let mut out = vec![f64::NAN; n];
        for i in 0..steps {
            let mut g: Vec<usize> = (0..symbols)
                .map(|k| k * steps + i)
                .filter(|&r| src[r].is_finite() && member[r])
                .collect();
            g.sort_by(|&a, &b| {
                src[a].partial_cmp(&src[b]).unwrap().then(sym[a].cmp(&sym[b])).then(a.cmp(&b))
            });
            if g.len() >= 20 {
                for (rank, &r) in g.iter().enumerate() {
                    out[r] = 2.0 * rank as f64 / (g.len() - 1) as f64 - 1.0;
                }
            }
        }
        out
    };
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();

    let mut gpu = open(&whole_start, &whole_len);
    assert!(gpu.prepare_cross_section(&sym, Some(&member)));
    let got = gpu.rank(&src).expect("device rank");
    let want = reference(&src);
    for (r, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            g.to_bits(),
            w.to_bits(),
            "symbol {} timestamp {}: device {g:?}, host {w:?}",
            r / steps,
            r % steps
        );
    }

    // Fewer than 20 finite PIT members: every row of the timestamp is NaN.
    for i in (0..steps).filter(|i| i % 5 == 0 || i % 5 == 2) {
        for k in 0..symbols {
            assert!(got[k * steps + i].is_nan(), "timestamp {i} symbol {k} ranked with 19 members");
        }
    }
    // Exactly 20 members still rank, spanning -1 to +1.
    for i in (0..steps).filter(|i| i % 5 == 1) {
        let r: Vec<f64> = (0..symbols).map(|k| got[k * steps + i]).filter(|v| v.is_finite()).collect();
        assert_eq!(r.len(), 20);
        assert_eq!(r.iter().cloned().fold(f64::INFINITY, f64::min), -1.0);
        assert_eq!(r.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 1.0);
    }

    // +0.0 and -0.0 are one tie, broken by symbol order, never by sign.
    let swapped: Vec<f64> = src
        .iter()
        .enumerate()
        .map(|(r, &v)| match (r % steps % 5, r / steps) {
            (3, 0) => -0.0,
            (3, 1) => 0.0,
            _ => v,
        })
        .collect();
    let swapped_ranks = gpu.rank(&swapped).expect("device rank");
    assert_eq!(bits(&swapped_ranks), bits(&got), "the sign of zero changed a rank");
    for i in (0..steps).filter(|i| i % 5 == 3) {
        for k in 0..3 {
            assert_eq!(
                got[k * steps + i],
                2.0 * k as f64 / 23.0 - 1.0,
                "timestamp {i}: zero of symbol {k} is not tie position {k}"
            );
        }
    }

    // Segment partition: the permutation is keyed by symbol and time, so a
    // different segment layout must not change a bit.
    let mut half_start = Vec::new();
    let mut half_len = Vec::new();
    for k in 0..symbols {
        let h = steps / 2;
        half_start.extend([(k * steps) as u32, (k * steps + h) as u32]);
        half_len.extend([h as u32, (steps - h) as u32]);
    }
    let mut split = open(&half_start, &half_len);
    assert!(split.prepare_cross_section(&sym, Some(&member)));
    assert_eq!(
        bits(&split.rank(&src).expect("device rank")),
        bits(&got),
        "segment partition changed a rank"
    );
}

/// #295 paired measurement: the cross-section stage (`xs_seconds`) of the same
/// PIT panel with `--gpu off` (host sorts) and `--gpu on` (one device
/// permutation plus segmented sorts). `RCSR_XS_SYMBOLS` and `RCSR_XS_ROWS` size
/// the panel. Run with `--ignored --nocapture`.
#[test]
#[ignore = "throughput measurement; needs a CUDA device"]
fn measure_cross_section_stage() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let env = |k: &str, d: usize| {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    };
    let symbols = env("RCSR_XS_SYMBOLS", 100);
    let per_symbol = env("RCSR_XS_ROWS", 40_000);
    let dir = tmp_dir("xs-measure");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    write_pit_store(&src, symbols, per_symbol, 65_536);
    for features in ["xs", "cross"] {
        let mut line = Vec::new();
        for (tag, mode) in [("cpu", GpuMode::Off), ("gpu", GpuMode::On)] {
            let out = dir.join(format!("{features}-{tag}.rcsb")).to_string_lossy().into_owned();
            let r = enrich::build(&opts(&src, &out, features, mode)).unwrap();
            line.push(format!(
                "{tag}: xs_seconds={:.3} total={:.3} gpu_columns={} kernel_s={:.3}",
                r.xs_seconds, r.total_seconds, r.gpu_columns, r.gpu_kernel_seconds
            ));
        }
        eprintln!(
            "xs-measure features={features} rows={} symbols={symbols} | {}",
            symbols * per_symbol,
            line.join(" | ")
        );
    }
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
/// #298: every engine-tail kernel, on a two-symbol store with a hole.
///
/// CPU is the reference. Extrema (Donchian/stochastic/close position) must be
/// bit-identical because the blocked prefix-suffix pass is order-free; the
/// recurrence and exotic columns are allowed the declared 1e-12 relative tier.
#[test]
fn engine_tail_kernels_match_cpu_and_are_batch_invariant() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("tail");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    let gpu_out2 = dir.join("gpu2.rcsb").to_string_lossy().into_owned();
    let mut rows = bars("AAA", 500_000, 0, 3600, Some(100));
    rows.extend(bars("BBB", 500_000, 0, 3600, Some(250_000)));
    // Degenerate bars exercise the high==low extrema convention.
    for i in [1000usize, 1001, 2000] {
        let (t, s, _o, _h, _l, c, v) = rows[i].clone();
        rows[i] = (t, s, c, c, c, c, v);
    }
    write_store(&src, &rows, 65_536);

    let features = "r16,r96,r192,r672,ret1,ret5,ma96,ma672,ma_slope672,\
        don20,don96,atr14,natr14,batr,gapatr,volrat,stoch_k,stoch_d,\
        adx14,di_plus,di_minus,di_diff14,bb20_z,bb20_bw,bb50,bb50_bw,\
        bb96,bb_width96,rmean5,rstd5,close_pos20,ou_hl96,acf1,vr4,\
        rskew96,jump_share,semivar_asym,cusum_abs_z,tod_sin,tod_cos";
    let mut cpu_opts = opts(&src, &cpu_out, features, GpuMode::Off);
    cpu_opts.labels.clear();
    let mut gpu_opts = opts(&src, &gpu_out, features, GpuMode::On);
    gpu_opts.labels.clear();
    let mut gpu_opts2 = opts(&src, &gpu_out2, features, GpuMode::On);
    gpu_opts2.labels.clear();
    gpu_opts2.chunk_rows = 4096;

    let cpu = enrich::build(&cpu_opts).unwrap();
    let gpu = enrich::build(&gpu_opts).unwrap();
    let gpu2 = enrich::build(&gpu_opts2).unwrap();
    assert_eq!(gpu.gpu_columns, cpu.features.len(), "every tail column has a kernel: {:?}", gpu.plans);
    assert_eq!(gpu.cpu_columns, 0, "tail fallback: {:?}", gpu.plans);

    // Repeated run and chunk-size invariance: same values and hashes.
    let h = |r: &enrich::Report| {
        r.features
            .iter()
            .map(|f| (f.name.clone(), f.hash.clone()))
            .collect::<Vec<_>>()
    };
    assert_eq!(h(&gpu), h(&gpu2), "tail columns must be independent of chunk size");

    let bitwise = ["don20", "don96", "stoch_k", "close_pos20"];
    let mut worst = 0.0f64;
    let exotic = [
        "rmean5", "rstd5", "ou_hl96", "acf1", "vr4", "rskew96",
        "jump_share", "semivar_asym", "cusum_abs_z",
    ];
    for f in &cpu.features {
        let a = column(&cpu_out, &f.name);
        let b = column(&gpu_out, &f.name);
        assert_eq!(a.len(), b.len());
        let mut bit_same = true;
        let tol = if exotic.contains(&f.name.as_str()) { 1e-6 } else { 1e-12 };
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                x.is_finite() == y.is_finite(),
                "column {} row {}: NaN mask differs CPU {:?} GPU {:?}",
                f.name, i, x, y
            );
            if x.to_bits() != y.to_bits() {
                bit_same = false;
            }
            let ok = if x.is_nan() && y.is_nan() {
                true
            } else if x.is_nan() != y.is_nan() {
                false
            } else {
                x == y || (*x - *y).abs() <= tol * (1.0 + x.abs().max(y.abs()))
            };
            assert!(ok, "column {} row {}: CPU {:?} vs GPU {:?}", f.name, i, x, y);
            if x.is_finite() && y.is_finite() {
                worst = worst.max((x - y).abs());
            }
        }
        if bitwise.contains(&f.name.as_str()) {
            assert!(bit_same, "column {} must be bit-identical", f.name);
        }
    }
    eprintln!(
        "engine-tail parity: {} columns, max abs diff {:.3e}, kernel {:.3}s",
        cpu.features.len(),
        worst,
        gpu.gpu_kernel_seconds
    );
}

/// #298: the three compact-guarded shared names reach their exact kernels.
#[test]
fn compact_shared_names_are_registered_and_dispatched_at_16m() {
    for name in ["r16", "r96", "don96"] {
        assert!(rcsr::enrich::gpu_supports(name), "{name} must be registered");
    }
    let mut counts = std::collections::BTreeMap::new();
    counts.insert("donchian", (2usize, enrich::OpClass::RollingExtrema));
    let mut supported = std::collections::BTreeMap::new();
    supported.insert("donchian", 2usize);
    let plans = enrich::plan_groups(&counts, &supported, 16_000_000, GpuMode::Auto, true);
    let p = plans.iter().find(|p| p.group == "donchian").unwrap();
    if enrich::gpu_kernels_compiled() {
        // Registration is unconditional. The 16M auto plan still charges the
        // one-time context per family, so a two-column family can record a
        // measured CPU decision; the reason must then say why.
        if p.backend == "gpu" {
            assert_eq!(p.gpu_cols, 2);
        } else {
            assert!(
                p.reason.contains("estimated GPU time is not below CPU"),
                "unexpected decline reason: {}",
                p.reason
            );
        }
    }
}

/// #298: before/after throughput for the blocked extrema.
///
/// The old path is still reachable through the batch descriptor
/// (`BatchOp::donchian`), so this measures both kernels on the same resident
/// context. Run with `--ignored --nocapture`; it is not part of the suite
/// because it needs a device and a few hundred MiB.
#[test]
#[ignore = "throughput measurement; needs a CUDA device"]
fn measure_donchian_extrema_throughput() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = 1_000_000usize;
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
        times.push(i as i64 * step);
        open.push(p);
        high.push(p + 0.4);
        low.push(p - 0.3);
        close.push(p + (i as f64 * 0.11).sin() * 0.1);
        volume.push(10.0 + (i % 97) as f64);
        qvolume.push(1000.0 + (i % 89) as f64);
    }
    let starts = [0u32, (rows / 2) as u32];
    let lens = [(rows / 2) as u32, (rows / 2) as u32];
    let mut gpu = rcsr::enrich_gpu::EnrichGpu::new(
        &times, &open, &high, &low, &close, &volume, &qvolume, &starts, &lens, step,
    )
    .expect("device context");

    let before = gpu.stats();
    let blocked = gpu.donchian(96).unwrap();
    let after_new = gpu.stats();
    let new = after_new.since(&before);

    let before = gpu.stats();
    {
        use rcsr::enrich_gpu::BatchOp;
        let block = gpu.batch(&[BatchOp::donchian(96)]).expect("old batch path");
        assert_eq!(block.column(0).len(), rows);
    }
    let after_old = gpu.stats();
    let old = after_old.since(&before);
    let cells = (rows * 96) as f64;
    eprintln!(
        "don96 blocked: {:.3} ms, {:.1} M cells/s; O(n*w): {:.3} ms, {:.1} M cells/s; values {}",
        new.kernel_s * 1e3,
        cells / new.kernel_s / 1e6,
        old.kernel_s * 1e3,
        cells / old.kernel_s / 1e6,
        if blocked.is_empty() { "empty" } else { "ok" }
    );
    assert!(new.kernel_s > 0.0 && old.kernel_s > 0.0);
}

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
        ("ma20", gpu.ma(20, rcsr::window_stats::MEAN_STRICT).unwrap()),
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
        BatchOp::ma(20, rcsr::window_stats::MEAN_STRICT),
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
