//! #296: device MTF resample and causal as-of join parity.
//!
//! The CPU `mtf_all` columns are the reference. These tests cover the complete
//! 36-column family, a gap/stale bucket and a partial trailing bucket, prefix
//! replay at many cuts, and symbol-batch invariance. Direct context tests run
//! without the cost model's 1M-row dispatch floor; the store test drives the
//! real `--gpu on` path.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::mtf_gpu;
use rcsr::store::{self, FileStore};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-mtf-gpu-{}-{}-{}",
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

fn write_store(
    path: &str,
    rows: &[(i64, String, f64, f64, f64, f64, f64)],
    group_rows: usize,
) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,keep\n");
    for (i, (t, s, o, h, l, c, v)) in rows.iter().enumerate() {
        text.push_str(&format!("{},{},{},{},{},{},{},{}\n", t, s, o, h, l, c, v, i));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
}

fn opts(data: &str, out: &str, features: &str, gpu: GpuMode, chunk_rows: usize) -> Options {
    Options {
        data: data.to_string(),
        out: Some(out.to_string()),
        features: vec![features.to_string()],
        allow_lockbox: true,
        gpu,
        chunk_rows,
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

fn wavy(symbol: &str, n: usize, start: i64, step: i64, phase: f64, gap_at: Option<usize>) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..n)
        .map(|i| {
            let mut t = start + i as i64 * step;
            if let Some(g) = gap_at {
                if i >= g {
                    t += 3 * step;
                }
            }
            let x = i as f64;
            let p = 100.0 + 5.0 * (x * 0.37 + phase).sin() + 0.02 * x;
            let o = p + 0.4 * (x * 0.11 + phase).cos();
            let c = p + 0.5 * (x * 0.7).sin();
            let h = o.max(c) + 0.8 + (i % 5) as f64 * 0.03;
            let l = o.min(c) - 0.7 - (i % 3) as f64 * 0.02;
            (t, symbol.to_string(), o, h, l, c, 10.0 + (i % 7) as f64)
        })
        .collect()
}

fn split_segments(rows: &[(i64, String, f64, f64, f64, f64, f64)]) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
    let mut times = Vec::with_capacity(rows.len());
    let mut open = Vec::with_capacity(rows.len());
    let mut high = Vec::with_capacity(rows.len());
    let mut low = Vec::with_capacity(rows.len());
    let mut close = Vec::with_capacity(rows.len());
    let mut seg_start = Vec::new();
    let mut seg_len = Vec::new();
    let mut last = None;
    for (idx, (t, s, o, h, l, c, _)) in rows.iter().enumerate() {
        if last.as_deref() != Some(s.as_str()) {
            seg_start.push(idx as u32);
            if let Some(prev) = last.as_ref() {
                let start = *seg_start.get(seg_start.len() - 2).unwrap() as usize;
                seg_len.push((idx - start) as u32);
                let _ = prev;
            }
            last = Some(s.clone());
        }
        times.push(*t);
        open.push(*o);
        high.push(*h);
        low.push(*l);
        close.push(*c);
    }
    if let Some(start) = seg_start.last() {
        seg_len.push((rows.len() - *start as usize) as u32);
    }
    (times, open, high, low, close, seg_start, seg_len)
}

fn compute(rows: &[(i64, String, f64, f64, f64, f64, f64)], names: &[String]) -> std::collections::BTreeMap<String, Vec<f64>> {
    let (times, open, high, low, close, seg_start, seg_len) = split_segments(rows);
    let step = if rows.len() > 1 { rows[1].0 - rows[0].0 } else { 900 };
    let (cols, _) = mtf_gpu::compute(
        &times, &open, &high, &low, &close, &seg_start, &seg_len, step, names,
    )
    .expect("device MTF context");
    cols
}

fn close_enough(a: f64, b: f64) -> bool {
    if a.is_nan() && b.is_nan() {
        return true;
    }
    if a.is_nan() != b.is_nan() {
        return false;
    }
    a == b || (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn gpu_mtf_36_columns_match_cpu_and_dispatch_without_fallback() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // Above the 1M-row dispatch floor, with two symbols and a real gap.
    let mut rows = wavy("AAA", 525_000, 0, 900, 0.0, Some(100_000));
    rows.extend(wavy("BBB", 525_000, 0, 900, 1.7, Some(300_000)));
    let dir = tmp_dir("parity");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 65_536);

    let cpu = enrich::build(&opts(&src, &cpu_out, "mtf", GpuMode::Off, 65_536)).unwrap();
    let gpu = enrich::build(&opts(&src, &gpu_out, "mtf", GpuMode::On, 65_536)).unwrap();
    let auto_out = dir.join("auto.rcsb").to_string_lossy().into_owned();
    let auto = enrich::build(&opts(&src, &auto_out, "mtf", GpuMode::Auto, 65_536)).unwrap();

    assert_eq!(cpu.features.len(), 36);
    assert_eq!(gpu.features.len(), 36);
    assert_eq!(gpu.backend, "gpu");
    assert_eq!(gpu.gpu_columns, 36, "all 36 MTF columns must dispatch");
    assert_eq!(gpu.cpu_columns, 0);
    assert_eq!(gpu.gpu_fallback, 0);
    assert!(gpu.gpu_kernel_seconds > 0.0);
    assert!(gpu.gpu_transfer_bytes > 0);
    let plan = gpu.plans.iter().find(|p| p.group == "mtf").expect("mtf plan");
    assert_eq!(plan.backend, "gpu");
    assert_eq!(plan.gpu_cols, 36);

    // The 1.05M paired phase is slower on the device, so the new path stays
    // opt-in: `auto` keeps the CPU reference until a 16M crossover measurement.
    assert_eq!(auto.gpu_columns, 0);
    let auto_plan = auto.plans.iter().find(|p| p.group == "mtf").expect("auto mtf plan");
    assert_eq!(auto_plan.backend, "cpu");
    assert!(
        auto_plan.reason.contains("crossover") || auto_plan.reason.contains("not below"),
        "reason: {}",
        auto_plan.reason
    );
    for (a, b) in auto.features.iter().zip(cpu.features.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.hash, b.hash, "auto MTF must stay the CPU reference");
    }

    eprintln!(
        "MTF phase: CPU {:.3}s vs GPU {:.3}s (kernel {:.3}s, transfer {} bytes)",
        cpu.feature_seconds, gpu.feature_seconds, gpu.gpu_kernel_seconds, gpu.gpu_transfer_bytes
    );

    for name in enrich::mtf_names() {
        let a = column(&cpu_out, &name);
        let b = column(&gpu_out, &name);
        assert_eq!(a.len(), b.len(), "{name} length");
        let transcendental = name.ends_with("_vol");
        for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
            if transcendental {
                assert!(close_enough(*x, *y), "{name} row {i}: CPU {x:?} GPU {y:?}");
            } else {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "{name} row {i} must be bit-identical: CPU {x:?} GPU {y:?}"
                );
            }
        }
    }
}

#[test]
fn gap_is_nan_bar_age_is_finite_and_a_partial_bucket_is_never_visible() {
    // Twelve complete hours with hour 5 missing, then a partial hour 12.
    let step = 900i64;
    let mut rows = Vec::new();
    for hour in 0..12i64 {
        if hour == 5 {
            continue;
        }
        for q in 0..4i64 {
            let t = hour * 3600 + q * step;
            let p = 100.0 + hour as f64 + q as f64 * 0.1;
            rows.push((t, "AAA".to_string(), p, p + 0.5, p - 0.4, p + 0.05, 10.0));
        }
    }
    // Only three of the four 15m bars of hour 12 exist: the 12:00 bucket is
    // partial and must not be presented as a closed hourly bar.
    for q in 0..3i64 {
        let t = 12 * 3600 + q * step;
        let p = 120.0 + q as f64;
        rows.push((t, "AAA".to_string(), p, p + 0.5, p - 0.4, p + 0.05, 10.0));
    }
    let wanted = enrich::mtf_names()
        .into_iter()
        .filter(|n| n.ends_with("_ret1") || n.ends_with("_ret4") || n.ends_with("_close_pos") || n.ends_with("_bar_age"))
        .collect::<Vec<_>>();
    let cols = compute(&rows, &wanted);
    let times: Vec<i64> = rows.iter().map(|r| r.0).collect();
    let at = |t: i64| times.iter().position(|x| *x == t).unwrap();

    // First decision after the missing hour expects bucket 05, but the last
    // closed bucket is 04: values are NaN and age spans the hole exactly.
    let i = at(6 * 3600 + step);
    let ret1 = &cols["mtf_1h_ret1"];
    let pos = &cols["mtf_1h_close_pos"];
    let age = &cols["mtf_1h_bar_age"];
    assert!(ret1[i].is_nan(), "ret1 must not bridge the gap");
    assert!(pos[i].is_nan(), "close_pos must not bridge the gap");
    assert!(age[i].is_finite(), "bar_age must stay finite");
    assert_eq!(age[i], (6 * 3600 + step - 5 * 3600) as f64);

    // Bucket 06 exists, so its bar-position feature is valid, but ret4 still
    // spans the hole.
    let j = at(7 * 3600);
    assert!(cols["mtf_1h_close_pos"][j].is_finite());
    assert!(cols["mtf_1h_ret4"][j].is_nan());
    assert_eq!(cols["mtf_1h_bar_age"][j], 0.0);

    // At 12:30 the in-progress 12:00 bucket has not closed: the join uses 11:00
    // and reports 1800 s of staleness, never the partial bar.
    let k = at(12 * 3600 + 2 * step);
    assert_eq!(cols["mtf_1h_bar_age"][k], 1800.0);
    let hour_close = |h: i64| 100.0 + h as f64 + 0.3 + 0.05;
    let want = hour_close(11) / hour_close(10) - 1.0;
    assert!(close_enough(cols["mtf_1h_ret1"][k], want));
}

#[test]
fn a_missing_bucket_is_nan_for_every_timeframe() {
    // (timeframe, drop_lo hour, drop_hi hour, decision time)
    let cases = [
        ("1h", 5usize, 6usize, 6 * 3600 + 900),
        ("4h", 4usize, 8usize, 8 * 3600 + 900),
        ("1d", 24usize, 48usize, 48 * 3600 + 900),
    ];
    for (tf, drop_lo, drop_hi, decision) in cases {
        let mut rows = Vec::new();
        for hour in 0..144i64 {
            if (hour as usize) >= drop_lo && (hour as usize) < drop_hi {
                continue;
            }
            for q in 0..4i64 {
                let t = hour * 3600 + q * 900;
                let p = 100.0 + hour as f64 * 0.05 + q as f64 * 0.01;
                rows.push((t, "AAA".to_string(), p, p + 0.4, p - 0.3, p + 0.02, 10.0));
            }
        }
        let wanted = enrich::mtf_names()
            .into_iter()
            .filter(|n| n.starts_with(&format!("mtf_{tf}_")))
            .collect::<Vec<_>>();
        let cols = compute(&rows, &wanted);
        let times: Vec<i64> = rows.iter().map(|r| r.0).collect();
        let i = times.iter().position(|t| *t == decision).unwrap();
        for leaf in enrich::MTF_LEAVES {
            let name = format!("mtf_{tf}_{leaf}");
            let got = &cols[&name];
            if *leaf == "bar_age" {
                let expected = (decision - (drop_lo as i64 * 3600)) as f64;
                assert_eq!(got[i], expected, "{name} bar_age after a gap");
            } else {
                assert!(got[i].is_nan(), "{name} must be NaN when its bucket is missing");
            }
        }
    }
}

#[test]
fn prefix_replay_at_many_cuts_is_bit_identical() {
    let mut rows = wavy("AAA", 4_000, 0, 900, 0.0, Some(1_234));
    rows.extend(wavy("BBB", 4_000, 0, 900, 2.1, Some(777)));
    let wanted = enrich::mtf_names();
    let full = compute(&rows, &wanted);
    // ~20 evenly spaced cuts, including before/after symbol boundaries.
    let step = (rows.len() / 20).max(1);
    for cut in (1..=rows.len()).step_by(step) {
        let prefix = &rows[..cut];
        let got = compute(prefix, &wanted);
        for name in &wanted {
            let a = &full[name];
            let b = &got[name];
            assert_eq!(b.len(), cut);
            for i in 0..cut {
                assert_eq!(
                    a[i].to_bits(),
                    b[i].to_bits(),
                    "{name} changed at row {i} under prefix cut {cut}"
                );
            }
        }
    }
}

#[test]
fn symbol_batches_and_chunk_sizes_write_the_same_bytes() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let wanted = enrich::mtf_names();
    let a = wavy("AAA", 5_000, 0, 900, 0.0, Some(1_000));
    let b = wavy("BBB", 5_000, 0, 900, 1.1, Some(2_000));
    let mut both = a.clone();
    both.extend(b.clone());
    let combined = compute(&both, &wanted);
    let left = compute(&a, &wanted);
    let right = compute(&b, &wanted);
    for name in &wanted {
        let mut joined = left[name].clone();
        joined.extend(right[name].iter().copied());
        for (i, (x, y)) in combined[name].iter().zip(joined.iter()).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name} symbol batch differs at {i}");
        }
    }

    // The real enrich path ignores --chunk-rows for the device pass; two chunk
    // sizes must still produce identical column hashes.
    let mut rows = wavy("AAA", 525_000, 0, 900, 0.0, Some(50_000));
    rows.extend(wavy("BBB", 525_000, 0, 900, 0.9, Some(400_000)));
    let dir = tmp_dir("batch");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out_a = dir.join("a.rcsb").to_string_lossy().into_owned();
    let out_b = dir.join("b.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 65_536);
    let ra = enrich::build(&opts(&src, &out_a, "mtf", GpuMode::On, 7)).unwrap();
    let rb = enrich::build(&opts(&src, &out_b, "mtf", GpuMode::On, 65_536)).unwrap();
    assert_eq!(ra.gpu_columns, 36);
    assert_eq!(rb.gpu_columns, 36);
    let ha: Vec<_> = ra.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    let hb: Vec<_> = rb.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    assert_eq!(ha, hb, "chunk size changed the MTF bytes");
}

fn holey(symbol: &str, span: usize, step: i64, hole_every: usize, phase: f64) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let all = wavy(symbol, span, 0, step, phase, None);
    all.into_iter()
        .enumerate()
        .filter(|(i, _)| hole_every == 0 || i % hole_every != hole_every / 2)
        .map(|(_, r)| r)
        .collect()
}

#[test]
fn long_holey_symbol_stays_on_the_device() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // 21,000-bar span, 21 single-bar holes: 20,979 rows.
    let rows = holey("AAA", 21_000, 900, 1_000, 0.0);
    assert_eq!(rows.len(), 20_979);
    let wanted = enrich::mtf_names();
    let cols = compute(&rows, &wanted);
    assert_eq!(cols.len(), 36);

    let dir = tmp_dir("holey");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 4_096);
    enrich::build(&opts(&src, &cpu_out, "mtf", GpuMode::Off, 4_096)).unwrap();
    for name in &wanted {
        let want = column(&cpu_out, name);
        let got = &cols[name];
        assert_eq!(want.len(), got.len(), "{name} length");
        for (i, (x, y)) in want.iter().zip(got.iter()).enumerate() {
            if name.ends_with("_vol") {
                assert!(close_enough(*x, *y), "{name} row {i}: CPU {x:?} GPU {y:?}");
            } else {
                assert_eq!(x.to_bits(), y.to_bits(), "{name} row {i}: CPU {x:?} GPU {y:?}");
            }
        }
    }
}

fn sweep_rows(nseg: usize, step: i64, hole_k: usize, big_gap: bool) -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    let mut rows = Vec::new();
    for s in 0..nseg {
        let sym = format!("S{s}");
        let bars = 3_000usize;
        let mut shift = (s as i64 * 1_234) % 3_600;
        for i in 0..bars {
            if big_gap && s == 0 && i == bars / 2 {
                shift += 2 * 86_400 + 17;
            }
            if hole_k > 0 && (i * 7 + s) % hole_k == 0 {
                continue;
            }
            let t = shift + i as i64 * step;
            let x = (i + s) as f64;
            let p = 50.0 + 3.0 * (x * 0.21).sin();
            rows.push((t, sym.clone(), p, p + 1.0, p - 1.0, p + 0.3, 5.0));
        }
    }
    rows
}

#[test]
fn bucket_capacity_bounds_every_timeframe() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    for step in [60i64, 300, 420, 900, 3600] {
        for nseg in [1usize, 3, 7] {
            for hole_k in [0usize, 997, 50, 3] {
                for big_gap in [false, true] {
                    let rows = sweep_rows(nseg, step, hole_k, big_gap);
                    let (times, open, high, low, close, seg_start, seg_len) = split_segments(&rows);
                    let capacity = mtf_gpu::bucket_capacity(&times, &seg_start, &seg_len);
                    let case = format!("step {step} nseg {nseg} hole 1/{hole_k} gap {big_gap}");
                    assert!(capacity <= times.len(), "{case}: capacity {capacity} > n {}", times.len());
                    let mut gpu = mtf_gpu::MtfGpu::new(&times, &open, &high, &low, &close, &seg_start, &seg_len, step)
                        .unwrap_or_else(|e| panic!("{case}: {e}"));
                    assert_eq!(gpu.capacity(), capacity);
                    for tf in [3_600i64, 14_400, 86_400] {
                        gpu.prepare(tf).unwrap_or_else(|e| panic!("{case} tf {tf}: {e}"));
                        assert!(
                            gpu.total_buckets() <= capacity,
                            "{case} tf {tf}: {} buckets > capacity {capacity}",
                            gpu.total_buckets()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn sub_hour_and_non_hour_multiple_timeframes_are_rejected_with_their_own_code() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = holey("AAA", 2_000, 900, 0, 0.0);
    let (times, open, high, low, close, seg_start, seg_len) = split_segments(&rows);
    let mut gpu = mtf_gpu::MtfGpu::new(&times, &open, &high, &low, &close, &seg_start, &seg_len, 900).unwrap();
    for tf in [900i64, 1_800, 5_400] {
        let err = gpu.prepare(tf).unwrap_err();
        assert_eq!((err.stage, err.code), ("prepare", 6), "tf {tf}");
    }
    gpu.prepare(3_600).unwrap();
}

// 48 long holey symbols: 1,006,992 rows, above the 1M-row dispatch floor.
fn big_holey_rows() -> Vec<(i64, String, f64, f64, f64, f64, f64)> {
    (0..48)
        .flat_map(|s| holey(&format!("H{s:02}"), 21_000, 900, 1_000, s as f64 * 0.3))
        .collect()
}

#[test]
fn enrich_on_the_long_holey_fixture_reports_no_fallback() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = big_holey_rows();
    let dir = tmp_dir("holey-enrich");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 65_536);
    let r = enrich::build(&opts(&src, &out, "mtf", GpuMode::On, 65_536)).unwrap();
    assert_eq!(r.gpu_fallback, 0);
    assert_eq!(r.gpu_columns, 36);
}

#[test]
fn a_forced_device_failure_warns_once_and_matches_the_cpu_bytes() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = big_holey_rows();
    let dir = tmp_dir("forced");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let on_out = dir.join("on.rcsb").to_string_lossy().into_owned();
    let off_out = dir.join("off.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 65_536);
    let run = |out: &str, gpu: &str, force: bool| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"));
        cmd.args(["enrich", "--data", &src, "--out", out, "--features", "mtf", "--allow-lockbox", "--gpu", gpu]);
        if force {
            cmd.env("RCSR_MTF_GPU_FORCE_TF_SECONDS", "900");
        }
        cmd.output().unwrap()
    };
    let on = run(&on_out, "on", true);
    assert!(on.status.success(), "{}", String::from_utf8_lossy(&on.stderr));
    let stderr = String::from_utf8_lossy(&on.stderr);
    let warnings: Vec<&str> = stderr.lines().filter(|l| l.contains("device MTF failed")).collect();
    assert_eq!(warnings.len(), 1, "stderr: {stderr}");
    assert!(warnings[0].contains("stage prepare code 6"), "{}", warnings[0]);
    let off = run(&off_out, "off", false);
    assert!(off.status.success());
    for name in enrich::mtf_names() {
        let a = column(&on_out, &name);
        let b = column(&off_out, &name);
        assert!(a.iter().zip(b.iter()).all(|(x, y)| x.to_bits() == y.to_bits()), "{name}");
    }
}

extern "C" {
    fn rc_mtf_test_force_cub_fail(on: i32);
}

/// #410: a refused CUB scan returns the scan's own status code, which the
/// caller turns into a labelled CPU fallback, and the next prepare works.
#[test]
fn refused_cub_scan_fails_prepare_with_its_own_code_then_recovers() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let rows = holey("AAA", 2_000, 900, 0, 0.0);
    let (times, open, high, low, close, seg_start, seg_len) = split_segments(&rows);
    let mut gpu = mtf_gpu::MtfGpu::new(&times, &open, &high, &low, &close, &seg_start, &seg_len, 900).unwrap();

    unsafe { rc_mtf_test_force_cub_fail(1) };
    let err = gpu.prepare(3_600).unwrap_err();
    unsafe { rc_mtf_test_force_cub_fail(0) };
    assert_eq!((err.stage, err.code), ("prepare", 3));

    gpu.prepare(3_600).expect("no error may stay pending");
}
