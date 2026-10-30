//! #297: fused flow and price-action producers.
//!
//! The CPU producers `flow_all` and `price_action_all` are the reference. The
//! device tests build a 1,000,000-row store with gaps, zero volume, zero
//! trades and `H == L` bars, enrich it with `--gpu off` and `--gpu on`, and
//! compare every column at its declared tier. Device tests skip cleanly when no
//! CUDA device is present.

use rcsr::features_gpu as fg;

fn assert_registry_tiers() {
    let reg = rcsr::enrich::default_registry();
    for col in fg::FLOW_DEVICE_COLUMNS.iter().chain(fg::PRICE_ACTION_DEVICE_COLUMNS.iter()) {
        let entry = reg
            .iter()
            .find(|e| e.name == col.name)
            .unwrap_or_else(|| panic!("{} has no registry entry", col.name));
        assert!(entry.available, "{} must be available", col.name);
        assert_eq!(
            entry.parity_tier,
            col.tier.as_str(),
            "{} parity tier",
            col.name
        );
    }
    for name in ["derivatives"] {
        let e = reg.iter().find(|e| e.family == name).expect("derivative entry");
        assert_eq!(e.parity_tier, "unavailable");
    }
}

#[test]
fn device_column_inventory_and_tiers_are_declared() {
    assert_eq!(fg::FLOW_DEVICE_COLUMNS.len(), 12);
    assert_eq!(fg::PRICE_ACTION_DEVICE_COLUMNS.len(), 11);
    // The flow registry family is 17 columns: five raw flow inputs, ten
    // canonical FLOW_NAMES, and cmf20/vwap20_dev. The fused producer emits the
    // twelve derived columns (ten canonical + the two engine names).
    let reg = rcsr::enrich::default_registry();
    assert_eq!(reg.iter().filter(|e| e.family == "flow").count(), 17);
    assert_eq!(
        reg.iter().filter(|e| e.family == "price_action").count(),
        11
    );
    assert_eq!(fg::base_passes("flow"), 2);
    assert_eq!(fg::base_passes("price_action"), 2);
    assert_eq!(fg::base_passes("returns"), 0);

    let flow_engine = [
        "vwap",
        "taker_ratio96",
        "taker_imb16",
        "trade_intensity96",
        "avg_trade_size96",
        "amihud96",
        "zero_ret_share96",
        "obv_z96",
        "cvd_proxy",
        "vwap_dev96",
        "cmf20",
        "vwap20_dev",
    ];
    for name in flow_engine {
        assert!(fg::is_device_flow(name), "{name} must be a device flow column");
    }
    for (name, tier) in [
        ("trade_intensity96", "1e-12"),
        ("avg_trade_size96", "1e-12"),
        ("amihud96", "1e-12"),
        ("vwap", "bitwise"),
        ("cvd_proxy", "bitwise"),
    ] {
        assert_eq!(fg::device_tier(name), Some(tier), "{name}");
    }
    for name in [
        "clv16",
        "body16",
        "rng_atr",
        "prev_range_pos",
        "inside_s",
        "outside_s",
        "engulf_s",
        "harami_s",
        "run8",
        "nr4",
        "trend96",
    ] {
        assert_eq!(fg::device_tier(name), Some("bitwise"), "{name}");
    }
    assert_registry_tiers();
}

#[test]
fn fused_cost_model_is_sublinear_in_columns() {
    let one = fg::estimate_fused_ms("flow", 1_000_000, 1).unwrap();
    let twelve = fg::estimate_fused_ms("flow", 1_000_000, 12).unwrap();
    assert!(one > 0.0);
    // The base pass is paid once; adding a column must cost less than a full
    // new pass. This is the property that makes fusion worth it.
    assert!(
        twelve < 12.0 * one,
        "12 columns ({twelve}) must cost less than 12 separate passes ({})",
        12.0 * one
    );
    assert!(fg::estimate_fused_ms("returns", 1_000_000, 1).is_none());
}

#[cfg(feature = "cuda")]
#[test]
fn fused_families_dispatch_only_with_enough_segments() {
    use std::collections::BTreeMap;
    let mut counts = BTreeMap::new();
    counts.insert("flow", (12usize, rcsr::enrich::OpClass::RollingStd));
    let mut supported = BTreeMap::new();
    supported.insert("flow", 12usize);

    // 109 symbols and 1 M rows: the measured fused cost beats the measured CPU
    // reference estimate, so `auto` dispatches.
    let panel = rcsr::enrich::plan_groups_with_segments(
        &counts,
        &supported,
        1_000_000,
        109,
        rcsr::enrich::GpuMode::Auto,
        true,
    );
    assert_eq!(panel[0].backend, "gpu", "reason: {}", panel[0].reason);

    // Two symbols cannot fill the segment-parallel consumers; the cost model
    // declines instead of silently running a slow kernel.
    let pair = rcsr::enrich::plan_groups_with_segments(
        &counts,
        &supported,
        1_000_000,
        2,
        rcsr::enrich::GpuMode::Auto,
        true,
    );
    assert_eq!(pair[0].backend, "cpu");
    assert!(
        pair[0].reason.contains("segment floor"),
        "reason: {}",
        pair[0].reason
    );

    // `--gpu on` still forces the parity fixture.
    let forced = rcsr::enrich::plan_groups_with_segments(
        &counts,
        &supported,
        1_000_000,
        2,
        rcsr::enrich::GpuMode::On,
        true,
    );
    assert_eq!(forced[0].backend, "gpu");
}

#[cfg(feature = "cuda")]
mod device {
    use super::fg;
    use rcsr::data;
    use rcsr::enrich::{self, GpuMode, Options};
    use rcsr::enrich_gpu::EnrichGpu;
    use rcsr::store::{self, FileStore};

    const STEP: i64 = 3600;
    const FEATURES: &str = "flow,price_action,cmf20,vwap20_dev";

    #[derive(Clone)]
    struct Row {
        ts: i64,
        symbol: &'static str,
        open: f64,
        high: f64,
        low: f64,
        close: f64,
        volume: f64,
        quote_volume: f64,
        trades: f64,
        taker_buy_base: f64,
        taker_buy_quote: f64,
    }

    fn rows(symbol: &'static str, n: usize, start: i64, gap_at: usize) -> Vec<Row> {
        (0..n)
            .map(|i| {
                let mut t = start + i as i64 * STEP;
                if i >= gap_at {
                    t += 3 * STEP;
                }
                let base = 100.0 + (i as f64 * 0.013).sin() * 7.0 + i as f64 * 0.0001;
                let zero_volume = i % 997 == 0;
                let zero_trades = i % 997 == 1;
                let flat = i % 991 == 0;
                let volume = if zero_volume { 0.0 } else { 10.0 + (i % 97) as f64 };
                let high = if flat { base } else { base + 0.3 + (i as f64 * 0.07).cos().abs() * 0.2 };
                let low = if flat { base } else { base - 0.4 - (i as f64 * 0.05).sin().abs() * 0.2 };
                let close = if flat { base } else { base + (i as f64 * 0.11).sin() * 0.1 };
                let trades = if zero_trades { 0.0 } else { 1.0 + (i % 31) as f64 };
                let taker_buy_base = volume * (0.4 + 0.2 * (i as f64 * 0.03).cos());
                Row {
                    ts: t,
                    symbol,
                    open: base,
                    high,
                    low,
                    close,
                    volume,
                    quote_volume: close * volume * (1.0 + 0.0001 * (i % 7) as f64),
                    trades,
                    taker_buy_base,
                    taker_buy_quote: close * taker_buy_base,
                }
            })
            .collect()
    }

    fn write_store(path: &str, rows: &[Row], group_rows: usize) {
        let mut text = String::from(
            "ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote\n",
        );
        for r in rows {
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{},{},{},{}\n",
                r.ts,
                r.symbol,
                r.open,
                r.high,
                r.low,
                r.close,
                r.volume,
                r.quote_volume,
                r.trades,
                r.taker_buy_base,
                r.taker_buy_quote
            ));
        }
        let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
        store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
    }

    fn tmp_dir(tag: &str) -> std::path::PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "rcsr-features-gpu-{}-{}-{}",
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

    fn opts(data: &str, out: &str, gpu: GpuMode, chunk_rows: usize) -> Options {
        Options {
            data: data.to_string(),
            out: Some(out.to_string()),
            features: vec![FEATURES.to_string()],
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

    fn close_enough(a: f64, b: f64) -> bool {
        if a.is_nan() && b.is_nan() {
            return true;
        }
        if a.is_nan() != b.is_nan() {
            return false;
        }
        a == b || (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
    }

    /// Flatten `rows` into the resident device context shape.
    fn flat(rows: &[Row]) -> (Vec<i64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<u32>, Vec<u32>) {
        let mut ts = Vec::with_capacity(rows.len());
        let mut o = Vec::with_capacity(rows.len());
        let mut h = Vec::with_capacity(rows.len());
        let mut l = Vec::with_capacity(rows.len());
        let mut c = Vec::with_capacity(rows.len());
        let mut v = Vec::with_capacity(rows.len());
        let mut qv = Vec::with_capacity(rows.len());
        let mut tr = Vec::with_capacity(rows.len());
        let mut tb = Vec::with_capacity(rows.len());
        let mut tbq = Vec::with_capacity(rows.len());
        let mut starts = Vec::new();
        let mut lens = Vec::new();
        let mut prev: Option<&str> = None;
        for (i, r) in rows.iter().enumerate() {
            if prev != Some(r.symbol) {
                starts.push(i as u32);
                lens.push(0u32);
                prev = Some(r.symbol);
            }
            if let Some(last) = lens.last_mut() {
                *last += 1;
            }
            ts.push(r.ts);
            o.push(r.open);
            h.push(r.high);
            l.push(r.low);
            c.push(r.close);
            v.push(r.volume);
            qv.push(r.quote_volume);
            tr.push(r.trades);
            tb.push(r.taker_buy_base);
            tbq.push(r.taker_buy_quote);
        }
        (ts, o, h, l, c, v, qv, tr, tb, tbq, starts, lens)
    }

    fn all_present() -> u32 {
        fg::PRESENT_VOLUME
            | fg::PRESENT_QUOTE_VOLUME
            | fg::PRESENT_TRADES
            | fg::PRESENT_TAKER_BUY
            | fg::PRESENT_TAKER_BUY_QV
    }

    fn context(rows: &[Row]) -> Option<EnrichGpu> {
        let (ts, o, h, l, c, v, qv, tr, tb, tbq, starts, lens) = flat(rows);
        EnrichGpu::new_with_flow(
            &ts,
            &o,
            &h,
            &l,
            &c,
            &v,
            &qv,
            &tr,
            &tb,
            &tbq,
            all_present(),
            &starts,
            &lens,
            STEP,
        )
    }

    #[test]
    fn fused_flow_reads_optional_raw_columns_once() {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let rows = rows("AAA", 2_000, 0, 700);
        let mut gpu = context(&rows).unwrap();
        let names: Vec<&str> = fg::FLOW_DEVICE_COLUMNS.iter().map(|c| c.name).collect();
        let out = gpu.flow_all(&names).expect("fused flow");
        assert_eq!(out.len(), 12);
        assert_eq!(
            gpu.base_reads(),
            2,
            "the fused base must read trades and taker_buy_base once"
        );
        assert_eq!(gpu.base_passes(), 2, "elementwise base + cumulative state");
        // A second family run re-runs the base; the per-column count must not
        // grow with the twelve consumers.
        let _ = gpu.flow_all(&names).expect("fused flow rerun");
        assert_eq!(gpu.base_reads(), 4);
    }

    #[test]
    fn missing_raw_columns_produce_nan_not_zero() {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let rows = rows("AAA", 1_000, 0, 300);
        let (ts, o, h, l, c, v, qv, _, tb, tbq, starts, lens) = flat(&rows);
        let nan = vec![f64::NAN; rows.len()];
        let mut gpu = EnrichGpu::new_with_flow(
            &ts,
            &o,
            &h,
            &l,
            &c,
            &v,
            &qv,
            &nan,
            &tb,
            &tbq,
            fg::PRESENT_VOLUME
                | fg::PRESENT_QUOTE_VOLUME
                | fg::PRESENT_TAKER_BUY
                | fg::PRESENT_TAKER_BUY_QV,
            &starts,
            &lens,
            STEP,
        )
        .unwrap();
        let out = gpu
            .flow_all(&["avg_trade_size96", "trade_intensity96"])
            .expect("fused flow without trades");
        for (name, vals) in out {
            assert!(
                vals.iter().all(|v| v.is_nan()),
                "{name} must be NaN when the raw trades column is absent"
            );
        }
        assert_eq!(gpu.base_reads(), 1, "only taker_buy_base was present");
    }

    #[test]
    fn fused_gpu_is_prefix_causal() {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let dir = tmp_dir("prefix");
        let full_src = dir.join("full.rcsb").to_string_lossy().into_owned();
        let cut_src = dir.join("cut.rcsb").to_string_lossy().into_owned();
        let full_out = dir.join("full-out.rcsb").to_string_lossy().into_owned();
        let cut_out = dir.join("cut-out.rcsb").to_string_lossy().into_owned();
        let full_rows = rows("AAA", 2_000, 0, 700);
        let cut_rows = full_rows[..1_500].to_vec();
        write_store(&full_src, &full_rows, 4_096);
        write_store(&cut_src, &cut_rows, 4_096);
        let full = enrich::build(&opts(&full_src, &full_out, GpuMode::On, 4_096)).unwrap();
        let cut = enrich::build(&opts(&cut_src, &cut_out, GpuMode::On, 4_096)).unwrap();
        assert_eq!(full.features.len(), cut.features.len());
        for f in &full.features {
            let a = column(&full_out, &f.name);
            let b = column(&cut_out, &f.name);
            for (i, (x, y)) in a.iter().take(cut_rows.len()).zip(b.iter()).enumerate() {
                assert!(
                    x.is_nan() == y.is_nan() && x.to_bits() == y.to_bits(),
                    "column {} row {}: a future bar changed the prefix (full {:?}, cut {:?})",
                    f.name,
                    i,
                    x,
                    y
                );
            }
        }
    }

    #[test]
    fn flow_and_price_action_match_the_cpu_reference() {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        let dir = tmp_dir("parity");
        let src = dir.join("in.rcsb").to_string_lossy().into_owned();
        let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
        let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
        let gpu_out2 = dir.join("gpu2.rcsb").to_string_lossy().into_owned();
        let gpu_out_chunk = dir.join("gpu-chunk.rcsb").to_string_lossy().into_owned();

        let mut all = rows("AAA", 500_000, 0, 120);
        all.extend(rows("BBB", 500_000, 0, 250_000));
        write_store(&src, &all, 65_536);

        let cpu = enrich::build(&opts(&src, &cpu_out, GpuMode::Off, 65_536)).unwrap();
        let gpu = enrich::build(&opts(&src, &gpu_out, GpuMode::On, 65_536)).unwrap();
        let gpu2 = enrich::build(&opts(&src, &gpu_out2, GpuMode::On, 65_536)).unwrap();
        let chunk = enrich::build(&opts(&src, &gpu_out_chunk, GpuMode::On, 1)).unwrap();

        assert_eq!(cpu.gpu_columns, 0);
        assert_eq!(gpu.features.len(), 23, "10 flow + 2 engine flow + 11 price action");
        assert_eq!(gpu.gpu_columns, 23, "all fused columns dispatched: {:?}", gpu.plans);
        assert_eq!(gpu.cpu_columns, 0);
        assert_eq!(gpu.gpu_fallback, 0);
        assert_eq!(gpu.backend, "gpu");
        assert!(
            gpu.projection_hash.is_some(),
            "the fused parity fixture must run through the #294 projected transport"
        );
        for raw in ["trades", "taker_buy_base", "taker_buy_quote"] {
            assert!(
                gpu.projection_columns.iter().any(|c| c == raw),
                "{raw} must be in the projected column list"
            );
        }
        assert!(gpu.gpu_kernel_seconds > 0.0);
        assert!(gpu.gpu_transfer_bytes > 0);

        // The full flow family is 17 registered columns. Five are raw inputs,
        // copied through by enrich; the five raw flow fields must be present in
        // the fixture, and all twelve derived flow plus eleven price-action
        // columns must be in the output.
        let out_store = FileStore::open(&gpu_out).unwrap();
        for raw in [
            "volume",
            "quote_volume",
            "trades",
            "taker_buy_base",
            "taker_buy_quote",
        ] {
            assert!(
                out_store.header().column_index(raw).is_some(),
                "raw flow column {raw} missing from the full-set fixture"
            );
        }
        for col in fg::FLOW_DEVICE_COLUMNS
            .iter()
            .chain(fg::PRICE_ACTION_DEVICE_COLUMNS.iter())
        {
            assert!(
                out_store.header().column_index(col.name).is_some(),
                "{} missing from the fused output",
                col.name
            );
        }

        // Repeated runs and different chunk sizes hash identically.
        let hashes = |r: &enrich::Report| -> Vec<(String, String)> {
            r.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect()
        };
        assert_eq!(hashes(&gpu), hashes(&gpu2), "repeated GPU run must be identical");
        assert_eq!(
            hashes(&gpu),
            hashes(&chunk),
            "--chunk-rows must not change arithmetic"
        );

        let mut worst = 0.0f64;
        let mut bitwise = 0usize;
        let mut failures: Vec<String> = Vec::new();
        for f in &cpu.features {
            let a = column(&cpu_out, &f.name);
            let b = column(&gpu_out, &f.name);
            let tier = fg::device_tier(&f.name)
                .unwrap_or_else(|| panic!("{} has no declared tier", f.name));
            let mut same_bits = true;
            let mut first: Option<String> = None;
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                if x.is_nan() != y.is_nan() {
                    first.get_or_insert_with(|| {
                        format!("column {} row {}: NaN mask differs (CPU {x:?}, GPU {y:?})", f.name, i)
                    });
                    continue;
                }
                if x.to_bits() != y.to_bits() {
                    same_bits = false;
                }
                if tier == "bitwise" && x.to_bits() != y.to_bits() {
                    first.get_or_insert_with(|| {
                        format!("column {} row {}: declared bitwise (CPU {x:?}, GPU {y:?})", f.name, i)
                    });
                } else if tier != "bitwise" && !close_enough(*x, *y) {
                    first.get_or_insert_with(|| {
                        format!("column {} row {}: CPU {x:?} vs GPU {y:?} beyond 1e-12", f.name, i)
                    });
                }
                if x.is_finite() && y.is_finite() {
                    worst = worst.max((x - y).abs());
                }
            }
            if same_bits {
                bitwise += 1;
            }
            if let Some(msg) = first {
                eprintln!("MISMATCH {msg}");
                failures.push(msg);
            } else {
                eprintln!("ok {} ({tier}, bit-identical={same_bits})", f.name);
            }
        }
        assert!(failures.is_empty(), "{} column(s) disagree: {failures:#?}", failures.len());
        eprintln!(
            "CPU feature phase {:.3}s, GPU feature phase {:.3}s",
            cpu.feature_seconds, gpu.feature_seconds
        );
        eprintln!(
            "flow/price-action parity: 23 columns, {bitwise} bit-identical, max abs {worst:.3e}; \
             kernel {:.4}s, transfer {} bytes",
            gpu.gpu_kernel_seconds, gpu.gpu_transfer_bytes
        );
    }

    #[test]
    fn fused_cost_is_measured_on_one_million_rows() {
        if !rcsr::gpu::probe().available() {
            eprintln!("skipping: no CUDA device");
            return;
        }
        // A real 15 m panel has ~109 symbols. Segment-parallel consumers only
        // pay off with that many segments, so the cost model is measured on a
        // 109-symbol 1 M-row fixture rather than the two-symbol parity pair.
        let n = 9_175usize;
        let mut all = Vec::with_capacity(109 * n);
        for s in 0..109u32 {
            let sym: &'static str = Box::leak(format!("S{:03}", s).into_boxed_str());
            all.extend(rows(sym, n, 0, n / 2));
        }
        let mut gpu = context(&all).unwrap();

        let flow: Vec<&str> = fg::FLOW_DEVICE_COLUMNS.iter().map(|c| c.name).collect();
        let pa: Vec<&str> = fg::PRICE_ACTION_DEVICE_COLUMNS.iter().map(|c| c.name).collect();

        // Base-only passes: the cost model splits the fused base from the
        // per-column consumer scan.
        let before_base = gpu.stats();
        let _ = gpu.flow_all(&[]).unwrap();
        let flow_base_stats = gpu.stats().since(&before_base);
        let before_pa_base = gpu.stats();
        let _ = gpu.price_action_all(&[]).unwrap();
        let pa_base_stats = gpu.stats().since(&before_pa_base);

        if std::env::var("RCSR_FEATURE_COST_TRACE").is_ok() {
            for c in &flow {
                let before = gpu.stats();
                let _ = gpu.flow_all(&[*c]).unwrap();
                let after = gpu.stats();
                let d = after.since(&before);
                eprintln!(
                    "  flow {c}: wall {:.3} ms, kernel {:.3} ms",
                    d.wall_s * 1e3,
                    d.kernel_s * 1e3
                );
            }
            for c in &pa {
                let before = gpu.stats();
                let _ = gpu.price_action_all(&[*c]).unwrap();
                let after = gpu.stats();
                let d = after.since(&before);
                eprintln!(
                    "  pa {c}: wall {:.3} ms, kernel {:.3} ms",
                    d.wall_s * 1e3,
                    d.kernel_s * 1e3
                );
            }
        }

        let passes_before = gpu.base_passes();
        let t0 = std::time::Instant::now();
        let _ = gpu.flow_all(&flow).unwrap();
        let flow_s = t0.elapsed().as_secs_f64();
        let flow_stats = gpu.stats();
        let t1 = std::time::Instant::now();
        let _ = gpu.price_action_all(&pa).unwrap();
        let pa_s = t1.elapsed().as_secs_f64();
        let all_stats = gpu.stats();

        let mrow = all.len() as f64 / 1_000_000.0;
        let flow_kernel_ms = flow_stats.kernel_s * 1e3 / mrow;
        let pa_kernel_ms = (all_stats.kernel_s - flow_stats.kernel_s) * 1e3 / mrow;
        let flow_base_ms = flow_base_stats.kernel_s * 1e3 / mrow;
        let pa_base_ms = pa_base_stats.kernel_s * 1e3 / mrow;
        let flow_col_ms = (flow_kernel_ms - flow_base_ms) / 12.0;
        let pa_col_ms = (pa_kernel_ms - pa_base_ms) / 11.0;
        let d2h_mb_s = all_stats.d2h_bytes as f64 / 1e6 / all_stats.d2h_s.max(1e-9);
        eprintln!(
            "measured fused cost per Mrow on {} rows x 109 symbols: flow base {flow_base_ms:.3} ms, \
             column {flow_col_ms:.3} ms, total {flow_kernel_ms:.3} ms (12 columns); \
             price_action base {pa_base_ms:.3} ms, column {pa_col_ms:.3} ms, total \
             {pa_kernel_ms:.3} ms (11 columns); wall {:.3}/{:.3} ms; d2h {d2h_mb_s:.1} MB/s",
            all.len(),
            flow_s * 1e3 / mrow,
            pa_s * 1e3 / mrow
        );
        // The declared constants are kernel-time constants; the wall clock also
        // includes one download per column, which is issue #300's output path.
        assert!(
            flow_kernel_ms <= fg::FLOW_COST.base_ms_per_mrow * 4.0
                + fg::FLOW_COST.column_ms_per_mrow * 12.0 * 4.0,
            "flow kernel measured {flow_kernel_ms:.3} ms/Mrow"
        );
        assert!(
            pa_kernel_ms <= fg::PRICE_ACTION_COST.base_ms_per_mrow * 4.0
                + fg::PRICE_ACTION_COST.column_ms_per_mrow * 11.0 * 4.0,
            "price-action kernel measured {pa_kernel_ms:.3} ms/Mrow"
        );
        assert_eq!(
            gpu.base_passes() - passes_before,
            4,
            "two base kernels per family"
        );
    }
}
