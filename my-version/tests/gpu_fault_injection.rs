//! #384: injected device failures during `enrich --gpu on`.
//!
//! Every failure is injected before the driver is called: a refused
//! allocation returns before cudaMalloc, a launch failure returns before the
//! kernel is queued. No test exhausts device memory. Run under
//! `compute-sanitizer --tool memcheck` first.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options, Report};
use rcsr::enrich_gpu;
use rcsr::store::{self, FileStore};

#[path = "common/sanitizer.rs"]
#[allow(dead_code)]
mod sanitizer;
#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const SYMBOLS: usize = 6;
const PER_SYMBOL: usize = 1_200;

fn write_store(path: &str) {
    let mut text =
        String::from("ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote,member_asof\n");
    for k in 0..SYMBOLS {
        let mut close = 100.0 + k as f64;
        for i in 0..PER_SYMBOL {
            let t = 1_700_000_000i64 + i as i64 * 900;
            let ret = if i % 5 == 0 { 0.002 } else { 0.0005 * ((k % 3) as f64 - 1.0) };
            let next = close * (1.0 + ret) + (i as f64 * 0.07).sin() * 0.05;
            let member = if k == 3 && i % 11 == 0 { t + 900 } else { t };
            let high = next.max(close) + 0.1;
            let low = next.min(close) - 0.1;
            let volume = 10.0 + (i % 97) as f64;
            let name = match k {
                0 => "BTCUSDT".to_string(),
                1 => "ETHUSDT".to_string(),
                _ => format!("SYM{k:02}"),
            };
            text.push_str(&format!(
                "{t},{name},{close},{high},{low},{next},{volume},{},{},{},{},{member}\n",
                next * volume,
                5.0 + (i % 53) as f64,
                volume * 0.4,
                volume * 0.4 * next
            ));
            close = next;
        }
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, 2_048, Some(0)).unwrap();
}

fn columns(path: &str, names: &[String]) -> Vec<Vec<f64>> {
    let s = FileStore::open(path).unwrap();
    names
        .iter()
        .map(|name| {
            let c = s.header().column_index(name).unwrap();
            let mut out = Vec::new();
            for g in 0..s.n_groups() {
                out.extend(s.read_f64_group(g, c).unwrap());
            }
            out
        })
        .collect()
}

fn close_enough(a: f64, b: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return a.is_nan() && b.is_nan();
    }
    a == b || (a - b).abs() <= 1e-9 * a.abs().max(b.abs()).max(1.0)
}

struct Reset;

impl Drop for Reset {
    fn drop(&mut self) {
        enrich_gpu::set_alloc_fault_for_tests(0);
        enrich_gpu::set_launch_fault_for_tests(0);
    }
}

static HOOKS: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Fixture {
    _lock: std::sync::MutexGuard<'static, ()>,
    scratch: Scratch,
    src: String,
    features: String,
    names: Vec<String>,
    cpu_columns: Vec<Vec<f64>>,
}

impl Fixture {
    fn options(&self, out: &str, gpu: GpuMode) -> Options {
        Options {
            data: self.src.clone(),
            out: Some(out.to_string()),
            features: vec![self.features.clone()],
            labels: vec!["fwd:1h,cost=10bp,volnorm=20".to_string()],
            allow_lockbox: true,
            gpu,
            ..Default::default()
        }
    }

    fn gpu_run(&self, tag: &str) -> (Report, String) {
        let out = self.scratch.join(&format!("{tag}.rcsb")).to_string_lossy().into_owned();
        let report = enrich::build(&self.options(&out, GpuMode::On))
            .unwrap_or_else(|e| panic!("{tag}: build failed: {e}"));
        (report, out)
    }

    /// Output equals the CPU arm: bit-equal on columns that fell back to the
    /// CPU, within the parity tolerance on device columns.
    fn assert_matches_cpu(&self, tag: &str, report: &Report, out: &str) {
        let got = columns(out, &self.names);
        for ((name, a), b) in self.names.iter().zip(&self.cpu_columns).zip(&got) {
            let fell_back = report.gpu_fallback_columns.iter().any(|(n, _)| n == name);
            assert_eq!(a.len(), b.len(), "{tag}: column {name} length");
            for (i, (x, y)) in a.iter().zip(b).enumerate() {
                let same = if fell_back { x.to_bits() == y.to_bits() } else { close_enough(*x, *y) };
                assert!(same, "{tag}: column {name} row {i}: CPU {x:?} run {y:?} (fallback {fell_back})");
            }
        }
    }

    fn assert_labelled(&self, tag: &str, report: &Report) {
        assert_eq!(
            report.gpu_fallback,
            report.gpu_fallback_columns.len(),
            "{tag}: gpu_fallback count"
        );
        for (name, reason) in &report.gpu_fallback_columns {
            assert!(!name.is_empty() && !reason.trim().is_empty(), "{tag}: unlabelled fallback");
        }
    }
}

fn fixture(tag: &str) -> Option<Fixture> {
    if !rcsr::gpu::probe().available() {
        rcsr::backend::skip_without_device("skipping: no CUDA device");
        return None;
    }
    let lock = HOOKS.lock().unwrap_or_else(|e| e.into_inner());
    enrich::set_gpu_floor_for_tests(0);
    let names: Vec<String> = enrich::default_registry()
        .into_iter()
        .filter(|e| e.available && enrich::gpu_supports(&e.name))
        .map(|e| e.name)
        .collect();
    let scratch = Scratch::new(tag);
    let src = scratch.join("in.rcsb").to_string_lossy().into_owned();
    write_store(&src);
    let cpu_out = scratch.join("cpu.rcsb").to_string_lossy().into_owned();
    let fx = Fixture { _lock: lock, scratch, src, features: names.join(","), names: Vec::new(), cpu_columns: Vec::new() };
    let cpu = enrich::build(&fx.options(&cpu_out, GpuMode::Off)).unwrap();
    let names: Vec<String> = cpu.features.iter().map(|f| f.name.clone()).collect();
    let cpu_columns = columns(&cpu_out, &names);
    Some(Fixture { names, cpu_columns, ..fx })
}

/// Every `k` natively. Under memcheck a fixed subset derived from `total`: the
/// first four, every eighth of the range, and the last.
fn sweep(total: u64) -> Vec<u64> {
    if !sanitizer::active() {
        return (1..=total).collect();
    }
    let step = (total / 8).max(1);
    let mut ks: Vec<u64> = (1..=4.min(total)).chain((5..=total).step_by(step as usize)).collect();
    ks.push(total);
    ks.sort_unstable();
    ks.dedup();
    ks
}

/// Refuse the k-th device allocation for every k the build makes. The run must
/// finish, label every CPU fallback and match the CPU arm.
#[test]
fn every_refused_device_allocation_falls_back_cleanly() {
    let Some(fx) = fixture("alloc") else { return };
    let _reset = Reset;
    enrich_gpu::set_alloc_fault_for_tests(u64::MAX);
    let (base, base_out) = fx.gpu_run("count");
    let total = enrich_gpu::alloc_count_for_tests();
    enrich_gpu::set_alloc_fault_for_tests(0);
    assert!(total > 20, "the build made only {total} guarded allocations");
    fx.assert_matches_cpu("baseline", &base, &base_out);
    eprintln!("columns compared per k: {}", fx.names.len());
    let ks = sweep(total);
    eprintln!("allocations swept: {ks:?} of {total}");
    let mut with_fallback = 0usize;
    for &k in &ks {
        let tag = format!("alloc-{k}");
        enrich_gpu::set_alloc_fault_for_tests(k);
        let (report, out) = fx.gpu_run(&tag);
        enrich_gpu::set_alloc_fault_for_tests(0);
        fx.assert_labelled(&tag, &report);
        fx.assert_matches_cpu(&tag, &report, &out);
        let extra = report.gpu_fallback.saturating_sub(base.gpu_fallback);
        eprintln!("alloc {k}/{total}: fallback columns +{extra}");
        if extra > 0 {
            with_fallback += 1;
        }
        std::fs::remove_file(&out).ok();
    }
    eprintln!("allocations swept: {} of {total}, with CPU fallback: {with_fallback}", ks.len());
    assert!(with_fallback > 0, "no refused allocation produced a CPU fallback");
}

/// Make each launch site fail before the kernel is queued.
#[test]
fn every_failed_kernel_launch_falls_back_cleanly() {
    let Some(fx) = fixture("launch") else { return };
    let _reset = Reset;
    for (site_name, site) in enrich_gpu::LAUNCH_FAULT_SITES {
        let tag = format!("launch-{site}");
        enrich_gpu::set_launch_fault_for_tests(site);
        let (report, out) = fx.gpu_run(&tag);
        let hits = enrich_gpu::launch_fault_hits_for_tests();
        enrich_gpu::set_launch_fault_for_tests(0);
        eprintln!(
            "launch fault {site_name}: hits {hits}, fallback columns {}",
            report.gpu_fallback
        );
        assert!(hits > 0, "{site_name}: the injected failure was never reached");
        fx.assert_labelled(&tag, &report);
        fx.assert_matches_cpu(&tag, &report, &out);
        if matches!(site, 4..=6) {
            assert!(report.gpu_fallback > 0, "{site_name}: no labelled CPU fallback");
        }
        std::fs::remove_file(&out).ok();
    }
}
