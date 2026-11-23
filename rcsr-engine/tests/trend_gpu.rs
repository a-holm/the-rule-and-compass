//! #932: device trend family parity.
//!
//! The CPU `trend::compute` columns are the reference. The device evaluates the
//! 27 leaves on the daily buckets of the MTF context and gathers them through
//! its causal join. Every leaf except the twelve `_z` leaves is bitwise equal;
//! the `_z` leaves are held within 1e-12 relative. The fixtures are small so the
//! whole file also runs under compute-sanitizer memcheck.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::mtf_gpu;
use rcsr::ohlcv::Bar;
use rcsr::store::{self, FileStore};
use rcsr::trend;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

type Row = (i64, String, f64, f64, f64, f64, f64);

const DAY: i64 = 86_400;

fn device() -> bool {
    if rcsr::gpu::probe().available() {
        true
    } else {
        rcsr::backend::skip_without_device("skipping: no CUDA device");
        false
    }
}

fn walk(symbol: &str, hours: usize, start: i64, step: i64, seed: u64) -> Vec<Row> {
    let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    let mut close = 50.0 + (seed % 29) as f64;
    (0..hours)
        .map(|i| {
            let open = close;
            close = open * (1.0 + 0.012 * (uniform() - 0.495));
            let high = open.max(close) * (1.0 + 0.004 * uniform());
            let low = open.min(close) * (1.0 - 0.004 * uniform());
            (start + i as i64 * step, symbol.to_string(), open, high, low, close, 10.0 + 50.0 * uniform())
        })
        .collect()
}

struct Columns {
    times: Vec<i64>,
    open: Vec<f64>,
    high: Vec<f64>,
    low: Vec<f64>,
    close: Vec<f64>,
    seg_start: Vec<u32>,
    seg_len: Vec<u32>,
    symbols: Vec<(String, usize, usize)>,
}

fn split(rows: &[Row]) -> Columns {
    let mut c = Columns {
        times: Vec::new(),
        open: Vec::new(),
        high: Vec::new(),
        low: Vec::new(),
        close: Vec::new(),
        seg_start: Vec::new(),
        seg_len: Vec::new(),
        symbols: Vec::new(),
    };
    for (i, (t, s, o, h, l, cl, _)) in rows.iter().enumerate() {
        if c.symbols.last().map_or(true, |(name, _, _)| name != s) {
            c.symbols.push((s.clone(), i, 0));
            c.seg_start.push(i as u32);
            c.seg_len.push(0);
        }
        c.symbols.last_mut().unwrap().2 += 1;
        *c.seg_len.last_mut().unwrap() += 1;
        c.times.push(*t);
        c.open.push(*o);
        c.high.push(*h);
        c.low.push(*l);
        c.close.push(*cl);
    }
    c
}

/// Device columns for `names` over every symbol of `rows`.
fn gpu(rows: &[Row], step: i64, names: &[String]) -> std::collections::BTreeMap<String, Vec<f64>> {
    let c = split(rows);
    mtf_gpu::compute(&c.times, &c.open, &c.high, &c.low, &c.close, &c.seg_start, &c.seg_len, step, names)
        .expect("device trend context")
        .0
}

/// CPU reference columns, symbol by symbol, in row order.
fn cpu(rows: &[Row], step: i64, names: &[String]) -> std::collections::BTreeMap<String, Vec<f64>> {
    let c = split(rows);
    let mut out: std::collections::BTreeMap<String, Vec<f64>> = names.iter().map(|n| (n.clone(), Vec::new())).collect();
    for (_, start, len) in &c.symbols {
        let bars: Vec<Bar> = rows[*start..start + len]
            .iter()
            .map(|r| Bar::new(r.0, r.2, r.3, r.4, r.5, r.6))
            .collect();
        let cols = trend::compute(&bars, step, names).unwrap();
        for n in names {
            out.get_mut(n).unwrap().extend_from_slice(&cols[n]);
        }
    }
    out
}

fn close_rel(a: f64, b: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a.is_nan() == b.is_nan() && (a == b || (a - b).abs() <= 1e-12 * a.abs().max(b.abs())))
}

/// Bitwise for every leaf but the `_z` leaves, which stay within 1e-12 relative.
fn assert_parity(tag: &str, want: &std::collections::BTreeMap<String, Vec<f64>>, got: &std::collections::BTreeMap<String, Vec<f64>>) -> usize {
    let mut finite = 0usize;
    for (name, a) in want {
        let b = &got[name];
        assert_eq!(a.len(), b.len(), "{tag}: {name} length");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            if name.ends_with("_z") {
                assert!(close_rel(*x, *y), "{tag}: {name} row {i}: CPU {x:?} GPU {y:?}");
            } else {
                assert_eq!(x.to_bits(), y.to_bits(), "{tag}: {name} row {i} must be bit-identical: CPU {x:?} GPU {y:?}");
            }
            finite += x.is_finite() as usize;
        }
    }
    finite
}

fn write_store(path: &str, rows: &[Row], group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
    for (t, s, o, h, l, c, v) in rows {
        text.push_str(&format!("{t},{s},{o},{h},{l},{c},{v}\n"));
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

fn two_symbols(days: usize) -> Vec<Row> {
    let mut rows = walk("AAA", days * 24, 0, 3600, 1);
    rows.extend(walk("BBB", (days - 20) * 24, 7 * DAY + 5 * 3600, 3600, 2));
    rows
}

#[test]
fn trend_gpu_matches_cpu() {
    if !device() {
        return;
    }
    let rows = two_symbols(400);
    let names = trend::trend_names();
    let (want, got) = (cpu(&rows, 3600, &names), gpu(&rows, 3600, &names));
    let finite = assert_parity("direct", &want, &got);
    assert!(finite > 27 * 2_000, "only {finite} finite cells compared");
    let z: Vec<&String> = names.iter().filter(|n| n.ends_with("_z")).collect();
    let z_cells = z.iter().map(|n| want[*n].iter().filter(|v| v.is_finite()).count()).sum::<usize>();
    let z_bitwise = z
        .iter()
        .map(|n| want[*n].iter().zip(&got[*n]).filter(|(a, b)| a.is_finite() && a.to_bits() == b.to_bits()).count())
        .sum::<usize>();
    eprintln!("trend _z leaves: {z_bitwise} of {z_cells} finite cells bit-identical");

    // `mtf,trend` shares the daily resample: both families match the CPU.
    let mut both = enrich::mtf_names();
    both.extend(names.clone());
    let dev = gpu(&rows, 3600, &both);
    assert_parity("with mtf", &cpu(&rows, 3600, &names), &dev);

    // The enrich path under --gpu on dispatches all 27 columns to the device
    // and writes the CPU bytes for the bitwise leaves.
    enrich::set_gpu_floor_for_tests(1);
    let dir = Scratch::new("trend-gpu-parity");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 4096);
    let c = enrich::build(&opts(&src, &cpu_out, "trend", GpuMode::Off, 65_536)).unwrap();
    let g = enrich::build(&opts(&src, &gpu_out, "trend", GpuMode::On, 65_536)).unwrap();
    assert_eq!(c.features.len(), 27);
    assert_eq!(g.gpu_columns, 27, "all 27 trend columns must dispatch");
    assert_eq!(g.cpu_columns, 0);
    assert_eq!(g.gpu_fallback, 0);
    let plan = g.plans.iter().find(|p| p.group == "trend").expect("trend plan");
    assert_eq!(plan.backend, "gpu");
    assert_eq!(plan.op, "trend-join");
    for (a, b) in c.features.iter().zip(&g.features) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.first_finite_row, b.first_finite_row, "{}", a.name);
        if !a.name.ends_with("_z") {
            assert_eq!(a.hash, b.hash, "{} bytes differ", a.name);
        }
    }
    for name in &names {
        let (a, b) = (column(&cpu_out, name), column(&gpu_out, name));
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(close_rel(*x, *y), "{name} row {i}: CPU {x:?} GPU {y:?}");
        }
    }
}

#[test]
fn trend_gpu_missing_bucket() {
    if !device() {
        return;
    }
    // AAA misses day 40 entirely; BBB lacks one hour of day 60 and day 0 starts
    // mid-day.
    let mut rows: Vec<Row> = walk("AAA", 420 * 24, 0, 3600, 3)
        .into_iter()
        .filter(|r| r.0.div_euclid(DAY) != 40)
        .collect();
    rows.extend(walk("BBB", 410 * 24, 13 * 3600, 3600, 4).into_iter().filter(|r| r.0 != 60 * DAY + 9 * 3600));
    let names = trend::trend_names();
    let want = cpu(&rows, 3600, &names);
    assert_parity("missing", &want, &gpu(&rows, 3600, &names));
    // The CPU reference really is NaN after the hole: day 41 reads day 40.
    let i = rows.iter().position(|r| r.0 == 41 * DAY + 6 * 3600).unwrap();
    for name in &names {
        assert!(want[name][i].is_nan(), "{name} must be NaN after the missing day");
    }
}

#[test]
fn trend_gpu_partial_trailing_bucket() {
    if !device() {
        return;
    }
    // Both series end mid-day: the running day is never visible and its rows
    // read the day before.
    let mut rows = walk("AAA", 400 * 24 + 11, 0, 3600, 5);
    rows.extend(walk("BBB", 380 * 24 + 3, 0, 3600, 6));
    let names = trend::trend_names();
    let want = cpu(&rows, 3600, &names);
    assert_parity("trailing", &want, &gpu(&rows, 3600, &names));
    let last = 400 * 24 + 10;
    assert_eq!(rows[last].0, 400 * DAY + 10 * 3600);
    assert_eq!(want["trend_mom4w"][last].to_bits(), want["trend_mom4w"][400 * 24].to_bits(), "the running day feeds no value");
    // A 15 m base runs the same daily contract.
    let quarter = walk("CCC", 380 * 96 + 37, 0, 900, 7);
    assert_parity("15m", &cpu(&quarter, 900, &names), &gpu(&quarter, 900, &names));
}

#[test]
fn trend_gpu_prefix_replay() {
    if !device() {
        return;
    }
    let rows = two_symbols(390);
    let names = trend::trend_names();
    let full = gpu(&rows, 3600, &names);
    // Cuts land mid-day, so a prefix ends inside a running day.
    let mut cuts = 0;
    for cut in (1..=20).map(|k| k * (rows.len() - 24) / 20 + 7 + k % 13) {
        let got = gpu(&rows[..cut], 3600, &names);
        for name in &names {
            for i in 0..cut {
                assert_eq!(full[name][i].to_bits(), got[name][i].to_bits(), "{name} changed at row {i} under prefix cut {cut}");
            }
        }
        cuts += 1;
    }
    assert_eq!(cuts, 20);
}

#[test]
fn trend_gpu_chunk_invariance() {
    if !device() {
        return;
    }
    let names = trend::trend_names();
    let a = walk("AAA", 390 * 24, 0, 3600, 8);
    let b = walk("BBB", 385 * 24, 0, 3600, 9);
    let mut both = a.clone();
    both.extend(b.clone());
    let combined = gpu(&both, 3600, &names);
    let left = gpu(&a, 3600, &names);
    let right = gpu(&b, 3600, &names);
    for name in &names {
        let mut joined = left[name].clone();
        joined.extend_from_slice(&right[name]);
        for (i, (x, y)) in combined[name].iter().zip(&joined).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{name} symbol batch differs at row {i}");
        }
    }
    enrich::set_gpu_floor_for_tests(1);
    let dir = Scratch::new("trend-gpu-chunks");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out_a = dir.join("a.rcsb").to_string_lossy().into_owned();
    let out_b = dir.join("b.rcsb").to_string_lossy().into_owned();
    write_store(&src, &both, 4096);
    let ra = enrich::build(&opts(&src, &out_a, "trend", GpuMode::On, 7)).unwrap();
    let rb = enrich::build(&opts(&src, &out_b, "trend", GpuMode::On, 65_536)).unwrap();
    assert_eq!(ra.gpu_columns, 27);
    assert_eq!(rb.gpu_columns, 27);
    let ha: Vec<_> = ra.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    let hb: Vec<_> = rb.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect();
    assert_eq!(ha, hb, "chunk size changed the trend bytes");
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn trend_causal_under_future_edits_gpu() {
    if !device() {
        return;
    }
    let mut rows = walk("AAA", 400 * 24, 0, 3600, 31);
    rows.extend(walk("BBB", 400 * 24, 0, 3600, 32));
    let names = trend::trend_names();
    let base = gpu(&rows, 3600, &names);
    let hours = 400 * 24;
    let mut rng = Rng(0x2545F4914F6CDD1D);
    let mut moved = 0usize;
    for k in 0..20 {
        let cuts = [rows[rng.below(hours)].0, rows[hours + rng.below(hours)].0];
        let mut edited = rows.clone();
        for (sym, cut) in ["AAA", "BBB"].iter().zip(cuts) {
            for r in edited.iter_mut().filter(|r| r.1 == *sym && r.0 > cut) {
                let (o, c) = (20.0 + 300.0 * rng.unit(), 20.0 + 300.0 * rng.unit());
                r.2 = o;
                r.5 = c;
                r.3 = o.max(c) * (1.0 + 0.05 * rng.unit());
                r.4 = o.min(c) * (1.0 - 0.05 * rng.unit());
                r.6 = 1.0 + 1000.0 * rng.unit();
            }
        }
        let got = gpu(&edited, 3600, &names);
        for name in &names {
            for (i, r) in rows.iter().enumerate() {
                let cut = if r.1 == "AAA" { cuts[0] } else { cuts[1] };
                if r.0 <= cut {
                    assert_eq!(base[name][i].to_bits(), got[name][i].to_bits(), "cut {k}: {name} of {} changed at row {i} (cut {cut})", r.1);
                } else if base[name][i].to_bits() != got[name][i].to_bits() {
                    moved += 1;
                }
            }
        }
    }
    assert!(moved > 0, "the future edits must change some value after the cuts");
}

#[test]
fn trend_gpu_refused_arena_falls_back_to_cpu() {
    if !device() {
        return;
    }
    // Count the guarded allocations of a trend-only run, then refuse the last
    // one: the trend arena. The run must label the fallback and write the CPU
    // bytes.
    enrich::set_gpu_floor_for_tests(1);
    let rows = two_symbols(80);
    let dir = Scratch::new("trend-gpu-alloc");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let cpu_out = dir.join("cpu.rcsb").to_string_lossy().into_owned();
    let gpu_out = dir.join("gpu.rcsb").to_string_lossy().into_owned();
    write_store(&src, &rows, 4096);
    let c = enrich::build(&opts(&src, &cpu_out, "trend", GpuMode::Off, 65_536)).unwrap();
    let names = trend::trend_names();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(u64::MAX);
    let _ = gpu(&rows, 3600, &names);
    let total = rcsr::enrich_gpu::alloc_count_for_tests();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    assert!(total > 6, "only {total} guarded allocations");
    let c2 = split(&rows);
    rcsr::enrich_gpu::set_alloc_fault_for_tests(total);
    let refused = mtf_gpu::compute(&c2.times, &c2.open, &c2.high, &c2.low, &c2.close, &c2.seg_start, &c2.seg_len, 3600, &names);
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    let err = refused.err().expect("a refused trend arena fails the device pass");
    assert_eq!((err.stage, err.code), ("trend", 7));

    rcsr::enrich_gpu::set_alloc_fault_for_tests(u64::MAX);
    let g0 = enrich::build(&opts(&src, &gpu_out, "trend", GpuMode::On, 65_536)).unwrap();
    let run_total = rcsr::enrich_gpu::alloc_count_for_tests();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    assert_eq!(g0.gpu_columns, 27);
    std::fs::remove_file(&gpu_out).ok();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(run_total);
    let g = enrich::build(&opts(&src, &gpu_out, "trend", GpuMode::On, 65_536)).unwrap();
    rcsr::enrich_gpu::set_alloc_fault_for_tests(0);
    assert_eq!(g.gpu_fallback, 27, "every trend column falls back");
    assert_eq!(g.gpu_columns, 0);
    assert!(g.gpu_fallback_columns.iter().all(|(n, r)| n.starts_with("trend_") && r.starts_with("trend: stage trend code 7")), "{:?}", g.gpu_fallback_columns);
    for (a, b) in c.features.iter().zip(&g.features) {
        assert_eq!(a.hash, b.hash, "{} fallback bytes differ from the CPU", a.name);
    }
}
