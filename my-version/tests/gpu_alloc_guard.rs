//! #293: no production device allocation scales with P x N.
//!
//! This is its own test binary: the counting host allocator and the device
//! ledger must not be polluted by unrelated tests running in parallel.

#![cfg(feature = "cuda")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

struct Counting;

static MAX_SINGLE: AtomicUsize = AtomicUsize::new(0);

/// `MAX_SINGLE` is process-global: observed runs take turns.
static RUN: Mutex<()> = Mutex::new(());

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        MAX_SINGLE.fetch_max(layout.size(), Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        MAX_SINGLE.fetch_max(new_size, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[global_allocator]
static ALLOC: Counting = Counting;

const ROWS: usize = 131_072;
const PLAN_ROWS: usize = 1_000_000;
const CANDIDATES: usize = 400;

/// Tags of the requests that may scale with the row count: the dataset, its
/// f64 twin, the side vectors, the rank target and the fit/score region.
const N_SCALED_TAGS: &[&str] = &[
    "cols",
    "y",
    "cols_f64",
    "y_f64",
    "w_f64",
    "tsign",
    "tsign_f64",
    "rank_b",
    "fit_region",
];

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) >> 8) as f64 / (1u32 << 24) as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn config(metric: Metric) -> Config {
    Config {
        generations: 3,
        islands: 4,
        pop: 100,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 3,
        fit_strategy: FitStrategy::Gpu,
        metric,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        ..Default::default()
    }
}

struct Observed {
    max_host: usize,
    max_device: u64,
    largest_term: u64,
    max_tag: String,
    refused: u64,
    fit_gpu: usize,
}

fn run(metric: Metric, rows: usize) -> Option<Observed> {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let _one_at_a_time = RUN.lock().unwrap_or_else(|e| e.into_inner());
    let train = sample(rows);
    let test = sample(1_000);
    let plan = search::device_plan_for(&config(metric), &train).expect("a plan");
    let t = &plan.terms;
    let largest_term = [t.data_f32, t.side_vectors, t.fit_twin, t.buffers, t.stat_wave, t.tile_arena, t.tile_pinned, t.partials]
        .into_iter()
        .max()
        .unwrap_or(0);
    MAX_SINGLE.store(0, Ordering::Relaxed);
    let out = search::run(&config(metric), &train, &test);
    let max_host = MAX_SINGLE.load(Ordering::Relaxed);
    assert_eq!(out.backend, rcsr::backend::Kind::Gpu, "{:?}", out.backend_reason);
    let gpu = out.gpu.expect("gpu run stats");
    let obs = gpu.observed.expect("the device allocation ledger");
    eprintln!(
        "{metric:?}: host max {max_host} B, device max {} B ({}), requests {}, peak live {} B, device fits {}",
        obs.max_request_bytes, obs.max_request_tag, obs.requests, obs.peak_live_bytes, gpu.fit_gpu
    );
    Some(Observed {
        max_host,
        max_device: obs.max_request_bytes,
        largest_term,
        max_tag: obs.max_request_tag,
        refused: obs.refused,
        fit_gpu: gpu.fit_gpu,
    })
}

const PLAN_P_N_4: u64 = (CANDIDATES * PLAN_ROWS * 4) as u64;

fn check(o: &Observed, metric: Metric, rows: usize) {
    let p_n_4 = (CANDIDATES * rows * 4) as u64;
    assert!(o.fit_gpu > 0, "the device fit did not run");
    assert!(
        (o.max_host as u64) < p_n_4 / 8,
        "largest host allocation {} B reaches P*N*4/8",
        o.max_host
    );
    if metric == Metric::RankCorrelation {
        // Score waves are capped at 512 MiB (F1).
        let bound = 512 * 1024 * 1024;
        assert!(o.max_device <= bound, "largest device request {} B above {}", o.max_device, bound);
        if p_n_4 > bound as u64 {
            assert!(o.max_device < p_n_4);
        }
    } else {
        assert!(o.max_device < p_n_4 / 2, "largest device request {} B", o.max_device);
    }
    assert!(
        N_SCALED_TAGS.contains(&o.max_tag.as_str()),
        "largest device request is tagged {}",
        o.max_tag
    );
    assert!(
        o.max_device <= o.largest_term,
        "largest device request {} B above the largest planned term {} B",
        o.max_device,
        o.largest_term
    );
    assert_eq!(o.refused, 0);
}

#[test]
fn the_plan_at_one_million_rows_has_no_candidates_times_rows_term() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let _one_at_a_time = RUN.lock().unwrap_or_else(|e| e.into_inner());
    for metric in [Metric::Squared, Metric::RankCorrelation] {
        let plan = search::device_plan_for(&config(metric), &sample(PLAN_ROWS)).expect("a plan");
        let t = &plan.terms;
        for (name, v) in [
            ("data_f32", t.data_f32),
            ("side_vectors", t.side_vectors),
            ("fit_twin", t.fit_twin),
            ("buffers", t.buffers),
            ("stat_wave", t.stat_wave),
            ("tile_arena", t.tile_arena),
            ("tile_pinned", t.tile_pinned),
            ("partials", t.partials),
        ] {
            assert!(v < PLAN_P_N_4 / 2, "{metric:?}: term {name} = {v} B");
        }
        if metric == Metric::RankCorrelation {
            assert!(t.stat_wave <= 512 << 20, "{plan:?}");
            assert!(t.stat_wave > 0, "{plan:?}");
        }
    }
}

#[test]
fn squared() {
    let Some(o) = run(Metric::Squared, ROWS) else { return };
    check(&o, Metric::Squared, ROWS);
}

#[test]
fn rank() {
    let Some(o) = run(Metric::RankCorrelation, ROWS) else { return };
    check(&o, Metric::RankCorrelation, ROWS);
}

#[test]
#[ignore = "1M-row allocation measurement"]
fn squared_at_one_million_rows() {
    let Some(o) = run(Metric::Squared, PLAN_ROWS) else { return };
    check(&o, Metric::Squared, PLAN_ROWS);
}

#[test]
#[ignore = "1M-row allocation measurement"]
fn rank_at_one_million_rows() {
    let Some(o) = run(Metric::RankCorrelation, PLAN_ROWS) else { return };
    check(&o, Metric::RankCorrelation, PLAN_ROWS);
}
