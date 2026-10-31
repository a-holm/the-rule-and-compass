//! #293: no production device allocation scales with P x N.
//!
//! This is its own test binary: the counting host allocator and the device
//! ledger must not be polluted by unrelated tests running in parallel.

#![cfg(feature = "cuda")]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

struct Counting;

static MAX_SINGLE: AtomicUsize = AtomicUsize::new(0);

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

const ROWS: usize = 1_000_000;
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
        gpu_notes: false,
        ..Default::default()
    }
}

struct Observed {
    max_host: usize,
    max_device: u64,
    max_tag: String,
    refused: u64,
    fit_gpu: usize,
}

fn run(metric: Metric) -> Option<Observed> {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return None;
    }
    let train = sample(ROWS);
    let test = sample(1_000);
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
        max_tag: obs.max_request_tag,
        refused: obs.refused,
        fit_gpu: gpu.fit_gpu,
    })
}

const P_N_4: u64 = (CANDIDATES * ROWS * 4) as u64;

#[test]
fn squared() {
    let Some(o) = run(Metric::Squared) else { return };
    assert!(o.fit_gpu > 0, "the device fit did not run");
    assert!(
        (o.max_host as u64) < P_N_4 / 8,
        "largest host allocation {} B reaches P*N*4/8",
        o.max_host
    );
    assert!(o.max_device < P_N_4 / 2, "largest device request {} B", o.max_device);
    assert!(
        N_SCALED_TAGS.contains(&o.max_tag.as_str()),
        "largest device request is tagged {}",
        o.max_tag
    );
    assert_eq!(o.refused, 0);
}

#[test]
fn rank() {
    let Some(o) = run(Metric::RankCorrelation) else { return };
    assert!(o.fit_gpu > 0, "the device fit did not run");
    assert!(
        (o.max_host as u64) < P_N_4 / 8,
        "largest host allocation {} B reaches P*N*4/8",
        o.max_host
    );
    // Score waves are capped at 512 MiB (F1).
    let bound = 512 * 1024 * 1024;
    assert!(o.max_device <= bound, "largest device request {} B above {}", o.max_device, bound);
    assert!(o.max_device < P_N_4);
    assert!(
        N_SCALED_TAGS.contains(&o.max_tag.as_str()),
        "largest device request is tagged {}",
        o.max_tag
    );
    assert_eq!(o.refused, 0);
}
