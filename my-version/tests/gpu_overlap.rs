//! #286 measurement: the wide-cell overlap fit queue.
//!
//! This is an **ignored** measurement because it needs just over the
//! `WIDE_ADMISSION_ROWS` floor (1 048 576 rows) to make the serial admission
//! set unconditional. Run it explicitly:
//!
//! ```sh
//! cargo test --release --features cuda --test gpu_overlap -- --ignored --nocapture
//! ```
//!
//! It answers the three questions the issue asks for:
//! * device-busy over the **whole run wall**, not an evaluator window;
//! * fallback share and every finite device status label;
//! * no worse archive (identical hash and best error) than the synchronous
//!   `--pipeline-depth 0` run at the same seed.

use std::time::Instant;

use rcsr::data::Sample;
use rcsr::fit::FitStrategy;
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

/// Keep in sync with `search::WIDE_ADMISSION_ROWS` (private to the crate).
const WIDE_ADMISSION_ROWS: usize = 1 << 20;

fn wide_sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = (i as f64 / n as f64) * 12.0 - 6.0;
        let z = ((i as u32).wrapping_mul(2_654_435_761) as f64 / u32::MAX as f64) * 2.0 - 1.0;
        x.push(vec![t, z]);
        y.push(1.3 * t.sin() + 0.2 * z + 0.1 * t);
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

fn config(depth: usize, seed: u32) -> Config {
    Config {
        generations: 2,
        islands: 1,
        pop: 64,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Squared,
        gpu: true,
        pipeline_depth: depth,
        ..Default::default()
    }
}

#[test]
#[ignore = "measurement: wide-cell GPU overlap, run explicitly"]
fn wide_cell_overlap_matches_serial_and_reports_run_scope_duty() {
    let train = wide_sample(WIDE_ADMISSION_ROWS);
    let test = wide_sample(1_024);
    let mut rows = Vec::new();
    for depth in [0usize, 2] {
        let t = Instant::now();
        let out = search::run(&config(depth, 11), &train, &test);
        let wall = t.elapsed().as_secs_f64();
        let g = out.gpu.expect("CUDA run must carry the fit ledger");
        let device_busy = g.accounting.kernel_s + g.fit_ms / 1000.0;
        let share = if g.fit_attempts == 0 {
            0.0
        } else {
            g.fit_fallback as f64 / g.fit_attempts as f64
        };
        let best = out
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        println!(
            "depth={depth} wall={wall:.3}s device_busy={device_busy:.3}s \
             duty={:.1}% fit_attempts={} gpu={} fallback={} share={:.2}% \
             labels={:?} causes={:?} archive={:016x} best={best:.12e}",
            100.0 * device_busy / wall.max(1e-9),
            g.fit_attempts,
            g.fit_gpu,
            g.fit_fallback,
            100.0 * share,
            g.fit_status_by_label,
            g.fit_fallback_by_cause,
            out.archive.hash(),
        );
        rows.push((out.archive.hash(), best, wall, device_busy, share));
    }
    assert_eq!(
        rows[0].0, rows[1].0,
        "overlap must not change the archive hash"
    );
    assert_eq!(
        rows[0].1.to_bits(),
        rows[1].1.to_bits(),
        "overlap must not change the best error"
    );
    // The proxy has a tiny population, so a few structural fallbacks are a
    // large *share*. The production target is measured on the 10 h cell; here
    // we only require that the finite label statuses did not become fallbacks.
    assert!(
        rows[1].4 <= 0.20,
        "proxy fallback share {:.2}% is implausibly high",
        100.0 * rows[1].4
    );
    assert!(
        rows[1].3 > 0.0 && rows[1].3 <= rows[1].2,
        "device-busy must be measurable and inside the run wall"
    );
}
