//! #290: the device calibration probe for the dispatch cost model.
//!
//! One short run on a small resident dataset times the terms
//! [`crate::dispatch::Calibration`] prices: the scoring rate, the batched fit
//! rate, host-to-device and device-to-host bandwidth, and the launch overhead.
//! Each term is the median of its timed reps and carries the spread of those
//! reps. The probe runs once per process; a process without a usable device
//! gets `None` and the model keeps its corpus anchors.

use crate::dispatch::{
    median_and_spread, DeviceRates, Spread, FIT_REF_CONSTANTS, FIT_REF_ROWS, FIT_REF_SWEEPS,
};
use crate::fit::DeviceLoss;
use crate::gpu::{GpuEvaluator, Population};
use std::sync::OnceLock;
use std::time::Instant;

const ROWS: usize = 16_384;
const EVAL_CANDIDATES: usize = 1024;
const FIT_CANDIDATES: usize = 64;
const FIT_SWEEPS: usize = 8;
const COPY_BYTES: usize = 16 << 20;
const REPS: usize = 15;
const FIT_REPS: usize = 5;
const FIT_WARMUP: usize = 2;
const WARMUP: usize = 4;
const EVAL_WARMUP: std::time::Duration = std::time::Duration::from_millis(40);
const LAUNCH_REPS: usize = 33;
const DEVICE_BUDGET_BYTES: u64 = 256 << 20;

/// The device rates of this process, measured on first use.
pub fn device_rates() -> Option<DeviceRates> {
    static RATES: OnceLock<Option<DeviceRates>> = OnceLock::new();
    *RATES.get_or_init(measure)
}

/// Run the probe now, whether or not an earlier call cached a result.
pub fn measure() -> Option<DeviceRates> {
    if !crate::gpu::probe().available() {
        return None;
    }
    let start = Instant::now();
    let names = vec!["x0".to_string()];
    let x: Vec<f64> = (0..ROWS).map(|i| i as f64 / ROWS as f64 * 2.0 - 1.0).collect();
    let y: Vec<f64> = x.iter().map(|&v| 0.8 * v * v + 0.25 * v + 0.1).collect();
    let x32: Vec<f32> = x.iter().map(|&v| v as f32).collect();
    let y32: Vec<f32> = y.iter().map(|&v| v as f32).collect();
    let mut ev = GpuEvaluator::with_budget(&x32, ROWS, 1, &y32, Some(DEVICE_BUDGET_BYTES))?;

    let score_tree = crate::parse::parse("x0*x0+0.5*x0-1.0", &names).ok()?;
    let nodes = crate::expr::complexity(&score_tree) as f64;
    let pop = Population::from_trees(std::iter::repeat(&score_tree).take(EVAL_CANDIDATES));
    let (prog, offsets) = pop.programs();
    let warm = Instant::now();
    while warm.elapsed() < EVAL_WARMUP {
        let (_, ms) = ev.eval_batch(prog, offsets);
        if ms < 0.0 {
            return None;
        }
    }
    let mut eval_rates = Vec::with_capacity(REPS);
    for _ in 0..REPS {
        let (_, ms) = ev.eval_batch(prog, offsets);
        if ms < 0.0 {
            return None;
        }
        eval_rates.push(EVAL_CANDIDATES as f64 * ROWS as f64 * nodes / (ms as f64 / 1e3));
    }

    let one = Population::from_trees(std::iter::once(&score_tree));
    let (prog1, offsets1) = one.programs();
    let mut launch = Vec::with_capacity(LAUNCH_REPS);
    for rep in 0..WARMUP + LAUNCH_REPS {
        let before = ev.stats().launch_s;
        let (_, ms) = ev.eval_batch(prog1, offsets1);
        if ms < 0.0 {
            return None;
        }
        if rep >= WARMUP {
            launch.push(ev.stats().launch_s - before);
        }
    }

    if !ev.set_fit_data(&x, &y, &[]) {
        return None;
    }
    let fit_tree = crate::parse::parse("0.7*x0*x0+0.3*x0+0.2", &names).ok()?;
    let plan = crate::fit::gpu_fit_plan(&fit_tree)?;
    let constants = plan.nconst as f64;
    let plans: Vec<&crate::fit::GpuFitPlan> = (0..FIT_CANDIDATES).map(|_| &plan).collect();
    let mut fit_rates = Vec::with_capacity(FIT_REPS);
    for rep in 0..FIT_WARMUP + FIT_REPS {
        let (_, ms) = ev.fit_batch(&plans, FIT_SWEEPS, DeviceLoss::Squared);
        if ms < 0.0 {
            return None;
        }
        if rep >= FIT_WARMUP {
            let per_s = FIT_CANDIDATES as f64 / (ms as f64 / 1e3);
            fit_rates.push(
                per_s * (ROWS as f64 / FIT_REF_ROWS) * (constants / FIT_REF_CONSTANTS)
                    * (FIT_SWEEPS as f64 / FIT_REF_SWEEPS),
            );
        }
    }

    let (h2d_ms, d2h_ms) = ev.copy_probe(COPY_BYTES, WARMUP + REPS)?;
    let bytes = COPY_BYTES as f64;
    let h2d: Vec<f64> = h2d_ms[WARMUP..].iter().map(|ms| bytes / (ms / 1e3)).collect();
    let d2h: Vec<f64> = d2h_ms[WARMUP..].iter().map(|ms| bytes / (ms / 1e3)).collect();
    let both: Vec<f64> = h2d_ms[WARMUP..]
        .iter()
        .zip(&d2h_ms[WARMUP..])
        .map(|(h, d)| 2.0 * bytes / ((h + d) / 1e3))
        .collect();

    let (r_eval, s_eval) = median_and_spread(&eval_rates)?;
    let (r_fit, s_fit) = median_and_spread(&fit_rates)?;
    let (bw, s_bw) = median_and_spread(&both)?;
    let (bw_h2d, _) = median_and_spread(&h2d)?;
    let (bw_d2h, _) = median_and_spread(&d2h)?;
    let (launch_s, s_launch) = median_and_spread(&launch)?;
    Some(DeviceRates {
        r_eval,
        r_fit,
        bw,
        bw_h2d,
        bw_d2h,
        launch_s,
        spread: Spread {
            r_eval: s_eval,
            r_fit: s_fit,
            bw: s_bw,
            launch_s: s_launch,
            cpu_eval: 1.0,
        },
        samples: REPS as u32,
        probe_s: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dispatch::Calibration;

    #[test]
    fn device_probe_three_runs_after_the_cold_start_agree_within_the_larger_of_their_own_spreads() {
        if !crate::gpu::probe().available() {
            eprintln!("no usable CUDA device: device probe test skipped");
            return;
        }
        let runs: Vec<Calibration> = (0..4)
            .map(|_| {
                let rates = measure().expect("a usable device yields device rates");
                eprintln!("{:?}", rates);
                Calibration::default().with_device_rates(rates)
            })
            .collect();
        // The first probe of a process runs on a cold device and is not compared.
        let runs = &runs[1..];
        for run in runs {
            assert!(run.is_usable());
            assert_eq!(run.device_source, "probe");
        }
        for (i, j) in [(0, 1), (1, 2), (0, 2)] {
            for (term, ratio, band) in runs[i].agreement(&runs[j]) {
                eprintln!("probes {} {} {}: ratio {:.4} band {:.4}", i, j, term, ratio, band);
                assert!(
                    ratio <= band,
                    "probes {} and {} differ on {} by {:.4}x, past the band {:.4}x (the larger of the two probes' own spreads)",
                    i, j, term, ratio, band
                );
            }
        }
    }
}
