//! Enrich planner probe for `rcsr calibrate` (#778).
//!
//! Builds a synthetic 15m panel under the temp directory, runs the real
//! enrich producers on it with two column counts per operation class, and fits
//! the planner rates from the measured stage seconds. Only this module times
//! anything; the planner takes the fitted rates as arguments.

use crate::enrich::{
    self, EnrichCalibration, EnrichRates, GpuMode, OpClass, Options, OP_CLASSES,
    GPU_D2H_BYTES_PER_SECOND, GPU_TRANSFER_BYTES_PER_SECOND,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;

/// Symbols and bars per symbol of the probe store: the shape of the 1M-row,
/// 109-symbol panel the planner anchors come from. The product reaches the device
/// dispatch floor, so `--gpu on` really runs the kernels.
pub const PROBE_SYMBOLS: usize = 109;
pub const PROBE_BARS: usize = 9_175;

/// Symbols of the probe store that times the shared host passes.
const PASS_SYMBOLS: usize = 27;

/// The probe symbol that serves as the benchmark series: the shared compact pass
/// computes the benchmark columns only when the series exists.
const PROBE_BENCHMARK: &str = "S000";

/// A symbol the probe store does not hold: a run with it has no benchmark series.
const PROBE_NO_BENCHMARK: &str = "NOBENCH";

/// Which stage seconds carry a class: the feature stage, or the cross-section
/// stage for the rank class.
#[derive(Clone, Copy)]
enum Stage {
    Feature,
    CrossSection,
}

/// Device runs of each large column set, each against the one small run of its
/// class: the class rate is the median over the repeats and its spread is
/// their largest over their smallest.
const DEVICE_REPEATS: usize = 2;

/// Extra cells of the large column set over the small one, and what the
/// device runs of those cells cost beyond the small set.
struct Slope {
    idx: usize,
    d_cells: f64,
    d_secs: f64,
    d_write: f64,
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    v.sort_by(|a, b| a.total_cmp(b));
    v.get(v.len() / 2).copied()
}

struct Plan {
    op: OpClass,
    stage: Stage,
    small: &'static str,
    large: &'static str,
    /// Columns in `small` and `large`.
    counts: (usize, usize),
    /// The host pass costs the same for any column count: time the large set
    /// once and keep the anchor per-column part.
    host_once: bool,
}

const PLANS: [Plan; 6] = [
    Plan {
        op: OpClass::Elementwise,
        stage: Stage::Feature,
        small: "lag_r1_1,lag_r1_2",
        large: "lag_r1_1,lag_r1_2,lag_r1_3,lag_r1_4,lag_r1_5,lag_r1_6,lag_r1_7,lag_r1_8,lag_r1_9,lag_r1_10",
        counts: (2, 10),
        host_once: false,
    },
    Plan {
        op: OpClass::RollingStd,
        stage: Stage::Feature,
        small: "vol5,vol20",
        large: "vol5,vol20,vol60,vol120,vz96",
        counts: (2, 5),
        host_once: false,
    },
    Plan {
        op: OpClass::Rank,
        stage: Stage::CrossSection,
        small: "xs_rank_r1",
        large: "xs_rank_r1,xs_rank_r16,xs_rank_r20,xs_rank_vol20",
        counts: (1, 4),
        host_once: false,
    },
    Plan {
        op: OpClass::Mtf,
        stage: Stage::Feature,
        small: "mtf_1h_ret1",
        large: "mtf_1h_ret1,mtf_1h_ret4,mtf_1h_ret24,mtf_1h_ma_ratio,mtf_1h_vol,mtf_1h_atr_pct,mtf_1h_rsi,mtf_1h_close_pos,mtf_1h_donchian_pos,mtf_1h_range_pct,mtf_1h_trend_slope,mtf_1h_bar_age",
        counts: (1, 12),
        host_once: false,
    },
    Plan {
        op: OpClass::Py15mPass,
        stage: Stage::Feature,
        small: "r192",
        large: "r192,ema8,ema20,ema50,ema100,ema200,macd,macd_sig",
        counts: (1, 8),
        host_once: false,
    },
    Plan {
        op: OpClass::CompactPass,
        stage: Stage::Feature,
        small: "vol16",
        large: "vol16,vol96,bb96,bb_width96,volrat",
        counts: (1, 5),
        host_once: true,
    },
];

struct ProbeDir(std::path::PathBuf);

impl Drop for ProbeDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn probe_csv(symbols: usize) -> String {
    let mut text = String::with_capacity(symbols * PROBE_BARS * 64);
    text.push_str("ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base\n");
    // A deterministic random walk with the raw flow columns of a real panel.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for s in 0..symbols {
        let mut close = 100.0 + s as f64;
        for i in 0..PROBE_BARS {
            let open = close;
            let ret = 0.004 * (uniform() + uniform() + uniform() - 1.5);
            close = open * ret.exp();
            let high = open.max(close) * (1.0 + 0.002 * uniform());
            let low = open.min(close) * (1.0 - 0.002 * uniform());
            let volume = 10.0 + 90.0 * uniform() * uniform();
            let _ = writeln!(
                text,
                "{},S{:03},{:.4},{:.4},{:.4},{:.4},{:.3},{:.3},{},{:.3}",
                1_700_000_000 + i as i64 * 900,
                s,
                open,
                high,
                low,
                close,
                volume,
                volume * close,
                (20.0 + volume * 3.0 * uniform()) as u64,
                volume * (0.3 + 0.4 * uniform())
            );
        }
    }
    text
}

/// One timed run: the seconds of its stage and the seconds the run spent
/// writing output values.
struct Sample {
    stage_s: f64,
    write_s: f64,
}

struct Runner {
    store: String,
    out: String,
}

impl Runner {
    /// Stage seconds of one dry run with `features` on `gpu`.
    fn run(&self, features: &str, gpu: GpuMode, stage: Stage) -> Result<f64, String> {
        Ok(self.sample(features, gpu, stage)?.stage_s)
    }

    /// Stage seconds and output write seconds of one dry run.
    fn sample(&self, features: &str, gpu: GpuMode, stage: Stage) -> Result<Sample, String> {
        self.sample_with(features, gpu, stage, PROBE_BENCHMARK)
    }

    /// As [`Runner::run`] with `cross_symbol` as the benchmark symbol; a name
    /// the store does not hold leaves the benchmark series out.
    fn run_with(
        &self,
        features: &str,
        gpu: GpuMode,
        stage: Stage,
        cross_symbol: &str,
    ) -> Result<f64, String> {
        Ok(self.sample_with(features, gpu, stage, cross_symbol)?.stage_s)
    }

    fn sample_with(
        &self,
        features: &str,
        gpu: GpuMode,
        stage: Stage,
        cross_symbol: &str,
    ) -> Result<Sample, String> {
        let opts = Options {
            data: self.store.clone(),
            out: Some(self.out.clone()),
            features: vec![features.to_string()],
            gpu,
            dry_run: true,
            allow_lockbox: true,
            calibration: crate::dispatch::CalibrationRequest::Anchors,
            cross_symbol: cross_symbol.to_string(),
            ..Options::default()
        };
        let report = enrich::build(&opts)?;
        if gpu == GpuMode::On && report.gpu_columns == 0 {
            return Err(format!(
                "the device did not run '{}' ({} fallback columns)",
                features, report.gpu_fallback
            ));
        }
        Ok(Sample {
            stage_s: match stage {
                Stage::Feature => report.feature_seconds,
                Stage::CrossSection => report.xs_seconds,
            },
            write_s: report.gpu_write_seconds,
        })
    }
}

impl Runner {
    /// Context seconds (host read plus resident upload) of a device run in a
    /// fresh process: the device initialisation is part of what a real run pays.
    fn fresh_context(&self, features: &str) -> Result<f64, String> {
        let exe = std::env::current_exe().map_err(|e| format!("current exe: {}", e))?;
        let out = std::process::Command::new(exe)
            .args(["enrich", "--data", &self.store, "--out", &self.out])
            .args(["--features", features, "--gpu", "on", "--dry-run", "--json"])
            .args(["--calibration", "anchors", "--cross-symbol", PROBE_BENCHMARK])
            .output()
            .map_err(|e| format!("cannot start rcsr enrich: {}", e))?;
        if !out.status.success() {
            return Err(format!(
                "rcsr enrich exited {:?}: {}",
                out.status.code(),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        let doc = crate::bench_stats::parse_json(&String::from_utf8_lossy(&out.stdout))?;
        let field = |outer: &str, key: &str| {
            doc.get(outer)
                .and_then(|o| o.get(key))
                .and_then(|v| v.as_f64())
                .ok_or_else(|| format!("rcsr enrich report has no {}.{}", outer, key))
        };
        Ok(field("timing", "feature_context_s")? + field("gpu", "h2d_s")?)
    }
}

/// Device terms fitted from the paired runs.
struct DeviceFit {
    /// Output write bandwidth the runs reported; `None` when no run wrote.
    write_bps: Option<f64>,
    /// Class index, kernel rate (median over the repeats) and spread (largest
    /// over smallest repeat).
    classes: Vec<(usize, f64, f64)>,
}

/// The write term is the output write the same device runs report, seconds per
/// cell, so a busy disk at probe time does not move the plans. A class's kernel
/// rate is what its extra cells cost after the bus and the write. A class whose
/// runs report their write subtracts that figure, so the write noise of a run
/// cancels in its own kernel term. The rate is limited to `RATE_CLAMP` times
/// the anchor: a kernel hidden behind the transfers is not resolved by timing.
fn fit_device(slopes: &[Slope]) -> DeviceFit {
    let anchors = EnrichRates::anchors();
    let per_cell_bus = 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND);
    let written: Vec<f64> = slopes
        .iter()
        .filter(|s| s.d_write > 0.0)
        .map(|s| 8.0 * s.d_cells / s.d_write)
        .collect();
    let write_bps = median(written);
    let per_cell_write = write_bps.map_or(0.0, |w| 8.0 / w);
    let mut per_class: BTreeMap<usize, Vec<f64>> = BTreeMap::new();
    for s in slopes {
        let write_s = if s.d_write > 0.0 { s.d_write } else { s.d_cells * per_cell_write };
        let kernel_s = s.d_secs - write_s - s.d_cells * per_cell_bus;
        let anchor = anchors.classes[s.idx].gpu_cells_per_s;
        per_class
            .entry(s.idx)
            .or_default()
            .push(s.d_cells / kernel_s.max(s.d_cells / (enrich::RATE_CLAMP * anchor)));
    }
    let classes = per_class
        .into_iter()
        .map(|(idx, samples)| {
            let (lo, hi) = samples
                .iter()
                .fold((f64::INFINITY, 0.0f64), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            let rate = median(samples).unwrap_or(anchors.classes[idx].gpu_cells_per_s);
            (idx, rate, hi / lo)
        })
        .collect();
    DeviceFit { write_bps, classes }
}

/// Measure the enrich planner rates on this machine. With `device` false only
/// the host terms are timed and the device terms keep their anchors.
pub fn measure(device: bool) -> Result<EnrichCalibration, String> {
    let dir = std::env::temp_dir().join(format!("rcsr-calibrate-enrich-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let dir = ProbeDir(dir);
    let rows = PROBE_SYMBOLS * PROBE_BARS;
    let table = crate::data::parse_table_with_time(&probe_csv(PROBE_SYMBOLS), false, Some("ts"))?;
    let store = dir.0.join("probe.rcsb").to_string_lossy().into_owned();
    crate::store::write_store_with_time(&store, &table, 65_536, Some(0))?;
    drop(table);
    let runner = Runner {
        store,
        out: dir.0.join("out.rcsb").to_string_lossy().into_owned(),
    };
    // A shared pass costs the same per row on any number of symbols, so its
    // host cost is timed on the first symbols only, which hold the benchmark
    // series at full length.
    let pass_rows = PASS_SYMBOLS * PROBE_BARS;
    let pass_table = crate::data::parse_table_with_time(&probe_csv(PASS_SYMBOLS), false, Some("ts"))?;
    let pass_store = dir.0.join("pass.rcsb").to_string_lossy().into_owned();
    crate::store::write_store_with_time(&pass_store, &pass_table, 65_536, Some(0))?;
    drop(pass_table);
    let pass_runner = Runner {
        store: pass_store,
        out: dir.0.join("pass-out.rcsb").to_string_lossy().into_owned(),
    };

    let anchors = EnrichRates::anchors();
    let mut rates = anchors;
    let mut slopes: Vec<Slope> = Vec::new();
    let mrow = rows as f64 / 1_000_000.0;
    let mut spread_classes = [1.0; OP_CLASSES];
    let mut context_ms_per_mrow = None;
    for plan in PLANS.iter() {
        let (n1, n2) = plan.counts;
        let cpu_large = if plan.host_once {
            pass_runner.run(plan.large, GpuMode::Off, plan.stage)?
        } else {
            runner.run(plan.large, GpuMode::Off, plan.stage)?
        };
        let anchor = anchors.class(plan.op);
        let mut rate = anchor;
        if plan.host_once {
            let pass_mrow = pass_rows as f64 / 1_000_000.0;
            rate.cpu_fixed_ms = (cpu_large * 1000.0 / pass_mrow).max(0.0);
            if plan.op == OpClass::CompactPass {
                let bare =
                    pass_runner.run_with(plan.large, GpuMode::Off, plan.stage, PROBE_NO_BENCHMARK)?;
                rates.compact_no_benchmark_ms = (bare * 1000.0 / pass_mrow).max(0.0);
            }
        } else {
            let cpu_small = runner.run(plan.small, GpuMode::Off, plan.stage)?;
            let col_ms = ((cpu_large - cpu_small) * 1000.0 / mrow / (n2 - n1) as f64).max(0.0);
            rate.cpu_col_ms = col_ms;
            rate.cpu_fixed_ms = (cpu_small * 1000.0 / mrow - col_ms * n1 as f64).max(0.0);
        }
        if device {
            let d_cells = rows as f64 * (n2 - n1) as f64;
            let small = runner.sample(plan.small, GpuMode::On, plan.stage)?;
            for _ in 0..DEVICE_REPEATS {
                let large = runner.sample(plan.large, GpuMode::On, plan.stage)?;
                slopes.push(Slope {
                    idx: plan.op.index(),
                    d_cells,
                    d_secs: large.stage_s - small.stage_s,
                    d_write: large.write_s - small.write_s,
                });
            }
            if plan.op == OpClass::Elementwise && matches!(plan.stage, Stage::Feature) {
                let context_s = runner.fresh_context(plan.small)?;
                context_ms_per_mrow = Some((context_s * 1000.0 / mrow).max(1.0));
            }
        }
        rates.classes[plan.op.index()] = rate;
    }
    let fit = fit_device(&slopes);
    rates.write_bps = match fit.write_bps {
        Some(w) => w,
        None => enrich::measure_write_bandwidth(&dir.0, 128 * 1024 * 1024)?,
    };
    for (idx, rate, spread) in fit.classes {
        rates.classes[idx].gpu_cells_per_s = rate;
        spread_classes[idx] = spread;
    }
    if let Some(ctx) = context_ms_per_mrow {
        rates.context_ms_per_mrow = ctx;
    }
    rates.validate()?;
    Ok(EnrichCalibration {
        rows,
        rates,
        spread_context: 1.0,
        spread_classes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slope(idx: usize, d_secs: f64, d_write: f64) -> Slope {
        Slope { idx, d_cells: 4.0e6, d_secs, d_write }
    }

    #[test]
    fn the_write_noise_of_a_run_cancels_in_its_kernel_rate() {
        let rank = OpClass::Rank.index();
        let kernel = 0.05;
        let bus = 4.0e6 * 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND);
        // Two repeats whose writes differ by 15% but whose kernel time is equal.
        let a = slope(OpClass::CompactPass.index(), kernel + bus + 0.20, 0.20);
        let b = slope(OpClass::CompactPass.index(), kernel + bus + 0.23, 0.23);
        let fit = fit_device(&[a, b, slope(rank, 0.4, 0.0)]);
        let (_, rate, spread) = *fit.classes.iter().find(|c| c.0 == OpClass::CompactPass.index()).unwrap();
        assert!((rate - 4.0e6 / kernel).abs() < 1.0, "{rate}");
        assert!((spread - 1.0).abs() < 1e-9, "{spread}");
        let w = fit.write_bps.unwrap();
        assert!(w > 1.5e8 && w < 2.0e8, "{w}");
    }

    #[test]
    fn a_hidden_kernel_gives_the_clamped_rate_and_no_write_means_no_fit() {
        let ew = OpClass::Elementwise.index();
        let anchor = EnrichRates::anchors().classes[ew].gpu_cells_per_s;
        let fit = fit_device(&[slope(ew, 0.0, 0.0)]);
        assert_eq!(fit.write_bps, None);
        assert_eq!(fit.classes[0].1, enrich::RATE_CLAMP * anchor);
    }

    #[test]
    fn the_spread_is_the_largest_repeat_over_the_smallest() {
        let idx = OpClass::Mtf.index();
        let bus = 4.0e6 * 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND);
        let fit = fit_device(&[slope(idx, bus + 0.1 + 0.1, 0.1), slope(idx, bus + 0.15 + 0.1, 0.1)]);
        assert!((fit.classes[0].2 - 1.5).abs() < 1e-9, "{:?}", fit.classes);
    }

    #[test]
    fn probe_sets_have_the_declared_column_counts_and_classes() {
        for plan in PLANS.iter() {
            assert_eq!(plan.small.split(',').count(), plan.counts.0, "{:?}", plan.op);
            assert_eq!(plan.large.split(',').count(), plan.counts.1, "{:?}", plan.op);
            assert!(plan.counts.0 < plan.counts.1);
        }
        assert!(PROBE_SYMBOLS * PROBE_BARS >= enrich::GPU_MIN_ROWS);
    }
}
