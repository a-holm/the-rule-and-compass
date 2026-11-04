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
use std::fmt::Write as _;

/// Symbols and bars per symbol of the probe store. The product is the device
/// dispatch floor, so `--gpu on` really runs the kernels.
pub const PROBE_SYMBOLS: usize = 50;
pub const PROBE_BARS: usize = 20_000;

/// Which stage seconds carry a class: the feature stage, or the cross-section
/// stage for the rank class.
#[derive(Clone, Copy)]
enum Stage {
    Feature,
    CrossSection,
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

const PLANS: [Plan; 8] = [
    Plan {
        op: OpClass::Elementwise,
        stage: Stage::Feature,
        small: "lag_r1_1,lag_r1_2",
        large: "lag_r1_1,lag_r1_2,lag_r1_3,lag_r1_4,lag_r1_5,lag_r1_6,lag_r1_7,lag_r1_8,lag_r1_9,lag_r1_10",
        counts: (2, 10),
        host_once: false,
    },
    Plan {
        op: OpClass::Calendar,
        stage: Stage::Feature,
        small: "dow_sin,dow_cos",
        large: "dow_sin,dow_cos,hour_sin,hour_cos,month_sin,month_cos,dom_sin,dom_cos",
        counts: (2, 8),
        host_once: false,
    },
    Plan {
        op: OpClass::RollingSum,
        stage: Stage::Feature,
        small: "ma5,ma10",
        large: "ma5,ma10,ma20,ma50,ma100,ma200",
        counts: (2, 6),
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

/// The probe's wall clock budget, seconds, for the `rcsr calibrate` summary.
pub const PROBE_BUDGET_S: f64 = 60.0;

struct ProbeDir(std::path::PathBuf);

impl Drop for ProbeDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn probe_csv() -> String {
    let mut text = String::with_capacity(PROBE_SYMBOLS * PROBE_BARS * 64);
    text.push_str("ts,symbol,open,high,low,close,volume\n");
    for s in 0..PROBE_SYMBOLS {
        let phase = s as f64 * 0.37;
        for i in 0..PROBE_BARS {
            let x = i as f64;
            let p = 100.0 + 5.0 * (x * 0.37 + phase).sin() + 0.002 * x;
            let o = p + 0.4 * (x * 0.11 + phase).cos();
            let c = p + 0.5 * (x * 0.7).sin();
            let h = o.max(c) + 0.8 + (i % 5) as f64 * 0.03;
            let l = o.min(c) - 0.7 - (i % 3) as f64 * 0.02;
            let _ = writeln!(
                text,
                "{},S{:03},{:.4},{:.4},{:.4},{:.4},{}",
                1_700_000_000 + i as i64 * 900,
                s,
                o,
                h,
                l,
                c,
                10 + i % 7
            );
        }
    }
    text
}

struct Runner {
    store: String,
    out: String,
}

impl Runner {
    /// Stage seconds of one dry run with `features` on `gpu`.
    fn run(&self, features: &str, gpu: GpuMode, stage: Stage) -> Result<f64, String> {
        let opts = Options {
            data: self.store.clone(),
            out: Some(self.out.clone()),
            features: vec![features.to_string()],
            gpu,
            dry_run: true,
            allow_lockbox: true,
            calibration: crate::dispatch::CalibrationRequest::Anchors,
            ..Options::default()
        };
        let report = enrich::build(&opts)?;
        if gpu == GpuMode::On && report.gpu_columns == 0 {
            return Err(format!(
                "the device did not run '{}' ({} fallback columns)",
                features, report.gpu_fallback
            ));
        }
        Ok(match stage {
            Stage::Feature => report.feature_seconds,
            Stage::CrossSection => report.xs_seconds,
        })
    }
}

/// Measure the enrich planner rates on this machine. With `device` false only
/// the host terms are timed and the device terms keep their anchors.
pub fn measure(device: bool) -> Result<EnrichCalibration, String> {
    let dir = std::env::temp_dir().join(format!("rcsr-calibrate-enrich-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    let dir = ProbeDir(dir);
    let rows = PROBE_SYMBOLS * PROBE_BARS;
    let table = crate::data::parse_table_with_time(&probe_csv(), false, Some("ts"))?;
    let store = dir.0.join("probe.rcsb").to_string_lossy().into_owned();
    crate::store::write_store_with_time(&store, &table, 65_536, Some(0))?;
    drop(table);
    let runner = Runner {
        store,
        out: dir.0.join("out.rcsb").to_string_lossy().into_owned(),
    };

    let anchors = EnrichRates::anchors();
    let mut rates = anchors;
    rates.write_bps = enrich::measure_write_bandwidth(&dir.0, 128 * 1024 * 1024)?;
    let mrow = rows as f64 / 1_000_000.0;
    let mut spread_classes = [1.0; OP_CLASSES];
    let mut context_ms_per_mrow = None;
    let mut spread_context = 1.0;
    for plan in PLANS.iter() {
        let (n1, n2) = plan.counts;
        let cpu_large = runner.run(plan.large, GpuMode::Off, plan.stage)?;
        let anchor = anchors.class(plan.op);
        let mut rate = anchor;
        if plan.host_once {
            rate.cpu_fixed_ms = (cpu_large * 1000.0 / mrow).max(0.0);
        } else {
            let cpu_small = runner.run(plan.small, GpuMode::Off, plan.stage)?;
            let col_ms = ((cpu_large - cpu_small) * 1000.0 / mrow / (n2 - n1) as f64).max(0.0);
            rate.cpu_col_ms = col_ms;
            rate.cpu_fixed_ms = (cpu_small * 1000.0 / mrow - col_ms * n1 as f64).max(0.0);
        }
        if device {
            let gpu_small = runner.run(plan.small, GpuMode::On, plan.stage)?;
            let gpu_large = runner.run(plan.large, GpuMode::On, plan.stage)?;
            let per_cell_transfer =
                8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND + 1.0 / rates.write_bps);
            let d_cells = rows as f64 * (n2 - n1) as f64;
            let kernel_s = (gpu_large - gpu_small) - d_cells * per_cell_transfer;
            rate.gpu_cells_per_s = d_cells / kernel_s.max(d_cells / 1.0e12);
            if plan.op == OpClass::Elementwise && matches!(plan.stage, Stage::Feature) {
                let repeat = runner.run(plan.small, GpuMode::On, plan.stage)?;
                let (lo, hi) = (gpu_small.min(repeat), gpu_small.max(repeat));
                spread_context = if lo > 0.0 { hi / lo } else { 1.0 };
                let cells = rows as f64 * n1 as f64;
                let context_s = lo - cells / rate.gpu_cells_per_s - cells * per_cell_transfer;
                context_ms_per_mrow =
                    Some(((context_s * 1000.0 - anchors.fixed_overhead_ms) / mrow).max(1.0));
            }
        }
        rates.classes[plan.op.index()] = rate;
        spread_classes[plan.op.index()] = 1.0;
    }
    if let Some(ctx) = context_ms_per_mrow {
        rates.context_ms_per_mrow = ctx;
    }
    rates.validate()?;
    Ok(EnrichCalibration {
        rows,
        rates,
        spread_context,
        spread_classes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_sets_have_the_declared_column_counts_and_classes() {
        for plan in PLANS.iter() {
            assert_eq!(plan.small.split(',').count(), plan.counts.0, "{:?}", plan.op);
            assert_eq!(plan.large.split(',').count(), plan.counts.1, "{:?}", plan.op);
            assert!(plan.counts.0 < plan.counts.1);
        }
        assert_eq!(PROBE_SYMBOLS * PROBE_BARS, enrich::GPU_MIN_ROWS);
    }
}
