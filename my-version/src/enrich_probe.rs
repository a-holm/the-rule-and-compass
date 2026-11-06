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

/// Symbols and bars per symbol of the probe store: the shape of the 1M-row,
/// 109-symbol panel the planner anchors come from. The product reaches the device
/// dispatch floor, so `--gpu on` really runs the kernels.
pub const PROBE_SYMBOLS: usize = 109;
pub const PROBE_BARS: usize = 9_175;

/// The probe symbol that serves as the benchmark series: the shared compact pass
/// computes the benchmark columns only when the series exists.
const PROBE_BENCHMARK: &str = "S000";

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

fn probe_csv() -> String {
    let mut text = String::with_capacity(PROBE_SYMBOLS * PROBE_BARS * 64);
    text.push_str("ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base\n");
    // A deterministic random walk with the raw flow columns of a real panel.
    let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
    let mut uniform = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64
    };
    for s in 0..PROBE_SYMBOLS {
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
            cross_symbol: PROBE_BENCHMARK.to_string(),
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
    // (class index, extra cells, device seconds for those cells)
    let mut slopes: Vec<(usize, f64, f64)> = Vec::new();
    let mrow = rows as f64 / 1_000_000.0;
    let mut spread_classes = [1.0; OP_CLASSES];
    let mut context_ms_per_mrow = None;
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
            let d_cells = rows as f64 * (n2 - n1) as f64;
            slopes.push((plan.op.index(), d_cells, gpu_large - gpu_small));
            if plan.op == OpClass::Elementwise && matches!(plan.stage, Stage::Feature) {
                let context_s = runner.fresh_context(plan.small)?;
                context_ms_per_mrow = Some((context_s * 1000.0 / mrow).max(1.0));
            }
        }
        rates.classes[plan.op.index()] = rate;
        spread_classes[plan.op.index()] = 1.0;
    }
    // The write term is what a run pays per output cell beyond the bus and the
    // kernel, fitted from the same runs that set the device rates, so a busy
    // disk at probe time does not move the plans. Without a device the
    // filesystem figure stands.
    let per_cell_bus = 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND);
    let mut implied: Vec<f64> = slopes
        .iter()
        .filter_map(|(idx, cells, secs)| {
            let left = secs / cells - per_cell_bus - 1.0 / anchors.classes[*idx].gpu_cells_per_s;
            (left > 0.0).then(|| 8.0 / left)
        })
        .collect();
    implied.sort_by(|a, b| a.total_cmp(b));
    rates.write_bps = match implied.get(implied.len() / 2) {
        Some(w) => *w,
        None => enrich::measure_write_bandwidth(&dir.0, 128 * 1024 * 1024)?,
    };
    let per_cell_transfer = per_cell_bus + 8.0 / rates.write_bps;
    for (idx, cells, secs) in &slopes {
        let kernel_s = secs - cells * per_cell_transfer;
        let anchor = anchors.classes[*idx].gpu_cells_per_s;
        rates.classes[*idx].gpu_cells_per_s =
            cells / kernel_s.max(cells / (enrich::RATE_CLAMP * anchor));
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
