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

/// Pairs of small and large device runs per class.
const DEVICE_REPEATS: usize = 2;

/// Device runs of one class: the small and the large column set, repeated in
/// pairs. The rate is fitted from the smallest sample of each set and the
/// spread is the largest over the smallest of the per-pair rates.
struct ClassRuns {
    idx: usize,
    /// Extra cells of the large set over the small one.
    d_cells: f64,
    small: Vec<Sample>,
    large: Vec<Sample>,
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

const PLANS: [Plan; 7] = [
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
        op: OpClass::Trend,
        stage: Stage::Feature,
        small: "trend_ma20_ratio",
        large: "trend_ma20_ratio,trend_ma50_ratio,trend_ma100_ratio,trend_ma200_ratio,trend_ma20_slope,trend_ma50_slope,trend_ma100_slope,trend_ma200_slope,trend_don364_pos,trend_hi364_dist,trend_lo364_dist,trend_mom4w,trend_mom12w,trend_mom26w,trend_mom52w,trend_ma20_ratio_z,trend_ma50_ratio_z,trend_ma100_ratio_z,trend_ma200_ratio_z,trend_ma20_slope_z,trend_ma50_slope_z,trend_ma100_slope_z,trend_ma200_slope_z,trend_mom4w_z,trend_mom12w_z,trend_mom26w_z,trend_mom52w_z",
        counts: (1, 27),
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
    /// Per class, in index order.
    classes: Vec<ClassFit>,
}

struct ClassFit {
    idx: usize,
    /// Kernel rate; the anchor when the class is not resolved.
    rate: f64,
    /// Largest over smallest repeat of a resolved class; 1.0 otherwise.
    spread: f64,
    /// The kernel term after the bus and the write exceeds the noise of the
    /// repeats. A resolved rate is limited to `RATE_CLAMP` times the anchor.
    resolved: bool,
    /// Seconds of the extra cells left for the kernel, before any limit.
    kernel_s: f64,
    /// Smaller of the stage-seconds ranges of the two sets over the repeats.
    noise_s: f64,
}

/// Largest spread a file records.
const MAX_SPREAD: f64 = 1000.0;

fn least(v: impl Iterator<Item = f64>) -> f64 {
    v.fold(f64::INFINITY, f64::min)
}

/// The write term is the output write the same device runs report, seconds per
/// cell, so a busy disk at probe time does not move the plans. A class's kernel
/// rate is what its extra cells cost after the bus and the write. Timing noise
/// on a fixed workload only adds time, so each run set is represented by its
/// smallest sample. A class whose runs report their write subtracts that
/// figure. A kernel that does not exceed the noise of the repeats (the smaller of the two sets' ranges, since a stall hits one set) is hidden
/// behind the transfers: timing does not resolve it and the class keeps its
/// anchor. A resolved rate is limited to `RATE_CLAMP` times the anchor.
fn fit_device(runs: &[ClassRuns]) -> DeviceFit {
    let anchors = EnrichRates::anchors();
    let per_cell_bus = 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND);
    let diff = |r: &ClassRuns| {
        let d_secs = least(r.large.iter().map(|s| s.stage_s)) - least(r.small.iter().map(|s| s.stage_s));
        let d_write = least(r.large.iter().map(|s| s.write_s)) - least(r.small.iter().map(|s| s.write_s));
        (d_secs, d_write)
    };
    let written: Vec<f64> = runs
        .iter()
        .map(|r| (r, diff(r).1))
        .filter(|(_, w)| *w > 0.0)
        .map(|(r, w)| 8.0 * r.d_cells / w)
        .collect();
    let write_bps = median(written);
    let per_cell_write = write_bps.map_or(0.0, |w| 8.0 / w);
    let kernel = |r: &ClassRuns, d_secs: f64, d_write: f64| {
        let write_s = if d_write > 0.0 { d_write } else { r.d_cells * per_cell_write };
        d_secs - write_s - r.d_cells * per_cell_bus
    };
    let classes = runs
        .iter()
        .map(|r| {
            let anchor = anchors.classes[r.idx].gpu_cells_per_s;
            let (d_secs, d_write) = diff(r);
            let kernel_s = kernel(r, d_secs, d_write);
            let pairs: Vec<f64> = r
                .small
                .iter()
                .zip(&r.large)
                .map(|(s, l)| kernel(r, l.stage_s - s.stage_s, l.write_s - s.write_s))
                .collect();
            let (lo, hi) = pairs
                .iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), v| (lo.min(*v), hi.max(*v)));
            // A stall hits one sample of one set: the set that stayed clean sets
            // the noise.
            let range = |v: &[Sample]| {
                let (lo, hi) = v
                    .iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), s| (lo.min(s.stage_s), hi.max(s.stage_s)));
                hi - lo
            };
            let noise_s = if pairs.len() > 1 { range(&r.small).min(range(&r.large)) } else { f64::INFINITY };
            let fastest = r.d_cells / (enrich::RATE_CLAMP * anchor);
            let resolved = kernel_s > noise_s;
            ClassFit {
                idx: r.idx,
                rate: if resolved { r.d_cells / kernel_s.max(fastest) } else { anchor },
                spread: if resolved { (hi / lo.max(hi / MAX_SPREAD)).max(1.0) } else { 1.0 },
                resolved,
                kernel_s,
                noise_s,
            }
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
    let mut slopes: Vec<ClassRuns> = Vec::new();
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
            let mut run = ClassRuns { idx: plan.op.index(), d_cells, small: Vec::new(), large: Vec::new() };
            for _ in 0..DEVICE_REPEATS {
                run.small.push(runner.sample(plan.small, GpuMode::On, plan.stage)?);
                run.large.push(runner.sample(plan.large, GpuMode::On, plan.stage)?);
            }
            slopes.push(run);
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
    for c in fit.classes {
        rates.classes[c.idx].gpu_cells_per_s = c.rate;
        spread_classes[c.idx] = c.spread;
        if std::env::var_os("RCSR_PROBE_TRACE").is_some() {
            eprintln!("probe-trace idx={} resolved={} kernel_s={:.6} noise_s={:.6} rate={:.4e}", c.idx, c.resolved, c.kernel_s, c.noise_s, c.rate);
        }
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

    const CELLS: f64 = 4.0e6;

    fn bus() -> f64 {
        CELLS * 8.0 * (1.0 / GPU_TRANSFER_BYTES_PER_SECOND + 1.0 / GPU_D2H_BYTES_PER_SECOND)
    }

    /// Pairs of (small stage, small write, large stage, large write) seconds.
    fn runs(op: OpClass, pairs: &[(f64, f64, f64, f64)]) -> ClassRuns {
        let sample = |stage_s, write_s| Sample { stage_s, write_s };
        ClassRuns {
            idx: op.index(),
            d_cells: CELLS,
            small: pairs.iter().map(|p| sample(p.0, p.1)).collect(),
            large: pairs.iter().map(|p| sample(p.2, p.3)).collect(),
        }
    }

    fn class(fit: &DeviceFit, op: OpClass) -> &ClassFit {
        fit.classes.iter().find(|c| c.idx == op.index()).unwrap()
    }

    #[test]
    fn the_write_noise_of_a_run_cancels_in_its_kernel_rate() {
        let kernel = 0.05;
        // Two pairs whose writes differ by 15% but whose kernel time is equal.
        let r = runs(
            OpClass::CompactPass,
            &[(0.2, 0.05, 0.2 + kernel + bus() + 0.20, 0.25), (0.2, 0.05, 0.2 + kernel + bus() + 0.23, 0.28)],
        );
        let fit = fit_device(&[r]);
        let c = class(&fit, OpClass::CompactPass);
        assert!(c.resolved && (c.rate - CELLS / kernel).abs() < 1.0, "{} {}", c.rate, c.kernel_s);
        assert!((c.spread - 1.0).abs() < 1e-9, "{}", c.spread);
        let w = fit.write_bps.unwrap();
        assert!(w > 1.5e8 && w < 2.0e8, "{w}");
    }

    #[test]
    fn a_slow_outlier_in_one_run_does_not_move_the_rate() {
        // Noise only adds time: the second small run is 30 ms slow, the pair
        // rates differ, the smallest samples give the clean figure.
        let kernel = 0.05;
        let large = 0.2 + kernel + bus() + 0.2;
        let r = runs(
            OpClass::Mtf,
            &[(0.2, 0.05, large, 0.25), (0.23, 0.05, large, 0.25)],
        );
        let fit = fit_device(&[r]);
        let c = class(&fit, OpClass::Mtf);
        assert!(c.resolved, "{} {}", c.kernel_s, c.noise_s);
        assert!((c.rate - CELLS / kernel).abs() < 1.0, "{}", c.rate);
        assert!(c.spread > 1.2, "the repeats still disagree: {}", c.spread);
    }

    #[test]
    fn a_kernel_hidden_behind_the_transfers_keeps_the_anchor() {
        let ew = OpClass::Elementwise;
        let anchor = EnrichRates::anchors().class(ew).gpu_cells_per_s;
        // The extra cells cost only bus and write: the kernel term is zero.
        let hidden = runs(ew, &[(0.2, 0.05, 0.2 + bus() + 0.2, 0.25), (0.2, 0.05, 0.2 + bus() + 0.2, 0.25)]);
        let fit = fit_device(&[hidden]);
        let c = class(&fit, ew);
        assert!(!c.resolved);
        assert_eq!((c.rate, c.spread), (anchor, 1.0));
        // A kernel below the noise of both sets is not resolved either.
        let tiny = 0.02;
        let noisy = runs(
            ew,
            &[(0.2, 0.05, 0.2 + tiny + bus() + 0.2, 0.25), (0.23, 0.05, 0.2 + tiny + 0.03 + bus() + 0.2, 0.25)],
        );
        let c = fit_device(&[noisy]).classes.remove(0);
        assert!(!c.resolved);
        assert_eq!(c.rate, anchor);
        // A single repeat has no noise estimate and is not resolved.
        let single = runs(ew, &[(0.2, 0.05, 0.2 + 0.1 + bus() + 0.2, 0.25)]);
        assert!(!fit_device(&[single]).classes[0].resolved);
    }

    #[test]
    fn a_stall_in_one_set_does_not_unresolve_the_class() {
        let kernel = 0.05;
        let large = 0.2 + kernel + bus() + 0.2;
        let r = runs(OpClass::Mtf, &[(0.2, 0.05, large, 0.25), (0.2, 0.05, large + 0.1, 0.25)]);
        let c = fit_device(&[r]).classes.remove(0);
        assert!(c.resolved, "{} {}", c.kernel_s, c.noise_s);
        assert!((c.rate - CELLS / kernel).abs() < 1.0, "{}", c.rate);
    }

    #[test]
    fn a_resolved_kernel_faster_than_four_times_the_anchor_is_limited() {
        let ew = OpClass::Elementwise;
        let anchor = EnrichRates::anchors().class(ew).gpu_cells_per_s;
        let fast = CELLS / (enrich::RATE_CLAMP * anchor) / 2.0;
        let r = runs(ew, &[(0.2, 0.05, 0.2 + fast + bus() + 0.2, 0.25), (0.2, 0.05, 0.2 + fast + bus() + 0.2, 0.25)]);
        let c = fit_device(&[r]).classes.remove(0);
        assert!(c.resolved);
        assert_eq!(c.rate, enrich::RATE_CLAMP * anchor);
    }

    #[test]
    fn no_reported_write_means_no_write_fit() {
        let fit = fit_device(&[runs(OpClass::Rank, &[(0.2, 0.0, 0.5, 0.0), (0.2, 0.0, 0.5, 0.0)])]);
        assert_eq!(fit.write_bps, None);
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
