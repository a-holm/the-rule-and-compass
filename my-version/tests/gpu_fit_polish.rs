//! #496: `--fit-polish` is saved search state and the device run honours it.
//! Device tests skip cleanly on a machine without CUDA.

#![cfg_attr(not(feature = "cuda"), allow(dead_code, unused_imports))]

use rcsr::checkpoint::Checkpoint;
use rcsr::data::Sample;
use rcsr::fit::{FitPolish, FitStrategy};
use rcsr::metrics::Metric;
use rcsr::search::{self, Config};

fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        y.push(1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2 + (((i * 37) % 11) as f64 - 5.0) * 1e-3);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

fn config(seed: u32) -> Config {
    Config {
        generations: 4,
        islands: 1,
        pop: 24,
        max_complexity: 12,
        fit_iters: 4,
        patience: 0,
        seed,
        fit_strategy: FitStrategy::Gpu,
        metric: Metric::Squared,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        ..Default::default()
    }
}

#[test]
fn fit_polish_defaults_to_f64_and_round_trips_through_the_checkpoint() {
    let train = sample(160, 1);
    let test = sample(40, 2);
    let base = Config { gpu: false, ..config(5) };
    assert_eq!(base.fit_polish, FitPolish::F64);
    let save = |cfg: &Config| {
        let mut text: Option<String> = None;
        search::run_resumable(cfg, &train, &test, None, 0, |cp| {
            text = Some(cp.to_text());
        })
        .expect("fresh run");
        text.expect("a checkpoint callback")
    };
    let default_text = save(&base);
    assert!(
        !default_text.contains("fit_polish"),
        "the default leaves the checkpoint bytes unchanged"
    );
    let cp = Checkpoint::from_text(&default_text).expect("parse");
    assert_eq!(cp.config.fit_polish, FitPolish::F64);
    for mode in [FitPolish::F32, FitPolish::Off] {
        let cfg = Config { fit_polish: mode, ..base.clone() };
        let text = save(&cfg);
        assert!(text.contains(&format!("config fit_polish {}", mode.name())));
        let cp = Checkpoint::from_text(&text).expect("parse");
        assert_eq!(cp.config.fit_polish, mode);
        let again = Checkpoint::from_text(&cp.to_text()).expect("reparse");
        assert_eq!(again.config.fit_polish, mode);
        assert_ne!(
            rcsr::checkpoint::config_hash(&again.config),
            rcsr::checkpoint::config_hash(&base)
        );
    }
}

#[cfg(feature = "cuda")]
#[test]
fn device_search_honours_fit_polish_and_splits_the_kernel_seconds() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(1200, 1);
    let test = sample(200, 2);
    let run = |mode: FitPolish| {
        let cfg = Config { fit_polish: mode, ..config(7) };
        let out = search::run(&cfg, &train, &test);
        let g = out.gpu.clone().expect("a device run reports gpu stats");
        let best = out.archive.items.iter().map(|s| s.error).fold(f64::INFINITY, f64::min);
        (g, best, out.archive.hash())
    };
    let (f64_stats, f64_best, f64_hash) = run(FitPolish::F64);
    let (default_stats, _, default_hash) = {
        let cfg = config(7);
        let out = search::run(&cfg, &train, &test);
        let g = out.gpu.clone().expect("gpu stats");
        let best = out.archive.items.iter().map(|s| s.error).fold(f64::INFINITY, f64::min);
        (g, best, out.archive.hash())
    };
    assert_eq!(default_hash, f64_hash, "the default is the f64 polish");
    assert_eq!(default_stats.fit_polish, FitPolish::F64);
    assert_eq!(f64_stats.fit_polish, FitPolish::F64);
    let mut shares = Vec::new();
    for (mode, stats, best) in [
        (FitPolish::F64, &f64_stats, f64_best),
        (FitPolish::F32, &run(FitPolish::F32).0, f64::NAN),
        (FitPolish::Off, &run(FitPolish::Off).0, f64::NAN),
    ] {
        assert_eq!(stats.fit_polish, mode);
        let l = &stats.ledger;
        assert!(l.fit_kernel_s() > 0.0, "{mode:?}: the fit kernel ran");
        let sum = l.fit_kernel_lm_s() + l.fit_kernel_polish_s();
        assert!(
            (sum - l.fit_kernel_s()).abs() <= 1e-12 * l.fit_kernel_s().max(1.0),
            "{mode:?}: {} + {} != {}",
            l.fit_kernel_lm_s(),
            l.fit_kernel_polish_s(),
            l.fit_kernel_s()
        );
        shares.push(l.fit_kernel_polish_s() / l.fit_kernel_s());
        let _ = best;
    }
    assert!(shares[0] > shares[2], "off spends less of the kernel on polish: {shares:?}");
    assert!(f64_best.is_finite());
}
