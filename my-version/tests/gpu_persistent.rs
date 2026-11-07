//! #334: the persistent fit and score buffers do not change a search.
//!
//! The expected archive hashes pin the search result; the persistent buffers must
//! not move them. Each run uses a GPU fit strategy so the batched fit, rank,
//! median, interquartile and maximum stages all execute on the device.
//! Device-dependent tests skip cleanly on a machine without CUDA.

#[cfg(feature = "cuda")]
use rcsr::data::Sample;
#[cfg(feature = "cuda")]
use rcsr::fit::FitStrategy;
#[cfg(feature = "cuda")]
use rcsr::metrics::Metric;
#[cfg(feature = "cuda")]
use rcsr::search::{self, Config, Outcome};

#[cfg(feature = "cuda")]
fn sample(n: usize, seed: u32) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let t = i as f64 / (n - 1) as f64 * 6.0 - 3.0;
        let s = ((i as u32).wrapping_mul(2_654_435_761) ^ seed) as f64 / u32::MAX as f64;
        x.push(vec![t, s * 2.0 - 1.0]);
        let v = 1.7 * t.sin() + 0.4 * t + 0.3 * s - 0.2;
        y.push(if i % 7 == 3 { 0.0 } else { v });
    }
    Sample {
        x,
        y,
        w: Vec::new(),
        ys: Vec::new(),
        pairs: None,
    }
}

#[cfg(feature = "cuda")]
fn config(metric: Metric) -> Config {
    Config {
        generations: 4,
        islands: 1,
        pop: 20,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 11,
        fit_strategy: FitStrategy::Gpu,
        metric,
        gpu: true,
        dispatch: rcsr::dispatch::DispatchMode::Device,
        gpu_notes: false,
        ..Default::default()
    }
}

#[cfg(feature = "cuda")]
fn archive_hash(out: &Outcome) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut eat = |bytes: &[u8]| {
        for &b in bytes {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for s in &out.archive.items {
        eat(&s.error.to_bits().to_le_bytes());
        eat(&(s.complexity as u64).to_le_bytes());
        eat(rcsr::expr::to_string(&s.tree, &["x0".to_string(), "x1".to_string()]).as_bytes());
    }
    h
}

/// `(metric, hash of the archive)`.
#[cfg(feature = "cuda")]
const GOLDEN: &[(Metric, u64)] = &[
    (Metric::Squared, 0xa92e_ae3a_b38e_20cd),
    (Metric::Absolute, 0x6a1c_d97e_7fbf_c519),
    (Metric::RankCorrelation, 0x0dbc_187c_7c8f_6de1),
    (Metric::Median, 0xdd06_72b6_d7be_afeb),
    (Metric::Interquartile, 0xbd8b_f909_d817_ba39),
    // 2 048 rows is below the crossover of the maximum fit cost model, so the CPU polish fits.
    (Metric::Maximum, 0x4cef_73c5_37d0_7645),
];

#[cfg(feature = "cuda")]
#[test]
fn gpu_search_archives_match_the_pre_change_path() {
    if !rcsr::gpu::probe().available() {
        rcsr::backend::skip_without_device("skipping GPU test: no usable CUDA device");
        return;
    }
    let train = sample(2_048, 5);
    let test = sample(256, 6);
    let mut device_fits = 0usize;
    for metric in [
        Metric::Squared,
        Metric::Absolute,
        Metric::RankCorrelation,
        Metric::Median,
        Metric::Interquartile,
        Metric::Maximum,
    ] {
        let out = search::run(&config(metric), &train, &test);
        let h = archive_hash(&out);
        let fit_gpu = out.gpu.as_ref().map(|g| g.fit_gpu).unwrap_or(0);
        eprintln!("{metric:?}: device fits {fit_gpu}");
        device_fits += fit_gpu;
        let (_, want) = GOLDEN
            .iter()
            .find(|(m, _)| *m == metric)
            .expect("a golden hash");
        assert_eq!(h, *want, "{metric:?}: archive differs from the pre-change path ({h:#018x})");
    }
    assert!(device_fits > 0, "no metric used the device fit arm");
}

/// The body of the C function whose signature starts with `head`, up to its
/// closing brace at column zero.
fn body<'a>(src: &'a str, head: &str) -> &'a str {
    let start = src.find(head).unwrap_or_else(|| panic!("{head} not found"));
    let end = src[start..].find("\n}\n").expect("function end") + start;
    &src[start..end]
}

/// #334 code inspection: the fit and score paths use no legacy-stream launch,
/// no event on the legacy stream, no synchronous `cudaMemcpy` and no
/// allocator call of their own.
#[test]
fn fit_and_score_paths_use_the_fit_stream_and_no_blocking_copies() {
    let src = include_str!("../src/cuda/eval.cu");
    let entry_points = [
        "float rc_fit_batch(",
        "float rc_fit_batch_from(",
        "float rc_rank_score_batch(",
        "float rc_order_stat_score_batch(",
        "float rc_max_scan_batch(",
    ];
    let helpers = [
        "static int rc_rank_prepare(",
        "static int rc_rank_runs(",
        "static int rc_fs_f64_twins(",
        "static int ft_launch(",
        "static float rc_fit_launch(",
    ];
    let delegating = ["float rc_fit_batch(", "float rc_fit_batch_from("];
    for head in entry_points.iter().chain(helpers.iter()) {
        let text = body(src, head);
        let code: String = text
            .lines()
            .filter(|l| !l.trim_start().starts_with("/*") && !l.trim_start().starts_with('*'))
            .collect::<Vec<_>>()
            .join("\n");
        assert!(!code.contains("cudaMemcpy("), "{head}: synchronous cudaMemcpy");
        assert!(!code.contains("rc_upload(") && !code.contains("rc_download("), "{head}: blocking copy helper");
        assert!(!code.contains("cudaMemset("), "{head}: synchronous memset");
        assert!(!code.contains("cudaDeviceSynchronize"), "{head}: device sync");
        assert!(!code.contains("cudaStreamSynchronize(0)"), "{head}: legacy sync");
        for launch in code.split("<<<").skip(1) {
            let cfg = &launch[..launch.find(">>>").expect("launch end")];
            assert!(
                cfg.contains("s_fit") || cfg.contains(", s)") || cfg.ends_with(", s"),
                "{head}: launch <<<{cfg}>>> is not on the fit stream"
            );
        }
    }
    for head in &entry_points {
        let code = body(src, head);
        assert!(!code.contains("cudaMalloc") && !code.contains("rc_alloc("), "{head}: allocates");
        assert!(!code.contains("cudaFree"), "{head}: frees");
        if delegating.contains(head) {
            assert!(code.contains("rc_fit_launch("), "{head}: does not delegate to rc_fit_launch");
        } else {
            assert!(code.contains("rc_fs_finish("), "{head}: no event synchronisation");
        }
        assert_eq!(code.matches("cudaEventSynchronize").count(), 0);
    }
    let launch = body(src, "static float rc_fit_launch(");
    assert!(!launch.contains("cudaMalloc") && !launch.contains("rc_alloc("), "rc_fit_launch: allocates");
    assert!(!launch.contains("cudaFree"), "rc_fit_launch: frees");
    assert!(launch.contains("rc_fs_finish("), "rc_fit_launch: no event synchronisation");
    assert_eq!(launch.matches("cudaEventSynchronize").count(), 0);
}
