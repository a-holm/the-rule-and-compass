//! #300: the host-thread partition of the feature phase closes on a GPU build.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options, Report};
use rcsr::store;

const ROWS: usize = 24_000;
const FEATURES: &str = "r1,r5,ma20,vol20,rsi6,don20,macd,macd_sig,macd_hist,hour_sin";
const CLOSE_EPS: f64 = 1e-9;
const JSON_EPS: f64 = 5e-7;

fn write_input(path: &str) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
    for sym in ["AAA", "BBB"] {
        for i in 0..ROWS / 2 {
            let p = 100.0 + (i as f64 * 0.013).sin() * 7.0 + i as f64 * 0.0001;
            text.push_str(&format!(
                "{},{},{},{},{},{},{}\n",
                i as i64 * 3600,
                sym,
                p,
                p + 0.5,
                p - 0.5,
                p + (i as f64 * 0.11).sin() * 0.1,
                10.0 + (i % 97) as f64
            ));
        }
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, 4096, Some(0)).unwrap();
}

fn object_field(json: &str, object: &str, key: &str) -> f64 {
    let start = json.find(&format!("\"{object}\":{{")).unwrap();
    let body = &json[start..];
    let body = &body[..body.find('}').unwrap()];
    let tag = format!("\"{key}\":");
    let at = body.find(&tag).unwrap_or_else(|| panic!("{object}.{key} missing")) + tag.len();
    let rest = &body[at..];
    let end = rest.find(|c| c == ',' || c == '}').unwrap_or(rest.len());
    rest[..end].parse().unwrap()
}

fn parts(r: &Report) -> [(&'static str, f64); 5] {
    [
        ("read", r.feature_read_seconds),
        ("context", r.feature_context_seconds),
        ("gpu", r.feature_gpu_seconds),
        ("write", r.gpu_write_seconds),
        ("cpu", r.feature_cpu_seconds),
    ]
}

#[test]
fn feature_phase_partition_closes_and_json_matches_the_report() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    enrich::set_gpu_floor_for_tests(0);
    let dir = std::env::temp_dir().join(format!("rcsr-enrich-ledger-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    write_input(&src);
    let r = enrich::build(&Options {
        data: src,
        out: Some(out),
        features: vec![FEATURES.to_string()],
        allow_lockbox: true,
        gpu: GpuMode::On,
        ..Default::default()
    })
    .unwrap();

    assert_eq!(r.backend, "gpu");
    assert!(r.gpu_columns > 0);
    for (name, v) in parts(&r).into_iter().chain([
        ("other", r.feature_other_seconds),
        ("kernel", r.gpu_kernel_seconds),
        ("h2d", r.gpu_h2d_seconds),
        ("d2h", r.gpu_d2h_seconds),
        ("feature", r.feature_seconds),
    ]) {
        assert!(v.is_finite(), "{name} = {v}");
        if name != "other" {
            assert!(v >= 0.0, "{name} = {v}");
        }
    }
    assert!(r.feature_seconds > 0.0);
    assert!(r.feature_context_seconds > 0.0, "the device context is built in the phase");
    assert!(r.feature_gpu_seconds > 0.0);
    assert!(r.gpu_write_seconds > 0.0);
    assert!(r.gpu_kernel_seconds > 0.0);
    assert!(r.gpu_d2h_seconds > 0.0);

    let accounted: f64 = parts(&r).iter().map(|(_, v)| v).sum();
    assert!(
        r.feature_other_seconds >= -CLOSE_EPS,
        "other = {} (feature {} accounted {})",
        r.feature_other_seconds,
        r.feature_seconds,
        accounted
    );
    assert!(
        (accounted + r.feature_other_seconds - r.feature_seconds).abs() <= CLOSE_EPS * r.feature_seconds,
        "partition {} + {} != {}",
        accounted,
        r.feature_other_seconds,
        r.feature_seconds
    );
    for (name, v) in parts(&r) {
        assert!(v <= r.feature_seconds + CLOSE_EPS, "{name} exceeds the phase");
    }

    let json = r.json();
    let checks = [
        ("gpu", "kernel_s", r.gpu_kernel_seconds),
        ("gpu", "h2d_s", r.gpu_h2d_seconds),
        ("gpu", "d2h_s", r.gpu_d2h_seconds),
        ("gpu", "write_s", r.gpu_write_seconds),
        ("timing", "feature_s", r.feature_seconds),
        ("timing", "feature_read_s", r.feature_read_seconds),
        ("timing", "feature_context_s", r.feature_context_seconds),
        ("timing", "feature_gpu_s", r.feature_gpu_seconds),
        ("timing", "feature_cpu_s", r.feature_cpu_seconds),
        ("timing", "feature_other_s", r.feature_other_seconds),
    ];
    for (object, key, want) in checks {
        let got = object_field(&json, object, key);
        assert!((got - want).abs() <= JSON_EPS, "{object}.{key}: json {got} report {want}");
    }
    let text = r.text_lines().join("\n");
    assert!(text.contains("feature_parts read="), "{text}");
    let _ = std::fs::remove_dir_all(&dir);
}
