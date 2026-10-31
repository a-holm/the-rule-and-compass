//! #381: tile outputs that stay resident on the device must not starve the
//! fused flow and price-action planes.
//!
//! The device byte budget hook limits what the context may add after creation,
//! so the test exhausts that share without exhausting the device.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options};
use rcsr::store;

const SYMBOLS: usize = 4;
const PER_SYMBOL: usize = 6_000;
const STEP: i64 = 900;

fn write_store(path: &str) -> usize {
    let mut text = String::from(
        "ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote\n",
    );
    for k in 0..SYMBOLS {
        for i in 0..PER_SYMBOL {
            let t = 1_700_000_000i64 + i as i64 * STEP;
            let p = 100.0 + k as f64 + (i as f64 * 0.013).sin() * 7.0;
            let v = 10.0 + (i % 97) as f64;
            text.push_str(&format!(
                "{},S{},{},{},{},{},{},{},{},{},{}\n",
                t,
                k,
                p,
                p + 0.3,
                p - 0.4,
                p + (i as f64 * 0.11).sin() * 0.1,
                v,
                v * p,
                10.0 + (i % 53) as f64,
                v * 0.4,
                v * 0.4 * p
            ));
        }
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, 8_192, Some(0)).unwrap();
    SYMBOLS * PER_SYMBOL
}

struct BudgetReset;

impl Drop for BudgetReset {
    fn drop(&mut self) {
        rcsr::enrich_gpu::set_device_budget_for_tests(0);
    }
}

#[test]
fn resident_tile_outputs_leave_room_for_the_fused_families() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    enrich::set_gpu_floor_for_tests(0);
    let names: Vec<String> = enrich::default_registry()
        .into_iter()
        .filter(|e| e.available && enrich::gpu_supports(&e.name))
        .map(|e| e.name)
        .filter(|n| {
            rcsr::enrich_gpu::tile_feature(n).is_some()
                || rcsr::features_gpu::is_device_flow(n)
                || rcsr::features_gpu::is_device_price_action(n)
        })
        .collect();
    let tiles = names.iter().filter(|n| rcsr::enrich_gpu::tile_feature(n).is_some()).count();
    assert!(tiles > 16, "the budget must be smaller than the tile share");
    let dir = std::env::temp_dir().join(format!("rcsr-budget-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    let out = dir.join("out.rcsb").to_string_lossy().into_owned();
    let rows = write_store(&src);
    // Room for twelve resident columns: more than the eight flow planes, fewer
    // than the tile outputs that would otherwise stay resident.
    let _reset = BudgetReset;
    rcsr::enrich_gpu::set_device_budget_for_tests(12 * rows * 8);
    let report = enrich::build(&Options {
        data: src,
        out: Some(out),
        features: vec![names.join(",")],
        allow_lockbox: true,
        gpu: GpuMode::On,
        ..Default::default()
    })
    .unwrap();
    assert_eq!(report.gpu_fallback, 0, "fallback columns: {:?}", report.gpu_fallback_columns);
    std::fs::remove_dir_all(&dir).ok();
}
