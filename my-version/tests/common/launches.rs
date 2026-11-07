//! #868: successful device launches a child run reports in its `--json` output. A forced device
//! run that reports 0 launches ran on the CPU or failed, so the test that claims the device asserts
//! `> 0`; a test that means to reach a refusal asserts `device_failures` instead.
#![allow(dead_code)]

use rcsr::devlog::Json;

fn count(v: Option<&Json>) -> u64 {
    v.and_then(|v| v.as_f64()).unwrap_or(0.0) as u64
}

fn gpu_block(json: &str) -> Option<Json> {
    Json::parse(json.trim())?.get("gpu").cloned()
}

/// Successful kernel runs: `gpu.batches` minus `gpu.failed_batches` (a failed batch is counted in
/// `batches` too) plus `gpu.fit.calls` (the fit counter rises only for a launch that returned a
/// positive time; a failed fit returns -1 and is not counted). 0 without a `gpu` block.
pub fn device_launches(json: &str) -> u64 {
    let Some(gpu) = gpu_block(json) else {
        return 0;
    };
    let ok_batches = count(gpu.get("batches")).saturating_sub(count(gpu.get("failed_batches")));
    ok_batches + count(gpu.get("fit").and_then(|f| f.get("calls")))
}

/// `gpu.failed_batches`: batches the device refused or failed.
pub fn device_failures(json: &str) -> u64 {
    gpu_block(json).map_or(0, |gpu| count(gpu.get("failed_batches")))
}
