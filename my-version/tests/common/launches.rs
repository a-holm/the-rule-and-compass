//! #868: device launches a child run reports in its `--json` output. A forced device run that
//! reports 0 launches ran on the CPU, so the test that claims the device asserts `> 0`.
#![allow(dead_code)]

use rcsr::devlog::Json;

/// `gpu.batches` (evaluator batches) plus `gpu.fit.calls` (fit launches). 0 when the report
/// has no `gpu` block or the text is not JSON.
pub fn device_launches(json: &str) -> u64 {
    let Some(doc) = Json::parse(json.trim()) else {
        return 0;
    };
    let Some(gpu) = doc.get("gpu") else {
        return 0;
    };
    let count = |v: Option<&Json>| v.and_then(|v| v.as_f64()).unwrap_or(0.0) as u64;
    count(gpu.get("batches")) + count(gpu.get("fit").and_then(|f| f.get("calls")))
}
