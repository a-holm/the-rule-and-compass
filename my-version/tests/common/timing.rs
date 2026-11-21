//! Report comparison that drops measured time by key, on parsed JSON.
#![allow(dead_code)]

use rcsr::devlog::Json;

/// Members whose value is wall-clock content or derived from it. `ledger` and
/// `batch_timings` are dropped whole, so a new ledger field never reaches a comparison. `scan`
/// counts the reads of one store scan and differs between stores of the same rows.
pub fn is_timing_key(key: &str) -> bool {
    matches!(
        key,
        "ledger" | "scan" | "batch_timings" | "timing" | "closure" | "numerator" | "device_fraction" | "duty_run" | "declared_in" | "bytes_per_s"
            | "rows_per_s" | "report_sha256" | "total_seconds" | "prep_seconds" | "search_seconds" | "baselines_seconds"
            | "diagnostics_seconds" | "report_seconds" | "gates_seconds" | "gpu_seconds" | "unaccounted_seconds"
            | "fit_fallback_seconds" | "fallback_seconds" | "nonfinite_seconds" | "phases_seconds" | "host_seconds"
            | "gate_seconds" | "device_seconds_incl_fit"
    ) || key.contains("wall")
        || key.contains("elapsed")
        || key.contains("denominator")
        || key.ends_with("_s")
        || key.ends_with("_ms")
}

/// `seconds` is wall-clock in the `gpu.fit` and `prune` objects and data elsewhere (`lockbox_embargo.seconds`).
fn drops_seconds(path: &[String]) -> bool {
    match path {
        [.., last] if last == "prune" => true,
        [.., gpu, fit] => gpu == "gpu" && fit == "fit",
        _ => false,
    }
}

/// `value` without its timing members, at any depth.
pub fn drop_timing(value: Json) -> Json {
    drop_timing_at(value, &[])
}

/// `path` is the key chain of the objects above `value`; an array adds no element.
fn drop_timing_at(value: Json, path: &[String]) -> Json {
    match value {
        Json::Obj(items) => Json::Obj(
            items
                .into_iter()
                .filter(|(key, _)| !is_timing_key(key) && !(key == "seconds" && drops_seconds(path)))
                .map(|(key, v)| {
                    let mut below = path.to_vec();
                    below.push(key.clone());
                    (key, drop_timing_at(v, &below))
                })
                .collect(),
        ),
        Json::Arr(items) => Json::Arr(items.into_iter().map(|v| drop_timing_at(v, path)).collect()),
        other => other,
    }
}
