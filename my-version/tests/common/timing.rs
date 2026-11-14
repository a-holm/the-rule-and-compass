//! Report comparison that drops measured time by key, on parsed JSON.
#![allow(dead_code)]

use rcsr::devlog::Json;

/// Members whose value is wall-clock content or derived from it. `ledger` and
/// `batch_timings` are dropped whole, so a new ledger field never reaches a comparison.
pub fn is_timing_key(key: &str) -> bool {
    matches!(
        key,
        "ledger" | "batch_timings" | "timing" | "closure" | "numerator" | "device_fraction" | "duty_run" | "declared_in" | "bytes_per_s"
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

/// `value` without its timing members, at any depth.
pub fn drop_timing(value: Json) -> Json {
    match value {
        Json::Obj(items) => Json::Obj(
            items
                .into_iter()
                .filter(|(key, _)| !is_timing_key(key))
                .map(|(key, v)| (key, drop_timing(v)))
                .collect(),
        ),
        Json::Arr(items) => Json::Arr(items.into_iter().map(drop_timing).collect()),
        other => other,
    }
}
