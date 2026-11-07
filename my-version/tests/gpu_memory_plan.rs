//! #293: the device memory plan, its named refusals and the allocation label.
//!
//! Device tests skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/launches.rs"]
mod launches;

use std::path::PathBuf;
use std::process::Output;

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;

fn has_device() -> bool {
    let present = rcsr::gpu::probe().available();
    if !present {
        rcsr::backend::skip_without_device("skipping: no usable CUDA device");
    }
    present
}

fn tmp_csv(rows: usize) -> (Scratch, PathBuf) {
    let dir = Scratch::new("sine");
    let path = dir.join("sine.csv");
    let mut text = String::from("x,y\n");
    for i in 0..rows {
        let x = i as f64 * 0.001;
        text.push_str(&format!("{:.6},{:.9}\n", x, x.sin()));
    }
    std::fs::write(&path, text).unwrap();
    (dir, path)
}

fn rcsr(csv: &PathBuf, extra: &[&str]) -> Output {
    // Every test of this file needs the device, with and without `--gpu`.
    rcsr_cmd::rcsr_on_device()
        .args(["--data", csv.to_str().unwrap(), "--target", "y", "--vars", "x"])
        .args(["--generations", "2", "--seed", "1"])
        .args(extra)
        .output()
        .expect("run rcsr")
}

fn number_after(text: &str, key: &str) -> u64 {
    let at = text.find(key).unwrap_or_else(|| panic!("no {key} in {text}"));
    text[at + key.len()..]
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap()
}

#[test]
fn a_tiny_budget_under_gpu_exits_2_with_the_plan() {
    if !has_device() {
        return;
    }
    let (_dir, csv) = tmp_csv(2000);
    let out = rcsr(&csv, &["--gpu", "--gpu-memory", "1KiB"]);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("GPU memory plan refused"), "{stderr}");
    assert!(stderr.contains("budget 1.00 KiB (1,024 bytes)"), "{stderr}");
    assert!(stderr.contains("smallest fix"), "{stderr}");
    let need = stderr.split("need ").nth(1).expect("need line");
    let bytes = need
        .split('(')
        .nth(1)
        .and_then(|s| s.split(" bytes").next())
        .expect("byte count")
        .replace(',', "");
    assert!(bytes.parse::<u64>().unwrap() > 1024, "{stderr}");
}

#[test]
fn a_tiny_budget_without_gpu_keeps_the_cpu_and_names_the_cause() {
    if !has_device() {
        return;
    }
    let (_dir, csv) = tmp_csv(2000);
    let out = rcsr(&csv, &["--gpu-memory", "1KiB", "--json", "--quiet"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(stdout.contains("\"backend\":\"cpu\""), "{stdout}");
    assert!(stdout.contains("\"backend_reason\":\"device_memory"), "{stdout}");
}

#[test]
fn an_explicit_device_fit_that_does_not_fit_exits_2_at_the_fit_stage() {
    if !has_device() {
        return;
    }
    let rows = 100_000usize;
    let (_dir, csv) = tmp_csv(rows);
    let train = rows * 3 / 4;
    let budget = (train * 2 * 4 + (2 << 20)).to_string();
    let out = rcsr(
        &csv,
        &["--gpu", "--dispatch", "device", "--fit-strategy", "gpu", "--gpu-memory", budget.as_str()],
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("refused the fit stage"), "{stderr}");
    assert!(stderr.contains("--fit-strategy exact"), "{stderr}");
}

#[test]
fn a_normal_gpu_run_reports_a_resident_plan() {
    if !has_device() {
        return;
    }
    let (_dir, csv) = tmp_csv(5000);
    let out = rcsr(&csv, &["--gpu", "--dispatch", "device", "--json", "--quiet"]);
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let launches = launches::device_launches(&stdout);
    eprintln!("forced device run: {launches} device launches counted");
    assert!(launches > 0, "a forced device run launched no kernel: {stdout}");
    assert!(stdout.contains("\"memory_plan\":{\"mode\":\"resident\""), "{stdout}");
    let plan = &stdout[stdout.find("\"memory_plan\"").unwrap()..];
    let required = number_after(plan, "\"required_bytes\":");
    let budget = number_after(plan, "\"budget_bytes\":");
    assert!(required > 0 && required <= budget, "{required} {budget}");
    let l2 = number_after(plan, "\"l2_bytes\":");
    assert!(l2 > 0);
    assert_eq!(l2, rcsr::gpu::device_info().unwrap().l2_bytes);
    assert!(plan.contains("\"refusal\":null"), "{plan}");
}

#[test]
fn a_refused_device_allocation_is_reported_with_its_bytes() {
    if !has_device() || rcsr::gpu::GpuEvaluator::unified_memory() {
        return;
    }
    let info = rcsr::gpu::device_info().expect("device info");
    let rows = 1usize << 30;
    let vars = (info.total / (4 * rows as u64)) as usize + 2;
    let cols = vec![0.0f32; 4];
    let y = vec![0.0f32; 4];
    assert!(rcsr::gpu::GpuEvaluator::new(&cols, rows, vars, &y).is_none());
    assert_eq!(rcsr::gpu::last_alloc_failure(), Some(rows as u64 * vars as u64 * 4));
}
