//! #962: a chunked search resumed at its final generation reports the search backend.
//! Every child runs on the CPU; a GPU chunk is stood in for by a sidecar that records `gpu`.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const G: &str = "8";

fn setup(tag: &str) -> (Scratch, PathBuf, String) {
    let scratch = Scratch::new(tag);
    let mut text = String::from("x,y\n");
    for i in 0..60 {
        let x = -2.0 + 4.0 * i as f64 / 59.0;
        text.push_str(&format!("{:.6},{:.6}\n", x, 2.0 * x + 1.0));
    }
    let csv = scratch.join("d.csv");
    std::fs::write(&csv, text).unwrap();
    let dir = scratch.join("run");
    std::fs::create_dir(&dir).unwrap();
    (scratch, dir, csv.to_str().unwrap().to_string())
}

/// One process of a chain: `interim` runs one generation and stops on the time budget.
fn chunk(dir: &Path, csv: &str, resume: bool, interim: bool) -> Output {
    let mut args = vec![
        "--data", csv, "--target", "y", "--vars", "x", "--seed", "5", "--pop", "30", "--max-complexity", "8", "--cpu",
        "--no-dev-log", "--quiet", "--generations", G, "--save", "ck.rcsr", "--json-out", "run.json",
    ];
    if interim {
        args.extend(["--time-budget", "0", "--interim-chunk"]);
    }
    if resume {
        args.extend(["--resume", "ck.rcsr"]);
        if !interim {
            args.extend(["--time-budget", "100000"]);
        }
    }
    let out = rcsr_cmd::rcsr().current_dir(dir).args(&args).output().expect("run rcsr");
    assert!(out.status.success(), "{:?}", out);
    out
}

fn run_json(dir: &Path) -> Json {
    Json::parse(std::fs::read_to_string(dir.join("run.json")).unwrap().trim()).expect("run.json")
}

fn text(j: &Json, key: &str) -> String {
    j.get(key).and_then(Json::as_str).unwrap_or_else(|| panic!("run.json has no string {key}")).to_string()
}

/// `(backend, search_backend, process_backend, search.cumulative.backend)`.
fn backends(dir: &Path) -> (String, String, String, String) {
    let j = run_json(dir);
    let cumulative = j.get("search").and_then(|s| s.get("cumulative")).and_then(|c| c.get("backend")).and_then(Json::as_str);
    (text(&j, "backend"), text(&j, "search_backend"), text(&j, "process_backend"), cumulative.unwrap_or("absent").to_string())
}

fn sidecar(dir: &Path) -> PathBuf {
    dir.join("ck.rcsr.cost")
}

fn set_sidecar_backend(dir: &Path, backend: &str) {
    let text = std::fs::read_to_string(sidecar(dir)).unwrap();
    assert!(text.contains("\"backend\":\"cpu\""), "{text}");
    std::fs::write(sidecar(dir), text.replace("\"backend\":\"cpu\"", &format!("\"backend\":\"{backend}\""))).unwrap();
}

/// Two interim chunks, then a final chunk that ends at generation G.
fn finished_chain(dir: &Path, csv: &str) {
    chunk(dir, csv, false, true);
    chunk(dir, csv, true, true);
    chunk(dir, csv, true, false);
}

fn strs(v: [&str; 4]) -> (String, String, String, String) {
    (v[0].to_string(), v[1].to_string(), v[2].to_string(), v[3].to_string())
}

#[test]
fn resume_at_final_generation_reports_search_backend() {
    let (_s, dir, csv) = setup("final-generation");
    finished_chain(&dir, &csv);
    assert_eq!(backends(&dir), strs(["cpu", "cpu", "cpu", "cpu"]));
    assert_eq!(run_json(&dir).get("generations_run").and_then(Json::as_usize), Some(8));
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["cpu", "cpu", "cpu", "cpu"]));

    set_sidecar_backend(&dir, "gpu");
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["gpu", "gpu", "cpu", "gpu"]));
    let reason = text(&run_json(&dir), "backend_reason");
    assert!(reason.contains("report-only resume"), "{reason}");
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["gpu", "gpu", "cpu", "gpu"]), "a report-only step does not turn gpu into mixed");
}

#[test]
fn fresh_run_backend_fields_unchanged() {
    let (_s, dir, csv) = setup("fresh");
    chunk(&dir, &csv, false, false);
    assert_eq!(backends(&dir), strs(["cpu", "cpu", "cpu", "cpu"]));
    let j = run_json(&dir);
    assert_eq!(text(&j, "backend_reason"), "--cpu/--no-gpu");
}

#[test]
fn resume_with_generations_left_keeps_process_backend() {
    let (_s, dir, csv) = setup("generations-left");
    chunk(&dir, &csv, false, true);
    set_sidecar_backend(&dir, "gpu");
    chunk(&dir, &csv, true, false);
    let (backend, search, process, cumulative) = backends(&dir);
    assert_eq!((backend.as_str(), process.as_str()), ("cpu", "cpu"));
    assert_eq!((search.as_str(), cumulative.as_str()), ("mixed", "mixed"));
    assert!(run_json(&dir).get("generations_run").and_then(Json::as_usize) > Some(1));
}

#[test]
fn missing_or_stale_sidecar_gives_unknown_search_backend() {
    let (_s, dir, csv) = setup("unknown");
    finished_chain(&dir, &csv);
    let current = std::fs::read_to_string(sidecar(&dir)).unwrap();

    std::fs::remove_file(sidecar(&dir)).unwrap();
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["cpu", "unknown", "cpu", "absent"]));
    assert!(text(&run_json(&dir), "backend_reason").contains("search backend unknown"));

    std::fs::write(sidecar(&dir), current.replace("\"generation\":8", "\"generation\":3")).unwrap();
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["cpu", "unknown", "cpu", "absent"]));

    let v1 = current.replace("\"v\":2", "\"v\":1").replace(",\"backend\":\"cpu\"", "");
    assert!(!v1.contains("backend"), "{v1}");
    std::fs::write(sidecar(&dir), v1).unwrap();
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["cpu", "unknown", "cpu", "unknown"]));
}

#[test]
fn gpu_then_cpu_chunks_report_mixed() {
    let (_s, dir, csv) = setup("mixed");
    chunk(&dir, &csv, false, true);
    set_sidecar_backend(&dir, "gpu");
    chunk(&dir, &csv, true, true);
    assert_eq!(backends(&dir), strs(["cpu", "mixed", "cpu", "mixed"]));
    chunk(&dir, &csv, true, false);
    chunk(&dir, &csv, true, false);
    assert_eq!(backends(&dir), strs(["mixed", "mixed", "cpu", "mixed"]));
}
