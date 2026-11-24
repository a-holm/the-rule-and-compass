//! #1444: the h1-mini fixture panel is byte-deterministic, the strip list S is pinned, and the hash tool
//! prints hashes only. The controls on the pinned binary are `scripts/h1-mini/` runs, not tests (docs/h1-mini.md).

#[path = "common/h1_mini.rs"]
mod h1_mini;
#[path = "common/scratch.rs"]
mod scratch;

use std::path::{Path, PathBuf};
use std::process::Command;

const STRIP_JQ: &str = "del(.timing, .elapsed_seconds, .gpu, .dispatch, .checkpoint, .tool, .device_probed, .backend_reason, .threads, .process_backend,\n    .baselines.knn.seconds, .fitting.gpu_batches, .preparation.store_memory,\n    .search.cumulative.chunks, .search.cumulative.device_hours, .search.cumulative.elapsed_seconds, .search.cumulative.ledger_wall_s)\n";
const STRIP_CHUNK_JQ: &str = "del(.timing, .elapsed_seconds, .gpu, .dispatch, .checkpoint, .tool, .device_probed, .backend_reason, .threads, .process_backend,\n    .baselines.knn.seconds, .fitting.gpu_batches, .preparation.store_memory,\n    .search.cumulative.chunks, .search.cumulative.device_hours, .search.cumulative.elapsed_seconds, .search.cumulative.ledger_wall_s,\n    .search.books, .search.fit_accept)\n";

fn script_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("scripts").join("h1-mini")
}

fn jq(filter_file: &Path, input: &str) -> String {
    let dir = scratch::Scratch::new("h1-mini-jq");
    let file = dir.join("in.json");
    std::fs::write(&file, input).unwrap();
    let out = Command::new("jq").arg("-S").arg("-c").arg("-f").arg(filter_file).arg(&file).output().expect("run jq");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap().trim().to_string()
}

#[test]
fn h1_mini_panel_is_byte_identical_for_the_same_seed() {
    let a = h1_mini::h1_mini_panel_csv(11);
    assert_eq!(a, h1_mini::h1_mini_panel_csv(11));
    assert_ne!(a, h1_mini::h1_mini_panel_csv(12));
    assert_eq!(a.iter().filter(|&&c| c == b'\n').count(), h1_mini::H1_MINI_SYMBOLS * h1_mini::H1_MINI_BARS + 1);
    assert!(h1_mini::H1_MINI_SYMBOLS >= 8);
    assert_eq!(rcsr::manifest::sha256_hex(&a), PANEL_11_SHA256);
}

const PANEL_11_SHA256: &str = "5cb7c9f18158220c44fa64bdadb1ff742e15b1b117203926def025577082fe4a";

#[test]
fn the_strip_list_is_pinned() {
    let text = std::fs::read_to_string(script_dir().join("strip.jq")).unwrap();
    assert_eq!(text, STRIP_JQ);
}

#[test]
fn the_strip_removes_exactly_the_listed_keys() {
    let input = r#"{
      "timing":{"total":1},"elapsed_seconds":2,"gpu":{"a":1},"dispatch":{"route_ledger":[1]},"checkpoint":{"saved_to":"x"},
      "tool":"t","device_probed":true,"backend_reason":"r","threads":4,"process_backend":"p",
      "baselines":{"knn":{"seconds":1.5,"error":0.25},"other":1},
      "fitting":{"gpu_batches":3,"iters":12},
      "preparation":{"store_memory":9,"rows":100},
      "search":{"cumulative":{"chunks":2,"device_hours":0.1,"elapsed_seconds":3,"ledger_wall_s":4,"generations":30},"islands":4},
      "pareto":[{"expression_exact":"x","error":0.5}],
      "wall_s":7,"search_seconds":8,"rows_per_s":9
    }"#;
    let kept = r#"{"baselines":{"knn":{"error":0.25},"other":1},"fitting":{"iters":12},"pareto":[{"error":0.5,"expression_exact":"x"}],"preparation":{"rows":100},"rows_per_s":9,"search":{"cumulative":{"generations":30},"islands":4},"search_seconds":8,"wall_s":7}"#;
    assert_eq!(jq(&script_dir().join("strip.jq"), input), kept);
}

fn fake_run(dir: &Path, marker: &str) {
    let ckpt = format!(
        "rcsr-checkpoint 1\nbuild commit {marker}\narchive 1 {marker}\narchive_hash {marker}\nfinal 1 4000 {marker}\nsol {marker}\nfsol {marker}\nind {marker}\nrng {marker}\nmix {marker}\nqd_known {marker}\n"
    );
    std::fs::write(dir.join("unit.ckpt"), ckpt).unwrap();
    let run = format!(
        "{{\"timing\":{{\"total\":{marker:?}}},\"objective\":{{\"finalist\":{{\"expression_exact\":\"f{marker}\"}}}},\"pareto\":[{{\"expression_exact\":\"x{marker}\"}}],\"split\":\"s{marker}\"}}"
    );
    std::fs::write(dir.join("run.json"), run).unwrap();
}

fn hash_tool(args: &[&Path]) -> (bool, String) {
    let out = Command::new("bash").arg(script_dir().join("hash.sh")).args(args).output().expect("run hash.sh");
    (out.status.success(), String::from_utf8(out.stdout).unwrap())
}

#[test]
fn the_hash_tool_prints_hashes_and_no_run_value() {
    let dir = scratch::Scratch::new("h1-mini-hash");
    let a = dir.join("a");
    let b = dir.join("b");
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    fake_run(&a, "SECRETVALUE1");
    fake_run(&b, "SECRETVALUE1");
    let (ok, first) = hash_tool(&[&a]);
    assert!(ok);
    let names: Vec<&str> = first.lines().map(|l| l.split(' ').next().unwrap()).collect();
    assert_eq!(
        names,
        ["ckpt_front", "ckpt_ind", "ckpt_rng_mix", "ckpt_qd", "run_pareto", "run_pareto_expressions", "run_finalist", "run_stripped", "run_stripped_chunked"]
    );
    for line in first.lines() {
        let (_, h) = line.split_once(' ').unwrap();
        assert_eq!(h.len(), 64, "{line}");
        assert!(h.bytes().all(|c| c.is_ascii_hexdigit()), "{line}");
    }
    assert!(!first.contains("SECRETVALUE1"));
    assert_eq!(hash_tool(&[&b]).1, first);
    fake_run(&b, "SECRETVALUE2");
    let (_, second) = hash_tool(&[&b]);
    for (x, y) in first.lines().zip(second.lines()) {
        assert_ne!(x, y);
    }
}

#[test]
fn the_keys_mode_names_differing_paths_and_no_value() {
    let dir = scratch::Scratch::new("h1-mini-keys");
    let a = dir.join("a");
    let b = dir.join("b");
    std::fs::create_dir(&a).unwrap();
    std::fs::create_dir(&b).unwrap();
    fake_run(&a, "SECRETVALUE1");
    fake_run(&b, "SECRETVALUE2");
    let (ok, out) = hash_tool(&[Path::new("keys"), &a, &b]);
    assert!(ok);
    assert_eq!(out.lines().collect::<Vec<_>>(), ["objective.finalist.expression_exact", "pareto.0.expression_exact", "split"]);
    assert!(!out.contains("SECRETVALUE"));
}

#[test]
fn the_chunked_strip_adds_the_per_process_search_counters() {
    let text = std::fs::read_to_string(script_dir().join("strip-chunk.jq")).unwrap();
    assert_eq!(text, STRIP_CHUNK_JQ);
    let input = r#"{"search":{"books":{"signed":1},"fit_accept":{"accepted":2},"cumulative":{"chunks":3,"generations":30},"islands":4},"pareto":[]}"#;
    assert_eq!(jq(&script_dir().join("strip-chunk.jq"), input), r#"{"pareto":[],"search":{"cumulative":{"generations":30},"islands":4}}"#);
}
