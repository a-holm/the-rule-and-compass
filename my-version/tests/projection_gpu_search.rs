//! #293 AC5: a 364-column store searched on the GPU with 20 variables stays
//! resident, reads only the projected columns and stays inside the host and
//! device caps.
//!
//! The device tests skip cleanly when no CUDA device is present. The `ignored`
//! tests are the full-scale record: `build_wide_364_store` writes the
//! 4,775,757-row store (the search then runs through the CLI, see
//! docs/memory-model.md), `resident_throughput_record` times resident scoring.

#![cfg(feature = "cuda")]

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

use rcsr::data;
use rcsr::devlog::Json;
use rcsr::expr::{constant, input, node, Node, Op};
use rcsr::genome::{GenomeMeta, GenomePool};
use rcsr::gpu::{compile_program, GpuEvaluator};
use rcsr::metrics::ScoreObjective;
use rcsr::store::{self, AppendColumn};

const GIB: u64 = 1 << 30;
const VARS: usize = 20;
const TOTAL_COLUMNS: usize = 364;
const GROUP_ROWS: usize = 65_536;
const FULL_ROWS: usize = 4_775_757;
const FULL_STORE: &str = "/mnt/lager/tmp/opencode/293-wide364.rcsb";

fn has_device() -> bool {
    rcsr::gpu::probe().available()
}

fn tmp_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-projection-gpu-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Writes `ts, symbol, y, x0..x19` as CSV, the 23-column base of the wide store.
fn write_base_csv(path: &Path, rows: usize, symbols: usize) {
    let file = std::fs::File::create(path).unwrap();
    let mut out = std::io::BufWriter::with_capacity(1 << 20, file);
    let mut head = String::from("ts,symbol,y");
    for c in 0..VARS {
        head.push_str(&format!(",x{c}"));
    }
    writeln!(out, "{head}").unwrap();
    let mut rng = Lcg(0x293_364);
    let mut x = [0.0f64; VARS];
    for r in 0..rows {
        let sym = r % symbols;
        let ts = 1_600_000_000i64 + (r / symbols) as i64 * 60;
        for v in x.iter_mut() {
            *v = rng.unit() * 4.0 - 2.0;
        }
        let y = 0.7 * x[0] + 0.3 * (2.0 * x[1]).sin() + 0.2 * (rng.unit() - 0.5);
        let mut line = format!("{ts},S{sym},{y:.6}");
        for v in &x {
            line.push_str(&format!(",{v:.6}"));
        }
        writeln!(out, "{line}").unwrap();
    }
    out.flush().unwrap();
}

/// Writes `files` filler column files of `rows` little-endian f64 values.
fn write_fillers(dir: &Path, rows: usize, files: usize) -> Vec<PathBuf> {
    (0..files)
        .map(|k| {
            let path = dir.join(format!("filler{k}.f64"));
            let mut out = std::io::BufWriter::with_capacity(
                1 << 20,
                std::fs::File::create(&path).unwrap(),
            );
            let mut rng = Lcg(0xf111 + k as u64);
            for _ in 0..rows {
                out.write_all(&(rng.unit() * 2.0 - 1.0).to_le_bytes()).unwrap();
            }
            out.flush().unwrap();
            path
        })
        .collect()
}

/// A `TOTAL_COLUMNS`-column store: the 23-column base, then x20..x360 appended
/// in one pass from a few shared filler column files.
fn build_wide_store(path: &str, rows: usize, symbols: usize, scratch: &Path) {
    let csv = scratch.join("base.csv");
    write_base_csv(&csv, rows, symbols);
    let table = data::read_table_with_time(csv.to_str().unwrap(), false, Some("ts")).unwrap();
    assert_eq!(table.len(), rows);
    let base = scratch.join("base.rcsb");
    store::write_store_with_time(base.to_str().unwrap(), &table, GROUP_ROWS, Some(0)).unwrap();
    drop(table);
    std::fs::remove_file(&csv).unwrap();

    let fillers = write_fillers(scratch, rows, 4);
    let extras: Vec<AppendColumn> = (VARS..TOTAL_COLUMNS - 3)
        .map(|c| AppendColumn {
            name: format!("x{c}"),
            file: fillers[c % fillers.len()].clone(),
            replace: false,
        })
        .collect();
    let source = store::FileStore::open(base.to_str().unwrap()).unwrap();
    store::write_store_appending(path, &source, &extras).unwrap();
    drop(source);
    std::fs::remove_file(&base).unwrap();
    for f in fillers {
        std::fs::remove_file(f).unwrap();
    }
}

struct Run {
    code: Option<i32>,
    stdout: String,
    stderr: String,
    manifest: Json,
    peak_rss_bytes: u64,
}

fn search(store_path: &str, dir: &Path, extra: &[&str]) -> Run {
    let vars: Vec<String> = (0..VARS).map(|c| format!("x{c}")).collect();
    let manifest = dir.join("manifest.json");
    let devlog = dir.join("dev.log");
    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["--data", store_path, "--target", "y", "--vars", &vars.join(",")])
        .args(["--time", "ts", "--group-by", "symbol"])
        .args(["--gpu", "--fit-strategy", "gpu", "--seed", "7"])
        .args(["--json", "--quiet"])
        .args(["--run-manifest", manifest.to_str().unwrap()])
        .args(["--dev-log", devlog.to_str().unwrap()])
        .args(extra)
        .output()
        .expect("run rcsr");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let manifest_text = std::fs::read_to_string(&manifest).unwrap_or_default();
    let log = std::fs::read_to_string(&devlog).unwrap_or_default();
    let end = log
        .lines()
        .filter_map(Json::parse)
        .filter(|j| j.get("event").and_then(Json::as_str) == Some("run_end"))
        .last();
    let peak_rss_bytes = end
        .as_ref()
        .and_then(|e| e.get("peak_rss_bytes"))
        .and_then(Json::as_f64)
        .unwrap_or(0.0) as u64;
    Run {
        code: out.status.code(),
        stdout,
        stderr,
        manifest: Json::parse(&manifest_text).unwrap_or(Json::Null),
        peak_rss_bytes,
    }
}

fn field<'a>(json: &'a Json, path: &[&str]) -> Option<&'a Json> {
    path.iter().try_fold(json, |j, k| j.get(k))
}

fn projected_columns(run: &Run) -> Vec<String> {
    match field(&run.manifest, &["projection", "columns"]) {
        Some(Json::Arr(items)) => items
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect(),
        other => panic!("no projection.columns in the manifest: {other:?}"),
    }
}

struct Plan {
    mode: String,
    peak_live_bytes: u64,
    required_bytes: u64,
}

fn plan(run: &Run) -> Plan {
    let out = Json::parse(&run.stdout).unwrap_or_else(|| panic!("stdout is not JSON: {}", run.stdout));
    let p = field(&out, &["gpu", "memory_plan"])
        .unwrap_or_else(|| panic!("no gpu.memory_plan: {}", run.stdout));
    Plan {
        mode: p.get("mode").and_then(Json::as_str).unwrap().to_string(),
        peak_live_bytes: field(p, &["observed", "peak_live_bytes"])
            .and_then(Json::as_f64)
            .expect("observed.peak_live_bytes") as u64,
        required_bytes: p.get("required_bytes").and_then(Json::as_f64).unwrap() as u64,
    }
}

fn assert_ac5(run: &Run) -> Plan {
    assert_eq!(run.code, Some(0), "{}", run.stderr);
    let cols = projected_columns(run);
    assert_eq!(cols.len(), 23, "ts + symbol + y + 20 vars: {cols:?}");
    for c in 0..VARS {
        assert!(cols.contains(&format!("x{c}")), "missing x{c}");
    }
    assert!(!cols.contains(&"x20".to_string()), "unrequested column read");
    let p = plan(run);
    assert_eq!(p.mode, "resident");
    assert!(run.peak_rss_bytes > 0, "no peak_rss_bytes in the dev log");
    assert!(
        run.peak_rss_bytes < 8 * GIB,
        "peak RSS {} B over 8 GiB",
        run.peak_rss_bytes
    );
    assert!(
        p.peak_live_bytes < 21 * GIB / 2,
        "device use {} B over 10.5 GiB",
        p.peak_live_bytes
    );
    p
}

#[test]
fn a_364_column_store_with_20_vars_searches_resident_on_the_gpu() {
    if !has_device() {
        return;
    }
    let dir = tmp_dir("small");
    let store_path = dir.join("wide364.rcsb");
    build_wide_store(store_path.to_str().unwrap(), 65_536, 8, &dir);
    let run = search(
        store_path.to_str().unwrap(),
        &dir,
        &["--generations", "3", "--pop", "64"],
    );
    let p = assert_ac5(&run);
    eprintln!(
        "ac5 small: peak_rss {} B, device peak {} B, required {} B",
        run.peak_rss_bytes, p.peak_live_bytes, p.required_bytes
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// One append pass: writes about 14 GB to `FULL_STORE`.
#[test]
#[ignore]
fn build_wide_364_store() {
    let scratch = tmp_dir("full-build");
    let started = std::time::Instant::now();
    build_wide_store(FULL_STORE, FULL_ROWS, 73, &scratch);
    let header = store::FileStore::open(FULL_STORE).unwrap();
    eprintln!(
        "built {FULL_STORE}: {} bytes in {:.1} s",
        std::fs::metadata(FULL_STORE).unwrap().len(),
        started.elapsed().as_secs_f64()
    );
    drop(header);
    let _ = std::fs::remove_dir_all(&scratch);
}

fn pool_of(trees: &[Node]) -> GenomePool {
    let programs: Vec<_> = trees
        .iter()
        .map(|t| compile_program(t).expect("a native program"))
        .collect();
    let width = programs.iter().map(|p| p.words().len()).max().unwrap();
    let mut pool = GenomePool::new(trees.len(), width);
    pool.begin_generation();
    for p in &programs {
        assert!(pool.push(p.words(), GenomeMeta::default()));
    }
    pool
}

/// Resident scoring of P candidates over 4,775,757 x 20 f32 rows.
#[test]
#[ignore]
fn resident_throughput_record() {
    if !has_device() {
        return;
    }
    let n = FULL_ROWS;
    let mut rng = Lcg(0x71ed);
    let mut cols = vec![0.0f32; n * VARS];
    for c in cols.iter_mut() {
        *c = (rng.unit() * 4.0 - 2.0) as f32;
    }
    let y: Vec<f32> = (0..n)
        .map(|i| 0.7 * cols[i] + 0.3 * cols[n + i].sin() + 0.1 * cols[2 * n + i])
        .collect();
    let mut ev = GpuEvaluator::new(&cols, n, VARS, &y).expect("resident evaluator");
    drop(cols);
    for (p, target_s) in [(400usize, 0.25f64), (4_096, 2.0)] {
        let trees: Vec<Node> = (0..p)
            .map(|i| {
                node(
                    Op::Add,
                    vec![
                        node(Op::Mul, vec![constant(0.001 * (i + 1) as f64), input(i % VARS)]),
                        node(Op::Sin, vec![input((i * 7 + 3) % VARS)]),
                    ],
                )
            })
            .collect();
        let pool = pool_of(&trees);
        let pcount = vec![2i32; p];
        let _ = ev.score_pool(&pool, ScoreObjective::Squared, &pcount);
        let mut times = Vec::new();
        for _ in 0..7 {
            let t = std::time::Instant::now();
            let _ = ev.score_pool(&pool, ScoreObjective::Squared, &pcount);
            times.push(t.elapsed().as_secs_f64());
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        eprintln!(
            "throughput {n} x {VARS}, P = {p}: median {:.4} s, min {:.4} s, target {:.2} s",
            times[3], times[0], target_s
        );
    }
}
