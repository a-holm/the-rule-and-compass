//! #1247: `--sample-rows` with `--pnl-mask`, `rcsr compose` or `rcsr switch` exits 2 before any row is read.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::{data, store};
use scratch::Scratch;
use std::path::{Path, PathBuf};
use std::process::Output;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;
const STAMPS: usize = 480;
const SYMBOLS: [&str; 3] = ["a", "b", "c"];

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn regime(k: usize) -> f64 {
    let m = 0.5 + 0.4 * frac(k as f64 * 0.324_717_957_244_746);
    if (k / 48) % 2 == 0 {
        m
    } else {
        -m
    }
}

fn write_csv(dir: &Path) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y\n");
    for k in 0..STAMPS {
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x1 = frac(i * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let g = regime(k);
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1 + 0.0005 * noise;
            text.push_str(&format!("{},{},{},{},{},{},{},{}\n", T0 + STEP * k as i64, sym, x1, x2, g, h, r, r));
        }
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_store(dir: &Path, csv: &str) -> String {
    let table = data::parse_table(&std::fs::read_to_string(csv).unwrap()).unwrap();
    let path = dir.join("d.rcsb");
    store::write_store(path.to_str().unwrap(), &table, 251).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"stride_seconds\":900,\"entry_offset_seconds\":0,\"delay_seconds\":0,\
         \"entry_price\":\"open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("s.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_menu(dir: &Path) -> String {
    let path = dir.join("menu.json");
    std::fs::write(
        &path,
        "{\"schema\":\"rcsr-gate-menu/1\",\"switch_budget\":null,\"gates\":[{\"name\":\"g\",\"codeword\":\"000\",\"expr\":\"g\",\"band\":0,\"dwell\":1}]}",
    )
    .unwrap();
    path.to_string_lossy().into_owned()
}

const RUN_FLAGS: [&str; 20] = [
    "--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--split", "temporal", "--objective", "pnl", "--cost-bps", "5",
    "--validation-frac", "0.3", "--test-frac", "0.2", "--no-dev-log", "--quiet",
];

fn run(args: &[&str]) -> Output {
    rcsr_cmd::rcsr().args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

struct Fixture {
    _scratch: Scratch,
    dir: PathBuf,
    store: String,
    spec: String,
    menu: String,
    pool: String,
}

fn fixture() -> Fixture {
    let scratch = Scratch::new("gate-sample");
    let dir = scratch.path().to_path_buf();
    let csv = write_csv(&dir);
    let store = write_store(&dir, &csv);
    let spec = write_spec(&dir);
    let menu = write_menu(&dir);
    let mut args = vec!["--data", &csv, "--target-spec", &spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--pnl-mask", "g", "--cpu", "--seed", "3", "--generations", "2", "--pop", "16", "--json"]);
    let o = run(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let body = String::from_utf8_lossy(&o.stdout).trim().to_string();
    let cut = body.rfind(",\"pareto\":[").expect("run.json ends with its pareto");
    let member = "{\"archive_rank\":1,\"expression\":\"x1\",\"expression_exact\":\"x1\",\"complexity\":2}";
    let pool = dir.join("bull.json");
    std::fs::write(&pool, format!("{},\"pareto\":[{}]}}\n", &body[..cut], member)).unwrap();
    let pool = pool.to_string_lossy().into_owned();
    Fixture { _scratch: scratch, dir, store, spec, menu, pool }
}

fn masked(f: &Fixture, extra: &[&str]) -> Output {
    let mut args = vec!["--data", &f.store, "--target-spec", &f.spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--pnl-mask", "g", "--cpu", "--seed", "3", "--generations", "1", "--pop", "8", "--json"]);
    args.extend_from_slice(extra);
    run(&args)
}

fn compose(f: &Fixture, extra: &[&str]) -> Output {
    let mut args = vec!["compose", "--data", &f.store, "--target-spec", &f.spec, "--gates", &f.menu, "--bull", &f.pool, "--json"];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend_from_slice(extra);
    run(&args)
}

fn switch(f: &Fixture, extra: &[&str]) -> Output {
    let mut args = vec![
        "switch", "--data", &f.store, "--target-spec", &f.spec, "--a", "x1", "--b", "0 - x1", "--gate", "g", "--json",
    ];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend_from_slice(extra);
    run(&args)
}

fn assert_refused(o: &Output, flags: &[&str], dir: &Path) {
    let err = stderr(o);
    assert_eq!(code(o), 2, "{err}");
    for flag in flags {
        assert!(err.contains(flag), "{flag}: {err}");
    }
    assert!(!err.contains("sampling (explicit)"), "{err}");
    assert!(String::from_utf8_lossy(&o.stdout).is_empty());
    let mut written: Vec<String> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    written.sort();
    assert_eq!(written, ["bull.json", "d.csv", "d.rcsb", "menu.json", "s.json"], "the refusal writes nothing");
}

#[test]
fn sample_rows_refused_with_gate_commands() {
    let f = fixture();
    assert_refused(&masked(&f, &["--sample-rows", "1000"]), &["--sample-rows", "--pnl-mask"], &f.dir);
    assert_refused(&compose(&f, &["--sample-rows", "1000"]), &["--sample-rows", "rcsr compose"], &f.dir);
    assert_refused(&switch(&f, &["--sample-rows", "1000"]), &["--sample-rows", "rcsr switch"], &f.dir);
    // n at or above the row count is a full load; the flag is what is refused.
    assert_refused(&masked(&f, &["--sample-rows", "999999"]), &["--sample-rows", "--pnl-mask"], &f.dir);
}

#[test]
fn gate_commands_without_sample_rows_still_run() {
    let f = fixture();
    for o in [masked(&f, &[]), compose(&f, &[]), switch(&f, &[])] {
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
}

#[test]
fn sample_rows_without_a_gate_still_runs() {
    let f = fixture();
    let mut args = vec!["--data", &f.store, "--target-spec", &f.spec];
    args.extend_from_slice(&RUN_FLAGS);
    args.extend(["--fitness", "pnl", "--cpu", "--seed", "3", "--generations", "1", "--pop", "8", "--sample-rows", "1000", "--json"]);
    let o = run(&args);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}
