//! #1262: an `rcsr` child must not resolve a bare relative name against the caller's working
//! directory. The test process runs inside a directory that holds files named like the
//! relative arguments of the gate-search command; the child must not read them.
//! One test only: `set_current_dir` is process-wide.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::{data, store};
use scratch::Scratch;
use std::path::Path;

const T0: i64 = 1_600_000_000;
const STEP: i64 = 900;
const STAMPS: usize = 480;
const SYMBOLS: [&str; 3] = ["a", "b", "c"];
const DECOYS: [&str; 8] = ["x1", "cash", "0 - x1", "x1 + 0", "missing.csv", "x.csv", "x", "x.ckpt"];
const JUNK: &str = "decoy: not a run.json\n";

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn write_store(dir: &Path) -> String {
    let mut text = String::from("t,s,x1,x2,g,h,r,y\n");
    for k in 0..STAMPS {
        let g = if (k / 48) % 2 == 0 { 0.5 } else { -0.5 };
        for (j, sym) in SYMBOLS.iter().enumerate() {
            let i = (k + 40) as f64 * 3.0 + j as f64;
            let x1 = frac(i * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(i * 0.754_877_666_246_692_7) - 0.5;
            let h = frac(i * 0.414_213_562_373_095) - 0.5;
            let noise = frac(i * 0.324_717_957_244_746) - 0.5;
            let r = if g > 0.0 { 0.01 } else { -0.01 } * x1 + 0.0005 * noise;
            text.push_str(&format!("{},{},{},{},{},{},{},{}\n", T0 + STEP * k as i64, sym, x1, x2, g, h, r, r));
        }
    }
    let table = data::parse_table(&text).unwrap();
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

#[test]
fn a_child_ignores_decoy_files_in_the_callers_directory() {
    let data_dir = Scratch::new("cwd-data");
    let decoy_dir = Scratch::new("cwd-decoys");
    let store = write_store(data_dir.path());
    let spec = write_spec(data_dir.path());
    for name in DECOYS {
        std::fs::write(decoy_dir.join(name), JUNK).unwrap();
    }
    std::env::set_current_dir(decoy_dir.path()).unwrap();
    let out = rcsr_cmd::rcsr()
        .args(["--data", &store, "--target-spec", &spec, "--gate-search", "--branch-on", "x1", "--branch-off", "cash"])
        .args(["--target", "y", "--vars", "x1,x2", "--time", "t", "--group-by", "s", "--split", "temporal"])
        .args(["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--test-frac", "0.2"])
        .args(["--no-dev-log", "--quiet", "--seed", "3", "--generations", "1", "--pop", "8", "--json"])
        .output()
        .expect("run rcsr");
    let err = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{err}");
    for name in DECOYS {
        assert_eq!(std::fs::read_to_string(decoy_dir.join(name)).unwrap(), JUNK, "{name} was changed");
    }
}
