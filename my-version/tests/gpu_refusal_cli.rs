//! #744: a refused kernel launch reaches the fail-closed path of the CLI.
//!
//! `RCSR_FORCE_LAUNCH_REFUSAL=<kernel prefix>` makes the launch guard refuse
//! every launch of the matching kernels. It is set on child processes only.

#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const ROWS: usize = 50_000;
const KERNEL: &str = "score_kernel";

struct Work {
    scratch: Scratch,
    data: PathBuf,
}

impl Work {
    fn new(tag: &str) -> Work {
        let scratch = Scratch::new(tag);
        let mut text = String::from("x1,x2,x3,y\n");
        let mut s = 12345u64;
        let mut next = move || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 33) as f64 / (1u64 << 31) as f64) * 4.0 - 2.0
        };
        for _ in 0..ROWS {
            let (a, b, c) = (next(), next(), next());
            text.push_str(&format!("{a:.6},{b:.6},{c:.6},{:.6}\n", 2.0 * a + b * b - 0.5 * c * a));
        }
        let data = scratch.join("data.csv");
        std::fs::write(&data, text).unwrap();
        Work { scratch, data }
    }

    fn json(&self, name: &str) -> PathBuf {
        self.scratch.join(format!("{name}.json"))
    }

    fn log(&self, name: &str) -> PathBuf {
        self.scratch.join(format!("{name}.log"))
    }

    fn run(&self, name: &str, args: &[&str], force: Option<&str>) -> Output {
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr"));
        cmd.arg("--data").arg(&self.data);
        cmd.args(["--target", "y", "--generations", "2", "--pop", "60", "--seed", "3"]);
        cmd.args(["--threads", "1", "--quiet"]);
        cmd.args(args);
        cmd.arg("--json-out").arg(self.json(name));
        cmd.arg("--dev-log").arg(self.log(name));
        cmd.env_remove("RCSR_FORCE_LAUNCH_REFUSAL");
        cmd.env_remove("RCSR_DEVICE_PTR_CHECK");
        if let Some(k) = force {
            cmd.env("RCSR_FORCE_LAUNCH_REFUSAL", k);
        }
        cmd.output().expect("run rcsr")
    }
}

fn text(b: &[u8]) -> String {
    String::from_utf8_lossy(b).to_string()
}

/// Archive hash of the `run_end` event in a development log.
fn archive_hash(log: &Path) -> String {
    let text = std::fs::read_to_string(log).expect("dev log");
    let end = text.lines().find(|l| l.contains("\"event\":\"run_end\"")).expect("run_end event");
    let at = end.find("\"hash\":\"").expect("archive hash") + 8;
    end[at..].split('"').next().unwrap().to_string()
}

fn gpu_available() -> bool {
    if rcsr::gpu::probe().available() {
        true
    } else {
        eprintln!("skipping GPU test: no usable CUDA device");
        false
    }
}

#[test]
fn explicit_gpu_run_fails_closed_when_a_launch_is_refused() {
    if !gpu_available() {
        return;
    }
    let w = Work::new("explicit");
    let ok = w.run("control", &["--gpu"], None);
    assert_eq!(ok.status.code(), Some(0), "{}", text(&ok.stderr));
    assert!(std::fs::read_to_string(w.json("control")).unwrap().contains("\"backend\":\"gpu\""));

    let out = w.run("forced", &["--gpu"], Some(KERNEL));
    let err = text(&out.stderr);
    assert_eq!(out.status.code(), Some(6), "{err}");
    assert!(err.contains(&format!("refused launch of {KERNEL}")), "{err}");
    assert!(err.contains("--gpu was requested but the device search failed"), "{err}");
    assert!(!w.json("forced").exists(), "a failed --gpu run must not write a result");
}

#[test]
fn auto_dispatch_falls_back_to_the_cpu_with_the_cause_and_the_cpu_archive() {
    if !gpu_available() {
        return;
    }
    let w = Work::new("auto");
    let control = w.run("control", &[], None);
    assert_eq!(control.status.code(), Some(0), "{}", text(&control.stderr));
    assert!(
        std::fs::read_to_string(w.json("control")).unwrap().contains("\"backend\":\"gpu\""),
        "the control run must use the device, otherwise the test proves nothing"
    );

    let forced = w.run("forced", &[], Some(KERNEL));
    let err = text(&forced.stderr);
    assert_eq!(forced.status.code(), Some(0), "{err}");
    assert!(err.contains(&format!("refused launch of {KERNEL}")), "{err}");
    let json = std::fs::read_to_string(w.json("forced")).unwrap();
    assert!(json.contains("\"backend\":\"cpu\""), "{json}");
    assert!(json.contains("\"backend_reason\":\"CUDA batch evaluation failed\""), "{json}");

    let cpu = w.run("cpu", &["--cpu"], None);
    assert_eq!(cpu.status.code(), Some(0), "{}", text(&cpu.stderr));
    assert_eq!(archive_hash(&w.log("forced")), archive_hash(&w.log("cpu")));
}
