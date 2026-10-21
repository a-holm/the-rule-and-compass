//! Compile-only validation for `RCSR_CUDA_ARCH=sm_121` (#28).
//!
//! This test needs a CUDA toolkit that knows `sm_121` (CUDA >= 12.9) but **no
//! GPU**. It compiles the real `src/cuda/eval.cu` to a cubin and inspects the
//! result with `cuobjdump` and `ptxas -v`. When no suitable toolkit is found it
//! prints a skip note and returns, so the ordinary CPU-only suite stays green.
//!
//! Point it at a toolkit with `RCSR_SM121_NVCC` (or the usual `RCSR_NVCC`):
//!
//! ```sh
//! RCSR_SM121_NVCC=/path/to/cuda-13/bin/nvcc \
//!   cargo test --release --test sm121_compile -- --nocapture
//! ```
//!
//! What this proves: an `sm_121` cubin and its SASS exist, and `ptxas` reports
//! the register/stack/smem budget. What it does **not** prove: that a GB10
//! loaded or ran the kernels. That is the hardware-gated half of #28.

#[path = "../src/cuda/arch.rs"]
mod arch;

use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// Candidate `nvcc` binaries, in priority order.
fn candidate_nvccs() -> Vec<PathBuf> {
    let mut out = Vec::new();
    for var in ["RCSR_SM121_NVCC", "RCSR_NVCC"] {
        if let Ok(v) = std::env::var(var) {
            if !v.trim().is_empty() {
                out.push(PathBuf::from(v));
            }
        }
    }
    out.push(PathBuf::from("nvcc"));
    // Common system layouts, so the test works without an env var on a host
    // with `/usr/local/cuda-13.x`.
    if let Ok(entries) = std::fs::read_dir("/usr/local") {
        let mut cudas: Vec<PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("cuda"))
                    .unwrap_or(false)
            })
            .collect();
        cudas.sort();
        cudas.reverse();
        for dir in cudas {
            out.push(dir.join("bin/nvcc"));
        }
    }
    out
}

fn output_text(out: &Output) -> String {
    let mut text = String::from_utf8_lossy(&out.stdout).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stderr));
    text
}

fn nvcc_version(nvcc: &Path) -> Option<(u32, u32)> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    arch::parse_nvcc_version(&output_text(&out))
}

/// `cuobjdump` normally sits next to `nvcc`; fall back to `PATH`.
fn sibling_tool(nvcc: &Path, name: &str) -> PathBuf {
    match nvcc.parent() {
        Some(dir) if !dir.as_os_str().is_empty() => dir.join(name),
        _ => PathBuf::from(name),
    }
}

fn run(cmd: &mut Command) -> Output {
    let display = format!("{:?}", cmd);
    cmd.output()
        .unwrap_or_else(|e| panic!("could not run {}: {}", display, e))
}

/// `(registers, stack bytes, smem bytes)` for one kernel from `ptxas -v`.
fn ptxas_resources(log: &str, kernel: &str) -> Option<(u32, u64, u64)> {
    let mut lines = log.lines();
    while let Some(line) = lines.next() {
        if line.contains("Compiling entry function") && line.contains(kernel) {
            for detail in lines.by_ref().take(4) {
                if let Some(rest) = detail.split("Used ").nth(1) {
                    let registers: u32 = rest.split(' ').next()?.parse().ok()?;
                    let stack = detail
                        .split(" bytes cumulative stack size")
                        .next()
                        .and_then(|s| s.split_whitespace().last())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    let smem = detail
                        .split("bytes smem")
                        .next()
                        .and_then(|s| s.split_whitespace().last())
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    return Some((registers, stack, smem));
                }
            }
        }
    }
    None
}

#[test]
fn sm121_requires_cuda_12_9() {
    let sm121 = arch::parse_arch("sm_121").unwrap();
    let sm121a = arch::parse_arch("sm_121a").unwrap();
    assert_eq!(arch::minimum_cuda(sm121), Some((12, 9)));
    assert_eq!(arch::minimum_cuda(sm121a), Some((12, 9)));
    // CUDA 12.8 added Blackwell but its nvcc stops at compute_120.
    assert!(!arch::version_at_least((12, 8), (12, 9)));
    assert!(arch::version_at_least((12, 9), (12, 9)));
    assert!(arch::version_at_least((13, 0), (12, 9)));
}

#[test]
fn compiles_eval_cu_for_sm_121_without_a_gpu() {
    let Some((nvcc, version)) = candidate_nvccs()
        .into_iter()
        .find_map(|nvcc| nvcc_version(&nvcc).map(|v| (nvcc, v)))
    else {
        eprintln!(
            "skip sm_121 compile check: no working nvcc \
             (set RCSR_SM121_NVCC to a CUDA >= 12.9 toolkit)"
        );
        return;
    };
    if !arch::version_at_least(version, (12, 9)) {
        eprintln!(
            "skip sm_121 compile check: `{}` is CUDA {}; sm_121 needs 12.9+",
            nvcc.display(),
            arch::format_version(version)
        );
        return;
    }

    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let eval_cu = manifest.join("src/cuda/eval.cu");
    let dir = std::env::temp_dir().join(format!("rcsr-sm121-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create temporary build directory");
    let cubin = dir.join("eval.sm_121.cubin");

    // The same flags `build.rs` uses, minus `-shared`: a cubin is what
    // `cuobjdump` and `ptxas -v` can inspect directly.
    let compiled = run(Command::new(&nvcc)
        .args([
            "-O3",
            "-std=c++14",
            "-arch=sm_121",
            "-DRCSR_UNIFIED=1",
            "-cubin",
            "-Xptxas",
            "-v",
            "-o",
        ])
        .arg(&cubin)
        .arg(&eval_cu)
        .current_dir(&manifest));
    let log = output_text(&compiled);
    assert!(
        compiled.status.success(),
        "nvcc {} failed to compile {} for sm_121:\n{}",
        nvcc.display(),
        eval_cu.display(),
        log
    );
    assert!(
        log.contains("for 'sm_121'"),
        "ptxas did not report an sm_121 target:\n{}",
        log
    );

    // SASS: the cubin must name the architecture and disassemble as sm_121.
    let cuobjdump = sibling_tool(&nvcc, "cuobjdump");
    let elf = run(Command::new(&cuobjdump).arg("--list-elf").arg(&cubin));
    let elf_text = output_text(&elf);
    assert!(
        elf.status.success(),
        "`{} --list-elf` failed:\n{}",
        cuobjdump.display(),
        elf_text
    );
    assert!(
        elf_text.contains(".sm_121.cubin"),
        "cubin list does not contain an sm_121 ELF image:\n{}",
        elf_text
    );

    let sass = run(Command::new(&cuobjdump).arg("--dump-sass").arg(&cubin));
    let sass_text = output_text(&sass);
    assert!(
        sass.status.success(),
        "`{} --dump-sass` failed:\n{}",
        cuobjdump.display(),
        sass_text
    );
    assert!(
        sass_text.contains("code for sm_121"),
        "SASS is not tagged sm_121"
    );
    assert!(
        sass_text.contains("EF_CUDA_SM121"),
        "SASS header flags do not contain EF_CUDA_SM121"
    );

    // Resource budget: record it, and fail loudly if ptxas stopped reporting.
    for (kernel, max_registers) in [("eval_pool_kernel", 64u32), ("fit_kernel", 128u32)] {
        let (registers, stack, smem) = ptxas_resources(&log, kernel)
            .unwrap_or_else(|| panic!("no ptxas resource line for {}:\n{}", kernel, log));
        eprintln!(
            "sm_121 {}: {} registers, {} B stack, {} B smem",
            kernel, registers, stack, smem
        );
        assert!(
            registers <= max_registers,
            "{} uses {} registers for sm_121, above the {} budget",
            kernel,
            registers,
            max_registers
        );
    }

    eprintln!(
        "compiled sm_121 with CUDA {}; this is compile evidence, not a GB10 run",
        arch::format_version(version)
    );
    let _ = std::fs::remove_dir_all(&dir);
}
