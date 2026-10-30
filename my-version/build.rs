//! Build script.
//!
//! The CUDA backend is **opt-in** (`cargo build --features cuda`). Without the
//! feature this script does nothing, so the crate stays dependency-free and
//! builds anywhere. With the feature it compiles `src/cuda/eval.cu` into a
//! shared object and links it.
//!
//! Overridable via environment:
//!   RCSR_CUDA_ARCH    target architecture (default `sm_86`, i.e. Ampere)
//!   RCSR_NVCC         nvcc binary (default `nvcc`)
//!   RCSR_GPU_UNIFIED  `1` selects `cudaMallocManaged` for every buffer
//!                     (unified memory). Defaults to `1` for `sm_121`.
//!
//! Architecture names are validated by `src/cuda/arch.rs`, which is also
//! compiled as a module by `src/gpu.rs` so the parser has unit tests. The
//! important behaviour here is the failure path: when the toolkit is too old
//! for the requested target we print the target, the detected toolkit, and the
//! minimum supported version instead of the raw nvcc diagnostics.

include!("src/cuda/arch.rs");

use std::process::Command;

/// Concatenate captured nvcc output for diagnostics (stderr first: nvcc writes
/// its errors there).
fn output_text(out: &std::process::Output) -> String {
    let mut text = String::from_utf8_lossy(&out.stderr).into_owned();
    text.push_str(&String::from_utf8_lossy(&out.stdout));
    text
}

/// Run `nvcc --version` and parse the toolkit version out of it.
fn detect_nvcc_version(nvcc: &str) -> Option<(u32, u32)> {
    let out = Command::new(nvcc).arg("--version").output().ok()?;
    if !out.status.success() {
        return None;
    }
    parse_nvcc_version(&output_text(&out))
}

/// Print each line as a build-script error, then fail. Keeping the explanation
/// out of the raw `panic!` payload makes `cargo` show it verbatim under
/// "error: failed to run custom build command".
fn fail(lines: &[String]) -> ! {
    for line in lines {
        eprintln!("error: {}", line);
    }
    panic!("CUDA backend build failed (see the error above)");
}

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/eval.cu");
    println!("cargo:rerun-if-changed=src/cuda/enrich.cu");
    println!("cargo:rerun-if-changed=src/cuda/tail.cu");
    println!("cargo:rerun-if-changed=src/cuda/mtf.cu");
    println!("cargo:rerun-if-changed=src/cuda/features.cu");
    println!("cargo:rerun-if-changed=src/cuda/enrich_common.h");
    println!("cargo:rerun-if-changed=src/cuda/prep.cu");
    println!("cargo:rerun-if-changed=src/cuda/pnl.cu");
    println!("cargo:rerun-if-changed=src/cuda/arch.rs");
    println!("cargo:rerun-if-env-changed=RCSR_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=RCSR_NVCC");
    println!("cargo:rerun-if-env-changed=RCSR_GPU_UNIFIED");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let nvcc = std::env::var("RCSR_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let arch_text = std::env::var("RCSR_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let so = format!("{}/librcsrcuda.so", out);

    let arch = match parse_arch(&arch_text) {
        Ok(a) => a,
        Err(e) => fail(&[
            format!("invalid RCSR_CUDA_ARCH: {}", e),
            String::new(),
            "Set it to a target the toolkit supports, for example:".to_string(),
            "  RCSR_CUDA_ARCH=sm_86     # RTX 3060 / Ampere".to_string(),
            "  RCSR_CUDA_ARCH=sm_121    # GB10 / DGX Spark (needs CUDA 12.9+)".to_string(),
        ]),
    };

    let detected = detect_nvcc_version(&nvcc);
    let detected_text = detected
        .map(format_version)
        .unwrap_or_else(|| "unknown (could not run `nvcc --version`)".to_string());

    // Tell the user about unified memory before compiling, so the configuration
    // is visible even when the build succeeds.
    let unified = match std::env::var("RCSR_GPU_UNIFIED") {
        Ok(v) => matches!(v.trim(), "1" | "true" | "yes" | "on"),
        Err(_) => arch.major == 12 && arch.minor == 1,
    };
    if unified {
        println!(
            "cargo:warning=CUDA backend: unified memory (cudaMallocManaged) enabled for {}",
            arch.name()
        );
    }

    // Fail early, with a readable explanation, when the toolkit predates the
    // target. nvcc's own message ("Unsupported gpu architecture ...") does not
    // say which CUDA version is needed, which is the whole problem.
    if let (Some(required), Some(found)) = (minimum_cuda(arch), detected) {
        if !version_at_least(found, required) {
            let mut lines = vec![
                format!(
                    "RCSR_CUDA_ARCH={} needs CUDA {} or newer, but `{}` is CUDA {}.",
                    arch.name(),
                    format_version(required),
                    nvcc,
                    detected_text
                ),
            ];
            if arch.major == 12 && arch.minor == 1 {
                lines.push(
                    "sm_121 is the GB10 in the DGX Spark; its DGX OS images ship CUDA 13.0."
                        .to_string(),
                );
            }
            lines.push(String::new());
            lines.push("Install a newer toolkit and point RCSR_NVCC at it, for example:".to_string());
            lines.push(format!(
                "  RCSR_NVCC=/usr/local/cuda-{}/bin/nvcc RCSR_CUDA_ARCH={} cargo build --release --features cuda",
                format_version(required),
                arch.name()
            ));
            lines.push("or build for the GPU in this machine, for example:".to_string());
            lines.push("  RCSR_CUDA_ARCH=sm_86 cargo build --release --features cuda".to_string());
            fail(&lines);
        }
    }

    let arch_flag = format!("-arch={}", arch.name());
    let mut cmd = Command::new(&nvcc);
    // `-fmad=false` keeps multiply-add association identical to the Rust f64
    // reference; the enrichment kernels must not silently contract `a*b+c`.
    cmd.args(["-O3", "-fmad=false", "-std=c++14", arch_flag.as_str(), "-shared"]);
    cmd.args(["-Xcompiler", "-fPIC"]);
    if unified {
        cmd.arg("-DRCSR_UNIFIED=1");
    }
    cmd.args([
        "-o",
        so.as_str(),
        "src/cuda/eval.cu",
        "src/cuda/enrich.cu",
        "src/cuda/tail.cu",
        "src/cuda/mtf.cu",
        "src/cuda/features.cu",
        "src/cuda/prep.cu",
        "src/cuda/pnl.cu",
        "-lcudart",
    ]);

    let result = cmd.output().unwrap_or_else(|e| {
        fail(&[
            format!("could not run `{}`: {}", nvcc, e),
            "Install the CUDA toolkit or set RCSR_NVCC to the nvcc binary.".to_string(),
        ])
    });

    if !result.status.success() {
        // Do not dump the whole nvcc log: pick the first diagnostic so the
        // action (the target and toolkit) stays the headline.
        let text = output_text(&result);
        let first = text
            .lines()
            .map(str::trim)
            .find(|l| !l.is_empty())
            .unwrap_or("(nvcc produced no diagnostics)");
        let mut lines = vec![format!(
            "nvcc (CUDA {}) rejected -arch={}.",
            detected_text,
            arch.name()
        )];
        if let Some(required) = minimum_cuda(arch) {
            lines.push(format!(
                "{} needs CUDA {} or newer; check the installed toolkit and RCSR_NVCC.",
                arch.name(),
                format_version(required)
            ));
        }
        lines.push(format!("nvcc: {}", first));
        lines.push("Run `nvcc --version` to see the detected toolkit.".to_string());
        fail(&lines);
    }

    println!("cargo:rustc-link-search=native={}", out);
    println!("cargo:rustc-link-lib=dylib=rcsrcuda");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out);
}
