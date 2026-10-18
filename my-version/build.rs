//! Build script.
//!
//! The CUDA backend is **opt-in** (`cargo build --features cuda`). Without the
//! feature this script does nothing, so the crate stays dependency-free and
//! builds anywhere. With the feature it compiles `src/cuda/eval.cu` into a
//! shared object and links it.
//!
//! Overridable via environment:
//!   RCSR_CUDA_ARCH  target architecture (default `sm_86`, i.e. Ampere)
//!   RCSR_NVCC       nvcc binary (default `nvcc`)

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=src/cuda/eval.cu");
    println!("cargo:rerun-if-env-changed=RCSR_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=RCSR_NVCC");

    if std::env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    let out = std::env::var("OUT_DIR").expect("OUT_DIR");
    let nvcc = std::env::var("RCSR_NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let arch = std::env::var("RCSR_CUDA_ARCH").unwrap_or_else(|_| "sm_86".to_string());
    let so = format!("{}/librcsrcuda.so", out);

    let status = Command::new(&nvcc)
        .args([
            "-O3",
            "-std=c++14",
            &format!("-arch={}", arch),
            "-shared",
            "-Xcompiler",
            "-fPIC",
            "-o",
            &so,
            "src/cuda/eval.cu",
            "-lcudart",
        ])
        .status()
        .unwrap_or_else(|e| panic!("could not run `{}`: {}", nvcc, e));
    assert!(status.success(), "nvcc failed to build the CUDA backend");

    println!("cargo:rustc-link-search=native={}", out);
    println!("cargo:rustc-link-lib=dylib=rcsrcuda");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out);
}
