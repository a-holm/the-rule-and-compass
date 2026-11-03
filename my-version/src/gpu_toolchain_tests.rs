//! Toolchain facts recorded by `build.rs` (#663).

use super::*;

extern "C" {
    fn rc_fit_occupancy(blocks: *mut i32, regs: *mut i32) -> i32;
}

#[test]
fn nvcc_reported_no_diagnostics() {
    let count = option_env!("RCSR_BUILD_NVCC_DIAGNOSTICS").expect("build.rs records the nvcc diagnostics count");
    assert_eq!(
        count,
        "0",
        "nvcc printed {} diagnostic line(s); read {}/nvcc.log",
        count,
        env!("OUT_DIR")
    );
}

#[test]
fn linked_runtime_matches_the_recorded_toolkit() {
    let build = crate::provenance::CUDA_BUILD.expect("a CUDA build records its toolkit");
    let runtime = cuda_runtime_version().expect("cudaRuntimeGetVersion");
    let major: u32 = build.toolkit.split('.').next().unwrap().parse().unwrap();
    assert_eq!(runtime.0, major, "runtime {:?} against toolkit {}", runtime, build.toolkit);
}

#[test]
fn fit_kernel_occupancy_report() {
    let (mut blocks, mut regs) = (0i32, 0i32);
    let rc = unsafe { rc_fit_occupancy(&mut blocks, &mut regs) };
    assert_eq!(rc, 0);
    let build = crate::provenance::CUDA_BUILD.expect("a CUDA build records its toolkit");
    println!(
        "fit_kernel<16,256,128> toolkit {} registers {} resident_blocks_per_sm {}",
        build.toolkit, regs, blocks
    );
    assert!(blocks >= 1);
}
