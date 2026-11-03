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

#[test]
fn fit_kernel_large_runs_three_blocks_per_sm() {
    let (blocks20, regs20, smem20) = GpuEvaluator::fit_occupancy_for(20).expect("occupancy of the p 17..20 kernel");
    let (blocks16, regs16, smem16) = GpuEvaluator::fit_occupancy_for(16).expect("occupancy of the p <= 16 kernel");
    let build = crate::provenance::CUDA_BUILD.expect("a CUDA build records its toolkit");
    println!(
        "toolkit {} fit_kernel<20,128,128> registers {} smem {} blocks {} ; fit_kernel<16,256,128> registers {} smem {} blocks {}",
        build.toolkit, regs20, smem20, blocks20, regs16, smem16, blocks16
    );
    assert_eq!(smem20, 31_624);
    assert_eq!(smem16, 28_040);
    assert_eq!(blocks20, 3, "the p 17..20 kernel runs three blocks per SM");
    assert_eq!(blocks16, 2, "the p <= 16 kernel keeps two blocks per SM");
}
