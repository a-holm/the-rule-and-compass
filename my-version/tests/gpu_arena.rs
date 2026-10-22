//! #285: persistent device arena, pinned staging and asynchronous copies.
//!
//! These tests pin the schedule contract of the batched evaluator:
//!
//! * after the first batch, a stable batch shape performs **no allocator
//!   call** and keeps the pinned ring at a constant size (<= 64 MiB);
//! * the asynchronous path and the forced synchronous fallback produce
//!   bit-for-bit identical errors for the same programs;
//! * the second upload is enqueued while the first kernel is still
//!   outstanding (`overlapped_uploads`), and the event ordering still yields
//!   the same results as two sequential batches;
//! * the fallback carries a recorded, human-readable reason.
//!
//! Device-dependent tests skip cleanly on a machine without CUDA.

#[cfg(feature = "cuda")]
use rcsr::expr::{self, Op};
#[cfg(feature = "cuda")]
use rcsr::gpu::{GpuEvaluator, Population};
#[cfg(feature = "cuda")]
use rcsr::rng::Mt;

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

/// A deterministic two-variable sample, column-major for the kernel.
#[cfg(feature = "cuda")]
fn sample(n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cols = vec![0.0f32; 2 * n];
    let mut y = vec![0.0f32; n];
    for r in 0..n {
        let a = (r as f32) / (n as f32) * 4.0 - 2.0;
        let b = (((r * 13) % n) as f32) / (n as f32) * 2.0 - 1.0;
        cols[r] = a;
        cols[n + r] = b;
        y[r] = 1.7 * a - 0.4 * b + 0.3 * (a * b).sin() + 0.5;
    }
    (cols, y)
}

#[cfg(feature = "cuda")]
fn population(nprog: usize, seed: u32, depth: usize) -> Population {
    let mut rng = Mt::new(seed);
    let ops = [Op::Add, Op::Sub, Op::Mul, Op::Div, Op::Sin];
    let trees: Vec<_> = (0..nprog)
        .map(|_| expr::random_tree(&mut rng, 2, &ops, depth))
        .collect();
    Population::from_trees(trees.iter())
}

/// Repeat a compiled batch to `nprog` slots without rebuilding the trees.
#[cfg(feature = "cuda")]
fn repeat(prog: &[u32], offsets: &[u32], nprog: usize) -> (Vec<u32>, Vec<u32>) {
    let mut out = Vec::with_capacity(nprog * prog.len());
    let mut off = Vec::with_capacity(nprog + 1);
    off.push(0u32);
    for _ in 0..nprog {
        out.extend_from_slice(prog);
        off.push(out.len() as u32);
    }
    let _ = offsets;
    (out, off)
}

/// 100 generations of the same shape allocate nothing and keep the ring fixed.
#[test]
fn steady_state_allocates_nothing_and_keeps_the_pinned_ring_bounded() {
    if !device_available() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (cols, y) = sample(2_048);
    let mut ev = GpuEvaluator::new(&cols, 2_048, 2, &y).expect("device evaluator");
    let pop = population(96, 5, 4);
    let (prog, off) = pop.programs();

    // Warm batch: this is the only place allocation is allowed.
    let (warm_out, warm_ms) = ev.eval_batch(prog, off);
    assert!(warm_ms >= 0.0, "warm batch ran");
    assert_eq!(warm_out.len(), pop.len());
    let warm = ev.arena_report();
    assert!(
        warm.async_copy,
        "expected the pinned asynchronous path, got reason: {}",
        warm.reason
    );
    assert!(warm.pinned_bytes > 0);
    assert!(
        warm.pinned_bytes <= 64 * 1024 * 1024,
        "pinned ring {} exceeds the 64 MiB cap",
        warm.pinned_bytes
    );

    for generation in 0..100 {
        let (out, ms) = ev.eval_batch(prog, off);
        assert!(ms >= 0.0, "generation {generation} ran");
        assert_eq!(out.len(), pop.len());
    }
    let after = ev.arena_report();
    assert_eq!(
        after.steady_alloc_calls - warm.steady_alloc_calls,
        0,
        "allocator calls after warm-up: {} -> {}",
        warm.steady_alloc_calls,
        after.steady_alloc_calls
    );
    assert_eq!(
        after.pinned_bytes, warm.pinned_bytes,
        "the pinned ring grew across a stable batch shape"
    );
    assert_eq!(after.batches - warm.batches, 100);
}

/// The pinned async path and the forced blocking path agree bit for bit.
#[test]
fn async_and_synchronous_paths_are_numerically_identical() {
    if !device_available() {
        eprintln!("skip: no CUDA device");
        return;
    }
    let (cols, y) = sample(4_096);
    let pop = population(64, 11, 5);
    let (prog, off) = pop.programs();

    let mut fast = GpuEvaluator::new(&cols, 4_096, 2, &y).expect("device evaluator");
    let (fast_out, fast_ms) = fast.eval_batch(prog, off);
    assert!(fast_ms >= 0.0);

    let mut slow = GpuEvaluator::new(&cols, 4_096, 2, &y).expect("device evaluator");
    assert!(
        slow.force_synchronous_copy(true),
        "the default mode must be asynchronous"
    );
    let report = slow.arena_report();
    assert!(!report.async_copy);
    assert!(
        report.reason.contains("forced"),
        "fallback reason must be recorded, got {:?}",
        report.reason
    );
    let (slow_out, slow_ms) = slow.eval_batch(prog, off);
    assert!(slow_ms >= 0.0);

    let fast_bits: Vec<u32> = fast_out.iter().map(|v| v.to_bits()).collect();
    let slow_bits: Vec<u32> = slow_out.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        fast_bits, slow_bits,
        "the asynchronous arena changed the arithmetic"
    );
}

/// Batch B is uploaded while batch A's kernel is still outstanding; collecting
/// A after that reordering must return exactly the sequential result.
#[test]
fn upload_is_enqueued_behind_an_outstanding_kernel_and_results_match() {
    if !device_available() {
        eprintln!("skip: no CUDA device");
        return;
    }
    // A large enough prog x row product that the kernel is still running when
    // the host has staged and enqueued the second upload.
    let n = 20_000;
    let (cols, y) = sample(n);
    let mut ev = GpuEvaluator::new(&cols, n, 2, &y).expect("device evaluator");
    let base = population(8, 23, 4);
    let (base_prog, base_off) = base.programs();
    let (prog, off) = repeat(base_prog, base_off, 20_000);

    let (expected, warm_ms) = ev.eval_batch(&prog, &off);
    assert!(warm_ms >= 0.0);
    let before = ev.arena_report();

    let ((a, ms_a), (b, ms_b)) = ev.eval_batch_pipelined(&prog, &off, &prog, &off);
    assert!(ms_a >= 0.0 && ms_b >= 0.0, "both pipelined batches ran");
    let after = ev.arena_report();
    assert!(
        after.overlapped_uploads > before.overlapped_uploads,
        "the second upload was not issued while the first kernel was outstanding \
         (overlapped_uploads {} -> {})",
        before.overlapped_uploads,
        after.overlapped_uploads
    );
    assert_eq!(a.len(), expected.len());
    assert_eq!(b.len(), expected.len());
    let expected_bits: Vec<u32> = expected.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected_bits,
        "batch A changed after the upload reordering"
    );
    assert_eq!(
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected_bits,
        "batch B differs from the sequential score"
    );
    assert_eq!(after.steady_alloc_calls, before.steady_alloc_calls);
}