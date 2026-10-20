//! #118: GPU accounting shared by the CUDA evaluator, the search report and
//! the benchmarks.
//!
//! The C backend times the full batch path (`stats_begin` → reserve/upload →
//! launch → sync → download) with disjoint host windows. This type accumulates
//! those windows across batches and derives the honest kernel-time duty
//! fraction. It is compiled without CUDA so the arithmetic has tests on every
//! machine.
//!
//! What this is *not*: `duty_cycle` is `kernel_s / wall_s`, a kernel-time
//! fraction. It is not `nvidia-smi`'s 1 Hz "some kernel was resident" sample and
//! it is not SM occupancy. Achieved occupancy needs `ncu`; this type leaves it
//! out entirely rather than inventing it.

/// #118: accumulated GPU accounting for a run, in seconds and bytes.
///
/// `wall_s` is measured by the Rust caller around the whole evaluator call, so
/// it includes the C host windows and the small Rust glue between them.
/// `kernel_s` is GPU-event time inside that wall and is deliberately not part of
/// [`GpuStats::accounted_s`]: it overlaps the launch/sync windows and would
/// double count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuStats {
    /// GPU-event time between the launch and completion events.
    pub kernel_s: f64,
    /// Host time in the kernel launch calls (including event records).
    pub launch_s: f64,
    /// Host time blocked in `cudaEventSynchronize`.
    pub sync_s: f64,
    /// Host time in `cudaMalloc`/`cudaFree`/`cudaMallocManaged` (synchronising).
    pub alloc_s: f64,
    /// Host time in the host→device program upload.
    pub h2d_s: f64,
    /// Host time in the device→host SSE download.
    pub d2h_s: f64,
    /// Host glue outside the measured C windows: Rust-side allocation and the
    /// gaps between reserve/upload/eval calls. It is the difference between the
    /// independently measured outer wall and the C envelope, so the ledger
    /// closes to wall instead of hiding an `other` bucket.
    pub host_s: f64,
    /// Independently measured wall of every evaluator call.
    pub wall_s: f64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    /// Number of evaluator batches (generation evaluations).
    pub batches: u64,
    /// Number of compiled programs scored.
    pub programs: u64,
    /// Number of rows in the training/evaluation sample.
    pub rows: u64,
    /// Candidate evaluations: one per program scored, i.e. `programs`.
    pub evals: u64,
    /// Row evaluations: `programs × rows`.
    pub row_evals: u64,
    /// Batches that returned an error after being timed.
    pub failed_batches: u64,
}

impl GpuStats {
    /// Add one batch's ledger. `wall_ms` is the independently measured envelope
    /// of the whole evaluator call. `failed` marks a batch that returned an
    /// error: its wall and windows are real, but it contributes no evaluations,
    /// so a failed batch can never inflate `evals/s`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_batch(
        &mut self,
        kernel_ms: f64,
        launch_ms: f64,
        sync_ms: f64,
        alloc_ms: f64,
        h2d_ms: f64,
        d2h_ms: f64,
        host_ms: f64,
        wall_ms: f64,
        h2d_bytes: u64,
        d2h_bytes: u64,
        programs: u64,
        rows: u64,
        failed: bool,
    ) {
        // A failed launch can return a negative kernel time; the event pair did
        // not produce a reading, so keep the accumulated time honest.
        self.kernel_s += kernel_ms.max(0.0) / 1e3;
        self.launch_s += launch_ms.max(0.0) / 1e3;
        self.sync_s += sync_ms.max(0.0) / 1e3;
        self.alloc_s += alloc_ms.max(0.0) / 1e3;
        self.h2d_s += h2d_ms.max(0.0) / 1e3;
        self.d2h_s += d2h_ms.max(0.0) / 1e3;
        self.host_s += host_ms.max(0.0) / 1e3;
        self.wall_s += wall_ms.max(0.0) / 1e3;
        self.h2d_bytes += h2d_bytes;
        self.d2h_bytes += d2h_bytes;
        self.batches += 1;
        if failed {
            self.failed_batches += 1;
        } else {
            self.programs += programs;
            self.rows = rows;
            self.evals += programs;
            self.row_evals += programs * rows;
        }
    }

    /// Subtract an earlier snapshot, for timing a benchmark window.
    pub fn since(&self, earlier: &GpuStats) -> GpuStats {
        GpuStats {
            kernel_s: self.kernel_s - earlier.kernel_s,
            launch_s: self.launch_s - earlier.launch_s,
            sync_s: self.sync_s - earlier.sync_s,
            alloc_s: self.alloc_s - earlier.alloc_s,
            h2d_s: self.h2d_s - earlier.h2d_s,
            d2h_s: self.d2h_s - earlier.d2h_s,
            host_s: self.host_s - earlier.host_s,
            wall_s: self.wall_s - earlier.wall_s,
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            batches: self.batches.saturating_sub(earlier.batches),
            programs: self.programs.saturating_sub(earlier.programs),
            rows: self.rows,
            evals: self.evals.saturating_sub(earlier.evals),
            row_evals: self.row_evals.saturating_sub(earlier.row_evals),
            failed_batches: self.failed_batches.saturating_sub(earlier.failed_batches),
        }
    }

    /// Both copies, the way a report usually wants them.
    pub fn transfer_s(&self) -> f64 {
        self.h2d_s + self.d2h_s
    }

    /// The measured C windows alone. `wall_s − window_s` is the host glue that
    /// [`Self::host_s`] records.
    pub fn window_s(&self) -> f64 {
        self.alloc_s + self.h2d_s + self.launch_s + self.sync_s + self.d2h_s
    }

    /// Disjoint phases that should add up to `wall_s` (kernel time overlaps and
    /// is excluded). `host_s` is the residue outside the C windows, measured
    /// against the independent outer wall rather than guessed.
    pub fn accounted_s(&self) -> f64 {
        self.window_s() + self.host_s
    }

    /// Wall not spent inside the kernel: host overhead plus GPU idle.
    pub fn idle_s(&self) -> f64 {
        (self.wall_s - self.kernel_s).max(0.0)
    }

    /// Kernel-time fraction of the measured full path. A duty estimate, not
    /// occupancy.
    pub fn duty_cycle(&self) -> f64 {
        if self.wall_s > 0.0 {
            (self.kernel_s / self.wall_s).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    /// Candidate evaluations per full-path second.
    pub fn evals_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.evals as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Row evaluations (`programs × rows`) per full-path second.
    pub fn row_evals_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.row_evals as f64 / self.wall_s
        } else {
            0.0
        }
    }

    pub fn generations_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.batches as f64 / self.wall_s
        } else {
            0.0
        }
    }

    /// Decades of error reduction against the constant-mean baseline per
    /// GPU-hour: `-log10(best_error / baseline_error) / hours`. Positive means
    /// the searched model beats the trivial mean predictor. `None` when either
    /// error is not finite/positive or no GPU time was measured.
    pub fn quality_per_gpu_hour(&self, best_error: f64, baseline_error: f64) -> Option<f64> {
        if !best_error.is_finite()
            || best_error <= 0.0
            || !baseline_error.is_finite()
            || baseline_error <= 0.0
            || !(self.wall_s > 0.0)
        {
            return None;
        }
        Some((-(best_error / baseline_error).log10()) / (self.wall_s / 3600.0))
    }

    /// The additive host decomposition agrees with the independently measured
    /// wall to within `tol` (relative). Used as an accounting test and reported
    /// as `residual_fraction` so a reader can see the unmeasured glue.
    pub fn decomposition_within(&self, tol: f64) -> bool {
        self.wall_s > 0.0 && (self.accounted_s() - self.wall_s).abs() <= tol * self.wall_s
    }

    /// Signed relative residual `(accounted − wall) / wall`.
    pub fn residual_fraction(&self) -> f64 {
        if self.wall_s > 0.0 {
            (self.accounted_s() - self.wall_s) / self.wall_s
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #118: the host windows are disjoint and add up to wall; kernel time is
    /// overlapping and only feeds duty/idle, never the additive sum.
    #[test]
    fn gpu_accounting_decomposes_wall_and_reports_honest_duty() {
        let mut st = GpuStats::default();
        st.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 0.0, 5.0, 800, 400, 10, 128, false);
        assert!(st.decomposition_within(1e-12));
        assert!((st.accounted_s() - 0.005).abs() < 1e-12);
        assert!((st.wall_s - 0.005).abs() < 1e-12);
        assert_eq!(st.evals, 10);
        assert_eq!(st.programs, 10);
        assert_eq!(st.row_evals, 1280);
        assert_eq!(st.rows, 128);
        assert_eq!(st.batches, 1);
        assert_eq!(st.h2d_bytes + st.d2h_bytes, 1200);
        assert!((st.evals_per_s() - 10.0 / 0.005).abs() < 1e-6);
        assert!((st.row_evals_per_s() - 1280.0 / 0.005).abs() < 1e-6);
        // Kernel is 2 ms of a 5 ms path: 40% duty, 3 ms idle.
        assert!((st.duty_cycle() - 0.4).abs() < 1e-12);
        assert!((st.idle_s() - 0.003).abs() < 1e-12);
        // 1e-6 against a 1.0 baseline is six decades removed.
        assert!(
            (st.quality_per_gpu_hour(1e-6, 1.0).unwrap() - 6.0 / (0.005 / 3600.0)).abs() < 1e-6
        );
        assert!(st.quality_per_gpu_hour(0.0, 1.0).is_none());

        let mut twice = st;
        twice.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 0.0, 5.0, 800, 400, 10, 128, false);
        assert_eq!(twice.batches, 2);
        assert_eq!(twice.evals, 20);
        assert!((twice.wall_s - 2.0 * st.wall_s).abs() < 1e-12);
    }

    /// #118: a failed batch keeps its measured wall and transfer bytes but adds
    /// no evaluations, so it cannot make throughput or quality look better.
    #[test]
    fn failed_batch_is_timed_but_adds_no_evals() {
        let mut st = GpuStats::default();
        st.add_batch(-1.0, 0.2, 0.3, 0.4, 0.1, 0.0, 0.0, 1.0, 64, 0, 8, 1000, true);
        assert_eq!(st.batches, 1);
        assert_eq!(st.failed_batches, 1);
        assert_eq!(st.evals, 0);
        assert_eq!(st.row_evals, 0);
        assert_eq!(st.evals_per_s(), 0.0);
        assert!((st.wall_s - 0.001).abs() < 1e-12);
        assert!((st.accounted_s() - 0.001).abs() < 1e-12);
        assert!(st.decomposition_within(1e-9));

        st.add_batch(2.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 4.0, 64, 4, 8, 1000, false);
        assert_eq!(st.evals, 8);
        assert_eq!(st.failed_batches, 1);
    }

    /// #118: `host_s` is the residue between the outer wall and the C envelope;
    /// adding it makes the phase sum close without an opaque `other` bucket.
    #[test]
    fn host_glue_closes_the_phase_ledger() {
        let mut st = GpuStats::default();
        st.add_batch(1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.6, 8, 0, 2, 10, false);
        assert!((st.window_s() - 0.0005).abs() < 1e-12);
        assert!((st.host_s - 0.0001).abs() < 1e-12);
        assert!((st.accounted_s() - 0.0006).abs() < 1e-12);
        assert!(st.decomposition_within(1e-12));
        assert!(st.residual_fraction().abs() < 1e-12);
    }

    /// #118: `since()` differences a benchmark window without losing bytes.
    #[test]
    fn since_subtracts_a_snapshot() {
        let mut st = GpuStats::default();
        st.add_batch(1.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 2.0, 10, 20, 4, 100, false);
        let before = st;
        st.add_batch(2.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 4.0, 30, 40, 8, 100, false);
        let d = st.since(&before);
        assert_eq!(d.batches, 1);
        assert_eq!(d.evals, 8);
        assert_eq!(d.h2d_bytes, 30);
        assert_eq!(d.d2h_bytes, 40);
        assert!((d.wall_s - 0.004).abs() < 1e-12);
        assert!((d.accounted_s() - 0.001).abs() < 1e-12); // five 0.2 ms windows
    }
}
