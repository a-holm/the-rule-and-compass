//! #118: GPU accounting shared by the CUDA evaluator and the search report.
//!
//! The C backend times the full `rc_set_programs` + `rc_eval` path with
//! disjoint host windows (allocation, H2D, launch, event sync, D2H). This type
//! accumulates those windows across batches and derives the honest kernel-time
//! duty fraction. It is compiled without CUDA so the accounting has tests on
//! every machine.

/// #118: accumulated GPU accounting for a run, in seconds and bytes.
///
/// `wall_s` covers the whole `rc_set_programs` + `rc_eval` path, so it includes
/// the `cudaMalloc`/`cudaFree` and copies that the old event-only timer missed.
/// `duty_cycle` is deliberately the honest kernel-time fraction
/// (`kernel_s / wall_s`), not `nvidia-smi`'s 1 Hz sampled utilisation and not
/// SM occupancy. Occupancy needs `ncu`; it is never invented here.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct GpuStats {
    pub kernel_s: f64,
    pub launch_s: f64,
    pub sync_s: f64,
    pub alloc_s: f64,
    pub h2d_s: f64,
    pub d2h_s: f64,
    pub wall_s: f64,
    pub h2d_bytes: u64,
    pub d2h_bytes: u64,
    /// Number of batches (generation evaluations).
    pub batches: u64,
    /// Number of compiled programs scored.
    pub programs: u64,
    /// Number of row evaluations (`programs × rows`).
    pub evals: u64,
}

impl GpuStats {
    /// Add one batch's ledger (milliseconds, bytes, programs and rows).
    #[allow(clippy::too_many_arguments)]
    pub fn add_batch(
        &mut self,
        kernel_ms: f64,
        launch_ms: f64,
        sync_ms: f64,
        alloc_ms: f64,
        h2d_ms: f64,
        d2h_ms: f64,
        wall_ms: f64,
        h2d_bytes: u64,
        d2h_bytes: u64,
        programs: u64,
        rows: u64,
    ) {
        self.kernel_s += kernel_ms / 1e3;
        self.launch_s += launch_ms / 1e3;
        self.sync_s += sync_ms / 1e3;
        self.alloc_s += alloc_ms / 1e3;
        self.h2d_s += h2d_ms / 1e3;
        self.d2h_s += d2h_ms / 1e3;
        self.wall_s += wall_ms / 1e3;
        self.h2d_bytes += h2d_bytes;
        self.d2h_bytes += d2h_bytes;
        self.batches += 1;
        self.programs += programs;
        self.evals += programs * rows;
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
            wall_s: self.wall_s - earlier.wall_s,
            h2d_bytes: self.h2d_bytes.saturating_sub(earlier.h2d_bytes),
            d2h_bytes: self.d2h_bytes.saturating_sub(earlier.d2h_bytes),
            batches: self.batches.saturating_sub(earlier.batches),
            programs: self.programs.saturating_sub(earlier.programs),
            evals: self.evals.saturating_sub(earlier.evals),
        }
    }

    pub fn transfer_s(&self) -> f64 {
        self.h2d_s + self.d2h_s
    }

    /// Host windows that should add up to `wall_s` (kernel time is overlapping).
    pub fn accounted_s(&self) -> f64 {
        self.alloc_s + self.h2d_s + self.launch_s + self.sync_s + self.d2h_s
    }

    /// Wall not spent inside the kernel: host overhead plus GPU idle.
    pub fn idle_s(&self) -> f64 {
        (self.wall_s - self.kernel_s).max(0.0)
    }

    /// Kernel-time fraction of the measured full path. This is a duty estimate,
    /// not occupancy.
    pub fn duty_cycle(&self) -> f64 {
        if self.wall_s > 0.0 {
            (self.kernel_s / self.wall_s).clamp(0.0, 1.0)
        } else {
            0.0
        }
    }

    pub fn evals_per_s(&self) -> f64 {
        if self.wall_s > 0.0 {
            self.evals as f64 / self.wall_s
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
    pub fn quality_per_gpu_hour(
        &self,
        best_error: f64,
        baseline_error: f64,
    ) -> Option<f64> {
        if !best_error.is_finite()
            || best_error <= 0.0
            || !baseline_error.is_finite()
            || baseline_error <= 0.0
            || !(self.wall_s > 0.0)
        {
            return None;
        }
        Some((-((best_error / baseline_error).log10())) / (self.wall_s / 3600.0))
    }

    /// The additive host decomposition agrees with the independently measured
    /// wall to within `tol` (relative). Used as an accounting test.
    pub fn decomposition_within(&self, tol: f64) -> bool {
        self.wall_s > 0.0 && (self.accounted_s() - self.wall_s).abs() <= tol * self.wall_s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// #118: the ledger's host windows are disjoint and add up to wall; the
    /// kernel time is overlapping and only feeds duty/idle, never the sum.
    #[test]
    fn gpu_accounting_decomposes_wall_and_reports_honest_duty() {
        let mut st = GpuStats::default();
        st.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 5.0, 800, 400, 10, 128);
        assert!(st.decomposition_within(1e-12));
        assert!((st.accounted_s() - 0.005).abs() < 1e-12);
        assert!((st.wall_s - 0.005).abs() < 1e-12);
        assert_eq!(st.evals, 1280);
        assert_eq!(st.batches, 1);
        assert_eq!(st.h2d_bytes + st.d2h_bytes, 1200);
        assert!((st.evals_per_s() - 1280.0 / 0.005).abs() < 1e-6);
        // Kernel is 2 ms of a 5 ms path: 40% duty, 3 ms idle.
        assert!((st.duty_cycle() - 0.4).abs() < 1e-12);
        assert!((st.idle_s() - 0.003).abs() < 1e-12);
        // 1e-6 against a 1.0 baseline is six decades removed.
        assert!((st.quality_per_gpu_hour(1e-6, 1.0).unwrap() - 6.0 / (0.005 / 3600.0)).abs() < 1e-6);
        assert!(st.quality_per_gpu_hour(0.0, 1.0).is_none());

        let mut twice = st;
        twice.add_batch(2.0, 0.1, 3.0, 1.0, 0.4, 0.5, 5.0, 800, 400, 10, 128);
        assert_eq!(twice.batches, 2);
        assert_eq!(twice.evals, 2560);
        assert!((twice.wall_s - 2.0 * st.wall_s).abs() < 1e-12);
    }
}
