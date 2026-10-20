//! Backend selection and the one-time device probe (issue #141).
//!
//! The CUDA build treats the GPU as the **preferred** backend: a default run
//! probes the device once and uses it when it is available. A build without the
//! `cuda` feature (or a run on a machine with no usable device) falls back to
//! the CPU and records why. `--no-gpu`/`--cpu` is the deliberate override and
//! `--gpu` is an explicit assertion that fails when no device is present.
//!
//! The probe is deliberately separate from opening a [`crate::gpu::GpuEvaluator`]:
//! it must run before the dataset is uploaded, so the CLI can decide the
//! backend and the resource summary without allocating device memory.

/// Which evaluator guides the search.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Gpu,
    Cpu,
}

impl Kind {
    pub fn name(self) -> &'static str {
        match self {
            Kind::Gpu => "gpu",
            Kind::Cpu => "cpu",
        }
    }
}

/// The result of probing for a usable CUDA device once at startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Probe {
    pub kind: Kind,
    /// The device description when a device answered, otherwise `None`.
    pub device: Option<String>,
    /// Why the CPU was selected (or the device is unusable). `None` for a
    /// healthy GPU probe and for a deliberate CPU run.
    pub reason: Option<String>,
}

impl Probe {
    pub fn cpu(reason: impl Into<String>) -> Self {
        Probe {
            kind: Kind::Cpu,
            device: None,
            reason: Some(reason.into()),
        }
    }

    pub fn gpu(device: String) -> Self {
        Probe {
            kind: Kind::Gpu,
            device: Some(device),
            reason: None,
        }
    }

    pub fn available(&self) -> bool {
        self.kind == Kind::Gpu
    }
}

/// Probe the CUDA device once. A non-CUDA build reports why it cannot.
pub fn probe() -> Probe {
    #[cfg(feature = "cuda")]
    {
        crate::gpu::probe()
    }
    #[cfg(not(feature = "cuda"))]
    {
        Probe::cpu("built without --features cuda")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn non_cuda_probe_names_the_missing_feature() {
        if !cfg!(feature = "cuda") {
            let p = probe();
            assert_eq!(p.kind, Kind::Cpu);
            assert!(p.device.is_none());
            assert_eq!(p.reason.as_deref(), Some("built without --features cuda"));
        }
    }

    #[test]
    fn cpu_probe_carries_a_reason() {
        let p = Probe::cpu("RCSR_GPU=0 (device use disabled)");
        assert_eq!(p.kind, Kind::Cpu);
        assert!(!p.available());
        assert!(p.reason.is_some());
    }
}
