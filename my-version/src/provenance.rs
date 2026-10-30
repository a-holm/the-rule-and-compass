//! Build and run identity (issue #256).
//!
//! A manifest or checkpoint is evidence only if it names the binary that
//! produced it. The commit is recorded by `build.rs` when the binary is built,
//! never by running `git` in the process working directory, so a binary that
//! runs inside another checkout still reports the tree it was built from. The
//! SHA-256 of the running executable is the authority; the commit is a label.

use std::sync::OnceLock;

/// The commit the binary was built from (`unknown` outside a git work tree
/// unless `RCSR_BUILD_COMMIT` was set at build time).
pub const BUILD_COMMIT: &str = match option_env!("RCSR_BUILD_COMMIT") {
    Some(c) => c,
    None => "unknown",
};

/// Whether the work tree had uncommitted files when `build.rs` last ran. It is
/// not refreshed by source edits alone; the binary SHA-256 is the identity.
pub fn build_dirty() -> bool {
    option_env!("RCSR_BUILD_DIRTY") == Some("true")
}

/// What a checkpoint records about the process that wrote it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildIdentity {
    pub version: String,
    /// Build-time commit, see [`BUILD_COMMIT`].
    pub commit: String,
    pub binary_sha256: String,
    /// Device name (`cpu` for a CPU run).
    pub device: String,
    /// Driver version (`n/a` for a CPU run).
    pub driver: String,
    pub threads: usize,
}

impl BuildIdentity {
    /// The identity of this process. The executable hash and the device probe
    /// run once per process and backend.
    pub fn current(gpu: bool) -> BuildIdentity {
        static CPU: OnceLock<BuildIdentity> = OnceLock::new();
        static GPU: OnceLock<BuildIdentity> = OnceLock::new();
        let slot = if gpu { &GPU } else { &CPU };
        slot.get_or_init(|| {
            let (device, driver, _cuda) = crate::manifest::device_info(if gpu { "gpu" } else { "cpu" });
            BuildIdentity {
                version: env!("CARGO_PKG_VERSION").to_string(),
                commit: BUILD_COMMIT.to_string(),
                binary_sha256: exe_sha256(),
                device,
                driver,
                threads: std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1),
            }
        })
        .clone()
    }

    /// Every field that differs from `now`, as `name: saved -> current`.
    /// Device and driver are compared only when both runs used the GPU, so a
    /// permitted backend change does not report a device mismatch.
    pub fn differences(&self, now: &BuildIdentity) -> Vec<String> {
        let mut out = Vec::new();
        let mut cmp = |name: &str, a: &str, b: &str| {
            if a != b {
                out.push(format!("{}: saved {} -> current {}", name, a, b));
            }
        };
        cmp("version", &self.version, &now.version);
        cmp("commit", &self.commit, &now.commit);
        cmp("binary_sha256", &self.binary_sha256, &now.binary_sha256);
        if self.device != "cpu" && now.device != "cpu" {
            cmp("device", &self.device, &now.device);
            cmp("driver", &self.driver, &now.driver);
        }
        cmp(
            "threads",
            &self.threads.to_string(),
            &now.threads.to_string(),
        );
        out
    }
}

/// SHA-256 of the running executable, hashed once per process.
pub fn exe_sha256() -> String {
    static HASH: OnceLock<String> = OnceLock::new();
    HASH.get_or_init(|| {
        let exe = std::env::current_exe().unwrap_or_default();
        crate::manifest::sha256_file(&exe).unwrap_or_else(|_| "unknown".to_string())
    })
    .clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn differences_name_each_changed_field() {
        let a = BuildIdentity::current(false);
        assert!(a.differences(&a).is_empty());
        let mut b = a.clone();
        b.commit = "deadbeef".into();
        b.threads += 1;
        let d = a.differences(&b);
        assert_eq!(d.len(), 2, "{:?}", d);
        assert!(d[0].starts_with("commit: "));
        assert!(d[1].starts_with("threads: "));
    }

    #[test]
    fn device_is_ignored_across_a_backend_change() {
        let cpu = BuildIdentity::current(false);
        let mut gpu = cpu.clone();
        gpu.device = "NVIDIA".into();
        gpu.driver = "1".into();
        assert!(cpu.differences(&gpu).is_empty());
    }

    #[test]
    fn the_build_commit_is_recorded_at_build_time() {
        assert!(!BUILD_COMMIT.is_empty());
        let id = BuildIdentity::current(false);
        assert_eq!(id.commit, BUILD_COMMIT);
        assert_eq!(id.binary_sha256.len(), 64);
    }
}
