//! #731: a per-call test directory, unique, fresh and removed on drop.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// A directory that belongs to one caller: unique inside the process (counter),
/// across processes (pid), fresh (a stale directory from a reused pid is removed
/// first) and removed on drop. `RCSR_KEEP_TEST_DIRS=1` keeps it for debugging.
/// A test that hands a path to a child process holds the `Scratch` until the
/// child has exited.
pub struct Scratch {
    path: PathBuf,
}

impl Scratch {
    pub fn new(tag: &str) -> Scratch {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let bin = std::env::current_exe()
            .ok()
            .and_then(|p| p.file_stem().map(|s| s.to_string_lossy().into_owned()))
            .unwrap_or_default();
        let bin = bin.rsplit_once('-').map_or(bin.as_str(), |(name, _)| name);
        let path = std::env::temp_dir()
            .join(format!("rcsr-test-{bin}-{}-{n}-{tag}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("remove a stale scratch directory");
        }
        std::fs::create_dir(&path).expect("create the scratch directory");
        Scratch { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var_os("RCSR_KEEP_TEST_DIRS").is_none() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
