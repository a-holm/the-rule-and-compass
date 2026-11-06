//! #775: a per-call test directory for the unit tests under `src/`, unique (pid and
//! counter), fresh and removed on drop. `RCSR_KEEP_TEST_DIRS` keeps it for debugging.
//! Same contract as `tests/common/scratch.rs`, which `src/` cannot reach.

use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

pub(crate) struct Scratch {
    path: PathBuf,
}

impl Scratch {
    pub(crate) fn new(tag: &str) -> Scratch {
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("rcsr-test-lib-{}-{n}-{tag}", std::process::id()));
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("remove a stale scratch directory");
        }
        std::fs::create_dir_all(&path).expect("create the scratch directory");
        Scratch { path }
    }
}

impl Deref for Scratch {
    type Target = Path;
    fn deref(&self) -> &Path {
        &self.path
    }
}

impl AsRef<Path> for Scratch {
    fn as_ref(&self) -> &Path {
        &self.path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        if std::env::var_os("RCSR_KEEP_TEST_DIRS").is_none() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }
}
