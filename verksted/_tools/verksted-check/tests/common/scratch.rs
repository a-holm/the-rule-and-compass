//! #731: a per-call test directory, unique, fresh and removed on drop. A sibling
//! `<dir>.owner` file is locked while its `Scratch` lives, so `sweep_dead_owners`
//! can tell the directory of a killed process from a live one.
#![allow(dead_code)]

use std::fs::{File, OpenOptions, TryLockError};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

const PREFIX: &str = "rcsr-test-";
const OWNER: &str = ".owner";

/// A directory that belongs to one caller: unique inside the process (counter),
/// across processes (pid), fresh (a stale directory from a reused pid is removed
/// first) and removed on drop. `RCSR_KEEP_TEST_DIRS=1` keeps it for debugging.
/// A test that hands a path to a child process holds the `Scratch` until the
/// child has exited.
pub struct Scratch {
    path: PathBuf,
    owner: PathBuf,
    _lock: File,
}

fn keep() -> bool {
    std::env::var_os("RCSR_KEEP_TEST_DIRS").is_some()
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
        let tmp = std::env::temp_dir();
        let name = format!("{PREFIX}{bin}-{}-{n}-{tag}", std::process::id());
        let path = tmp.join(&name);
        let owner = tmp.join(format!("{name}{OWNER}"));
        // The sweep only matches `*.owner`, so the file is locked before it can be seen.
        let pending = tmp.join(format!(".{name}{OWNER}-pending"));
        let lock = File::create(&pending).expect("create the scratch owner file");
        lock.lock().expect("lock the scratch owner file");
        std::fs::rename(&pending, &owner).expect("publish the scratch owner file");
        if path.exists() {
            std::fs::remove_dir_all(&path).expect("remove a stale scratch directory");
        }
        std::fs::create_dir(&path).expect("create the scratch directory");
        Scratch { path, owner, _lock: lock }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.path.join(name)
    }

    /// Removes the directory and its owner file; safe to call twice. With
    /// `RCSR_KEEP_TEST_DIRS` set it keeps the directory and removes only the owner
    /// file, so no sweep deletes it.
    pub fn cleanup(&self) {
        if !keep() {
            let _ = std::fs::remove_dir_all(&self.path);
        }
        let _ = std::fs::remove_file(&self.owner);
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        self.cleanup();
    }
}

/// `sweep_dead_owners_in` over the temp dir.
pub fn sweep_dead_owners() -> usize {
    sweep_dead_owners_in(&std::env::temp_dir())
}

/// Removes every `rcsr-test-*` directory in `tmp` whose owner file nobody holds
/// locked, that is, whose process ended without dropping its `Scratch`. Returns
/// how many it removed. Does nothing with `RCSR_KEEP_TEST_DIRS` set.
pub fn sweep_dead_owners_in(tmp: &Path) -> usize {
    if keep() {
        return 0;
    }
    let Ok(entries) = std::fs::read_dir(tmp) else { return 0 };
    let mut removed = 0;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let Some(dir) = name.strip_suffix(OWNER).filter(|d| d.starts_with(PREFIX)) else { continue };
        let owner = entry.path();
        let Ok(file) = OpenOptions::new().read(true).write(true).open(&owner) else { continue };
        match file.try_lock() {
            Ok(()) => {}
            Err(TryLockError::WouldBlock) | Err(TryLockError::Error(_)) => continue,
        }
        // A reused name may have been published again since `open`.
        let same = match (file.metadata(), std::fs::metadata(&owner)) {
            (Ok(a), Ok(b)) => a.dev() == b.dev() && a.ino() == b.ino(),
            _ => false,
        };
        if !same {
            continue;
        }
        let _ = std::fs::remove_dir_all(tmp.join(dir));
        let _ = std::fs::remove_file(&owner);
        removed += 1;
    }
    removed
}
