//! Crash-safe file publication (issue #140).
//!
//! A long run may be killed at any moment. Every file the engine publishes —
//! checkpoints, run manifests, JSON results and fronts — is therefore written
//! to a sibling temporary file, flushed with `fsync`, and renamed over the
//! destination. A reader never observes a half-written file.
//!
//! For checkpoints the previous good generation is retained as `<path>.prev`.
//! The retention uses a hard link where the filesystem supports it (so the old
//! inode survives the rename) and a copy otherwise.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Deliberate write failures used by the fault-injection tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    /// Write the complete file and publish it.
    None,
    /// Write only the first half, sync it, and return an error without
    /// touching the destination (a killed process looks like this).
    Partial,
}

fn sibling(path: &Path, suffix: &str) -> PathBuf {
    let mut p = path.as_os_str().to_owned();
    p.push(suffix);
    PathBuf::from(p)
}

/// Write `bytes` to `path` atomically.
///
/// The temporary file is `<path>.tmp`; the previous generation (if any) is
/// retained as `<path>.prev`. On error the destination is left untouched (the
/// temporary file may remain, and is overwritten by the next attempt).
pub fn write_atomic(path: impl AsRef<Path>, bytes: &[u8], fault: Fault) -> Result<(), String> {
    let path = path.as_ref();
    let tmp = sibling(path, ".tmp");
    let parent = path.parent().filter(|p| !p.as_os_str().is_empty());

    {
        let mut file = std::fs::File::create(&tmp)
            .map_err(|e| format!("cannot create temporary file '{}': {}", tmp.display(), e))?;
        let (slice, injected) = match fault {
            Fault::None => (bytes, false),
            Fault::Partial => (&bytes[..bytes.len() / 2], true),
        };
        file.write_all(slice)
            .map_err(|e| format!("cannot write temporary file '{}': {}", tmp.display(), e))?;
        file.sync_all()
            .map_err(|e| format!("cannot fsync temporary file '{}': {}", tmp.display(), e))?;
        if injected {
            return Err(format!(
                "injected partial write to '{}' ({} of {} bytes); destination left intact",
                tmp.display(),
                slice.len(),
                bytes.len()
            ));
        }
    }

    // Keep the previous good generation. A hard link is atomic and preserves
    // the old inode even when the destination is replaced below.
    if path.exists() {
        let prev = sibling(path, ".prev");
        let _ = std::fs::remove_file(&prev);
        if std::fs::hard_link(path, &prev).is_err() {
            std::fs::copy(path, &prev).map_err(|e| {
                format!(
                    "cannot retain previous generation '{}': {}",
                    prev.display(),
                    e
                )
            })?;
        }
        if let Ok(f) = std::fs::File::open(&prev) {
            let _ = f.sync_all();
        }
    }

    std::fs::rename(&tmp, path).map_err(|e| {
        format!(
            "cannot publish '{}' (temporary file '{}'): {}",
            path.display(),
            tmp.display(),
            e
        )
    })?;

    // Make the directory entry durable too.
    if let Some(dir) = parent {
        if let Ok(f) = std::fs::File::open(dir) {
            let _ = f.sync_all();
        }
    }
    Ok(())
}

/// Atomically publish a UTF-8 text file.
pub fn write_text_atomic(path: impl AsRef<Path>, text: &str) -> Result<(), String> {
    write_atomic(path, text.as_bytes(), Fault::None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "rcsr-atomic-{}-{}-{}",
            std::process::id(),
            name,
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        p
    }

    #[test]
    fn a_partial_write_leaves_the_previous_generation_intact() {
        let path = temp("partial");
        write_text_atomic(&path, "generation one").unwrap();
        let err = write_atomic(&path, b"generation two", Fault::Partial).unwrap_err();
        assert!(err.contains("partial"), "{}", err);
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "generation one");
        write_text_atomic(&path, "generation two").unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "generation two");
        assert_eq!(
            std::fs::read_to_string(sibling(&path, ".prev")).unwrap(),
            "generation one"
        );
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(sibling(&path, ".prev"));
        let _ = std::fs::remove_file(sibling(&path, ".tmp"));
    }
}
