/// True when `verksted/_tools/sanitize.sh` or `sanitize-all.sh` started this process under memcheck.
pub fn active() -> bool {
    std::env::var_os("RCSR_UNDER_SANITIZER").is_some()
}

/// `native` normally, `sanitized` under memcheck. Callers pass a size that still crosses
/// every chunk, tile and gap boundary the native size crosses.
pub fn pick<T>(native: T, sanitized: T) -> T {
    if active() {
        sanitized
    } else {
        native
    }
}
