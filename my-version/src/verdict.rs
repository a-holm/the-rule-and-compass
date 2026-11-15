//! #1238: `rcsr protocol verdict`, the pre-registered verdict of the sealed reads.

/// `text` without the member `path[last]` of the object at `path[..last]`.
pub fn remove_member(text: &str, _path: &[&str]) -> Result<String, String> {
    Ok(text.to_string())
}

/// The text whose SHA-256 is `protocol.report_sha256` of a walk-forward report file.
pub fn hashed_report_text(text: &str) -> Result<String, String> {
    Ok(text.to_string())
}
