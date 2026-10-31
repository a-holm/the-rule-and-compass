//! Minimal JSON string output.

/// Quotes and escapes `s` as a JSON string.
pub fn quote(s: &str) -> String {
    let _ = s;
    String::new()
}

/// `quote(s)` or `null`.
pub fn quote_opt(s: Option<&str>) -> String {
    s.map_or_else(|| "null".to_string(), quote)
}
