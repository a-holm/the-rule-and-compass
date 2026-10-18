// CUDA architecture names: parsing and toolkit requirements.
//
// This file is shared deliberately:
//
// * `build.rs` pulls it in with `include!` so the build script can validate
//   `RCSR_CUDA_ARCH` and explain a mismatch *before* dumping nvcc output;
// * `src/gpu.rs` declares it as an ordinary module, so `cargo test --features
//   cuda` exercises the parser with real unit tests.
//
// It has no dependencies beyond `core`, which keeps it usable from both.

/// A parsed `sm_<major><minor>[a]` target.
///
/// A trailing `a` marks an architecture-specific ("accelerated") target such as
/// `sm_90a`; nvcc treats those as a distinct target, not as the base one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Arch {
    pub major: u32,
    pub minor: u32,
    pub suffix_a: bool,
}

impl Arch {
    /// The canonical spelling, e.g. `sm_86` or `sm_121a`.
    pub fn name(&self) -> String {
        format!(
            "sm_{}{}{}",
            self.major,
            self.minor,
            if self.suffix_a { "a" } else { "" }
        )
    }
}

/// Parse an `RCSR_CUDA_ARCH` value.
///
/// Accepted grammar: `sm_` followed by two or three ASCII digits (the last is
/// the minor version) and an optional trailing `a`:
///
/// * `sm_86`   -> major 8, minor 6
/// * `sm_90a`  -> major 9, minor 0, accelerated
/// * `sm_121`  -> major 12, minor 1
/// * `sm_121a` -> major 12, minor 1, accelerated
///
/// Everything else is rejected with an actionable message. Leading/trailing
/// whitespace is ignored so a copy-pasted environment value still works.
pub fn parse_arch(s: &str) -> Result<Arch, String> {
    let s = s.trim();
    let rest = s
        .strip_prefix("sm_")
        .ok_or_else(|| format!("`{}` must start with `sm_` (for example `sm_86` or `sm_121`)", s))?;
    let (digits, suffix_a) = match rest.strip_suffix('a') {
        Some(d) => (d, true),
        None => (rest, false),
    };

    if digits.len() < 2 || digits.len() > 3 {
        return Err(format!(
            "`{}` has {} digit(s) after `sm_`; expected two or three (`sm_86`, `sm_121`)",
            s,
            digits.len()
        ));
    }
    if !digits.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!(
            "`{}` contains a non-digit after `sm_` (expected e.g. `sm_86` or `sm_121a`)",
            s
        ));
    }
    if digits.starts_with('0') {
        return Err(format!(
            "`{}` has a leading zero; write the major version without padding (e.g. `sm_86`)",
            s
        ));
    }

    let (major_digits, minor_digit) = digits.split_at(digits.len() - 1);
    let major: u32 = major_digits
        .parse()
        .map_err(|_| format!("`{}` has an unreadable major version", s))?;
    let minor: u32 = minor_digit
        .parse()
        .map_err(|_| format!("`{}` has an unreadable minor version", s))?;

    Ok(Arch {
        major,
        minor,
        suffix_a,
    })
}

/// The oldest CUDA toolkit that can compile for this target, as
/// `(major, minor)`, or `None` when the mapping is unknown.
///
/// The values are the toolkit releases that introduced each compute capability;
/// they are deliberately conservative (an older toolkit cannot compile the
/// target at all). `sm_121` is the GB10 in the DGX Spark; CUDA 12.8 added the
/// Blackwell family, and the DGX OS images ship CUDA 13.
pub fn minimum_cuda(a: Arch) -> Option<(u32, u32)> {
    match (a.major, a.minor) {
        (0, _) => None,
        (1..=7, _) => Some((9, 0)),
        (8, 0) => Some((11, 0)),
        (8, 6) => Some((11, 1)),
        (8, 7) => Some((11, 4)),
        (8, 9) => Some((11, 8)),
        (9, 0) => {
            if a.suffix_a {
                Some((12, 0))
            } else {
                Some((11, 8))
            }
        }
        (10, _) | (12, _) => Some((12, 8)),
        _ => None,
    }
}

/// Read the toolkit version out of `nvcc --version` output.
///
/// Matches the `release X.Y,` field, so it works with both the English and the
/// localised line layout (the release field itself is stable).
pub fn parse_nvcc_version(text: &str) -> Option<(u32, u32)> {
    let at = text.find("release ")? + "release ".len();
    let version = text[at..].split([',', ' ', '\n', '\r']).next()?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next().unwrap_or("0").parse().ok()?;
    Some((major, minor))
}

/// `found >= required`, comparing major then minor.
pub fn version_at_least(found: (u32, u32), required: (u32, u32)) -> bool {
    found.0 > required.0 || (found.0 == required.0 && found.1 >= required.1)
}

/// Format a `(major, minor)` toolkit version as `X.Y`.
pub fn format_version(v: (u32, u32)) -> String {
    format!("{}.{}", v.0, v.1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_architectures() {
        let cases = [
            ("sm_86", 8, 6, false),
            ("sm_80", 8, 0, false),
            ("sm_90", 9, 0, false),
            ("sm_90a", 9, 0, true),
            ("sm_100", 10, 0, false),
            ("sm_121", 12, 1, false),
            ("sm_121a", 12, 1, true),
            ("  sm_86  ", 8, 6, false),
        ];
        for (text, major, minor, suffix_a) in cases {
            let a = parse_arch(text).unwrap_or_else(|e| panic!("{}: {}", text, e));
            assert_eq!(
                (a.major, a.minor, a.suffix_a),
                (major, minor, suffix_a),
                "{}",
                text
            );
            assert_eq!(a.name(), text.trim(), "round-trip for {}", text);
        }
    }

    #[test]
    fn rejects_malformed_architectures() {
        for bad in [
            "", "86", "sm", "sm_", "sm_a", "sm_8", "sm_1234", "sm_8x", "sm_086", "sm-86",
            "sm_86b", "compute_86", "sm__86",
        ] {
            assert!(parse_arch(bad).is_err(), "`{}` should be rejected", bad);
        }
    }

    #[test]
    fn reports_minimum_toolkit_per_architecture() {
        let sm86 = parse_arch("sm_86").unwrap();
        let sm90 = parse_arch("sm_90").unwrap();
        let sm90a = parse_arch("sm_90a").unwrap();
        let sm121 = parse_arch("sm_121").unwrap();

        assert_eq!(minimum_cuda(sm86), Some((11, 1)));
        assert_eq!(minimum_cuda(sm90), Some((11, 8)));
        assert_eq!(minimum_cuda(sm90a), Some((12, 0)));
        assert_eq!(minimum_cuda(sm121), Some((12, 8)));
        assert_eq!(minimum_cuda(parse_arch("sm_100").unwrap()), Some((12, 8)));
    }

    #[test]
    fn parses_nvcc_version_lines() {
        let cuda12 = "nvcc: NVIDIA (R) Cuda compiler driver\n\
                      Copyright (c) 2005-2023 NVIDIA Corporation\n\
                      Built on Fri_Jan__6_16:45:21_PST_2023\n\
                      Cuda compilation tools, release 12.0, V12.0.140\n\
                      Build cuda_12.0.r12.0/compiler.32267302_0";
        assert_eq!(parse_nvcc_version(cuda12), Some((12, 0)));

        let cuda13 = "Cuda compilation tools, release 13.0, V13.0.48";
        assert_eq!(parse_nvcc_version(cuda13), Some((13, 0)));
        assert_eq!(parse_nvcc_version("no version here"), None);
    }

    #[test]
    fn compares_versions() {
        assert!(version_at_least((12, 0), (11, 8)));
        assert!(version_at_least((12, 8), (12, 8)));
        assert!(!version_at_least((12, 0), (12, 8)));
        assert!(!version_at_least((11, 9), (12, 0)));
        assert_eq!(format_version((12, 8)), "12.8");
    }
}
