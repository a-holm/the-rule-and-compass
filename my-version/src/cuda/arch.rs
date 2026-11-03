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
/// target at all). CUDA 12.8 added Blackwell (`sm_100`, `sm_120`);
/// `sm_121` — the GB10 in the DGX Spark — first appears in CUDA 12.9, and
/// 12.8's `nvcc --list-gpu-arch` stops at `compute_120`. The DGX OS images
/// ship CUDA 13.
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
        (10, _) | (12, 0) => Some((12, 8)),
        (12, _) => Some((12, 9)),
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


/// The CUDA major version the build prefers when it finds several toolkits.
pub const PREFERRED_CUDA_MAJOR: u32 = 13;

/// True when a toolkit of version `v` can compile for `a`. CUDA 13.0 removed
/// offline compilation for Maxwell, Pascal and Volta (`sm_50` to `sm_72`).
pub fn toolkit_supports(a: Arch, v: (u32, u32)) -> bool {
    let new_enough = minimum_cuda(a).map_or(true, |m| version_at_least(v, m));
    new_enough && !arch_dropped(a, v)
}

/// True when `a` is an architecture that toolkit `v` no longer compiles for.
pub fn arch_dropped(a: Arch, v: (u32, u32)) -> bool {
    let dropped = a.major < 7 || (a.major == 7 && a.minor < 5);
    dropped && v.0 >= 13
}

/// Where the chosen `nvcc` came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NvccSource {
    Override,
    Preferred,
    Path,
}

impl NvccSource {
    pub fn name(self) -> &'static str {
        match self {
            NvccSource::Override => "RCSR_NVCC",
            NvccSource::Preferred => "cuda-13",
            NvccSource::Path => "PATH",
        }
    }
}

/// The toolkit selection rule: a non-empty override wins; else the newest
/// preferred-major toolkit that can target `arch`; else `nvcc` on `PATH`.
/// `preferred` holds each discovered `/usr/local/cuda-<major>.<minor>/bin/nvcc`
/// with the version its `--version` reported, so this function stays pure.
pub fn choose_nvcc(
    override_nvcc: Option<&str>,
    preferred: &[(String, (u32, u32))],
    arch: Arch,
) -> (String, NvccSource) {
    if let Some(p) = override_nvcc.map(str::trim).filter(|p| !p.is_empty()) {
        return (p.to_string(), NvccSource::Override);
    }
    preferred
        .iter()
        .filter(|(_, v)| v.0 == PREFERRED_CUDA_MAJOR && toolkit_supports(arch, *v))
        .max_by(|a, b| a.1.cmp(&b.1).then_with(|| b.0.cmp(&a.0)))
        .map(|(p, _)| (p.clone(), NvccSource::Preferred))
        .unwrap_or_else(|| ("nvcc".to_string(), NvccSource::Path))
}

/// The full toolkit version from the `V<major>.<minor>.<patch>` token of
/// `nvcc --version`, for example `13.0.88`; falls back to `X.Y` from the
/// release field.
pub fn parse_nvcc_build_version(text: &str) -> Option<String> {
    for token in text.split(|c: char| c.is_whitespace() || c == ',') {
        let Some(v) = token.strip_prefix('V') else { continue };
        let parts: Vec<&str> = v.split('.').collect();
        if parts.len() == 3 && parts.iter().all(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())) {
            return Some(v.to_string());
        }
    }
    parse_nvcc_version(text).map(format_version)
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
        assert_eq!(minimum_cuda(sm121), Some((12, 9)));
        assert_eq!(minimum_cuda(parse_arch("sm_121a").unwrap()), Some((12, 9)));
        assert_eq!(minimum_cuda(parse_arch("sm_100").unwrap()), Some((12, 8)));
        assert_eq!(minimum_cuda(parse_arch("sm_120").unwrap()), Some((12, 8)));
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

    fn tk(path: &str, major: u32, minor: u32) -> (String, (u32, u32)) {
        (path.to_string(), (major, minor))
    }

    fn pick(over: Option<&str>, found: &[(String, (u32, u32))], arch: &str) -> (String, NvccSource) {
        choose_nvcc(over, found, parse_arch(arch).unwrap())
    }

    #[test]
    fn toolkit_supports_follows_minimum_and_dropped_architectures() {
        let s = |a: &str, v: (u32, u32)| toolkit_supports(parse_arch(a).unwrap(), v);
        for v in [(12, 0), (12, 9), (13, 0), (13, 4), (14, 0)] {
            assert!(s("sm_86", v), "sm_86 on {:?}", v);
            assert!(s("sm_75", v), "sm_75 on {:?}", v);
        }
        assert!(s("sm_70", (12, 0)) && s("sm_70", (12, 9)));
        assert!(!s("sm_70", (13, 0)) && !s("sm_70", (13, 4)) && !s("sm_70", (14, 0)));
        assert!(!s("sm_121", (12, 0)) && s("sm_121", (12, 9)) && s("sm_121", (13, 0)));
        assert!(arch_dropped(parse_arch("sm_70").unwrap(), (13, 0)));
        assert!(!arch_dropped(parse_arch("sm_70").unwrap(), (12, 9)));
        assert!(!arch_dropped(parse_arch("sm_75").unwrap(), (13, 0)));
    }

    #[test]
    fn choose_nvcc_empty_override_is_unset() {
        let found = [tk("/usr/local/cuda-13.0/bin/nvcc", 13, 0)];
        let (p, src) = pick(Some(""), &found, "sm_86");
        assert_eq!((p.as_str(), src), ("/usr/local/cuda-13.0/bin/nvcc", NvccSource::Preferred));
        let (p, src) = pick(Some("  "), &found, "sm_86");
        assert_eq!((p.as_str(), src), ("/usr/local/cuda-13.0/bin/nvcc", NvccSource::Preferred));
    }

    #[test]
    fn choose_nvcc_override_wins_over_a_present_13() {
        let found = [tk("/usr/local/cuda-13.0/bin/nvcc", 13, 0)];
        let (p, src) = pick(Some("/usr/bin/nvcc"), &found, "sm_86");
        assert_eq!((p.as_str(), src), ("/usr/bin/nvcc", NvccSource::Override));
    }

    #[test]
    fn choose_nvcc_falls_back_to_path() {
        let (p, src) = pick(None, &[], "sm_86");
        assert_eq!((p.as_str(), src), ("nvcc", NvccSource::Path));
        let only14 = [tk("/usr/local/cuda-14.0/bin/nvcc", 14, 0)];
        let (p, src) = pick(None, &only14, "sm_86");
        assert_eq!((p.as_str(), src), ("nvcc", NvccSource::Path));
        let only12 = [tk("/usr/local/cuda-12.6/bin/nvcc", 12, 6)];
        let (p, src) = pick(None, &only12, "sm_86");
        assert_eq!((p.as_str(), src), ("nvcc", NvccSource::Path));
    }

    #[test]
    fn choose_nvcc_takes_the_newest_13() {
        let found = [
            tk("/usr/local/cuda-13.0/bin/nvcc", 13, 0),
            tk("/usr/local/cuda-13.4/bin/nvcc", 13, 4),
            tk("/usr/local/cuda-13.1/bin/nvcc", 13, 1),
            tk("/usr/local/cuda-14.0/bin/nvcc", 14, 0),
        ];
        let (p, src) = pick(None, &found, "sm_86");
        assert_eq!((p.as_str(), src), ("/usr/local/cuda-13.4/bin/nvcc", NvccSource::Preferred));
        let (p, _) = pick(None, &[found[1].clone(), found[0].clone()], "sm_86");
        assert_eq!(p, "/usr/local/cuda-13.4/bin/nvcc");
    }

    #[test]
    fn choose_nvcc_skips_13_for_a_dropped_architecture() {
        let found = [tk("/usr/local/cuda-13.0/bin/nvcc", 13, 0)];
        let (p, src) = pick(None, &found, "sm_70");
        assert_eq!((p.as_str(), src), ("nvcc", NvccSource::Path));
    }

    #[test]
    fn choose_nvcc_prefers_13_for_sm121() {
        let found = [tk("/usr/local/cuda-13.0/bin/nvcc", 13, 0)];
        let (p, src) = pick(None, &found, "sm_121");
        assert_eq!((p.as_str(), src), ("/usr/local/cuda-13.0/bin/nvcc", NvccSource::Preferred));
    }

    #[test]
    fn parses_the_full_toolkit_version() {
        let cuda12 = "Cuda compilation tools, release 12.0, V12.0.140\nBuild cuda_12.0.r12.0/compiler.32267302_0";
        assert_eq!(parse_nvcc_build_version(cuda12).as_deref(), Some("12.0.140"));
        let cuda13 = "Cuda compilation tools, release 13.0, V13.0.88\nBuild cuda_13.0.r13.0/compiler.36424714_0";
        assert_eq!(parse_nvcc_build_version(cuda13).as_deref(), Some("13.0.88"));
        assert_eq!(
            parse_nvcc_build_version("Cuda compilation tools, release 13.0, odd").as_deref(),
            Some("13.0")
        );
        assert_eq!(parse_nvcc_build_version("nothing"), None);
    }
}
