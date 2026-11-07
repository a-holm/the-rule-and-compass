//! #730: the checked-in memcheck plan names only binaries and tests that exist.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

#[path = "common/sanitizer.rs"]
mod sanitizer;
#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

const PLAN: &str = include_str!("sanitizer.plan");

struct Line {
    no: usize,
    binary: String,
    test: String,
    mode: String,
    args: String,
}

fn parse(text: &str) -> Result<Vec<Line>, String> {
    let mut out = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let no = i + 1;
        let mut it = line.split_whitespace();
        let (binary, test, mode) = match (it.next(), it.next(), it.next()) {
            (Some(b), Some(t), Some(m)) => (b, t, m),
            _ => return Err(format!("line {no}: need <binary> <test> <mode> <reason>")),
        };
        let reason = it.collect::<Vec<_>>().join(" ");
        if !["skip", "api-errors-no", "no-cuda", "no-cuda-app-only", "app-only", "run-bin"].contains(&mode) {
            return Err(format!("line {no}: unknown mode '{mode}'"));
        }
        if (mode.starts_with("no-cuda") || mode == "app-only" || mode == "run-bin") && test != "*" {
            return Err(format!("line {no}: {mode} needs '*' as the test"));
        }
        if reason.is_empty() {
            return Err(format!("line {no}: empty reason"));
        }
        out.push(Line {
            no,
            binary: binary.to_string(),
            test: test.to_string(),
            mode: mode.to_string(),
            args: reason,
        });
    }
    Ok(out)
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).expect("read dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

fn fn_names(source: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for part in source.split("fn ").skip(1) {
        let name: String = part
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() && part[name.len()..].trim_start().starts_with(['(', '<']) {
            names.insert(name);
        }
    }
    names
}

fn sources(binary: &str, root: &Path) -> Result<Vec<PathBuf>, String> {
    let mut files = Vec::new();
    if binary == "lib" || binary.starts_with("bin-") {
        if let Some(name) = binary.strip_prefix("bin-") {
            let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
            if !manifest.contains(&format!("name = \"{name}\"")) {
                return Err(format!("no [[bin]] named '{name}' in Cargo.toml"));
            }
        }
        rust_files(&root.join("src"), &mut files);
        return Ok(files);
    }
    let file = root.join("tests").join(format!("{binary}.rs"));
    if !file.is_file() {
        return Err(format!("tests/{binary}.rs does not exist"));
    }
    Ok(vec![file])
}

/// A run-bin line names a cargo bin and flags that the bin's sources parse.
fn check_run_bin(l: &Line, root: &Path) -> Vec<String> {
    let manifest = fs::read_to_string(root.join("Cargo.toml")).expect("Cargo.toml");
    if !manifest.contains(&format!("name = \"{}\"", l.binary)) {
        return vec![format!("line {}: no [[bin]] named '{}' in Cargo.toml", l.no, l.binary)];
    }
    let mut files = Vec::new();
    rust_files(&root.join("src").join("bin"), &mut files);
    let text: String = files.iter().map(|f| fs::read_to_string(f).expect("read source")).collect();
    l.args
        .split_whitespace()
        .filter(|a| a.starts_with("--") && !text.contains(&format!("\"{a}\"")))
        .map(|a| format!("line {}: '{}' does not parse the flag {a}", l.no, l.binary))
        .collect()
}

fn check(plan: &str, root: &Path) -> Vec<String> {
    let lines = match parse(plan) {
        Ok(l) => l,
        Err(e) => return vec![e],
    };
    let mut errors = Vec::new();
    for l in &lines {
        if l.mode == "run-bin" {
            errors.extend(check_run_bin(l, root));
            continue;
        }
        let files = match sources(&l.binary, root) {
            Ok(f) => f,
            Err(e) => {
                errors.push(format!("line {}: {e}", l.no));
                continue;
            }
        };
        if l.test == "*" {
            if lines.iter().any(|o| o.binary == l.binary && o.test != "*") {
                errors.push(format!("line {}: '{}' has both '*' and named lines", l.no, l.binary));
            }
            continue;
        }
        let mut names = BTreeSet::new();
        for f in &files {
            names.extend(fn_names(&fs::read_to_string(f).expect("read source")));
        }
        if !names.contains(&l.test) {
            errors.push(format!("line {}: no fn {}( in the sources of '{}'", l.no, l.test, l.binary));
            continue;
        }
        for other in &names {
            if *other != l.test && other.contains(&l.test) {
                errors.push(format!(
                    "line {}: '{}' is a substring of '{other}', so libtest --skip would match both",
                    l.no, l.test
                ));
            }
        }
    }
    errors
}

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn the_checked_in_plan_matches_the_test_sources() {
    let errors = check(PLAN, &root());
    assert!(errors.is_empty(), "{errors:#?}");
}

#[test]
fn a_plan_line_naming_a_missing_test_fails() {
    let plan = "gpu_ring no_such_test_exists api-errors-no reason\n";
    let errors = check(plan, &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("no fn no_such_test_exists("), "{errors:?}");
}

#[test]
fn a_plan_line_naming_a_missing_binary_fails() {
    let errors = check("no_such_binary * skip reason\n", &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("tests/no_such_binary.rs"), "{errors:?}");
}

#[test]
fn star_and_named_lines_for_one_binary_fail() {
    let plan = "gpu_ring * skip a\ngpu_ring stale_cuda_error_does_not_fail_the_transport skip b\n";
    let errors = check(plan, &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("both '*' and named"), "{errors:?}");
}

#[test]
fn an_empty_reason_or_unknown_mode_fails() {
    assert!(parse("gpu_ring * skip\n").is_err());
    assert!(parse("gpu_ring * maybe why\n").is_err());
    assert!(parse("gpu_ring * no-cuda\n").is_err());
}

#[test]
fn a_planned_name_that_is_a_substring_of_another_test_fails() {
    let scratch = Scratch::new("sanplan");
    let dir = scratch.path();
    fs::create_dir_all(dir.join("tests")).unwrap();
    fs::write(
        dir.join("tests/demo.rs"),
        "#[test]\nfn short() {}\n#[test]\nfn short_and_long() {}\n",
    )
    .unwrap();
    let errors = check("demo short skip why\n", &dir);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("substring"), "{errors:?}");
}

#[test]
fn pick_follows_the_marker() {
    let want = if std::env::var_os("RCSR_UNDER_SANITIZER").is_some() { 2 } else { 1 };
    assert_eq!(sanitizer::pick(1, 2), want);
    assert_eq!(sanitizer::active(), want == 2);
}

#[test]
fn no_cuda_takes_only_a_star_line() {
    assert!(parse("gpu_ring * no-cuda reason\n").is_ok());
    let Err(err) = parse("gpu_ring some_test no-cuda reason\n") else { panic!("accepted") };
    assert!(err.contains("no-cuda needs '*'"), "{err}");
}

#[test]
fn a_no_cuda_line_naming_a_missing_binary_fails() {
    let errors = check("no_such_binary * no-cuda reason\n", &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("tests/no_such_binary.rs"), "{errors:?}");
}

#[test]
fn no_cuda_app_only_takes_only_a_star_line() {
    assert!(parse("launchers * no-cuda-app-only reason\n").is_ok());
    assert!(parse("launchers some_test no-cuda-app-only reason\n").is_err());
    assert!(parse("launchers * no-cuda-app-only\n").is_err());
}

#[test]
fn app_only_takes_only_a_star_line() {
    assert!(parse("launchers * app-only reason\n").is_ok());
    assert!(parse("launchers some_test app-only reason\n").is_err());
    assert!(parse("launchers * app-only\n").is_err());
}

#[test]
fn run_bin_takes_only_a_star_line_with_arguments() {
    assert!(parse("rcsr-gpu-bench * run-bin --batch 64\n").is_ok());
    assert!(parse("rcsr-gpu-bench some_test run-bin --batch 64\n").is_err());
    assert!(parse("rcsr-gpu-bench * run-bin\n").is_err());
}

#[test]
fn a_run_bin_line_names_a_real_bin_and_flags_it_parses() {
    let errors = check("no-such-bin * run-bin --batch 64\n", &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("no [[bin]] named 'no-such-bin'"), "{errors:?}");
    let errors = check("rcsr-gpu-bench * run-bin --no-such-flag 1\n", &root());
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("does not parse the flag --no-such-flag"), "{errors:?}");
    assert!(check("rcsr-gpu-bench * run-bin --batch 64 --reps 1 --depth 3\n", &root()).is_empty());
}

#[test]
fn the_gpu_bench_binary_runs_under_memcheck_with_the_smallest_arguments() {
    let lines = parse(PLAN).expect("plan parses");
    let run: Vec<&Line> = lines.iter().filter(|l| l.binary == "rcsr-gpu-bench").collect();
    assert_eq!(run.len(), 1, "one rcsr-gpu-bench line");
    assert_eq!((run[0].mode.as_str(), run[0].args.as_str()), ("run-bin", "--batch 64 --reps 1 --depth 3"));
    let unit = lines.iter().find(|l| l.binary == "bin-rcsr-gpu-bench").expect("unit-test line");
    assert_eq!(unit.mode, "skip", "the 0-test unit binary must not be listed as a no-CUDA run");
}
