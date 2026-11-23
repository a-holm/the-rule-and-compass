//! #733: a checkpoint records the calibration source of its run, and `--resume`
//! refuses a run that would price with another one.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const DATA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");

fn run(cache: &Path, args: &[&str]) -> Output {
    rcsr_cmd::rcsr()
        .args(["--data", DATA, "--target", "y", "--vars", "x", "--cpu", "--quiet"])
        .args(args)
        .env("XDG_CACHE_HOME", cache)
        .output()
        .expect("run rcsr")
}

fn ok(out: Output) -> Output {
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    out
}

fn save(cache: &Path, path: &Path, generations: &str, extra: &[&str]) -> Output {
    let mut args = vec![
        "--seed", "3", "--islands", "1", "--pop", "20", "--generations", generations, "--save",
        path.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    ok(run(cache, &args))
}

fn resume(cache: &Path, path: &Path, extra: &[&str]) -> Output {
    let mut args = vec!["--generations", "3", "--resume", path.to_str().unwrap()];
    args.extend_from_slice(extra);
    run(cache, &args)
}

fn calibrate(cache: &Path, file: &Path) {
    let out = rcsr_cmd::rcsr()
        .args(["calibrate", "--cpu", "--no-enrich", "--out"])
        .arg(file)
        .env("XDG_CACHE_HOME", cache)
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

fn sha(path: &Path) -> String {
    rcsr::manifest::sha256_hex(&std::fs::read(path).unwrap())
}

fn lines(path: &Path) -> Vec<String> {
    std::fs::read_to_string(path).unwrap().lines().map(String::from).collect()
}

#[test]
fn a_fresh_checkpoint_carries_the_source() {
    let dir = Scratch::new("fresh");
    let file = dir.join("cal.json");
    calibrate(dir.path(), &file);

    let anchors = dir.join("anchors.rcsr");
    save(dir.path(), &anchors, "2", &[]);
    let text = lines(&anchors);
    assert!(text.contains(&"config calibration_source anchors".to_string()));
    assert!(!text.iter().any(|l| l.starts_with("config calibration_sha256")));

    let filed = dir.join("file.rcsr");
    save(dir.path(), &filed, "2", &["--calibration", file.to_str().unwrap()]);
    let text = lines(&filed);
    assert!(text.contains(&"config calibration_source file".to_string()));
    assert!(text.contains(&format!("config calibration_sha256 {}", sha(&file))));

    let probe = dir.join("probe.rcsr");
    save(dir.path(), &probe, "2", &["--calibration", "probe"]);
    assert!(lines(&probe).contains(&"config calibration_source probe".to_string()));
}

#[test]
fn a_mismatching_source_is_refused_naming_both() {
    let dir = Scratch::new("mismatch");
    let file = dir.join("cal.json");
    calibrate(dir.path(), &file);
    let file_arg = file.to_str().unwrap();

    let anchors = dir.join("anchors.rcsr");
    save(dir.path(), &anchors, "2", &[]);
    let out = resume(dir.path(), &anchors, &["--calibration", file_arg]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("anchors") && err.contains(&sha(&file)), "{err}");

    let filed = dir.join("file.rcsr");
    save(dir.path(), &filed, "2", &["--calibration", file_arg]);
    let out = resume(dir.path(), &filed, &["--calibration", "anchors"]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains(&sha(&file)) && err.contains("anchors"), "{err}");

    let other = dir.join("other.json");
    calibrate(dir.path(), &other);
    let out = resume(dir.path(), &filed, &["--calibration", other.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2), "another file is another source");

    ok(resume(dir.path(), &filed, &["--calibration", file_arg]));
}

#[test]
fn a_probe_run_resumes_only_with_the_probe_flag() {
    let dir = Scratch::new("probe");
    let probe = dir.join("probe.rcsr");
    save(dir.path(), &probe, "2", &["--calibration", "probe"]);
    let out = resume(dir.path(), &probe, &[]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("not reproducible"));
    let out = ok(resume(dir.path(), &probe, &["--calibration", "probe"]));
    assert!(String::from_utf8_lossy(&out.stderr).contains("# resume: --calibration probe"));
}

#[test]
fn a_checkpoint_without_the_source_needs_the_flag() {
    let dir = Scratch::new("legacy");
    let cp = dir.join("legacy.rcsr");
    save(dir.path(), &cp, "2", &[]);
    let kept: String = lines(&cp)
        .into_iter()
        .filter(|l| !l.starts_with("config calibration_"))
        .map(|l| format!("{l}\n"))
        .collect();
    std::fs::write(&cp, kept).unwrap();

    let out = resume(dir.path(), &cp, &[]);
    assert_eq!(out.status.code(), Some(2));
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(err.contains("--calibration"), "{err}");

    let out = ok(resume(dir.path(), &cp, &["--calibration", "anchors"]));
    assert!(String::from_utf8_lossy(&out.stderr)
        .contains("# resume: checkpoint has no saved calibration source; using --calibration anchors"));
}

fn pareto(out: &Output) -> String {
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let tail = text.split("\"pareto\":").nth(1).expect("a pareto array");
    rcsr::manifest::sha256_hex(tail.as_bytes())
}

#[test]
fn a_matching_resume_continues_bit_identically() {
    let dir = Scratch::new("identical");
    let file = dir.join("cal.json");
    calibrate(dir.path(), &file);
    let file_arg = file.to_str().unwrap();
    let common = ["--seed", "3", "--islands", "1", "--pop", "20", "--json", "--calibration", file_arg];

    let mut unbroken = vec!["--generations", "3"];
    unbroken.extend_from_slice(&common);
    let unbroken = ok(run(dir.path(), &unbroken));

    let cp = dir.join("half.rcsr");
    save(dir.path(), &cp, "2", &["--calibration", file_arg]);
    let resumed = ok(resume(dir.path(), &cp, &["--json", "--calibration", file_arg]));
    assert_eq!(pareto(&unbroken), pareto(&resumed));
}
