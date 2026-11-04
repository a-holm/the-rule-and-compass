//! #495: `--refit-band` defaults to `inf`; `0.001` stays an accepted opt-in value.

use rcsr::search::Config;
use std::process::Command;

fn rcsr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().expect("run rcsr")
}

#[test]
fn refit_band_default_is_inf() {
    assert!(Config::default().refit_band.is_infinite());
    let schema = String::from_utf8(rcsr(&["schema"]).stdout).expect("utf8 schema");
    let at = schema.find("\"--refit-band\"").expect("--refit-band in schema");
    let entry = &schema[at..at + 200];
    assert!(entry.contains("\"default\":\"inf\""), "{entry}");
    let help = String::from_utf8(rcsr(&["--help"]).stdout).expect("utf8 help");
    assert!(help.contains("inf (default) refits every"), "help names the default");
}

#[test]
fn refit_band_accepts_inf_zero_and_small_values() {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    for band in ["inf", "0.001", "0"] {
        let out = rcsr(&[
            "--data", data, "--target", "y", "--vars", "x", "--seed", "3", "--islands", "1", "--pop", "20",
            "--generations", "2", "--cpu", "--quiet", "--refit-band", band,
        ]);
        assert!(out.status.success(), "{band}: {}", String::from_utf8_lossy(&out.stderr));
    }
    let bad = rcsr(&["--data", data, "--target", "y", "--vars", "x", "--refit-band", "-1"]);
    assert_eq!(bad.status.code(), Some(2));
}

fn save_checkpoint(path: &str, extra: &[&str]) {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x", "--seed", "3", "--islands", "1", "--pop", "20",
        "--generations", "2", "--cpu", "--quiet", "--save", path,
    ];
    args.extend_from_slice(extra);
    let out = rcsr(&args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
}

fn resume(path: &str, extra: &[&str]) -> std::process::Output {
    let data = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv");
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x", "--cpu", "--quiet", "--generations", "3", "--resume", path,
    ];
    args.extend_from_slice(extra);
    rcsr(&args)
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

fn scratch(tag: &str) -> (Scratch, String) {
    let scratch = Scratch::new(tag);
    let path = scratch.join("run.rcsr").to_string_lossy().into_owned();
    (scratch, path)
}

#[test]
fn fresh_checkpoint_always_carries_the_band() {
    let (_scratch, p) = scratch("fresh");
    save_checkpoint(&p, &[]);
    let text = std::fs::read_to_string(&p).expect("read");
    assert!(text.lines().any(|l| l == "config refit_band inf"), "no refit_band inf line");
    let (_scratch, q) = scratch("explicit");
    save_checkpoint(&q, &["--refit-band", "0.001"]);
    let text = std::fs::read_to_string(&q).expect("read");
    assert!(text.lines().any(|l| l.starts_with("config refit_band ") && l != "config refit_band inf"));
    let out = resume(&q, &[]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = resume(&q, &["--refit-band", "0.001"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let out = resume(&q, &["--refit-band", "inf"]);
    assert_eq!(out.status.code(), Some(2), "a different explicit band is refused");
}

#[test]
fn checkpoint_without_the_band_needs_an_explicit_flag() {
    let (_scratch, p) = scratch("legacy");
    save_checkpoint(&p, &[]);
    let text = std::fs::read_to_string(&p).expect("read");
    let kept: Vec<&str> = text.lines().filter(|l| !l.starts_with("config refit_band ")).collect();
    let block: String = kept.iter().filter(|l| l.starts_with("config ")).map(|l| format!("{l}\n")).collect();
    let hash = format!("config_hash {:016x}", rcsr::checkpoint::hash_config_text(&block));
    let stripped: String = kept
        .iter()
        .map(|l| if l.starts_with("config_hash ") { format!("{hash}\n") } else { format!("{l}\n") })
        .collect();
    std::fs::write(&p, stripped).expect("write");
    let out = resume(&p, &[]);
    assert_eq!(out.status.code(), Some(2), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--refit-band"));
    let out = resume(&p, &["--refit-band", "0.001"]);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(String::from_utf8_lossy(&out.stderr).contains("# resume: checkpoint has no saved refit band"));
}
