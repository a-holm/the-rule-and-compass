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
