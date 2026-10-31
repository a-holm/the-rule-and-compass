use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

mod common;
use common::scratch;

fn fx(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/universe").join(name)
}

fn run(args: &[&str]) -> (i32, String, String) {
    let o = Command::new(env!("CARGO_BIN_EXE_rcsr-acquire")).args(args).output().expect("run binary");
    (
        o.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&o.stdout).into_owned(),
        String::from_utf8_lossy(&o.stderr).into_owned(),
    )
}

fn universe(out: &Path, extra: &[&str]) -> (i32, String, String) {
    let archive = fx("archive.txt");
    let products = fx("products.json");
    let equities = format!("{},{}", fx("nasdaq.txt").display(), fx("other.txt").display());
    let mut args = vec![
        "universe",
        "--archive",
        archive.to_str().unwrap(),
        "--products",
        products.to_str().unwrap(),
        "--equities",
        &equities,
        "--out",
        out.to_str().unwrap(),
    ];
    args.extend_from_slice(extra);
    run(&args)
}

fn read(p: &Path) -> String {
    fs::read_to_string(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))
}

fn golden(name: &str) -> String {
    read(&fx(name))
}

#[test]
fn universe_matches_the_golden_files() {
    let out = scratch("universe-golden");
    let (code, _, err) = universe(&out, &[]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(read(&out.join("universe_all_usdt_pit.txt")), golden("expected_universe.txt"));
    assert_eq!(read(&out.join("symbol_exclusions.csv")), golden("expected_exclusions.csv"));
}

#[test]
fn every_exclusion_row_names_a_rule() {
    let out = scratch("universe-rules");
    assert_eq!(universe(&out, &[]).0, 0);
    let rules = [
        "non-ascii",
        "tokenised-stock",
        "stablecoin",
        "commodity-token",
        "fiat",
        "leveraged-token",
        "stablecoin-legacy",
    ];
    let csv = read(&out.join("symbol_exclusions.csv"));
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("symbol,rule,detail"));
    let mut n = 0;
    for l in lines {
        let f: Vec<&str> = l.splitn(3, ',').collect();
        assert_eq!(f.len(), 3, "{l}");
        assert!(rules.contains(&f[1]), "unknown rule in {l}");
        assert!(!f[2].is_empty(), "empty detail in {l}");
        n += 1;
    }
    assert!(n > 0);
}

#[test]
fn two_runs_are_byte_identical() {
    let a = scratch("universe-run-a");
    let b = scratch("universe-run-b");
    assert_eq!(universe(&a, &["--top-by-quote-volume", "4", "--tickers", fx("tickers.json").to_str().unwrap()]).0, 0);
    assert_eq!(universe(&b, &["--top-by-quote-volume", "4", "--tickers", fx("tickers.json").to_str().unwrap()]).0, 0);
    for f in ["universe_all_usdt_pit.txt", "symbol_exclusions.csv", "universe_top_4_by_quote_volume.txt"] {
        assert_eq!(fs::read(a.join(f)).unwrap(), fs::read(b.join(f)).unwrap(), "{f}");
    }
}

#[test]
fn top_by_quote_volume_ranks_the_filtered_universe() {
    let out = scratch("universe-top");
    let (code, _, err) = universe(&out, &["--top-by-quote-volume", "4", "--tickers", fx("tickers.json").to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(read(&out.join("universe_top_4_by_quote_volume.txt")), golden("expected_top4.txt"));
}

#[test]
fn products_and_equities_are_optional() {
    let out = scratch("universe-no-optional");
    let archive = fx("archive.txt");
    let (code, _, err) = run(&["universe", "--archive", archive.to_str().unwrap(), "--out", out.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    let kept = read(&out.join("universe_all_usdt_pit.txt"));
    assert!(kept.lines().any(|l| l == "TSLABUSDT"), "no tag file: the name rules cannot see it");
    assert!(!kept.lines().any(|l| l == "DUSDT"), "name rules still apply");
}

#[test]
fn custom_config_replaces_the_defaults() {
    let out = scratch("universe-config");
    let cfg = out.join("config.toml");
    fs::write(&cfg, "# only fiat\nfiat = [\"EUR\"]\nstablecoins = []\nleveraged_suffixes = []\n").unwrap();
    let (code, _, err) = universe(&out, &["--config", cfg.to_str().unwrap()]);
    assert_eq!(code, 0, "{err}");
    let csv = read(&out.join("symbol_exclusions.csv"));
    assert!(csv.contains("EURUSDT,fiat,"));
    assert!(!csv.contains("USDCUSDT"));
    assert!(!csv.contains("ETHUPUSDT"));
}

#[test]
fn usage_errors_exit_2() {
    let out = scratch("universe-usage");
    let o = out.to_str().unwrap();
    let missing = out.join("absent.txt");
    let archive = fx("archive.txt");
    let a = archive.to_str().unwrap();
    for args in [
        vec!["universe", "--out", o],
        vec!["universe", "--archive", missing.to_str().unwrap(), "--out", o],
        vec!["universe", "--archive", a, "--top-by-quote-volume", "3", "--out", o],
        vec!["universe", "--archive", a, "--top-by-quote-volume", "0", "--tickers", a, "--out", o],
        vec!["universe", "--archive", a, "--config", missing.to_str().unwrap(), "--out", o],
        vec!["universe", "--archive", a, "--products", a, "--out", o],
    ] {
        let (code, _, err) = run(&args);
        assert_eq!(code, 2, "{args:?}: {err}");
        assert!(err.contains("rcsr-acquire:"), "{args:?}");
    }
}

#[test]
fn help_lists_universe() {
    let (code, out, _) = run(&["--help"]);
    assert_eq!(code, 0);
    for needle in ["rcsr-acquire universe", "--archive", "--top-by-quote-volume", "symbol_exclusions.csv"] {
        assert!(out.contains(needle), "missing {needle}");
    }
}
