//! #1083: the search setting table. The golden tests pin the bytes the migrated
//! settings produced before the table existed: the checkpoint config text and
//! hash, their `--help` blocks, their `rcsr schema` rows and the bench knobs JSON.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::bench::{self, SuiteConfig};
use rcsr::checkpoint::{config_canonical, config_hash};
use rcsr::search::{self, Config};

fn golden(name: &str, expected: &str, actual: &str) {
    if expected != actual {
        panic!("{name} differs from the pinned bytes; actual:\n<<<{name}\n{actual}>>>{name}\n");
    }
}

/// Every migrated setting off its default.
fn non_default() -> Config {
    Config {
        selection: search::Selection::Afp,
        restart_policy: search::RestartPolicy::Island,
        restart_isolation: 7,
        mutation_mix: search::MutationMix::Extended,
        crossover_internal: 0.25,
        adaptive_mix: true,
        reject_constant_offspring: true,
        reject_fixed_position: true,
        fit_thresholds: true,
        patience: 7,
        ..Config::default()
    }
}

fn config_text(cfg: &Config) -> String {
    format!("{}config_hash {:016x}\n", config_canonical(cfg), config_hash(cfg))
}

#[test]
fn default_config_text_and_hash_are_pinned() {
    golden(
        "default_config",
        include_str!("fixtures/settings/default_config.txt"),
        &config_text(&Config::default()),
    );
}

#[test]
fn non_default_config_text_and_hash_are_pinned() {
    golden(
        "non_default_config",
        include_str!("fixtures/settings/non_default_config.txt"),
        &config_text(&non_default()),
    );
}

/// The `"knobs":{...}` object of the bench config JSON.
fn knobs_json(cfg: &SuiteConfig) -> String {
    let json = bench::config_json(cfg);
    let start = json.find("\"knobs\":{").expect("knobs object");
    let end = start + json[start..].find('}').expect("knobs end");
    json[start..=end].to_string()
}

fn non_default_suite() -> SuiteConfig {
    let mut cfg = SuiteConfig::default();
    let k = &mut cfg.knobs;
    k.selection = Some(search::Selection::Afp);
    k.restart_policy = Some(search::RestartPolicy::Island);
    k.restart_isolation = Some(7);
    k.mutation_mix = Some(search::MutationMix::Extended);
    k.crossover_internal = Some(0.25);
    k.adaptive_mix = true;
    k.reject_constant_offspring = true;
    k.fit_thresholds = true;
    k.patience = Some(7);
    cfg
}

#[test]
fn bench_knobs_json_is_pinned() {
    let actual = format!("{}\n{}\n", knobs_json(&SuiteConfig::default()), knobs_json(&non_default_suite()));
    golden("knobs", include_str!("fixtures/settings/knobs.txt"), &actual);
}

/// Pinned blocks: `@@ <previous flag> <next flag>` then the exact lines.
fn blocks(fixture: &str) -> Vec<(String, String, Vec<String>)> {
    let mut out: Vec<(String, String, Vec<String>)> = Vec::new();
    for line in fixture.lines() {
        if let Some(rest) = line.strip_prefix("@@ ") {
            let mut names = rest.split(' ');
            let prev = names.next().expect("previous flag").to_string();
            let next = names.next().expect("next flag").to_string();
            out.push((prev, next, Vec::new()));
        } else {
            out.last_mut().expect("a block header first").2.push(line.to_string());
        }
    }
    out
}

/// `block` appears as consecutive lines of `doc`; the option before it is
/// `prev` and the line after it starts the option `next`. `start` is the text an
/// option line starts with.
fn assert_block(doc: &str, prev: &str, next: &str, block: &[String], start: fn(&str) -> String) {
    let lines: Vec<&str> = doc.lines().collect();
    let at = (0..lines.len())
        .find(|&i| lines[i..].len() >= block.len() && lines[i..i + block.len()].iter().zip(block).all(|(a, b)| a == b))
        .unwrap_or_else(|| panic!("block after {prev} is not in the document:\n{}", block.join("\n")));
    let before = lines[..at]
        .iter()
        .rev()
        .find(|l| l.starts_with(&start("--")))
        .expect("an option before the block");
    assert!(before.starts_with(&start(prev)), "the option before the block is not {prev}: {before}");
    let after = lines.get(at + block.len()).expect("a line after the block");
    assert!(after.starts_with(&start(next)), "the option after the block is not {next}: {after}");
}

#[test]
fn help_blocks_of_migrated_settings_are_pinned() {
    let out = rcsr_cmd::rcsr().arg("--help").output().expect("run rcsr --help");
    assert!(out.status.success());
    let help = String::from_utf8(out.stdout).expect("utf-8 help");
    let fixture = include_str!("fixtures/settings/help_blocks.txt");
    let pinned = blocks(fixture);
    assert_eq!(pinned.len(), 4, "help fixture blocks");
    for (prev, next, block) in &pinned {
        assert_block(&help, prev, next, block, |flag| format!("  {flag}"));
    }
}

#[test]
fn schema_rows_of_migrated_settings_are_pinned() {
    let out = rcsr_cmd::rcsr().arg("schema").output().expect("run rcsr schema");
    assert!(out.status.success());
    let schema = String::from_utf8(out.stdout).expect("utf-8 schema");
    let fixture = include_str!("fixtures/settings/schema_rows.txt");
    let pinned = blocks(fixture);
    assert_eq!(pinned.len(), 5, "schema fixture blocks");
    for (prev, next, block) in &pinned {
        assert_block(&schema, prev, next, block, |flag| format!("    {{\"name\":\"{flag}"));
    }
}
