//! #1083: the search setting table. The golden tests pin the bytes the migrated
//! settings produced before the table existed: the checkpoint config text and
//! hash, their `--help` blocks, their `rcsr schema` rows, the bench knobs JSON and
//! the resume conflict lines. The table tests check every row in every reader.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

#[path = "common/scratch.rs"]
mod scratch;

use rcsr::bench::{self, SuiteConfig};
use rcsr::checkpoint::{config_canonical, config_hash, Checkpoint};
use rcsr::data::Sample;
use rcsr::schema;
use rcsr::search::{self, Config};
use rcsr::settings::{self, Group, Setting};
use std::collections::BTreeMap;

fn golden(name: &str, expected: &str, actual: &str) {
    if expected != actual {
        panic!("{name} differs from the pinned bytes; actual:\n<<<{name}\n{actual}>>>{name}\n");
    }
}

/// A command-line value off the default for every value row. A new value row
/// needs one here.
fn sample(row: &dyn Setting) -> Option<&'static str> {
    if !row.takes_value() {
        return None;
    }
    Some(match row.flag() {
        "--patience" => "7",
        "--selection" => "afp",
        "--restart-policy" => "island",
        "--restart-isolation" => "7",
        "--mutation-mix" => "extended",
        "--crossover-internal" => "0.25",
        other => panic!("no sample value for {other}: add one to tests/settings_table.rs"),
    })
}

/// Every migrated setting off its default.
fn non_default() -> Config {
    let mut cfg = Config::default();
    for row in settings::rows() {
        row.apply_cli(&mut cfg, sample(row)).expect("sample value");
    }
    cfg
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

/// The base build wrote, compared and checked `restart_isolation` only under
/// `--restart-policy island`.
#[test]
fn restart_isolation_is_read_only_under_the_island_policy() {
    let cfg = Config { restart_isolation: 7, ..Config::default() };
    golden(
        "default_config",
        include_str!("fixtures/settings/default_config.txt"),
        &config_text(&cfg),
    );
    assert!(Config::default().search_setting_differences(&cfg).is_empty());
    assert!(cfg.search_setting_differences(&Config::default()).is_empty());
    let mut named = settings::Explicit::default();
    named.mark(settings::find("--restart-isolation").expect("row"));
    let mut lines = Vec::new();
    settings::conflicts(Group::Island, &cfg, &Config::default(), &named, &mut lines);
    assert!(lines.is_empty(), "{lines:?}");
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
    for row in settings::rows().filter(|r| r.bench().is_some()) {
        cfg.knobs.settings.set(row, sample(row)).expect("sample value");
    }
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

fn help() -> String {
    let out = rcsr_cmd::rcsr().arg("--help").output().expect("run rcsr --help");
    assert!(out.status.success());
    String::from_utf8(out.stdout).expect("utf-8 help")
}

#[test]
fn help_blocks_of_migrated_settings_are_pinned() {
    let help = help();
    let pinned = blocks(include_str!("fixtures/settings/help_blocks.txt"));
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
    let pinned = blocks(include_str!("fixtures/settings/schema_rows.txt"));
    assert_eq!(pinned.len(), 5, "schema fixture blocks");
    for (prev, next, block) in &pinned {
        assert_block(&schema, prev, next, block, |flag| format!("    {{\"name\":\"{flag}"));
    }
}

fn line_csv(n: usize) -> String {
    let mut text = String::from("x,y\n");
    for i in 0..n {
        let x = -2.0 + 4.0 * i as f64 / (n - 1) as f64;
        text.push_str(&format!("{:.6},{:.6}\n", x, 2.0 * x + 1.0));
    }
    text
}

/// The resume conflict lines of every row, in the order the base build printed
/// them.
#[test]
fn resume_conflict_lines_are_pinned() {
    let dir = scratch::Scratch::new("settings-conflicts");
    let csv = dir.join("line.csv");
    std::fs::write(&csv, line_csv(40)).expect("write csv");
    let cp = dir.join("island.rcsr");
    let (csv, cp) = (csv.to_str().expect("utf-8"), cp.to_str().expect("utf-8"));
    let base = ["--data", csv, "--target", "y", "--vars", "x", "--seed", "2", "--pop", "25", "--islands", "2"];
    let made = rcsr_cmd::rcsr()
        .args(base)
        .args(["--generations", "2", "--restart-policy", "island", "--save", cp])
        .output()
        .expect("run the save");
    assert!(made.status.success(), "save run failed: {}", String::from_utf8_lossy(&made.stderr));
    let resumed = rcsr_cmd::rcsr()
        .args(base)
        .args(["--resume", cp, "--patience", "7", "--selection", "afp", "--restart-policy", "archive"])
        .args(["--restart-isolation", "7", "--mutation-mix", "extended", "--crossover-internal", "0.25"])
        .args(["--adaptive-mix", "--reject-constant-offspring", "--reject-fixed-position", "--fit-thresholds"])
        .output()
        .expect("run the resume");
    assert_eq!(resumed.status.code(), Some(2), "{}", String::from_utf8_lossy(&resumed.stderr));
    let err = String::from_utf8_lossy(&resumed.stderr);
    let lines: Vec<&str> = err.lines().filter_map(|l| l.strip_prefix("  - ")).collect();
    assert_eq!(
        lines,
        [
            "--patience 7 (saved 40)",
            "--selection afp (saved lexicographic)",
            "--restart-policy archive (saved restart_policy is island)",
            "--restart-isolation 7 (saved restart_isolation is 25)",
            "--mutation-mix extended (saved classic)",
            "--crossover-internal 0.25 (saved 0)",
            "--adaptive-mix (saved adaptive_mix is false)",
            "--reject-constant-offspring (saved reject_constant_offspring is false)",
            "--reject-fixed-position (saved reject_fixed_position is false)",
            "--fit-thresholds (saved fit_thresholds is false)",
        ],
        "{err}"
    );
}

#[test]
fn every_row_has_one_schema_entry_with_its_texts_and_default() {
    let options = schema::options();
    assert!(options.iter().all(|o| o.0.starts_with("--")), "an anchor row reached the schema");
    for group in Group::ALL {
        let anchors = schema::OPTIONS.iter().filter(|o| settings::anchored(o) == Some(group)).count();
        assert_eq!(anchors, 1, "schema::OPTIONS anchors group {} once", group.name());
    }
    for row in settings::rows() {
        let entries: Vec<_> = options.iter().filter(|o| o.0 == row.flag()).collect();
        assert_eq!(entries, [&row.schema_row()], "{}", row.flag());
        assert_eq!(row.schema_row().2, row.show(&Config::default()), "schema default of {}", row.flag());
        assert!(schema::has_option(row.flag()));
        assert_eq!(settings::find_key(row.key()).map(|r| r.flag()), Some(row.flag()));
    }
}

#[test]
fn every_row_has_its_help_block_once() {
    let help = help();
    for group in Group::ALL {
        assert!(!settings::help(group).is_empty(), "group {} has no help", group.name());
    }
    for row in settings::rows() {
        assert_eq!(help.matches(&format!("{}\n", row.help())).count(), 1, "{}", row.flag());
    }
}

fn config_fields(text: &str) -> BTreeMap<String, String> {
    text.lines()
        .filter_map(|l| l.strip_prefix("config "))
        .filter_map(|l| l.split_once(' '))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn group_text(cfg: &Config) -> String {
    let mut text = String::new();
    for group in Group::ALL {
        settings::write(group, cfg, &mut text);
    }
    text
}

#[test]
fn every_row_round_trips_and_a_missing_key_reads_its_default() {
    let off = non_default();
    let fields = config_fields(&group_text(&off));
    let mut back = Config::default();
    settings::read(&mut back, &fields).expect("read the rows");
    for row in settings::rows() {
        assert!(!row.differs(&back, &off), "{} does not round-trip", row.flag());
        assert!(fields.contains_key(row.key()), "{} is not written off its default", row.flag());
        let mut without = fields.clone();
        without.remove(row.key());
        let mut cfg = off.clone();
        match settings::read(&mut cfg, &without) {
            Ok(()) => {
                assert!(!row.required(), "{} is required but loaded without its key", row.flag());
                assert!(!row.differs(&cfg, &Config::default()), "{} missing is not its default", row.flag());
            }
            Err(e) => {
                assert!(row.required(), "{}: {e}", row.flag());
                assert!(e.contains(&format!("missing config field '{}'", row.key())), "{e}");
            }
        }
    }
    let default_fields = config_fields(&group_text(&Config::default()));
    for row in settings::rows() {
        assert_eq!(default_fields.contains_key(row.key()), row.required(), "{} at its default", row.flag());
    }
}

fn sample_rows(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        x.push(vec![a]);
        y.push(2.0 * a + 1.0);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new(), pairs: None }
}

#[test]
fn a_checkpoint_carries_every_row() {
    let cfg = Config { generations: 2, islands: 2, pop: 12, gpu: false, threads: 1, ..non_default() };
    let (train, test) = (sample_rows(60), sample_rows(20));
    let mut text = None;
    search::run_resumable(&cfg, &train, &test, None, 0, |cp| text = Some(cp.to_text())).expect("search");
    let cp = Checkpoint::from_text(&text.expect("exit checkpoint")).expect("load");
    for row in settings::rows() {
        assert!(!row.differs(&cp.config, &cfg), "{} is lost in the checkpoint", row.flag());
    }
}

#[test]
fn a_changed_row_is_a_difference_only_while_active() {
    let off = non_default();
    for row in settings::rows() {
        let mut other = off.clone();
        row.copy(&Config::default(), &mut other);
        assert!(row.active(&off), "{} is inactive in the sample config", row.flag());
        assert_eq!(off.search_setting_differences(&other), [row.key()], "{}", row.flag());
        let base = Config::default();
        if !row.active(&base) {
            let mut changed = base.clone();
            row.copy(&off, &mut changed);
            assert!(base.search_setting_differences(&changed).is_empty(), "{} while inactive", row.flag());
        }
    }
}

#[test]
fn a_named_row_is_a_resume_conflict_only_while_active() {
    let saved = non_default();
    for row in settings::rows() {
        let mut cfg = saved.clone();
        row.copy(&Config::default(), &mut cfg);
        let mut named = settings::Explicit::default();
        let mut lines = Vec::new();
        for group in Group::ALL {
            settings::conflicts(group, &cfg, &saved, &named, &mut lines);
        }
        assert!(lines.is_empty(), "{} conflicts without being named", row.flag());
        named.mark(row);
        for group in Group::ALL {
            settings::conflicts(group, &cfg, &saved, &named, &mut lines);
        }
        assert_eq!(lines.len(), 1, "{}: {lines:?}", row.flag());
        assert!(lines[0].starts_with(row.flag()), "{}", lines[0]);
        if !row.active(&Config::default()) {
            let mut quiet = Vec::new();
            let mut changed = Config::default();
            row.copy(&saved, &mut changed);
            for group in Group::ALL {
                settings::conflicts(group, &changed, &Config::default(), &named, &mut quiet);
            }
            assert!(quiet.is_empty(), "{} conflicts while inactive: {quiet:?}", row.flag());
        }
    }
}

#[test]
fn bench_rows_round_trip_through_the_knobs() {
    for row in settings::rows().filter(|r| r.bench().is_some()) {
        let mut cli = Config::default();
        row.apply_cli(&mut cli, sample(row)).expect("sample value");
        let mut knobs = settings::Overrides::default();
        knobs.set(row, sample(row)).expect("knob");
        let mut applied = Config::default();
        knobs.apply(&mut applied);
        assert!(!row.differs(&applied, &cli), "{} knob", row.flag());
        let mut json = String::new();
        knobs.json(row.group(), &mut json);
        assert!(json.contains(&format!(",\"{}\":{}", row.key(), row.json(&cli))), "{}: {json}", row.flag());
        assert!(settings::bench_usage(row.group()).contains(row.bench().expect("bench").usage));
    }
}

#[test]
fn gui_rows_parse_and_the_others_are_refused_by_name() {
    for row in settings::rows() {
        let mut args = vec![row.flag().to_string()];
        args.extend(sample(row).map(str::to_string));
        let parsed = rcsr::gui::parse_web_args(&args);
        if row.gui() {
            assert!(parsed.is_ok(), "{}: {parsed:?}", row.flag());
        } else {
            let e = parsed.expect_err("not a GUI option");
            assert!(e.contains("not supported by the web front end"), "{}: {e}", row.flag());
        }
    }
}
