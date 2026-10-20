//! Behavioural acceptance for the schema contract (issue #58).
//!
//! The old test compared two JSON documents. These tests check the thing the
//! document promises: every documented root option is recognised by the CLI,
//! listed in `--help` and the README, and is either implemented by the web
//! front end or rejected with a deliberate message (never "unknown argument").

use rcsr::gui;
use rcsr::schema;
use std::process::Command;

fn rcsr(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn every_schema_option_is_accepted_by_the_cli() {
    for (name, _, _, _) in schema::OPTIONS {
        // No value: a flag reaches "missing --data"; a value option reaches
        // "requires a value". Either way the parser must not call it unknown.
        let out = rcsr(&[name]);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("unknown argument"),
            "schema documents {name}, but the CLI rejects it: {stderr}"
        );
        assert_eq!(
            out.status.code(),
            Some(2),
            "{name} should end in a usage error without --data, got {:?}: {stderr}",
            out.status.code()
        );
    }
}

#[test]
fn every_schema_option_is_in_help_and_the_readme_list() {
    let help = rcsr(&["--help"]);
    assert!(help.status.success());
    let help = String::from_utf8_lossy(&help.stdout);
    let readme = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../README.md"),
    )
    .expect("README.md");
    let list = readme
        .split("### Full option list")
        .nth(1)
        .expect("README full option list")
        .split("```")
        .nth(1)
        .expect("README option block");
    for (name, _, _, _) in schema::OPTIONS {
        assert!(help.contains(name), "--help is missing {name}");
        assert!(
            list.contains(name),
            "README's full option list is missing {name}"
        );
    }
}

#[test]
fn every_schema_option_is_supported_or_explicitly_rejected_by_the_gui() {
    for (name, _, _, _) in schema::OPTIONS {
        match gui::parse_web_args(&[name.to_string()]) {
            Ok(()) => {}
            Err(message) => {
                assert!(
                    !message.contains("unknown argument"),
                    "the GUI calls the documented option {name} unknown: {message}"
                );
                assert!(
                    message.contains("not supported by the web front end")
                        || message.contains("requires a value")
                        || message.contains("needs")
                        || message.contains("unknown --"),
                    "{name} failed for an undocumented reason: {message}"
                );
            }
        }
    }
}

#[test]
fn subcommands_carry_their_own_documented_options() {
    let doc = schema::json();
    assert!(gui::is_valid_json(&doc), "schema is not valid JSON");
    let import = String::from_utf8_lossy(&rcsr(&["import", "--help"]).stdout).to_string();
    let verify =
        String::from_utf8_lossy(&rcsr(&["verify-manifest", "--help"]).stdout).to_string();
    let walk = String::from_utf8_lossy(&rcsr(&["walkforward", "--help"]).stdout).to_string();
    for (block, end, usage, label) in [
        ("\"import\"", "\"verify-manifest\"", import.as_str(), "import"),
        ("\"verify-manifest\"", "\"exit_codes\"", verify.as_str(), "verify-manifest"),
        ("\"walkforward\"", "\"import\"", walk.as_str(), "walkforward"),
    ] {
        let start = doc.find(block).unwrap_or_else(|| panic!("schema has no {label}"));
        let tail = &doc[start + block.len()..];
        let stop = tail.find(end).unwrap_or(tail.len());
        let section = &tail[..stop];
        assert!(!section.is_empty(), "schema has an empty {label} block");
        // Every option named in the block must be in that subcommand's help.
        for part in section.split("\"name\":\"--").skip(1) {
            let name = format!("--{}", part.split('"').next().unwrap_or(""));
            assert!(
                usage.contains(&name),
                "{label} help is missing {name} that schema documents"
            );
        }
    }
}

#[test]
fn the_option_table_has_no_duplicates_or_empty_fields() {
    let mut seen = std::collections::HashSet::new();
    for (name, ty, default, description) in schema::OPTIONS {
        assert!(name.starts_with("--"), "malformed option name {name}");
        assert!(seen.insert(*name), "duplicate option {name}");
        assert!(!ty.is_empty(), "{name} has no type");
        assert!(!default.is_empty(), "{name} has no default");
        assert!(!description.is_empty(), "{name} has no description");
        assert!(schema::has_option(name), "has_option misses {name}");
    }
    assert!(!schema::has_option("--not-a-flag"));
}
