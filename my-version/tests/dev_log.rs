//! #211: the engine-level development log.
//!
//! The log is **on by default** at `summary` level — it is the in-engine
//! replacement for the old external run monitor — and `--no-dev-log` turns it
//! off. These tests drive the real binary and pin the properties the issue
//! demands:
//!
//! * the default is a summary log at `rcsr.dev.ndjson` in the expected place,
//!   and `--no-dev-log` writes nothing;
//! * logging at either level produces **identical** archive, results and
//!   configuration hash (the log is a reading, not a search input);
//! * the file is **bounded** and a killed run leaves a **valid prefix** that
//!   `rcsr dev-report` still reads;
//! * `summary` stays concise and `full` carries the complete record, both
//!   readable by the analyzer.

use rcsr::devlog::Json;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv").to_string()
}

/// A fresh directory per test, so parallel `cargo test` runs cannot collide.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-devlog-{}-{}", name, std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// The baseline command: small, deterministic, `--cpu` so the pin does not
/// depend on whether the build has CUDA or a device.
///
/// It carries `--no-dev-log` because the log is now on by default and most
/// tests below are not about it; the tests that are pass their own flags.
fn base_args(data: &str) -> Vec<String> {
    [
        "--data",
        data,
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--islands",
        "2",
        "--pop",
        "40",
        "--generations",
        "8",
        "--cpu",
        "--quiet",
        "--no-dev-log",
    ]
    .iter()
    .map(|s| s.to_string())
    .collect()
}

fn run(args: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(args)
        .output()
        .expect("run rcsr")
}

/// Run with `dir` as the working directory, so a default log lands there.
fn run_in(dir: &Path, args: &[String]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .current_dir(dir)
        .args(args)
        .output()
        .expect("run rcsr")
}

fn parse_events(path: &Path) -> Vec<Json> {
    let body = std::fs::read_to_string(path).expect("read the dev log");
    body.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| Json::parse(l).unwrap_or_else(|| panic!("line is not JSON: {}", l)))
        .collect()
}

fn events_of<'a>(events: &'a [Json], kind: &str) -> Vec<&'a Json> {
    events
        .iter()
        .filter(|e| e.get("event").and_then(|v| v.as_str()) == Some(kind))
        .collect()
}

fn f(j: &Json, key: &str) -> Option<f64> {
    j.get(key).and_then(|v| v.as_f64())
}

fn field(j: &Json, key: &str) -> String {
    j.get(key).and_then(|v| v.as_str()).unwrap_or("").to_string()
}

fn generation_numbers(events: &[Json]) -> Vec<usize> {
    events_of(events, "generation")
        .iter()
        .map(|g| g.get("generation").and_then(|v| v.as_usize()).unwrap())
        .collect()
}

/// The log is on by default: a plain run writes `rcsr.dev.ndjson` in the
/// working directory at `summary` level, with a bounded number of events.
#[test]
fn the_log_is_on_by_default_at_summary_level() {
    let dir = scratch("default");
    let data = fixture();
    let args: Vec<String> = base_args(&data)
        .into_iter()
        .filter(|a| a != "--no-dev-log")
        .collect();
    let out = run_in(&dir, &args);
    assert!(
        out.status.success(),
        "rcsr failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let path = dir.join("rcsr.dev.ndjson");
    assert!(path.exists(), "the default log was not written in the cwd");

    let events = parse_events(&path);
    assert_eq!(
        events.first().unwrap().get("event").and_then(|v| v.as_str()),
        Some("run_start")
    );
    assert_eq!(
        events.last().unwrap().get("event").and_then(|v| v.as_str()),
        Some("run_end")
    );
    let start = events_of(&events, "run_start")[0];
    assert_eq!(field(start, "level"), "summary", "summary is the default level");
    // 8 generations <= 10, so the summary cadence is one event per generation.
    assert_eq!(f(start, "every"), Some(1.0));
    assert_eq!(generation_numbers(&events), (1..=8).collect::<Vec<_>>());

    // Every generation event is a *summary* record: the monitoring fields are
    // present and the full-only fields are not.
    for g in events_of(&events, "generation") {
        assert_eq!(field(g, "level"), "summary");
        for key in ["generation", "t", "wall", "phases", "error", "archive", "fit", "gpu", "fallback", "convergence"] {
            assert!(g.get(key).is_some(), "summary event has no '{}'", key);
        }
        assert!(f(g.get("error").unwrap(), "best").is_some());
        assert!(f(g.get("error").unwrap(), "median").is_some());
        assert!(g.get("archive").unwrap().get("size").is_some());
        for key in ["variation", "islands", "rng"] {
            assert!(
                g.get(key).is_none(),
                "summary must not carry the full-only field '{}'",
                key
            );
        }
        assert!(
            g.get("archive").unwrap().get("hash").is_none(),
            "summary must not carry the archive hash"
        );
    }
}

/// `--no-dev-log` writes nothing at all.
#[test]
fn no_dev_log_writes_nothing() {
    let dir = scratch("off");
    let data = fixture();
    let mut args = base_args(&data);
    args.push("--out".into());
    args.push(dir.join("front.tsv").to_string_lossy().into_owned());
    let out = run_in(&dir, &args);
    assert!(
        out.status.success(),
        "rcsr failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let created: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(created, vec!["front.tsv".to_string()], "nothing else was written");
}

/// The default path follows the result artifacts: next to `--json-out`, else
/// next to `--save`, else the working directory.
#[test]
fn the_default_path_follows_json_out_then_save() {
    let dir = scratch("path");
    let data = fixture();
    let results = dir.join("results");
    std::fs::create_dir_all(&results).unwrap();

    // `--json-out` wins.
    let args: Vec<String> = base_args(&data)
        .into_iter()
        .filter(|a| a != "--no-dev-log")
        .chain([
            "--json-out".to_string(),
            results.join("out.json").to_string_lossy().into_owned(),
            "--save".to_string(),
            results.join("state.ckpt").to_string_lossy().into_owned(),
            "--save-every".to_string(),
            "4".to_string(),
        ])
        .collect();
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        results.join("rcsr.dev.ndjson").exists(),
        "the log must sit next to --json-out"
    );
    assert!(!dir.join("rcsr.dev.ndjson").exists());
    let events = parse_events(&results.join("rcsr.dev.ndjson"));
    assert!(
        !events_of(&events, "checkpoint").is_empty(),
        "checkpoint events are written at both levels"
    );

    // `--save` alone is enough.
    std::fs::remove_file(results.join("rcsr.dev.ndjson")).unwrap();
    let other = dir.join("state");
    std::fs::create_dir_all(&other).unwrap();
    let args: Vec<String> = base_args(&data)
        .into_iter()
        .filter(|a| a != "--no-dev-log")
        .chain([
            "--save".to_string(),
            other.join("s.ckpt").to_string_lossy().into_owned(),
        ])
        .collect();
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(
        other.join("rcsr.dev.ndjson").exists(),
        "the log must sit next to --save"
    );
}

/// A summary log keeps the cadence small: `clamp(generations/10, 10, 25)`.
#[test]
fn the_summary_cadence_is_bounded() {
    let dir = scratch("cadence");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--generations".into());
    args.push("200".into());
    // The cadence is a property of the budget, so the run must not stop early.
    args.push("--patience".into());
    args.push("0".into());
    args.push("--pop".into());
    args.push("8".into());
    args.push("--dev-log".into());
    args.push(dir.join("run.ndjson").to_string_lossy().into_owned());
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let events = parse_events(&dir.join("run.ndjson"));
    let start = events_of(&events, "run_start")[0];
    assert_eq!(f(start, "every"), Some(20.0), "200/10 = 20");
    let numbers = generation_numbers(&events);
    assert_eq!(numbers, (1..=10).map(|i| i * 20).collect::<Vec<_>>());
}

/// The core guarantee: with the log on — at either level — the front, the JSON
/// result and the canonical configuration hash are byte-identical to the same
/// run with it off. Only wall-clock timings may differ, so they are removed
/// before comparing.
#[test]
fn logging_on_or_off_leaves_the_archive_results_and_config_hash_identical() {
    let dir = scratch("identical");
    let data = fixture();
    let mut outputs = Vec::new();
    let arms: [(&str, Vec<String>); 3] = [
        ("off", vec!["--no-dev-log".to_string()]),
        ("summary", vec![]),
        ("full", vec!["--dev-log-level".to_string(), "full".to_string()]),
    ];
    for (name, extra) in arms {
        let mut args = base_args(&data);
        args.retain(|a| a != "--no-dev-log");
        args.extend(extra);
        args.push("--out".into());
        args.push(dir.join(format!("{}.tsv", name)).to_string_lossy().into_owned());
        args.push("--json-out".into());
        args.push(dir.join(format!("{}.json", name)).to_string_lossy().into_owned());
        args.push("--run-manifest".into());
        args.push(
            dir.join(format!("{}.manifest", name))
                .to_string_lossy()
                .into_owned(),
        );
        // An explicit path keeps the logged arms inside this test's directory.
        if name != "off" {
            args.push("--dev-log".into());
            args.push(dir.join(format!("{}.ndjson", name)).to_string_lossy().into_owned());
        }
        let out = run_in(&dir, &args);
        assert!(
            out.status.success(),
            "rcsr failed: {}",
            String::from_utf8_lossy(&out.stderr)
        );
        let tsv = std::fs::read(dir.join(format!("{}.tsv", name))).unwrap();
        let json = std::fs::read_to_string(dir.join(format!("{}.json", name))).unwrap();
        let manifest = std::fs::read_to_string(dir.join(format!("{}.manifest", name))).unwrap();
        outputs.push((name.to_string(), tsv, json, manifest));
    }
    // `elapsed_seconds` and every `*_seconds` in `timing` are wall clock; the
    // rest of the JSON document must be identical.
    let strip = |text: &str| -> Vec<String> {
        let v = Json::parse(text).expect("the JSON result parses");
        let mut fields = match v {
            Json::Obj(f) => f,
            _ => panic!("the JSON result is an object"),
        };
        fields.retain(|(k, _)| k != "elapsed_seconds" && k != "timing" && k != "baselines");
        fields
            .into_iter()
            .map(|(k, v)| format!("{:?}", (k, v)))
            .collect()
    };
    let config_of = |manifest: &str| -> String {
        manifest
            .lines()
            .find(|l| l.trim_start().starts_with("\"config\":"))
            .expect("a config line in the manifest")
            .to_string()
    };
    let (off_name, off_tsv, off_json, off_manifest) = &outputs[0];
    assert_eq!(off_name, "off");
    for (name, tsv, json, manifest) in &outputs[1..] {
        assert_eq!(off_tsv, tsv, "the {} run changed the reported front", name);
        assert_eq!(strip(off_json), strip(json), "the {} run changed the JSON result", name);
        assert_eq!(
            config_of(off_manifest),
            config_of(manifest),
            "the {} run changed the configuration hash",
            name
        );
    }
    assert!(config_of(off_manifest).contains("\"hash\":"));
    // The two logged arms really are a summary and a full log.
    let summary = parse_events(&dir.join("summary.ndjson"));
    let full = parse_events(&dir.join("full.ndjson"));
    assert_eq!(field(events_of(&summary, "run_start")[0], "level"), "summary");
    assert_eq!(field(events_of(&full, "run_start")[0], "level"), "full");
    assert!(events_of(&summary, "generation")[0].get("variation").is_none());
    assert!(events_of(&full, "generation")[0].get("variation").is_some());
}

/// Every documented event and field of a `full` log is present, with the
/// documented types.
#[test]
fn the_full_log_emits_the_documented_events_and_fields() {
    let dir = scratch("fields");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--dev-log-level".into());
    args.push("full".into());
    args.push("--dev-log".into());
    args.push(dir.join("run.ndjson").to_string_lossy().into_owned());
    args.push("--save".into());
    args.push(dir.join("state.ckpt").to_string_lossy().into_owned());
    args.push("--save-every".into());
    args.push("4".into());
    let out = run_in(&dir, &args);
    assert!(
        out.status.success(),
        "rcsr failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );

    let events = parse_events(&dir.join("run.ndjson"));
    for e in &events {
        assert_eq!(
            e.get("schema").and_then(|v| v.as_str()),
            Some("rcsr-devlog/1"),
            "every event carries the schema"
        );
        assert_eq!(f(e, "v"), Some(1.0));
    }
    assert_eq!(
        events.first().unwrap().get("event").and_then(|v| v.as_str()),
        Some("run_start")
    );
    assert_eq!(
        events.last().unwrap().get("event").and_then(|v| v.as_str()),
        Some("run_end")
    );

    let start = events_of(&events, "run_start")[0];
    for key in [
        "command",
        "version",
        "commit",
        "dirty",
        "binary_sha256",
        "data",
        "config",
        "config_hash",
        "seed",
        "backend",
        "device",
        "threads",
        "host",
        "start",
        "level",
        "every",
        "max_bytes",
    ] {
        assert!(start.get(key).is_some(), "run_start has no '{}'", key);
    }
    assert_eq!(field(start, "level"), "full");
    let config = start.get("config").expect("config object");
    assert_eq!(f(config, "pop"), Some(40.0), "the resolved config is recorded");
    assert_eq!(f(config, "seed"), Some(3.0));

    let gens = events_of(&events, "generation");
    assert_eq!(gens.len(), 8, "full writes every generation");
    let g = gens[3];
    for key in [
        "generation",
        "total",
        "t",
        "wall",
        "phases",
        "error",
        "archive",
        "variation",
        "gpu",
        "fit",
        "fallback",
        "qd",
        "convergence",
        "islands",
        "rng",
    ] {
        assert!(g.get(key).is_some(), "generation has no '{}'", key);
    }
    assert_eq!(field(g, "level"), "full");
    let phases = g.get("phases").unwrap();
    for key in ["prep", "search", "eval", "select", "archive", "fit_cpu", "eval_cpu"] {
        assert!(f(phases, key).is_some(), "phases has no '{}'", key);
    }
    let err = g.get("error").unwrap();
    for key in ["best", "p05", "p25", "median", "p75", "p95"] {
        assert!(f(err, key).is_some(), "error has no '{}'", key);
    }
    let archive = g.get("archive").unwrap();
    for key in ["size", "hypervolume", "hash", "offers", "admissions"] {
        assert!(archive.get(key).is_some(), "archive has no '{}'", key);
    }
    assert!(
        archive
            .get("hash")
            .and_then(|v| v.as_str())
            .unwrap()
            .starts_with("0x"),
        "the archive hash is a hex string"
    );
    let var = g.get("variation").unwrap();
    for key in [
        "attempts",
        "duplicates",
        "too_complex",
        "offspring",
        "fitted",
        "refits",
        "dedup_rate",
    ] {
        assert!(var.get(key).is_some(), "variation has no '{}'", key);
    }
    let fit = g.get("fit").unwrap();
    for key in ["attempts", "gpu", "fallback", "refits", "fallback_by_cause"] {
        assert!(fit.get(key).is_some(), "fit has no '{}'", key);
    }
    let conv = g.get("convergence").unwrap();
    for key in ["confidence", "stability", "maturity", "percent_converged"] {
        assert!(f(conv, key).is_some(), "convergence has no '{}'", key);
    }
    assert_eq!(
        match g.get("islands") {
            Some(Json::Arr(items)) => items.len(),
            other => panic!("islands is an array, got {:?}", other),
        },
        2,
        "one entry per island"
    );
    assert_eq!(generation_numbers(&events), (1..=8).collect::<Vec<_>>());

    let cps = events_of(&events, "checkpoint");
    assert!(
        cps.len() >= 2,
        "save-every 4 plus the exit snapshot, got {}",
        cps.len()
    );
    for cp in &cps {
        assert!(cp.get("bytes").and_then(|v| v.as_usize()).unwrap() > 0);
        assert!(cp.get("hash").and_then(|v| v.as_str()).unwrap().starts_with("0x"));
    }

    let end = events.last().unwrap();
    for key in ["generations", "stop", "wall", "search", "archive", "json", "peak_rss_bytes"] {
        assert!(end.get(key).is_some(), "run_end has no '{}'", key);
    }
    let stop = end.get("stop").and_then(|v| v.as_str()).unwrap();
    assert!(
        ["solved", "patience", "front_stable", "time_budget", "generations", "already_stopped"].contains(&stop),
        "unknown stop reason '{}'",
        stop
    );
}

/// `--dev-log-every N` overrides the level's cadence and keeps every Nth
/// generation and nothing else.
#[test]
fn dev_log_every_overrides_the_cadence_at_both_levels() {
    let dir = scratch("every");
    let data = fixture();
    for level in ["summary", "full"] {
        let mut args = base_args(&data);
        args.retain(|a| a != "--no-dev-log");
        args.push("--dev-log".into());
        args.push(dir.join(format!("{}.ndjson", level)).to_string_lossy().into_owned());
        args.push("--dev-log-level".into());
        args.push(level.into());
        args.push("--dev-log-every".into());
        args.push("3".into());
        let out = run_in(&dir, &args);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let events = parse_events(&dir.join(format!("{}.ndjson", level)));
        assert_eq!(generation_numbers(&events), vec![3, 6], "level {}", level);
    }
}

/// The byte cap bounds the generation stream, and `run_end` is always written.
#[test]
fn the_byte_cap_bounds_the_file_and_the_end_event_always_lands() {
    let dir = scratch("cap");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--dev-log-level".into());
    args.push("full".into());
    args.push("--dev-log".into());
    args.push(dir.join("run.ndjson").to_string_lossy().into_owned());
    args.push("--dev-log-max-bytes".into());
    args.push("4KiB".into());
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let path = dir.join("run.ndjson");
    let size = std::fs::metadata(&path).unwrap().len();
    // 4 KiB of generation events plus the two unbounded terminal events.
    assert!(
        size < 4096 + 4096,
        "the log grew past the cap: {} bytes",
        size
    );
    let events = parse_events(&path);
    let dropped = events
        .last()
        .unwrap()
        .get("dropped")
        .and_then(|v| v.as_usize())
        .unwrap();
    assert!(dropped > 0, "the cap must be reported in run_end");
    assert!(
        events_of(&events, "generation").len() < 8,
        "the cap must drop some generation events"
    );
    assert_eq!(
        events.first().unwrap().get("event").and_then(|v| v.as_str()),
        Some("run_start")
    );
}

/// A killed run leaves complete lines; `dev-report` reads the prefix, says how
/// many lines it skipped and still ranks the bottlenecks.
#[test]
fn a_killed_run_leaves_a_valid_prefix_that_dev_report_reads() {
    let dir = scratch("killed");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--dev-log-level".into());
    args.push("full".into());
    args.push("--dev-log".into());
    args.push(dir.join("run.ndjson").to_string_lossy().into_owned());
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let path = dir.join("run.ndjson");
    let full = std::fs::read(&path).unwrap();
    // Cut inside the last event, as a SIGKILL during the write would.
    std::fs::write(&path, &full[..full.len() - 12]).unwrap();

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["dev-report", path.to_str().unwrap()])
        .output()
        .expect("run dev-report");
    assert!(
        out.status.success(),
        "dev-report failed on a torn log: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("1 line(s) were skipped"), "{}", text);
    assert!(text.contains("## Bottleneck ranking"));
    assert!(
        text.contains("## Where the generation wall went"),
        "the prefix is still a usable log"
    );
}

/// The analyzer turns a real run into the summary the issue asks for: phases,
/// fit/eval share, fallback causes, convergence, archive growth and a ranked
/// bottleneck list.
#[test]
fn dev_report_summarises_the_run_and_ranks_bottlenecks() {
    let dir = scratch("report");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--dev-log-level".into());
    args.push("full".into());
    args.push("--dev-log".into());
    args.push(dir.join("run.ndjson").to_string_lossy().into_owned());
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["dev-report", dir.join("run.ndjson").to_str().unwrap()])
        .output()
        .expect("run dev-report");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    for section in [
        "## Run",
        "## Where the generation wall went",
        "## Fit vs evaluation",
        "## Variation effort",
        "## Fallback causes",
        "## Convergence",
        "## Archive growth",
        "## Bottleneck ranking",
        "## Other findings",
    ] {
        assert!(text.contains(section), "missing section {}", section);
    }
    // The ranking is ordered by measured time, largest first.
    let ranking: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with("1. **") || l.starts_with("2. **"))
        .collect();
    assert_eq!(ranking.len(), 2, "at least two ranked bottlenecks");
    let seconds = |line: &str| -> f64 {
        line.split('—')
            .nth(1)
            .and_then(|rest| rest.trim().split(' ').next())
            .and_then(|v| v.parse().ok())
            .unwrap_or(-1.0)
    };
    assert!(
        seconds(ranking[0]) >= seconds(ranking[1]),
        "the bottleneck ranking is not ordered: {:?}",
        ranking
    );
    // The run is described honestly: seed, backend and stop reason are all read
    // back from the log, not invented.
    assert!(text.contains("| seed | 3 |"), "{}", text);
    assert!(text.contains("| backend | cpu |"), "{}", text);
    assert!(
        text.contains("| dedup rate |"),
        "a full log carries the variation counters"
    );
}

/// The analyzer reads the default (summary) log too, says what the level left
/// out, and still ranks the bottlenecks.
#[test]
fn dev_report_reads_a_summary_log_and_names_what_it_omits() {
    let dir = scratch("report-summary");
    let data = fixture();
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--generations".into());
    args.push("40".into());
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let log = dir.join("rcsr.dev.ndjson");

    let out = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["dev-report", log.to_str().unwrap()])
        .output()
        .expect("run dev-report");
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(text.contains("`summary` level"), "{}", text);
    assert!(text.contains("This is a **summary** log"), "{}", text);
    assert!(
        text.contains("Not recorded at `summary` level"),
        "the report must name what the level omits: {}",
        text
    );
    assert!(text.contains("only in `--dev-log-level full`"));
    assert!(text.contains("## Bottleneck ranking"));
    assert!(text.contains("| seed | 3 |"));
    assert!(!text.contains("NaN"), "no NaN in a report: {}", text);
}

/// No raw rows: the log records counts, hashes and error statistics, never an
/// input or target value.
#[test]
fn the_log_contains_no_raw_data_rows() {
    let dir = scratch("norows");
    let data = dir.join("rows.csv");
    // Distinctive values so a leak is visible in a string search.
    let mut csv = String::from("a,b,y\n");
    for i in 0..40 {
        let a = 1234.5678 + i as f64;
        let b = 98765.4321 - i as f64;
        csv.push_str(&format!("{},{},{}\n", a, b, 2.0 * a + b));
    }
    std::fs::write(&data, csv).unwrap();
    let args = vec![
        "--data".to_string(),
        data.to_string_lossy().into_owned(),
        "--target".into(),
        "y".into(),
        "--vars".into(),
        "a,b".into(),
        "--seed".into(),
        "1".into(),
        "--generations".into(),
        "4".into(),
        "--pop".into(),
        "30".into(),
        "--cpu".into(),
        "--quiet".into(),
        "--dev-log-level".into(),
        "full".into(),
        "--dev-log".into(),
        dir.join("run.ndjson").to_string_lossy().into_owned(),
    ];
    let out = run_in(&dir, &args);
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    let body = std::fs::read_to_string(dir.join("run.ndjson")).unwrap();
    for needle in ["1234.5678", "98765.4321", "a,b,y"] {
        assert!(
            !body.contains(needle),
            "the log leaked raw data ('{}'): {}",
            needle,
            &body[..body.len().min(400)]
        );
    }
    // The data file appears only as its path, SHA-256 and sample fingerprints.
    let events = parse_events(&dir.join("run.ndjson"));
    let start = events_of(&events, "run_start")[0];
    let data_obj = start.get("data").unwrap();
    assert!(data_obj.get("path").is_some());
    assert!(data_obj.get("sha256").is_some());
}

/// The log never touches stdout: a `--json` result stays one JSON object and an
/// `--ndjson` stream never carries a `rcsr-devlog` line.
#[test]
fn the_log_is_independent_of_the_result_streams() {
    let dir = scratch("streams");
    let data = fixture();
    for format in ["--json", "--ndjson"] {
        let mut args = base_args(&data);
        args.retain(|a| a != "--no-dev-log");
        args.push(format.into());
        let b = run_in(&dir, &args);
        assert!(b.status.success());
        let sb = String::from_utf8_lossy(&b.stdout).into_owned();
        assert!(
            !sb.contains("rcsr-devlog"),
            "the development log leaked onto stdout"
        );
        if format == "--json" {
            Json::parse(&sb).expect("--json still prints one object");
        }
        // The default log file is written in the working directory.
        assert!(dir.join("rcsr.dev.ndjson").exists());
    }
}

/// Flag misuse is a usage error, not a silent no-op.
#[test]
fn dev_log_flag_misuse_is_a_usage_error() {
    let data = fixture();
    // `--no-dev-log` contradicts every flag that configures the log.
    for extra in [
        vec!["--dev-log", "/tmp/x.ndjson"],
        vec!["--dev-log-level", "full"],
        vec!["--dev-log-every", "2"],
        vec!["--dev-log-max-bytes", "1MiB"],
    ] {
        let mut args = base_args(&data);
        args.extend(extra.iter().map(|s| s.to_string()));
        let out = run(&args);
        assert_eq!(out.status.code(), Some(2), "expected a usage error for {:?}", extra);
    }
    // An unknown level is refused.
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--dev-log-level".into());
    args.push("verbose".into());
    let out = run(&args);
    assert_eq!(out.status.code(), Some(2));

    // Logging an --eval-expr run is refused: there is no search to log.
    let mut args = base_args(&data);
    args.retain(|a| a != "--no-dev-log");
    args.push("--eval-expr".into());
    args.push("x".into());
    args.push("--dev-log".into());
    args.push("/tmp/never.ndjson".into());
    let out = run(&args);
    assert_eq!(out.status.code(), Some(2));

    // dev-report needs a readable log.
    let out = run(&["dev-report".to_string()]);
    assert_eq!(out.status.code(), Some(2));
    let out = run(&["dev-report".to_string(), "/nonexistent/log.ndjson".to_string()]);
    assert_eq!(out.status.code(), Some(3));
}

/// #599: the data digest in the run record equals the plain file hash, is
/// cached beside the data file, and a changed file invalidates the cache.
#[test]
fn data_digest_is_cached_beside_the_file_and_follows_changes() {
    let dir = scratch("digest");
    let data = dir.join("data.csv");
    std::fs::copy(fixture(), &data).expect("copy fixture");
    let sidecar = dir.join("data.csv.sha256.json");
    let args: Vec<String> = base_args(data.to_str().unwrap())
        .into_iter()
        .filter(|a| a != "--no-dev-log")
        .collect();
    let digest = |dir: &Path| -> String {
        let _ = std::fs::remove_file(dir.join("rcsr.dev.ndjson"));
        let out = run_in(dir, &args);
        assert!(out.status.success(), "rcsr failed: {}", String::from_utf8_lossy(&out.stderr));
        let events = parse_events(&dir.join("rcsr.dev.ndjson"));
        field(events_of(&events, "run_start")[0].get("data").unwrap(), "sha256")
    };

    let first = digest(&dir);
    assert_eq!(first, rcsr::manifest::sha256_file(&data).unwrap());
    assert!(sidecar.exists(), "the first run must leave a digest cache");
    assert_eq!(digest(&dir), first, "a cached digest must equal the computed one");

    let mut bytes = std::fs::read(&data).unwrap();
    bytes.extend_from_slice(b"9,9\n");
    std::fs::write(&data, &bytes).unwrap();
    let changed = digest(&dir);
    assert_eq!(changed, rcsr::manifest::sha256_file(&data).unwrap());
    assert_ne!(changed, first, "a changed file must not reuse the cache");
}
