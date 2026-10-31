//! `rcsr-launch`: tracked multi-window launcher for long `rcsr` runs (#140).
//!
//! It follows the launcher contract in `docs/run-integrity.md`:
//!
//! * the backend is explicit (`--backend gpu|cpu`) and is reused on resume;
//! * the numeric seed is `SHA-256(label:seed)`, first 8 hex digits mod 100000;
//! * every run passes `--run-manifest`, `--json-out` and `--progress`;
//! * the final JSON is published atomically by the engine (`--json-out`);
//! * stdout and stderr go to a per-window log;
//! * the exact command line and resolved seed are written next to the outputs.
//!
//! ```text
//! rcsr-launch --data panel.rcsb --target y1 --vars close,volume \
//!   --time ts --backend gpu --seed 7 \
//!   --window 2024H1,1704067200,1719792000 \
//!   --window 2024H2,1719792000,1735689600 --out-dir runs
//! ```
//!
//! `--dry-run` prints the resolved commands without starting a search.

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

use rcsr::manifest::sha256_hex;

const USAGE: &str = "rcsr-launch: run one `rcsr` search per time window

USAGE:
  rcsr-launch --data FILE --target COL --vars A,B --backend gpu|cpu \\
    --window LABEL,START,END [--window ...] [options]

OPTIONS:
  --rcsr PATH                  path to the rcsr binary (default: rcsr)
  --data FILE                  input data (required)
  --target COL                 target column (required)
  --vars A,B                   predictor columns (required)
  --time COL                   time column (default: ts)
  --train-select TEMPLATE      expression with {start} and {train_end}
                               (default: \"ts >= 0 and ts < {train_end}\")
  --test-select TEMPLATE       expression with {start} and {end}
                               (default: \"ts >= {start} and ts < {end}\")
  --label-horizon-seconds N    removed before the test window (default: 86400)
  --backend gpu|cpu            explicit backend (required)
  --seed N                     base seed (default: 0)
  --generations N              default: 1000000
  --time-budget SECONDS        default: 0 (none)
  --save-every N               default: 25
  --out-dir DIR                default: runs
  --window LABEL,START,END     epoch seconds; repeatable (required)
  --resume                     resume each window from its checkpoint
  --allow-backend-change       pass through to rcsr
  --dry-run                    print the commands, start nothing
  -h, --help                   this text
";

struct Args {
    rcsr: String,
    data: String,
    target: String,
    vars: String,
    time: String,
    train_select: String,
    test_select: String,
    label_horizon_seconds: i64,
    backend: String,
    seed: i64,
    generations: u64,
    time_budget: f64,
    save_every: u64,
    out_dir: String,
    windows: Vec<(String, i64, i64)>,
    resume: bool,
    allow_backend_change: bool,
    dry_run: bool,
}

/// A process-independent seed: SHA-256(label:seed), first 8 hex digits.
fn stable_seed(label: &str, seed: i64) -> u32 {
    let digest = sha256_hex(format!("{label}:{seed}").as_bytes());
    u32::from_str_radix(&digest[..8], 16).expect("hex digest") % 100_000
}

fn parse_window(text: &str) -> Result<(String, i64, i64), String> {
    let parts: Vec<&str> = text.split(',').collect();
    if parts.len() != 3 {
        return Err("window must be LABEL,START,END (epoch seconds)".to_string());
    }
    if parts[0].is_empty() {
        return Err("window label is empty".to_string());
    }
    let epoch = |s: &str| -> Result<i64, String> {
        let v: f64 = s
            .trim()
            .parse()
            .map_err(|_| format!("invalid epoch seconds: {s:?}"))?;
        if !v.is_finite() {
            return Err(format!("invalid epoch seconds: {s:?}"));
        }
        Ok(v as i64)
    };
    Ok((parts[0].to_string(), epoch(parts[1])?, epoch(parts[2])?))
}

/// Fills `{name}` placeholders. `{{` and `}}` are literal braces.
fn fill(template: &str, vars: &[(&str, i64)]) -> Result<String, String> {
    let mut out = String::new();
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '{' if chars.peek() == Some(&'{') => {
                chars.next();
                out.push('{');
            }
            '}' if chars.peek() == Some(&'}') => {
                chars.next();
                out.push('}');
            }
            '{' => {
                let mut name = String::new();
                loop {
                    match chars.next() {
                        Some('}') => break,
                        Some(ch) => name.push(ch),
                        None => return Err(format!("unclosed '{{' in {template:?}")),
                    }
                }
                match vars.iter().find(|(k, _)| *k == name) {
                    Some((_, v)) => out.push_str(&v.to_string()),
                    None => return Err(format!("unknown placeholder {{{name}}} in {template:?}")),
                }
            }
            '}' => return Err(format!("single '}}' in {template:?}")),
            _ => out.push(c),
        }
    }
    Ok(out)
}

/// POSIX shell quoting, so the recorded command line can be pasted.
fn shell_quote(s: &str) -> String {
    let safe = |c: char| c.is_ascii_alphanumeric() || "@%+=:,./-_".contains(c);
    if !s.is_empty() && s.chars().all(safe) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\"'\"'"))
    }
}

fn float_text(v: f64) -> String {
    format!("{v:?}")
}

fn parse_args(argv: &[String]) -> Result<Option<Args>, String> {
    let mut a = Args {
        rcsr: "rcsr".into(),
        data: String::new(),
        target: String::new(),
        vars: String::new(),
        time: "ts".into(),
        train_select: "ts >= 0 and ts < {train_end}".into(),
        test_select: "ts >= {start} and ts < {end}".into(),
        label_horizon_seconds: 86400,
        backend: String::new(),
        seed: 0,
        generations: 1_000_000,
        time_budget: 0.0,
        save_every: 25,
        out_dir: "runs".into(),
        windows: Vec::new(),
        resume: false,
        allow_backend_change: false,
        dry_run: false,
    };
    let (mut have_data, mut have_target, mut have_vars) = (false, false, false);
    let mut i = 0;
    while i < argv.len() {
        let (flag, inline) = match argv[i].split_once('=') {
            Some((f, v)) if f.starts_with("--") => (f.to_string(), Some(v.to_string())),
            _ => (argv[i].clone(), None),
        };
        i += 1;
        match flag.as_str() {
            "-h" | "--help" => return Ok(None),
            "--resume" => a.resume = true,
            "--allow-backend-change" => a.allow_backend_change = true,
            "--dry-run" => a.dry_run = true,
            "--rcsr" | "--data" | "--target" | "--vars" | "--time" | "--train-select"
            | "--test-select" | "--label-horizon-seconds" | "--backend" | "--seed"
            | "--generations" | "--time-budget" | "--save-every" | "--out-dir" | "--window" => {
                let value = match inline {
                    Some(v) => v,
                    None => {
                        let v = argv
                            .get(i)
                            .ok_or_else(|| format!("{flag} needs a value"))?
                            .clone();
                        i += 1;
                        v
                    }
                };
                let num = |what: &str| format!("{flag}: invalid {what}: {value:?}");
                match flag.as_str() {
                    "--rcsr" => a.rcsr = value,
                    "--data" => {
                        a.data = value;
                        have_data = true;
                    }
                    "--target" => {
                        a.target = value;
                        have_target = true;
                    }
                    "--vars" => {
                        a.vars = value;
                        have_vars = true;
                    }
                    "--time" => a.time = value,
                    "--train-select" => a.train_select = value,
                    "--test-select" => a.test_select = value,
                    "--label-horizon-seconds" => {
                        a.label_horizon_seconds = value.parse().map_err(|_| num("integer"))?
                    }
                    "--backend" => {
                        if value != "gpu" && value != "cpu" {
                            return Err(format!("--backend must be gpu or cpu, got {value:?}"));
                        }
                        a.backend = value;
                    }
                    "--seed" => a.seed = value.parse().map_err(|_| num("integer"))?,
                    "--generations" => a.generations = value.parse().map_err(|_| num("integer"))?,
                    "--time-budget" => a.time_budget = value.parse().map_err(|_| num("number"))?,
                    "--save-every" => a.save_every = value.parse().map_err(|_| num("integer"))?,
                    "--out-dir" => a.out_dir = value,
                    _ => a.windows.push(parse_window(&value).map_err(|e| format!("--window: {e}"))?),
                }
            }
            other => return Err(format!("unrecognized argument: {other}")),
        }
    }
    let mut missing = Vec::new();
    if !have_data {
        missing.push("--data");
    }
    if !have_target {
        missing.push("--target");
    }
    if !have_vars {
        missing.push("--vars");
    }
    if a.backend.is_empty() {
        missing.push("--backend");
    }
    if a.windows.is_empty() {
        missing.push("--window");
    }
    if !missing.is_empty() {
        return Err(format!("missing required arguments: {}", missing.join(", ")));
    }
    Ok(Some(a))
}

struct Window {
    cmd: Vec<String>,
    manifest: String,
    result: String,
    log: String,
    command_file: String,
    seed: u32,
}

fn build_command(a: &Args, label: &str, start: i64, end: i64) -> Result<Window, String> {
    let path = |suffix: &str| {
        Path::new(&a.out_dir)
            .join(format!("{label}.{suffix}"))
            .to_string_lossy()
            .into_owned()
    };
    let checkpoint = path("ckpt");
    let manifest = path("manifest.json");
    let result = path("result.json");
    let log = path("stderr.log");
    let dev_log = path("dev.ndjson");
    let command_file = path("command.txt");
    let seed = stable_seed(label, a.seed);
    let train_end = start - a.label_horizon_seconds;
    let train = fill(&a.train_select, &[("start", start), ("train_end", train_end)])?;
    let test = fill(&a.test_select, &[("start", start), ("end", end)])?;
    let mut cmd: Vec<String> = [
        a.rcsr.as_str(),
        "--data",
        &a.data,
        "--target",
        &a.target,
        "--vars",
        &a.vars,
        "--time",
        &a.time,
        "--train-select",
        &train,
        "--test-select",
        &test,
        "--seed",
        &seed.to_string(),
        "--generations",
        &a.generations.to_string(),
        "--save",
        &checkpoint,
        "--save-every",
        &a.save_every.to_string(),
        "--run-manifest",
        &manifest,
        "--json-out",
        &result,
        "--progress",
        "--dev-log",
        &dev_log,
        "--dev-log-level",
        "summary",
        if a.backend == "gpu" { "--gpu" } else { "--cpu" },
    ]
    .iter()
    .map(|s| s.to_string())
    .collect();
    if a.time_budget != 0.0 {
        cmd.push("--time-budget".into());
        cmd.push(float_text(a.time_budget));
    }
    if a.resume {
        cmd.push("--resume".into());
        cmd.push(checkpoint);
    }
    if a.allow_backend_change {
        cmd.push("--allow-backend-change".into());
    }
    Ok(Window { cmd, manifest, result, log, command_file, seed })
}

fn run(argv: &[String]) -> Result<i32, String> {
    let a = match parse_args(argv)? {
        Some(a) => a,
        None => {
            print!("{USAGE}");
            return Ok(0);
        }
    };
    fs::create_dir_all(&a.out_dir).map_err(|e| format!("create {}: {e}", a.out_dir))?;
    let mut failed: Vec<(String, i32)> = Vec::new();
    for (label, start, end) in &a.windows {
        let w = build_command(&a, label, *start, *end)?;
        let quoted: Vec<String> = w.cmd.iter().map(|s| shell_quote(s)).collect();
        let line = format!(
            "# resolved_seed={} label={} backend={}\n{}",
            w.seed,
            label,
            a.backend,
            quoted.join(" ")
        );
        fs::write(&w.command_file, format!("{line}\n"))
            .map_err(|e| format!("write {}: {e}", w.command_file))?;
        println!("{line}");
        if a.dry_run {
            continue;
        }
        let log = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&w.log)
            .map_err(|e| format!("open {}: {e}", w.log))?;
        let err = log.try_clone().map_err(|e| format!("clone log handle: {e}"))?;
        let status = Command::new(&w.cmd[0])
            .args(&w.cmd[1..])
            .stdout(Stdio::from(log))
            .stderr(Stdio::from(err))
            .status()
            .map_err(|e| format!("start {}: {e}", w.cmd[0]))?;
        let rc = status.code().unwrap_or(-1);
        if rc != 0 {
            eprintln!("error: {label} exited {rc}; see {}", w.log);
            failed.push((label.clone(), rc));
        } else {
            println!("# result: {}", w.result);
            println!("# manifest: {}", w.manifest);
        }
        std::io::stdout().flush().ok();
    }
    if failed.is_empty() {
        return Ok(0);
    }
    for (label, rc) in &failed {
        eprintln!("failed: {label} rc={rc}");
    }
    Ok(1)
}

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    match run(&argv) {
        Ok(code) => std::process::exit(code),
        Err(e) => {
            eprintln!("rcsr-launch: error: {e}");
            std::process::exit(2);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_fips_vector() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn quoting_matches_posix_rules() {
        assert_eq!(shell_quote("a/b-c.d"), "a/b-c.d");
        assert_eq!(shell_quote("ts >= 0"), "'ts >= 0'");
        assert_eq!(shell_quote("it's"), "'it'\"'\"'s'");
        assert_eq!(shell_quote(""), "''");
    }

    #[test]
    fn templates_fill_and_reject_unknown_names() {
        let v = [("start", 5), ("end", 9)];
        assert_eq!(fill("ts >= {start} and ts < {end}", &v).unwrap(), "ts >= 5 and ts < 9");
        assert_eq!(fill("{{x}}", &v).unwrap(), "{x}");
        assert!(fill("{nope}", &v).is_err());
    }
}
