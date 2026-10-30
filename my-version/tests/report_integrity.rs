//! Reporting, checkpoint and provenance integrity (issue #256).
//!
//! A report field must say what it measures, a checkpoint must describe the
//! result it belongs to, and a time-budget stop must be resumable.

use rcsr::checkpoint::{Checkpoint, ErrorKind};
use rcsr::data::Sample;
use rcsr::manifest;
use rcsr::provenance::{self, BuildIdentity};
use rcsr::search::{self, Config, Outcome};
use std::path::PathBuf;
use std::process::{Command, Output};
use std::sync::{Arc, Mutex};

fn sample(n: usize) -> Sample {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for i in 0..n {
        let a = (i as f64) / (n as f64) * 4.0 - 2.0;
        let b = (((i * 7) % n) as f64) / (n as f64) * 3.0 - 1.0;
        let noise = (((i * 37) % 11) as f64 - 5.0) * 1e-3;
        x.push(vec![a, b]);
        y.push(2.0 * a + 3.0 * b + 1.0 + noise);
    }
    Sample { x, y, w: Vec::new(), ys: Vec::new() }
}

fn config(generations: usize) -> Config {
    Config {
        generations,
        islands: 2,
        pop: 30,
        max_complexity: 10,
        fit_iters: 4,
        patience: 0,
        seed: 11,
        gpu: false,
        ..Default::default()
    }
}

fn archive_bits(out: &Outcome) -> Vec<(u64, u64, usize)> {
    out.archive
        .items
        .iter()
        .map(|s| (s.error.to_bits(), s.cost.to_bits(), s.complexity))
        .collect()
}

/// Run to completion and return the outcome plus the last checkpoint written.
fn run_saving(
    cfg: &Config,
    train: &Sample,
    test: &Sample,
    resume: Option<Checkpoint>,
) -> (Outcome, Checkpoint) {
    let last: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
    let sink = last.clone();
    let out = search::run_resumable(cfg, train, test, resume, 1, move |cp| {
        *sink.lock().unwrap() = Some(cp.to_text());
    })
    .expect("search");
    let text = last.lock().unwrap().clone().expect("a checkpoint");
    (out, Checkpoint::from_text(&text).expect("checkpoint parses"))
}

#[test]
fn a_time_budget_stop_is_not_convergence_and_is_resumable() {
    let (train, test) = (sample(120), sample(40));
    let progress: Arc<Mutex<Vec<usize>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = progress.clone();
    let mut cfg = config(8);
    cfg.time_budget = Some(0.0);
    cfg.progress = Some(Arc::new(move |p: search::Progress| {
        sink.lock().unwrap().push(p.generation);
    }));
    let (stopped, cp) = run_saving(&cfg, &train, &test, None);

    assert_eq!(stopped.stop_reason, "time_budget");
    assert_eq!(stopped.generations, 1);
    assert!(
        !stopped.converged_early,
        "a time-budget stop is not convergence"
    );
    assert!(stopped.stopped_early());
    // The terminating generation reports progress like every other one.
    assert_eq!(*progress.lock().unwrap(), vec![1]);
    // The checkpoint is not terminal.
    assert!(!cp.stopped_early);
    assert_eq!(cp.generations, 1);

    // A resume with a new budget runs the remaining generations and lands on
    // exactly the archive of an uninterrupted run.
    let mut resumed_cfg = config(8);
    resumed_cfg.time_budget = None;
    let resumed = search::run_resumable(&resumed_cfg, &train, &test, Some(cp), 0, |_| {})
        .expect("resume after a time-budget stop");
    assert_eq!(resumed.generations, 8);
    assert_eq!(resumed.stop_reason, "generations");
    let straight = search::run_resumable(&config(8), &train, &test, None, 0, |_| {}).unwrap();
    assert_eq!(archive_bits(&resumed), archive_bits(&straight));
    assert_eq!(resumed.archive.hash(), straight.archive.hash());
}

#[test]
fn the_time_budget_may_be_raised_on_resume() {
    let (train, test) = (sample(100), sample(30));
    let mut first = config(6);
    first.time_budget = Some(0.0);
    let (_, cp) = run_saving(&first, &train, &test, None);
    let mut again = config(6);
    again.time_budget = Some(3600.0);
    assert!(
        again.same_search_settings(&cp.config),
        "time_budget is not a search setting: {:?}",
        again.search_setting_differences(&cp.config)
    );
    let out = search::run_resumable(&again, &train, &test, Some(cp), 0, |_| {}).unwrap();
    assert_eq!(out.generations, 6);
}

#[test]
fn converged_early_follows_the_stop_reason_and_percent_converged() {
    let (train, test) = (sample(100), sample(30));
    let mut budget = config(10);
    budget.time_budget = Some(0.0);
    let mut patience = config(60);
    patience.patience = 2;
    for cfg in [budget, patience, config(4)] {
        let out = search::run_resumable(&cfg, &train, &test, None, 0, |_| {}).unwrap();
        let engine_stop = matches!(out.stop_reason, "solved" | "patience" | "already_stopped");
        let expected = engine_stop && out.convergence.percent_converged >= 0.75;
        assert_eq!(
            out.converged_early, expected,
            "stop_reason {} percent {}",
            out.stop_reason, out.convergence.percent_converged
        );
    }
}

#[test]
fn the_exit_checkpoint_carries_the_reported_archive() {
    let (train, test) = (sample(100), sample(30));
    let (out, cp) = run_saving(&config(5), &train, &test, None);
    let fin = cp.final_archive.as_ref().expect("a final archive block");
    assert_eq!(fin.hash(), out.archive.hash(), "the reported archive hash");
    assert_eq!(fin.items.len(), out.archive.items.len());
    for (a, b) in fin.items.iter().zip(out.archive.items.iter()) {
        assert_eq!(a.error.to_bits(), b.error.to_bits());
    }
    assert_eq!(cp.error_kind, ErrorKind::F64);
    let text = cp.to_text();
    assert!(text.contains("\nerror_kind f64\n"));
    assert!(text.contains(&format!("\nfinal {} ", fin.items.len())));
    assert!(text.contains(&format!(" {:016x}\n", out.archive.hash())));

    // The loader checks the hashes: an edited final error is refused.
    let line = text
        .lines()
        .find(|l| l.starts_with("fsol "))
        .expect("an fsol line")
        .to_string();
    let mut parts: Vec<String> = line.split(' ').map(str::to_string).collect();
    parts[3] = "0.123456789".to_string();
    let tampered = text.replacen(&line, &parts.join(" "), 1);
    let err = Checkpoint::from_text(&tampered).err().expect("tampered final");
    assert!(err.contains("final archive hash"), "{}", err);

    // The search archive is still there and resumes unchanged.
    let longer = search::run_resumable(&config(9), &train, &test, Some(cp), 0, |_| {}).unwrap();
    assert_eq!(longer.generations, 9);
}

#[test]
fn a_mid_run_checkpoint_has_no_final_block_and_is_labelled() {
    let (train, test) = (sample(100), sample(30));
    let mut texts: Vec<String> = Vec::new();
    search::run_resumable(&config(4), &train, &test, None, 1, |cp| {
        texts.push(cp.to_text());
    })
    .unwrap();
    assert!(texts.len() >= 5, "one per generation plus the exit checkpoint");
    let mid = &texts[0];
    assert!(!mid.contains("\nfinal "));
    assert!(mid.contains("\nerror_kind f64\n"));
    assert!(mid.contains("\narchive_hash "));
    assert!(texts.last().unwrap().contains("\nfinal "));
}

/// A checkpoint minus the budget line and the post-refit block, which is the
/// part that must not depend on how long the run was meant to be.
fn trajectory_text(text: &str) -> String {
    text.lines()
        .filter(|l| {
            !l.starts_with("config generations ")
                && !l.starts_with("final ")
                && !l.starts_with("fsol ")
        })
        .map(|l| format!("{}\n", l))
        .collect()
}

#[test]
fn a_prefix_run_reproduces_the_boundary_checkpoint_byte_for_byte() {
    let (train, test) = (sample(100), sample(30));
    let mut long_run: Vec<String> = Vec::new();
    search::run_resumable(&config(7), &train, &test, None, 1, |cp| {
        long_run.push(cp.to_text());
    })
    .unwrap();
    let mut prefix: Vec<String> = Vec::new();
    search::run_resumable(&config(4), &train, &test, None, 0, |cp| {
        prefix.push(cp.to_text());
    })
    .unwrap();
    // The 4-generation run's exit checkpoint is the long run's 4th boundary.
    let exit = prefix.last().expect("exit checkpoint");
    assert_eq!(trajectory_text(exit), trajectory_text(&long_run[3]));
}

#[test]
fn the_checkpoint_binds_the_build_identity() {
    let (train, test) = (sample(80), sample(20));
    let (_, cp) = run_saving(&config(2), &train, &test, None);
    let id = cp.identity.clone().expect("identity");
    assert_eq!(id, BuildIdentity::current(false));
    assert_eq!(id.commit, provenance::BUILD_COMMIT);
    assert_eq!(id.binary_sha256.len(), 64);
    let text = cp.to_text();
    for key in [
        "build version ",
        "build commit ",
        "build binary_sha256 ",
        "build device ",
        "build driver ",
        "build threads ",
    ] {
        assert!(text.contains(&format!("\n{}", key)), "missing {}", key);
    }
    assert_eq!(
        Checkpoint::from_text(&text).unwrap().identity.as_ref(),
        Some(&id)
    );

    assert!(cp.check_identity(&id, false).is_ok());
    let mut other = id.clone();
    other.binary_sha256 = "0".repeat(64);
    other.threads += 1;
    let err = cp.check_identity(&other, false).unwrap_err();
    assert!(err.contains("binary_sha256"), "{}", err);
    assert!(err.contains("threads"), "{}", err);
    assert!(err.contains("--allow-build-change"), "{}", err);
    assert!(cp.check_identity(&other, true).is_ok());
}

#[test]
fn a_checkpoint_without_identity_still_loads() {
    let (train, test) = (sample(80), sample(20));
    let (_, cp) = run_saving(&config(2), &train, &test, None);
    let stripped: String = cp
        .to_text()
        .lines()
        .filter(|l| {
            !l.starts_with("build ")
                && !l.starts_with("error_kind ")
                && !l.starts_with("archive_hash ")
                && !l.starts_with("final ")
                && !l.starts_with("fsol ")
        })
        .map(|l| format!("{}\n", l))
        .collect();
    let old = Checkpoint::from_text(&stripped).expect("an older checkpoint loads");
    assert!(old.identity.is_none());
    assert!(old.final_archive.is_none());
    assert_eq!(old.error_kind, ErrorKind::Unknown);
    let id = BuildIdentity::current(false);
    assert!(old.check_identity(&id, false).is_ok());
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn rcsr(args: &[&str], envs: &[(&str, &str)], dir: &std::path::Path) -> Output {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_rcsr"));
    cmd.args(args).current_dir(dir);
    for (k, v) in envs {
        cmd.env(k, v);
    }
    cmd.output().expect("run rcsr")
}

struct Dir(PathBuf);

impl Dir {
    fn new(tag: &str) -> Dir {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static SEQ: AtomicUsize = AtomicUsize::new(0);
        let n = SEQ.fetch_add(1, Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("rcsr-256-{}-{}-{}", std::process::id(), n, tag));
        std::fs::create_dir_all(&dir).unwrap();
        Dir(dir)
    }
    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Dir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn data_path() -> &'static str {
    concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/clean.csv")
}

fn base_args<'a>(extra: &[&'a str]) -> Vec<&'a str> {
    let mut v = vec![
        "--data",
        data_path(),
        "--target",
        "y",
        "--vars",
        "x",
        "--seed",
        "3",
        "--cpu",
        "--pop",
        "20",
        "--max-complexity",
        "8",
    ];
    v.extend_from_slice(extra);
    v
}

/// The number after `"key":` in a JSON text, as the source text.
fn number_after(text: &str, key: &str) -> String {
    let needle = format!("\"{}\":", key);
    let at = text.find(&needle).unwrap_or_else(|| panic!("no {} in {}", key, text)) + needle.len();
    text[at..]
        .chars()
        .take_while(|c| c.is_ascii_digit() || matches!(c, '.' | '-' | 'e' | 'E' | '+'))
        .collect()
}

#[test]
fn the_manifest_names_the_build_and_shares_the_runtime_with_the_result() {
    let dir = Dir::new("manifest");
    let man = dir.path("run.manifest.json");
    let out = dir.path("run.json");
    let args = base_args(&["--generations", "4", "--run-manifest", &man, "--json-out", &out, "--quiet"]);
    // The working directory is a temp dir outside any git work tree: the commit
    // can only come from the build.
    let run = rcsr(&args, &[], &dir.0);
    assert!(run.status.success(), "{}", String::from_utf8_lossy(&run.stderr));
    let m = std::fs::read_to_string(&man).unwrap();
    let r = std::fs::read_to_string(&out).unwrap();

    let commit = manifest::string_field(&m, "commit").unwrap();
    assert_eq!(commit, provenance::BUILD_COMMIT);
    let binary = manifest::object(&m, "binary").unwrap();
    assert_eq!(
        manifest::string_field(binary, "build_commit").as_deref(),
        Some(provenance::BUILD_COMMIT)
    );
    assert_eq!(
        manifest::string_field(binary, "sha256").unwrap(),
        manifest::sha256_file(env!("CARGO_BIN_EXE_rcsr")).unwrap()
    );

    // One runtime: the manifest total is the result's timing.total_seconds.
    let phases = manifest::object(&m, "phases_seconds").unwrap();
    assert_eq!(
        number_after(phases, "total"),
        number_after(&r, "total_seconds")
    );
    assert_eq!(
        number_after(phases, "search"),
        number_after(&r, "elapsed_seconds")
    );
    // The phases and the remainder add up to the runtime.
    let total: f64 = number_after(&r, "total_seconds").parse().unwrap();
    let parts: f64 = ["prep_seconds", "search_seconds", "baselines_seconds", "diagnostics_seconds", "report_seconds", "unaccounted_seconds"]
        .iter()
        .map(|k| number_after(&r, k).parse::<f64>().unwrap())
        .sum();
    assert!((parts - total).abs() < 1e-9, "{} vs {}", parts, total);
    assert!(r.contains("\"stop_reason\":\"generations\""), "{}", r);

    // start_utc is the process start, end_utc comes after the work.
    let start = manifest::string_field(&m, "start_utc").unwrap();
    let end = manifest::string_field(&m, "end_utc").unwrap();
    assert!(start <= end, "{} {}", start, end);
}

#[test]
fn verify_manifest_accepts_a_preserved_binary_by_hash() {
    let dir = Dir::new("verify");
    let man = dir.path("run.manifest.json");
    let out = dir.path("run.json");
    let args = base_args(&["--generations", "3", "--run-manifest", &man, "--json-out", &out, "--quiet"]);
    assert!(rcsr(&args, &[], &dir.0).status.success());

    let preserved = dir.path("rcsr.preserved");
    std::fs::copy(env!("CARGO_BIN_EXE_rcsr"), &preserved).unwrap();
    let ok = rcsr(&["verify-manifest", &man, "--binary", &preserved], &[], &dir.0);
    assert!(ok.status.success(), "{}", String::from_utf8_lossy(&ok.stderr));
    assert!(String::from_utf8_lossy(&ok.stdout).contains("PASS binary sha256"));

    // Any other file is not the binary that ran.
    let wrong = rcsr(&["verify-manifest", &man, "--binary", data_path()], &[], &dir.0);
    assert_eq!(wrong.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&wrong.stderr).contains("FAIL binary sha256"));
}

#[test]
fn json_out_keeps_the_human_timing_line_unless_quiet() {
    let dir = Dir::new("lines");
    let out = dir.path("run.json");
    let loud = rcsr(&base_args(&["--generations", "2", "--json-out", &out]), &[], &dir.0);
    assert!(loud.status.success());
    let err = String::from_utf8_lossy(&loud.stderr);
    assert!(err.contains("# elapsed:"), "{}", err);
    assert!(err.contains("# time:"), "{}", err);
    let quiet = rcsr(&base_args(&["--generations", "2", "--json-out", &out, "--quiet"]), &[], &dir.0);
    assert!(quiet.status.success());
    assert!(!String::from_utf8_lossy(&quiet.stderr).contains("# time:"));
}

fn saves(stderr: &[u8]) -> usize {
    String::from_utf8_lossy(stderr)
        .lines()
        .filter(|l| l.starts_with("# checkpoint: saved to"))
        .count()
}

#[test]
fn a_slow_run_checkpoints_by_wall_time_not_only_by_generation_count() {
    let dir = Dir::new("cadence");
    let ckpt = dir.path("run.ckpt");
    let result = dir.path("r.json");
    let args = base_args(&["--generations", "6", "--save", &ckpt, "--save-every", "25", "--json-out", &result]);
    let default = rcsr(&args, &[], &dir.0);
    assert!(default.status.success());
    assert_eq!(saves(&default.stderr), 1, "only the exit checkpoint");

    let timed = rcsr(&args, &[("RCSR_CHECKPOINT_INTERVAL_S", "0")], &dir.0);
    assert!(timed.status.success());
    assert!(
        saves(&timed.stderr) >= 6,
        "a checkpoint per generation when the interval has passed: {}",
        String::from_utf8_lossy(&timed.stderr)
    );
}

#[test]
fn the_cli_resumes_a_time_budget_stop_with_a_new_budget() {
    let dir = Dir::new("resume");
    let ckpt = dir.path("run.ckpt");
    let (a_json, b_json) = (dir.path("a.json"), dir.path("b.json"));
    let first = rcsr(
        &base_args(&["--generations", "6", "--time-budget", "0", "--save", &ckpt, "--json-out", &a_json]),
        &[],
        &dir.0,
    );
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    let a = std::fs::read_to_string(dir.path("a.json")).unwrap();
    assert!(a.contains("\"stop_reason\":\"time_budget\""), "{}", a);
    assert!(a.contains("\"converged_early\":false"), "{}", a);
    assert!(a.contains("\"generations_run\":1"), "{}", a);

    let second = rcsr(
        &["--data", data_path(), "--target", "y", "--vars", "x", "--cpu", "--resume", &ckpt, "--time-budget", "3600", "--json-out", &b_json, "--quiet"],
        &[],
        &dir.0,
    );
    assert!(second.status.success(), "{}", String::from_utf8_lossy(&second.stderr));
    let b = std::fs::read_to_string(dir.path("b.json")).unwrap();
    assert!(b.contains("\"generations_run\":6"), "{}", b);
    assert!(b.contains("\"stop_reason\":\"generations\""), "{}", b);
}

#[test]
fn the_cli_refuses_a_checkpoint_from_another_build_unless_allowed() {
    let dir = Dir::new("identity");
    let ckpt = dir.path("run.ckpt");
    let a_json = dir.path("a.json");
    let first = rcsr(
        &base_args(&["--generations", "2", "--save", &ckpt, "--json-out", &a_json, "--quiet"]),
        &[],
        &dir.0,
    );
    assert!(first.status.success());
    let text = std::fs::read_to_string(&ckpt).unwrap();
    let sha = text
        .lines()
        .find_map(|l| l.strip_prefix("build binary_sha256 "))
        .expect("binary sha in the checkpoint")
        .to_string();
    std::fs::write(&ckpt, text.replacen(&sha, &"f".repeat(64), 1)).unwrap();

    let sink = dir.path("c.json");
    let resume = |extra: &[&str]| {
        let mut args = vec![
            "--data", data_path(), "--target", "y", "--vars", "x", "--cpu", "--generations", "4",
            "--resume", ckpt.as_str(), "--json-out", sink.as_str(), "--quiet",
        ];
        args.extend_from_slice(extra);
        rcsr(&args, &[], &dir.0)
    };
    let refused = resume(&[]);
    assert_eq!(refused.status.code(), Some(3), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("binary_sha256"));
    let allowed = resume(&["--allow-build-change"]);
    assert!(allowed.status.success(), "{}", String::from_utf8_lossy(&allowed.stderr));
}
