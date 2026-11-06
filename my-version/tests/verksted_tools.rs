//! The workshop scripts parse and their hermetic tests pass (#667).
//! A script fails when it exits non-zero, or when its whole process tree stays blocked: every thread
//! in S, T or t and no counter moving for `idle_snapshots` snapshots in a row. A healthy script under
//! load is runnable (R) or in disk wait (D), so load never fires the guard.

use std::collections::BTreeSet;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

struct Guard {
    poll: Duration,
    snapshot_every: u32,
    idle_snapshots: u32,
}

const GATE: Guard = Guard { poll: Duration::from_millis(100), snapshot_every: 10, idle_snapshots: 30 };

#[derive(Clone, Debug, PartialEq, Eq)]
struct Task {
    pid: u32,
    tid: u32,
    state: char,
    utime: u64,
    stime: u64,
    vcsw: u64,
    nvcsw: u64,
}

/// Fields after the `(comm)` of a /proc stat line: [0] is field 3 (state), [1] ppid, [11] utime, [12] stime.
fn stat_fields(text: &str) -> Option<Vec<String>> {
    let close = text.rfind(')')?;
    Some(text.get(close + 2..)?.split_whitespace().map(str::to_string).collect())
}

/// pid and ppid of every process on the machine.
fn parents() -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    for e in fs::read_dir("/proc").into_iter().flatten().flatten() {
        let Some(pid) = e.file_name().to_str().and_then(|s| s.parse().ok()) else { continue };
        let Some(f) = fs::read_to_string(format!("/proc/{pid}/stat")).ok().as_deref().and_then(stat_fields) else {
            continue;
        };
        if let Some(ppid) = f.get(1).and_then(|s| s.parse().ok()) {
            out.push((pid, ppid));
        }
    }
    out
}

/// Every thread of `root` and its descendants, sorted by (pid, tid).
fn tree(root: u32) -> Vec<Task> {
    let pp = parents();
    let mut members = vec![root];
    let mut i = 0;
    while i < members.len() {
        let p = members[i];
        members.extend(pp.iter().filter(|(_, parent)| *parent == p).map(|(child, _)| *child));
        i += 1;
    }
    let mut out = Vec::new();
    for pid in members {
        for e in fs::read_dir(format!("/proc/{pid}/task")).into_iter().flatten().flatten() {
            let Some(tid) = e.file_name().to_str().and_then(|s| s.parse().ok()) else { continue };
            let base = format!("/proc/{pid}/task/{tid}");
            let Some(f) = fs::read_to_string(format!("{base}/stat")).ok().as_deref().and_then(stat_fields) else {
                continue;
            };
            let status = fs::read_to_string(format!("{base}/status")).unwrap_or_default();
            let count = |key: &str| {
                status.lines().find_map(|l| l.strip_prefix(key)).and_then(|v| v.trim().parse().ok()).unwrap_or(0)
            };
            let num = |i: usize| f.get(i).and_then(|s| s.parse().ok()).unwrap_or(0);
            out.push(Task {
                pid,
                tid,
                state: f.first().and_then(|s| s.chars().next()).unwrap_or('?'),
                utime: num(11),
                stime: num(12),
                vcsw: count("voluntary_ctxt_switches:"),
                nvcsw: count("nonvoluntary_ctxt_switches:"),
            });
        }
    }
    out.sort_by_key(|t| (t.pid, t.tid));
    out
}

/// No thread of the tree ran between the two snapshots, and every thread is blocked.
fn idle(prev: &[Task], cur: &[Task]) -> bool {
    !cur.is_empty() && prev == cur && cur.iter().all(|t| matches!(t.state, 'S' | 'T' | 't'))
}

fn describe(tasks: &[Task]) -> String {
    tasks
        .iter()
        .map(|t| {
            let base = format!("/proc/{}/task/{}", t.pid, t.tid);
            let read = |f: &str| fs::read_to_string(format!("{base}/{f}")).unwrap_or_default().trim().to_string();
            format!("pid {} tid {} state {} wchan {} comm {}", t.pid, t.tid, t.state, read("wchan"), read("comm"))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Runs `bash <args>` with its output in `log`. None on exit 0, else the reason.
fn run_bash(args: &[&str], log: &Path, guard: &Guard) -> Option<String> {
    let file = fs::File::create(log).expect("create log");
    let mut cmd = Command::new("bash");
    for (key, _) in std::env::vars_os() {
        if key.to_string_lossy().starts_with("RCSR_") {
            cmd.env_remove(key);
        }
    }
    // Script tests start with the machine paths pointing at nothing; a test that needs a directory sets its own.
    cmd.env("RCSR_MACHINE_DIR", "/nonexistent-machine-dir");
    cmd.env("RCSR_TEST_LOCK", "/nonexistent-machine-dir/full-suite.lock");
    let mut child = cmd
        .args(args)
        .stdin(Stdio::null())
        .stdout(file.try_clone().expect("clone log handle"))
        .stderr(file)
        .spawn()
        .expect("spawn bash");
    let (mut polls, mut streak, mut prev) = (0u32, 0u32, Vec::new());
    loop {
        if let Some(status) = child.try_wait().expect("poll child") {
            return if status.success() { None } else { Some(format!("exit {status}")) };
        }
        polls += 1;
        if polls % guard.snapshot_every == 0 {
            let cur = tree(child.id());
            streak = if idle(&prev, &cur) { streak + 1 } else { 0 };
            if streak >= guard.idle_snapshots {
                let table = describe(&cur);
                let pids: BTreeSet<String> = cur.iter().map(|t| t.pid.to_string()).collect();
                let _ = Command::new("kill").arg("-KILL").args(&pids).status();
                let _ = child.wait();
                return Some(format!(
                    "hung: every thread of the process tree blocked, nothing ran for {} snapshots\n{table}",
                    guard.idle_snapshots
                ));
            }
            prev = cur;
        }
        std::thread::sleep(guard.poll);
    }
}

fn tools_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../verksted/_tools")
}

fn shell_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let mut entries: Vec<PathBuf> =
        fs::read_dir(dir).expect("read dir").map(|e| e.expect("dir entry").path()).collect();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            if path.file_name().map(|n| n != "target").unwrap_or(true) {
                shell_files(&path, out);
            }
        } else if path.extension().map(|e| e == "sh").unwrap_or(false) {
            out.push(path);
        }
    }
}

fn tail(text: &str, n: usize) -> String {
    let lines: Vec<&str> = text.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[test]
fn tools_parse() {
    let mut files = Vec::new();
    shell_files(&tools_dir(), &mut files);
    assert!(!files.is_empty(), "no scripts under {}", tools_dir().display());
    let mut failures = Vec::new();
    for file in &files {
        let out = Command::new("bash").arg("-n").arg(file).output().expect("run bash -n");
        if !out.status.success() {
            failures.push(format!("{}:\n{}", file.display(), String::from_utf8_lossy(&out.stderr)));
        }
    }
    assert!(failures.is_empty(), "bash -n failed:\n{}", failures.join("\n"));
}

fn script_tests() -> Vec<PathBuf> {
    let mut tests: Vec<PathBuf> = fs::read_dir(tools_dir().join("tests"))
        .map(|rd| rd.map(|e| e.expect("dir entry").path()).collect())
        .unwrap_or_default();
    tests.retain(|p| {
        p.file_name().and_then(|n| n.to_str()).is_some_and(|n| n.starts_with("test-") && n.ends_with(".sh"))
    });
    tests.sort();
    tests
}

#[test]
fn tools_tests_pass() {
    let tests = script_tests();
    assert!(!tests.is_empty(), "no script tests under {}", tools_dir().display());
    let scratch = Scratch::new("logs");
    let log_dir = scratch.path();
    let mut failures = Vec::new();
    for test in &tests {
        let name = test.file_name().and_then(|n| n.to_str()).unwrap_or("test").to_string();
        let log = log_dir.join(format!("{name}.log"));
        let _ = std::io::stderr().write_all(format!("verksted_tools: start {name}\n").as_bytes());
        if let Some(why) = run_bash(&[test.to_str().expect("utf-8 path")], &log, &GATE) {
            let output = fs::read_to_string(&log).unwrap_or_default();
            failures.push(format!("{name}: {why}\n{}", tail(&output, 20)));
        }
    }
    assert!(failures.is_empty(), "script tests failed:\n{}", failures.join("\n\n"));
}

/// Every script test starts with the machine paths pointing at nothing: a test that forgets an override cannot reach the real lock.
#[test]
fn script_tests_start_with_the_machine_paths_poisoned() {
    let scratch = Scratch::new("poison");
    let log = scratch.path().join("env.log");
    let script = "printf '%s\\n%s\\n' \"${RCSR_MACHINE_DIR-unset}\" \"${RCSR_TEST_LOCK-unset}\"";
    assert_eq!(run_bash(&["-c", script], &log, &FAST), None);
    let text = fs::read_to_string(&log).expect("read env log");
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 2, "{text}");
    for line in lines {
        assert!(line != "unset" && !Path::new(line).exists(), "script tests must start with a nonexistent path, got '{line}'");
    }
}

#[test]
fn script_tests_never_reach_the_machine_directory() {
    let mut hits = Vec::new();
    for test in script_tests() {
        let text: String = fs::read_to_string(&test)
            .expect("read test")
            .lines()
            .filter(|l| !l.trim_start().starts_with('#'))
            .collect::<Vec<_>>()
            .join("\n");
        let stubs_machine = text.contains("RCSR_LOCK_SCRIPT") || text.contains("RCSR_SLOT_SCRIPT");
        if stubs_machine && !text.contains("RCSR_MACHINE_DIR") {
            hits.push(test.display().to_string());
        }
    }
    assert!(
        hits.is_empty(),
        "script tests that stub the lock or slot without RCSR_MACHINE_DIR (the real lock path is reachable):\n{}",
        hits.join("\n")
    );
}

/// No script test sleeps or reads the wall clock; tests/lib/fakes.sh holds the only timer stand-ins.
#[test]
fn tools_tests_read_no_clock() {
    const FORBIDDEN: [&str; 6] = ["sleep", "EPOCHREALTIME", "SECONDS", "read -t", "flock -w", "date +%s"];
    let mut hits = Vec::new();
    for test in script_tests() {
        let text = fs::read_to_string(&test).expect("read test");
        for (n, line) in text.lines().enumerate() {
            if line.trim_start().starts_with('#') {
                continue;
            }
            for word in FORBIDDEN {
                if has_word(line, word) {
                    hits.push(format!("{}:{}: {word}", test.display(), n + 1));
                }
            }
        }
    }
    assert!(hits.is_empty(), "wall-clock use in script tests:\n{}", hits.join("\n"));
}

fn has_word(line: &str, word: &str) -> bool {
    let w = |c: char| c.is_ascii_alphanumeric() || c == '_';
    line.match_indices(word).any(|(i, _)| {
        !line[..i].chars().next_back().is_some_and(w) && !line[i + word.len()..].chars().next().is_some_and(w)
    })
}

const FAST: Guard = Guard { poll: Duration::from_millis(10), snapshot_every: 5, idle_snapshots: 3 };

#[test]
fn guard_reports_a_deadlocked_script() {
    let scratch = Scratch::new("deadlock");
    let log = scratch.path().join("deadlock.log");
    let reason = run_bash(&["-c", "read -r _ < <(exec sleep infinity)"], &log, &FAST).expect("a hang is reported");
    assert!(reason.starts_with("hung:"), "{reason}");
    let pids: BTreeSet<&str> = reason.lines().filter_map(|l| l.strip_prefix("pid ")?.split(' ').next()).collect();
    assert_eq!(pids.len(), 2, "bash and sleep: {reason}");
}

#[test]
fn guard_lets_a_cpu_bound_script_finish() {
    let scratch = Scratch::new("cpu");
    let log = scratch.path().join("cpu.log");
    let script = "i=0; while [ \"$i\" -lt 200000 ]; do i=$((i + 1)); done";
    assert_eq!(run_bash(&["-c", script], &log, &FAST), None);
}

#[test]
fn idle_needs_every_thread_blocked_and_nothing_moving() {
    let t = |tid: u32, state: char| Task { pid: 1, tid, state, utime: 0, stime: 0, vcsw: 0, nvcsw: 0 };
    let base = vec![t(1, 'S'), t(2, 'T')];
    assert!(idle(&base, &base));
    let running = vec![t(1, 'S'), t(2, 'R')];
    assert!(!idle(&running, &running));
    let disk = vec![t(1, 'S'), t(2, 'D')];
    assert!(!idle(&disk, &disk));
    let mut moved = base.clone();
    moved[0].vcsw += 1;
    assert!(!idle(&base, &moved));
    let mut extra = base.clone();
    extra.push(t(3, 'S'));
    assert!(!idle(&base, &extra));
    assert!(!idle(&[], &[]));
}
