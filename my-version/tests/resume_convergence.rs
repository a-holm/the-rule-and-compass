//! #1348: the `convergence` block and `converged_early` of a resumed run equal
//! those of the uninterrupted run, byte for byte, through the CLI.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const GENERATIONS: &str = "24";
const CUTS: [usize; 4] = [1, 5, 12, 23];

struct Lcg(u64);

impl Lcg {
    fn unit(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / (1u64 << 53) as f64
    }
}

/// One noisy series: a front that keeps moving, so every reading differs by generation.
fn series_csv() -> String {
    let mut rng = Lcg(7);
    let mut text = String::from("x1,x2,y\n");
    for _ in 0..90 {
        let (a, b) = (rng.unit() * 4.0 - 2.0, rng.unit() * 4.0 - 2.0);
        let y = (1.7 * a).sin() + 0.4 * a * b + 0.3 * (rng.unit() - 0.5);
        text.push_str(&format!("{a:.6},{b:.6},{y:.6}\n"));
    }
    text
}

/// Four symbols over 40 time steps each.
fn panel_csv() -> String {
    let mut rng = Lcg(11);
    let mut text = String::from("t,sym,x1,x2,y\n");
    for s in 0..4 {
        for t in 0..40 {
            let (a, b) = (rng.unit() * 4.0 - 2.0, rng.unit() * 4.0 - 2.0);
            let y = (1.3 * a).sin() - 0.5 * b * a + 0.1 * s as f64 + 0.3 * (rng.unit() - 0.5);
            text.push_str(&format!("{},S{s},{a:.6},{b:.6},{y:.6}\n", 1_700_000_000 + 900 * t));
        }
    }
    text
}

#[derive(Clone, Copy)]
enum Fixture {
    Series,
    Panel,
    PanelLexicase,
}

fn base_args(fixture: Fixture, extra: &[&str]) -> Vec<String> {
    let mut args: Vec<&str> = vec![
        "--target", "y", "--vars", "x1,x2", "--seed", "3", "--pop", "40", "--islands", "1", "--threads", "1",
        "--max-complexity", "10", "--patience", "0", "--cpu", "--no-dev-log", "--quiet",
    ];
    match fixture {
        Fixture::Series => args.extend(["--data", "d.csv"]),
        Fixture::Panel => args.extend(["--data", "d.csv", "--group-by", "sym", "--time", "t", "--split", "temporal"]),
        Fixture::PanelLexicase => args.extend([
            "--data", "d.csv", "--group-by", "sym", "--time", "t", "--split", "temporal", "--selection", "lexicase",
            "--variation", "node",
        ]),
    }
    args.extend(extra);
    args.into_iter().map(str::to_string).collect()
}

fn rcsr_in(dir: &Path, args: &[String]) -> Output {
    let refs: Vec<&str> = args.iter().map(String::as_str).collect();
    rcsr_cmd::rcsr().current_dir(dir).args(&refs).output().expect("run rcsr")
}

fn ok(out: Output, what: &str) {
    assert!(out.status.success(), "{what} failed: {}", String::from_utf8_lossy(&out.stderr));
}

fn run(dir: &Path, fixture: Fixture, extra: &[&str], what: &str) {
    ok(rcsr_in(dir, &base_args(fixture, extra)), what);
}

fn block(json: &str, key: &str) -> String {
    let at = json.find(&format!("\"{key}\":")).unwrap_or_else(|| panic!("no {key} in {json}"));
    let rest = &json[at..];
    let end = rest.find('}').expect("closing brace");
    rest[..=end].to_string()
}

/// The two readings the issue names, as written to run.json.
fn readings(dir: &Path, name: &str) -> (String, String) {
    let json = std::fs::read_to_string(dir.join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    let early = json.split("\"converged_early\":").nth(1).expect("converged_early");
    (block(&json, "convergence"), early.split(',').next().unwrap().to_string())
}

fn setup(tag: &str, fixture: Fixture) -> Scratch {
    let scratch = Scratch::new(tag);
    let csv = match fixture {
        Fixture::Series => series_csv(),
        _ => panel_csv(),
    };
    std::fs::write(scratch.join("d.csv"), csv).expect("csv");
    scratch
}

fn check(fixture: Fixture, tag: &str) {
    let scratch = setup(tag, fixture);
    let dir: &Path = scratch.path();
    run(dir, fixture, &["--generations", GENERATIONS, "--json-out", "full.json"], "full run");
    let full = readings(dir, "full.json");
    assert!(full.0.contains("\"confidence\":"), "{}", full.0);

    for cut in CUTS {
        let cp = format!("cut{cut}.rcsr");
        let json = format!("cut{cut}.json");
        run(dir, fixture, &["--generations", &cut.to_string(), "--save", &cp], "cut run");
        run(dir, fixture, &["--generations", GENERATIONS, "--resume", &cp, "--json-out", &json], "resume");
        assert_eq!(readings(dir, &json), full, "resume at generation {cut}");
    }

    // A chain of resumes, each saving for the next.
    run(dir, fixture, &["--generations", "1", "--save", "chain.rcsr"], "chain 1");
    for upto in ["5", "12"] {
        run(dir, fixture, &["--generations", upto, "--resume", "chain.rcsr", "--save", "chain.rcsr"], "chain step");
    }
    run(dir, fixture, &["--generations", GENERATIONS, "--resume", "chain.rcsr", "--json-out", "chain.json"], "chain end");
    assert_eq!(readings(dir, "chain.json"), full, "chained resume");

    // One-generation chunks, each interim until the budget is spent.
    let mut resume: Vec<&str> = Vec::new();
    for chunk in 0..(2 * GENERATIONS.parse::<usize>().unwrap()) {
        let mut extra = vec!["--generations", GENERATIONS, "--save", "ck.rcsr", "--time-budget", "0", "--interim-chunk", "--json-out", "ck.json"];
        extra.extend(&resume);
        run(dir, fixture, &extra, "interim chunk");
        resume = vec!["--resume", "ck.rcsr"];
        let json = std::fs::read_to_string(dir.join("ck.json")).expect("ck.json");
        if !json.contains("\"interim\":true") {
            assert!(chunk >= 1, "the chain ran more than one chunk");
            break;
        }
    }
    let last = std::fs::read_to_string(dir.join("ck.json")).expect("ck.json");
    assert!(last.contains("\"pareto\":["), "the chunk chain ended with a full report");
    assert_eq!(readings(dir, "ck.json"), full, "interim chunk chain");
}

#[test]
fn a_resumed_series_run_reports_the_uninterrupted_convergence_block() {
    check(Fixture::Series, "resume-conv-series");
}

#[test]
fn a_resumed_panel_run_reports_the_uninterrupted_convergence_block() {
    check(Fixture::Panel, "resume-conv-panel");
}

#[test]
fn a_resumed_lexicase_panel_run_reports_the_uninterrupted_convergence_block() {
    check(Fixture::PanelLexicase, "resume-conv-lexicase");
}

/// `converged_early` needs an engine stop (`patience` or `solved`) and a
/// converged block; a resume must land on the same answer as the full run.
#[test]
fn a_resumed_run_that_stops_on_patience_reports_the_same_converged_early() {
    let scratch = Scratch::new("resume-conv-patience");
    let dir: &Path = scratch.path();
    let csv: String = std::iter::once("x1,x2,y\n".to_string())
        .chain((0..60).map(|i| {
            let x = -2.0 + 4.0 * i as f64 / 59.0;
            format!("{x:.6},{:.6},{:.6}\n", x * 0.5, 2.0 * x + 1.0)
        }))
        .collect();
    std::fs::write(dir.join("d.csv"), csv).expect("csv");
    let args = |extra: &[&str]| -> Vec<String> {
        let mut a: Vec<&str> = vec![
            "--data", "d.csv", "--target", "y", "--vars", "x1,x2", "--seed", "5", "--pop", "30", "--islands", "1",
            "--threads", "1", "--max-complexity", "8", "--cpu", "--no-dev-log", "--quiet", "--generations", "60",
            "--patience", "6",
        ];
        a.extend(extra);
        a.into_iter().map(str::to_string).collect()
    };
    ok(rcsr_in(dir, &args(&["--json-out", "full.json"])), "full run");
    let full = std::fs::read_to_string(dir.join("full.json")).unwrap();
    assert!(full.contains("\"stop_reason\":\"patience\""), "the fixture stops on patience: {}", block(&full, "convergence"));
    let want = readings(dir, "full.json");
    for cut in [2usize, 5] {
        ok(rcsr_in(dir, &args(&["--generations", &cut.to_string(), "--save", "cp.rcsr"])), "cut");
        ok(rcsr_in(dir, &args(&["--resume", "cp.rcsr", "--json-out", "res.json"])), "resume");
        let res = std::fs::read_to_string(dir.join("res.json")).unwrap();
        assert!(res.contains("\"stop_reason\":\"patience\""));
        assert_eq!(readings(dir, "res.json"), want, "cut {cut}");
    }
}

/// A checkpoint written before the tracker was saved resumes, and run.json
/// says its convergence block is not exact.
#[test]
fn an_old_checkpoint_resumes_and_marks_the_convergence_block_inexact() {
    let scratch = setup("resume-conv-old", Fixture::Series);
    let dir: &Path = scratch.path();
    run(dir, Fixture::Series, &["--generations", "5", "--save", "cp.rcsr"], "cut");
    let text = std::fs::read_to_string(dir.join("cp.rcsr")).unwrap();
    assert!(text.lines().any(|l| l.starts_with("progress ")), "a new checkpoint carries the tracker");
    let old: String = text.lines().filter(|l| !l.starts_with("progress")).map(|l| format!("{l}\n")).collect();
    std::fs::write(dir.join("old.rcsr"), old).unwrap();

    let args = base_args(Fixture::Series, &["--generations", GENERATIONS, "--resume", "old.rcsr", "--json-out", "old.json"]);
    let out = rcsr_in(dir, &args);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let json = std::fs::read_to_string(dir.join("old.json")).unwrap();
    assert!(block(&json, "checkpoint").contains("\"convergence_exact\":false"), "{}", block(&json, "checkpoint"));

    run(dir, Fixture::Series, &["--generations", GENERATIONS, "--resume", "cp.rcsr", "--json-out", "new.json"], "resume");
    let json = std::fs::read_to_string(dir.join("new.json")).unwrap();
    assert!(block(&json, "checkpoint").contains("\"convergence_exact\":true"));
    run(dir, Fixture::Series, &["--generations", GENERATIONS, "--json-out", "fresh.json"], "fresh");
    let json = std::fs::read_to_string(dir.join("fresh.json")).unwrap();
    assert!(block(&json, "checkpoint").contains("\"convergence_exact\":true"));
}
