//! #928: `--select-rule`, `--reselect`, `--reselect-run`.
//!
//! The fixture has `r = 0.03 x1 + 0.004 x2 + noise` on every row. The search
//! front holds a steady member `0.03 x1` (complexity 3) and a more complex
//! member that adds the `x2` term. The complex member has the larger validation
//! net, the simple member is within one standard error of its pooled alpha, so
//! `validation_objective` picks the complex member and `robust` the simple one.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/scratch.rs"]
mod scratch;

use rcsr::devlog::Json;
use scratch::Scratch;
use std::path::Path;
use std::process::Output;

const T0: i64 = 1_600_000_000;
const N: usize = 400;
const ROBUST: &[&str] = &["--select-rule", "robust", "--select-blocks", "4", "--select-se", "1"];

fn frac(v: f64) -> f64 {
    v - v.floor()
}

/// `rotate_from..N` has its `r` and `y` rotated by 7 rows.
fn write_csv(dir: &Path, name: &str, rotate_from: Option<usize>) -> String {
    let mut r: Vec<f64> = (0..N)
        .map(|i| {
            let f = i as f64;
            let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let noise = frac(f * 0.324_717_957_244_746) - 0.5;
            0.03 * x1 + 0.004 * x2 + 0.0005 * noise
        })
        .collect();
    if let Some(from) = rotate_from {
        r[from..].rotate_left(7);
    }
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..N {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        text.push_str(&format!("{},{},{},{},{}\n", T0 + 900 * i as i64, x1, x2, r[i], r[i]));
    }
    let path = dir.join(name);
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_spec(dir: &Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
         \"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
         \"horizon_seconds\":900,\"entry_offset_seconds\":900,\"delay_seconds\":0,\
         \"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
         \"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
         \"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("spec.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

struct Fixture {
    _scratch: Scratch,
    dir: std::path::PathBuf,
    data: String,
    spec: String,
}

fn fixture() -> Fixture {
    let scratch = Scratch::new("reselect");
    let dir = scratch.path().to_path_buf();
    let data = write_csv(&dir, "d.csv", None);
    let spec = write_spec(&dir);
    Fixture { _scratch: scratch, dir, data, spec }
}

impl Fixture {
    fn path(&self, name: &str) -> String {
        self.dir.join(name).to_string_lossy().into_owned()
    }

    fn rcsr(&self, data: &str, extra: &[&str]) -> Output {
        let mut args = vec![
            "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal",
            "--target-spec", self.spec.as_str(), "--cpu", "--seed", "3", "--metric", "squared", "--quiet",
            "--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3",
        ];
        args.extend_from_slice(extra);
        rcsr_cmd::rcsr().args(&args).output().expect("run rcsr")
    }

    /// A 80-generation search saved to `<tag>.ckpt`, report in `<tag>.run.json`.
    fn search(&self, tag: &str, extra: &[&str]) -> Output {
        let (ck, run, log) = (self.path(&format!("{tag}.ckpt")), self.path(&format!("{tag}.run.json")), self.path(&format!("{tag}.dev.ndjson")));
        let mut args = vec!["--generations", "80", "--pop", "128", "--save", ck.as_str(), "--json-out", run.as_str(), "--dev-log", log.as_str()];
        args.extend_from_slice(extra);
        self.rcsr(&self.data, &args)
    }

    fn reselect(&self, from: &str, tag: &str, extra: &[&str]) -> Output {
        let (ck, run, log) = (self.path(&format!("{from}.ckpt")), self.path(&format!("{tag}.run.json")), self.path(&format!("{tag}.dev.ndjson")));
        let mut args = vec!["--reselect", ck.as_str(), "--json-out", run.as_str(), "--dev-log", log.as_str()];
        args.extend_from_slice(extra);
        self.rcsr(&self.data, &args)
    }

    fn run_json(&self, tag: &str) -> Json {
        let text = std::fs::read_to_string(self.path(&format!("{tag}.run.json"))).expect("run.json");
        Json::parse(text.trim()).expect("run.json parses")
    }
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn ok(o: &Output) {
    assert_eq!(code(o), 0, "{}", stderr(o));
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}"));
    }
    cur
}

fn finalist(j: &Json) -> (usize, String) {
    let f = at(j, &["objective", "finalist"]);
    (
        at(f, &["archive_rank"]).as_usize().unwrap(),
        at(f, &["expression_exact"]).as_str().unwrap().to_string(),
    )
}

fn without_reselected_from(j: &Json) -> Json {
    match at(j, &["objective", "selection_rule"]) {
        Json::Obj(fields) => Json::Obj(fields.iter().filter(|(k, _)| k != "reselected_from").cloned().collect()),
        other => other.clone(),
    }
}

fn complexity_of(j: &Json, archive_rank: usize) -> f64 {
    match at(j, &["pareto"]) {
        Json::Arr(a) => a
            .iter()
            .find(|m| m.get("archive_rank").and_then(Json::as_usize) == Some(archive_rank))
            .and_then(|m| m.get("complexity").and_then(Json::as_f64))
            .expect("member"),
        _ => panic!("pareto is not an array"),
    }
}

fn checkpoint_line(dir: &Fixture, tag: &str, key: &str) -> String {
    let text = std::fs::read_to_string(dir.path(&format!("{tag}.ckpt"))).unwrap();
    text.lines().find(|l| l.starts_with(key)).unwrap_or_else(|| panic!("no '{key}' line")).to_string()
}

fn generation_events(dir: &Fixture, tag: &str) -> usize {
    let text = std::fs::read_to_string(dir.path(&format!("{tag}.dev.ndjson"))).unwrap_or_default();
    text.lines().filter(|l| l.contains("\"event\":\"generation\"")).count()
}

#[test]
fn reselect_checkpoint_matches_in_loop() {
    let fx = fixture();
    ok(&fx.search("a", &ROBUST));
    ok(&fx.reselect("a", "b", ROBUST));
    let (a, b) = (fx.run_json("a"), fx.run_json("b"));
    assert_eq!(finalist(&a), finalist(&b));
    assert_eq!(without_reselected_from(&a), without_reselected_from(&b));
    let from = at(&b, &["objective", "selection_rule", "reselected_from"]);
    assert_eq!(at(from, &["kind"]).as_str(), Some("checkpoint"));
    assert_eq!(at(from, &["sha256"]).as_str().map(str::len), Some(64));
    assert_eq!(at(from, &["build_differs"]), &Json::Bool(false));
}

#[test]
fn reselect_changes_finalist_without_search() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    ok(&fx.reselect("a", "b", ROBUST));
    let (a, b) = (fx.run_json("a"), fx.run_json("b"));
    let (rank_a, rank_b) = (finalist(&a).0, finalist(&b).0);
    assert_ne!(rank_a, rank_b, "the rule changes the finalist");
    assert!(complexity_of(&b, rank_b) < complexity_of(&a, rank_a), "the steady member is simpler");
    assert_eq!(at(&a, &["generations_run"]), at(&b, &["generations_run"]));
    assert_eq!(at(&a, &["search", "candidates_evaluated"]), at(&b, &["search", "candidates_evaluated"]));
    assert!(generation_events(&fx, "a") > 0, "the search wrote generation events");
    assert_eq!(generation_events(&fx, "b"), 0, "the re-selection ran no generation");
    let sel = at(&b, &["objective", "selection_rule"]);
    assert_eq!(at(sel, &["rule"]).as_str(), Some("robust"));
    assert_eq!(at(sel, &["verdict"]).as_str(), Some("selected"));
}

#[test]
fn reselect_run_json_matches_checkpoint() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    ok(&fx.reselect("a", "b", ROBUST));
    let run = fx.path("a.run.json");
    let out = fx.path("c.run.json");
    let mut args = vec!["--reselect-run", run.as_str(), "--json-out", out.as_str(), "--no-dev-log"];
    args.extend_from_slice(ROBUST);
    ok(&fx.rcsr(&fx.data, &args));
    let (b, c) = (fx.run_json("b"), fx.run_json("c"));
    assert_eq!(finalist(&b), finalist(&c));
    assert_eq!(
        at(&b, &["objective", "selection_rule", "members"]),
        at(&c, &["objective", "selection_rule", "members"])
    );
    let from = at(&c, &["objective", "selection_rule", "reselected_from"]);
    assert_eq!(at(from, &["kind"]).as_str(), Some("run"));
}

#[test]
fn resume_at_saved_generations_selects_same_finalist() {
    let fx = fixture();
    // `--generations <generations_run>` is a resume with nothing left to run
    // only when the search used its whole budget: a lower value is refused.
    ok(&fx.search("a", &["--patience", "0"]));
    ok(&fx.reselect("a", "b", ROBUST));
    let saved = at(&fx.run_json("a"), &["generations_run"]).as_usize().unwrap().to_string();
    assert_eq!(saved, "80", "the search ran its whole budget");
    let (ck, out) = (fx.path("a.ckpt"), fx.path("r.run.json"));
    let log = fx.path("r.dev.ndjson");
    let mut args = vec!["--resume", ck.as_str(), "--generations", saved.as_str(), "--json-out", out.as_str(), "--dev-log", log.as_str()];
    args.extend_from_slice(ROBUST);
    ok(&fx.rcsr(&fx.data, &args));
    assert_eq!(finalist(&fx.run_json("b")), finalist(&fx.run_json("r")));
    assert_eq!(generation_events(&fx, "r"), 0, "the resume had no generation left to run");
}

#[test]
fn select_rule_not_in_config_hash() {
    let fx = fixture();
    ok(&fx.search("plain", &[]));
    ok(&fx.search("rule", &ROBUST));
    assert_eq!(checkpoint_line(&fx, "plain", "config_hash "), checkpoint_line(&fx, "rule", "config_hash "));
    let ck = fx.path("plain.ckpt");
    let out = fx.path("again.run.json");
    let log = fx.path("again.dev.ndjson");
    let mut args = vec!["--resume", ck.as_str(), "--json-out", out.as_str(), "--dev-log", log.as_str()];
    args.extend_from_slice(ROBUST);
    ok(&fx.rcsr(&fx.data, &args));
    assert_eq!(generation_events(&fx, "again"), 0, "the resume continued the checkpoint");
    let refused = fx.rcsr(&fx.data, &["--resume", ck.as_str(), "--pop", "64", "--no-dev-log"]);
    assert_eq!(code(&refused), 2, "{}", stderr(&refused));
}

#[test]
fn reselect_refusals() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    let ck = fx.path("a.ckpt");
    let text = std::fs::read_to_string(&ck).unwrap();

    let mid = fx.path("mid.ckpt");
    let kept: Vec<&str> = text.lines().filter(|l| !l.starts_with("final ") && !l.starts_with("fsol ")).collect();
    std::fs::write(&mid, kept.join("\n") + "\n").unwrap();
    let o = fx.rcsr(&fx.data, &["--reselect", mid.as_str(), "--select-rule", "robust", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("run it to the end first"), "{}", stderr(&o));

    let saved = at(&fx.run_json("a"), &["generations_run"]).as_usize().unwrap();
    let more = (saved + 1).to_string();
    let o = fx.rcsr(&fx.data, &["--reselect", ck.as_str(), "--select-rule", "robust", "--generations", more.as_str(), "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));

    let other = write_csv(&fx.dir, "other.csv", Some(300));
    let o = fx.rcsr(&other, &["--reselect", ck.as_str(), "--select-rule", "robust", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));

    let gpu = fx.path("gpu.ckpt");
    std::fs::write(&gpu, text.replacen("\nbackend cpu\n", "\nbackend gpu\n", 1)).unwrap();
    let o = fx.rcsr(&fx.data, &["--reselect", gpu.as_str(), "--select-rule", "robust", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let o = fx.rcsr(&fx.data, &["--reselect", gpu.as_str(), "--select-rule", "robust", "--allow-backend-change", "--no-dev-log"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));

    let o = fx.rcsr(&fx.data, &["--reselect", ck.as_str(), "--resume", ck.as_str(), "--select-rule", "robust"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
}

#[test]
fn reselect_accepts_other_build() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    let ck = fx.path("a.ckpt");
    let text = std::fs::read_to_string(&ck).unwrap();
    let commit = checkpoint_line(&fx, "a", "build commit ");
    let other = fx.path("old.ckpt");
    std::fs::write(&other, text.replacen(&commit, "build commit 0000000000000000000000000000000000000000", 1)).unwrap();
    let resumed = fx.rcsr(&fx.data, &["--resume", other.as_str(), "--no-dev-log"]);
    assert_eq!(code(&resumed), 3, "a resume refuses another build: {}", stderr(&resumed));
    let out = fx.path("old.run.json");
    ok(&fx.rcsr(&fx.data, &["--reselect", other.as_str(), "--json-out", out.as_str(), "--no-dev-log", "--select-rule", "robust"]));
    let j = fx.run_json("old");
    let from = at(&j, &["objective", "selection_rule", "reselected_from"]);
    assert_eq!(at(from, &["build_differs"]), &Json::Bool(true));
    assert!(at(from, &["saved_build"]).get("commit").and_then(Json::as_str).is_some_and(|c| c.starts_with("0000")));
}

#[test]
fn robust_selection_ignores_test_rows() {
    let fx = fixture();
    let rotated = write_csv(&fx.dir, "rot.csv", Some(300));
    let run = |data: &str, tag: &str| {
        let out = fx.path(&format!("{tag}.run.json"));
        let mut args = vec!["--generations", "80", "--pop", "128", "--json-out", out.as_str(), "--no-dev-log"];
        args.extend_from_slice(ROBUST);
        ok(&fx.rcsr(data, &args));
        fx.run_json(tag)
    };
    let (a, b) = (run(&fx.data, "x"), run(&rotated, "y"));
    assert_eq!(finalist(&a), finalist(&b));
    assert_eq!(at(&a, &["objective", "selection_rule"]), at(&b, &["objective", "selection_rule"]));
    assert_ne!(
        at(&a, &["objective", "test", "net_pnl"]).as_f64(),
        at(&b, &["objective", "test", "net_pnl"]).as_f64(),
        "the test book itself changed"
    );
}

#[test]
fn one_block_without_se_matches_validation_objective() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    ok(&fx.reselect("a", "b", &["--select-rule", "robust", "--select-blocks", "1", "--select-se", "0"]));
    assert_eq!(finalist(&fx.run_json("a")), finalist(&fx.run_json("b")));
}

#[test]
fn select_rule_refusals_are_usage_errors() {
    let fx = fixture();
    let o = fx.rcsr(&fx.data, &["--select-blocks", "4", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let o = fx.rcsr(&fx.data, &["--select-rule", "robust", "--select-blocks", "9", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let o = fx.rcsr(&fx.data, &["--select-rule", "median", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    let o = fx.rcsr(&fx.data, &["--generations", "2", "--pop", "16", "--select-rule", "robust", "--select-blocks", "8", "--validation-frac", "0.02", "--no-dev-log"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
}

#[test]
fn reselect_reads_the_final_archive_not_the_search_archive() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    let mut cp = rcsr::checkpoint::Checkpoint::load(fx.path("a.ckpt")).expect("checkpoint loads");
    let reported: Vec<f64> = cp.final_archive.as_ref().expect("final archive").items.iter().map(|s| s.cost).collect();
    assert!(reported.len() >= 2);
    cp.archive.items.truncate(1);
    let front = rcsr::reselect::front_of(&cp).expect("front");
    assert_eq!(front.items.iter().map(|s| s.cost).collect::<Vec<_>>(), reported);
    cp.final_archive = None;
    let refusal = rcsr::reselect::front_of(&cp).err().expect("a checkpoint without a final archive is refused");
    assert!(refusal.contains("run it to the end first"), "{refusal}");
}
