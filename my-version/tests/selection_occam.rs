//! #1184: `--select-rule occam`: in-loop rule, re-selection, refusals and default output.
//!
//! The fixture is the one of tests/reselect.rs: `r = 0.03 x1 + 0.004 x2 + noise`.

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
const OCCAM: &[&str] = &["--select-rule", "occam", "--select-min-entries", "5", "--select-delta", "0.05"];


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
    let scratch = Scratch::new("selection_occam");
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

fn occam_run(fx: &Fixture, tag: &str) -> Json {
    ok(&fx.search(tag, OCCAM));
    fx.run_json(tag)
}

fn pareto(j: &Json) -> &Vec<Json> {
    match at(j, &["pareto"]) {
        Json::Arr(a) => a,
        _ => panic!("pareto is not an array"),
    }
}

#[test]
fn reselect_run_gives_the_in_loop_selection_rule() {
    let fx = fixture();
    let a = occam_run(&fx, "a");
    let run = fx.path("a.run.json");
    let out = fx.path("c.run.json");
    let mut args = vec!["--reselect-run", run.as_str(), "--json-out", out.as_str(), "--no-dev-log"];
    args.extend_from_slice(OCCAM);
    ok(&fx.rcsr(&fx.data, &args));
    let c = fx.run_json("c");
    assert_eq!(without_reselected_from(&a), without_reselected_from(&c));
    assert_eq!(finalist(&a).0, finalist(&c).0);
    let from = at(&c, &["objective", "selection_rule", "reselected_from"]);
    assert_eq!(at(from, &["kind"]).as_str(), Some("run"));
    let sel = at(&a, &["objective", "selection_rule"]);
    assert_eq!(at(sel, &["rule"]).as_str(), Some("occam"));
    assert!(matches!(at(sel, &["verdict"]).as_str(), Some("certified" | "no_signal")));
}

#[test]
fn occam_fields_appear_with_the_rule_and_name_the_code() {
    let fx = fixture();
    let a = occam_run(&fx, "a");
    let sel = at(&a, &["objective", "selection_rule"]);
    assert_eq!(at(sel, &["delta"]).as_f64(), Some(0.05));
    assert_eq!(at(sel, &["min_entries"]).as_usize(), Some(5));
    let spec = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/occam-code-v1.json")).unwrap();
    assert_eq!(at(sel, &["code", "spec_sha256"]).as_str(), Some(rcsr::manifest::sha256_hex(&spec).as_str()));
    assert_eq!(at(sel, &["code", "sha256"]).as_str().map(str::len), Some(64));
    let mut scored = 0;
    for m in pareto(&a) {
        let o = m.get("occam").expect("every member has an occam member");
        if matches!(o, Json::Null) {
            continue;
        }
        scored += 1;
        let bits = at(o, &["bits"]).as_f64().unwrap();
        let (s, c) = (at(o, &["bits_structure"]).as_f64().unwrap(), at(o, &["bits_constants"]).as_f64().unwrap());
        assert!((bits - s - c).abs() < 1e-9);
        assert!(at(o, &["blocks"]).as_usize().unwrap() >= 20);
        match at(o, &["p_occam"]).as_f64() {
            Some(p_occam) => assert!((0.0..=1.0).contains(&p_occam), "{p_occam}"),
            None => assert!(at(o, &["p"]).as_f64().is_none(), "p_occam is null only without a p-value"),
        }
    }
    assert!(scored > 0);
}

#[test]
fn a_run_without_the_rule_has_no_occam_output() {
    let fx = fixture();
    ok(&fx.search("a", &[]));
    let text = std::fs::read_to_string(fx.path("a.run.json")).unwrap();
    assert!(!text.contains("\"occam\""), "default run JSON carries an occam field");
    let a = fx.run_json("a");
    let sel = at(&a, &["objective", "selection_rule"]);
    assert_eq!(sel, &Json::Obj(vec![("rule".to_string(), Json::Str("validation_objective".to_string()))]));
}

#[test]
fn occam_refusals_are_usage_errors() {
    let fx = fixture();
    for extra in [
        &["--select-delta", "0.05", "--no-dev-log"][..],
        &["--select-min-entries", "5", "--no-dev-log"][..],
        &["--select-rule", "occam", "--select-delta", "0", "--no-dev-log"][..],
        &["--select-rule", "occam", "--select-delta", "1", "--no-dev-log"][..],
        &["--select-rule", "occam", "--select-delta", "1.5", "--no-dev-log"][..],
        &["--select-rule", "robust", "--select-delta", "0.05", "--no-dev-log"][..],
    ] {
        let o = fx.rcsr(&fx.data, extra);
        assert_eq!(code(&o), 2, "{extra:?}: {}", stderr(&o));
    }
}

#[test]
fn reselect_checkpoint_gives_the_in_loop_selection_rule() {
    let fx = fixture();
    let a = occam_run(&fx, "a");
    ok(&fx.reselect("a", "b", OCCAM));
    let b = fx.run_json("b");
    assert_eq!(without_reselected_from(&a), without_reselected_from(&b));
    assert_eq!(finalist(&a), finalist(&b));
}
