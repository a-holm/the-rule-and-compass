//! #1130: `panel snapshot`, `base`, `build`, `schema` and `validate` with
//! `--lock-record`. The fixture holds a symbol and a price spike inside the range.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/lock_record.rs"]
mod lr;
#[path = "common/scratch.rs"]
mod scratch;

use lr::{run, s, Pipeline, LOCK_HOUR, PLANT};
use rcsr::bench_stats::{parse_json, Json};
use std::path::Path;

fn read(p: &Path) -> String {
    std::fs::read_to_string(p).unwrap()
}

fn doc(p: &Path) -> Json {
    parse_json(&read(p)).unwrap()
}

fn member(j: &Json, path: &str) -> String {
    let mut cur = j;
    for k in path.split('.') {
        cur = cur.get(k).unwrap_or_else(|| panic!("{path}"));
    }
    format!("{cur:?}")
}

fn manifest(p: &Pipeline) -> Json {
    doc(Path::new(&format!("{}.manifest.json", s(&p.panel))))
}

/// `availability_end` of every column of a schema file.
fn availability(schema: &str) -> Vec<String> {
    schema
        .lines()
        .filter(|l| l.starts_with("{\"name\":"))
        .map(|l| {
            let at = l.find("\"availability_start\":").unwrap();
            l[at..l.find(",\"pit_rule\"").unwrap()].to_string()
        })
        .collect()
}

#[test]
fn the_sidecars_carry_nothing_from_the_range_and_equal_the_truncated_build() {
    let sc = scratch::Scratch::new("panel-lr-redact");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let full = Pipeline::new(&dir.join("full"), false);
    full.build(Some(&record));
    let cut = Pipeline::new(&dir.join("cut"), true);
    cut.build(None);

    let start = lr::lock_start();
    for f in full.sidecars() {
        let text = read(&f);
        assert!(!text.contains(PLANT), "{} names the planted symbol", f.display());
        let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &s(&f)]);
        assert_eq!(code, 0, "{}: {out}{err}", f.display());
    }

    let (m, t) = (manifest(&full), manifest(&cut));
    for key in ["rows_per_year", "symbols_per_year", "panel.rows", "panel.symbols", "panel.delisted", "panel.gaps", "panel.first_ts", "panel.last_ts", "panel.last_date", "membership", "v2.quality"] {
        assert_eq!(member(&m, key), member(&t, key), "{key}");
    }
    let Some(Json::Obj(_)) = m.get("lock_record") else { panic!("lock_record") };
    assert!(member(&m, "lock_record.rows_left_out") != "Num(0.0)");

    assert_eq!(availability(&read(Path::new(&format!("{}.schema.json", s(&full.panel))))), availability(&read(Path::new(&format!("{}.schema.json", s(&cut.panel))))));
    assert_eq!(member(&doc(Path::new(&format!("{}.schema.json", s(&full.panel)))), "rows"), member(&doc(Path::new(&format!("{}.schema.json", s(&cut.panel)))), "rows"));

    let snap = doc(&full.snapshot);
    let Some(Json::Arr(entries)) = snap.get("entries") else { panic!("entries") };
    assert_eq!(entries.len(), 4, "the planted symbol has no entry; two symbols keep January and a partial February");
    for e in entries {
        assert!((e.get("last_ts").and_then(Json::as_f64).unwrap() as i64) < start);
    }
    let feb = entries.iter().find(|e| e.get("path").and_then(Json::as_str).is_some_and(|p| p.contains("AAAUSDT-15m-2021-02"))).unwrap();
    assert_eq!(feb.get("rows").and_then(Json::as_f64), Some(((LOCK_HOUR - lr::JAN_END) * 4) as f64));
    assert_eq!(member(&snap, "lock_record.rows_left_out"), format!("Num({}.0)", ((lr::FEB_END - LOCK_HOUR) * 2 + (lr::FEB_END - lr::PLANT_FROM)) * 4));
}

#[test]
fn the_unguarded_build_is_flagged_with_exact_counts() {
    let sc = scratch::Scratch::new("panel-lr-control");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    p.build(None);
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &s(&p.dir)]);
    assert_eq!(code, 4, "{err}");
    let d = s(&p.dir);
    for line in [
        format!("{d}/snapshot.json:entries[].last_ts:3"),
        format!("{d}/snapshot.json:entries[].first_ts:1"),
        format!("{d}/snapshot.json:symbols[].last_ts:3"),
        format!("{d}/panel.rcsb.manifest.json:membership.rebalances[]:1"),
        format!("{d}/panel.rcsb.manifest.json:snapshot.entries[].last_ts:3"),
    ] {
        assert!(out.lines().any(|l| l == line), "missing {line} in:\n{out}");
    }
}

#[test]
fn validate_uses_the_development_counts_when_the_record_is_given() {
    let sc = scratch::Scratch::new("panel-lr-validate");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    p.build(Some(&record));
    let base = [s(&p.panel), s(&p.base), s(&p.membership)];
    let args = |extra: &[&str]| -> (i32, String, String) {
        let mut a = vec!["panel", "validate", "--data", &base[0], "--base", &base[1], "--membership", &base[2]];
        a.extend(extra);
        run(&a)
    };
    let (code, out, err) = args(&["--lock-record", &s(&record)]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!out.contains(PLANT));
    let (code, out, _) = args(&[]);
    assert_eq!(code, 3, "without the record the per-year counts of the guarded manifest differ: {out}");
    assert!(out.contains("\"id\":\"V12\"") && out.contains("\"status\":\"fail\""), "{out}");
}

#[test]
fn schema_with_a_record_leaves_the_range_out_and_still_verifies() {
    let sc = scratch::Scratch::new("panel-lr-schema");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    p.build(None);
    let plain = read(Path::new(&format!("{}.schema.json", s(&p.panel))));
    let (code, _, err) = run(&["panel", "schema", "--data", &s(&p.panel), "--lock-record", &s(&record)]);
    assert_eq!(code, 0, "{err}");
    let schema_path = format!("{}.schema.json", s(&p.panel));
    let guarded = read(Path::new(&schema_path));
    assert_ne!(plain, guarded);
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &schema_path]);
    assert_eq!(code, 0, "{out}{err}");
    let (code, _, err) = run(&["panel", "schema", "--data", &s(&p.panel), "--verify"]);
    assert_eq!(code, 0, "{err}");
}
