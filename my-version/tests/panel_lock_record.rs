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

#[test]
fn every_file_the_pipeline_wrote_is_free_of_the_planted_symbol_and_passes_lock_check() {
    let sc = scratch::Scratch::new("panel-lr-allfiles");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    p.build(Some(&record));
    let files = lr::written_files(&p.dir);
    assert!(files.len() >= 4, "{files:?}");
    for f in &files {
        assert!(!read(f).contains(PLANT), "{} names the planted symbol", f.display());
    }
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &s(&p.dir.join("snapshot.json")), &format!("{}.manifest.json", s(&p.base)), &format!("{}.manifest.json", s(&p.panel)), &format!("{}.schema.json", s(&p.panel))]);
    assert_eq!(code, 0, "{out}{err}");
    let all: Vec<String> = files.iter().filter(|f| f.extension().is_some_and(|e| e == "json" || e == "csv")).map(|f| s(f)).collect();
    let mut a = vec!["lock".to_string(), "check".into(), "--record".into(), s(&record)];
    a.extend(all);
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, out, err) = run(&a);
    assert_eq!(code, 0, "{out}{err}");
}

#[test]
fn daily_archives_inside_the_range_are_not_listed_by_the_base_manifest_or_the_snapshot() {
    let sc = scratch::Scratch::new("panel-lr-daily");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let raw = dir.join("raw");
    lr::daily_archives(&raw);
    let (base, snap) = (dir.join("base_h1.rcsb"), dir.join("snapshot.json"));
    let (code, _, err) = run(&["panel", "base", "--input", &s(&raw), "--jobs", "2", "--cpu", "--out", &s(&base), "--lock-record", &s(&record)]);
    assert_eq!(code, 0, "{err}");
    let (code, _, err) = run(&["panel", "snapshot", "--input", &s(&raw), "--jobs", "2", "--out", &s(&snap), "--lock-record", &s(&record)]);
    assert_eq!(code, 0, "{err}");
    let manifest = format!("{}.manifest.json", s(&base));
    for f in [&manifest, &s(&snap)] {
        let text = read(Path::new(f));
        assert!(text.contains("2021-02-01") && text.contains("2021-01"), "development archives stay listed in {f}");
        assert!(!text.contains("2021-02-05") && !text.contains("2021-02-06"), "an in-range daily archive is listed in {f}");
        let (code, out, err) = run(&["lock", "check", "--record", &s(&record), f]);
        assert_eq!(code, 0, "{f}: {out}{err}");
    }
}

#[test]
fn the_base_lockbox_block_uses_development_rows_only() {
    let sc = scratch::Scratch::new("panel-lr-overlap");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    let guarded = format!("{},{}", lr::lock_start() - 24 * lr::H, lr::lock_end());
    let (code, _, err) = run(&[
        "panel", "base", "--input", &s(&p.raw), "--jobs", "2", "--cpu", "--out", &s(&p.base), "--locked-test", &guarded, "--allow-lockbox", "--lock-record", &s(&record),
    ]);
    assert_eq!(code, 0, "{err}");
    let manifest = format!("{}.manifest.json", s(&p.base));
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &manifest]);
    assert_eq!(code, 0, "{out}{err}");
    assert!(!read(Path::new(&manifest)).contains(PLANT));
}

#[test]
fn append_with_a_record_leaves_the_range_out_of_the_schema_and_every_written_file() {
    let sc = scratch::Scratch::new("panel-lr-append");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let p = Pipeline::new(&dir.join("full"), false);
    p.build(Some(&record));
    let fam = s(&dir.join("fam.rcsb"));
    let (code, _, err) = run(&["enrich", "--data", &s(&p.base), "--out", &fam, "--cpu", "--features", "ret1", "--locked-test", "none"]);
    assert_eq!(code, 0, "{err}");
    let (rec, plain) = (s(&dir.join("with.rcsb")), s(&dir.join("without.rcsb")));
    let panel = s(&p.panel);
    let go = |out: &str, extra: &[&str]| {
        let mut a = vec!["panel", "append", "--data", &panel, "--family", &fam, "--columns", "ret1", "--out", out];
        a.extend(extra);
        run(&a)
    };
    let (code, out, err) = go(&rec, &["--lock-record", &s(&record)]);
    assert_eq!(code, 0, "{out}{err}");
    let schema = format!("{rec}.schema.json");
    let text = read(Path::new(&schema));
    assert!(text.contains("\"lock_record\""), "{text}");
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &schema]);
    assert_eq!(code, 0, "{out}{err}");
    let (code, out, err) = go(&plain, &[]);
    assert_eq!(code, 0, "{out}{err}");
    let (code, out, _) = run(&["lock", "check", "--record", &s(&record), &format!("{plain}.schema.json")]);
    assert_eq!(code, 4, "the control without the record is flagged: {out}");
    for f in lr::written_files(&dir.join("full")) {
        assert!(!read(&f).contains(PLANT), "{}", f.display());
    }
}

#[test]
fn the_campaign_rebuild_shape_finishes_under_the_record() {
    let sc = scratch::Scratch::new("panel-lr-shape");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let (rec, fwd) = (s(&record), format!("{},{}", lr::T0 + 50_000 * lr::H, lr::T0 + 60_000 * lr::H));
    let p = Pipeline::new(&dir.join("full"), false);
    p.build_with(Some(&record), Some(&fwd));
    let fam = s(&dir.join("full/base_enriched.rcsb"));
    let mut a: Vec<String> = ["enrich", "--data", &s(&p.base), "--out", &fam, "--cpu", "--features", "r1", "--locked-test", &fwd, "--lock-record", &rec]
        .iter()
        .map(|x| x.to_string())
        .collect();
    for h in 1..=7 {
        a.push("--label".into());
        a.push(format!("long:{h},name=h{h},fee=10,stride={h}"));
    }
    let args: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, out, err) = run(&args);
    assert_eq!(code, 0, "{out}{err}");
    let outp = s(&dir.join("full/panel_enriched.rcsb"));
    let (code, out, err) = run(&["panel", "append", "--data", &s(&p.panel), "--family", &fam, "--columns", "r1", "--out", &outp, "--locked-test", &fwd, "--lock-record", &rec]);
    assert_eq!(code, 0, "{out}{err}");
    let (code, out, err) = run(&["panel", "schema", "--data", &outp, "--verify"]);
    assert_eq!(code, 0, "{out}{err}");
    let files = lr::written_files(&p.dir);
    assert!(files.len() >= 8, "{files:?}");
    for f in &files {
        assert!(!read(f).contains(PLANT), "{} names the planted symbol", f.display());
    }
    let mut a = vec!["lock".to_string(), "check".into(), "--record".into(), rec.clone()];
    a.extend(files.iter().filter(|f| f.extension().is_some_and(|e| e == "json" || e == "csv")).map(|f| s(f)));
    let a: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, out, err) = run(&a);
    assert_eq!(code, 0, "{out}{err}");
}

fn v1_csv(path: &Path, to: i64, with_plant: bool) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume\n");
    let mut row = |sym: &str, hr: i64, base: f64| {
        let p = base * (1.0 + 0.0001 * hr as f64);
        text.push_str(&format!("{},{},{p},{},{},{p},10\n", lr::T0 + hr * lr::H, sym, p * 1.01, p * 0.99));
    };
    for hr in 0..to {
        row("AAA", hr, 10.0);
        row("LATE", hr, 20.0);
        if hr < 50 {
            row("GONE", hr, 30.0);
        }
        if with_plant && hr >= 850 {
            row("ZZPLANT", hr, 7.0);
        }
    }
    std::fs::write(path, text).unwrap();
}

#[test]
fn the_v1_panel_build_under_the_record_equals_the_truncated_build_and_names_nothing_from_the_range() {
    let sc = scratch::Scratch::new("panel-lr-v1");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let mut m = String::from("as_of,symbol,member,adv_usd,n_bars\n");
    for sym in ["AAA", "LATE", "GONE"] {
        m.push_str(&format!("{},{sym},1,1000.0,100\n", lr::T0));
    }
    m.push_str(&format!("{},LATE,0,1000.0,100\n{},AAA,1,1000.0,100\n", lr::T0 + 400 * lr::H, lr::T0 + 400 * lr::H));
    let full_m = format!("{m}{},ZZPLANT,1,1000.0,100\n", lr::T0 + 900 * lr::H);
    let (mf, mc) = (dir.join("full-m.csv"), dir.join("cut-m.csv"));
    std::fs::write(&mf, full_m).unwrap();
    std::fs::write(&mc, &m).unwrap();
    let go = |name: &str, to: i64, plant: bool, memb: &Path, rec: Option<&Path>| -> String {
        let csv = dir.join(format!("{name}.csv"));
        v1_csv(&csv, to, plant);
        let store = dir.join(format!("{name}.rcsb"));
        let (code, _, err) = run(&["import", &s(&csv), &s(&store)]);
        assert_eq!(code, 0, "{err}");
        let out = s(&dir.join(format!("{name}-panel.rcsb")));
        let (store_s, memb_s) = (s(&store), s(memb));
        let mut a = vec!["panel", "build", "--data", &store_s, "--membership", &memb_s, "--bar-seconds", "3600", "--out", &out];
        let r = rec.map(s);
        if let Some(r) = &r {
            a.extend(["--lock-record", r]);
        }
        let (code, o, err) = run(&a);
        assert_eq!(code, 0, "{o}{err}");
        format!("{out}.manifest.json")
    };
    let full = go("full", 1000, true, &mf, Some(&record));
    let cut = go("cut", LOCK_HOUR, false, &mc, None);
    let (f, c) = (doc(Path::new(&full)), doc(Path::new(&cut)));
    for key in ["panel.delisted", "panel.symbols", "panel.rows", "panel.gaps", "panel.last_ts", "membership"] {
        assert_eq!(member(&f, key), member(&c, key), "{key}");
    }
    assert_eq!(member(&f, "panel.delisted"), "Num(1.0)", "GONE only: LATE trades on into the range");
    assert!(!read(Path::new(&full)).contains("ZZPLANT"));
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &full]);
    assert_eq!(code, 0, "{out}{err}");
}
