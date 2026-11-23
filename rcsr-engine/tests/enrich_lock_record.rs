//! #1130: `rcsr enrich --lock-record` leaves the campaign range out of the label
//! report and the feature statistics. The fixture plants an extreme move and a
//! symbol that trades only inside the range.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/lock_record.rs"]
mod lr;
#[path = "common/long_horizon.rs"]
mod lh;
#[path = "common/scratch.rs"]
mod scratch;

use lr::{run, s, H, LOCK_HOUR, T0};
use rcsr::bench_stats::{parse_json, Json};
use rcsr::ohlcv::Bar;
use std::path::{Path, PathBuf};

const N: usize = 1000;
const PLANT: &str = "ZZPLANT";
/// The planted bar: its one-bar label row is the extreme move.
const PLANT_BAR: usize = 900;
/// A development move that stays in the report.
const DEV_BAR: usize = 100;

fn series() -> Vec<(&'static str, Vec<Bar>)> {
    let mut a = lh::bars(T0, N, 0.0);
    let b = lh::bars(T0, N, 1.0);
    let mut z = lh::bars(T0, N, 2.0);
    let k = a[DEV_BAR];
    a[DEV_BAR] = Bar::new(k.time, k.open, k.open * 1.5 + 0.3, k.open - 0.3, k.open * 1.5, k.volume);
    let k = z[PLANT_BAR];
    z[PLANT_BAR] = Bar::new(k.time, k.open, k.open * 1.9 + 0.3, k.open - 0.3, k.open * 1.9, k.volume);
    vec![("AAA", a), ("BBB", b), (PLANT, z)]
}

fn store(dir: &Path, name: &str, rows: usize) -> String {
    let data = series();
    let cut: Vec<(&str, &[Bar])> = data.iter().map(|(n, b)| (*n, &b[..rows])).collect();
    let p = s(&dir.join(name));
    lh::write_store(&p, &cut);
    p
}

fn enrich(src: &str, out: &str, labels: bool, record: Option<&Path>) -> String {
    let mut a: Vec<String> = ["enrich", "--data", src, "--out", out, "--cpu", "--features", "ret1"].iter().map(|x| x.to_string()).collect();
    if labels {
        a.push("--label".into());
        a.push(lh::src(1, ""));
    }
    if let Some(r) = record {
        a.push("--lock-record".into());
        a.push(s(r));
    }
    let args: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, _, err) = run(&args);
    assert_eq!(code, 0, "{err}");
    std::fs::read_to_string(format!("{out}.enrich.json")).unwrap()
}

fn num(j: &Json, key: &str) -> f64 {
    j.get(key).and_then(Json::as_f64).unwrap_or_else(|| panic!("{key}"))
}

fn features(text: &str) -> Vec<(String, f64, f64)> {
    let doc = parse_json(text).unwrap();
    let Some(Json::Arr(items)) = doc.get("features") else { panic!("features") };
    items
        .iter()
        .map(|f| (f.get("name").and_then(Json::as_str).unwrap().to_string(), num(f, "prevalence"), num(f, "nan_rows")))
        .filter(|(n, _, _)| !n.starts_with("lbl_") && n != "vol96")
        .collect()
}

fn report(text: &str) -> Json {
    let doc = parse_json(text).unwrap();
    let Some(Json::Arr(items)) = doc.get("long_labels") else { panic!("long_labels") };
    items[0].get("report").unwrap().clone()
}

fn expected_left_out() -> usize {
    let spec = lh::spec(&lh::src(1, ""));
    let purge = spec.purge_seconds().max(1);
    series()
        .iter()
        .map(|(_, bars)| {
            let l = rcsr::labels::build(bars, &spec, None).unwrap();
            (0..bars.len()).filter(|&t| l.fwd[t].is_finite() && bars[t].time + purge > lr::lock_start()).count()
        })
        .sum()
}

struct Runs {
    _scratch: scratch::Scratch,
    dir: PathBuf,
    record: PathBuf,
    plain: String,
    locked: String,
}

fn runs(tag: &str) -> Runs {
    let sc = scratch::Scratch::new(tag);
    let dir = sc.path().to_path_buf();
    let record = lr::write_record(&dir);
    let src = store(&dir, "in.rcsb", N);
    std::fs::create_dir_all(dir.join("plain")).unwrap();
    std::fs::create_dir_all(dir.join("locked")).unwrap();
    let plain = enrich(&src, &s(&dir.join("plain/out.rcsb")), true, None);
    let locked = enrich(&src, &s(&dir.join("locked/out.rcsb")), true, Some(&record));
    Runs { _scratch: sc, dir, record, plain, locked }
}

#[test]
fn the_label_report_has_no_top_move_inside_the_range_and_counts_the_rows_it_left_out() {
    let r = runs("enrich-lr-labels");
    let planted_time = (T0 + (PLANT_BAR as i64 - 1) * H).to_string();
    assert!(r.plain.contains(PLANT) && r.plain.contains(&planted_time), "the unguarded report names the planted move");
    assert!(!r.locked.contains(&format!("\"time\":{planted_time}")), "the planted time");
    let rep = report(&r.locked);
    let Some(Json::Obj(_)) = rep.get("top_abs_moves") else { panic!("top_abs_moves") };
    let Some(Json::Arr(moves)) = rep.get("top_abs_moves").and_then(|t| t.get("moves")) else { panic!("moves") };
    assert!(!moves.is_empty());
    for m in moves {
        assert!((num(m, "time") as i64) < lr::lock_start(), "a top move reaches the range");
    }
    let dev_time = T0 + (DEV_BAR as i64 - 1) * H;
    assert!(moves.iter().any(|m| num(m, "time") as i64 == dev_time), "the development move stays");
    let also = rep.get("locked_test_also").expect("locked_test_also");
    assert_eq!(num(also, "rows_left_out") as usize, expected_left_out());
    assert!(expected_left_out() > 0);
}

#[test]
fn feature_statistics_equal_a_run_on_the_store_truncated_before_the_range() {
    let r = runs("enrich-lr-truncated");
    let short = store(&r.dir, "short.rcsb", LOCK_HOUR as usize);
    let truncated = enrich(&short, &s(&r.dir.join("short-out.rcsb")), false, None);
    let (a, b) = (features(&r.locked), features(&truncated));
    assert!(!a.is_empty());
    assert_eq!(a, b);
    assert_ne!(features(&r.plain), b, "the unguarded statistics include the range");
}

#[test]
fn lock_check_flags_the_unguarded_sidecars_and_passes_the_guarded_ones() {
    let r = runs("enrich-lr-check");
    let rec = s(&r.record);
    let (code, out, err) = run(&["lock", "check", "--record", &rec, &s(&r.dir.join("plain"))]);
    assert_eq!(code, 4, "{out}{err}");
    assert!(out.contains("top_abs_moves"), "{out}");
    let (code, out, err) = run(&["lock", "check", "--record", &rec, &s(&r.dir.join("locked"))]);
    assert_eq!(code, 0, "{out}{err}");
}

#[test]
fn a_record_that_is_not_usable_is_refused_before_any_work() {
    let sc = scratch::Scratch::new("enrich-lr-bad");
    let dir = sc.path();
    let src = store(dir, "in.rcsb", 200);
    let bad = dir.join("bad.json");
    std::fs::write(&bad, r#"{"range":{"start_epoch":10,"end_epoch":20,"end_exclusive":false}}"#).unwrap();
    let out = s(&dir.join("out.rcsb"));
    let (code, _, err) = run(&["enrich", "--data", &src, "--out", &out, "--cpu", "--features", "ret1", "--lock-record", &s(&bad)]);
    assert_eq!(code, 2, "{err}");
    assert!(err.contains("range.end_exclusive"), "{err}");
    assert!(!Path::new(&out).exists());
}

fn enrich_label(src: &str, out: &str, features: &str, label: &str, record: Option<&Path>) -> (i32, String) {
    let mut a: Vec<String> = ["enrich", "--data", src, "--out", out, "--cpu", "--features", features, "--label", label].iter().map(|x| x.to_string()).collect();
    if let Some(r) = record {
        a.push("--lock-record".into());
        a.push(s(r));
    }
    let args: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, _, err) = run(&args);
    (code, err)
}

#[test]
fn a_chained_run_under_the_record_refuses_labels_carried_from_a_run_without_it() {
    let sc = scratch::Scratch::new("enrich-lr-chain");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let src = store(dir, "in.rcsb", N);
    let (a, b) = (s(&dir.join("a.rcsb")), s(&dir.join("b.rcsb")));
    assert_eq!(enrich_label(&src, &a, "r1", &lh::src(1, ""), None).0, 0);
    let (code, err) = enrich_label(&a, &b, "r4", &lh::src(2, ""), Some(&record));
    assert_ne!(code, 0, "{err}");
    assert!(err.contains("lbl_h1_y") && err.contains("without this lock record"), "{err}");
    assert!(!Path::new(&b).exists() && !Path::new(&labels_path(&b)).exists());

    let (a2, b2) = (s(&dir.join("a2.rcsb")), s(&dir.join("b2.rcsb")));
    assert_eq!(enrich_label(&src, &a2, "r1", &lh::src(1, ""), Some(&record)).0, 0);
    let (code, err) = enrich_label(&a2, &b2, "r4", &lh::src(2, ""), Some(&record));
    assert_eq!(code, 0, "{err}");
    let text = std::fs::read_to_string(labels_path(&b2)).unwrap();
    assert!(text.contains("lbl_h1_y") && text.contains("lbl_h2_y"), "chained labels under the record are kept");
    let (code, out, err) = run(&["lock", "check", "--record", &s(&record), &labels_path(&b2)]);
    assert_eq!(code, 0, "{out}{err}");
}

#[test]
fn an_in_place_rerun_under_the_record_refuses_labels_it_does_not_recompute() {
    let sc = scratch::Scratch::new("enrich-lr-inplace");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let src = store(dir, "in.rcsb", N);
    let out = s(&dir.join("x.rcsb"));
    assert_eq!(enrich_label(&src, &out, "r1", &lh::src(1, ""), None).0, 0);
    let (code, err) = enrich_label(&src, &out, "r4", &lh::src(2, ""), Some(&record));
    assert_ne!(code, 0, "{err}");
    assert!(err.contains("lbl_h1_y"), "{err}");
    let (code, out_text, err) = run(&["lock", "check", "--record", &s(&record), &labels_path(&out)]);
    assert_eq!(code, 4, "the refused run leaves the earlier sidecar untouched: {out_text}{err}");
}

fn enrich_features(src: &str, out: &str, features: &str, record: Option<&Path>) -> String {
    let mut a: Vec<String> = ["enrich", "--data", src, "--out", out, "--cpu", "--features", features].iter().map(|x| x.to_string()).collect();
    if let Some(r) = record {
        a.push("--lock-record".into());
        a.push(s(r));
    }
    let args: Vec<&str> = a.iter().map(String::as_str).collect();
    let (code, _, err) = run(&args);
    assert_eq!(code, 0, "{err}");
    std::fs::read_to_string(format!("{out}.enrich.json")).unwrap()
}

fn all_features(text: &str) -> Vec<(String, f64, f64)> {
    features(text)
}

#[test]
fn every_feature_statistic_equals_a_run_on_the_store_truncated_before_the_range() {
    let sc = scratch::Scratch::new("enrich-lr-allfeat");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let src = store(dir, "in.rcsb", N);
    let short = store(dir, "short.rcsb", LOCK_HOUR as usize);
    let list = "r1,r4,ma5,vol20";
    let locked = enrich_features(&src, &s(&dir.join("locked.rcsb")), list, Some(&record));
    let truncated = enrich_features(&short, &s(&dir.join("truncated.rcsb")), list, None);
    let (a, b) = (all_features(&locked), all_features(&truncated));
    assert!(a.len() >= 3, "{a:?}");
    assert_eq!(a, b);
}

#[test]
fn a_stale_carrier_that_lacks_the_record_range_is_refused_before_the_output_is_written() {
    let sc = scratch::Scratch::new("enrich-lr-carrier");
    let dir = sc.path();
    let record = lr::write_record(dir);
    let src = store(dir, "in.rcsb", N);
    let declaration = format!(
        "{{\"locked_test\":{{\"start\":{},\"end\":{},\"start_date\":\"2021-01-05\",\"end_date\":\"2021-01-09\",\"source\":\"fixture\"}}}}\n",
        T0 + 100 * H,
        T0 + 200 * H
    );
    std::fs::write(format!("{src}.manifest.json"), &declaration).unwrap();
    let run_enrich = |out: &str| {
        let (code, _, err) = run(&["enrich", "--data", &src, "--out", out, "--cpu", "--features", "r1", "--lock-record", &s(&record)]);
        (code, err)
    };
    let clean = s(&dir.join("clean.rcsb"));
    let (code, err) = run_enrich(&clean);
    assert_eq!(code, 0, "{err}");
    let text = std::fs::read_to_string(format!("{clean}.enrich.json")).unwrap();
    assert!(text.contains("\"locked_test_also\""), "the record is the second range: {}", &text[..300.min(text.len())]);

    let stale = s(&dir.join("stale.rcsb"));
    std::fs::write(format!("{stale}.manifest.json"), &declaration).unwrap();
    let (code, err) = run_enrich(&stale);
    assert_ne!(code, 0, "{err}");
    assert!(err.contains("stale") && err.contains("manifest.json"), "{err}");
    assert!(!Path::new(&stale).exists() && !Path::new(&format!("{stale}.enrich.json")).exists());
}

fn labels_path(store: &str) -> String {
    s(&Path::new(store).with_extension("labels.json"))
}
