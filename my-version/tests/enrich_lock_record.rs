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
        .filter(|(n, _, _)| !n.starts_with("lbl_"))
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
    assert!(!r.locked.contains(PLANT), "no planted symbol anywhere in the enrich report");
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
