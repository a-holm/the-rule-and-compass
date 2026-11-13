//! #1130: `rcsr lock check` on synthetic sidecars with planted in-range entries.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;
#[path = "common/lock_record.rs"]
mod lr;
#[path = "common/scratch.rs"]
mod scratch;

use lr::{run, s, H, T0};
use std::path::Path;

const IN1: i64 = T0 + 1000 * H;
const IN2: i64 = T0 + 1200 * H + 17;
const DEV: i64 = T0 + 100 * H;
const PLANT: &str = "PLANTSYM";
const PLANT_FWD: &str = "0.693147";

fn write(dir: &Path, name: &str, text: &str) {
    let p = dir.join(name);
    std::fs::create_dir_all(p.parent().unwrap()).unwrap();
    std::fs::write(p, text).unwrap();
}

/// Sidecars that carry in-range entries (`planted`) or only development ones.
fn fixture(dir: &Path, planted: bool) {
    let (t1, t2) = if planted { (IN1, IN2) } else { (DEV, DEV + H) };
    let (d1, d2) = if planted { ("2021-02-12T16:00:00Z", "2021-02-20") } else { ("2021-01-05T04:00:00Z", "2021-01-06") };
    write(
        dir,
        "base.labels.json",
        &format!(
            "{{\"long_labels\":[{{\"report\":{{\"top_abs_moves\":{{\"n\":3,\"moves\":[\
             {{\"symbol\":\"{PLANT}\",\"time\":{t1},\"time_iso\":\"{d1}\",\"fwd\":{PLANT_FWD}}},\
             {{\"symbol\":\"AAA\",\"time\":{DEV},\"time_iso\":\"2021-01-05T04:00:00Z\",\"fwd\":0.1}},\
             {{\"symbol\":\"BBB\",\"time\":{t2},\"time_iso\":\"{d1}\",\"fwd\":0.2}}]}},\
             \"locked_test\":{{\"start\":{},\"start_date\":\"2021-02-03\",\"rows_left_out\":0}}}}}}]}}\n",
            lr::lock_start()
        ),
    );
    let reb = if planted { t1 } else { DEV + 2 * H };
    write(
        dir,
        "panel.rcsb.manifest.json",
        &format!(
            "{{\"locked_test_also\":{{\"start\":{},\"end\":{},\"start_date\":\"2021-02-03\"}},\
             \"membership\":{{\"rebalances\":[{T0},{reb}]}},\"panel\":{{\"last_ts\":{t2},\"last_date\":\"{d2}\"}},\
             \"output\":{{\"bytes\":{},\"mtime\":{t1}}}}}\n",
            lr::lock_start(),
            lr::lock_end(),
            t1
        ),
    );
    write(dir, "sub/snapshot.json", &format!("{{\"entries\":[{{\"first_ts\":{},\"last_ts\":{}}},{{\"first_ts\":5,\"last_ts\":{DEV}}}]}}\n", t1 * 1000, t2));
    write(dir, "members.csv", &format!("as_of,symbol\n{DEV},AAA\n{t1},{PLANT}\n{t2},BBB\n"));
    write(dir, "column.rcsb", "binary");
    write(dir, "notes.txt", &format!("{t1}"));
    write(dir, "locked-tail-fixture.json", &lr::record_text(lr::lock_start(), lr::lock_end()));
}

fn check(dir: &Path) -> (i32, String, String) {
    run(&["lock", "check", "--record", &s(&dir.join("locked-tail-fixture.json")), &s(dir)])
}

#[test]
fn counts_are_printed_per_path_and_key_and_the_exit_is_four() {
    let sc = scratch::Scratch::new("lock-check-planted");
    let dir = sc.path();
    fixture(dir, true);
    let (code, out, err) = check(dir);
    assert_eq!(code, 4, "{err}");
    let d = s(dir);
    let mut want = vec![
        format!("{d}/base.labels.json:long_labels[].report.top_abs_moves.moves[].time:2"),
        format!("{d}/base.labels.json:long_labels[].report.top_abs_moves.moves[].time_iso:2"),
        format!("{d}/members.csv:as_of:2"),
        format!("{d}/panel.rcsb.manifest.json:membership.rebalances[]:1"),
        format!("{d}/panel.rcsb.manifest.json:panel.last_date:1"),
        format!("{d}/panel.rcsb.manifest.json:panel.last_ts:1"),
        format!("{d}/sub/snapshot.json:entries[].first_ts:1"),
        format!("{d}/sub/snapshot.json:entries[].last_ts:1"),
    ];
    want.sort();
    let mut got: Vec<String> = out.lines().map(str::to_string).collect();
    got.sort();
    assert_eq!(got, want, "stderr: {err}");
}

#[test]
fn the_output_names_no_timestamp_symbol_or_value_from_inside_the_range() {
    let sc = scratch::Scratch::new("lock-check-leak");
    let dir = sc.path();
    fixture(dir, true);
    let (_, out, err) = check(dir);
    let text = format!("{out}{err}");
    for needle in [PLANT, PLANT_FWD, &IN1.to_string(), &IN2.to_string(), &(IN1 * 1000).to_string(), "2021-02-12", "2021-02-20", "16:00"] {
        assert!(!text.contains(needle), "the output names {needle}: {text}");
    }
}

#[test]
fn sidecars_without_an_entry_in_the_range_pass() {
    let sc = scratch::Scratch::new("lock-check-clean");
    let dir = sc.path();
    fixture(dir, false);
    let (code, out, err) = check(dir);
    assert_eq!(code, 0, "{out}{err}");
    assert!(out.is_empty(), "{out}");
}

#[test]
fn a_record_that_is_not_usable_exits_two_and_names_the_field() {
    let sc = scratch::Scratch::new("lock-check-record");
    let dir = sc.path();
    fixture(dir, false);
    let bad = |name: &str, text: &str, field: &str| {
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        let (code, _, err) = run(&["lock", "check", "--record", &s(&p), &s(dir)]);
        assert_eq!(code, 2, "{name}: {err}");
        assert!(err.contains(field), "{name}: {err}");
    };
    bad("open.json", r#"{"range":{"start_epoch":10,"end_epoch":20,"end_exclusive":false}}"#, "range.end_exclusive");
    bad("unmarked.json", r#"{"range":{"start_epoch":10,"end_epoch":20}}"#, "range.end_exclusive");
    bad("reversed.json", r#"{"range":{"start_epoch":20,"end_epoch":10,"end_exclusive":true}}"#, "range.end_epoch");
    bad("text.json", r#"{"range":{"start_epoch":"soon","end_epoch":10,"end_exclusive":true}}"#, "range.start_epoch");
    bad("garbage.json", "{not json", "not valid JSON");
    let (code, _, err) = run(&["lock", "check", "--record", &s(&dir.join("absent.json")), &s(dir)]);
    assert_eq!(code, 3, "{err}");
    let (code, _, err) = run(&["lock", "check", "--record", &s(&dir.join("locked-tail-fixture.json"))]);
    assert_eq!(code, 2, "{err}");
}
