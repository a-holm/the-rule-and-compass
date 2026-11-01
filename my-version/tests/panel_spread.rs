//! #261: `rcsr panel spread` reproduces the market-neutral bucket spread and
//! refuses a range whose label windows reach a declared locked test.

use rcsr::devlog::Json;
use rcsr::store;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::sync::atomic::{AtomicUsize, Ordering};

const T0: i64 = 1_600_000_000;
const STEP: i64 = 3600;
const HOURS: i64 = 10;

static NEXT: AtomicUsize = AtomicUsize::new(0);

fn scratch() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-panel-spread-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed)
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn p(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn rcsr(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(args).output().expect("run rcsr")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

const FWD_SPEC: &str = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"lbl_h1_y\",\"name\":\"h1\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"lbl_h1_fwd\",\"return_units\":\"log\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":3600,\"delay_seconds\":0,\"entry_price\":\"open[t+1]\",\
\"exit_price\":\"close[t+h]\",\"offset\":0.0,\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\
\"units\":\"log_return\",\"cost_bps\":20.0,\"cost_in_label\":false,\"construction_columns\":[\"lbl_h1_fwd\"],\
\"note\":null,\"long_only\":true}]}\n";

/// 10 hourly timestamps x 4 symbols. The feature ranks the symbols 1..4 and
/// the label is `-0.0001 * feature + hour effect`, so the lowest bucket beats the highest by 3 bp.
fn panel(dir: &Path) -> String {
    let mut text = String::from("ts,symbol,clv1,lbl_h1_fwd\n");
    for h in 0..HOURS {
        for k in 0..4 {
            let feat = ((k + h) % 4) as f64;
            let label = -0.0001 * feat + 0.001 * h as f64;
            text.push_str(&format!("{},S{},{},{}\n", T0 + STEP * h, k, feat, label));
        }
    }
    let table = rcsr::data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = p(dir, "panel.rcsb");
    store::write_store_with_time(&path, &table, 7, Some(0)).unwrap();
    std::fs::write(p(dir, "panel.targets.json"), FWD_SPEC).unwrap();
    path
}

fn base<'a>(panel: &'a str, range: &'a str) -> Vec<&'a str> {
    vec![
        "panel", "spread", "--data", panel, "--feature", "clv1", "--label", "lbl_h1_fwd", "--buckets", "4",
        "--range", range, "--min-members", "4", "--draws", "200",
    ]
}

#[test]
fn spread_prints_the_hand_value_with_an_interval() {
    let dir = scratch();
    let panel = panel(&dir);
    let range = format!("{}..{}", T0, T0 + STEP * HOURS);
    let o = rcsr(&base(&panel, &range));
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let j = Json::parse(String::from_utf8_lossy(&o.stdout).trim()).expect("json on stdout");
    let s = j.get("spread_bp").and_then(|v| v.as_f64()).unwrap();
    assert!((s - 3.0).abs() < 1e-9, "{s}");
    let lo = j.get("ci_lo_bp").and_then(|v| v.as_f64()).unwrap();
    let hi = j.get("ci_hi_bp").and_then(|v| v.as_f64()).unwrap();
    assert!(lo <= s && s <= hi);
    assert_eq!(j.get("timestamps").and_then(|v| v.as_f64()), Some(HOURS as f64));
}

#[test]
fn a_range_reaching_the_declared_locked_test_is_refused() {
    let dir = scratch();
    let panel = panel(&dir);
    let lock = format!("{},{}", T0 + STEP * 6, T0 + STEP * HOURS + STEP);
    // Overlaps the lock outright.
    let range = format!("{}..{}", T0, T0 + STEP * HOURS);
    let mut a = base(&panel, &range);
    a.extend(["--lockbox", &lock]);
    let o = rcsr(&a);
    assert_eq!(code(&o), 3, "{}", stderr(&o));
    assert!(stderr(&o).contains("declared locked test"), "{}", stderr(&o));

    // Ends inside the label purge before the lock (entry offset + horizon = 7200 s).
    let range = format!("{}..{}", T0, T0 + STEP * 5 + 1);
    let mut a = base(&panel, &range);
    a.extend(["--lockbox", &lock]);
    let o = rcsr(&a);
    assert_eq!(code(&o), 3, "{}", stderr(&o));
    assert!(stderr(&o).contains("label window"), "{}", stderr(&o));

    // Ends before the purge: allowed.
    let range = format!("{}..{}", T0, T0 + STEP * 4);
    let mut a = base(&panel, &range);
    a.extend(["--lockbox", &lock]);
    let o = rcsr(&a);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}

#[test]
fn lockbox_none_is_the_only_opt_out() {
    let dir = scratch();
    let panel = panel(&dir);
    let range = format!("{}..{}", T0, T0 + STEP * HOURS);
    let mut a = base(&panel, &range);
    a.extend(["--lockbox", "none"]);
    let o = rcsr(&a);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
}
