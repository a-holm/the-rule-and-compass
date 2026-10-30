//! Golden checks of the declared target spec on the pinned compact cell (#254).
//!
//! The cell is 1.9 GB and stays out of git. Run with
//! `RCSR_COMPACT_CELL=<path>/panel_hourly_compact.rcsb cargo test --release --test target_golden -- --ignored`.

use rcsr::devlog::Json;
use std::process::{Command, Output};

const INPUTS: &str = "r16,r96,r672,ma96,ma672,ma_slope672,macd_hist,natr14,vol16,vol96,vol_ts,vol_ratio,vol_z2880,parkinson96,rv_ratio,rsi14,bb96,bb_width96,adx14,di_diff14,cmf20,mfi14,vwap_dev96,vol_z96,taker_ratio96,taker_imb16,trade_intensity96,clv1,clv4,body1,batr,wasym1,lower_wick,rng1,volrat,gapatr,ou_hl96,acf1,vr4,rskew96,jump_share,semivar_asym,cusum_abs_z,btc_r96,btc_vol96,rel_btc96,beta384,tod_sin,tod_cos,dow_sin,dow_cos,don96";

fn cell() -> String {
    std::env::var("RCSR_COMPACT_CELL")
        .expect("set RCSR_COMPACT_CELL to the path of panel_hourly_compact.rcsb")
}

fn spec() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/targets/compact_15m_100.targets.json"
    )
    .to_string()
}

fn run(target: &str, expr: &str, extra: &[&str]) -> Output {
    let data = cell();
    let spec = spec();
    let mut args = vec![
        "--data", &data, "--target", target, "--vars", INPUTS, "--time", "ts", "--train-select",
        "ts >= 0 and ts < 1735473600", "--test-select", "ts >= 1735689600 and ts < 1751328000",
        "--fill", "drop", "--metric", "squared", "--cpu", "--no-dev-log", "--quiet",
        "--target-spec", &spec, "--eval-expr", expr, "--json",
    ];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(&args)
        .output()
        .expect("run rcsr")
}

fn json(o: &Output) -> Json {
    Json::parse(String::from_utf8_lossy(&o.stdout).trim()).unwrap_or_else(|| {
        panic!("not json: {}", String::from_utf8_lossy(&o.stderr));
    })
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing {k}"));
    }
    cur
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} not a number"))
}

#[test]
#[ignore = "needs RCSR_COMPACT_CELL (1.9 GB, not in git)"]
fn compact_cell_zero_parameter_rung_beats_the_searched_front() {
    let refused = run("y60", "(0-0.002)/(6*vol96)", &[]);
    assert_eq!(refused.status.code(), Some(5), "{}", String::from_utf8_lossy(&refused.stderr));

    let out = run("y60", "(0-0.002)/(6*vol96)", &["--allow-target-structure"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let j = json(&out);
    assert_eq!(num(&j, &["train_rows"]), 2_456_306.0);
    assert_eq!(num(&j, &["test_rows"]), 411_316.0);
    let rung_test = num(&j, &["target_structure", "test_error"]);
    assert!((rung_test - 0.104_582_698).abs() <= 5e-10, "{rung_test}");
    assert!(rung_test < 0.104_616_110 && rung_test < 0.104_658_228);
    let rung_train = num(&j, &["target_structure", "train_error"]);
    assert!((rung_train - 0.103_513_258_9).abs() <= 5e-10, "{rung_train}");
    assert!(rung_train < 0.104_616_110);
    assert!((num(&j, &["eval", "test_error"]) - 0.104_582_697_76).abs() <= 5e-10);
    assert!((num(&j, &["eval", "train_error"]) - 0.103_513_550_90).abs() <= 5e-10);
    assert_eq!(num(&j, &["target_spec", "horizon_seconds"]), 3600.0);
    assert_eq!(
        at(&j, &["target_spec", "window_seconds"]),
        &Json::Arr(vec![Json::Num(900.0), Json::Num(4500.0)])
    );
    assert_eq!(num(&j, &["target_spec", "required_purge_seconds"]), 4500.0);
    assert_eq!(at(&j, &["target_spec", "check", "status"]).as_str(), Some("overridden"));
    match at(&j, &["target_spec", "check", "proxies"]) {
        Json::Arr(p) => {
            assert_eq!(p[0].get("column").and_then(|c| c.as_str()), Some("parkinson96"));
            let rho = p[0].get("spearman").and_then(|r| r.as_f64()).unwrap();
            assert!((0.97..=0.99).contains(&rho), "{rho}");
        }
        other => panic!("{other:?}"),
    }
}

#[test]
#[ignore = "needs RCSR_COMPACT_CELL (1.9 GB, not in git)"]
fn compact_cell_raw_target_has_no_structure() {
    let out = run("fwd60", "0", &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let j = json(&out);
    assert_eq!(at(&j, &["target_spec", "structural_term"]), &Json::Bool(false));
    let mse = num(&j, &["target_structure", "test_error"]);
    assert!((mse - 1.322_536_784e-4).abs() <= 1e-12, "{mse}");
}
