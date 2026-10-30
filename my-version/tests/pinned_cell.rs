//! Validation gates on the pinned compact 10 h cell (#257, evidence E1a-E3).
//!
//! The cell is 1.9 GB and stays out of git. Run with
//! `RCSR_COMPACT_CELL=<path>/panel_hourly_compact.rcsb cargo test --release --test pinned_cell -- --ignored --nocapture`.
//! The six printed laws of the 10 h run are seeded with `--fit-iters 0`, so the
//! constants stay as printed and the errors are those of the audit.

use rcsr::devlog::Json;
use std::process::{Command, Output};

const INPUTS: &str = "r16,r96,r672,ma96,ma672,ma_slope672,macd_hist,natr14,vol16,vol96,vol_ts,vol_ratio,vol_z2880,parkinson96,rv_ratio,rsi14,bb96,bb_width96,adx14,di_diff14,cmf20,mfi14,vwap_dev96,vol_z96,taker_ratio96,taker_imb16,trade_intensity96,clv1,clv4,body1,batr,wasym1,lower_wick,rng1,volrat,gapatr,ou_hl96,acf1,vr4,rskew96,jump_share,semivar_asym,cusum_abs_z,btc_r96,btc_vol96,rel_btc96,beta384,tod_sin,tod_cos,dow_sin,dow_cos,don96";

const LAWS: [&str; 6] = [
    "-0.0827",
    "(-0.1071 * gauss((19.9879 * (0.001 + rel_btc96))))",
    "(-0.055 * (0.293 ^ (268.4249 * (-0.0054 + vol96))))",
    "(-0.0188 / (-0.175 + logistic((268.4249 * (-0.0054 + vol96)))))",
    "(-0.0252 + (-0.0261 / (-0.2172 + exp((268.4249 * (-0.0054 + vol96))))))",
    "((-0.0252 + (-0.0261 / (-0.2172 + exp((268.4249 * (-0.0054 + vol96)))))) + (0.0074 * (-0.0108 - (475.2189 * (-0 + macd_hist)))))",
];

fn cell() -> String {
    std::env::var("RCSR_COMPACT_CELL")
        .expect("set RCSR_COMPACT_CELL to the path of panel_hourly_compact.rcsb")
}

fn run(extra: &[&str]) -> Json {
    let data = cell();
    let spec = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/targets/compact_15m_100.targets.json"
    );
    let mut args: Vec<&str> = vec![
        "--data", &data, "--target", "y60", "--vars", INPUTS, "--time", "ts", "--train-select",
        "ts >= 0 and ts < 1735473600", "--test-select", "ts >= 1735689600 and ts < 1751328000",
        "--fill", "drop", "--metric", "squared", "--cpu", "--no-dev-log", "--quiet",
        "--target-spec", spec, "--allow-target-structure", "--generations", "1", "--pop", "8",
        "--fit-iters", "0", "--seed", "1", "--json",
    ];
    for law in LAWS {
        args.push("--seed-expr");
        args.push(law);
    }
    args.extend_from_slice(extra);
    let out: Output = Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().expect("run rcsr");
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    Json::parse(String::from_utf8_lossy(&out.stdout).trim()).expect("json")
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    let mut cur = j;
    for k in path {
        cur = cur.get(k).unwrap_or_else(|| panic!("missing {k} in {path:?}"));
    }
    cur
}

fn num(j: &Json, path: &[&str]) -> f64 {
    at(j, path).as_f64().unwrap_or_else(|| panic!("{path:?} not a number"))
}

fn members(j: &Json) -> &Vec<Json> {
    match at(j, &["pareto"]) {
        Json::Arr(a) => a,
        other => panic!("{other:?}"),
    }
}

fn expression(m: &Json) -> &str {
    at(m, &["expression"]).as_str().expect("expression")
}

/// The seeded r5 law: the exponential hyperbola without the macd term.
fn r5(j: &Json) -> &Json {
    members(j)
        .iter()
        .find(|m| expression(m).contains("exp(") && !expression(m).contains("macd_hist"))
        .expect("r5 law in the front")
}

/// The seeded r6 law: r5 plus the linear macd term.
fn r6(j: &Json) -> &Json {
    members(j)
        .iter()
        .find(|m| expression(m).contains("exp(") && expression(m).contains("macd_hist"))
        .expect("r6 law in the front")
}

fn print_summary(tag: &str, j: &Json) {
    println!("== {tag}");
    for r in ["zero_param", "affine", "binned20", "construction_ols"] {
        let rung = at(j, &["baselines", "target_structure", "rungs", r]);
        println!("rung {r}: {rung:?}");
    }
    println!("selection: {:?}", at(j, &["selection"]));
    println!("noise: {:?}", at(j, &["noise"]));
    println!("timing: {:?}", at(j, &["timing"]));
    for m in members(j) {
        println!(
            "rank {} cx {} test {} expr {} beats_zero_param_test {:?} beats_mean_test {:?} gates {:?} cmp_test {:?} ret {:?}",
            num(m, &["rank"]),
            num(m, &["complexity"]),
            num(m, &["test_error"]),
            expression(m),
            at(m, &["beats_zero_param_test"]),
            at(m, &["beats_mean_test"]),
            at(m, &["gates"]),
            at(m, &["comparisons", "test"]),
            at(m, &["return_space"]),
        );
    }
}

fn rung_checks(j: &Json) {
    let t = |name: &str| num(j, &["baselines", "target_structure", "rungs", name, "test_error"]);
    assert!((t("zero_param") - 0.104_582_698).abs() <= 1e-8, "{}", t("zero_param"));
    // The affine rung is fitted on fit-train only, so it stays close to the
    // zero-parameter rung out of sample.
    assert!((t("affine") - t("zero_param")).abs() <= 1e-4, "{}", t("affine"));
    assert!((t("binned20") - 0.104_662).abs() <= 1e-5, "{}", t("binned20"));
    assert!((t("construction_ols") - 0.105_127).abs() <= 2e-6, "{}", t("construction_ols"));
    for m in members(j) {
        assert_eq!(at(m, &["beats_zero_param_test"]), &Json::Bool(false), "rank {}", num(m, &["rank"]));
    }
}

#[test]
#[ignore = "needs RCSR_COMPACT_CELL (1.9 GB, not in git)"]
fn e1a_rungs_and_flags_without_a_validation_slice() {
    let j = run(&[]);
    print_summary("E1a", &j);
    rung_checks(&j);
    assert_eq!(at(&j, &["selection", "verdict"]).as_str(), Some("unvalidated"));
}

#[test]
#[ignore = "needs RCSR_COMPACT_CELL (1.9 GB, not in git)"]
fn e1b_e2_e3_verdict_noise_and_identification_with_a_validation_slice() {
    let j = run(&["--validation-frac", "0.1"]);
    print_summary("E1b", &j);
    // The rungs fit on 90 % of the training rows here, so only the
    // zero-parameter rung (no fitted parameter) keeps its E1a value.
    let zero = num(&j, &["baselines", "target_structure", "rungs", "zero_param", "test_error"]);
    assert!((zero - 0.104_582_698).abs() <= 1e-8, "{zero}");
    for m in members(&j) {
        assert_eq!(at(m, &["beats_zero_param_test"]), &Json::Bool(false));
    }
    assert_eq!(at(&j, &["selection", "verdict"]).as_str(), Some("no_signal"));
    assert_eq!(at(&j, &["selection", "selected_rank"]), &Json::Null);
    let search = num(&j, &["timing", "search_seconds"]);
    let gates = num(&j, &["timing", "gates_seconds"]);
    println!("timing.gates {gates} vs search {search} ({:.2} %)", 100.0 * gates / search);

    // E2: paired Driscoll-Kraay SE of the printed r5 law on the test block.
    let r5 = r5(&j);
    let vs_mean = num(r5, &["comparisons", "test", "mean", "se"]);
    let vs_zero = num(r5, &["comparisons", "test", "target_structure.zero_param", "se"]);
    println!("E2 r5 vs mean se {vs_mean} delta {}", num(r5, &["comparisons", "test", "mean", "delta"]));
    println!("E2 r5 vs zero_param se {vs_zero} delta {}", num(r5, &["comparisons", "test", "target_structure.zero_param", "delta"]));
    assert!((1.5e-4..=2.3e-4).contains(&vs_mean), "{vs_mean}");
    assert!((1.6e-5..=2.5e-5).contains(&vs_zero), "{vs_zero}");

    // E3: the printed r6 law is structurally unidentified.
    let r6 = r6(&j);
    println!("E3 r6 gates {:?} constant_errors {:?}", at(r6, &["gates"]), at(r6, &["constant_errors"]));
    assert_eq!(at(r6, &["gates", "identified"]), &Json::Bool(false));
    println!("E3 r6 involved {:?} rank {:?}", at(r6, &["constant_errors", "involved"]), at(r6, &["constant_errors", "rank"]));
    assert_eq!(at(r6, &["constant_errors", "ill_conditioned"]), &Json::Bool(true));
}
