//! Add-one-family ablation: arm definitions and manifests, placebo and
//! permuted columns, scoring of a real run, and the ship/drop verdict.

use rcsr::ablation::{self, Definition, SeedResult};
use rcsr::data;
use rcsr::devlog::Json;
use rcsr::ladder::{self, Fold, Options, Panel};
use rcsr::protocol::{self, WindowRow};
use rcsr::rng::Mt;
use rcsr::walkforward::Window;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

fn definition() -> Definition {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    ablation::parse_definition(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("rcsr-ablation-{}-{}", std::process::id(), name));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn gauss(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

// --- definition, manifests, plan ---------------------------------------------

#[test]
fn the_committed_definition_has_the_six_arms() {
    let d = definition();
    let sizes: Vec<(String, usize)> = d.arms.iter().map(|a| (a.name.clone(), d.features(a).len())).collect();
    assert_eq!(
        sizes,
        vec![
            ("A0".to_string(), 10),
            ("A1".to_string(), 16),
            ("A2".to_string(), 22),
            ("A3".to_string(), 26),
            ("A4".to_string(), 29),
            ("A5".to_string(), 22)
        ]
    );
    assert!((8..=12).contains(&d.b0.len()), "B0 holds 8 to 12 columns");
    let a4 = d.features(d.arm("A4").unwrap());
    assert_eq!(a4.iter().filter(|f| f.starts_with("placebo_")).count(), 3);
    assert_eq!(&a4[..26], &d.features(d.arm("A3").unwrap())[..], "A4 is A3 plus the placebo columns");
    let a5 = d.features(d.arm("A5").unwrap());
    assert_eq!(a5.iter().filter(|f| f.starts_with("perm_")).count(), 12);
    assert_eq!(&d.features(d.arm("A2").unwrap())[..16], &d.features(d.arm("A1").unwrap())[..]);
    let hashes: Vec<String> = d.arms.iter().map(|a| ablation::feature_list_sha256(&d.features(a))).collect();
    let mut uniq = hashes.clone();
    uniq.sort();
    uniq.dedup();
    assert_eq!(uniq.len(), hashes.len(), "every arm has its own feature-list hash");
    assert_eq!(d.rule.families.len(), 3);
    assert_eq!(d.generation_budget, 300);
}

#[test]
fn a_definition_with_an_unknown_group_is_refused() {
    let text = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json")).unwrap();
    let bad = text.replace("\"add\": [\"candle_continuous\"]}", "\"add\": [\"nope\"]}");
    assert!(ablation::parse_definition(&bad).unwrap_err().contains("unknown group"));
}

#[test]
fn the_plan_writes_one_manifest_per_arm_and_every_command() {
    let d = definition();
    let files = ablation::plan_files(&ablation::PlanInput {
        def: &d,
        data: "panel.rcsb",
        target: "y60",
        time: "ts",
        group: Some("symbol"),
        return_col: Some("fwd60"),
        windows: "protocol/windows.csv",
        scheme: "expanding",
        out_dir: "out",
        finalist: false,
        search_args: "--objective pnl --cost-bps 12 --validation-frac 0.2",
        score_args: "",
        verdict_args: "",
    });
    let get = |n: &str| files.iter().find(|(k, _)| k == n).unwrap_or_else(|| panic!("no {n}")).1.clone();
    for arm in &d.arms {
        let m = Json::parse(&get(&format!("{}.arm.json", arm.name))).unwrap();
        assert_eq!(m.get("arm").unwrap().as_str(), Some(arm.name.as_str()));
        let feats: Vec<String> = match m.get("features").unwrap() {
            Json::Arr(a) => a.iter().map(|x| x.as_str().unwrap().to_string()).collect(),
            _ => panic!(),
        };
        assert_eq!(feats, d.features(arm));
        assert_eq!(m.get("feature_list_sha256").unwrap().as_str().unwrap(), ablation::feature_list_sha256(&feats));
        assert_eq!(m.get("generation_budget").unwrap().as_f64(), Some(300.0));
        match m.get("seeds").unwrap() {
            Json::Arr(a) => assert_eq!(a.len(), 3),
            _ => panic!(),
        }
    }
    let script = get("commands.sh");
    assert_eq!(script.matches("--arm A").count(), 6 * 3, "one search per arm and seed");
    assert_eq!(script.matches("ablation score").count(), 6 * 3);
    assert!(script.contains("ablation prepare") && script.contains("ablation verdict"));
    assert!(script.contains("--generations 300"));

    let finalist = ablation::plan_files(&ablation::PlanInput {
        def: &d,
        data: "p.csv",
        target: "y",
        time: "ts",
        group: None,
        return_col: None,
        windows: "w.csv",
        scheme: "expanding",
        out_dir: "out",
        finalist: true,
        search_args: "",
        score_args: "",
        verdict_args: "",
    });
    let s = &finalist.iter().find(|(k, _)| k == "commands.sh").unwrap().1;
    assert_eq!(s.matches("--arm A").count(), 6 * 5, "five seeds for finalists");
}

// --- null columns -------------------------------------------------------------

fn ar1(n_groups: usize, len: usize, rho: f64, seed: u32) -> (Vec<f64>, Vec<u64>) {
    let mut rng = Mt::new(seed);
    let (mut x, mut g) = (Vec::new(), Vec::new());
    for k in 0..n_groups {
        let mut z = gauss(&mut rng);
        for _ in 0..len {
            z = rho * z + (1.0 - rho * rho).sqrt() * gauss(&mut rng);
            x.push(z);
            g.push(k as u64);
        }
    }
    (x, g)
}

fn corr(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut c, mut va, mut vb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        c += (x - ma) * (y - mb);
        va += (x - ma).powi(2);
        vb += (y - mb).powi(2);
    }
    c / (va * vb).sqrt()
}

#[test]
fn a_placebo_matches_prevalence_and_autocorrelation_and_is_unrelated() {
    let (z, g) = ar1(4, 3000, 0.7, 5);
    // A sparse, persistent pattern flag: 1 when the latent state is above its 85th percentile.
    let mut sorted = z.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let cut = sorted[(0.85 * sorted.len() as f64) as usize];
    let mut template: Vec<f64> = z.iter().map(|v| if *v > cut { 1.0 } else { 0.0 }).collect();
    template[0] = f64::NAN;
    template[3000] = f64::NAN;
    let placebo = ablation::placebo_column(&template, &g, 11, 1);
    let prevalence = |v: &[f64]| {
        let f: Vec<f64> = v.iter().copied().filter(|x| x.is_finite()).collect();
        f.iter().sum::<f64>() / f.len() as f64
    };
    assert_eq!(prevalence(&placebo), prevalence(&template), "the marginal is matched exactly");
    assert_eq!(placebo.iter().filter(|v| v.is_nan()).count(), 2);
    assert!(placebo[0].is_nan() && placebo[3000].is_nan());
    let (ta, pa) = (ablation::lag1_autocorr(&template, &g), ablation::lag1_autocorr(&placebo, &g));
    assert!(ta > 0.3, "the template is persistent: {ta}");
    assert!((ta - pa).abs() < 0.1, "autocorrelation template {ta}, placebo {pa}");
    let pairs: Vec<(f64, f64)> = z.iter().zip(&placebo).filter(|(_, p)| p.is_finite()).map(|(a, b)| (*a, *b)).collect();
    let (a, b): (Vec<f64>, Vec<f64>) = pairs.into_iter().unzip();
    assert!(corr(&a, &b).abs() < 0.05, "the placebo is unrelated to the latent state");
    let again = ablation::placebo_column(&template, &g, 11, 1);
    let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    assert_eq!(bits(&placebo), bits(&again), "reproducible");
    assert_ne!(placebo, ablation::placebo_column(&template, &g, 11, 2), "each placebo is its own draw");
}

#[test]
fn block_permutation_keeps_values_and_pairs_but_breaks_alignment() {
    let (x, g) = ar1(3, 2000, 0.5, 8);
    let y: Vec<f64> = x.iter().map(|v| 2.0 * v + 1.0).collect();
    let out = ablation::permute_columns(&[&x, &y], &g, 96, 4);
    for k in 0..3 {
        let range = k * 2000..(k + 1) * 2000;
        let mut a: Vec<f64> = x[range.clone()].to_vec();
        let mut b: Vec<f64> = out[0][range.clone()].to_vec();
        a.sort_by(|p, q| p.total_cmp(q));
        b.sort_by(|p, q| p.total_cmp(q));
        assert_eq!(a, b, "group {k} keeps its multiset");
    }
    for i in 0..x.len() {
        assert_eq!(out[1][i], 2.0 * out[0][i] + 1.0, "columns of one family stay paired");
    }
    assert!(corr(&x, &out[0]).abs() < 0.2, "alignment with the original is gone: {}", corr(&x, &out[0]));
    assert_eq!(out, ablation::permute_columns(&[&x, &y], &g, 96, 4));
    let mut rng = Mt::new(1);
    let order = ablation::block_permutation(1000, 96, &mut rng);
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..1000).collect::<Vec<_>>());
}

#[test]
fn prepare_writes_placebo_and_permuted_columns_next_to_the_real_ones() {
    let text = r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1","f2"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":10,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam","pat"],"placebo":2},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#;
    let d = ablation::parse_definition(text).unwrap();
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    let mut rng = Mt::new(2);
    for s in ["b", "a"] {
        for t in (0..300).rev() {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", 1_000_000 + t * 60, s, gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), (t % 7 == 0) as i32));
        }
    }
    let table = data::parse_table(&csv).unwrap();
    let out = ablation::prepare_csv(&ablation::PrepareInput { table: &table, def: &d, time: "ts", group: "sym", keep: &["y".to_string()], seed: 3 }).unwrap();
    let prepared = data::parse_table(&out).unwrap();
    assert_eq!(prepared.names, vec!["ts", "sym", "y", "b", "f1", "f2", "p1", "placebo_1", "placebo_2", "perm_f1", "perm_f2"]);
    assert_eq!(prepared.len(), 600);
    let ts = prepared.col(0);
    assert!(ts[..300].windows(2).all(|w| w[0] <= w[1]), "rows are sorted by group and time");
    let p1 = prepared.col(prepared.index_of("placebo_1").unwrap());
    let pat = prepared.col(prepared.index_of("p1").unwrap());
    assert_eq!(p1.iter().sum::<f64>(), pat.iter().sum::<f64>(), "placebo_1 copies the prevalence of its template");
}

// --- scoring a run --------------------------------------------------------------

const T0: f64 = 1_600_000_000.0;

fn objective_csv(dir: &std::path::Path) -> String {
    let n = 400usize;
    let frac = |v: f64| v - v.floor();
    let mut text = String::from("t,x1,x2,r,y\n");
    for i in 0..n {
        let f = i as f64;
        let x1 = frac(f * 0.618_033_988_749_894_9) - 0.5;
        let x2 = frac(f * 0.754_877_666_246_692_7) - 0.5;
        let noise = frac(f * 0.324_717_957_244_746) - 0.5;
        let r = 0.01 * x1 + 0.0005 * noise;
        text.push_str(&format!("{},{},{},{},{}\n", (T0 as i64) + 900 * i as i64, x1, x2, r, r));
    }
    let path = dir.join("d.csv");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn write_target_spec(dir: &std::path::Path) -> String {
    let text = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"y\",\"name\":\"raw_target\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"r\",\"return_units\":\"log\",\
\"horizon_seconds\":3600,\"entry_offset_seconds\":900,\"delay_seconds\":0,\
\"entry_price\":\"next open\",\"exit_price\":\"close\",\"offset\":0.0,\
\"scale\":{\"kind\":\"constant\",\"value\":1.0},\"clip\":null,\"units\":\"log_return\",\
\"cost_bps\":null,\"cost_in_label\":false,\"construction_columns\":[],\"note\":null}]}";
    let path = dir.join("spec.json");
    std::fs::write(&path, text).unwrap();
    path.to_string_lossy().into_owned()
}

fn search(data: &str, spec: &str, extra: &[&str]) -> std::process::Output {
    let mut args = vec![
        "--data", data, "--target", "y", "--vars", "x1,x2", "--time", "t", "--split", "temporal", "--target-spec", spec, "--cpu",
        "--no-dev-log", "--seed", "3", "--generations", "15", "--pop", "32", "--metric", "squared", "--quiet",
    ];
    args.extend_from_slice(extra);
    Command::new(env!("CARGO_BIN_EXE_rcsr")).args(&args).output().expect("run rcsr")
}

#[test]
fn run_json_records_arm_feature_hash_and_generation_budget() {
    let dir = scratch("runjson");
    let data = objective_csv(&dir);
    let spec = write_target_spec(&dir);
    let out = search(&data, &spec, &["--json", "--arm", "A7"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let j = Json::parse(String::from_utf8_lossy(&out.stdout).trim()).unwrap();
    let a = j.get("ablation").expect("ablation object");
    assert_eq!(a.get("arm").unwrap().as_str(), Some("A7"));
    assert_eq!(a.get("feature_list_sha256").unwrap().as_str(), Some(ablation::feature_list_sha256(&["x1".to_string(), "x2".to_string()]).as_str()));
    assert_eq!(a.get("generation_budget").unwrap().as_f64(), Some(15.0));
    let reached = a.get("generations_run").unwrap().as_f64().unwrap();
    assert!(reached >= 1.0 && reached <= 15.0);
    let plain = search(&data, &spec, &["--json"]);
    assert!(Json::parse(String::from_utf8_lossy(&plain.stdout).trim()).unwrap().get("ablation").is_none(), "no --arm, no ablation object");
}

#[test]
fn a_real_run_is_scored_on_the_folds_without_the_test_block_in_selection() {
    let dir = scratch("score");
    let data = objective_csv(&dir);
    let spec = write_target_spec(&dir);
    let run_json = dir.join("A0-s3.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(290),
            iso(400)
        ),
    )
    .unwrap();
    let res = dir.join("A0-s3.result.json");
    let score = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r", "--time", "t",
            "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", res.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(score.status.code(), Some(0), "{}", String::from_utf8_lossy(&score.stderr));
    let r = SeedResult::parse(&std::fs::read_to_string(&res).unwrap()).unwrap();
    assert_eq!(r.arm, "A0");
    assert_eq!(r.seed, 3);
    assert_eq!(r.generation_budget, 15);
    assert_eq!(r.features, vec!["x1", "x2"]);
    assert_eq!(r.folds, vec!["w0"]);
    assert_eq!(r.thresholds, vec![0.6, 0.7, 0.8]);
    assert_eq!(r.net_by_fold_threshold.len(), 1);
    assert_eq!(r.net_by_fold_threshold[0].len(), 3);
    assert_eq!(r.net_period.len(), 110);
    assert!(r.return_mse.is_finite());
    assert!((r.net_by_fold_threshold[0][1] - r.net_period.iter().sum::<f64>()).abs() < 1e-12, "headline threshold is the series");
    assert!(r.net_period.iter().sum::<f64>() > 0.0, "the long book wins on a signal-driven return");
    assert_eq!(r.feature_list_sha256, ablation::feature_list_sha256(&r.features));

    let plain = dir.join("plain.run.json");
    let out = search(&data, &spec, &["--json-out", plain.to_str().unwrap(), "--arm", "A0"]);
    assert_eq!(out.status.code(), Some(0));
    let bad = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "score", "--run", plain.to_str().unwrap(), "--data", &data, "--target", "y", "--time", "t", "--windows", windows.to_str().unwrap(), "--out", dir.join("x.json").to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(bad.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("objective.finalist"), "a run without a validation-chosen finalist is refused");
}

#[test]
fn parse_run_refuses_a_hash_that_does_not_match_the_variables() {
    let text = r#"{"seed":1,"variables":["a","b"],"ablation":{"arm":"A0","feature_list_sha256":"00","generation_budget":5,"generations_run":5},
"objective":{"finalist":{"expression":"a"}}}"#;
    assert!(ablation::parse_run(text).unwrap_err().contains("does not match"));
}

#[test]
fn score_run_reads_the_expression_and_its_inputs() {
    let mut rng = Mt::new(4);
    let (mut time, mut group, mut s, mut n, mut ret) = (vec![], vec![], vec![], vec![], vec![]);
    for g in 0..4u64 {
        for t in 0..300 {
            let sig = gauss(&mut rng);
            time.push(T0 + t as f64 * 3600.0);
            group.push(g);
            s.push(sig);
            n.push(gauss(&mut rng));
            ret.push(0.004 * sig + 0.002 * gauss(&mut rng));
        }
    }
    let panel = Panel::new(vec!["s".into(), "n".into()], vec![s, n], time, group, ret.clone(), ret, vec![]).unwrap();
    let rows = vec![WindowRow {
        scheme: "expanding".into(),
        window: Window { label: "w".into(), train_start: T0, train_end: T0 + 200.0 * 3600.0, test_start: T0 + 205.0 * 3600.0, test_end: T0 + 300.0 * 3600.0 },
        validation_start: T0 + 120.0 * 3600.0,
    }];
    let folds: Vec<Fold> = ladder::make_folds(&panel, &rows, 3600.0, None).unwrap();
    let facts = ablation::RunFacts {
        arm: "A1".into(),
        feature_list_sha256: ablation::feature_list_sha256(&["s".into(), "n".into()]),
        generation_budget: 10,
        generations_run: 7,
        variables: vec!["s".into(), "n".into()],
        expression: "s".into(),
        seed: 2,
    };
    let r = ablation::score_run(&facts, &panel, &folds, &[0.5, 0.7], 1, &Options::default()).unwrap();
    assert_eq!(r.features_used, vec!["s"]);
    assert_eq!(r.generations_run, 7);
    assert_eq!(r.net_period.len(), 95);
    assert!(r.net_by_fold_threshold[0][1] > 0.0);
}

// --- verdict ------------------------------------------------------------------------

struct Scenario {
    mse: BTreeMap<&'static str, f64>,
    /// Mean per-period net per arm.
    net: BTreeMap<&'static str, f64>,
    mse_noise: f64,
    /// Arm and fold whose net difference is pushed negative.
    flip: Option<(&'static str, usize)>,
}

fn results(d: &Definition, sc: &Scenario, seeds: u32) -> Vec<SeedResult> {
    let mut out = Vec::new();
    for (ai, arm) in d.arms.iter().enumerate() {
        for s in 1..=seeds {
            let mut rng = Mt::new(1000 + s + 97 * ai as u32);
            let mut period = Vec::new();
            for _ in 0..360 {
                period.push(sc.net[arm.name.as_str()] + 0.002 * gauss(&mut rng));
            }
            // The first 100 periods belong to fold 0, and so on.
            let folds = 3usize;
            let thresholds = vec![0.6, 0.7, 0.8];
            let mut by = Vec::new();
            for f in 0..folds {
                let base: f64 = period[f * 120..(f + 1) * 120].iter().sum();
                let mut row = Vec::new();
                for q in 0..3 {
                    let mut v = base * (1.0 + 0.05 * q as f64);
                    if let Some((a, ff)) = sc.flip {
                        if a == arm.name && ff == f {
                            v -= 1.0;
                        }
                    }
                    row.push(v);
                }
                by.push(row);
            }
            let feats = d.features(arm);
            out.push(SeedResult {
                arm: arm.name.clone(),
                seed: s,
                feature_list_sha256: ablation::feature_list_sha256(&feats),
                generation_budget: d.generation_budget,
                generations_run: d.generation_budget,
                features: feats.clone(),
                features_used: feats[..2].to_vec(),
                return_mse: sc.mse[arm.name.as_str()] + sc.mse_noise * gauss(&mut rng),
                folds: vec!["f0".into(), "f1".into(), "f2".into()],
                thresholds,
                net_by_fold_threshold: by,
                net_period: period,
            });
        }
    }
    out
}

fn base_scenario() -> Scenario {
    let mse: BTreeMap<_, _> = [("A0", 1.0), ("A1", 0.90), ("A2", 0.95), ("A3", 0.95), ("A4", 0.95), ("A5", 1.0)].into_iter().collect();
    let net: BTreeMap<_, _> = [("A0", 0.0), ("A1", 0.0015), ("A2", 0.0005), ("A3", 0.0005), ("A4", 0.0005), ("A5", 0.0)].into_iter().collect();
    Scenario { mse, net, mse_noise: 0.004, flip: None }
}

fn verdict_of(sc: &Scenario, seeds: u32) -> Result<ablation::Verdict, String> {
    let d = definition();
    ablation::verdict(&d, &results(&d, sc, seeds), &BTreeMap::new())
}

fn family<'a>(v: &'a ablation::Verdict, name: &str) -> &'a ablation::FamilyVerdict {
    v.families.iter().find(|f| f.name == name).unwrap()
}

#[test]
fn a_real_family_ships_and_one_without_a_gain_over_its_control_is_dropped() {
    let v = verdict_of(&base_scenario(), 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "ship", "failed: {:?}", candle.failed);
    assert!(candle.comparisons[0].gain_mse > candle.comparisons[0].seed_range);
    assert!(candle.comparisons[0].spa.as_ref().unwrap().p_consistent < 0.05);
    let multi = family(&v, "multi_bar");
    assert_eq!(multi.verdict, "drop", "A2 equals A1: no gain");
    assert!(multi.failed.iter().any(|f| f.starts_with("gain_exceeds_seed_range")));
    let patterns = family(&v, "patterns_trend");
    assert_eq!(patterns.verdict, "drop", "A3 does not beat the placebo arm A4");
    assert!(patterns.failed.iter().any(|f| f.contains("vs A4")), "{:?}", patterns.failed);
    let js = v.json();
    for needle in ["\"verdict\":\"ship\"", "\"verdict\":\"drop\"", "\"n_trials\"", "\"dsr\"", "\"pbo\"", "\"bh_adjusted\"", "\"null_band\"", "\"generation_budget\"", "\"feature_list_sha256\""] {
        assert!(js.contains(needle), "verdict json lacks {needle}");
    }
    assert!(v.markdown().contains("| candle | A1 | ship |"));
    let d = definition();
    assert_eq!(v.n_trials, d.n_trials(3, 3));
    assert!(v.notes.iter().any(|n| n.contains("selection on training data repeated inside every fold")));
}

#[test]
fn a_gain_inside_the_null_band_is_dropped() {
    let mut sc = base_scenario();
    sc.mse.insert("A5", 0.90);
    sc.net.insert("A5", 0.0015);
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("outside_null_band")), "{:?}", candle.failed);
    let (lo, hi) = candle.null_band.unwrap();
    assert!(lo <= hi && hi > 0.05, "the permuted arm reaches the family's gain: {lo} {hi}");
}

#[test]
fn a_gain_smaller_than_the_seed_range_is_dropped() {
    let mut sc = base_scenario();
    sc.mse_noise = 0.2;
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("gain_exceeds_seed_range")));
}

#[test]
fn a_sign_flip_across_quarters_drops_the_family() {
    let mut sc = base_scenario();
    sc.flip = Some(("A1", 2));
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("no_sign_flip")), "{:?}", candle.failed);
    assert!(candle.comparisons[0].sign_by_fold.iter().any(|x| *x < 0.0));
    assert!(candle.comparisons[0].sign_by_fold.iter().any(|x| *x > 0.0));
}

#[test]
fn a_family_that_does_not_survive_fdr_or_the_deflated_sharpe_is_dropped() {
    let mut sc = base_scenario();
    sc.net.insert("A1", 0.0001);
    let v = verdict_of(&sc, 3).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert!(candle.failed.iter().any(|f| f.starts_with("spa_p_below_alpha") || f.starts_with("fdr_survives") || f.starts_with("dsr_above_min")), "{:?}", candle.failed);
}

#[test]
fn inconsistent_inputs_are_refused_not_averaged() {
    let d = definition();
    let sc = base_scenario();
    let all = results(&d, &sc, 3);
    // too few seeds
    let two: Vec<SeedResult> = all.iter().filter(|r| r.seed <= 2).cloned().collect();
    assert!(ablation::verdict(&d, &two, &BTreeMap::new()).unwrap_err().contains("seeds"));
    // a different generation budget in one run
    let mut budget = all.clone();
    budget[0].generation_budget = 299;
    assert!(ablation::verdict(&d, &budget, &BTreeMap::new()).unwrap_err().contains("generation budget"));
    // a feature list that is not the arm's
    let mut hash = all.clone();
    hash[4].feature_list_sha256 = "deadbeef".into();
    assert!(ablation::verdict(&d, &hash, &BTreeMap::new()).unwrap_err().contains("feature_list_sha256"));
    // a hash that differs from the arm manifest
    let mut expected = BTreeMap::new();
    expected.insert("A0".to_string(), "other".to_string());
    assert!(ablation::verdict(&d, &all, &expected).unwrap_err().contains("arm manifest"));
    // different seeds across the compared arms
    let mut seeds = all.clone();
    for r in seeds.iter_mut().filter(|r| r.arm == "A1" && r.seed == 3) {
        r.seed = 9;
    }
    assert!(ablation::verdict(&d, &seeds, &BTreeMap::new()).unwrap_err().contains("different seeds"));
}

#[test]
fn the_cli_verdict_reads_a_directory_of_result_files() {
    let dir = scratch("verdict");
    let d = definition();
    for r in results(&d, &base_scenario(), 3) {
        std::fs::write(dir.join(format!("{}-s{}.result.json", r.arm, r.seed)), r.json()).unwrap();
    }
    let def_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    let out = dir.join("verdict.json");
    let md = dir.join("verdict.md");
    let run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["ablation", "verdict", "--definition", def_path.to_str().unwrap(), "--results", dir.to_str().unwrap(), "--out", out.to_str().unwrap(), "--markdown", md.to_str().unwrap()])
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    let j = Json::parse(std::fs::read_to_string(&out).unwrap().trim()).unwrap();
    assert_eq!(j.get("schema").unwrap().as_str(), Some("rcsr-ablation-verdict/1"));
    assert!(std::fs::read_to_string(&md).unwrap().contains("ship"));
}

#[test]
fn the_cli_plan_writes_manifests_to_a_directory() {
    let dir = scratch("plan");
    let data = dir.join("panel.csv");
    let d = definition();
    let mut header = vec!["ts".to_string(), "sym".to_string(), "y".to_string()];
    header.extend(d.real_columns());
    std::fs::write(&data, header.join(",") + "\n").unwrap();
    let def_path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../protocol/ablation.json");
    let run = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "plan", "--definition", def_path.to_str().unwrap(), "--data", data.to_str().unwrap(), "--target", "y", "--time", "ts",
            "--group", "sym", "--windows", "protocol/windows.csv", "--out", dir.join("out").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(run.status.code(), Some(0), "{}", String::from_utf8_lossy(&run.stderr));
    for arm in ["A0", "A1", "A2", "A3", "A4", "A5"] {
        let m = std::fs::read_to_string(dir.join("out").join(format!("{arm}.arm.json"))).unwrap();
        assert!(m.contains(&format!("\"arm\":\"{arm}\"")) && m.contains("feature_list_sha256") && m.contains("\"generation_budget\":300"));
    }
    assert!(dir.join("out/commands.sh").exists() && dir.join("out/definition.json").exists());
    let missing = std::fs::write(&data, "ts,sym,y\n").and_then(|_| {
        Ok(Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "plan", "--definition", def_path.to_str().unwrap(), "--data", data.to_str().unwrap(), "--target", "y", "--time", "ts",
                "--windows", "w.csv", "--out", dir.join("out2").to_str().unwrap(),
            ])
            .output()
            .unwrap())
    });
    let missing = missing.unwrap();
    assert_eq!(missing.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&missing.stderr).contains("missing columns"));
}

#[test]
fn a_family_that_loses_only_on_pbo_is_dropped_with_reason_pbo() {
    let sc = base_scenario();
    let d = definition();
    let rs = results(&d, &sc, 3);
    let ok = ablation::verdict(&d, &rs, &BTreeMap::new()).unwrap();
    let pbo = ok.pbo.as_ref().expect("pbo computed").pbo;
    assert!(pbo < d.rule.max_pbo, "pbo {pbo}");
    assert_eq!(d.rule.max_pbo, 0.5);
    assert_eq!(family(&ok, "candle").verdict, "ship");

    let mut strict = d.clone();
    strict.rule.max_pbo = pbo / 2.0;
    let v = ablation::verdict(&strict, &rs, &BTreeMap::new()).unwrap();
    let candle = family(&v, "candle");
    assert_eq!(candle.verdict, "drop");
    assert_eq!(candle.failed.len(), 1, "{:?}", candle.failed);
    assert!(candle.failed[0].starts_with("pbo:"), "{:?}", candle.failed);
}

#[test]
fn a_declared_locked_test_is_refused_by_score_and_carried_by_prepare() {
    use rcsr::lockbox::{self, Lockbox};

    let dir = scratch("locked-test");
    let data = objective_csv(&dir);
    let spec = write_target_spec(&dir);
    let run_json = dir.join("A0-s3.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    let windows = dir.join("w.csv");
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(290),
            iso(400)
        ),
    )
    .unwrap();
    let lb = Lockbox { start: T0 as i64 + 900 * 300, end: T0 as i64 + 900 * 400 };
    lockbox::write_sidecar(&data, &lb, "test").unwrap();
    let score = |res: &str| {
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r",
                "--time", "t", "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", res,
            ])
            .output()
            .unwrap()
    };
    let refused = score(dir.join("refused.json").to_str().unwrap());
    assert_eq!(refused.status.code(), Some(3));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("declared locked test"), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(!dir.join("refused.json").exists());

    // prepare copies the declaration to a sidecar and a search on the CSV refuses the rows
    let def = dir.join("def.json");
    std::fs::write(
        &def,
        r#"{"schema":"rcsr-ablation/1","b0":["b"],"groups":{"fam":["f1","f2"],"pat":["p1"]},
"placebo_template_group":"pat","generation_budget":10,
"arms":[{"name":"A0","add":[]},{"name":"A1","add":["fam"]},{"name":"A5","add":[],"permute":["fam"]}],
"rule":{"families":[{"name":"fam","treatment":"A1","controls":["A0"],"null_arm":"A5","null_control":"A0"}]}}"#,
    )
    .unwrap();
    let mut csv = String::from("ts,sym,y,b,f1,f2,p1\n");
    let mut rng = Mt::new(2);
    for s in ["a", "b"] {
        for k in 0..400i64 {
            csv.push_str(&format!("{},{},{},{},{},{},{}\n", T0 as i64 + 900 * k, s, gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), gauss(&mut rng), (k % 7 == 0) as i32));
        }
    }
    let raw = dir.join("raw.csv");
    std::fs::write(&raw, csv).unwrap();
    lockbox::write_sidecar(raw.to_str().unwrap(), &lb, "test").unwrap();
    let prepared = dir.join("prepared.csv");
    let prep = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "ablation", "prepare", "--definition", def.to_str().unwrap(), "--data", raw.to_str().unwrap(), "--time", "ts", "--group", "sym",
            "--keep", "y", "--out", prepared.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert_eq!(prep.status.code(), Some(0), "{}", String::from_utf8_lossy(&prep.stderr));
    assert_eq!(lockbox::stored(prepared.to_str().unwrap()).unwrap().map(|(_, l)| l), Some(lb));
    let found = Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args([
            "--data", prepared.to_str().unwrap(), "--target", "y", "--vars", "b,f1,f2", "--time", "ts", "--split", "temporal", "--cpu",
            "--no-dev-log", "--seed", "3", "--generations", "3", "--pop", "16", "--quiet",
        ])
        .output()
        .unwrap();
    assert_eq!(found.status.code(), Some(3), "{}", String::from_utf8_lossy(&found.stderr));
    assert!(String::from_utf8_lossy(&found.stderr).contains("declared locked test"));
}

#[test]
fn stale_sidecar_is_refused() {
    use rcsr::lockbox::{self, Lockbox};

    let dir = scratch("stale-sidecar");
    let data = objective_csv(&dir);
    let spec900 = dir.join("spec900.json");
    std::fs::write(&spec900, std::fs::read_to_string(write_target_spec(&dir)).unwrap().replace("\"horizon_seconds\":3600", "\"horizon_seconds\":900")).unwrap();
    let spec = spec900.to_str().unwrap().to_string();
    let run_json = dir.join("A0.run.json");
    let out = search(&data, &spec, &["--objective", "pnl", "--cost-bps", "5", "--validation-frac", "0.3", "--arm", "A0", "--json-out", run_json.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let windows = dir.join("w.csv");
    let iso = |i: i64| protocol::iso(T0 + 900.0 * i as f64);
    std::fs::write(
        &windows,
        format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nexpanding,w0,{},{},{},{},{}\n",
            iso(0),
            iso(280),
            iso(200),
            iso(290),
            iso(400)
        ),
    )
    .unwrap();
    let score = || {
        Command::new(env!("CARGO_BIN_EXE_rcsr"))
            .args([
                "ablation", "score", "--run", run_json.to_str().unwrap(), "--data", &data, "--target", "y", "--return-col", "r",
                "--time", "t", "--windows", windows.to_str().unwrap(), "--purge-seconds", "1800", "--out", dir.join("o.json").to_str().unwrap(),
            ])
            .output()
            .unwrap()
    };

    let lb = Lockbox { start: T0 as i64 + 900 * 300, end: T0 as i64 + 900 * 400 };
    lockbox::write_sidecar(&data, &lb, "test").unwrap();
    let text = std::fs::read_to_string(lockbox::sidecar_path(&data)).unwrap();
    assert!(text.contains("\"data_bytes\":") && text.contains("\"data_sha256\":"), "{text}");
    assert!(lockbox::resolve(None, &data).is_ok());

    // same length, one digit changed: the SHA-256 differs
    let original = std::fs::read(&data).unwrap();
    let mut changed = original.clone();
    let i = changed.iter().rposition(|c| c.is_ascii_digit()).unwrap();
    changed[i] = if changed[i] == b'0' { b'1' } else { b'0' };
    std::fs::write(&data, &changed).unwrap();
    let e = lockbox::resolve(None, &data).unwrap_err();
    assert_eq!(e.exit, 3, "{}", e.message);
    assert!(e.message.contains("stale") && e.message.contains("sha256"), "{}", e.message);
    let found = search(&data, &spec, &[]);
    assert_eq!(found.status.code(), Some(3), "{}", String::from_utf8_lossy(&found.stderr));
    assert!(String::from_utf8_lossy(&found.stderr).contains("stale"), "{}", String::from_utf8_lossy(&found.stderr));
    let refused = score();
    assert_eq!(refused.status.code(), Some(3), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(String::from_utf8_lossy(&refused.stderr).contains("stale"), "{}", String::from_utf8_lossy(&refused.stderr));
    assert!(!dir.join("o.json").exists());

    // another length is refused by the byte count
    let mut longer = original.clone();
    longer.push(b'\n');
    std::fs::write(&data, &longer).unwrap();
    let e = lockbox::resolve(None, &data).unwrap_err();
    assert_eq!(e.exit, 3, "{}", e.message);
    assert!(e.message.contains("bytes"), "{}", e.message);

    // the original bytes are accepted again
    std::fs::write(&data, &original).unwrap();
    assert!(lockbox::resolve(None, &data).is_ok());
}
