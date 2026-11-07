//! #733: Auto dispatch routes from a pinned calibration, so a seeded run with
//! the default `--dispatch auto` gives one front and one route ledger in every
//! process, and `rcsr calibrate` is the only writer of the calibration file.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::path::Path;
use std::process::{Output, Stdio};

fn csv() -> String {
    let mut text = String::from("x0,x1,y\n");
    for i in 0..300 {
        let a = ((i * 37) % 101) as f64 / 25.0 - 2.0;
        let b = ((i * 53) % 97) as f64 / 24.0 - 2.0;
        let noise = (((i * 29) % 13) as f64 - 6.0) / 100.0;
        text.push_str(&format!("{a},{b},{}\n", 1.5 * (0.8 * a).sin() + 0.3 * b + noise));
    }
    text
}

/// One CLI search. `cache` is the private `XDG_CACHE_HOME`, so no file of the
/// machine's real cache is read.
fn search(cache: &Path, extra: &[&str]) -> Output {
    let mut child = rcsr_cmd::rcsr_for(extra)
        .args([
            "--data", "-", "--target", "y", "--vars", "x0,x1", "--islands", "1", "--pop", "30",
            "--generations", "3", "--seed", "1", "--patience", "0", "--json", "--no-dev-log",
            "--metric", "squared",
        ])
        .args(extra)
        .env("XDG_CACHE_HOME", cache)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    rcsr_cmd::feed_stdin(&mut child, csv().as_bytes());
    child.wait_with_output().unwrap()
}

fn ok_json(out: Output) -> String {
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn front(json: &str) -> String {
    json.split("\"pareto\":").nth(1).expect("a pareto array").to_string()
}

fn calibration_block(json: &str) -> String {
    let rest = json.split("\"calibration\":").nth(1).expect("dispatch.calibration");
    rest.split("\"route_ledger\"").next().unwrap().to_string()
}

/// `(stage, route, reason)` of every ledger row.
#[cfg(feature = "cuda")]
fn route_ledger(json: &str) -> Vec<(String, String, String)> {
    let ledger = json.split("\"route_ledger\":").nth(1).expect("dispatch.route_ledger");
    let field = |row: &str, key: &str| -> String {
        row.split(&format!("\"{key}\":\"")).nth(1).unwrap().split('"').next().unwrap().to_string()
    };
    ledger
        .split("{\"stage\":\"")
        .skip(1)
        .map(|row| {
            let row = format!("{{\"stage\":\"{row}");
            (field(&row, "stage"), field(&row, "route"), field(&row, "reason"))
        })
        .collect()
}

fn calibrate(cache: &Path, out: &Path, extra: &[&str]) -> Output {
    rcsr_cmd::rcsr_for(extra)
        .arg("calibrate")
        .arg("--out")
        .arg(out)
        .args(extra)
        .env("XDG_CACHE_HOME", cache)
        .output()
        .unwrap()
}

fn sha256_of(path: &Path) -> String {
    rcsr::manifest::sha256_hex(&std::fs::read(path).unwrap())
}

#[cfg(feature = "cuda")]
fn device_available() -> bool {
    rcsr::gpu::probe().available()
}

#[cfg(feature = "cuda")]
#[test]
fn two_auto_runs_give_one_front_and_one_route_ledger() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let cache = Scratch::new("auto-two");
    let a = ok_json(search(cache.path(), &["--gpu"]));
    let b = ok_json(search(cache.path(), &["--gpu"]));
    assert_eq!(front(&a), front(&b), "one seed, one front");
    assert_eq!(route_ledger(&a), route_ledger(&b), "one seed, one route ledger");
    assert!(calibration_block(&a).contains("\"source\":\"anchors\""), "{}", calibration_block(&a));
    let ledger = route_ledger(&a);
    let fit = ledger.iter().find(|r| r.0 == "fit").expect("a fit row");
    assert_eq!(fit.1, "device", "{ledger:?}");
}

#[cfg(feature = "cuda")]
#[test]
fn a_calibrated_file_gives_equal_fronts_and_its_own_hash() {
    if !device_available() {
        eprintln!("skipping GPU test: no usable CUDA device");
        return;
    }
    let dir = Scratch::new("file-two");
    let file = dir.join("cal.json");
    let out = calibrate(dir.path(), &file, &["--gpu"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let path = file.to_str().unwrap();
    let a = ok_json(search(dir.path(), &["--gpu", "--calibration", path]));
    let b = ok_json(search(dir.path(), &["--gpu", "--calibration", path]));
    assert_eq!(front(&a), front(&b));
    assert_eq!(route_ledger(&a), route_ledger(&b));
    let block = calibration_block(&a);
    assert!(block.contains("\"source\":\"file\""), "{block}");
    assert!(block.contains(&format!("\"file_sha256\":\"{}\"", sha256_of(&file))), "{block}");
}

#[test]
fn calibrate_writes_a_readable_file_atomically_and_a_run_never_writes_it() {
    let dir = Scratch::new("calibrate");
    let file = dir.join("cal.json");
    let out = calibrate(dir.path(), &file, &["--cpu", "--no-enrich"]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let text = std::fs::read_to_string(&file).unwrap();
    assert!(text.contains("\"schema\":\"rcsr-calibration/2\""), "{text}");
    assert!(text.contains("\"device\":null"), "{text}");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains(&sha256_of(&file)), "{stdout}");
    let leftovers: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    assert_eq!(leftovers, vec!["cal.json".to_string()], "no temporary file remains");

    let before = sha256_of(&file);
    let json = ok_json(search(dir.path(), &["--cpu", "--calibration", file.to_str().unwrap()]));
    assert!(calibration_block(&json).contains(&format!("\"file_sha256\":\"{before}\"")));
    assert_eq!(sha256_of(&file), before, "a search run leaves the file alone");

    let second = calibrate(dir.path(), &file, &["--cpu", "--no-enrich"]);
    assert_eq!(second.status.code(), Some(0));
    let valid = std::fs::read_to_string(&file).unwrap();
    assert!(valid.contains("\"schema\":\"rcsr-calibration/2\""));
}

#[test]
fn calibrate_without_a_usable_device_refuses_gpu_and_writes_nothing() {
    if cfg!(feature = "cuda") && rcsr::backend::probe().available() {
        eprintln!("skipping: a device is usable here");
        return;
    }
    let dir = Scratch::new("calibrate-gpu");
    let file = dir.join("cal.json");
    let out = calibrate(dir.path(), &file, &["--gpu"]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("--gpu needs a usable CUDA device"));
    assert!(!file.exists());
}

#[test]
fn a_corrupt_file_the_user_named_is_refused_with_a_message_and_no_output() {
    let dir = Scratch::new("corrupt");
    let good = dir.join("good.json");
    assert_eq!(calibrate(dir.path(), &good, &["--cpu", "--no-enrich"]).status.code(), Some(0));
    let text = std::fs::read_to_string(&good).unwrap();
    for (name, body) in [
        ("truncated.json", text[..text.len() / 2].to_string()),
        ("garbage.json", "not json".to_string()),
        ("negative.json", text.replacen("\"r_eval\":", "\"r_eval\":-", 1)),
        ("schema.json", text.replace("rcsr-calibration/2", "rcsr-calibration/9")),
    ] {
        let bad = dir.join(name);
        std::fs::write(&bad, body).unwrap();
        let out = search(dir.path(), &["--cpu", "--calibration", bad.to_str().unwrap()]);
        assert_eq!(out.status.code(), Some(2), "{name}");
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(stderr.contains(bad.to_str().unwrap()), "{name}: {stderr}");
        assert!(stderr.contains("calibration file"), "{name}: {stderr}");
        assert!(out.stdout.is_empty(), "{name}: no result on a refusal");
    }
    let missing = dir.join("missing.json");
    let out = search(dir.path(), &["--cpu", "--calibration", missing.to_str().unwrap()]);
    assert_eq!(out.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&out.stderr).contains("cannot read calibration file"));
}

#[test]
fn the_default_file_is_used_when_it_exists_and_refused_when_corrupt() {
    let cache = Scratch::new("default-file");
    let json = ok_json(search(cache.path(), &["--cpu"]));
    assert!(calibration_block(&json).contains("\"source\":\"anchors\""));
    assert!(!calibration_block(&json).contains("file_sha256"));

    let default = cache.join("rcsr").join("calibration.json");
    std::fs::create_dir_all(default.parent().unwrap()).unwrap();
    assert_eq!(calibrate(cache.path(), &default, &["--cpu", "--no-enrich"]).status.code(), Some(0));
    let json = ok_json(search(cache.path(), &["--cpu"]));
    let block = calibration_block(&json);
    assert!(block.contains("\"source\":\"file\""), "{block}");
    assert!(block.contains(&format!("\"file_sha256\":\"{}\"", sha256_of(&default))), "{block}");

    let json = ok_json(search(cache.path(), &["--cpu", "--calibration", "anchors"]));
    assert!(calibration_block(&json).contains("\"source\":\"anchors\""));

    std::fs::write(&default, "{").unwrap();
    let out = search(cache.path(), &["--cpu"]);
    assert_eq!(out.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("default calibration file"), "{stderr}");
}

#[test]
fn the_probe_source_is_labelled_and_the_cpu_front_does_not_depend_on_the_source() {
    let dir = Scratch::new("probe");
    let file = dir.join("cal.json");
    assert_eq!(calibrate(dir.path(), &file, &["--cpu", "--no-enrich"]).status.code(), Some(0));
    let probe = ok_json(search(dir.path(), &["--cpu", "--calibration", "probe"]));
    assert!(calibration_block(&probe).contains("\"source\":\"startup_probe\""));
    let anchors = ok_json(search(dir.path(), &["--cpu", "--calibration", "anchors"]));
    let filed = ok_json(search(dir.path(), &["--cpu", "--calibration", file.to_str().unwrap()]));
    assert_eq!(front(&probe), front(&anchors));
    assert_eq!(front(&anchors), front(&filed));
}

#[test]
fn the_run_manifest_records_the_calibration_source() {
    let dir = Scratch::new("manifest");
    let file = dir.join("cal.json");
    assert_eq!(calibrate(dir.path(), &file, &["--cpu", "--no-enrich"]).status.code(), Some(0));
    let manifest = dir.join("run.manifest.json");
    ok_json(search(
        dir.path(),
        &[
            "--cpu",
            "--calibration",
            file.to_str().unwrap(),
            "--run-manifest",
            manifest.to_str().unwrap(),
        ],
    ));
    let text = std::fs::read_to_string(&manifest).unwrap();
    assert!(text.contains("\"source\":\"file\""), "{text}");
    assert!(text.contains(&format!("\"file_sha256\":\"{}\"", sha256_of(&file))), "{text}");
    assert!(text.contains("\"file_path\""), "{text}");
}

#[test]
#[ignore = "runs the enrich probe on a synthetic panel; about a minute"]
fn calibrate_with_the_enrich_probe_writes_valid_rates() {
    let dir = Scratch::new("calibrate-enrich");
    let file = dir.join("cal.json");
    let out = calibrate(dir.path(), &file, &[]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let source = rcsr::dispatch::CalibrationRequest::File(file.clone())
        .resolve(None)
        .unwrap();
    let probe = rcsr::dispatch::Calibration::pinned(&source)
        .enrich
        .expect("the enrich object");
    assert_eq!(probe.rows, rcsr::enrich_probe::PROBE_SYMBOLS * rcsr::enrich_probe::PROBE_BARS);
    probe.rates.validate().unwrap();
    assert_eq!(probe.rates.clamped().validate(), Ok(()));
    let skipped = dir.join("skipped.json");
    let out = calibrate(dir.path(), &skipped, &["--no-enrich"]);
    assert_eq!(out.status.code(), Some(0));
    assert!(std::fs::read_to_string(&skipped).unwrap().contains("\"enrich\":null"));
}

/// #861: `rcsr dispatch-plan` prints the route table of the anchors against a
/// calibration file, byte for byte the same in every process, and its flips
/// stay few.
#[test]
fn dispatch_plan_prints_the_same_table_twice_and_few_flips() {
    let fixture = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/calibration_rtx3060.json");
    let run = || {
        rcsr_cmd::rcsr()
            .args(["dispatch-plan", "--calibration", "anchors", "--against", fixture])
            .output()
            .unwrap()
    };
    let (a, b) = (run(), run());
    assert_eq!(a.status.code(), Some(0), "{}", String::from_utf8_lossy(&a.stderr));
    assert_eq!(a.stdout, b.stdout, "two processes print different bytes");
    let text = String::from_utf8(a.stdout).unwrap();
    let cells = text.lines().filter(|l| l.starts_with("eval ") || l.starts_with("fit ")).count();
    assert_eq!(cells, 4950);
    let summary = text.lines().last().unwrap();
    let flips: usize = summary
        .strip_prefix("grid cells 4950, flips ")
        .and_then(|s| s.split(',').next())
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| panic!("summary line: {summary}"));
    assert!(flips <= 60, "{summary}");
    let json = rcsr_cmd::rcsr()
        .args(["dispatch-plan", "--against", fixture, "--json"])
        .output()
        .unwrap();
    assert_eq!(json.status.code(), Some(0));
    let json = String::from_utf8(json.stdout).unwrap();
    assert!(json.starts_with("{\"schema\":\"rcsr-dispatch-plan/1\""), "{json}");
    assert!(json.contains("\"cells\":4950"), "{json}");
    let probe = rcsr_cmd::rcsr().args(["dispatch-plan", "--calibration", "probe"]).output().unwrap();
    assert_eq!(probe.status.code(), Some(2));
    let missing = rcsr_cmd::rcsr().args(["dispatch-plan", "--against", "/no/such/file.json"]).output().unwrap();
    assert_eq!(missing.status.code(), Some(2));
}
