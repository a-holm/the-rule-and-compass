//! #733: Auto dispatch routes from a pinned calibration, so a seeded run with
//! the default `--dispatch auto` gives one front and one route ledger in every
//! process, and `rcsr calibrate` is the only writer of the calibration file.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

#[path = "common/scratch.rs"]
mod scratch;

use scratch::Scratch;
use std::io::Write;
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
    child.stdin.as_mut().unwrap().write_all(csv().as_bytes()).unwrap();
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

/// The stderr lines that start with `prefix`.
fn lines_with(out: &Output, prefix: &str) -> Vec<String> {
    String::from_utf8_lossy(&out.stderr)
        .lines()
        .filter(|l| l.starts_with(prefix))
        .map(str::to_string)
        .collect()
}

fn cpu_calibrate(dir: &Scratch, file: &Path, strict: bool, env: &[(&str, &str)]) -> Output {
    let mut cmd = rcsr_cmd::rcsr();
    cmd.arg("calibrate").arg("--out").arg(file).args(["--cpu", "--no-enrich"]);
    if strict {
        cmd.arg("--strict");
    }
    for (k, v) in env {
        cmd.env(k, v);
    }
    cmd.env("XDG_CACHE_HOME", dir.path()).output().unwrap()
}

/// #862: a noisy term gets one warning line naming it, the file is written, and
/// `--strict` exits 3 and writes nothing.
#[test]
fn calibrate_strict_refuses_noisy_terms() {
    let dir = Scratch::new("calibrate-strict");
    let noisy = [("RCSR_CALIBRATE_TEST_SPREAD", "bw=2.0"), ("RCSR_CALIBRATE_TEST_LOAD1", "0.25")];
    let file = dir.join("lenient.json");
    let out = cpu_calibrate(&dir, &file, false, &noisy);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    let warnings = lines_with(&out, "# calibrate:");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("noisy terms: bw 2.00"), "{}", warnings[0]);
    assert!(!warnings[0].contains("measured under load"), "{}", warnings[0]);
    let text = std::fs::read_to_string(&file).expect("the file exists");
    assert!(text.contains("\"load1_start\":0.25,\"load1_end\":0.25,"), "{text}");

    let strict_file = dir.join("strict.json");
    let out = cpu_calibrate(&dir, &strict_file, true, &noisy);
    assert_eq!(out.status.code(), Some(3), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(!strict_file.exists(), "--strict wrote a file");
    assert_eq!(lines_with(&out, "# calibrate:").len(), 1);

    let loaded = [("RCSR_CALIBRATE_TEST_LOAD1", "1000")];
    let over_file = dir.join("over.json");
    let out = cpu_calibrate(&dir, &over_file, false, &loaded);
    assert_eq!(out.status.code(), Some(0));
    let warnings = lines_with(&out, "# calibrate:");
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].starts_with("# calibrate: measured under load 1000.00 (limit "), "{}", warnings[0]);
    let refused = cpu_calibrate(&dir, &dir.join("over-strict.json"), true, &loaded);
    assert_eq!(refused.status.code(), Some(3));
    assert!(!dir.join("over-strict.json").exists());
}

/// #862: a probe at a low load and with quiet terms prints no `# calibrate:` line.
#[test]
fn calibrate_at_a_low_load_prints_no_warning() {
    let dir = Scratch::new("calibrate-quiet");
    let file = dir.join("quiet.json");
    let out = cpu_calibrate(&dir, &file, true, &[("RCSR_CALIBRATE_TEST_LOAD1", "0.0"), ("RCSR_CALIBRATE_TEST_SPREAD", "cpu_eval=1.2")]);
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    assert!(lines_with(&out, "# calibrate:").is_empty());
    assert!(file.exists());
}

/// #862: a search that prices with a file measured under load prints the notice
/// once, and its route ledger equals the one of the same rates without the fields.
#[test]
fn a_file_measured_under_load_gives_one_notice_and_the_same_routes() {
    let dir = Scratch::new("load-notice");
    let plain = rcsr::dispatch::Calibration::default();
    let loaded = rcsr::dispatch::Calibration { load1_start: Some(1000.0), load1_end: Some(1001.0), ..plain };
    let plain_file = dir.join("plain.json");
    let loaded_file = dir.join("loaded.json");
    std::fs::write(&plain_file, plain.file_text(None, "test", "2026-01-01T00:00:00Z")).unwrap();
    std::fs::write(&loaded_file, loaded.file_text(None, "test", "2026-01-01T00:00:00Z")).unwrap();
    let a = search(dir.path(), &["--calibration", plain_file.to_str().unwrap()]);
    let b = search(dir.path(), &["--calibration", loaded_file.to_str().unwrap()]);
    assert!(lines_with(&a, "# calibration:").is_empty());
    let notices = lines_with(&b, "# calibration:");
    assert_eq!(notices.len(), 1, "{notices:?}");
    assert!(notices[0].contains("measured under load 1000.00"), "{}", notices[0]);
    assert!(notices[0].contains(loaded_file.to_str().unwrap()), "{}", notices[0]);
    let (a, b) = (ok_json(a), ok_json(b));
    assert_eq!(route_ledger(&a), route_ledger(&b));
    assert_eq!(front(&a), front(&b));
}
