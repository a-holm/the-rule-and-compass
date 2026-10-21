//! #123: `rcsr prep` — raw archives/CSV to `.rcsb`, no intermediate CSV.
//!
//! The byte-identity test renders the prep table as canonical CSV (in memory),
//! feeds it through the existing `data` reader and `store` writer, and compares
//! the two stores. That is the reference path; `prep` itself never writes the
//! CSV.

use rcsr::features::LabelSpec;
use rcsr::ohlcv::Timeframe;
use rcsr::prep::{self, Layout, Options};
use rcsr::store;
use std::path::{Path, PathBuf};

fn fixture(name: &str) -> String {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/prep")
        .join(name)
        .to_string_lossy()
        .to_string()
}

fn tmp(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rcsr-prep-test-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_file(&p);
    p
}

fn tmp_dir(name: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!("rcsr-prep-cache-test-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

/// A three-symbol Binance-style directory with a distinct price level per
/// symbol, so a mis-merged or stale cached part is visible in the table.
fn write_three_symbols(dir: &Path) {
    let csv = std::fs::read_to_string(fixture("binance_1m.csv")).unwrap();
    for (sym, bump) in [("AAA", 0.0), ("BBB", 10.0), ("CCC", 20.0)] {
        let shifted: String = csv
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                let n = f.len();
                let mut out = String::new();
                for (i, x) in f.iter().enumerate() {
                    if i == 1 || i == 2 || i == 3 || i == 4 || i == 6 {
                        out.push_str(&format!("{:.4}", x.parse::<f64>().unwrap() + bump));
                    } else {
                        out.push_str(x);
                    }
                    if i != n - 1 {
                        out.push(',');
                    }
                }
                out
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join(format!("{}.csv", sym)), shifted + "\n").unwrap();
    }
}

fn cache_parts(cache: &Path, ext: &str) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = std::fs::read_dir(cache)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|x| x.to_str()) == Some(ext))
        .collect();
    out.sort();
    out
}

fn base(input: String) -> Options {
    Options {
        inputs: vec![input],
        format: Layout::Auto,
        jobs: 1,
        ..Default::default()
    }
}

#[test]
fn a_fixture_produces_a_store_byte_identical_to_the_importer() {
    let opts = base(fixture("binance_1m.csv"));
    let build = prep::build(&opts).unwrap();
    let prep_path = tmp("byte-identical-prep.rcsb");
    let ref_path = tmp("byte-identical-ref.rcsb");
    store::write_store_with_time(&prep_path.to_string_lossy(), &build.table, 65_536, Some(0)).unwrap();

    // The reference is the current importer path on the canonical CSV.
    let csv = prep::table_csv(&build.table);
    let table = rcsr::data::parse_table_with_time(&csv, false, Some("ts")).unwrap();
    store::write_store_with_time(&ref_path.to_string_lossy(), &table, 65_536, Some(0)).unwrap();

    let a = std::fs::read(&prep_path).unwrap();
    let b = std::fs::read(&ref_path).unwrap();
    assert_eq!(a, b, "prep store must equal the importer on the same values");
    let _ = std::fs::remove_file(&prep_path);
    let _ = std::fs::remove_file(&ref_path);
}

#[test]
fn resampling_aggregates_to_known_15m_bars_and_writes_features() {
    let mut opts = base(fixture("binance_1m.csv"));
    opts.resample = Some(Timeframe::M15);
    opts.features = rcsr::features::expand("ta:core,candle:geometry").unwrap();
    opts.label = Some(LabelSpec {
        horizon: 2,
        cost_bps: 10.0,
        vol_window: 0,
        stride: 1,
    });
    let build = prep::build(&opts).unwrap();
    assert_eq!(build.audit.rows_in, 60);
    assert_eq!(build.audit.rows_out, 4, "60 one-minute bars -> four 15m bars");
    assert_eq!(
        build.table.names[0..5],
        ["ts".to_string(), "open".into(), "high".into(), "low".into(), "close".into()]
    );
    assert!(build.table.names.contains(&"ret1".to_string()));
    assert!(build.table.names.contains(&"label".to_string()));
    // First 15m bar: open 100.00, close of minute 14 = 100 + 14*0.1 - 0.2.
    assert!((build.table.value(0, 1) - 100.0).abs() < 1e-9);
    assert!((build.table.value(0, 4) - 101.2).abs() < 1e-9);
    // Volume is the sum of the 15 constituent minutes (10..24).
    let expect: f64 = (10..25).map(|x| x as f64).sum();
    assert!((build.table.value(0, 5) - expect).abs() < 1e-9);
    assert!(build.audit.features.contains(&"ret1".to_string()));
    assert_eq!(build.audit.label_gap_rows, 0);
}

#[test]
fn millisecond_microsecond_and_second_inputs_normalise_to_the_same_times() {
    let dir = tmp("units");
    let _ = std::fs::create_dir_all(&dir);
    let text = std::fs::read_to_string(fixture("binance_1m.csv")).unwrap();
    let (secs, ms, us): (Vec<String>, Vec<String>, Vec<String>) = {
        let mut s = Vec::new();
        let mut m = Vec::new();
        let mut u = Vec::new();
        for (i, _line) in text.lines().enumerate() {
            let base = 1_704_067_200i64 + i as i64 * 60;
            s.push(format!("{base},rest"));
            m.push(format!("{},rest", base * 1_000));
            u.push(format!("{},rest", base * 1_000_000));
        }
        (s, m, u)
    };
    // Rebuild the 12-column rows with the alternate time unit.
    let rebuild = |times: &[String], path: &Path| {
        let mut out = String::new();
        for (i, t) in times.iter().enumerate() {
            let t = t.split(',').next().unwrap();
            let row: Vec<&str> = text.lines().nth(i).unwrap().split(',').collect();
            out.push_str(t);
            for f in &row[1..] {
                out.push(',');
                out.push_str(f);
            }
            out.push('\n');
        }
        std::fs::write(path, out).unwrap();
    };
    let p_s = dir.join("s.csv");
    let p_ms = dir.join("ms.csv");
    let p_us = dir.join("us.csv");
    rebuild(&secs, &p_s);
    rebuild(&ms, &p_ms);
    rebuild(&us, &p_us);
    let times = |p: &Path| {
        let b = prep::build(&base(p.to_string_lossy().to_string())).unwrap();
        (0..b.table.len()).map(|r| b.table.ints(0).unwrap()[r]).collect::<Vec<_>>()
    };
    let a = times(&p_s);
    assert_eq!(a, times(&p_ms));
    assert_eq!(a, times(&p_us));
    assert_eq!(a[0], 1_704_067_200);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_multi_symbol_directory_is_parallel_deterministic_and_keeps_symbols() {
    let dir = tmp("panel");
    let _ = std::fs::create_dir_all(&dir);
    let csv = std::fs::read_to_string(fixture("binance_1m.csv")).unwrap();
    for sym in ["AAA", "BBB", "CCC"] {
        // Shift each symbol's prices so a mis-merged panel is visible.
        let shifted: String = csv
            .lines()
            .map(|l| {
                let f: Vec<&str> = l.split(',').collect();
                let n = f.len();
                let bump = match sym {
                    "AAA" => 0.0,
                    "BBB" => 10.0,
                    _ => 20.0,
                };
                let mut out = String::new();
                for (i, x) in f.iter().enumerate() {
                    if i == 1 || i == 2 || i == 3 || i == 4 || i == 6 {
                        out.push_str(&format!("{:.4}", x.parse::<f64>().unwrap() + bump));
                    } else {
                        out.push_str(x);
                    }
                    if i != n - 1 {
                        out.push(',');
                    }
                }
                out
            })
            .collect::<Vec<_>>()
            .join("\n");
        std::fs::write(dir.join(format!("{}.csv", sym)), shifted + "\n").unwrap();
    }
    let mut opts = base(dir.to_string_lossy().to_string());
    opts.resample = Some(Timeframe::M15);
    opts.features = rcsr::features::expand("ta:core").unwrap();
    opts.jobs = 1;
    let serial = prep::build(&opts).unwrap();
    opts.jobs = 3;
    let parallel = prep::build(&opts).unwrap();
    assert_eq!(prep::table_csv(&serial.table), prep::table_csv(&parallel.table));
    assert_eq!(serial.manifest, parallel.manifest);
    let names = &parallel.table.names;
    assert_eq!(names[0], "ts");
    assert_eq!(names[1], "symbol");
    assert_eq!(names[2], "open");
    // Three symbols x 60 minutes -> 12 15m rows, symbol-sorted.
    assert_eq!(parallel.table.len(), 12);
    let syms: Vec<String> = (0..parallel.table.len())
        .map(|r| parallel.table.text(r, 1).unwrap().to_string())
        .collect();
    assert_eq!(syms, vec!["AAA"; 4].into_iter().chain(vec!["BBB"; 4]).chain(vec!["CCC"; 4]).collect::<Vec<_>>());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_zip_archive_is_read_without_an_intermediate_csv() {
    let dir = tmp("zip");
    let _ = std::fs::create_dir_all(&dir);
    let csv = std::fs::read(fixture("binance_1m.csv")).unwrap();
    let zip_path = dir.join("BTCUSDT-1m-2024-01.zip");
    std::fs::write(&zip_path, stored_zip("BTCUSDT-1m-2024-01.csv", &csv)).unwrap();
    let archive = rcsr::zip::ZipArchive::open(&zip_path.to_string_lossy()).unwrap();
    assert!(!archive.entries.is_empty(), "the test archive must have an entry");
    let b = prep::build(&base(zip_path.to_string_lossy().to_string())).unwrap();
    assert_eq!(b.audit.rows_in, 60);
    assert_eq!(b.audit.rows_out, 60);
    assert_eq!(b.table.len(), 60);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_same_spec_and_inputs_reproduce_the_same_bytes() {
    let mut opts = base(fixture("aliased_1m.csv"));
    opts.resample = Some(Timeframe::M15);
    opts.features = rcsr::features::expand("ta:core,exotic:cheap").unwrap();
    let a = prep::build(&opts).unwrap();
    let b = prep::build(&opts).unwrap();
    assert_eq!(prep::table_csv(&a.table), prep::table_csv(&b.table));
    assert_eq!(a.manifest, b.manifest);
    let pa = tmp("determinism-a.rcsb");
    let pb = tmp("determinism-b.rcsb");
    store::write_store_with_time(&pa.to_string_lossy(), &a.table, 65_536, Some(0)).unwrap();
    store::write_store_with_time(&pb.to_string_lossy(), &b.table, 65_536, Some(0)).unwrap();
    assert_eq!(std::fs::read(&pa).unwrap(), std::fs::read(&pb).unwrap());
    let _ = std::fs::remove_file(pa);
    let _ = std::fs::remove_file(pb);
}

#[test]
fn a_resumed_prep_reuses_parts_and_reproduces_the_store_bytes() {
    let dir = tmp_dir("resume");
    let input = dir.join("input");
    std::fs::create_dir_all(&input).unwrap();
    write_three_symbols(&input);
    let cache = dir.join("cache");
    let mut opts = base(input.to_string_lossy().to_string());
    opts.resample = Some(Timeframe::M15);
    opts.features = rcsr::features::expand("ta:core,candle:geometry").unwrap();
    opts.label = Some(LabelSpec { horizon: 2, cost_bps: 10.0, vol_window: 0, stride: 1 });
    opts.jobs = 3;
    opts.resume = Some(cache.to_string_lossy().to_string());

    let cold = prep::build(&opts).unwrap();
    let pa = tmp("resume-cold.rcsb");
    store::write_store_with_time(&pa.to_string_lossy(), &cold.table, 65_536, Some(0)).unwrap();
    assert_eq!(cache_parts(&cache, "spool").len(), 3, "one spool per symbol");
    assert_eq!(cache_parts(&cache, "key").len(), 3, "one key per symbol");

    // Warm run: same settings, different --jobs (it must not invalidate the
    // cache and must not change byte order).
    let before: Vec<_> = cache_parts(&cache, "spool")
        .iter()
        .map(|p| std::fs::metadata(p).unwrap().modified().unwrap())
        .collect();
    opts.jobs = 1;
    let warm = prep::build(&opts).unwrap();
    let pb = tmp("resume-warm.rcsb");
    store::write_store_with_time(&pb.to_string_lossy(), &warm.table, 65_536, Some(0)).unwrap();

    assert_eq!(prep::table_csv(&cold.table), prep::table_csv(&warm.table));
    assert_eq!(cold.manifest, warm.manifest, "a resumed build must reproduce the manifest");
    let a = std::fs::read(&pa).unwrap();
    let b = std::fs::read(&pb).unwrap();
    assert_eq!(a, b, "a resumed build must reproduce the store bytes");
    assert_eq!(
        rcsr::manifest::sha256_file(&pa).unwrap(),
        rcsr::manifest::sha256_file(&pb).unwrap()
    );

    // Reuse means "read, do not rewrite": every spool keeps its timestamp.
    let after: Vec<_> = cache_parts(&cache, "spool")
        .iter()
        .map(|p| std::fs::metadata(p).unwrap().modified().unwrap())
        .collect();
    assert_eq!(before, after, "reused spools must not be rewritten");

    // A third run with more workers is still the same table.
    opts.jobs = 4;
    let again = prep::build(&opts).unwrap();
    assert_eq!(prep::table_csv(&warm.table), prep::table_csv(&again.table));
    assert_eq!(warm.manifest, again.manifest);

    let _ = std::fs::remove_file(pa);
    let _ = std::fs::remove_file(pb);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_changed_archive_invalidates_only_its_symbol_part() {
    let dir = tmp_dir("invalidate");
    let input = dir.join("input");
    std::fs::create_dir_all(&input).unwrap();
    write_three_symbols(&input);
    let cache = dir.join("cache");
    let mut opts = base(input.to_string_lossy().to_string());
    opts.resample = Some(Timeframe::M15);
    opts.features = rcsr::features::expand("ta:core").unwrap();
    opts.jobs = 2;
    opts.resume = Some(cache.to_string_lossy().to_string());
    let cold = prep::build(&opts).unwrap();

    // Replace one symbol's archive with another symbol's bars while keeping
    // its byte length and modification time. The archive checksum, not the
    // name, size or mtime, must decide.
    let path = input.join("BBB.csv");
    let before = std::fs::read(&path).unwrap();
    let donor = std::fs::read(input.join("CCC.csv")).unwrap();
    assert_eq!(before.len(), donor.len(), "the fixture rewrite keeps the size");
    assert_ne!(before, donor);
    let mtime = std::fs::metadata(&path).unwrap().modified().unwrap();
    std::fs::write(&path, &donor).unwrap();
    std::fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(mtime)
        .unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), mtime);

    let mut fresh_opts = opts.clone();
    fresh_opts.resume = None;
    let reference = prep::build(&fresh_opts).unwrap();
    let resumed = prep::build(&opts).unwrap();
    assert_ne!(prep::table_csv(&cold.table), prep::table_csv(&resumed.table));
    assert_eq!(prep::table_csv(&reference.table), prep::table_csv(&resumed.table));
    assert_eq!(reference.manifest, resumed.manifest);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_resume_directory_with_other_settings_is_refused() {
    let dir = tmp_dir("settings");
    let input = dir.join("input");
    std::fs::create_dir_all(&input).unwrap();
    std::fs::write(input.join("AAA.csv"), std::fs::read(fixture("aliased_1m.csv")).unwrap()).unwrap();
    let cache = dir.join("cache");
    let mut opts = base(input.to_string_lossy().to_string());
    opts.jobs = 2;
    opts.resume = Some(cache.to_string_lossy().to_string());
    prep::build(&opts).unwrap();

    opts.features = rcsr::features::expand("ta:core").unwrap();
    let err = prep::build(&opts).unwrap_err();
    assert!(err.contains("different settings"), "{}", err);
    assert!(err.contains("--resume"), "{}", err);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn a_verified_splice_merges_the_two_legs_and_a_wrong_k_fails() {
    let dir = tmp("splice");
    let _ = std::fs::create_dir_all(&dir);
    // OLD trades at 100, NEW at 1.0 exactly when it takes over.
    std::fs::write(
        dir.join("OLD.csv"),
        "time,open,high,low,close,volume\n1704067200,100,100,100,100,10\n1704067260,100,100,100,100,10\n",
    )
    .unwrap();
    std::fs::write(
        dir.join("NEW.csv"),
        "time,open,high,low,close,volume\n1704067320,1,1,1,1,1000\n1704067380,1,1,1,1,1000\n",
    )
    .unwrap();
    let mut opts = base(dir.to_string_lossy().to_string());
    opts.format = Layout::Ohlcv;
    opts.jobs = 2;
    opts.migrations = prep::parse_migrations("OLD->NEW:100").unwrap();
    let b = prep::build(&opts).unwrap();
    // OLD is spliced into NEW, so one symbol and four rows; both legs at 1.0.
    assert_eq!(b.audit.rows_out, 4);
    assert_eq!(b.audit.migrations, 1);
    assert_eq!(b.table.value(0, 1), 1.0);
    assert_eq!(b.table.value(0, 5), 1000.0, "base volume scales by k");

    opts.migrations = prep::parse_migrations("OLD->NEW:1").unwrap();
    let err = prep::build(&opts).unwrap_err();
    assert!(err.contains("differs from k=1.000000"), "{}", err);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn dry_run_manifest_records_the_spec_hash_and_audit() {
    let mut opts = base(fixture("aliased_1m.csv"));
    opts.features = rcsr::features::expand("ta:core").unwrap();
    opts.label = Some(LabelSpec::parse("fwd:15m,cost=10bp,volnorm=4", Timeframe::M1).unwrap());
    opts.resample = Some(Timeframe::M15);
    opts.dry_run = true;
    let b = prep::build(&opts).unwrap();
    assert!(b.manifest.contains("\"spec_hash\""));
    assert!(b.manifest.contains("\"sha256\""));
    assert!(b.manifest.contains("\"audit\""));
    assert!(b.manifest.contains("\"time_unit\":\"s\""));
    assert!(b.manifest.contains("\"resample\":\"15m\""));
    assert_eq!(b.audit.rows_out, 4);
}

#[test]
fn a_spec_file_is_parsed_in_fixed_order_and_hashed_into_the_manifest() {
    let dir = tmp("spec");
    let _ = std::fs::create_dir_all(&dir);
    let spec = dir.join("panel.spec");
    // `label` appears before `resample` on purpose: the loader must still
    // resolve 1h against the 15m base.
    std::fs::write(&spec, "# panel spec\nlabel=fwd:1h,cost=7bp\nresample=15m\nfeatures=ta:core\n").unwrap();
    let mut opts = base(fixture("aliased_1m.csv"));
    prep::load_spec(&mut opts, &spec.to_string_lossy()).unwrap();
    assert_eq!(opts.resample, Some(Timeframe::M15));
    assert_eq!(opts.label.unwrap().horizon, 4);
    let b = prep::build(&opts).unwrap();
    let hash = rcsr::manifest::sha256_file(&spec).unwrap();
    assert!(b.manifest.contains(&hash));
    assert!(b.manifest.contains("\"spec_path\""));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn prep_help_and_schema_document_the_command() {
    let help = prep::usage();
    for flag in [
        "--input", "--format", "--resample", "--features", "--label", "--migrations",
        "--dry-run", "--jobs", "--resume",
    ] {
        assert!(help.contains(flag), "help is missing {}", flag);
    }
    let schema = rcsr::schema::json();
    assert!(schema.contains("\"prep\""));
    assert!(schema.contains("--dry-run"));
}

// ---------------------------------------------------------------------------
// A tiny stored-ZIP writer so the test has a real archive without a data blob.
// ---------------------------------------------------------------------------

fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let crc = crc32(data);
    out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(data);
    let cd = out.len();
    out.extend_from_slice(&0x0201_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(name.as_bytes());
    let cd_size = out.len() - cd;
    out.extend_from_slice(&0x0605_4b50u32.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&(cd_size as u32).to_le_bytes());
    out.extend_from_slice(&(cd as u32).to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out
}

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let m = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & m);
        }
    }
    !crc
}