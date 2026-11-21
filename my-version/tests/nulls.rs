//! Block-permutation null for autocorrelated targets (issue #114).
//!
//! The i.i.d. row shuffle destroys autocorrelation, so a real pipeline looks
//! better than a null that no longer resembles the data. These tests pin the
//! permutation mechanics and show, on synthetic AR(1) series, that the i.i.d.
//! null over-rejects while the block null is close to the nominal level.

#[path = "common/rcsr.rs"]
mod rcsr_cmd;

use rcsr::diagnostics::{block_count, block_permute, NullControl};
use rcsr::rng::Mt;
use std::fmt::Write as _;
use std::path::PathBuf;
use std::process::Output;

/// A stationary-ish AR(1) series; innovations are a sum of three uniforms.
fn ar1(n: usize, phi: f64, seed: u32) -> Vec<f64> {
    let mut rng = Mt::new(seed);
    let mut x = Vec::with_capacity(n);
    let mut prev = 0.0;
    for i in 0..n {
        let e = (rng.next_f64() + rng.next_f64() + rng.next_f64() - 1.5) * 2.0;
        let v = if i == 0 { e } else { phi * prev + e };
        x.push(v);
        prev = v;
    }
    x
}

/// Lag-1 autocorrelation as the test statistic.
fn lag1(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 3 {
        return 0.0;
    }
    let ma = x[..n - 1].iter().sum::<f64>() / (n - 1) as f64;
    let mb = x[1..].iter().sum::<f64>() / (n - 1) as f64;
    let mut num = 0.0;
    let mut a = 0.0;
    let mut b = 0.0;
    for i in 0..n - 1 {
        let u = x[i] - ma;
        let v = x[i + 1] - mb;
        num += u * v;
        a += u * u;
        b += v * v;
    }
    if a <= 0.0 || b <= 0.0 {
        0.0
    } else {
        num / (a.sqrt() * b.sqrt())
    }
}

/// One-sided permutation p-value for the statistic `-lag1` (lower = more
/// autocorrelated), using the same p-value formula as the engine.
fn p_value(y: &[f64], block_len: usize, reps: usize, seed: u32) -> f64 {
    let observed = -lag1(y);
    let mut rng = Mt::new(seed);
    let mut nulls = Vec::with_capacity(reps);
    for _ in 0..reps {
        let perm = block_permute(y, block_len, &mut rng);
        nulls.push(-lag1(&perm));
    }
    NullControl::summarise(observed, nulls).p_value
}

#[test]
fn block_permutation_is_deterministic_and_keeps_every_row() {
    let y = ar1(100, 0.8, 7);
    let mut a = Mt::new(123);
    let mut b = Mt::new(123);
    let pa = block_permute(&y, 24, &mut a);
    let pb = block_permute(&y, 24, &mut b);
    assert_eq!(pa, pb, "the same seed must reproduce the null exactly");
    let mut sorted = pa.clone();
    sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut original = y.clone();
    original.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert_eq!(sorted, original, "permutation must keep every value once");
    assert_eq!(block_count(100, 24), 5);
    assert_eq!(block_count(100, 1), 100);
}

#[test]
fn block_null_is_calibrated_on_exchangeable_data_and_does_not_over_reject_ar1() {
    const SERIES: usize = 200;
    const N: usize = 512;
    const REPS: usize = 99;

    // 1. On exchangeable data the block null must have a size close to the
    // nominal 5%: permuting blocks is just another random rearrangement.
    let mut iid_data_rejections = 0usize;
    let mut rng = Mt::new(99);
    for s in 0..SERIES {
        let y: Vec<f64> = (0..N).map(|_| rng.next_f64() - 0.5).collect();
        if p_value(&y, 64, REPS, 30_000 + s as u32) <= 0.05 {
            iid_data_rejections += 1;
        }
    }
    let iid_data_rate = iid_data_rejections as f64 / SERIES as f64;
    eprintln!("iid data, block null rejection rate: {iid_data_rate:.3}");
    assert!(
        iid_data_rate <= 0.12,
        "on exchangeable data the block null should be near nominal 5%, rate={iid_data_rate}"
    );

    // 2. On AR(1) data the i.i.d. shuffle is anti-conservative: it destroys the
    // serial dependence, so the real autocorrelation looks impossible. Blocks
    // of half the series keep the dependence and do not over-reject.
    let mut iid_rejections = 0usize;
    let mut block_rejections = 0usize;
    for s in 0..SERIES {
        let y = ar1(N, 0.9, s as u32 + 1);
        if p_value(&y, 1, REPS, 10_000 + s as u32) <= 0.05 {
            iid_rejections += 1;
        }
        if p_value(&y, N / 2, REPS, 20_000 + s as u32) <= 0.05 {
            block_rejections += 1;
        }
    }
    let iid_rate = iid_rejections as f64 / SERIES as f64;
    let block_rate = block_rejections as f64 / SERIES as f64;
    eprintln!("AR(1): iid rejection rate {iid_rate:.3}, block rejection rate {block_rate:.3}");
    assert!(
        iid_rate > 0.5,
        "the i.i.d. null should over-reject on AR(1) data, rate={iid_rate}"
    );
    assert!(
        block_rate <= 0.12,
        "the block null should stay near the nominal 5%, rate={block_rate}"
    );
    assert!(
        block_rate < iid_rate,
        "block permutation must be less anti-conservative: block={block_rate} iid={iid_rate}"
    );
}

#[test]
fn the_report_records_the_null_method_and_block_count() {
    let c = NullControl::summarise_block(0.1, vec![0.3, 0.2, 0.4], 24, 6);
    assert_eq!(c.method, "block");
    assert_eq!(c.block_len, 24);
    assert_eq!(c.blocks, 6);
    assert_eq!(c.observed, 0.1);
    assert!(c.p_value > 0.0 && c.p_value <= 1.0);
    let iid = NullControl::summarise(0.1, vec![0.3, 0.2]);
    assert_eq!(iid.method, "iid");
    assert_eq!(iid.block_len, 0);
}

#[path = "common/scratch.rs"]
mod scratch;
use scratch::Scratch;

struct TempCsv(PathBuf, #[allow(dead_code)] Scratch);

impl TempCsv {
    fn new(contents: &str) -> Self {
        let scratch = Scratch::new("null-block");
        let path = scratch.join("data.csv");
        std::fs::write(&path, contents).expect("write temp csv");
        TempCsv(path, scratch)
    }
    fn as_str(&self) -> &str {
        self.0.to_str().unwrap()
    }
}

fn run(args: &[&str]) -> Output {
    rcsr_cmd::rcsr()
        .args(args)
        .output()
        .expect("run rcsr")
}

#[test]
fn null_block_24_runs_on_a_time_ordered_sample_and_reports_p_and_blocks() {
    let mut text = String::from("t,x,y\n");
    for i in 0..200 {
        let x = (i as f64 * 0.017).sin();
        let y = 1.5 * x + 0.01 * (i as f64 * 0.37).cos();
        let _ = writeln!(text, "{},{:.6},{:.6}", i, x, y);
    }
    let csv = TempCsv::new(&text);
    let out = run(&[
        "--data",
        csv.as_str(),
        "--target",
        "y",
        "--vars",
        "x",
        "--split",
        "temporal",
        "--time",
        "t",
        "--null-block",
        "24",
        "--null-runs",
        "3",
        "--generations",
        "2",
        "--pop",
        "20",
        "--format",
        "json",
        "--quiet",
    ]);
    assert!(
        out.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("\"method\":\"block\""), "json: {stdout}");
    assert!(stdout.contains("\"block_len\":24"), "json: {stdout}");
    assert!(stdout.contains("\"blocks\":"), "json: {stdout}");
    assert!(stdout.contains("\"p_value\":"), "json: {stdout}");
    assert!(stdout.contains("\"observed\":"), "json: {stdout}");
}

// ---------------------------------------------------------------------------
// #1272: `rcsr panel permute-labels`, the store of the search-capacity null
// ---------------------------------------------------------------------------

use rcsr::devlog::Json;
use rcsr::lockbox::{self, Lockbox};
use rcsr::store::{self, DType, FileStore};
use std::collections::BTreeMap;
use std::path::Path;

/// 2021-01-01T00:00:00Z.
const T0: i64 = 1_609_459_200;
const HOUR: i64 = 3600;
const DAY: i64 = 86_400;
const SYMBOLS: [&str; 3] = ["AAA", "BBB", "CCC"];
/// The columns of the label family in store order.
const KINDS: [&str; 9] = ["fwd", "y", "long_net", "action", "t1", "exit_ret", "m_net", "m_barrier", "weight"];
/// The family columns that move; `t1` and `weight` stay on their rows.
const MOVED: [&str; 7] = ["fwd", "y", "long_net", "action", "exit_ret", "m_net", "m_barrier"];
const SPEC: &str = "{\"schema\":\"rcsr-target-spec/1\",\"targets\":[{\"column\":\"lbl_h24_y\",\"name\":\"h24\",\
\"kind\":\"forward_return\",\"deprecated\":false,\"return_column\":\"lbl_h24_fwd\",\"return_units\":\"log\",\
\"horizon_seconds\":86400,\"entry_offset_seconds\":3600,\"delay_seconds\":0,\"entry_price\":\"open[t+1]\",\
\"exit_price\":\"close[t+h]\",\"offset\":0.0,\"scale\":{\"kind\":\"constant\",\"value\":0.01},\"clip\":null,\
\"units\":\"log_return/0.01\",\"cost_bps\":20.0,\"cost_in_label\":false,\"construction_columns\":[\"lbl_h24_fwd\"],\
\"note\":null,\"long_only\":true}]}\n";

fn frac(v: f64) -> f64 {
    v - v.floor()
}

fn lbl(kind: &str) -> String {
    format!("lbl_h24_{kind}")
}

fn code(o: &Output) -> i32 {
    o.status.code().unwrap_or(-1)
}

fn stderr(o: &Output) -> String {
    String::from_utf8_lossy(&o.stderr).into_owned()
}

fn read_json(path: &str) -> Json {
    Json::parse(std::fs::read_to_string(path).unwrap().trim()).unwrap_or_else(|| panic!("{path} is not JSON"))
}

fn at<'a>(j: &'a Json, path: &[&str]) -> &'a Json {
    path.iter().fold(j, |cur, k| cur.get(k).unwrap_or_else(|| panic!("missing '{k}' in {path:?}")))
}

fn p(dir: &Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

/// Hourly rows of three symbols from `T0` for `hours` hours, time-major, without the hours
/// `skip` names. The label `h24` sits on the daily grid of `T0` (stride 24) and is NaN between;
/// `BBB` has a NaN `y`, `exit_ret`, `m_net` and `m_barrier` on every 5th grid day while its
/// `fwd`, `long_net`, `action`, `t1` and `weight` stay finite. Every finite `y` is distinct.
fn write_null_panel(dir: &Path, name: &str, hours: i64, skip: &dyn Fn(i64) -> bool) -> String {
    let mut text = format!("ts,symbol,r16,r96,{}\n", KINDS.map(lbl).join(","));
    let mut id = 0u64;
    for i in 0..hours {
        let ts = T0 + HOUR * i;
        if skip(ts) {
            continue;
        }
        for (k, sym) in SYMBOLS.iter().enumerate() {
            id += 1;
            let f = i as f64 + 0.37 * k as f64;
            let r16 = frac(f * 0.618_033_988_749_894_9) - 0.5;
            let r96 = frac(f * 0.754_877_666_246_692_7) - 0.5;
            let mut v = [f64::NAN; 9];
            if i % 24 == 0 {
                let fwd = 0.02 * r16 + 0.002 * (frac(f * 0.324_717_957_244_746) - 0.5) + id as f64 * 1e-12;
                let exit = 0.5 * fwd + 0.001;
                let bit = |b: bool| if b { 1.0 } else { 0.0 };
                v = [fwd, fwd / 0.01, fwd - 0.002, bit(fwd > 0.002), (ts + 25 * HOUR) as f64, exit, bit(exit > 0.002), bit(r96 > 0.0), 1.0 + 0.1 * frac(f * 0.5)];
                if k == 1 && (i / 24) % 5 == 2 {
                    for kind in ["y", "exit_ret", "m_net", "m_barrier"] {
                        v[KINDS.iter().position(|x| *x == kind).unwrap()] = f64::NAN;
                    }
                }
            }
            let cells: Vec<String> = v.iter().map(|x| format!("{}", x)).collect();
            let _ = writeln!(text, "{},{},{},{},{}", ts, sym, r16, r96, cells.join(","));
        }
    }
    let table = rcsr::data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    let path = p(dir, name);
    store::write_store_with_time(&path, &table, 1000, Some(0)).unwrap();
    path
}

fn write_text(dir: &Path, name: &str, text: &str) -> String {
    let path = p(dir, name);
    std::fs::write(&path, text).unwrap();
    path
}

fn permute(data: &str, spec: &str, out: &str, block: &str, seed: &str, extra: &[&str]) -> Output {
    let mut args = vec!["panel", "permute-labels", "--data", data, "--target-spec", spec, "--label", "h24", "--block", block, "--seed", seed, "--out", out];
    args.extend_from_slice(extra);
    run(&args)
}

/// The stored values of a column as bits (`i64` values as their two's complement), its codes
/// and its levels.
fn column_bits(path: &str, name: &str) -> (Vec<u64>, Vec<u32>, Vec<String>) {
    let s = FileStore::open(path).unwrap();
    let c = s.header().column_index(name).unwrap_or_else(|| panic!("{path}: no column {name}"));
    let (mut bits, mut codes) = (Vec::new(), Vec::new());
    for g in 0..s.n_groups() {
        match s.header().dtypes()[c] {
            DType::I64 => bits.extend(s.read_i64(g, c).unwrap().into_iter().map(|v| v as u64)),
            DType::F64 => bits.extend(s.read_f64_group(g, c).unwrap().into_iter().map(f64::to_bits)),
            DType::U32 => {}
        }
        codes.extend(s.read_codes(g, c).unwrap());
    }
    (bits, codes, s.header().levels(c).map(|l| l.to_vec()).unwrap_or_default())
}

fn floats(path: &str, name: &str) -> Vec<f64> {
    column_bits(path, name).0.into_iter().map(f64::from_bits).collect()
}

fn times(path: &str) -> Vec<i64> {
    column_bits(path, "ts").0.into_iter().map(|v| v as i64).collect()
}

fn symbols_of(path: &str) -> Vec<String> {
    let (_, codes, levels) = column_bits(path, "symbol");
    codes.iter().map(|&c| levels[c as usize].clone()).collect()
}

/// Source row of every finite `y`: the fixture makes each finite `y` distinct.
fn row_of_y(y: &[f64]) -> BTreeMap<u64, usize> {
    let map: BTreeMap<u64, usize> = y.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| (v.to_bits(), i)).collect();
    assert_eq!(map.len(), y.iter().filter(|v| v.is_finite()).count(), "the fixture gives every finite y its own value");
    map
}

fn sorted_bits(v: impl Iterator<Item = f64>) -> Vec<u64> {
    let mut b: Vec<u64> = v.map(f64::to_bits).collect();
    b.sort_unstable();
    b
}

#[test]
fn permute_labels_moves_values_within_each_symbol() {
    let s = Scratch::new("permute-within");
    let dir = s.path();
    let src = write_null_panel(dir, "src.rcsb", 60 * 24, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let out = p(dir, "null/n1.rcsb");
    let o = permute(&src, &spec, &out, "5", "1", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));

    let y = floats(&src, &lbl("y"));
    let syms = symbols_of(&src);
    assert_eq!(symbols_of(&out), syms);
    let live: Vec<usize> = (0..y.len()).filter(|&i| y[i].is_finite()).collect();
    assert_eq!(live.len(), 3 * 60 - 12, "60 grid days of 3 symbols, 12 NaN y days of BBB");
    for kind in MOVED {
        let (a, b) = (floats(&src, &lbl(kind)), floats(&out, &lbl(kind)));
        for sym in SYMBOLS {
            let rows: Vec<usize> = live.iter().copied().filter(|&i| syms[i] == sym).collect();
            assert_eq!(sorted_bits(rows.iter().map(|&i| a[i])), sorted_bits(rows.iter().map(|&i| b[i])), "{kind} of {sym}");
        }
        for i in (0..y.len()).filter(|&i| !y[i].is_finite()) {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "{kind} row {i} has a NaN y and keeps its value");
        }
    }
    let y_out = floats(&out, &lbl("y"));
    assert!(live.iter().any(|&i| y_out[i].to_bits() != y[i].to_bits()), "the permutation moves some y");
    for name in ["ts", "symbol", "r16", "r96", "lbl_h24_t1", "lbl_h24_weight"] {
        assert_eq!(column_bits(&src, name), column_bits(&out, name), "{name} is byte-identical");
    }

    let m = read_json(&format!("{out}.manifest.json"));
    assert_eq!(at(&m, &["schema"]).as_str(), Some("rcsr-null-store/1"));
    let null = at(&m, &["null"]);
    assert_eq!(at(null, &["label"]).as_str(), Some("h24"));
    assert_eq!(at(null, &["block"]).as_f64(), Some(5.0));
    assert_eq!(at(null, &["seed"]).as_f64(), Some(1.0));
    assert_eq!(at(null, &["order"]).as_str(), Some("per_symbol"));
    assert_eq!(at(null, &["source_sha256"]).as_str(), Some(rcsr::manifest::sha256_file(&src).unwrap().as_str()));
    assert_eq!(std::fs::read_to_string(p(dir, "null/n1.targets.json")).unwrap(), SPEC, "the spec carries over unchanged");

    // --columns keeps the view: time, symbol, the named features and the family, with the same order.
    let view = p(dir, "null/n1-view.rcsb");
    let o = permute(&src, &spec, &view, "5", "1", &["--columns", "r16"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let names = FileStore::open(&view).unwrap().header().names();
    let mut want = vec!["ts".to_string(), "symbol".to_string(), "r16".to_string()];
    want.extend(KINDS.map(lbl));
    assert_eq!(names, want);
    for kind in KINDS {
        assert_eq!(column_bits(&out, &lbl(kind)), column_bits(&view, &lbl(kind)), "{kind} with --columns");
    }
    // Without --block the command refuses: the block length is pre-registered.
    let o = run(&["panel", "permute-labels", "--data", &src, "--target-spec", &spec, "--label", "h24", "--seed", "1", "--out", &p(dir, "x.rcsb")]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("--block"), "{}", stderr(&o));
}

#[test]
fn permute_labels_moves_the_family_jointly() {
    let s = Scratch::new("permute-joint");
    let dir = s.path();
    let src = write_null_panel(dir, "src.rcsb", 40 * 24, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let out = p(dir, "n.rcsb");
    let o = permute(&src, &spec, &out, "3", "9", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let syms = symbols_of(&src);
    let rows = row_of_y(&floats(&src, &lbl("y")));
    let src_cols: Vec<Vec<f64>> = MOVED.iter().map(|k| floats(&src, &lbl(k))).collect();
    let out_cols: Vec<Vec<f64>> = MOVED.iter().map(|k| floats(&out, &lbl(k))).collect();
    let y_out = &out_cols[1];
    let mut moved = 0;
    for r in (0..y_out.len()).filter(|&r| y_out[r].is_finite()) {
        let from = rows[&y_out[r].to_bits()];
        assert_eq!(syms[from], syms[r], "row {r} takes a row of its own symbol");
        for (k, kind) in MOVED.iter().enumerate() {
            assert_eq!(out_cols[k][r].to_bits(), src_cols[k][from].to_bits(), "row {r}: {kind} comes from the source row of y ({from})");
        }
        moved += usize::from(from != r);
    }
    assert!(moved > 0, "the permutation moves some rows");
}

#[test]
fn permute_labels_bytes_follow_the_seed() {
    let s = Scratch::new("permute-seed");
    let dir = s.path();
    let src = write_null_panel(dir, "src.rcsb", 40 * 24, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let (a, b, c) = (p(dir, "a.rcsb"), p(dir, "b.rcsb"), p(dir, "c.rcsb"));
    for (out, seed) in [(&a, "4"), (&b, "4"), (&c, "5")] {
        let o = permute(&src, &spec, out, "4", seed, &[]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
    assert!(std::fs::read(&a).unwrap() == std::fs::read(&b).unwrap(), "the same seed gives the same store bytes");
    assert_eq!(at(&read_json(&format!("{a}.manifest.json")), &["null"]), at(&read_json(&format!("{b}.manifest.json")), &["null"]));
    let (ya, yc) = (floats(&a, &lbl("y")), floats(&c, &lbl("y")));
    assert!(ya.iter().zip(&yc).any(|(x, z)| x.to_bits() != z.to_bits()), "another seed gives another order");
}

#[test]
fn permute_labels_keeps_every_value_on_its_side_of_the_lock() {
    let s = Scratch::new("permute-lock");
    let dir = s.path();
    std::fs::create_dir_all(dir.join("src")).unwrap();
    let (ls, le) = (T0 + 20 * DAY, T0 + 24 * DAY);
    let src = write_null_panel(&dir.join("src"), "src.rcsb", 44 * 24, &|t| t >= ls && t < le);
    let spec = write_text(&dir.join("src"), "src.targets.json", SPEC);
    let range = Lockbox { start: ls, end: le };
    lockbox::write_sidecar_with(&src, &range, "test fixture", ",\"lockbox_embargo\":{\"seconds\":172800}").unwrap();
    let record = write_text(
        dir,
        "locked-tail.json",
        &format!("{{\"schema\":\"rcsr-locked-tail/1\",\"range\":{{\"start_epoch\":{ls},\"end_epoch\":{le},\"end_exclusive\":true}}}}\n"),
    );
    let rec = rcsr::lockrec::LockRecord::load(&record).unwrap();
    rcsr::registry::write_schema_for(&src, "2", Some(&rec)).unwrap();
    let out = p(dir, "out/n.rcsb");
    let o = permute(&src, &spec, &out, "3", "2", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));

    let m = read_json(&format!("{out}.manifest.json"));
    let null = at(&m, &["null"]);
    let purge = at(null, &["purge_seconds"]).as_f64().unwrap() as i64;
    let embargo = at(null, &["embargo_seconds"]).as_f64().unwrap() as i64;
    assert_eq!(purge, DAY + HOUR, "entry offset plus horizon of the spec");
    assert!(embargo >= 172_800, "at least the declared embargo, got {embargo}");
    assert_eq!(at(&m, &["lock_record", "record_sha256"]).as_str(), Some(rec.sha256.as_str()), "the lock record carries over");
    assert_eq!(lockbox::stored(&out).unwrap().map(|(_, r)| r), Some(range), "the declaration carries over");
    let schema = std::fs::read_to_string(format!("{out}.schema.json")).unwrap();
    assert!(schema.contains(&rec.sha256), "the schema carries the lock record");

    let side = |t: i64| -> &'static str {
        if t < ls - purge {
            "before"
        } else if t < ls {
            "purge"
        } else if t < le + embargo {
            "embargo"
        } else {
            "after"
        }
    };
    let time = times(&src);
    assert_eq!(time.iter().filter(|&&t| t >= ls && t < le).count(), 0, "the source has no row inside the range");
    let y = floats(&src, &lbl("y"));
    let rows = row_of_y(&y);
    let y_out = floats(&out, &lbl("y"));
    let src_cols: Vec<Vec<f64>> = KINDS.iter().map(|k| floats(&src, &lbl(k))).collect();
    let out_cols: Vec<Vec<f64>> = KINDS.iter().map(|k| floats(&out, &lbl(k))).collect();
    let mut banded = 0;
    for r in (0..y.len()).filter(|&r| y[r].is_finite()) {
        let from = rows[&y_out[r].to_bits()];
        match side(time[r]) {
            "purge" | "embargo" => {
                banded += 1;
                for (k, kind) in KINDS.iter().enumerate() {
                    assert_eq!(out_cols[k][r].to_bits(), src_cols[k][r].to_bits(), "row {r} in the {} band keeps {kind}", side(time[r]));
                }
            }
            s => assert_eq!(side(time[from]), s, "row {r} ({s}) takes a value from row {from} ({})", side(time[from])),
        }
    }
    assert!(banded >= 6, "the fixture holds label rows in the purge and embargo bands, got {banded}");
    let o = run(&["lock", "check", "--record", &record, &p(dir, "out")]);
    assert_eq!(code(&o), 0, "{}{}", String::from_utf8_lossy(&o.stdout), stderr(&o));

    // A source with rows inside the locked range is refused and writes nothing.
    std::fs::create_dir_all(dir.join("full")).unwrap();
    let full = write_null_panel(&dir.join("full"), "full.rcsb", 44 * 24, &|_| false);
    lockbox::write_sidecar(&full, &range, "test fixture").unwrap();
    let refused = p(dir, "refused/n.rcsb");
    let o = permute(&full, &spec, &refused, "3", "2", &[]);
    assert_eq!(code(&o), 3, "{}", stderr(&o));
    assert!(stderr(&o).contains("inside the locked range"), "{}", stderr(&o));
    assert!(!Path::new(&refused).exists());
}

/// A `--fitness pnl` search, `walkforward --from-run` and `trials --table --null-wf` on the
/// null store with the flags of the source; the table refuses the null report as a candidate.
#[test]
fn null_store_runs_search_walkforward_and_the_table() {
    let s = Scratch::new("permute-e2e");
    let dir = s.path();
    let src = write_null_panel(dir, "src.rcsb", 90 * 24, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let null = p(dir, "null/n1.rcsb");
    let o = permute(&src, &spec, &null, "5", "1", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let window = format!("v,{},{},{},{}", T0, T0 + 60 * DAY, T0 + 62 * DAY, T0 + 90 * DAY);
    let report = |data: &str, tag: &str| -> String {
        let o = run(&[
            "--data", data, "--target", "lbl_h24_y", "--vars", "r16,r96", "--time", "ts", "--group-by", "symbol", "--target-spec", &spec,
            "--split", "temporal", "--cpu", "--no-dev-log", "--seed", "3", "--generations", "3", "--pop", "16", "--json", "--quiet",
            "--objective", "pnl", "--fitness", "pnl", "--validation-frac", "0.25", "--test-frac", "0.2", "--fill", "drop",
        ]);
        assert_eq!(code(&o), 0, "{tag} search: {}", stderr(&o));
        let run_path = p(dir, &format!("{tag}.run.json"));
        std::fs::write(&run_path, &o.stdout).unwrap();
        let out = p(dir, &format!("{tag}.wf.json"));
        let o = run(&[
            "walkforward", "--data", data, "--time", "ts", "--group-by", "symbol", "--from-run", &run_path, "--window", &window,
            "--allow-in-sample-windows", "--emit-series", "--out", &out,
        ]);
        assert_eq!(code(&o), 0, "{tag} walkforward: {}", stderr(&o));
        out
    };
    let real = report(&src, "real");
    let nul = report(&null, "null");
    assert!(read_json(&real).get("null_store").is_none(), "a report of real data has no marker");
    let marker = at(&read_json(&nul), &["null_store"]).clone();
    assert_eq!(at(&marker, &["label"]).as_str(), Some("h24"));
    assert_eq!(at(&marker, &["seed"]).as_f64(), Some(1.0));

    let o = run(&["trials", "--wf", &real, "--table", "--null-wf", &nul, "--draws", "20"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let table = Json::parse(String::from_utf8_lossy(&o.stdout).trim()).expect("table JSON");
    let rows = match at(&table, &["rows"]) {
        Json::Arr(a) => a.clone(),
        _ => panic!("rows is not a list"),
    };
    let q = at(&rows[0], &["metrics", "null_q"]).as_f64().expect("null_q is a number");
    assert!((0.0..=1.0).contains(&q), "null_q {q}");

    for args in [vec!["trials", "--wf", real.as_str(), nul.as_str(), "--table", "--draws", "20"], vec!["trials", "--wf", nul.as_str(), "--table", "--draws", "20"]] {
        let o = run(&args);
        assert_eq!(code(&o), 2, "{}", stderr(&o));
        let e = stderr(&o);
        assert!(e.contains(nul.as_str()) && e.contains("null store") && e.contains("--null-wf"), "{e}");
    }
}
