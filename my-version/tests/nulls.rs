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
/// `fwd`, `long_net`, `action`, `t1` and `weight` stay finite. Every finite `y` and every finite
/// `fwd` is distinct.
fn write_null_panel(dir: &Path, name: &str, hours: i64, skip: &dyn Fn(i64) -> bool) -> String {
    write_warm_panel(dir, name, hours, 0, skip)
}

/// [`write_null_panel`] whose first `warmup` grid days of every symbol have a NaN `y`, `exit_ret`,
/// `m_net` and `m_barrier` and a finite return (the warm-up of a label's volatility window).
fn write_warm_panel(dir: &Path, name: &str, hours: i64, warmup: i64, skip: &dyn Fn(i64) -> bool) -> String {
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
                if (k == 1 && (i / 24) % 5 == 2) || i / 24 < warmup {
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
    let fwd = floats(&src, &lbl("fwd"));
    for kind in MOVED {
        let (a, b) = (floats(&src, &lbl(kind)), floats(&out, &lbl(kind)));
        for sym in SYMBOLS {
            let rows: Vec<usize> = live.iter().copied().filter(|&i| syms[i] == sym).collect();
            assert_eq!(sorted_bits(rows.iter().map(|&i| a[i])), sorted_bits(rows.iter().map(|&i| b[i])), "{kind} of {sym}");
            let gap: Vec<usize> = (0..y.len()).filter(|&i| syms[i] == sym && !y[i].is_finite() && fwd[i].is_finite()).collect();
            assert_eq!(sorted_bits(gap.iter().map(|&i| a[i])), sorted_bits(gap.iter().map(|&i| b[i])), "{kind} of the gap rows of {sym}");
        }
        for i in (0..y.len()).filter(|&i| !y[i].is_finite() && !fwd[i].is_finite()) {
            assert_eq!(a[i].to_bits(), b[i].to_bits(), "{kind} row {i} has no label and keeps its value");
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
    // Gap rows (a NaN y and a finite return) stay on their side of the lock too.
    let (fwd, fwd_out) = (&src_cols[0], &out_cols[0]);
    let by_fwd: BTreeMap<u64, usize> = fwd.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| (v.to_bits(), i)).collect();
    let gaps: Vec<usize> = (0..y.len()).filter(|&r| !y[r].is_finite() && fwd[r].is_finite()).collect();
    assert!(!gaps.is_empty(), "the fixture holds gap rows");
    for &r in &gaps {
        let from = by_fwd[&fwd_out[r].to_bits()];
        assert_eq!(side(time[from]), side(time[r]), "gap row {r} ({}) takes a return from row {from} ({})", side(time[r]), side(time[from]));
        if matches!(side(time[r]), "purge" | "embargo") {
            assert_eq!(from, r, "gap row {r} in the {} band keeps its return", side(time[r]));
        }
    }
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

/// Rows with a NaN `y` and a finite return (8 warm-up days per symbol, BBB's gap days) move in a
/// pool of their own: the priced rows of the null store are those of the source, and no such row
/// keeps its source return. The pools hold 8, 14 and 8 rows, at least two blocks of 3. A circular-
/// block order can map a whole block onto itself, so the last property holds for this fixture and
/// seed 1 (checked against a standalone copy of the order), not for every seed.
#[test]
fn permute_labels_moves_gap_rows_in_their_own_pool() {
    let s = Scratch::new("permute-gap");
    let dir = s.path();
    let src = write_warm_panel(dir, "src.rcsb", 40 * 24, 8, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let out = p(dir, "n.rcsb");
    let o = permute(&src, &spec, &out, "3", "1", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let syms = symbols_of(&src);
    let (y, y_out) = (floats(&src, &lbl("y")), floats(&out, &lbl("y")));
    let (fwd, fwd_out) = (floats(&src, &lbl("fwd")), floats(&out, &lbl("fwd")));
    let priced = |v: &[f64]| (0..v.len()).filter(|&i| v[i].is_finite()).collect::<Vec<_>>();
    assert_eq!(priced(&fwd_out), priced(&fwd), "the null book prices the rows of the real book");
    assert_eq!(priced(&y_out), priced(&y), "the fit rows are those of the source");
    let by_fwd: BTreeMap<u64, usize> = fwd.iter().enumerate().filter(|(_, v)| v.is_finite()).map(|(i, v)| (v.to_bits(), i)).collect();
    assert_eq!(by_fwd.len(), priced(&fwd).len(), "the fixture gives every finite return its own value");
    let gaps: Vec<usize> = (0..y.len()).filter(|&i| !y[i].is_finite() && fwd[i].is_finite()).collect();
    assert_eq!(gaps.len(), 8 + 14 + 8);
    let src_cols: Vec<Vec<f64>> = MOVED.iter().map(|k| floats(&src, &lbl(k))).collect();
    let out_cols: Vec<Vec<f64>> = MOVED.iter().map(|k| floats(&out, &lbl(k))).collect();
    for &r in &gaps {
        assert_ne!(fwd_out[r].to_bits(), fwd[r].to_bits(), "gap row {r} ({}) keeps its source return", syms[r]);
        let from = by_fwd[&fwd_out[r].to_bits()];
        assert!(gaps.contains(&from) && syms[from] == syms[r], "gap row {r} takes the return of row {from}, not a gap row of {}", syms[r]);
        for (k, kind) in MOVED.iter().enumerate() {
            assert_eq!(out_cols[k][r].to_bits(), src_cols[k][from].to_bits(), "gap row {r}: {kind} comes from row {from}");
        }
    }
    let m = read_json(&format!("{out}.manifest.json"));
    let pools = match at(&m, &["null", "pools"]) {
        Json::Arr(a) => a.clone(),
        _ => panic!("null.pools is not a list"),
    };
    for (sym, label_rows, gap_rows) in [("AAA", 32.0, 8.0), ("BBB", 26.0, 14.0), ("CCC", 32.0, 8.0)] {
        let pool = pools.iter().find(|p| at(p, &["symbol"]).as_str() == Some(sym)).unwrap_or_else(|| panic!("no pool of {sym}"));
        assert_eq!(at(pool, &["label_rows"]).as_f64(), Some(label_rows), "{sym}");
        assert_eq!(at(pool, &["gap_rows"]).as_f64(), Some(gap_rows), "{sym}");
    }
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

// ---------------------------------------------------------------------------
// #1304: the marker travels with the store; the ledger and plain `trials` refuse it
// ---------------------------------------------------------------------------

/// `walkforward --expr r16` on `data` over one window of the 90-day fixture, plus `extra`.
fn wf_expr(data: &str, spec: &str, out: &str, extra: &[&str]) -> Output {
    let window = format!("v,{},{},{},{}", T0, T0 + 60 * DAY, T0 + 62 * DAY, T0 + 90 * DAY);
    let mut args = vec![
        "walkforward", "--data", data, "--target", "lbl_h24_y", "--target-spec", spec, "--expr", "r16", "--time", "ts", "--group-by", "symbol",
        "--window", &window, "--emit-series", "--out", out,
    ];
    args.extend_from_slice(extra);
    run(&args)
}

/// The 90-day source store, its spec and the null store of label h24 (block 5, seed 1).
fn null_fixture(dir: &Path) -> (String, String, String) {
    let src = write_null_panel(dir, "src.rcsb", 90 * 24, &|_| false);
    let spec = write_text(dir, "src.targets.json", SPEC);
    let null = p(dir, "null/n1.rcsb");
    let o = permute(&src, &spec, &null, "5", "1", &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    (src, spec, null)
}

#[test]
fn null_marker_survives_store_copy() {
    let s = Scratch::new("null-copy");
    let dir = s.path();
    let (src, spec, null) = null_fixture(dir);
    let copy = p(dir, "copy.rcsb");
    std::fs::copy(&null, &copy).unwrap();
    assert!(!Path::new(&format!("{copy}.manifest.json")).exists(), "the copy has no sidecar");
    assert_eq!(rcsr::manifest::sha256_file(&copy).unwrap(), rcsr::manifest::sha256_file(&null).unwrap());
    let manifest_null = at(&read_json(&format!("{null}.manifest.json")), &["null"]).clone();
    let marker = rcsr::permute_labels::marker(&copy).expect("the copy of a null store is marked");
    assert_eq!(Json::parse(&marker).expect("the marker is JSON"), manifest_null, "the store header carries the manifest's null object");
    assert_eq!(rcsr::permute_labels::marker(&src), None, "the source store is real data");

    let out = p(dir, "copy.wf.json");
    let o = wf_expr(&copy, &spec, &out, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let report = read_json(&out);
    let m = report.get("null_store").expect("a report priced on the copy carries null_store");
    assert_eq!(at(m, &["label"]).as_str(), Some("h24"));
    assert_eq!(at(m, &["seed"]).as_f64(), Some(1.0));
    assert_eq!(at(m, &["block"]).as_f64(), Some(5.0));
    assert_eq!(at(m, &["source_sha256"]).as_str(), Some(rcsr::manifest::sha256_file(&src).unwrap().as_str()));
    let real = p(dir, "real.wf.json");
    let o = wf_expr(&src, &spec, &real, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert!(read_json(&real).get("null_store").is_none(), "a report of real data has no marker");
}

#[test]
fn walkforward_ledger_refuses_null_store() {
    let s = Scratch::new("null-ledger");
    let dir = s.path();
    let (src, spec, null) = null_fixture(dir);
    let copy = p(dir, "copy.rcsb");
    std::fs::copy(&null, &copy).unwrap();
    let ledger = p(dir, "ledger.jsonl");
    let o = wf_expr(&src, &spec, &p(dir, "real.wf.json"), &["--ledger", &ledger]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let before = std::fs::read(&ledger).unwrap();
    assert_eq!(before.iter().filter(|&&b| b == b'\n').count(), 1, "the real store adds one candidate");
    for data in [&null, &copy] {
        let out = p(dir, "null.wf.json");
        let o = wf_expr(data, &spec, &out, &["--ledger", &ledger]);
        assert_eq!(code(&o), 2, "{data}: {}{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
        let e = stderr(&o);
        assert!(e.contains(data.as_str()) && e.contains("null store") && e.contains("label h24, seed 1") && e.contains("--ledger"), "{e}");
        assert!(std::fs::read(&ledger).unwrap() == before, "{data}: the ledger keeps its bytes");
        assert!(!Path::new(&out).exists(), "{data}: no report is written");
    }
}

/// A null store written before #1304 is marked by its manifest only. A candidate registered
/// on it while unmarked and sealed is still refused a sealed read: the read would spend a
/// lockbox read on null data.
#[test]
fn walkforward_locked_test_refuses_null_store() {
    let s = Scratch::new("null-locked");
    let dir = s.path();
    let mut csv = String::from("ts,x,target\n");
    for i in 0..300 {
        let x = ((i * 37) % 17) as f64 - 8.0;
        let _ = writeln!(csv, "{},{},{}", i, x, 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004);
    }
    let data = write_text(dir, "d.csv", &csv);
    let (ledger, sealed) = (p(dir, "ledger.jsonl"), p(dir, "locked_candidates.json"));
    let wf = |extra: &[&str]| -> Output {
        let mut a = vec![
            "walkforward", "--data", &data, "--target", "target", "--expr", "x", "--time", "ts", "--horizon", "1", "--cost-bps", "5",
            "--lockbox", "200,300", "--ledger", &ledger, "--periods-per-year", "31536000", "--json",
        ];
        a.extend_from_slice(extra);
        run(&a)
    };
    let o = wf(&["--window", "dev,0,60,70,150"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = run(&["protocol", "seal", "--ledger", &ledger, "--out", &sealed]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let manifest = format!(
        "{{\"schema\":\"rcsr-null-store/1\",\"tool\":\"rcsr panel permute-labels\",\"null\":{{\"label\":\"h24\",\"block\":5,\"seed\":7,\"order\":\"per_symbol\"}},\"output_sha256\":\"{}\"}}\n",
        rcsr::manifest::sha256_file(&data).unwrap()
    );
    write_text(dir, "d.csv.manifest.json", &manifest);
    let before = std::fs::read(&ledger).unwrap();
    let o = wf(&["--locked-test", "--sealed", &sealed, "--window", "locked,0,150,210,300"]);
    assert_eq!(code(&o), 2, "stdout: {} stderr: {}", String::from_utf8_lossy(&o.stdout), stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("null store") && e.contains("label h24, seed 7") && e.contains("lockbox read"), "{e}");
    assert!(std::fs::read(&ledger).unwrap() == before, "the ledger records no read");
    assert!(!Path::new(&rcsr::protocol::marker_path(&data)).exists(), "no read marker is written");
    let o = wf(&["--window", "dev2,0,50,60,140"]);
    assert_eq!(code(&o), 2, "{}", stderr(&o));
    assert!(stderr(&o).contains("never a protocol candidate"), "{}", stderr(&o));
    assert!(std::fs::read(&ledger).unwrap() == before, "the ledger adds no candidate");
}

#[test]
fn trials_wf_refuses_null_report() {
    let s = Scratch::new("null-trials");
    let dir = s.path();
    let (src, spec, null) = null_fixture(dir);
    let (real, nul) = (p(dir, "real.wf.json"), p(dir, "null.wf.json"));
    for (data, out) in [(&src, &real), (&null, &nul)] {
        let o = wf_expr(data, &spec, out, &[]);
        assert_eq!(code(&o), 0, "{}", stderr(&o));
    }
    assert!(read_json(&nul).get("null_store").is_some());
    let o = run(&["trials", "--wf", &real, "--draws", "20"]);
    assert_eq!(code(&o), 0, "a real report: {}", stderr(&o));
    let refused: [Vec<&str>; 4] = [
        vec!["trials", "--wf", &nul, "--draws", "20"],
        vec!["trials", "--wf", &real, &nul, "--draws", "20"],
        vec!["trials", "--wf", &nul, "--members", "all", "--draws", "20"],
        vec!["trials", "--compare", &real, &nul, "--draws", "20"],
    ];
    for args in refused {
        let o = run(&args);
        assert_eq!(code(&o), 2, "{args:?}: {}{}", String::from_utf8_lossy(&o.stdout), stderr(&o));
        let e = stderr(&o);
        assert!(e.contains(nul.as_str()) && e.contains("null_store") && e.contains("label h24") && e.contains("seed 1"), "{args:?}: {e}");
        assert!(o.stdout.is_empty(), "{args:?}: no output");
    }
    let o = run(&["trials", "--wf", &real, "--table", "--null-wf", &nul, "--draws", "20"]);
    assert_eq!(code(&o), 0, "--null-wf still reads a null report: {}", stderr(&o));
}

/// A copy of the store `src` at `out` whose header carries `marker` as `null_store`.
fn mark_store(src: &str, out: &str, marker: &str) {
    copy_store(src, out, &|_| true, Some(marker));
}

/// A copy of the columns of `src` that `keep` accepts at `out`, whose header carries `marker`
/// as `null_store` (no member for `None`).
fn copy_store(src: &str, out: &str, keep: &dyn Fn(&str) -> bool, marker: Option<&str>) {
    let s = FileStore::open(src).unwrap();
    let h = s.header();
    let cols: Vec<usize> = (0..h.n_cols).filter(|&c| keep(&h.column(c).unwrap().name)).collect();
    let plans: Vec<store::ColumnPlan> = cols
        .iter()
        .map(|&c| {
            let i = h.column(c).unwrap();
            store::ColumnPlan { name: i.name, numeric: i.numeric, dtype: i.dtype, levels: i.levels, has_codes: i.has_codes }
        })
        .collect();
    let rows = h.group_rows;
    store::write_store_grouped_marked(out, &plans, h.n_rows, rows, marker, |lo, _, out_cols| {
        let g = lo / rows;
        for (k, (&c, plan)) in cols.iter().zip(&plans).enumerate() {
            match plan.dtype {
                DType::F64 => out_cols[k].f64s.extend(s.read_f64_group(g, c)?),
                DType::I64 => out_cols[k].i64s.extend(s.read_i64(g, c)?),
                DType::U32 => {}
            }
            if plan.has_codes {
                out_cols[k].codes.extend(s.read_codes(g, c)?);
            }
        }
        Ok(())
    })
    .unwrap();
}

/// `ladder --locked-test` spends the baseline read of a sealed list on any data: a store with
/// a `null_store` header member, or an older null store marked by its manifest, is refused
/// before the read is authorised; the real store then reads once.
#[test]
fn ladder_locked_test_refuses_null_store() {
    let s = Scratch::new("null-ladder");
    let dir = s.path();
    let mut csv = String::from("ts,sym,f1,f2,y\n");
    for g in 0..3i64 {
        for i in 0..500i64 {
            let f1 = (((i * 37 + g * 11) % 17) as f64 - 8.0) / 8.0;
            let f2 = (((i * 19 + g * 5) % 13) as f64 - 6.0) / 6.0;
            let y = 0.004 * f1 + (((i * 13 + g) % 7) as f64 - 3.0) * 0.001;
            let _ = writeln!(csv, "{},s{},{},{},{}", T0 + i * HOUR, g, f1, f2, y);
        }
    }
    let table = rcsr::data::parse_table_with_time(&csv, false, Some("ts")).unwrap();
    let (plain, marked, legacy) = (p(dir, "p.rcsb"), p(dir, "n.rcsb"), p(dir, "legacy.rcsb"));
    store::write_store_with_time(&plain, &table, 1000, Some(0)).unwrap();
    mark_store(&plain, &marked, "{\"label\":\"y\",\"block\":5,\"seed\":11,\"order\":\"per_symbol\"}");
    std::fs::copy(&plain, &legacy).unwrap();
    write_text(dir, "legacy.rcsb.manifest.json", "{\"schema\":\"rcsr-null-store/1\",\"null\":{\"label\":\"y\",\"block\":5,\"seed\":12}}\n");
    let windows = write_text(
        dir,
        "w.csv",
        &format!(
            "scheme,label,train_start,train_end,validation_start,test_start,test_end\nlocked,locked_0,{},{},{},{},{}\n",
            T0,
            T0 + 398 * HOUR,
            T0 + 250 * HOUR,
            T0 + 400 * HOUR,
            T0 + 500 * HOUR
        ),
    );
    let range = Lockbox { start: T0 + 400 * HOUR, end: T0 + 500 * HOUR };
    for d in [&plain, &marked, &legacy] {
        lockbox::write_sidecar(d, &range, "test").unwrap();
    }
    let (ledger, sealed) = (p(dir, "ledger.jsonl"), p(dir, "sealed.json"));
    rcsr::protocol::Ledger::new(&ledger)
        .add_candidate(&rcsr::protocol::Candidate {
            id: "c1".into(),
            spec_sha256: "a".repeat(64),
            data_sha256: "b".repeat(64),
            build_commit: "build".into(),
            report_sha256: "c".repeat(64),
            expression: "x".into(),
            locked_test: Some(rcsr::protocol::LockedRange { start: range.start, end: range.end, source: "store".into() }),
        })
        .unwrap();
    let o = run(&["protocol", "seal", "--ledger", &ledger, "--out", &sealed]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let ladder = |data: &str, out: &str| -> Output {
        run(&[
            "ladder", "--data", data, "--target", "y", "--time", "ts", "--group", "sym", "--features", "f1,f2", "--windows", &windows, "--scheme",
            "locked", "--purge-seconds", "7200", "--gbm-trees", "10", "--draws", "50", "--max-fit-rows", "600", "--out", out, "--locked-test",
            "--ledger", &ledger, "--sealed", &sealed,
        ])
    };
    let before = std::fs::read(&ledger).unwrap();
    for (data, seed) in [(&marked, 11), (&legacy, 12)] {
        let out = p(dir, "null.ladder.json");
        let o = ladder(data, &out);
        assert_eq!(code(&o), 2, "{data}: stdout: {} stderr: {}", String::from_utf8_lossy(&o.stdout), stderr(&o));
        let e = stderr(&o);
        assert!(e.contains(data.as_str()) && e.contains(&format!("null_store label y, seed {seed}")) && e.contains("lockbox read"), "{e}");
        assert!(std::fs::read(&ledger).unwrap() == before, "{data}: no read is recorded");
        assert!(!Path::new(&rcsr::protocol::marker_path(data)).exists(), "{data}: no read marker");
        assert!(!Path::new(&out).exists(), "{data}: no report");
    }
    let o = ladder(&plain, &p(dir, "real.ladder.json"));
    assert_eq!(code(&o), 0, "the real store reads once: {}", stderr(&o));
    assert!(std::fs::read_to_string(&ledger).unwrap().contains("\"event\":\"baseline_read\""));
}

/// `protocol regimes --after-read` writes the lock calendar into the ledger after the reads: a
/// store marked as a null store (an older one, by its manifest) is refused before the ledger is
/// touched.
#[test]
fn regimes_after_read_refuses_null_store() {
    let s = Scratch::new("null-regimes");
    let dir = s.path();
    let lock = (T0 + 12 * DAY, T0 + 17 * DAY);
    let mut text = String::from("ts,sym,x,close,y\n");
    for i in 0..17 * 24i64 {
        let day = (i / 24) as usize;
        let close = if day >= 12 { [100.0, 150.0, 100.0, 150.0, 100.0][day - 12] } else { 100.0 + day as f64 };
        let x = ((i * 37) % 17) as f64 - 8.0;
        let _ = writeln!(text, "{},BTC,{},{},{}", T0 + i * HOUR, x, close, 0.0003 * x + (((i * 13) % 7) as f64 - 3.0) * 0.0004);
    }
    let data = write_text(dir, "store.csv", &text);
    lockbox::write_sidecar(&data, &Lockbox { start: lock.0, end: lock.1 }, "test").unwrap();
    let (ledger, sealed) = (p(dir, "ledger.jsonl"), p(dir, "sealed.json"));
    let wf = |window: &str, extra: &[&str]| -> Output {
        let mut a = vec![
            "walkforward", "--data", &data, "--target", "y", "--time", "ts", "--horizon", "5", "--cost-bps", "5", "--periods-per-year", "6307200", "--json",
            "--ledger", &ledger, "--expr", "x", "--window", window,
        ];
        a.extend_from_slice(extra);
        run(&a)
    };
    let o = wf(&format!("dev,{},{},{},{}", T0, T0 + 100 * HOUR, T0 + 110 * HOUR, T0 + 200 * HOUR), &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = run(&["protocol", "seal", "--final", "--ledger", &ledger, "--out", &sealed]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let o = wf(&format!("locked,{},{},{},{}", lock.0 - 100 * HOUR, lock.0 - 10 * HOUR, lock.0, lock.1), &["--locked-test", "--sealed", &sealed]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    write_text(dir, "store.csv.manifest.json", "{\"schema\":\"rcsr-null-store/1\",\"null\":{\"label\":\"h24\",\"block\":5,\"seed\":4}}\n");
    let out = p(dir, "lock-cal.csv");
    let before = std::fs::read(&ledger).unwrap();
    let o = run(&[
        "protocol", "regimes", "--data", &data, "--time-col", "ts", "--price-col", "close", "--symbol-col", "sym", "--symbol", "BTC", "--out", &out, "--up",
        "0.3", "--down", "0.3", "--min-days", "0", "--sideways-days", "0", "--after-read", "--ledger", &ledger, "--sealed", &sealed, "--expect-reads", "1",
    ]);
    assert_eq!(code(&o), 2, "stdout: {} stderr: {}", String::from_utf8_lossy(&o.stdout), stderr(&o));
    let e = stderr(&o);
    assert!(e.contains("null_store label h24, seed 4") && e.contains("--after-read"), "{e}");
    assert!(std::fs::read(&ledger).unwrap() == before, "the ledger records no dating");
    assert!(!Path::new(&out).exists(), "no calendar is written");
}

// ---------------------------------------------------------------------------
// #1313: a store that takes columns from a null store is marked
// ---------------------------------------------------------------------------

fn append(target: &str, family: &str, columns: &str, out: &str, extra: &[&str]) -> Output {
    let mut args = vec!["panel", "append", "--data", target, "--family", family, "--columns", columns, "--out", out, "--locked-test", "none"];
    args.extend_from_slice(extra);
    run(&args)
}

fn ledger_lines(path: &str) -> usize {
    std::fs::read(path).map_or(0, |b| b.iter().filter(|&&c| c == b'\n').count())
}

fn header_marker(path: &str) -> Option<String> {
    FileStore::open(path).unwrap().header().null_store().map(str::to_string)
}

/// `panel append --family` takes the permuted labels of a null store (marked by its header, or
/// by the manifest of a null store written before #1304) into a panel without them: the output
/// carries the family's marker, so `walkforward --ledger` refuses it.
#[test]
fn panel_append_carries_the_family_null_marker() {
    let s = Scratch::new("null-append-family");
    let dir = s.path();
    let (src, spec, null) = null_fixture(dir);
    let nolbl = p(dir, "nolbl.rcsb");
    copy_store(&src, &nolbl, &|n| !n.starts_with("lbl_"), None);
    let legacy = p(dir, "legacy.rcsb");
    copy_store(&null, &legacy, &|_| true, None);
    write_text(dir, "legacy.rcsb.manifest.json", "{\"schema\":\"rcsr-null-store/1\",\"null\":{\"label\":\"h24\",\"block\":5,\"seed\":9,\"order\":\"per_symbol\"}}\n");
    assert_eq!(header_marker(&legacy), None, "the legacy store is marked by its manifest only");
    let ledger = p(dir, "ledger.jsonl");
    let o = wf_expr(&src, &spec, &p(dir, "real.wf.json"), &["--ledger", &ledger]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    let before = std::fs::read(&ledger).unwrap();
    for (family, seed) in [(&null, 1), (&legacy, 9)] {
        let want = rcsr::permute_labels::marker(family).expect("the family is a null store");
        let out = p(dir, &format!("gap{seed}.rcsb"));
        let o = append(&nolbl, family, "lbl_h24_fwd,lbl_h24_y", &out, &[]);
        assert_eq!(code(&o), 0, "{family}: {}", stderr(&o));
        let summary = stderr(&o);
        let wf_out = p(dir, &format!("gap{seed}.wf.json"));
        let o = wf_expr(&out, &spec, &wf_out, &["--ledger", &ledger]);
        assert_eq!(
            code(&o),
            2,
            "{family}: walkforward --ledger on the appended store; ledger lines {} -> {}; output null_store {:?}; {}",
            before.iter().filter(|&&c| c == b'\n').count(),
            ledger_lines(&ledger),
            header_marker(&out),
            stderr(&o)
        );
        let e = stderr(&o);
        assert!(e.contains(out.as_str()) && e.contains(&format!("label h24, seed {seed}")) && e.contains("--ledger"), "{e}");
        assert!(std::fs::read(&ledger).unwrap() == before, "{family}: the ledger keeps its bytes");
        assert!(!Path::new(&wf_out).exists(), "{family}: no report is written");
        assert_eq!(header_marker(&out).as_deref(), Some(want.as_str()), "{family}: the output carries the family's marker byte for byte");
        assert!(summary.contains(&format!("null_store label h24, seed {seed}")), "{summary}");
    }
}

/// A target and a family store marked by different null draws are refused before anything is
/// written; a target and a family of one draw carry its marker.
#[test]
fn panel_append_refuses_two_null_markers() {
    let s = Scratch::new("null-append-two");
    let dir = s.path();
    let (src, _spec, null) = null_fixture(dir);
    let marker = rcsr::permute_labels::marker(&null).unwrap();
    let other = marker.replacen("\"seed\":1,", "\"seed\":2,", 1);
    assert_ne!(other, marker);
    let (t_other, t_same) = (p(dir, "t-other.rcsb"), p(dir, "t-same.rcsb"));
    copy_store(&src, &t_other, &|n| !n.starts_with("lbl_"), Some(&other));
    copy_store(&src, &t_same, &|n| !n.starts_with("lbl_"), Some(&marker));
    let work = dir.join("refused");
    std::fs::create_dir_all(&work).unwrap();
    let out = p(&work, "out.rcsb");
    let scratch = p(&work, "scratch");
    let o = append(&t_other, &null, "lbl_h24_fwd,lbl_h24_y", &out, &["--scratch", &scratch]);
    assert_eq!(code(&o), 2, "output null_store {:?}: {}", Path::new(&out).exists().then(|| header_marker(&out)), stderr(&o));
    let e = stderr(&o);
    assert!(e.contains(t_other.as_str()) && e.contains(null.as_str()), "{e}");
    assert!(e.contains("null_store label h24, seed 2") && e.contains("null_store label h24, seed 1"), "{e}");
    let left: Vec<_> = std::fs::read_dir(&work).unwrap().map(|e| e.unwrap().file_name()).collect();
    assert!(left.is_empty(), "no output, sidecar, scratch directory or temporary column file: {left:?}");

    let out = p(dir, "same.rcsb");
    let o = append(&t_same, &null, "lbl_h24_fwd,lbl_h24_y", &out, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(header_marker(&out).as_deref(), Some(marker.as_str()));
}

/// A null store written before #1304 is marked by its manifest only. A store that takes its
/// columns (the target of `panel append`, the input of `rcsr enrich` and `rcsr panel rank`)
/// carries that marker in its header.
#[test]
fn every_appending_writer_carries_a_manifest_null_marker() {
    let s = Scratch::new("null-append-legacy");
    let dir = s.path();
    let (src, spec, null) = null_fixture(dir);
    let manifest = "{\"schema\":\"rcsr-null-store/1\",\"null\":{\"label\":\"h24\",\"block\":5,\"seed\":9,\"order\":\"per_symbol\"}}\n";
    let target = p(dir, "legacy-target.rcsb");
    copy_store(&null, &target, &|n| n != "lbl_h24_t1" && n != "lbl_h24_weight", None);
    write_text(dir, "legacy-target.rcsb.manifest.json", manifest);
    let want = rcsr::permute_labels::marker(&target).expect("the manifest marks the target");
    let out = p(dir, "appended.rcsb");
    let o = append(&target, &src, "lbl_h24_t1,lbl_h24_weight", &out, &[]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(header_marker(&out).as_deref(), Some(want.as_str()), "panel append: the target's manifest marker");

    let legacy = p(dir, "legacy.rcsb");
    copy_store(&null, &legacy, &|_| true, None);
    write_text(dir, "legacy.rcsb.manifest.json", manifest);
    let ranked = p(dir, "ranked.rcsb");
    let o = run(&[
        "panel", "rank", "--data", &legacy, "--column", "lbl_h24_fwd", "--name", "lbl_h24_xs", "--target-spec", &spec, "--min-members", "2", "--out",
        &ranked,
    ]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(header_marker(&ranked).as_deref(), Some(want.as_str()), "panel rank");

    let mut csv = String::from("ts,symbol,open,high,low,close,volume\n");
    for (k, sym) in SYMBOLS.iter().enumerate() {
        for i in 0..96i64 {
            let c = 100.0 + k as f64 + frac(i as f64 * 0.618_033_988_749_894_9);
            let _ = writeln!(csv, "{},{},{},{},{},{},{}", T0 + i * HOUR, sym, c, c + 0.5, c - 0.5, c + 0.1, 10 + i);
        }
    }
    let table = rcsr::data::parse_table_with_time(&csv, false, Some("ts")).unwrap();
    let bars = p(dir, "bars.rcsb");
    store::write_store_with_time(&bars, &table, 1000, Some(0)).unwrap();
    write_text(dir, "bars.rcsb.manifest.json", manifest);
    let enriched = p(dir, "enriched.rcsb");
    let o = run(&["enrich", "--data", &bars, "--out", &enriched, "--cpu", "--features", "r1"]);
    assert_eq!(code(&o), 0, "{}", stderr(&o));
    assert_eq!(header_marker(&enriched).as_deref(), Some(want.as_str()), "rcsr enrich");
}
