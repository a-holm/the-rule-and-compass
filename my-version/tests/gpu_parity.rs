//! #303: the epic B parity and batch-invariance harness.
//!
//! One table-driven gate for every device-registered enrichment column. The
//! column set is read from `enrich::default_registry()` filtered by
//! `enrich::gpu_supports`, never from a hand list, and each column is compared
//! with the CPU reference at the tier the registry declares:
//!
//! * `bitwise`: SHA-256 of the raw f64 bytes is equal, and no bit differs;
//! * `1e-12`: identical NaN/Inf masks and `|a-b| <= 1e-12 * (1 + max(|a|,|b|))`.
//!
//! A registered column with no tier (`reference`, empty, unknown) fails. The
//! harness also checks cold vs warm runs, `--chunk-rows` 1/4096/65536, store
//! group (symbol batching) layouts, prefix replay at 20 cuts with a
//! future-shift canary, panel prep and the PIT as-of joins. The `#[ignore]`
//! large-store test runs the same comparison on a real panel (see
//! `docs/ohlcv.md`). Device tests skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich::{self, GpuMode, Options, Report};
use rcsr::manifest::{sha256_hex, Sha256Hasher};
use rcsr::panel_pit::{self, PitBlock, PitParams};
use rcsr::panel_pit_gpu::{self, PitGpu};
use rcsr::panel_prep::{prepare as cpu_prepare, PrepBar, PrepOutput, PrepSegment};
use rcsr::panel_prep_gpu::{self, PrepGpu};
use rcsr::store::{self, FileStore};
use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

#[path = "common/pit.rs"]
mod pit;

const TOL: f64 = 1e-12;

/// Columns whose device output currently violates their declared tier. Each
/// entry is `(name, measured fixture result)`. They are excluded from the main
/// gate and asserted by `known_parity_bugs_stay_ignored`.
const KNOWN_BUGS: &[(&str, &str)] = &[
    ("amihud96", "flow, tier 1e-12: max rel 9.929e-1 (abs 1.635e2), first row 42074 CPU 0.171 GPU 85.46; also not prefix-causal (row 42074 GPU 85.46 full vs 4.137 after removing later rows)"),
    ("xs_rank_amihud96", "bitwise: 13 bit mismatches, max abs 6.087e-1; ranks of the device amihud96"),
    ("ou_hl96", "tier 1e-12: max rel 1.314e-7 (abs 1.382e-7); #298 accepted 1e-6, no matching tier"),
];

static HEAVY: Mutex<()> = Mutex::new(());

fn have_device() -> bool {
    if rcsr::gpu::probe().available() {
        true
    } else {
        eprintln!("skipping: no CUDA device");
        false
    }
}

// ---------------------------------------------------------------------------
// Tier table, driven by the registry
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct DeviceColumn {
    name: String,
    family: &'static str,
    tier: &'static str,
}

/// Every available registry column that has a device kernel in this build.
fn device_columns() -> Vec<DeviceColumn> {
    enrich::default_registry()
        .into_iter()
        .filter(|e| e.available && enrich::gpu_supports(&e.name))
        .map(|e| DeviceColumn { name: e.name, family: e.family, tier: e.parity_tier })
        .collect()
}

/// Violations of "every device column is in exactly one declared tier".
fn tier_violations(cols: &[(String, String)]) -> Vec<String> {
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    let mut out = Vec::new();
    for (name, tier) in cols {
        *seen.entry(name.as_str()).or_default() += 1;
        if tier != "bitwise" && tier != "1e-12" {
            out.push(format!("{name}: tier '{tier}' is not bitwise or 1e-12"));
        }
    }
    for (name, n) in seen {
        if n > 1 {
            out.push(format!("{name}: registered {n} times"));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Comparison
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Obs {
    rows: usize,
    finite: usize,
    mask_mismatch: usize,
    bit_mismatch: usize,
    max_abs: f64,
    max_rel: f64,
    /// First row (global index) where the mask or the 1e-12 bound fails, with both values.
    first_bad: Option<(usize, f64, f64)>,
    cpu_sha: Option<Sha256Hasher>,
    gpu_sha: Option<Sha256Hasher>,
}

impl Obs {
    fn new() -> Self {
        Obs {
            cpu_sha: Some(Sha256Hasher::new()),
            gpu_sha: Some(Sha256Hasher::new()),
            ..Default::default()
        }
    }

    fn push(&mut self, cpu: &[f64], gpu: &[f64]) {
        assert_eq!(cpu.len(), gpu.len(), "row count differs");
        let mut cb = Vec::with_capacity(cpu.len() * 8);
        let mut gb = Vec::with_capacity(cpu.len() * 8);
        for (&a, &b) in cpu.iter().zip(gpu.iter()) {
            let row = self.rows;
            self.rows += 1;
            cb.extend_from_slice(&a.to_le_bytes());
            gb.extend_from_slice(&b.to_le_bytes());
            if a.to_bits() != b.to_bits() {
                self.bit_mismatch += 1;
            }
            if a.is_nan() != b.is_nan() || a.is_infinite() != b.is_infinite() {
                self.mask_mismatch += 1;
                self.first_bad.get_or_insert((row, a, b));
                continue;
            }
            if a.is_finite() && b.is_finite() {
                self.finite += 1;
                let d = (a - b).abs();
                self.max_abs = self.max_abs.max(d);
                let rel = d / (1.0 + a.abs().max(b.abs()));
                self.max_rel = self.max_rel.max(rel);
                if rel > TOL {
                    self.first_bad.get_or_insert((row, a, b));
                }
            }
        }
        self.cpu_sha.as_mut().unwrap().update(&cb);
        self.gpu_sha.as_mut().unwrap().update(&gb);
    }

    /// `(cpu_sha, gpu_sha)` of the bytes pushed so far.
    fn shas(&mut self) -> (String, String) {
        (
            self.cpu_sha.take().unwrap().finalize(),
            self.gpu_sha.take().unwrap().finalize(),
        )
    }
}

fn observe(cpu: &[f64], gpu: &[f64]) -> (Obs, String, String) {
    let mut o = Obs::new();
    o.push(cpu, gpu);
    let (a, b) = o.shas();
    (o, a, b)
}

/// The tier verdict. `Ok` only when the column meets its declared tier.
fn verdict(tier: &str, o: &Obs, cpu_sha: &str, gpu_sha: &str) -> Result<(), String> {
    if o.mask_mismatch > 0 {
        return Err(format!("{} NaN/Inf mask mismatches, first {:?}", o.mask_mismatch, o.first_bad));
    }
    match tier {
        "bitwise" => {
            if cpu_sha != gpu_sha || o.bit_mismatch > 0 {
                Err(format!(
                    "bitwise column differs: {} bit mismatches, max abs {:.3e}",
                    o.bit_mismatch, o.max_abs
                ))
            } else {
                Ok(())
            }
        }
        "1e-12" => {
            if o.max_rel > TOL {
                Err(format!("max relative diff {:.3e} exceeds 1e-12 (max abs {:.3e}), first {:?}", o.max_rel, o.max_abs, o.first_bad))
            } else {
                Ok(())
            }
        }
        other => Err(format!("tier '{other}' is undeclared")),
    }
}

// ---------------------------------------------------------------------------
// Fixture: symbols x bars with gaps, NaNs, zero volume, H==L, ties, late
// listings (<20 members early), PIT non-members and stale MTF buckets.
// ---------------------------------------------------------------------------

const STEP: i64 = 900;
const T0: i64 = 1_700_000_000 - (1_700_000_000 % 86_400);
const LATE_START: usize = 3_000;

#[derive(Clone, Copy, Debug)]
struct Row {
    t: i64,
    sym: usize,
    o: f64,
    h: f64,
    l: f64,
    c: f64,
    v: f64,
    qv: f64,
    trades: f64,
    tb: f64,
    tq: f64,
    member: i64,
}

fn sym_name(k: usize) -> String {
    match k {
        0 => "BTCUSDT".to_string(),
        1 => "ETHUSDT".to_string(),
        _ => format!("SYM{k:02}"),
    }
}

fn mix(a: u64, b: u64) -> u64 {
    let mut x = a.wrapping_mul(0x9e37_79b9_7f4a_7c15) ^ b.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    x ^= x >> 29;
    x = x.wrapping_mul(0x94d0_49bb_1331_11eb);
    x ^ (x >> 32)
}

fn fixture(symbols: usize, per: usize) -> Vec<Row> {
    let mut rows = Vec::with_capacity(symbols * per);
    for k in 0..symbols {
        let mut close = 100.0 + k as f64;
        let start = if k >= 15 { LATE_START } else { 0 };
        for i in start..per {
            // Holes: every ~1k bars, a 5 h outage in symbol 3 (stale 1h/4h
            // bucket) and a >1 day outage in symbol 5 (stale 1d bucket).
            if i % 997 == 500 && k % 4 == 0 {
                continue;
            }
            if k == 3 && (per / 2..per / 2 + 20).contains(&i) {
                continue;
            }
            if k == 5 && (per * 3 / 4..per * 3 / 4 + 130).contains(&i) {
                continue;
            }
            let t = T0 + i as i64 * STEP;
            // Equal returns across symbols give tied cross-sectional ranks.
            let mut ret = if i % 5 == 0 { 0.002 } else { 0.0005 * ((k % 3) as f64 - 1.0) };
            if k % 2 == 0 && k > 1 {
                ret += ((mix(k as u64, i as u64) % 2001) as f64 - 1000.0) / 1.0e6;
            }
            let open = close;
            let mut next = close * (1.0 + ret);
            let mut high = open.max(next) + 0.1;
            let mut low = open.min(next) - 0.1;
            if i % 1013 == 3 {
                next = open;
                high = open;
                low = open;
            }
            let mut v = 10.0 + (i % 97) as f64;
            let mut trades = 10.0 + (i % 53) as f64;
            let mut tb = v * (0.3 + 0.4 * ((i % 7) as f64) / 7.0);
            if i % 701 == 0 && k % 2 == 1 {
                v = 0.0;
                trades = 0.0;
                tb = 0.0;
            }
            let qv = v * next;
            let tq = tb * next;
            let member = if k == 7 && i % 11 == 0 { t + STEP } else { t };
            let nan_close = k == 9 && i % 1500 == 7;
            rows.push(Row {
                t,
                sym: k,
                o: open,
                h: high,
                l: low,
                c: if nan_close { f64::NAN } else { next },
                v,
                qv,
                trades,
                tb,
                tq,
                member,
            });
            if !nan_close {
                close = next;
            }
        }
    }
    rows
}

fn write_fixture(path: &str, rows: &[Row], group_rows: usize) {
    let mut text = String::with_capacity(rows.len() * 120);
    text.push_str(
        "ts,symbol,open,high,low,close,volume,quote_volume,trades,taker_buy_base,taker_buy_quote,member_asof\n",
    );
    for r in rows {
        text.push_str(&format!(
            "{},{},{},{},{},{},{},{},{},{},{},{}\n",
            r.t,
            sym_name(r.sym),
            r.o,
            r.h,
            r.l,
            r.c,
            r.v,
            r.qv,
            r.trades,
            r.tb,
            r.tq,
            r.member
        ));
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
}

fn work_dir(tag: &str) -> std::path::PathBuf {
    let base = std::env::var("RCSR_PARITY_TMP")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let dir = base.join(format!("rcsr-parity-{}-{}-{}", tag, std::process::id(), mix(1, 2) % 1000));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn p(dir: &std::path::Path, name: &str) -> String {
    dir.join(name).to_string_lossy().into_owned()
}

fn options(data: &str, out: &str, names: &[String], gpu: GpuMode, chunk_rows: usize) -> Options {
    Options {
        data: data.to_string(),
        out: Some(out.to_string()),
        features: vec![names.join(",")],
        labels: Vec::new(),
        allow_lockbox: true,
        gpu,
        chunk_rows,
        ..Default::default()
    }
}

fn read_column(path: &str, name: &str) -> Vec<f64> {
    let s = FileStore::open(path).unwrap();
    let c = s.header().column_index(name).unwrap_or_else(|| panic!("column '{name}' missing in {path}"));
    let mut out = Vec::with_capacity(s.n_rows());
    for g in 0..s.n_groups() {
        out.extend(s.read_f64_group(g, c).unwrap());
    }
    out
}

fn hashes(r: &Report) -> Vec<(String, String)> {
    r.features.iter().map(|f| (f.name.clone(), f.hash.clone())).collect()
}

fn diff_hashes(a: &[(String, String)], b: &[(String, String)]) -> Vec<String> {
    let mut bad = Vec::new();
    if a.len() != b.len() {
        bad.push(format!("column count {} vs {}", a.len(), b.len()));
    }
    for ((n1, h1), (n2, h2)) in a.iter().zip(b.iter()) {
        if n1 != n2 || h1 != h2 {
            bad.push(n1.clone());
        }
    }
    bad
}

fn bug_names() -> Vec<&'static str> {
    KNOWN_BUGS.iter().map(|(n, _)| *n).collect()
}

fn gate_names(cols: &[DeviceColumn]) -> Vec<String> {
    cols.iter().map(|c| c.name.clone()).collect()
}

// ---------------------------------------------------------------------------
// Shared first runs (CPU reference and cold GPU)
// ---------------------------------------------------------------------------

struct Shared {
    dir: std::path::PathBuf,
    rows: Vec<Row>,
    src: String,
    src_small_groups: String,
    names: Vec<String>,
    cpu_out: String,
    cpu: Report,
    gpu_out: String,
    gpu_cold: Report,
}

const SYMBOLS: usize = 24;
const PER_SYMBOL: usize = 21_000;

/// `PER_SYMBOL` is overridable for quick iteration (`RCSR_PARITY_PER_SYMBOL`).
fn per_symbol() -> usize {
    std::env::var("RCSR_PARITY_PER_SYMBOL").ok().and_then(|v| v.parse().ok()).unwrap_or(PER_SYMBOL)
}

fn shared() -> &'static Shared {
    static S: OnceLock<Shared> = OnceLock::new();
    S.get_or_init(|| {
        enrich::set_gpu_floor_for_tests(0);
        let cols = device_columns();
        let names = gate_names(&cols);
        let dir = work_dir("fixture");
        let rows = fixture(SYMBOLS, per_symbol());
        let src = p(&dir, "in.rcsb");
        let src_small_groups = p(&dir, "in-small.rcsb");
        write_fixture(&src, &rows, 65_536);
        write_fixture(&src_small_groups, &rows, 8_191);
        let cpu_out = p(&dir, "cpu.rcsb");
        let gpu_out = p(&dir, "gpu-cold.rcsb");
        let t = std::time::Instant::now();
        let cpu = enrich::build(&options(&src, &cpu_out, &names, GpuMode::Off, 65_536)).unwrap();
        eprintln!("[parity] CPU arm: {} columns, {} rows, {:.1}s", cpu.features.len(), cpu.rows, t.elapsed().as_secs_f64());
        let t = std::time::Instant::now();
        let gpu_cold = enrich::build(&options(&src, &gpu_out, &names, GpuMode::On, 65_536)).unwrap();
        eprintln!(
            "[parity] GPU cold arm: {:.1}s, backend {} requested {} available {} gpu_columns {} cpu_columns {} fallback {} kernel {:.2}s",
            t.elapsed().as_secs_f64(), gpu_cold.backend, gpu_cold.gpu_requested, gpu_cold.gpu_available,
            gpu_cold.gpu_columns, gpu_cold.cpu_columns, gpu_cold.gpu_fallback, gpu_cold.gpu_kernel_seconds
        );
        for pl in &gpu_cold.plans {
            eprintln!("[parity]   plan {} rows {} cols {} gpu {} cpu {} backend {} reason {}", pl.group, pl.rows, pl.cols, pl.gpu_cols, pl.cpu_cols, pl.backend, pl.reason);
        }
        Shared { dir, rows, src, src_small_groups, names, cpu_out, cpu, gpu_out, gpu_cold }
    })
}

// ---------------------------------------------------------------------------
// Self-tests of the harness (no device needed)
// ---------------------------------------------------------------------------

#[test]
fn every_device_column_declares_exactly_one_tier() {
    let cols = device_columns();
    assert!(cols.len() >= 150, "the registry must yield the device set, got {}", cols.len());
    let table: Vec<(String, String)> =
        cols.iter().map(|c| (c.name.clone(), c.tier.to_string())).collect();
    let bad = tier_violations(&table);
    assert!(bad.is_empty(), "{} device columns without a valid tier:\n{}", bad.len(), bad.join("\n"));
}

#[test]
fn an_undeclared_or_duplicate_column_fails_the_tier_table() {
    let ok = vec![("a".to_string(), "bitwise".to_string()), ("b".to_string(), "1e-12".to_string())];
    assert!(tier_violations(&ok).is_empty());
    let none = vec![("a".to_string(), "reference".to_string())];
    assert_eq!(tier_violations(&none).len(), 1);
    let empty = vec![("a".to_string(), String::new())];
    assert_eq!(tier_violations(&empty).len(), 1);
    let unavailable = vec![("a".to_string(), "unavailable".to_string())];
    assert_eq!(tier_violations(&unavailable).len(), 1);
    let dup = vec![("a".to_string(), "bitwise".to_string()), ("a".to_string(), "bitwise".to_string())];
    assert_eq!(tier_violations(&dup).len(), 1);
}

#[test]
fn the_verdict_enforces_each_tier() {
    let a = vec![1.0, 2.0, f64::NAN, 4.0];
    let same = a.clone();
    let (o, c, g) = observe(&a, &same);
    assert!(verdict("bitwise", &o, &c, &g).is_ok());
    assert!(verdict("1e-12", &o, &c, &g).is_ok());

    // One ulp: fails bitwise, passes 1e-12.
    let mut ulp = a.clone();
    ulp[1] = f64::from_bits(ulp[1].to_bits() + 1);
    let (o, c, g) = observe(&a, &ulp);
    assert!(verdict("bitwise", &o, &c, &g).is_err(), "a declared bitwise column that differs must fail");
    assert!(verdict("1e-12", &o, &c, &g).is_ok());

    // 1e-9: fails both.
    let mut far = a.clone();
    far[3] += 4.0e-9;
    let (o, c, g) = observe(&a, &far);
    assert!(verdict("1e-12", &o, &c, &g).is_err());

    // One NaN mask mismatch fails every tier.
    let mut mask = a.clone();
    mask[2] = 0.0;
    let (o, c, g) = observe(&a, &mask);
    assert!(verdict("1e-12", &o, &c, &g).is_err());
    assert!(verdict("bitwise", &o, &c, &g).is_err());

    // NaN vs Inf is a mask mismatch.
    let mut inf = a.clone();
    inf[2] = f64::INFINITY;
    let (o, c, g) = observe(&a, &inf);
    assert_eq!(o.mask_mismatch, 1);
    assert!(verdict("1e-12", &o, &c, &g).is_err());

    // Different NaN payloads: the bytes differ, so bitwise fails (strict).
    let mut payload = a.clone();
    payload[2] = f64::from_bits(f64::NAN.to_bits() ^ 1);
    assert!(payload[2].is_nan());
    let (o, c, g) = observe(&a, &payload);
    assert_eq!(o.mask_mismatch, 0);
    assert!(verdict("1e-12", &o, &c, &g).is_ok());
}

#[test]
fn the_fixture_contains_every_required_edge_case() {
    let rows = fixture(SYMBOLS, PER_SYMBOL);
    assert!(rows.len() >= 470_000, "{} rows", rows.len());
    assert!(rows.iter().any(|r| r.c.is_nan()), "NaN close");
    assert!(rows.iter().any(|r| r.v == 0.0), "zero volume");
    assert!(rows.iter().any(|r| r.h == r.l), "H == L");
    assert!(rows.iter().any(|r| r.member > r.t), "PIT non-member row");
    // Gap, a stale 1d bucket (>1 day hole) and fewer than 20 early members.
    let sym5: Vec<i64> = rows.iter().filter(|r| r.sym == 5).map(|r| r.t).collect();
    assert!(sym5.windows(2).any(|w| w[1] - w[0] > 86_400), "stale 1d bucket");
    let early = rows.iter().filter(|r| r.t == T0 + 100 * STEP).count();
    assert!(early < 20, "{early} members at the early timestamp");
    let late = rows.iter().filter(|r| r.t == T0 + 4_000 * STEP).count();
    assert!(late >= 20, "{late} members at the late timestamp");
    // Tied returns: two symbols share the identical return path.
    let r1 = |k: usize| -> Vec<u64> {
        rows.iter().filter(|r| r.sym == k).take(50).map(|r| ((r.c / r.o - 1.0) * 1.0e9).round() as u64).collect()
    };
    assert_eq!(r1(3), r1(9), "symbols with equal k % 3 and odd k share their return path");
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn device_columns_match_the_cpu_at_their_declared_tier() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let cols = device_columns();
    let (cpu, gpu) = (&s.cpu, &s.gpu_cold);

    assert_eq!(gpu.backend, "gpu");
    assert_eq!(gpu.gpu_fallback, 0, "fallbacks: {:?}", gpu.plans);
    assert_eq!(gpu.cpu_columns, 0, "columns left on the CPU under --gpu on: {:?}", gpu.plans);
    assert_eq!(gpu.gpu_columns, gpu.features.len());
    assert!(gpu.gpu_kernel_seconds > 0.0 && gpu.gpu_transfer_bytes > 0);
    for n in &s.names {
        assert!(cpu.features.iter().any(|f| &f.name == n), "device column '{n}' was not materialised");
    }
    assert_eq!(cpu.features.len(), gpu.features.len());

    struct Fam {
        cols: usize,
        bit_identical: usize,
        mask: usize,
        max_abs: f64,
        max_rel: f64,
        tiers: Vec<&'static str>,
    }
    let mut fams: BTreeMap<&'static str, Fam> = BTreeMap::new();
    let mut failures = Vec::new();
    let mut all_finite = 0usize;
    for c in cols.iter().filter(|c| s.names.contains(&c.name)) {
        let a = read_column(&s.cpu_out, &c.name);
        let b = read_column(&s.gpu_out, &c.name);
        let (o, cs, gs) = observe(&a, &b);
        let rep_c = &cpu.features.iter().find(|f| f.name == c.name).unwrap().hash;
        let rep_g = &gpu.features.iter().find(|f| f.name == c.name).unwrap().hash;
        assert_eq!(&cs, rep_c, "{}: report hash is the SHA-256 of the stored column", c.name);
        assert_eq!(&gs, rep_g, "{}: report hash is the SHA-256 of the stored column", c.name);
        all_finite += o.finite;
        if std::env::var("RCSR_PARITY_VERBOSE").is_ok() {
            eprintln!("[col] {} {} {} identical={} max_abs={:.3e} max_rel={:.3e} mask={} first={:?}", c.family, c.name, c.tier, cs == gs, o.max_abs, o.max_rel, o.mask_mismatch, o.first_bad);
        }
        if let Err(e) = verdict(c.tier, &o, &cs, &gs) {
            if bug_names().contains(&c.name.as_str()) {
                eprintln!("[parity] known bug {}: {}", c.name, e);
            } else {
                failures.push(format!("{} [{}] {}: {}", c.name, c.family, c.tier, e));
            }
        } else if bug_names().contains(&c.name.as_str()) {
            failures.push(format!("{}: listed in KNOWN_BUGS but now within its tier", c.name));
        }
        let f = fams.entry(c.family).or_insert(Fam {
            cols: 0,
            bit_identical: 0,
            mask: 0,
            max_abs: 0.0,
            max_rel: 0.0,
            tiers: Vec::new(),
        });
        f.cols += 1;
        f.bit_identical += (cs == gs) as usize;
        f.mask += o.mask_mismatch;
        f.max_abs = f.max_abs.max(o.max_abs);
        f.max_rel = f.max_rel.max(o.max_rel);
        if !f.tiers.contains(&c.tier) {
            f.tiers.push(c.tier);
        }
        if o.finite == 0 {
            failures.push(format!("{}: no finite value on the fixture", c.name));
        }
    }
    eprintln!("[parity] rows={} columns={} finite_cells={}", cpu.rows, s.names.len(), all_finite);
    eprintln!("[parity] cpu feature_s={:.2} gpu feature_s={:.2} gpu kernel_s={:.2} d2h_s={:.2}",
        cpu.feature_seconds, gpu.feature_seconds, gpu.gpu_kernel_seconds, gpu.gpu_d2h_seconds);
    eprintln!("| family | columns | tiers | bit-identical | max abs | max rel | mask mismatches |");
    eprintln!("|---|---:|---|---:|---:|---:|---:|");
    for (name, f) in &fams {
        eprintln!(
            "| {} | {} | {} | {} | {:.3e} | {:.3e} | {} |",
            name, f.cols, f.tiers.join("+"), f.bit_identical, f.max_abs, f.max_rel, f.mask
        );
    }
    assert!(failures.is_empty(), "{} columns outside their declared tier:\n{}", failures.len(), failures.join("\n"));
}

fn assert_known_bug_column(name: &str) {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let c = device_columns().into_iter().find(|c| c.name == name).expect("registered column");
    let (o, cs, gs) = observe(&read_column(&s.cpu_out, name), &read_column(&s.gpu_out, name));
    verdict(c.tier, &o, &cs, &gs).unwrap_or_else(|e| panic!("{name} [{}] {}: {e}", c.family, c.tier));
}

#[test]
fn known_bugs_are_documented_and_registered() {
    let cols = device_columns();
    for (name, why) in KNOWN_BUGS {
        assert!(!why.is_empty(), "{name} needs its measured values");
        assert!(cols.iter().any(|c| c.name == *name), "{name} is not a device column");
    }
}

#[test]
#[ignore = "bug #355: amihud96 GPU differs from CPU beyond 1e-12 (rel 9.929e-1, first row 42074 CPU 0.171 GPU 85.46) and is not prefix-causal"]
fn bug_amihud96() {
    assert_known_bug_column("amihud96");
}

#[test]
#[ignore = "bug #355: xs_rank_amihud96 ranks differ (13 rows, max abs 6.087e-1); follows amihud96"]
fn bug_xs_rank_amihud96() {
    assert_known_bug_column("xs_rank_amihud96");
}

#[test]
fn ma96_matches_cpu_nan_mask_and_bits() {
    assert_known_bug_column("ma96");
}

#[test]
fn ma672_matches_cpu_nan_mask_and_bits() {
    assert_known_bug_column("ma672");
}

#[test]
#[ignore = "bug #357: ou_hl96 differs by 1.3e-7 relative; exceeds the 1e-12 tier (#298 accepted 1e-6)"]
fn bug_ou_hl96() {
    assert_known_bug_column("ou_hl96");
}

#[test]
fn cold_and_warm_gpu_runs_write_identical_bytes() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let warm_out = p(&s.dir, "gpu-warm.rcsb");
    let warm = enrich::build(&options(&s.src, &warm_out, &s.names, GpuMode::On, 65_536)).unwrap();
    let bad = diff_hashes(&hashes(&s.gpu_cold), &hashes(&warm));
    assert!(bad.is_empty(), "cold vs warm GPU columns differ: {bad:?}");
    assert_eq!(s.gpu_cold.output_sha256, warm.output_sha256, "cold vs warm store SHA-256");
    assert_eq!(s.gpu_cold.output_bytes, warm.output_bytes);
    // The CPU path is just as repeatable: its stored columns hash to the
    // reported digests used as the reference above.
    eprintln!("[parity] cold/warm store sha256 {} bytes {}", warm.output_sha256, warm.output_bytes);
    std::fs::remove_file(&warm_out).ok();
}

#[test]
fn chunk_rows_and_store_grouping_do_not_change_a_byte() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let base = hashes(&s.gpu_cold);
    let cases: [(&str, &str, usize); 4] = [
        ("chunk 1", &s.src, 1),
        ("chunk 4096", &s.src, 4_096),
        ("chunk 65536, 8191-row store groups", &s.src_small_groups, 65_536),
        ("chunk 4096, 8191-row store groups", &s.src_small_groups, 4_096),
    ];
    for (i, (label, src, chunk)) in cases.iter().enumerate() {
        let out = p(&s.dir, &format!("inv-{i}.rcsb"));
        let t = std::time::Instant::now();
        let r = enrich::build(&options(src, &out, &s.names, GpuMode::On, *chunk)).unwrap();
        let bad = diff_hashes(&base, &hashes(&r));
        eprintln!("[parity] {label}: {:.1}s, {} differing columns", t.elapsed().as_secs_f64(), bad.len());
        assert!(bad.is_empty(), "{label} changed {} columns: {:?}", bad.len(), bad);
        assert_eq!(r.gpu_fallback, 0, "{label}");
        std::fs::remove_file(&out).ok();
    }
}

fn mtf_device_ok(rows: &[Row]) -> bool {
    let times: Vec<i64> = rows.iter().map(|r| r.t).collect();
    let col = |f: fn(&Row) -> f64| -> Vec<f64> { rows.iter().map(f).collect() };
    let (mut start, mut len) = (Vec::new(), Vec::new());
    for (i, r) in rows.iter().enumerate() {
        if i == 0 || rows[i - 1].sym != r.sym {
            start.push(i as u32);
            len.push(0u32);
        }
        *len.last_mut().unwrap() += 1;
    }
    let names = enrich::mtf_names();
    rcsr::mtf_gpu::compute(
        &times, &col(|r| r.o), &col(|r| r.h), &col(|r| r.l), &col(|r| r.c), &start, &len, STEP, &names,
    )
    .map_or(false, |g| g.0.len() == names.len())
}

#[test]
fn mtf_device_context_accepts_a_long_holey_symbol() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let rows = fixture(SYMBOLS, PER_SYMBOL);
    let sym0: Vec<Row> = rows.iter().copied().filter(|r| r.sym == 0).collect();
    assert!(mtf_device_ok(&sym0), "the device MTF context failed on {} rows of one symbol", sym0.len());
}

/// The MTF family on the device at a size its context accepts: all 36 columns,
/// CPU vs device at the declared tier, with chunk invariance.
#[test]
fn mtf_family_matches_the_cpu_on_the_device() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    enrich::set_gpu_floor_for_tests(0);
    let rows = fixture(SYMBOLS, 6_000);
    assert!(mtf_device_ok(&rows), "device MTF context must accept the 6,000-bar fixture");
    let dir = work_dir("mtf");
    let src = p(&dir, "in.rcsb");
    write_fixture(&src, &rows, 65_536);
    let names = enrich::mtf_names();
    let cpu = enrich::build(&options(&src, &p(&dir, "cpu.rcsb"), &names, GpuMode::Off, 65_536)).unwrap();
    let gpu = enrich::build(&options(&src, &p(&dir, "gpu.rcsb"), &names, GpuMode::On, 65_536)).unwrap();
    assert_eq!(gpu.gpu_columns, names.len());
    assert_eq!(gpu.gpu_fallback, 0);
    let tiers: BTreeMap<String, &'static str> = device_columns().into_iter().map(|c| (c.name, c.tier)).collect();
    let mut identical = 0usize;
    for n in &names {
        let (o, cs, gs) = observe(&read_column(&p(&dir, "cpu.rcsb"), n), &read_column(&p(&dir, "gpu.rcsb"), n));
        identical += (cs == gs) as usize;
        verdict(tiers[n], &o, &cs, &gs).unwrap_or_else(|e| panic!("{n} [{}]: {e}", tiers[n]));
        assert!(o.finite > 0, "{n}: no finite value");
    }
    for chunk in [1usize, 4_096] {
        let r = enrich::build(&options(&src, &p(&dir, "inv.rcsb"), &names, GpuMode::On, chunk)).unwrap();
        assert!(diff_hashes(&hashes(&gpu), &hashes(&r)).is_empty(), "chunk {chunk} changed an MTF column");
    }
    eprintln!("[parity] mtf: {} columns on the device, {identical} bit-identical, cpu {:.1}s gpu {:.1}s", names.len(), cpu.feature_seconds, gpu.feature_seconds);
}

// ---------------------------------------------------------------------------
// Prefix replay and the future-shift canary
// ---------------------------------------------------------------------------

/// `Err(first differing column/row)` unless every kept row of `prefix` equals
/// the same row of `full` bit for bit.
fn prefix_equal(full: &[f64], kept: &[usize], prefix: &[f64]) -> Result<(), usize> {
    assert_eq!(kept.len(), prefix.len());
    for (j, &i) in kept.iter().enumerate() {
        if full[i].to_bits() != prefix[j].to_bits() {
            return Err(i);
        }
    }
    Ok(())
}

/// A look-ahead contaminated copy: every value comes from the next row of the
/// same symbol (the last row of a symbol becomes NaN).
fn future_shifted(col: &[f64], rows: &[Row]) -> Vec<f64> {
    let mut out = col.to_vec();
    for i in 0..col.len() {
        let same = i + 1 < col.len() && rows[i + 1].sym == rows[i].sym;
        out[i] = if same { col[i + 1] } else { f64::NAN };
    }
    out
}

#[test]
fn prefix_replay_at_twenty_cuts_is_bit_identical_and_the_canary_fails() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let s = shared();
    let span = per_symbol() as i64 * STEP;
    let full: Vec<(String, Vec<f64>)> =
        s.names.iter().map(|n| (n.clone(), read_column(&s.gpu_out, n))).collect();
    let mut canary_hits = 0usize;
    let mut canary_total = 0usize;
    let mut failures: BTreeMap<String, String> = BTreeMap::new();
    let mut known_prefix: BTreeMap<String, String> = BTreeMap::new();
    for k in 0..20i64 {
        let cut = T0 + span * (k + 1) / 21;
        let kept: Vec<usize> = (0..s.rows.len()).filter(|&i| s.rows[i].t <= cut).collect();
        let sub: Vec<Row> = kept.iter().map(|&i| s.rows[i]).collect();
        let src = p(&s.dir, "prefix-in.rcsb");
        let out = p(&s.dir, "prefix-out.rcsb");
        write_fixture(&src, &sub, 65_536);
        let r = enrich::build(&options(&src, &out, &s.names, GpuMode::On, 65_536)).unwrap();
        assert_eq!(r.gpu_fallback, 0, "cut {k}");
        for (name, col) in &full {
            let pre = read_column(&out, name);
            if let Err(row) = prefix_equal(col, &kept, &pre) {
                let msg = format!(
                    "cut {k} (t<={cut}): column '{name}' row {row} changed when later rows were removed: {:?} vs {:?}",
                    col[row],
                    pre[kept.iter().position(|&x| x == row).unwrap()]
                );
                if bug_names().contains(&name.as_str()) {
                    known_prefix.entry(name.clone()).or_insert(msg);
                } else {
                    failures.entry(name.clone()).or_insert(msg);
                }
                continue;
            }
            // Canary on the middle cut: a column that reads the future must be
            // flagged against the replay (constant prefixes cannot be flagged).
            if k == 10 {
                let distinct = pre.windows(2).any(|w| w[0].to_bits() != w[1].to_bits());
                if distinct {
                    canary_total += 1;
                    let bad = future_shifted(col, &s.rows);
                    if prefix_equal(&bad, &kept, &pre).is_err() {
                        canary_hits += 1;
                    }
                }
            }
        }
        std::fs::remove_file(&src).ok();
        std::fs::remove_file(&out).ok();
    }
    for (n, m) in &known_prefix {
        eprintln!("[parity] known bug (prefix replay) {n}: {m}");
    }
    assert!(failures.is_empty(), "{} columns are not prefix-causal:\n{}", failures.len(), failures.values().cloned().collect::<Vec<_>>().join("\n"));
    eprintln!("[parity] prefix replay: 20 cuts x {} columns bit-identical; canary flagged {canary_hits}/{canary_total}", full.len());
    assert!(canary_total >= full.len() / 2, "canary had too few non-constant columns: {canary_total}");
    assert_eq!(canary_hits, canary_total, "a future-shifted column passed the replay check");
}

// ---------------------------------------------------------------------------
// Panel prep (#301) and PIT as-of joins (#302)
// ---------------------------------------------------------------------------

fn prep_bars(rows: &[Row]) -> (Vec<PrepBar>, Vec<PrepSegment>) {
    let mut bars = Vec::new();
    let mut segs: Vec<PrepSegment> = Vec::new();
    let mut cur: Option<usize> = None;
    for (i, r) in rows.iter().enumerate() {
        if cur != Some(r.sym) {
            segs.push(PrepSegment { start: bars.len() as u32, len: 0 });
            cur = Some(r.sym);
        }
        let b = PrepBar::new(r.t, r.o, r.h, r.l, if r.c.is_nan() { r.o } else { r.c })
            .with_flow(r.v, r.v * 2.0, r.v / 10.0);
        bars.push(b);
        if i % 997 == 13 {
            bars.push(b);
        }
        if i % 1499 == 29 {
            let mut c = b;
            c.close += 0.5;
            bars.push(c);
        }
        segs.last_mut().unwrap().len = bars.len() as u32 - segs.last().unwrap().start;
    }
    (bars, segs)
}

fn prep_bytes(o: &PrepOutput) -> Vec<u8> {
    let mut v = Vec::with_capacity(o.buckets.len() * 120);
    for b in &o.buckets {
        v.extend_from_slice(&b.time.to_le_bytes());
        v.extend_from_slice(&b.segment.to_le_bytes());
        v.extend_from_slice(&b.chain.to_le_bytes());
        for x in [b.open, b.high, b.low, b.close, b.volume, b.quote_volume, b.trades, b.taker_buy_base, b.taker_buy_quote] {
            v.extend_from_slice(&x.to_bits().to_le_bytes());
        }
        v.extend_from_slice(&b.cov15m.to_le_bytes());
        v.extend_from_slice(&b.dup_rows.to_le_bytes());
        v.extend_from_slice(&b.gap.to_le_bytes());
        v.push(
            b.complete as u8
                | (b.traded as u8) << 1
                | (b.partial as u8) << 2
                | (b.duplicate_conflict as u8) << 3
                | (b.outage as u8) << 4,
        );
    }
    for c in &o.conflicts {
        v.extend_from_slice(&c.time.to_le_bytes());
        v.extend_from_slice(&c.segment.to_le_bytes());
        v.extend_from_slice(&c.chain.to_le_bytes());
        v.extend_from_slice(&c.dup_rows.to_le_bytes());
        v.extend_from_slice(&c.first_row.to_le_bytes());
        v.extend_from_slice(&c.len.to_le_bytes());
    }
    v
}

#[test]
fn panel_prep_is_byte_identical_cpu_gpu_cold_warm_and_across_block_sizes() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    let rows = fixture(SYMBOLS, per_symbol());
    let (bars, segments) = prep_bars(&rows);
    for keep_partial in [false, true] {
        let cpu = cpu_prepare(&bars, &segments, STEP, 3_600, keep_partial).unwrap();
        let want = sha256_hex(&prep_bytes(&cpu));
        assert!(!cpu.conflicts.is_empty() && cpu.buckets.len() > 100_000);

        let mut gpu = PrepGpu::new(&bars, &segments).expect("context");
        assert!(gpu.run(STEP, 3_600, keep_partial));
        let cold = sha256_hex(&prep_bytes(&gpu.output().expect("cold")));
        assert!(gpu.run(STEP, 3_600, keep_partial));
        let warm = sha256_hex(&prep_bytes(&gpu.output().expect("warm")));
        assert_eq!(cold, want, "cold device vs CPU (keep_partial={keep_partial})");
        assert_eq!(warm, want, "warm device vs CPU (keep_partial={keep_partial})");
        for threads in [32usize, 64, 256, 1024] {
            let mut g = PrepGpu::new(&bars, &segments).expect("context");
            assert!(g.run_with_threads(STEP, 3_600, keep_partial, threads));
            let got = sha256_hex(&prep_bytes(&g.output().expect("output")));
            assert_eq!(got, want, "threads={threads} keep_partial={keep_partial}");
        }
        eprintln!(
            "[parity] panel_prep rows={} buckets={} conflicts={} keep_partial={} tier=bitwise sha256={}",
            bars.len(), cpu.buckets.len(), cpu.conflicts.len(), keep_partial, want
        );
    }

    // Prefix replay at 20 hour-aligned cuts; the canary is the full bucket
    // closes shifted one bucket into the future.
    let full = panel_prep_gpu::prepare(&bars, &segments, STEP, 3_600, true).unwrap().0;
    let lo = bars.iter().map(|b| b.time).min().unwrap();
    let hi = bars.iter().map(|b| b.time).max().unwrap();
    for k in 0..20i64 {
        let cut = (lo + (hi - lo) * (k + 1) / 21) / 3_600 * 3_600;
        let mut pre_bars = Vec::new();
        let mut lens = Vec::new();
        for s in &segments {
            let before = pre_bars.len();
            pre_bars.extend(
                bars[s.start as usize..(s.start + s.len) as usize].iter().copied().filter(|b| b.time < cut),
            );
            lens.push((pre_bars.len() - before) as u32);
        }
        let mut start = 0u32;
        let pre_segments: Vec<PrepSegment> = lens
            .iter()
            .map(|len| {
                let sg = PrepSegment { start, len: *len };
                start += *len;
                sg
            })
            .collect();
        let got = panel_prep_gpu::prepare(&pre_bars, &pre_segments, STEP, 3_600, true).unwrap().0;
        let want: Vec<_> = full.buckets.iter().copied().filter(|b| b.time < cut).collect();
        assert_eq!(want.len(), got.buckets.len(), "cut {k}");
        for (a, b) in want.iter().zip(got.buckets.iter()) {
            assert_eq!(a, b, "bucket at {} changed under cut {k}", a.time);
        }
        if k == 10 {
            let mut shifted = want.clone();
            for i in 0..shifted.len().saturating_sub(1) {
                shifted[i].close = want[i + 1].close;
            }
            assert!(
                shifted.iter().zip(got.buckets.iter()).any(|(a, b)| a != b),
                "a future-shifted close must differ from the replay"
            );
        }
    }
}

fn pit_cases() -> Vec<(&'static str, PitBlock, PitParams)> {
    vec![
        ("membership", pit::membership_block(31, 7, 5000), PitParams::membership(STEP)),
        ("mtf1h", pit::mtf_block(32, 7, 5000, 3600), PitParams::mtf(3600, STEP)),
        ("mtf4h", pit::mtf_block(33, 7, 5000, 14_400), PitParams::mtf(14_400, STEP)),
        ("mtf1d", pit::mtf_block(34, 7, 5000, 86_400), PitParams::mtf(86_400, STEP)),
    ]
}

fn pit_gpu(block: &PitBlock, p: PitParams) -> panel_pit::PitOutput {
    panel_pit_gpu::join(block, p).expect("device join").0
}

#[test]
fn pit_asof_joins_are_byte_identical_and_causal() {
    if !have_device() {
        return;
    }
    let _g = HEAVY.lock().unwrap_or_else(|e| e.into_inner());
    for (name, block, p) in pit_cases() {
        let cpu = panel_pit::join(&block, p).unwrap();
        let want = pit::digest(&cpu);
        // Cold, warm, resident re-run, block and chunk sizes.
        let cold = pit_gpu(&block, p);
        pit::assert_same(name, &cpu, &cold);
        assert_eq!(cold.lookahead_violations, 0, "{name}");
        assert_eq!(pit::digest(&pit_gpu(&block, p)), want, "{name}: warm");
        let mut gpu = PitGpu::new(&block).unwrap();
        for threads in [32usize, 128, 1024] {
            for chunk in [1usize, 4096, 1 << 20] {
                assert!(gpu.run_with(p, threads, chunk), "{name} {threads}x{chunk}");
                assert_eq!(pit::digest(&gpu.output().unwrap()), want, "{name}: threads {threads} chunk {chunk}");
            }
        }
        // Prefix replay at 20 cuts.
        let full = pit_gpu(&block, p);
        let lo = *block.times.iter().min().unwrap();
        let hi = *block.times.iter().max().unwrap();
        for k in 0..20 {
            let cut = lo + (hi - lo) * (k + 1) / 21;
            let pre_block = pit::prefix(&block, cut, p.delay);
            if pre_block.times.is_empty() {
                continue;
            }
            let pre = pit_gpu(&pre_block, p);
            pit::assert_prefix_equal(&format!("{name} cut {k}"), &block, &full, &pre_block, &pre);
        }
        // Future-shift canary.
        let mut g = PitGpu::new(&block).unwrap();
        assert!(g.run(p));
        let out = g.output().unwrap();
        let mut shifted = out.leaf.clone();
        for s in &block.segments {
            let le = (s.leaf_start + s.leaf_len) as i32;
            for r in s.start as usize..(s.start + s.len) as usize {
                if shifted[r] >= 0 && shifted[r] + 1 < le {
                    shifted[r] += 1;
                }
            }
        }
        let bad = g.lookahead_violations(&shifted, p.delay).unwrap();
        assert!(bad > 0, "{name}: a future-shifted leaf must be reported");
        assert_eq!(bad, panel_pit::lookahead_violations(&block, &shifted, p.delay), "{name}");
        eprintln!("[parity] pit {name} tier=bitwise digest={want:016x} canary_violations={bad}");
    }
}

// ---------------------------------------------------------------------------
// Large-store harness (ignored): the same comparison on a real panel
// ---------------------------------------------------------------------------

/// Runs one arm of the paired large-store comparison, or compares the arms.
///
/// Environment:
/// * `RCSR_PARITY_STORE`: the input store, default
///   `/mnt/lager/tmp/opencode/prep-measure/panel15m.rcsb` (16,113,108 rows,
///   109 symbols, 15-minute bars; not in git).
/// * `RCSR_PARITY_WORK`: output directory (default `/mnt/lager/tmp/eureqa-gates/parity-large`).
/// * `RCSR_PARITY_ARM`: `cpu`, `gpu` (writes the store, then a second
///   `--dry-run` GPU run whose hashes must equal the first) or `compare`.
///
/// Each arm is a separate invocation so no single lock hold covers both.
#[test]
#[ignore = "large store, see docs/ohlcv.md (RCSR_PARITY_ARM=cpu|gpu|compare); the gpu arm is blocked by a projected-ring hang (#352)"]
fn large_store_parity_at_the_declared_tiers() {
    let store_path = std::env::var("RCSR_PARITY_STORE")
        .unwrap_or_else(|_| "/mnt/lager/tmp/opencode/prep-measure/panel15m.rcsb".to_string());
    if !std::path::Path::new(&store_path).exists() {
        eprintln!("skipping: {store_path} not found");
        return;
    }
    let work = std::path::PathBuf::from(
        std::env::var("RCSR_PARITY_WORK").unwrap_or_else(|_| "/mnt/lager/tmp/eureqa-gates/parity-large".to_string()),
    );
    std::fs::create_dir_all(&work).unwrap();
    let arm = std::env::var("RCSR_PARITY_ARM").expect("RCSR_PARITY_ARM=cpu|gpu|compare");
    let cols = device_columns();
    let table: Vec<(String, String)> = cols.iter().map(|c| (c.name.clone(), c.tier.to_string())).collect();
    let bad = tier_violations(&table);
    assert!(bad.is_empty(), "undeclared tiers: {bad:?}");
    let names = gate_names(&cols);
    let tsv = |arm: &str| p(&work, &format!("{arm}.hashes.tsv"));
    let save = |arm: &str, r: &Report| {
        let mut text = format!("# wall_s={:.1} feature_s={:.1} kernel_s={:.2} d2h_s={:.2} transfer_bytes={} rows={} gpu_columns={} fallback={} peak_rss={}\n",
            r.total_seconds, r.feature_seconds, r.gpu_kernel_seconds, r.gpu_d2h_seconds, r.gpu_transfer_bytes, r.rows, r.gpu_columns, r.gpu_fallback, r.peak_rss_bytes);
        for f in &r.features {
            text.push_str(&format!("{}\t{}\n", f.name, f.hash));
        }
        std::fs::write(tsv(arm), text).unwrap();
    };
    match arm.as_str() {
        "cpu" => {
            let out = p(&work, "cpu.rcsb");
            std::fs::remove_file(&out).ok();
            let r = enrich::build(&options(&store_path, &out, &names, GpuMode::Off, store::DEFAULT_GROUP_ROWS)).unwrap();
            eprintln!("[large] cpu: {} columns, wall {:.1}s, feature {:.1}s", r.features.len(), r.total_seconds, r.feature_seconds);
            save("cpu", &r);
        }
        "gpu" => {
            if !have_device() {
                return;
            }
            let out = p(&work, "gpu.rcsb");
            std::fs::remove_file(&out).ok();
            let r = enrich::build(&options(&store_path, &out, &names, GpuMode::On, store::DEFAULT_GROUP_ROWS)).unwrap();
            eprintln!(
                "[large] gpu: {} columns ({} on device, {} fallback), wall {:.1}s, feature {:.1}s, kernel {:.2}s, d2h {:.2}s",
                r.features.len(), r.gpu_columns, r.gpu_fallback, r.total_seconds, r.feature_seconds, r.gpu_kernel_seconds, r.gpu_d2h_seconds
            );
            assert_eq!(r.gpu_fallback, 0, "{:?}", r.plans);
            assert_eq!(r.cpu_columns, 0, "{:?}", r.plans);
            save("gpu", &r);
            let mut again = options(&store_path, &p(&work, "gpu2.rcsb"), &names, GpuMode::On, store::DEFAULT_GROUP_ROWS);
            again.dry_run = true;
            let r2 = enrich::build(&again).unwrap();
            let bad = diff_hashes(&hashes(&r), &hashes(&r2));
            eprintln!("[large] gpu repeat (dry-run): wall {:.1}s, differing columns {}", r2.total_seconds, bad.len());
            assert!(bad.is_empty(), "two GPU runs differ: {bad:?}");
        }
        "compare" => {
            let (cpu_path, gpu_path) = (p(&work, "cpu.rcsb"), p(&work, "gpu.rcsb"));
            let cs = FileStore::open(&cpu_path).unwrap();
            let gs = FileStore::open(&gpu_path).unwrap();
            assert_eq!(cs.n_rows(), gs.n_rows());
            eprintln!("| family | column | tier | bit-identical | max abs | max rel | mask mismatches |");
            eprintln!("|---|---|---|---|---:|---:|---:|");
            let mut fam: BTreeMap<&str, (usize, usize, f64, f64, usize, Vec<&str>)> = BTreeMap::new();
            let mut failures = Vec::new();
            for c in cols.iter().filter(|c| names.contains(&c.name)) {
                let ci = cs.header().column_index(&c.name).unwrap_or_else(|| panic!("{} missing in cpu store", c.name));
                let gi = gs.header().column_index(&c.name).unwrap_or_else(|| panic!("{} missing in gpu store", c.name));
                let mut o = Obs::new();
                for g in 0..cs.n_groups() {
                    o.push(&cs.read_f64_group(g, ci).unwrap(), &gs.read_f64_group(g, gi).unwrap());
                }
                let (a, b) = o.shas();
                let ident = a == b;
                if let Err(e) = verdict(c.tier, &o, &a, &b) {
                    if bug_names().contains(&c.name.as_str()) {
                        eprintln!("[large] known bug {}: {}", c.name, e);
                    } else {
                        failures.push(format!("{} [{}] {}: {}", c.name, c.family, c.tier, e));
                    }
                }
                let e = fam.entry(c.family).or_insert((0, 0, 0.0, 0.0, 0, Vec::new()));
                e.0 += 1;
                e.1 += ident as usize;
                e.2 = e.2.max(o.max_abs);
                e.3 = e.3.max(o.max_rel);
                e.4 += o.mask_mismatch;
                if !e.5.contains(&c.tier) {
                    e.5.push(c.tier);
                }
                if std::env::var("RCSR_PARITY_VERBOSE").is_ok() {
                    eprintln!("| {} | {} | {} | {} | {:.3e} | {:.3e} | {} |", c.family, c.name, c.tier, ident, o.max_abs, o.max_rel, o.mask_mismatch);
                }
            }
            eprintln!("| family | columns | tiers | bit-identical | max abs | max rel | mask mismatches |");
            for (name, (n, id, ma, mr, mm, tiers)) in &fam {
                eprintln!("| {name} | {n} | {} | {id} | {ma:.3e} | {mr:.3e} | {mm} |", tiers.join("+"));
            }
            // Report hashes of both arms for the bitwise columns.
            let load = |arm: &str| -> BTreeMap<String, String> {
                std::fs::read_to_string(tsv(arm))
                    .unwrap()
                    .lines()
                    .filter(|l| !l.starts_with('#'))
                    .filter_map(|l| l.split_once('\t').map(|(a, b)| (a.to_string(), b.to_string())))
                    .collect()
            };
            let (hc, hg) = (load("cpu"), load("gpu"));
            for c in cols.iter().filter(|c| c.tier == "bitwise" && names.contains(&c.name)) {
                if hc.get(&c.name) != hg.get(&c.name) {
                    failures.push(format!("{}: report SHA-256 differs between the arms", c.name));
                }
            }
            assert!(failures.is_empty(), "{} columns outside their tier:\n{}", failures.len(), failures.join("\n"));
        }
        other => panic!("unknown RCSR_PARITY_ARM '{other}'"),
    }
}
