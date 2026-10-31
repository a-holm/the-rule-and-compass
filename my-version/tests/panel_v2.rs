//! #259: the default hourly point-in-time panel (v2): snapshot, base,
//! build and validate on small synthetic archives.

use rcsr::manifest;
use rcsr::panel_v2::{
    self, BaseOptions, PanelOptions, ValidateOptions, FLAG_DUPLICATE_CONFLICT, FLAG_PARTIAL_BUCKET,
};
use rcsr::store::{self, FileStore};
use std::path::{Path, PathBuf};

/// 2021-01-01T00:00:00Z.
const T0: i64 = 1_609_459_200;
const H: i64 = 3600;

fn scratch(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("rcsr-panel-v2-{}-{}", std::process::id(), name));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

#[derive(Clone, Copy, PartialEq)]
enum Unit {
    Ms,
    Us,
}

fn quarter(t: i64, unit: Unit, base: f64, k: i64) -> String {
    let p = base * (1.0 + 0.001 * k as f64);
    let (o, c) = (p, p * 1.0005);
    let (h, l) = (c * 1.001, o * 0.999);
    let v = 10.0 + (k % 7) as f64;
    let qv = v * (o + c) / 2.0;
    let raw = match unit {
        Unit::Ms => t * 1000,
        Unit::Us => t * 1_000_000,
    };
    format!("{raw},{o},{h},{l},{c},{v},{raw},{qv},{},{},{},0", 5 + k % 3, v * 0.4, qv * 0.4)
}

/// Full hours `[from, from + n)` (hour indexes since T0) minus `skip`.
fn hours(base: f64, from: i64, n: i64, skip: &[i64], unit: Unit) -> Vec<String> {
    let mut out = Vec::new();
    for hr in from..from + n {
        if skip.contains(&hr) {
            continue;
        }
        for q in 0..4 {
            out.push(quarter(T0 + hr * H + q * 900, unit, base, hr * 4 + q));
        }
    }
    out
}

fn put_zip(root: &Path, symbol: &str, month: &str, lines: &[String]) -> PathBuf {
    let dir = root.join(symbol);
    std::fs::create_dir_all(&dir).unwrap();
    let name = format!("{symbol}-15m-{month}");
    let path = dir.join(format!("{name}.zip"));
    let csv = lines.join("\n") + "\n";
    std::fs::write(&path, stored_zip(&format!("{name}.csv"), csv.as_bytes())).unwrap();
    path
}

fn membership(path: &Path, rows: &[(i64, &str, u8)]) {
    let mut text = String::from("as_of,symbol,member,adv_usd,n_bars\n");
    for (t, s, m) in rows {
        text.push_str(&format!("{},{},{},1000.0,100\n", T0 + t * H, s, m));
    }
    std::fs::write(path, text).unwrap();
}

fn base_opts(input: &Path, out: &Path) -> BaseOptions {
    BaseOptions::new(input.to_str().unwrap(), out.to_str().unwrap())
}

fn panel_opts(base: &Path, m: &Path, out: &Path) -> PanelOptions {
    PanelOptions::new(base.to_str().unwrap(), m.to_str().unwrap(), out.to_str().unwrap())
}

struct Cols {
    names: Vec<String>,
    table: rcsr::data::Table,
}

impl Cols {
    fn open(path: &Path) -> Cols {
        let table = FileStore::open(path.to_str().unwrap()).unwrap().to_table().unwrap();
        Cols { names: table.names.clone(), table }
    }
    fn n(&self) -> usize {
        self.table.len()
    }
    fn f(&self, name: &str, r: usize) -> f64 {
        self.table.value(r, self.names.iter().position(|n| n == name).unwrap())
    }
    fn ts(&self, r: usize) -> i64 {
        self.f("ts", r) as i64
    }
    fn sym(&self, r: usize) -> String {
        let c = self.names.iter().position(|n| n == "symbol").unwrap();
        self.table.text(r, c).unwrap().to_string()
    }
    fn rows_of(&self, symbol: &str) -> Vec<usize> {
        (0..self.n()).filter(|&r| self.sym(r) == symbol).collect()
    }
}

#[test]
fn a_hour_with_a_missing_quarter_is_dropped_unless_keep_partial() {
    let dir = scratch("partial");
    let mut lines = hours(10.0, 0, 3, &[], Unit::Ms);
    // hour 1 loses its :15 bar
    lines.remove(5);
    put_zip(&dir.join("raw"), "AAAUSDT", "2021-01", &lines);

    let out = dir.join("base.rcsb");
    let s = panel_v2::build_base(&base_opts(&dir.join("raw"), &out)).unwrap();
    assert_eq!(s.rows, 2, "default drops the hour with 3 bars");
    let c = Cols::open(&out);
    assert!((0..c.n()).all(|r| c.f("cov15m", r) == 4.0 && c.f("flags", r) == 0.0));
    assert_eq!((c.ts(0), c.ts(1)), (T0, T0 + 2 * H));
    assert_eq!(c.f("gap", 1), 1.0, "the dropped hour is a gap, never bridged");

    let out2 = dir.join("base_keep.rcsb");
    let mut o = base_opts(&dir.join("raw"), &out2);
    o.keep_partial = true;
    panel_v2::build_base(&o).unwrap();
    let k = Cols::open(&out2);
    assert_eq!(k.n(), 3);
    let part = (0..3).find(|&r| k.ts(r) == T0 + H).unwrap();
    assert_eq!(k.f("cov15m", part), 3.0);
    assert_eq!(k.f("flags", part) as u32, FLAG_PARTIAL_BUCKET);

    // The 1h OHLC is the 4-bar aggregate only when cov15m == 4.
    let q: Vec<Vec<f64>> = (0..4)
        .map(|j| {
            quarter(T0 + j * 900, Unit::Ms, 10.0, j)
                .split(',')
                .map(|x| x.parse().unwrap())
                .collect()
        })
        .collect();
    let full = (0..3).find(|&r| k.ts(r) == T0).unwrap();
    assert_eq!(k.f("cov15m", full), 4.0);
    assert_eq!(k.f("open", full), q[0][1]);
    assert_eq!(k.f("close", full), q[3][4]);
    assert_eq!(k.f("high", full), q.iter().map(|x| x[2]).fold(f64::MIN, f64::max));
    assert_eq!(k.f("low", full), q.iter().map(|x| x[3]).fold(f64::MAX, f64::min));
    assert_eq!(k.f("volume", full), q.iter().map(|x| x[5]).sum::<f64>());
    assert_eq!(k.f("trades", full), q.iter().map(|x| x[8]).sum::<f64>());
    assert_eq!(k.f("taker_buy_base", full), q.iter().map(|x| x[9]).sum::<f64>());
}

#[test]
fn identical_duplicates_collapse_and_differing_ones_aggregate_and_flag() {
    let dir = scratch("dups");
    let one = hours(10.0, 0, 48, &[], Unit::Ms);
    let mut four = Vec::new();
    for _ in 0..4 {
        four.extend(one.iter().cloned());
    }
    put_zip(&dir.join("raw"), "JUVUSDT", "2021-01", &four);
    // Two differing copies of one instant.
    let mut conflict = hours(20.0, 0, 3, &[], Unit::Ms);
    let plain: Vec<f64> = conflict[..4].iter().map(|l| l.split(',').nth(5).unwrap().parse().unwrap()).collect();
    let mut twin: Vec<String> = conflict[0].split(',').map(|s| s.to_string()).collect();
    twin[4] = "99.5".to_string();
    twin[2] = "100".to_string();
    twin[5] = "7".to_string();
    conflict.insert(1, twin.join(","));
    put_zip(&dir.join("raw"), "CNFUSDT", "2021-01", &conflict);

    let out = dir.join("base.rcsb");
    panel_v2::build_base(&base_opts(&dir.join("raw"), &out)).unwrap();
    let c = Cols::open(&out);
    let juv = c.rows_of("JUVUSDT");
    assert_eq!(juv.len(), 48);
    let one_copy: f64 = one
        .iter()
        .map(|l| l.split(',').nth(5).unwrap().parse::<f64>().unwrap())
        .sum();
    let total: f64 = juv.iter().map(|&r| c.f("volume", r)).sum();
    assert_eq!(total, one_copy, "volume equals one copy, not four");
    assert!(juv.iter().all(|&r| c.f("dup_rows", r) == 16.0 && c.f("flags", r) == 0.0));

    let cnf = c.rows_of("CNFUSDT");
    let flagged: Vec<usize> = cnf.iter().copied().filter(|&r| c.f("flags", r) as u32 & FLAG_DUPLICATE_CONFLICT != 0).collect();
    assert_eq!(flagged.len(), 1);
    let r = flagged[0];
    assert_eq!(r, cnf[0]);
    assert_eq!(c.f("high", r), 100.0, "aggregate high is the max of the copies");
    assert_eq!(c.f("dup_rows", r), 5.0);
    assert_eq!(c.f("volume", r), plain.iter().sum::<f64>() + 7.0, "differing copies sum their flow once each");

    let m = std::fs::read_to_string(format!("{}.manifest.json", out.display())).unwrap();
    assert!(m.contains("\"duplicate_conflicts\":1"));
    assert!(m.contains("\"symbol\":\"CNFUSDT\",\"ts\":1609459200"), "the conflict timestamp is listed");
    assert!(m.contains("\"duplicate_rows\":577"));
}

#[test]
fn snapshot_hashes_every_zip_and_lists_the_juv_duplicate_and_the_aud_bar() {
    let dir = scratch("snap");
    let raw = dir.join("raw");
    let one = hours(10.0, 0, 6, &[], Unit::Ms);
    let mut four = Vec::new();
    for _ in 0..4 {
        four.extend(one.iter().cloned());
    }
    let juv = put_zip(&raw, "JUVUSDT", "2022-03", &four);
    // AUDUSDT 2020-11-13 04:15: high 0.72033 < close 0.72194.
    let aud_t = 1_605_240_900i64;
    let mut aud = vec![
        format!("{},0.72,0.73,0.71,0.721,5,0,3.6,2,1,0.7,0", (aud_t - 900) * 1000),
        format!("{},0.72,0.72033,0.71,0.72194,5,0,3.6,2,1,0.7,0", aud_t * 1000),
    ];
    aud.push(format!("{},0.72,0.73,0.71,0.721,5,0,3.6,2,1,0.7,0", (aud_t + 900) * 1000));
    let aud_zip = put_zip(&raw, "AUDUSDT", "2020-11", &aud);
    let us = put_zip(&raw, "USUSDT", "2025-01", &hours(5.0, 0, 2, &[], Unit::Us));

    let text = panel_v2::snapshot_text(raw.to_str().unwrap(), 2).unwrap();
    assert_eq!(text, panel_v2::snapshot_text(raw.to_str().unwrap(), 1).unwrap(), "deterministic");
    assert!(text.contains("\"network\":false"));
    assert!(text.contains("\"files\":3,"));
    for p in [&juv, &aud_zip, &us] {
        let sha = manifest::sha256_file(p).unwrap();
        assert!(text.contains(&format!("\"sha256\":\"{sha}\"")), "{} is hashed", p.display());
        let size = std::fs::metadata(p).unwrap().len();
        assert!(text.contains(&format!("\"bytes\":{size},")));
    }
    assert!(text.contains(&format!("\"first_ts\":{},\"last_ts\":{}", T0, T0 + 6 * H - 900)));
    assert!(text.contains("\"path\":\"JUVUSDT/JUVUSDT-15m-2022-03.zip\",\"symbol\":\"JUVUSDT\",\"rows\":96,\"duplicate_rows\":72,\"identical\":true"));
    assert!(text.contains("\"reason\":\"high_below_body\""));
    assert!(text.contains("\"date\":\"2020-11-13 04:15\""));
    assert!(text.contains("\"high\":0.72033,\"low\":0.71,\"close\":0.72194"));
    assert!(text.contains("\"units\":{\"s\":0,\"ms\":0,\"us\":8}"));
}

#[test]
fn the_lockbox_months_and_the_1m_interval_are_guarded() {
    let dir = scratch("lockbox");
    let raw = dir.join("raw");
    let dir1m = raw.join("BTCUSDT");
    std::fs::create_dir_all(&dir1m).unwrap();
    let csv: String = (0..4).map(|q| quarter(1_782_864_000 + q * 900, Unit::Ms, 5.0, q) + "\n").collect();
    std::fs::write(
        dir1m.join("BTCUSDT-1m-2026-07.zip"),
        stored_zip("BTCUSDT-1m-2026-07.csv", csv.as_bytes()),
    )
    .unwrap();
    let out = dir.join("b.rcsb");
    let mut o = base_opts(&raw, &out);
    o.interval_seconds = 60;
    let e = panel_v2::build_base(&o).unwrap_err();
    assert!(e.contains("lockbox") && e.contains("--allow-lockbox"), "{e}");
    assert!(!out.exists());
    o.interval_seconds = 900;
    let e = panel_v2::build_base(&o).unwrap_err();
    assert!(e.contains("lockbox"), "{e}");
    o.allow_lockbox = true;
    assert!(panel_v2::build_base(&o).is_ok(), "an explicit override is the only way in");
}

/// Six symbols with a venue outage (hours 100..103 missing everywhere), one
/// symbol outage, a delisting, a late listing, a relisting with a level jump
/// and a ms to us unit change.
fn market(dir: &Path, relist_gap_hours: i64) -> (PathBuf, PathBuf) {
    let raw = dir.join("raw");
    for (i, s) in ["AAAUSDT", "BBBUSDT", "CCCUSDT", "DDDUSDT", "EEEUSDT"].iter().enumerate() {
        let skip: Vec<i64> = if *s == "BBBUSDT" { vec![100, 101, 102, 200] } else { vec![100, 101, 102] };
        put_zip(&raw, s, "2021-01", &hours(10.0 + i as f64, 0, 360, &skip, Unit::Ms));
    }
    // Delists at hour 500 (data continues; exclusion at 300 removes it from the panel earlier for XXX).
    put_zip(&raw, "DELUSDT", "2021-01", &hours(7.0, 0, 360, &[100, 101, 102], Unit::Ms));
    // Late listing at hour 250.
    put_zip(&raw, "NEWUSDT", "2021-01", &hours(3.0, 250, 110, &[], Unit::Ms));
    // Two listings: the second after a long gap at 20x the price.
    put_zip(&raw, "LUNAUSDT", "2021-01", &hours(50.0, 0, 200, &[100, 101, 102], Unit::Ms));
    put_zip(&raw, "LUNAUSDT", "2021-03", &hours(1000.0, 200 + relist_gap_hours, 100, &[], Unit::Ms));
    let m = dir.join("membership.csv");
    membership(
        &m,
        &[
            (0, "AAAUSDT", 1),
            (0, "BBBUSDT", 1),
            (0, "CCCUSDT", 1),
            (0, "DDDUSDT", 1),
            (0, "EEEUSDT", 1),
            (0, "DELUSDT", 1),
            (0, "LUNAUSDT", 1),
            (0, "NEWUSDT", 0),
            (300, "DELUSDT", 0),
            (250, "NEWUSDT", 1),
        ],
    );
    (raw, m)
}

fn build_market(dir: &Path, relist_gap_hours: i64, relists: &str) -> Result<(PathBuf, PathBuf, PathBuf), String> {
    let (raw, m) = market(dir, relist_gap_hours);
    let base = dir.join("base.rcsb");
    panel_v2::build_base(&base_opts(&raw, &base))?;
    let out = dir.join("panel.rcsb");
    let mut p = panel_opts(&base, &m, &out);
    p.relists = panel_v2::parse_relists(relists)?;
    panel_v2::build_panel(&p)?;
    Ok((base, m, out))
}

#[test]
fn the_panel_has_the_declared_columns_and_only_point_in_time_members() {
    let dir = scratch("pit");
    // 40 days after hour 200: > 30 day gap, so a new listing.
    let (base, m, out) = build_market(&dir, 24 * 40, "").unwrap();
    let c = Cols::open(&out);
    let want = [
        "ts", "symbol", "listing_id", "member_asof", "open", "high", "low", "close", "volume",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "vwap", "gap", "cov15m",
        "flags", "traded", "dup_rows", "archive_unit",
    ];
    assert_eq!(c.names, want);
    // time-major, (ts, symbol) order
    for r in 1..c.n() {
        assert!((c.ts(r - 1), c.sym(r - 1)) < (c.ts(r), c.sym(r)));
    }
    // a listing starts at its first bar, and not before the decision
    let new = c.rows_of("NEWUSDT");
    assert_eq!(c.ts(new[0]), T0 + 250 * H);
    assert_eq!(new.len(), 110);
    // membership exclusion at hour 300 removes DEL; nothing is filled
    let del = c.rows_of("DELUSDT");
    assert_eq!(c.ts(*del.last().unwrap()), T0 + 299 * H);
    // LUNA is two listings and no row bridges the long outage
    let luna = c.rows_of("LUNAUSDT");
    let ids: std::collections::BTreeSet<i64> = luna.iter().map(|&r| c.f("listing_id", r) as i64).collect();
    assert_eq!(ids.len(), 2);
    let a = c.rows_of("AAAUSDT");
    let aid: std::collections::BTreeSet<i64> = a.iter().map(|&r| c.f("listing_id", r) as i64).collect();
    assert_eq!(aid.len(), 1);
    assert!(ids.is_disjoint(&aid));
    // no forward fill: the venue outage hours are absent for every symbol
    for r in 0..c.n() {
        assert!(![100, 101, 102].contains(&((c.ts(r) - T0) / H)));
    }
    // row after the outage carries the gap, never a filled bar
    let after = a.iter().find(|&&r| c.ts(r) == T0 + 103 * H).unwrap();
    assert_eq!(c.f("gap", *after), 3.0);
    assert!(c.f("vwap", *after) > 0.0);

    let manifest = std::fs::read_to_string(format!("{}.manifest.json", out.display())).unwrap();
    assert!(manifest.contains("\"venue_outage_hours\":3"), "{manifest}");
    assert!(manifest.contains("\"max_gap\":960"));
    assert!(manifest.contains("\"derivatives\":{\"available\":false,\"source\":null,\"columns\":[]}"));
    assert!(manifest.contains("\"multi_listing_symbols\":[{\"symbol\":\"LUNAUSDT\",\"listings\":2}]"));
    assert!(manifest.contains("\"lockbox\":") && manifest.contains("\"rows\":0"));
    assert!(manifest.contains("\"rows_not_in_force\":0"));
    for k in ["duplicate_conflicts", "invalid_ohlc", "partial_buckets", "zero_volume", "max_gap"] {
        assert!(manifest.contains(&format!("\"{k}\":")), "{k}");
    }
    assert!(manifest.contains("\"sha256\":"));

    // the full validation report passes, including the prefix replay
    let mut vo = ValidateOptions {
        panel: out.to_str().unwrap().to_string(),
        base: Some(base.to_str().unwrap().to_string()),
        membership: Some(m.to_str().unwrap().to_string()),
        replay: 4,
        panel_options: panel_opts(&base, &m, &out),
    };
    let rep = panel_v2::validate(&vo);
    assert!(rep.pass(), "{}", rep.json());
    let status = |id: &str| rep.checks.iter().find(|c| c.id == id).unwrap().status;
    for id in ["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V9", "V10", "V11", "V12"] {
        assert_eq!(status(id), "pass", "{id}: {}", rep.json());
    }
    vo.replay = 0;
    assert_eq!(panel_v2::validate(&vo).checks.iter().find(|c| c.id == "V7").unwrap().status, "skipped");
}

#[test]
fn an_undeclared_level_jump_fails_the_build_and_a_declared_relist_passes() {
    let dir = scratch("jump");
    // Only a 5 day gap: no automatic new listing, so 20x is an undeclared jump.
    let e = build_market(&dir, 24 * 5, "").unwrap_err();
    assert!(e.contains("undeclared level jump") && e.contains("LUNAUSDT"), "{e}");
    assert!(!dir.join("panel.rcsb").exists());
    // 2021-01-09 lies between the last bar before the gap (hour 199) and the first after it.
    let (_, _, out) = build_market(&dir, 24 * 5, "LUNAUSDT@2021-01-12").unwrap();
    let c = Cols::open(&out);
    let ids: std::collections::BTreeSet<i64> =
        c.rows_of("LUNAUSDT").iter().map(|&r| c.f("listing_id", r) as i64).collect();
    assert_eq!(ids.len(), 2);
}

#[test]
fn cold_and_warm_builds_are_byte_identical() {
    let dir = scratch("warm");
    let (raw, m) = market(&dir, 24 * 40);
    let cache = dir.join("cache");
    let sha = |p: &Path| manifest::sha256_file(p).unwrap();

    let mut o = base_opts(&raw, &dir.join("cold.rcsb"));
    o.resume = Some(cache.to_str().unwrap().to_string());
    let cold = panel_v2::build_base(&o).unwrap();
    assert_eq!(cold.reused, 0);
    let mut o2 = base_opts(&raw, &dir.join("warm.rcsb"));
    o2.resume = o.resume.clone();
    let warm = panel_v2::build_base(&o2).unwrap();
    assert_eq!(warm.reused, warm.symbols, "every symbol comes from the cache");
    assert_eq!(sha(&dir.join("cold.rcsb")), sha(&dir.join("warm.rcsb")));
    let plain = base_opts(&raw, &dir.join("plain.rcsb"));
    panel_v2::build_base(&plain).unwrap();
    assert_eq!(sha(&dir.join("plain.rcsb")), sha(&dir.join("cold.rcsb")), "the cache changes cost, not bytes");

    // a changed archive invalidates only its symbol
    put_zip(&raw, "AAAUSDT", "2021-01", &hours(10.0, 0, 361, &[100, 101, 102], Unit::Ms));
    let mut o3 = base_opts(&raw, &dir.join("changed.rcsb"));
    o3.resume = o.resume.clone();
    let changed = panel_v2::build_base(&o3).unwrap();
    assert_eq!(changed.reused, changed.symbols - 1);

    let p1 = dir.join("p1.rcsb");
    let p2 = dir.join("p2.rcsb");
    panel_v2::build_panel(&panel_opts(&dir.join("cold.rcsb"), &m, &p1)).unwrap();
    panel_v2::build_panel(&panel_opts(&dir.join("warm.rcsb"), &m, &p2)).unwrap();
    assert_eq!(sha(&p1), sha(&p2));
}

#[test]
fn the_archive_unit_follows_the_timestamp_unit_of_the_source() {
    let dir = scratch("unit");
    let raw = dir.join("raw");
    // 2024-12-31 22:00 .. 2025-01-01 02:00 across a ms and a us archive.
    let start = (1_735_682_400 - T0) / H;
    put_zip(&raw, "MIXUSDT", "2024-12", &hours(4.0, start, 2, &[], Unit::Ms));
    put_zip(&raw, "MIXUSDT", "2025-01", &hours(4.0, start + 2, 2, &[], Unit::Us));
    let out = dir.join("b.rcsb");
    panel_v2::build_base(&base_opts(&raw, &out)).unwrap();
    let c = Cols::open(&out);
    let units: Vec<f64> = (0..c.n()).map(|r| c.f("archive_unit", r)).collect();
    assert_eq!(units, vec![1.0, 1.0, 2.0, 2.0]);
    assert_eq!(c.f("gap", 3), 0.0, "the unit change does not create a gap");
}

#[test]
fn validation_fails_on_a_corrupt_panel() {
    let dir = scratch("corrupt");
    let (base, m, out) = build_market(&dir, 24 * 40, "").unwrap();
    let mut t = FileStore::open(out.to_str().unwrap()).unwrap().to_table().unwrap();
    let hi = t.index_of("high").unwrap();
    let lo = t.index_of("low").unwrap();
    let cov = t.index_of("cov15m").unwrap();
    let below = t.col_mut(lo)[5] - 1.0;
    t.col_mut(hi)[5] = below;
    t.col_mut(cov)[9] = 3.0;
    let bad = dir.join("bad.rcsb");
    store::write_store_with_time(bad.to_str().unwrap(), &t, 65_536, Some(0)).unwrap();
    let rep = panel_v2::validate(&ValidateOptions {
        panel: bad.to_str().unwrap().to_string(),
        base: Some(base.to_str().unwrap().to_string()),
        membership: Some(m.to_str().unwrap().to_string()),
        replay: 0,
        panel_options: panel_opts(&base, &m, &bad),
    });
    assert!(!rep.pass());
    let failed: Vec<&str> = rep.checks.iter().filter(|c| c.status == "fail").map(|c| c.id).collect();
    assert!(failed.contains(&"V4") && failed.contains(&"V5"), "{failed:?}");
}

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

fn audit(base: &Path, extra: &[&str]) -> std::process::Output {
    std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"))
        .args(["panel", "audit", "--data", base.to_str().unwrap(), "--bar-seconds", "3600"])
        .args(["--lookback", "3", "--min-history", "0", "--min-bars", "1", "--rebalance", "5", "--strict"])
        .args(extra)
        .output()
        .unwrap()
}

#[test]
fn the_audit_reports_jumps_and_fails_strict_on_an_undeclared_one() {
    let dir = scratch("audit");
    let (raw, _) = market(&dir, 24 * 5);
    let base = dir.join("base.rcsb");
    panel_v2::build_base(&base_opts(&raw, &base)).unwrap();
    let bad = audit(&base, &[]);
    let text = String::from_utf8_lossy(&bad.stdout).to_string() + &String::from_utf8_lossy(&bad.stderr);
    assert_eq!(bad.status.code(), Some(3), "{text}");
    assert!(text.contains("# panel: jump LUNAUSDT") && text.contains("declared=false"), "{text}");
    assert!(text.contains("undeclared level jump"), "{text}");

    let ok = audit(&base, &["--relist", "LUNAUSDT@2021-01-12"]);
    let text = String::from_utf8_lossy(&ok.stdout).to_string() + &String::from_utf8_lossy(&ok.stderr);
    assert_eq!(ok.status.code(), Some(0), "{text}");
    assert!(text.contains("declared=true"), "{text}");

    let long = scratch("audit_long");
    let (raw, _) = market(&long, 24 * 40);
    let base = long.join("base.rcsb");
    panel_v2::build_base(&base_opts(&raw, &base)).unwrap();
    let ok = audit(&base, &[]);
    assert_eq!(ok.status.code(), Some(0), "a listing break after a long outage is declared by the gap");
}

