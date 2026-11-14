//! #1130: shared fixtures of the lock-record tests: a record file, a small enrich
//! store with a planted extreme move inside the range, hourly archives with a
//! planted symbol and spike inside the range, and a membership table.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::process::Output;

/// 2021-01-01T00:00:00Z.
pub const T0: i64 = 1_609_459_200;
pub const H: i64 = 3600;
/// First hour inside the locked range, as an hour index since `T0`.
pub const LOCK_HOUR: i64 = 800;
/// The range ends long after the data.
pub const END_HOUR: i64 = 100_000;
/// A symbol that trades only inside the range.
pub const PLANT: &str = "ZZPLANTUSDT";

pub fn lock_start() -> i64 {
    T0 + LOCK_HOUR * H
}

pub fn lock_end() -> i64 {
    T0 + END_HOUR * H
}

pub fn record_text(start: i64, end: i64) -> String {
    format!(
        "{{\n  \"schema\": \"rcsr-locked-tail/1\",\n  \"range\": {{\n    \"start_epoch\": {start},\n    \"end_epoch\": {end},\n    \"end_exclusive\": true\n  }},\n  \"reason\": \"fixture\"\n}}\n"
    )
}

pub fn write_record(dir: &Path) -> PathBuf {
    let p = dir.join("locked-tail-fixture.json");
    std::fs::write(&p, record_text(lock_start(), lock_end())).unwrap();
    p
}

pub fn run(args: &[&str]) -> (i32, String, String) {
    let out: Output = crate::rcsr_cmd::rcsr().args(args).output().unwrap();
    (
        out.status.code().unwrap_or(-1),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

pub fn s(p: &Path) -> String {
    p.to_string_lossy().into_owned()
}

fn quarter(t: i64, base: f64, k: i64) -> String {
    let p = base * (1.0 + 0.001 * k as f64);
    let (o, c) = (p, p * 1.0005);
    let (h, l) = (c * 1.001, o * 0.999);
    let v = 10.0 + (k % 7) as f64;
    let qv = v * (o + c) / 2.0;
    let raw = t * 1000;
    format!("{raw},{o},{h},{l},{c},{v},{raw},{qv},{},{},{},0", 5 + k % 3, v * 0.4, qv * 0.4)
}

/// Full hours `[from, from + n)` (hour indexes since T0). `spike` hours trade at 2.5 times the price.
pub fn hours(base: f64, from: i64, n: i64, spike: &[i64]) -> Vec<String> {
    let mut out = Vec::new();
    for hr in from..from + n {
        let b = if spike.contains(&hr) { base * 2.5 } else { base };
        for q in 0..4 {
            out.push(quarter(T0 + hr * H + q * 900, b, hr * 4 + q));
        }
    }
    out
}

pub fn put_zip(root: &Path, symbol: &str, month: &str, lines: &[String]) -> PathBuf {
    let dir = root.join(symbol);
    std::fs::create_dir_all(&dir).unwrap();
    let name = format!("{symbol}-15m-{month}");
    let path = dir.join(format!("{name}.zip"));
    let csv = lines.join("\n") + "\n";
    std::fs::write(&path, stored_zip(&format!("{name}.csv"), csv.as_bytes())).unwrap();
    path
}

/// Hours of January 2021 and of February 2021 since `T0`.
pub const JAN_END: i64 = 744;
pub const FEB_END: i64 = 1416;
/// The hour at which the planted symbol starts to trade (inside the range).
pub const PLANT_FROM: i64 = 850;
/// The hour of the planted price spike of AAAUSDT (inside the range).
pub const SPIKE_HOUR: i64 = 1000;

/// January and February for AAAUSDT and BBBUSDT; February only for the planted symbol.
/// With `dev_only` the archives end before the range start.
pub fn archives(root: &Path, dev_only: bool) {
    let feb_end = if dev_only { LOCK_HOUR } else { FEB_END };
    for (sym, base) in [("AAAUSDT", 10.0), ("BBBUSDT", 20.0)] {
        put_zip(root, sym, "2021-01", &hours(base, 0, JAN_END, &[]));
        put_zip(root, sym, "2021-02", &hours(base, JAN_END, feb_end - JAN_END, &[SPIKE_HOUR]));
    }
    if !dev_only {
        put_zip(root, PLANT, "2021-02", &hours(7.0, PLANT_FROM, FEB_END - PLANT_FROM, &[]));
    }
}

/// Rebalances at hours 0 and 400 (development) and at hour 900 (inside the range).
pub fn membership(path: &Path, dev_only: bool) {
    let mut text = String::from("as_of,symbol,member,adv_usd,n_bars\n");
    for hr in [0i64, 400] {
        for sym in ["AAAUSDT", "BBBUSDT"] {
            text.push_str(&format!("{},{},1,1000.0,100\n", T0 + hr * H, sym));
        }
    }
    if !dev_only {
        for sym in ["AAAUSDT", "BBBUSDT", PLANT] {
            text.push_str(&format!("{},{},1,1000.0,100\n", T0 + 900 * H, sym));
        }
    }
    std::fs::write(path, text).unwrap();
}

fn stored_zip(name: &str, data: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let crc = crc32(data);
    out.extend_from_slice(&0x0403_4b50u32.to_le_bytes());
    out.extend_from_slice(&20u16.to_le_bytes());
    for _ in 0..4 {
        out.extend_from_slice(&0u16.to_le_bytes());
    }
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
    for _ in 0..4 {
        out.extend_from_slice(&0u16.to_le_bytes());
    }
    out.extend_from_slice(&crc.to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(data.len() as u32).to_le_bytes());
    out.extend_from_slice(&(name.len() as u16).to_le_bytes());
    for _ in 0..4 {
        out.extend_from_slice(&0u16.to_le_bytes());
    }
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

/// The store pipeline of the panel commands, with or without a record.
pub struct Pipeline {
    pub dir: PathBuf,
    pub raw: PathBuf,
    pub snapshot: PathBuf,
    pub base: PathBuf,
    pub membership: PathBuf,
    pub panel: PathBuf,
}

impl Pipeline {
    pub fn new(dir: &Path, dev_only: bool) -> Pipeline {
        std::fs::create_dir_all(dir).unwrap();
        let p = Pipeline {
            dir: dir.to_path_buf(),
            raw: dir.join("raw"),
            snapshot: dir.join("snapshot.json"),
            base: dir.join("base_h1.rcsb"),
            membership: dir.join("membership.csv"),
            panel: dir.join("panel.rcsb"),
        };
        archives(&p.raw, dev_only);
        membership(&p.membership, dev_only);
        p
    }

    /// snapshot, base and build; `record` adds `--lock-record` to each.
    pub fn build(&self, record: Option<&Path>) {
        let rec: Vec<String> = record.map(|r| vec!["--lock-record".to_string(), s(r)]).unwrap_or_default();
        let go = |args: Vec<String>| {
            let a: Vec<&str> = args.iter().map(String::as_str).collect();
            let (code, _, err) = run(&a);
            assert_eq!(code, 0, "{args:?}: {err}");
        };
        let mut a = vec!["panel", "snapshot", "--input", &s(&self.raw), "--jobs", "2", "--out", &s(&self.snapshot)]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        a.extend(rec.clone());
        go(a);
        let mut a = vec!["panel", "base", "--input", &s(&self.raw), "--jobs", "2", "--cpu", "--out", &s(&self.base)]
            .into_iter()
            .map(str::to_string)
            .collect::<Vec<_>>();
        a.extend(rec.clone());
        go(a);
        let mut a = vec![
            "panel", "build", "--data", &s(&self.base), "--membership", &s(&self.membership), "--snapshot", &s(&self.snapshot),
            "--bar-seconds", "3600", "--cpu", "--out", &s(&self.panel),
        ]
        .into_iter()
        .map(str::to_string)
        .collect::<Vec<_>>();
        a.extend(rec);
        go(a);
    }

    /// Every sidecar this pipeline wrote.
    pub fn sidecars(&self) -> Vec<PathBuf> {
        [
            self.snapshot.clone(),
            PathBuf::from(format!("{}.manifest.json", s(&self.base))),
            PathBuf::from(format!("{}.manifest.json", s(&self.panel))),
            PathBuf::from(format!("{}.schema.json", s(&self.panel))),
        ]
        .to_vec()
    }
}

/// Every text file a pipeline run wrote under `dir`, except the inputs (`raw/`, the membership table) and the record.
pub fn written_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        for e in std::fs::read_dir(&d).unwrap().flatten() {
            let p = e.path();
            let name = e.file_name().to_string_lossy().into_owned();
            if p.is_dir() {
                if name != "raw" {
                    stack.push(p);
                }
            } else if name != "membership.csv" && !name.starts_with("locked-tail") && !name.ends_with(".rcsb") {
                out.push(p);
            }
        }
    }
    out.sort();
    out
}

/// A raw directory with a monthly January archive and daily archives: 2021-02-01 (development) and
/// 2021-02-05 and 2021-02-06 (inside the range), for two symbols.
pub fn daily_archives(root: &Path) {
    for (sym, base) in [("AAAUSDT", 10.0), ("BBBUSDT", 20.0)] {
        put_zip(root, sym, "2021-01", &hours(base, 0, JAN_END, &[]));
        put_zip(root, sym, "2021-02-01", &hours(base, JAN_END, 24, &[]));
        put_zip(root, sym, "2021-02-05", &hours(base, JAN_END + 96, 24, &[]));
        put_zip(root, sym, "2021-02-06", &hours(base, JAN_END + 120, 24, &[]));
    }
}
