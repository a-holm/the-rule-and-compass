//! #704: recorded fit batches. `RCSR_FIT_RECORD=<dir>` makes the device fit
//! write every launch's raw kernel inputs and outputs to little-endian files,
//! and `rcsr-gpu-bench --fit-replay <dir>` replays them. Pure `std`; the codec
//! needs no device.
//!
//! ```text
//! data-<j>.bin  "RCFITDAT" u32 version=1 u32 n u32 v u32 flags
//!               u32 len(cols) u32 len(y) f32 cols f32 y
//!               [flags&4: u32 len(cols64) u32 len(y64) u32 len(w) f64 cols64 f64 y64 f64 w]
//!               [flags&2: u32 len(tsign) f64 tsign]
//! call-<k>.bin  "RCFITCAL" u32 version=1 u32 data u32 flags(1 = pair mask set)
//!               u32 iters i32 loss i32 polish u32 nprog u32 kmax
//!               u32 nwords u32 words u32 off[nprog+1] i32 pcount[nprog]
//!               f64 c0[nprog*kmax] u32 has_cbase [f64 cbase[nprog*kmax]] i32 gauge[2*nprog]
//!               f32 ms i32 status[nprog] f64 sse[nprog] f64 c[nprog*kmax]
//! ```
//! Flags of a data file: 1 weights present, 2 label sign present, 4 f64 twin
//! uploaded by `set_fit_data`.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::OnceLock;

const DATA_MAGIC: &[u8; 8] = b"RCFITDAT";
const CALL_MAGIC: &[u8; 8] = b"RCFITCAL";
const VERSION: u32 = 1;

/// Call flag: a slope pair mask was resident, so the replay cannot reproduce it.
pub const CALL_PAIRS: u32 = 1;

/// The f64 twin `set_fit_data` uploaded.
#[derive(Clone, Debug, PartialEq)]
pub struct RecTwin {
    pub cols: Vec<f64>,
    pub y: Vec<f64>,
    pub w: Vec<f64>,
}

/// The rows a fit launch reads.
#[derive(Clone, Debug, PartialEq)]
pub struct RecData {
    pub n: usize,
    pub v: usize,
    pub cols: Vec<f32>,
    pub y: Vec<f32>,
    pub twin: Option<RecTwin>,
    pub tsign: Vec<f64>,
}

/// One fit launch: the kernel inputs and the recorded result.
#[derive(Clone, Debug, PartialEq)]
pub struct RecCall {
    pub data: u32,
    pub flags: u32,
    pub iters: u32,
    pub loss: i32,
    pub polish: i32,
    pub nprog: u32,
    pub kmax: u32,
    pub words: Vec<u32>,
    pub off: Vec<u32>,
    pub pcount: Vec<i32>,
    pub c0: Vec<f64>,
    pub cbase: Option<Vec<f64>>,
    pub gauge: Vec<i32>,
    pub ms: f32,
    pub status: Vec<i32>,
    pub sse: Vec<f64>,
    pub c: Vec<f64>,
}

struct Writer(Vec<u8>);

impl Writer {
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.0.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fn f64s(&mut self, v: &[f64]) {
        for x in v {
            self.0.extend_from_slice(&x.to_bits().to_le_bytes());
        }
    }
    fn u32s(&mut self, v: &[u32]) {
        for x in v {
            self.u32(*x);
        }
    }
    fn i32s(&mut self, v: &[i32]) {
        for x in v {
            self.i32(*x);
        }
    }
}

struct Reader<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let end = self.at.checked_add(n).filter(|&e| e <= self.b.len());
        match end {
            Some(e) => {
                let s = &self.b[self.at..e];
                self.at = e;
                Ok(s)
            }
            None => Err("truncated fit record".to_string()),
        }
    }
    fn magic(&mut self, want: &[u8; 8]) -> Result<(), String> {
        if self.take(8)? != want {
            return Err(format!("bad magic, want {}", String::from_utf8_lossy(want)));
        }
        let v = self.u32()?;
        if v != VERSION {
            return Err(format!("unsupported fit record version {}", v));
        }
        Ok(())
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn i32(&mut self) -> Result<i32, String> {
        Ok(i32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_bits(self.u32()?))
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        let s = self.take(n.checked_mul(4).ok_or("size overflow")?)?;
        Ok(s.chunks_exact(4).map(|c| f32::from_bits(u32::from_le_bytes(c.try_into().unwrap()))).collect())
    }
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, String> {
        let s = self.take(n.checked_mul(8).ok_or("size overflow")?)?;
        Ok(s.chunks_exact(8).map(|c| f64::from_bits(u64::from_le_bytes(c.try_into().unwrap()))).collect())
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, String> {
        let s = self.take(n.checked_mul(4).ok_or("size overflow")?)?;
        Ok(s.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn i32s(&mut self, n: usize) -> Result<Vec<i32>, String> {
        let s = self.take(n.checked_mul(4).ok_or("size overflow")?)?;
        Ok(s.chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

impl RecData {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(DATA_MAGIC);
        w.u32(VERSION);
        w.u32(self.n as u32);
        w.u32(self.v as u32);
        let has_w = self.twin.as_ref().is_some_and(|t| !t.w.is_empty());
        let flags = u32::from(has_w) | (u32::from(!self.tsign.is_empty()) << 1) | (u32::from(self.twin.is_some()) << 2);
        w.u32(flags);
        w.u32(self.cols.len() as u32);
        w.u32(self.y.len() as u32);
        w.f32s(&self.cols);
        w.f32s(&self.y);
        if let Some(t) = &self.twin {
            w.u32(t.cols.len() as u32);
            w.u32(t.y.len() as u32);
            w.u32(t.w.len() as u32);
            w.f64s(&t.cols);
            w.f64s(&t.y);
            w.f64s(&t.w);
        }
        if !self.tsign.is_empty() {
            w.u32(self.tsign.len() as u32);
            w.f64s(&self.tsign);
        }
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { b: bytes, at: 0 };
        r.magic(DATA_MAGIC)?;
        let n = r.u32()? as usize;
        let v = r.u32()? as usize;
        let flags = r.u32()?;
        let ncols = r.u32()? as usize;
        let ny = r.u32()? as usize;
        let cols = r.f32s(ncols)?;
        let y = r.f32s(ny)?;
        let twin = if flags & 4 != 0 {
            let (nc, ny64, nw) = (r.u32()? as usize, r.u32()? as usize, r.u32()? as usize);
            Some(RecTwin { cols: r.f64s(nc)?, y: r.f64s(ny64)?, w: r.f64s(nw)? })
        } else {
            None
        };
        let tsign = if flags & 2 != 0 {
            let nt = r.u32()? as usize;
            r.f64s(nt)?
        } else {
            Vec::new()
        };
        if r.at != bytes.len() {
            return Err("trailing bytes in data record".to_string());
        }
        Ok(RecData { n, v, cols, y, twin, tsign })
    }
}

impl RecCall {
    pub fn encode(&self) -> Vec<u8> {
        let mut w = Writer(Vec::new());
        w.0.extend_from_slice(CALL_MAGIC);
        w.u32(VERSION);
        w.u32(self.data);
        w.u32(self.flags);
        w.u32(self.iters);
        w.i32(self.loss);
        w.i32(self.polish);
        w.u32(self.nprog);
        w.u32(self.kmax);
        w.u32(self.words.len() as u32);
        w.u32s(&self.words);
        w.u32s(&self.off);
        w.i32s(&self.pcount);
        w.f64s(&self.c0);
        w.u32(u32::from(self.cbase.is_some()));
        if let Some(cb) = &self.cbase {
            w.f64s(cb);
        }
        w.i32s(&self.gauge);
        w.u32(self.ms.to_bits());
        w.i32s(&self.status);
        w.f64s(&self.sse);
        w.f64s(&self.c);
        w.0
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        let mut r = Reader { b: bytes, at: 0 };
        r.magic(CALL_MAGIC)?;
        let data = r.u32()?;
        let flags = r.u32()?;
        let iters = r.u32()?;
        let loss = r.i32()?;
        let polish = r.i32()?;
        let nprog = r.u32()?;
        let kmax = r.u32()?;
        let (np, km) = (nprog as usize, kmax as usize);
        let nwords = r.u32()? as usize;
        let words = r.u32s(nwords)?;
        let off = r.u32s(np.checked_add(1).ok_or("size overflow")?)?;
        let pcount = r.i32s(np)?;
        let cells = np.checked_mul(km).ok_or("size overflow")?;
        let c0 = r.f64s(cells)?;
        let cbase = if r.u32()? != 0 { Some(r.f64s(cells)?) } else { None };
        let gauge = r.i32s(np.checked_mul(2).ok_or("size overflow")?)?;
        let ms = r.f32()?;
        let status = r.i32s(np)?;
        let sse = r.f64s(np)?;
        let c = r.f64s(cells)?;
        if r.at != bytes.len() {
            return Err("trailing bytes in call record".to_string());
        }
        Ok(RecCall { data, flags, iters, loss, polish, nprog, kmax, words, off, pcount, c0, cbase, gauge, ms, status, sse, c })
    }
}

/// FNV-1a, 64 bit.
pub struct Fnv(pub u64);

impl Fnv {
    pub fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    pub fn bytes(&mut self, b: &[u8]) {
        for &x in b {
            self.0 ^= x as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn candidate(&mut self, status: i32, sse: f64, c: &[f64]) {
        self.bytes(&status.to_le_bytes());
        self.bytes(&sse.to_bits().to_le_bytes());
        for x in c {
            self.bytes(&x.to_bits().to_le_bytes());
        }
    }
}

impl Default for Fnv {
    fn default() -> Self {
        Self::new()
    }
}

/// Hash of one candidate's `(status, sse bits, c bits)`; `c` holds its used constants.
pub fn candidate_hash(status: i32, sse: f64, c: &[f64]) -> u64 {
    let mut h = Fnv::new();
    h.candidate(status, sse, c);
    h.0
}

/// The constants the host reads: the first `pcount` of the candidate's `kmax` row. The
/// kernel leaves the padding slots of a row unwritten, so they are never compared.
pub fn used_constants<'a>(c: &'a [f64], kmax: usize, pcount: &[i32], k: usize) -> &'a [f64] {
    &c[k * kmax..k * kmax + pcount[k] as usize]
}

/// Hash of a whole result, candidate by candidate in order.
pub fn output_hash(status: &[i32], sse: &[f64], c: &[f64], kmax: usize, pcount: &[i32]) -> u64 {
    let mut h = Fnv::new();
    for (k, (&st, &s)) in status.iter().zip(sse).enumerate() {
        h.candidate(st, s, used_constants(c, kmax, pcount, k));
    }
    h.0
}

fn numbered(dir: &Path, prefix: &str) -> Result<Vec<(u32, PathBuf)>, String> {
    let mut out = Vec::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {}", dir.display(), e))?;
    for ent in rd.flatten() {
        let name = ent.file_name().to_string_lossy().into_owned();
        if let Some(num) = name.strip_prefix(prefix).and_then(|s| s.strip_suffix(".bin")) {
            if let Ok(k) = num.parse::<u32>() {
                out.push((k, ent.path()));
            }
        }
    }
    out.sort();
    Ok(out)
}

/// A recording read back from disk.
pub struct Recording {
    pub data: Vec<(u32, RecData)>,
    pub calls: Vec<(u32, RecCall)>,
    pub hash: u64,
}

/// Read every `data-<j>.bin` and `call-<k>.bin` of `dir`, in index order.
pub fn load_recording(dir: &Path) -> Result<Recording, String> {
    let mut hash = Fnv::new();
    let mut data = Vec::new();
    for (j, p) in numbered(dir, "data-")? {
        let b = std::fs::read(&p).map_err(|e| format!("{}: {}", p.display(), e))?;
        hash.bytes(&b);
        data.push((j, RecData::decode(&b).map_err(|e| format!("{}: {}", p.display(), e))?));
    }
    let mut calls = Vec::new();
    for (k, p) in numbered(dir, "call-")? {
        let b = std::fs::read(&p).map_err(|e| format!("{}: {}", p.display(), e))?;
        hash.bytes(&b);
        calls.push((k, RecCall::decode(&b).map_err(|e| format!("{}: {}", p.display(), e))?));
    }
    if calls.is_empty() {
        return Err(format!("{}: no call-<k>.bin files", dir.display()));
    }
    Ok(Recording { data, calls, hash: hash.0 })
}

static DIR: OnceLock<Option<PathBuf>> = OnceLock::new();
static NEXT_DATA: AtomicU32 = AtomicU32::new(0);
static NEXT_CALL: AtomicU32 = AtomicU32::new(0);

/// The recording directory of this process, read once from `RCSR_FIT_RECORD`.
/// A directory that already holds files disables recording with one warning.
pub fn record_dir() -> Option<&'static Path> {
    DIR.get_or_init(|| {
        let raw = std::env::var_os("RCSR_FIT_RECORD").filter(|v| !v.is_empty())?;
        let dir = PathBuf::from(raw);
        let non_empty = match std::fs::read_dir(&dir) {
            Ok(mut rd) => rd.next().is_some(),
            Err(_) => dir.exists(),
        };
        if non_empty {
            eprintln!("warning: RCSR_FIT_RECORD={} exists and is not an empty directory; recording is off", dir.display());
            return None;
        }
        if let Err(e) = std::fs::create_dir_all(&dir) {
            eprintln!("warning: RCSR_FIT_RECORD={}: {}; recording is off", dir.display(), e);
            return None;
        }
        Some(dir)
    })
    .as_deref()
}

fn write_numbered(prefix: &str, counter: &AtomicU32, bytes: &[u8]) -> Option<u32> {
    let dir = record_dir()?;
    let k = counter.fetch_add(1, Ordering::SeqCst);
    let path = dir.join(format!("{}{}.bin", prefix, k));
    match std::fs::write(&path, bytes) {
        Ok(()) => Some(k),
        Err(e) => {
            eprintln!("warning: fit record {}: {}", path.display(), e);
            None
        }
    }
}

/// Per-evaluator recorder state: a host copy of the rows the next launch will
/// read, written to a `data-<j>.bin` before the first call that uses it.
pub struct EvalRec {
    data: RecData,
    written: Option<u32>,
    pub polish: i32,
}

impl EvalRec {
    /// `None` unless `RCSR_FIT_RECORD` selected a usable directory.
    pub fn open(cols: &[f32], n: usize, v: usize, y: &[f32]) -> Option<Self> {
        record_dir()?;
        Some(EvalRec { data: RecData { n, v, cols: cols.to_vec(), y: y.to_vec(), twin: None, tsign: Vec::new() }, written: None, polish: 0 })
    }

    pub fn set_twin(&mut self, cols: &[f64], y: &[f64], w: &[f64]) {
        self.data.twin = Some(RecTwin { cols: cols.to_vec(), y: y.to_vec(), w: w.to_vec() });
        self.written = None;
    }

    pub fn set_sign(&mut self, sign: &[f64]) {
        self.data.tsign = sign.to_vec();
        self.written = None;
    }

    /// Write one launch; `call.data` is filled in here.
    pub fn record(&mut self, mut call: RecCall) {
        if self.written.is_none() {
            self.written = write_numbered("data-", &NEXT_DATA, &self.data.encode());
        }
        let Some(j) = self.written else { return };
        call.data = j;
        write_numbered("call-", &NEXT_CALL, &call.encode());
    }
}

/// One replayed call as the NDJSON line the replay prints.
pub struct ReplayLine {
    pub call: u32,
    pub inst: String,
    pub ms_median: f64,
    pub lm: Vec<u64>,
    pub polish: Vec<u64>,
    pub cand_hash: Vec<u64>,
}

/// The summary line of a replay.
pub struct ReplaySummary {
    pub calls: usize,
    pub candidates: usize,
    pub ms_total: f64,
    pub toolkit: String,
    pub device: String,
    pub recording_hash: String,
}

fn hex(v: u64) -> String {
    format!("\"0x{:016x}\"", v)
}

fn list<T: std::fmt::Display>(v: &[T]) -> String {
    let parts: Vec<String> = v.iter().map(|x| x.to_string()).collect();
    format!("[{}]", parts.join(","))
}

impl ReplayLine {
    pub fn json(&self, ms: &[f32], status: &[i32], out_hash: u64, same: bool, stable: bool) -> String {
        let hashes: Vec<String> = self.cand_hash.iter().map(|&h| hex(h)).collect();
        format!(
            "{{\"call\":{},\"nprog\":{},\"instantiation\":\"{}\",\"ms\":{},\"ms_median\":{},\"lm_cycles\":{},\"polish_cycles\":{},\"status\":{},\"cand_hash\":[{}],\"out_hash\":{},\"same_as_recorded\":{},\"stable\":{}}}",
            self.call,
            self.cand_hash.len(),
            self.inst,
            list(ms),
            self.ms_median,
            list(&self.lm),
            list(&self.polish),
            list(status),
            hashes.join(","),
            hex(out_hash),
            same,
            stable
        )
    }
}

impl ReplaySummary {
    pub fn json(&self, reps: usize) -> String {
        format!(
            "{{\"summary\":true,\"calls\":{},\"candidates\":{},\"reps\":{},\"ms_total_median\":{},\"toolkit\":\"{}\",\"device\":\"{}\",\"recording_hash\":{}}}",
            self.calls,
            self.candidates,
            reps,
            self.ms_total,
            self.toolkit,
            self.device,
            hex(u64::from_str_radix(&self.recording_hash, 16).unwrap_or(0))
        )
    }
}

/// The raw text of `"key":<value>` in a line of our own NDJSON.
fn field<'a>(line: &'a str, key: &str) -> Option<&'a str> {
    let pat = format!("\"{}\":", key);
    let start = line.find(&pat)? + pat.len();
    let rest = &line[start..];
    let end = match rest.as_bytes().first()? {
        b'[' => rest.find(']')? + 1,
        b'"' => rest[1..].find('"')? + 2,
        _ => rest.find([',', '}'])?,
    };
    Some(&rest[..end])
}

fn unquote(s: &str) -> &str {
    s.trim_matches('"')
}

fn nums<T: std::str::FromStr>(s: &str) -> Option<Vec<T>> {
    let inner = s.trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').map(|x| x.trim().parse().ok()).collect()
}

fn hashes(s: &str) -> Option<Vec<u64>> {
    let inner = s.trim_start_matches('[').trim_end_matches(']');
    if inner.is_empty() {
        return Some(Vec::new());
    }
    inner.split(',').map(|x| u64::from_str_radix(unquote(x.trim()).trim_start_matches("0x"), 16).ok()).collect()
}

/// Parse a replay file: the call lines in order and the summary.
pub fn parse_replay(text: &str) -> Result<(Vec<ReplayLine>, ReplaySummary), String> {
    let mut lines = Vec::new();
    let mut summary = None;
    for (no, raw) in text.lines().enumerate() {
        let bad = || format!("line {}: not a replay line", no + 1);
        if raw.trim().is_empty() {
            continue;
        }
        if field(raw, "summary").is_some() {
            summary = Some(ReplaySummary {
                calls: field(raw, "calls").and_then(|v| v.parse().ok()).ok_or_else(bad)?,
                candidates: field(raw, "candidates").and_then(|v| v.parse().ok()).ok_or_else(bad)?,
                ms_total: field(raw, "ms_total_median").and_then(|v| v.parse().ok()).ok_or_else(bad)?,
                toolkit: unquote(field(raw, "toolkit").ok_or_else(bad)?).to_string(),
                device: unquote(field(raw, "device").ok_or_else(bad)?).to_string(),
                recording_hash: unquote(field(raw, "recording_hash").ok_or_else(bad)?).trim_start_matches("0x").to_string(),
            });
            continue;
        }
        if field(raw, "skipped").is_some() {
            continue;
        }
        lines.push(ReplayLine {
            call: field(raw, "call").and_then(|v| v.parse().ok()).ok_or_else(bad)?,
            inst: unquote(field(raw, "instantiation").ok_or_else(bad)?).to_string(),
            ms_median: field(raw, "ms_median").and_then(|v| v.parse().ok()).ok_or_else(bad)?,
            lm: field(raw, "lm_cycles").and_then(nums).ok_or_else(bad)?,
            polish: field(raw, "polish_cycles").and_then(nums).ok_or_else(bad)?,
            cand_hash: field(raw, "cand_hash").and_then(hashes).ok_or_else(bad)?,
        });
    }
    Ok((lines, summary.ok_or("no summary line")?))
}

#[derive(Default, Clone, Copy)]
struct Tally {
    calls: usize,
    ms_a: f64,
    ms_b: f64,
    cands: usize,
    matching: usize,
    cyc_a_match: f64,
    cyc_b_match: f64,
    cyc_a_all: f64,
    cyc_b_all: f64,
}

impl Tally {
    fn add(&mut self, o: &Tally) {
        self.calls += o.calls;
        self.ms_a += o.ms_a;
        self.ms_b += o.ms_b;
        self.cands += o.cands;
        self.matching += o.matching;
        self.cyc_a_match += o.cyc_a_match;
        self.cyc_b_match += o.cyc_b_match;
        self.cyc_a_all += o.cyc_a_all;
        self.cyc_b_all += o.cyc_b_all;
    }

    fn row(&self, label: &str) -> String {
        let r = |b: f64, a: f64| if a > 0.0 { format!("{:.4}", b / a) } else { "n/a".to_string() };
        format!(
            "{:<10} calls {:>3}  ms {:>10.3} -> {:>10.3}  ratio b/a {}  matching {}/{}  cycle ratio matching {}  all {}",
            label,
            self.calls,
            self.ms_a,
            self.ms_b,
            r(self.ms_b, self.ms_a),
            self.matching,
            self.cands,
            r(self.cyc_b_match, self.cyc_a_match),
            r(self.cyc_b_all, self.cyc_a_all)
        )
    }
}

/// The compare report of two replays of one recording. `Err((2, why))` when the
/// files cannot be compared.
pub fn compare_replays(a: &str, b: &str, allow_device_mismatch: bool) -> Result<String, (i32, String)> {
    let (la, sa) = parse_replay(a).map_err(|e| (2, format!("a: {}", e)))?;
    let (lb, sb) = parse_replay(b).map_err(|e| (2, format!("b: {}", e)))?;
    if sa.recording_hash != sb.recording_hash {
        return Err((2, format!("different recordings: {} vs {}", sa.recording_hash, sb.recording_hash)));
    }
    if sa.device != sb.device && !allow_device_mismatch {
        return Err((2, format!("different devices: '{}' vs '{}' (use --allow-device-mismatch)", sa.device, sb.device)));
    }
    if la.len() != lb.len() || la.iter().zip(&lb).any(|(x, y)| x.call != y.call || x.cand_hash.len() != y.cand_hash.len()) {
        return Err((2, "the two replays cover different calls".to_string()));
    }
    let mut out = format!("a: toolkit {}  b: toolkit {}  device {}\n", sa.toolkit, sb.toolkit, sa.device);
    let mut total = Tally::default();
    let mut by_inst: Vec<(String, Tally)> = Vec::new();
    for (x, y) in la.iter().zip(&lb) {
        let mut t = Tally { calls: 1, ms_a: x.ms_median, ms_b: y.ms_median, cands: x.cand_hash.len(), ..Tally::default() };
        for i in 0..x.cand_hash.len() {
            let cyc = |l: &ReplayLine| (l.lm.get(i).copied().unwrap_or(0) + l.polish.get(i).copied().unwrap_or(0)) as f64;
            t.cyc_a_all += cyc(x);
            t.cyc_b_all += cyc(y);
            if x.cand_hash[i] == y.cand_hash[i] {
                t.matching += 1;
                t.cyc_a_match += cyc(x);
                t.cyc_b_match += cyc(y);
            }
        }
        out.push_str(&t.row(&format!("call {}", x.call)));
        out.push_str(&format!("  [{}]\n", x.inst));
        total.add(&t);
        match by_inst.iter_mut().find(|(n, _)| *n == x.inst) {
            Some((_, s)) => s.add(&t),
            None => by_inst.push((x.inst.clone(), t)),
        }
    }
    for (name, t) in &by_inst {
        out.push_str(&t.row(name));
        out.push('\n');
    }
    out.push_str(&total.row("total"));
    out.push('\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.0 >> 11
        }
        fn f64(&mut self) -> f64 {
            // Arbitrary bit patterns, NaN payloads and negative zero included.
            f64::from_bits(self.next() << 11 | self.next() & 0x7ff)
        }
        fn f32(&mut self) -> f32 {
            f32::from_bits(self.next() as u32)
        }
    }

    fn sample_call(rng: &mut Lcg) -> RecCall {
        let (nprog, kmax) = (5usize, 3usize);
        RecCall {
            data: 2,
            flags: CALL_PAIRS,
            iters: 40,
            loss: 4,
            polish: 1,
            nprog: nprog as u32,
            kmax: kmax as u32,
            words: (0..17).map(|_| rng.next() as u32).collect(),
            off: vec![0, 3, 7, 9, 12, 17],
            pcount: vec![1, 2, 3, 1, 3],
            c0: (0..nprog * kmax).map(|_| rng.f64()).collect(),
            cbase: Some((0..nprog * kmax).map(|_| rng.f64()).collect()),
            gauge: vec![-1, -1, 0, 1, 2, -1, -1, -1, 1, 0],
            ms: rng.f32(),
            status: vec![0, 4, 5, 6, 8],
            sse: (0..nprog).map(|_| rng.f64()).collect(),
            c: (0..nprog * kmax).map(|_| rng.f64()).collect(),
        }
    }

    fn bits(v: &[f64]) -> Vec<u64> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn fit_record_roundtrip_is_bitwise() {
        let mut rng = Lcg(704);
        let call = sample_call(&mut rng);
        let back = RecCall::decode(&call.encode()).unwrap();
        assert_eq!(back.words, call.words);
        assert_eq!(back.off, call.off);
        assert_eq!(back.pcount, call.pcount);
        assert_eq!(back.gauge, call.gauge);
        assert_eq!(back.status, call.status);
        assert_eq!(bits(&back.c0), bits(&call.c0));
        assert_eq!(bits(back.cbase.as_ref().unwrap()), bits(call.cbase.as_ref().unwrap()));
        assert_eq!(bits(&back.sse), bits(&call.sse));
        assert_eq!(bits(&back.c), bits(&call.c));
        assert_eq!(back.ms.to_bits(), call.ms.to_bits());
        assert_eq!((back.data, back.flags, back.iters, back.loss, back.polish), (2, CALL_PAIRS, 40, 4, 1));
        let mut none = call.clone();
        none.cbase = None;
        assert!(RecCall::decode(&none.encode()).unwrap().cbase.is_none());

        let data = RecData {
            n: 4,
            v: 2,
            cols: (0..8).map(|_| rng.f32()).collect(),
            y: (0..4).map(|_| rng.f32()).collect(),
            twin: Some(RecTwin { cols: (0..8).map(|_| rng.f64()).collect(), y: (0..4).map(|_| rng.f64()).collect(), w: (0..4).map(|_| rng.f64()).collect() }),
            tsign: (0..4).map(|_| rng.f64()).collect(),
        };
        let back = RecData::decode(&data.encode()).unwrap();
        let fb = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!((back.n, back.v), (4, 2));
        assert_eq!(fb(&back.cols), fb(&data.cols));
        assert_eq!(fb(&back.y), fb(&data.y));
        let (bt, dt) = (back.twin.as_ref().unwrap(), data.twin.as_ref().unwrap());
        assert_eq!(bits(&bt.cols), bits(&dt.cols));
        assert_eq!(bits(&bt.y), bits(&dt.y));
        assert_eq!(bits(&bt.w), bits(&dt.w));
        assert_eq!(bits(&back.tsign), bits(&data.tsign));
    }

    #[test]
    fn fit_record_rejects_a_bad_magic() {
        let mut rng = Lcg(7);
        let mut bytes = sample_call(&mut rng).encode();
        bytes[3] ^= 1;
        assert!(RecCall::decode(&bytes).unwrap_err().contains("bad magic"));
        let data = RecData { n: 1, v: 1, cols: vec![1.0], y: vec![2.0], twin: None, tsign: Vec::new() };
        let mut bytes = data.encode();
        bytes[0] ^= 1;
        assert!(RecData::decode(&bytes).is_err());
        let ok = data.encode();
        assert!(RecData::decode(&ok[..ok.len() - 1]).is_err());
    }
}
