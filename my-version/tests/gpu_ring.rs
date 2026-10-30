//! #294: pinned ring bound, copy/compute overlap and tile invariance.
//! #352: no consumer path can deadlock the ring, and every wait is bounded.
//!
//! Device tests skip cleanly when no CUDA device is present.

#![cfg(feature = "cuda")]

use rcsr::data;
use rcsr::enrich_gpu::{DeviceColumn, EnrichGpu, ProjectedSource};
use rcsr::projected_ring::{overlap_probe, ProjectedRing, RingGroup, RingOptions};
use rcsr::store::{self, FileStore, Projection};
use std::os::raw::{c_int, c_void};
use std::sync::mpsc;
use std::time::{Duration, Instant};

fn tmp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "rcsr-gpu-ring-{}-{}-{}",
        tag,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Two contiguous symbols with a gap, enough groups to exercise slot reuse.
fn write_store(path: &str, rows_per_symbol: usize, group_rows: usize) {
    let mut text = String::from("ts,symbol,open,high,low,close,volume,quote_volume\n");
    for (sym, start) in [("AAA", 0i64), ("BBB", 10_000_000)] {
        for i in 0..rows_per_symbol {
            let mut t = start + i as i64 * 3600;
            if i > rows_per_symbol / 2 {
                t += 3 * 3600;
            }
            let p = 100.0 + (i as f64 * 0.013).sin() * 7.0 + i as f64 * 0.0001;
            text.push_str(&format!(
                "{},{},{},{},{},{},{},{}\n",
                t,
                sym,
                p,
                p + 0.3 + (i as f64 * 0.07).cos().abs() * 0.2,
                p - 0.4 - (i as f64 * 0.05).sin().abs() * 0.2,
                p + (i as f64 * 0.11).sin() * 0.1,
                10.0 + (i % 97) as f64,
                1000.0 + (i % 53) as f64
            ));
        }
    }
    let table = data::parse_table_with_time(&text, false, Some("ts")).unwrap();
    store::write_store_with_time(path, &table, group_rows, Some(0)).unwrap();
}

struct Built {
    proj: Projection,
    sources: Vec<ProjectedSource>,
    seg_start: Vec<u32>,
    seg_len: Vec<u32>,
}

fn build(fs: &FileStore) -> Built {
    let names = [
        "ts",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
    ];
    let proj = Projection::resolve(fs, &names).unwrap();
    let dests = [
        DeviceColumn::Time,
        DeviceColumn::Value(0),
        DeviceColumn::Value(1),
        DeviceColumn::Value(2),
        DeviceColumn::Value(3),
        DeviceColumn::Value(4),
        DeviceColumn::Value(5),
    ];
    let sources = proj
        .columns()
        .iter()
        .zip(dests)
        .map(|(c, d)| ProjectedSource {
            header_index: c.index,
            destination: d,
        })
        .collect();
    // The store is symbol-major with two contiguous runs.
    let rows = fs.n_rows();
    let half = rows / 2;
    Built {
        proj,
        sources,
        seg_start: vec![0, half as u32],
        seg_len: vec![half as u32, (rows - half) as u32],
    }
}

fn run(fs: &FileStore, b: &Built, slots: usize, names: &[String]) -> (Vec<Vec<u64>>, u64) {
    let mut gpu = EnrichGpu::from_projected(
        fs,
        b.proj.clone(),
        &b.sources,
        &b.seg_start,
        &b.seg_len,
        3600,
        names,
        slots,
    )
    .expect("projected transport");
    let ring = gpu.ring_bytes();
    let mut out = Vec::new();
    for n in names {
        let vals = gpu.take_precomputed(n).expect("precomputed tile");
        out.push(vals.iter().map(|v| v.to_bits()).collect());
    }
    (out, ring)
}

#[test]
fn ring_is_bounded_constant_and_tiles_are_invariant() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("bound");
    let path = dir.join("in.rcsb").to_string_lossy().into_owned();
    let path_small_groups = dir.join("in2.rcsb").to_string_lossy().into_owned();
    // 100 groups of 64 rows for the bound, and a 2-group layout for the
    // group-size invariance check.
    write_store(&path, 3_200, 64);
    write_store(&path_small_groups, 3_200, 4096);

    let fs = FileStore::open(&path).unwrap();
    assert_eq!(fs.n_groups(), 100);
    let b = build(&fs);
    let names = [
        "r1".to_string(),
        "r5".to_string(),
        "lag_r1_3".to_string(),
        "range1".to_string(),
        "clv1".to_string(),
        "hour_sin".to_string(),
    ];

    let (two, ring_two) = run(&fs, &b, 2, &names);
    let (three, ring_three) = run(&fs, &b, 3, &names);
    assert!(ring_two > 0 && ring_two <= 64 * 1024 * 1024);
    assert!(ring_three > 0 && ring_three <= 64 * 1024 * 1024);
    // The pin is a function of one group and the slot count, not of how many
    // groups were read: 100 groups do not grow it.
    assert_eq!(ring_two % 2, 0);
    assert_eq!(ring_three % 3, 0);
    assert_eq!(ring_three / 3, ring_two / 2);
    assert_eq!(two, three, "slot count changed the values");

    let fs2 = FileStore::open(&path_small_groups).unwrap();
    assert_eq!(fs2.n_groups(), 2);
    let b2 = build(&fs2);
    let (two_groups, _) = run(&fs2, &b2, 3, &names);
    assert_eq!(three, two_groups, "group size changed the values");

    // The tiled kernels must match the whole-column reference bit for bit.
    let mut whole = {
        let mut cols: Vec<Vec<f64>> = Vec::new();
        for name in ["ts", "open", "high", "low", "close", "volume", "quote_volume"] {
            let c = fs.header().column_index(name).unwrap();
            let mut v = Vec::new();
            for g in 0..fs.n_groups() {
                v.extend(fs.read_f64_group(g, c).unwrap());
            }
            cols.push(v);
        }
        let times: Vec<i64> = cols[0].iter().map(|v| *v as i64).collect();
        EnrichGpu::new(
            &times, &cols[1], &cols[2], &cols[3], &cols[4], &cols[5], &cols[6],
            &b.seg_start, &b.seg_len, 3600,
        )
        .unwrap()
    };
    for (i, name) in names.iter().enumerate() {
        let expected = match name.as_str() {
            "r1" => whole.ret(1),
            "r5" => whole.ret(5),
            "lag_r1_3" => whole.lag(3),
            "range1" => whole.range(),
            "clv1" => whole.clv(),
            "hour_sin" => whole.calendar(2),
            _ => None,
        }
        .unwrap();
        let want: Vec<u64> = expected.iter().map(|v| v.to_bits()).collect();
        assert_eq!(two[i], want, "tiled '{}' differs from the whole kernel", name);
    }

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn copy_overlaps_compute_with_events_not_stream_sync() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let probe = overlap_probe(8 * 1024 * 1024, 8, 20_000_000).expect("overlap probe");
    assert!(probe.copies >= 8);
    assert!(!probe.failed);
    assert!(
        probe.overlapped,
        "copy {:.3} ms and compute {:.3} ms did not overlap",
        probe.copy_ms, probe.compute_ms
    );
    assert!(probe.copy_ms > 0.0 && probe.compute_ms > 0.0);
}

// ---------------------------------------------------------------------------
// #352: the ring protocol cannot deadlock, and every wait is bounded.
// ---------------------------------------------------------------------------

// Ring-level tests upload into the same enrichment arena the projected
// transport fills.
extern "C" {
    fn rc_enrich_create_arena(
        n: c_int,
        nseg: c_int,
        step: c_int,
        seg_start: *const c_int,
        seg_len: *const c_int,
        nan_mask: u32,
    ) -> *mut c_void;
    fn rc_enrich_destroy(ctx: *mut c_void);
    fn rc_enrich_times_ptr(ctx: *mut c_void) -> *mut i64;
    fn rc_enrich_value_ptr(ctx: *mut c_void, which: c_int) -> *mut f64;
}

const RING_GROUPS: usize = 64;
const RING_NAMES: [&str; 7] = [
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
];

/// A 64-group store and device columns for every projected column.
struct RingFixture {
    dir: std::path::PathBuf,
    path: String,
    ctx: usize,
    cols: Vec<usize>,
    group_rows: usize,
}

impl RingFixture {
    fn new(tag: &str) -> Self {
        let dir = tmp_dir(tag);
        let path = dir.join("in.rcsb").to_string_lossy().into_owned();
        write_store(&path, 2_048, 64);
        let fs = FileStore::open(&path).unwrap();
        assert_eq!(fs.n_groups(), RING_GROUPS);
        let n = fs.n_rows();
        let starts = [0 as c_int];
        let lens = [n as c_int];
        let ctx = unsafe {
            rc_enrich_create_arena(n as c_int, 1, 3600, starts.as_ptr(), lens.as_ptr(), 0)
        };
        assert!(!ctx.is_null(), "device arena");
        let mut cols = vec![unsafe { rc_enrich_times_ptr(ctx) } as usize];
        for which in 0..6 {
            cols.push(unsafe { rc_enrich_value_ptr(ctx, which) } as usize);
        }
        RingFixture {
            dir,
            path,
            ctx: ctx as usize,
            cols,
            group_rows: fs.group_rows(),
        }
    }

    fn handle(&self) -> RingHandle {
        RingHandle {
            path: self.path.clone(),
            cols: self.cols.clone(),
            group_rows: self.group_rows,
        }
    }
}

impl Drop for RingFixture {
    fn drop(&mut self) {
        unsafe { rc_enrich_destroy(self.ctx as *mut c_void) };
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

/// What a worker thread needs to drive one ring.
#[derive(Clone)]
struct RingHandle {
    path: String,
    cols: Vec<usize>,
    group_rows: usize,
}

impl RingHandle {
    fn ring(&self, slots: usize, options: RingOptions) -> ProjectedRing {
        let fs = FileStore::open(&self.path).unwrap();
        let proj = Projection::resolve(&fs, &RING_NAMES).unwrap();
        let ring = ProjectedRing::create_with(&fs, proj, slots, options).unwrap();
        assert_eq!(ring.slots(), slots);
        ring
    }

    fn upload(&self, ring: &mut ProjectedRing, group: RingGroup) {
        let at = group.group * self.group_rows * 8;
        let dests: Vec<(*mut c_void, usize)> =
            self.cols.iter().map(|p| (*p as *mut c_void, at)).collect();
        ring.upload(group, &dests).expect("upload");
    }
}

/// Run `f` on its own thread and fail, instead of hanging, if it does not
/// return within `limit`. A hung worker is left behind; the test still fails.
fn bounded<T: Send + 'static>(
    what: &str,
    limit: Duration,
    f: impl FnOnce() -> T + Send + 'static,
) -> T {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(f());
    });
    match rx.recv_timeout(limit) {
        Ok(v) => v,
        Err(mpsc::RecvTimeoutError::Timeout) => {
            panic!("{} did not return within {:?}", what, limit)
        }
        Err(mpsc::RecvTimeoutError::Disconnected) => panic!("{} panicked", what),
    }
}

#[derive(Clone, Copy, Debug)]
enum Stop {
    /// Upload `k` groups, then `finish`.
    Finish,
    /// Upload `k` groups, take group `k` without uploading it, then `finish`
    /// while holding it: the `from_projected` error paths.
    FinishHolding,
    /// Upload `k` groups, then drop the ring without `finish`.
    Drop,
}

#[test]
fn early_exit_after_any_group_never_hangs() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let fx = RingFixture::new("early-exit");
    let limit = Duration::from_secs(5);
    let mut cases = 0usize;
    for slots in [2usize, 3] {
        for k in 0..RING_GROUPS {
            for stop in [Stop::Finish, Stop::FinishHolding, Stop::Drop] {
                // Stop at once, and after the reader has filled every slot it
                // can reach: the shutdown race needs buffered free slots.
                for settle in [Duration::ZERO, Duration::from_millis(5)] {
                    let h = fx.handle();
                    let what = format!("slots={} k={} {:?} settle={:?}", slots, k, stop, settle);
                    let took = bounded(&what, Duration::from_secs(30), move || {
                        let mut ring = h.ring(slots, RingOptions::default());
                        for i in 0..k {
                            let g = ring.next_group().expect("group").expect("group");
                            assert_eq!(g.group, i);
                            h.upload(&mut ring, g);
                        }
                        let held = match stop {
                            Stop::FinishHolding if k < RING_GROUPS => {
                                Some(ring.next_group().expect("group").expect("group"))
                            }
                            _ => None,
                        };
                        std::thread::sleep(settle);
                        let t = Instant::now();
                        match stop {
                            Stop::Finish | Stop::FinishHolding => ring.finish().expect("finish"),
                            Stop::Drop => drop(ring),
                        }
                        let took = t.elapsed();
                        drop(held);
                        took
                    });
                    assert!(took < limit, "{}: stopping took {:?}", what, took);
                    cases += 1;
                }
            }
        }
    }
    assert_eq!(cases, 2 * RING_GROUPS * 3 * 2);
}

#[test]
fn dropped_group_returns_its_slot() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let fx = RingFixture::new("dropped-group");
    for slots in [2usize, 3] {
        // Skip every third group, then every group: no upload at all must
        // still reach the end of the store.
        for every in [3usize, 1] {
            let h = fx.handle();
            let what = format!("slots={} skip every {}", slots, every);
            let (seen, uploaded) = bounded(&what, Duration::from_secs(30), move || {
                let options = RingOptions {
                    wait: Some(Duration::from_secs(5)),
                    ..Default::default()
                };
                let mut ring = h.ring(slots, options);
                let mut seen = Vec::new();
                let mut uploaded = 0usize;
                while let Some(g) = ring.next_group() {
                    let g = g.expect("group");
                    seen.push(g.group);
                    if g.group % every == 0 {
                        drop(g);
                    } else {
                        h.upload(&mut ring, g);
                        uploaded += 1;
                    }
                }
                ring.finish().expect("finish");
                (seen, uploaded)
            });
            assert_eq!(seen, (0..RING_GROUPS).collect::<Vec<_>>(), "{}", what);
            let skipped = (0..RING_GROUPS).filter(|g| g % every == 0).count();
            assert_eq!(uploaded, RING_GROUPS - skipped, "{}", what);
        }
    }
}

#[test]
fn stalled_reader_times_out_with_ring_state() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    // Ring level: the wait expires within its bound and names the state.
    let fx = RingFixture::new("stall");
    let bound = Duration::from_millis(400);
    let h = fx.handle();
    let (err, waited, finished) = bounded("stalled ring", Duration::from_secs(30), move || {
        let options = RingOptions {
            wait: Some(bound),
            stall_before_group: Some(5),
            ..Default::default()
        };
        let mut ring = h.ring(3, options);
        assert_eq!(ring.wait_limit(), bound);
        for _ in 0..5 {
            let g = ring.next_group().expect("group").expect("group");
            h.upload(&mut ring, g);
        }
        let t = Instant::now();
        let err = ring.next_group().expect("an error, not end of store").unwrap_err();
        let waited = t.elapsed();
        let t = Instant::now();
        ring.finish().expect("a stalled reader honours cancellation");
        (err, waited, t.elapsed())
    });
    assert!(waited >= bound, "returned after {:?}, before the bound", waited);
    assert!(waited < bound + Duration::from_secs(2), "returned after {:?}", waited);
    assert!(finished < Duration::from_secs(5), "finish took {:?}", finished);
    for part in [
        "no group from the reader within 0.4 s",
        "reader stalled by the test hook before group 5",
        "slots free 2 of 3",
        "held by the consumer 0",
        "groups sent 5, received 5 of 64",
    ] {
        assert!(err.contains(part), "missing '{}' in: {}", part, err);
    }

    // Enrich level: the same stall through `rcsr enrich --gpu on` falls back
    // to the whole-column upload, prints one line and writes the same values.
    let dir = tmp_dir("stall-enrich");
    let src = dir.join("in.rcsb").to_string_lossy().into_owned();
    write_store(&src, 520_000, 65_536);
    let features = "r1,r5,lag_r1_3,range1,clv1,hour_sin,ma20";
    let enrich = |out: &str, stall: bool| {
        let mut cmd = std::process::Command::new(env!("CARGO_BIN_EXE_rcsr"));
        cmd.args(["enrich", "--data", &src, "--out", out, "--features", features]);
        cmd.args(["--gpu", "on", "--allow-lockbox"]);
        cmd.env_remove("RCSR_RING_WAIT_MS").env_remove("RCSR_TEST_RING_STALL_GROUP");
        if stall {
            cmd.env("RCSR_RING_WAIT_MS", "400").env("RCSR_TEST_RING_STALL_GROUP", "3");
        }
        let started = Instant::now();
        let output = cmd.output().expect("run rcsr enrich");
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        assert!(output.status.success(), "enrich failed: {}", stderr);
        let failed: Vec<String> = stderr
            .lines()
            .filter(|l| l.contains("projected transport failed"))
            .map(str::to_string)
            .collect();
        (failed, started.elapsed())
    };
    let stalled_out = dir.join("stalled.rcsb").to_string_lossy().into_owned();
    let normal_out = dir.join("normal.rcsb").to_string_lossy().into_owned();
    let (failed, took) = enrich(&stalled_out, true);
    assert_eq!(failed.len(), 1, "expected one fallback line, got {:?}", failed);
    assert!(
        failed[0].contains("reader stalled by the test hook before group 3")
            && failed[0].contains("whole-column fallback"),
        "{}",
        failed[0]
    );
    assert!(took < Duration::from_secs(120), "stalled enrich took {:?}", took);
    let (failed, _) = enrich(&normal_out, false);
    assert!(failed.is_empty(), "the unhooked run fell back: {:?}", failed);
    let a = FileStore::open(&stalled_out).unwrap();
    let b = FileStore::open(&normal_out).unwrap();
    for name in features.split(',') {
        let ca = a.header().column_index(name).unwrap();
        let cb = b.header().column_index(name).unwrap();
        for g in 0..a.n_groups() {
            let bits = |s: &FileStore, c: usize| -> Vec<u64> {
                s.read_f64_group(g, c).unwrap().iter().map(|v| v.to_bits()).collect()
            };
            assert_eq!(
                bits(&a, ca),
                bits(&b, cb),
                "column {} group {} differs after the fallback",
                name,
                g
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn reader_that_ignores_cancel_is_detached_not_joined() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let fx = RingFixture::new("detach");
    let bound = Duration::from_millis(300);
    let h = fx.handle();
    let (next, finish, took) = bounded("detached reader", Duration::from_secs(30), move || {
        let options = RingOptions {
            wait: Some(bound),
            stall_before_group: Some(2),
            stall_ignores_cancel: Some(Duration::from_secs(3)),
        };
        let mut ring = h.ring(2, options);
        for _ in 0..2 {
            let g = ring.next_group().expect("group").expect("group");
            h.upload(&mut ring, g);
        }
        let next = ring.next_group().expect("an error").unwrap_err();
        let t = Instant::now();
        let finish = ring.finish().unwrap_err();
        let took = t.elapsed();
        drop(ring);
        (next, finish, took)
    });
    assert!(next.contains("stalled by the test hook before group 2"), "{}", next);
    assert!(
        finish.contains("did not exit within 0.3 s") && finish.contains("stay allocated"),
        "{}",
        finish
    );
    assert!(took < Duration::from_secs(2), "finish took {:?}", took);
}

#[test]
fn stale_cuda_error_does_not_fail_the_transport() {
    if !rcsr::gpu::probe().available() {
        eprintln!("skipping: no CUDA device");
        return;
    }
    let dir = tmp_dir("stale");
    let path = dir.join("in.rcsb").to_string_lossy().into_owned();
    write_store(&path, 3_200, 64);
    // CUDA keeps the last error per host thread, so the failed allocation and
    // the transport run on the same worker.
    let (clean, after) = bounded("stale error", Duration::from_secs(60), move || {
        let fs = FileStore::open(&path).unwrap();
        let b = build(&fs);
        let names = ["r1".to_string(), "range1".to_string()];
        let (clean, _) = run(&fs, &b, 3, &names);
        // A failed allocation leaves its error as this thread's last CUDA
        // error, as an optional tile output that does not fit the device does.
        let starts = [0 as c_int];
        let lens = [1 as c_int];
        let arena = unsafe {
            rc_enrich_create_arena(c_int::MAX, 1, 3600, starts.as_ptr(), lens.as_ptr(), 0)
        };
        assert!(arena.is_null(), "a {} row arena must not fit the device", c_int::MAX);
        let (after, _) = run(&fs, &b, 3, &names);
        (clean, after)
    });
    assert_eq!(clean, after);
    let _ = std::fs::remove_dir_all(&dir);
}
