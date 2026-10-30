//! Bounded pinned-host ring for projected column uploads (issue #294).
//!
//! One reader thread `pread`s a group's projected blocks straight into a
//! `cudaHostAlloc` slot; the main thread enqueues one asynchronous H2D copy per
//! projected column and hands the slot to the per-tile compute stream. The
//! ring is capped at three slots and 64 MB total, so pinning is bounded by the
//! projection, not by the store size.
//!
//! No `cudaStreamSynchronize` is used per group. Slot reuse is guarded by the
//! copy event (`rc_ring_wait_copy`), and the final join is one stream
//! synchronisation after every group has been uploaded.
//!
//! Slot protocol (#352): the reader and the consumer share one mutex-guarded
//! state. A `RingGroup` owns its slot until `upload` hands it back or the group
//! is dropped. The reader only ever waits for a free slot, and `finish`/`Drop`
//! cancel it instead of draining it, so no consumer path can leave either
//! thread blocked on the other. Every wait the consumer makes on the reader is
//! bounded and reports the ring state when it expires.

#![cfg(feature = "cuda")]

use crate::memory::RING_CAP_BYTES;
use crate::store::{FileStore, ProjectedReadStats, Projection};
use std::collections::VecDeque;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int, c_void};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Shortest bound on a wait for the reader, whatever the slot size.
const WAIT_FLOOR: Duration = Duration::from_secs(60);
/// Slowest group read the default bound still tolerates.
const WAIT_MIN_BYTES_PER_S: u64 = 1 << 20;

enum Raw {}

extern "C" {
    fn rc_ring_create(slots: c_int, slot_bytes: usize) -> *mut Raw;
    fn rc_ring_destroy(ring: *mut Raw);
    fn rc_ring_host(ring: *mut Raw, slot: c_int) -> *mut c_void;
    fn rc_ring_upload(
        ring: *mut Raw,
        slot: c_int,
        dev_ptrs: *const *mut c_void,
        dev_offsets: *const usize,
        host_offsets: *const usize,
        bytes: *const usize,
        ncols: c_int,
    ) -> c_int;
    fn rc_ring_compute_begin(ring: *mut Raw, slot: c_int) -> c_int;
    fn rc_ring_compute_end(ring: *mut Raw, slot: c_int) -> c_int;
    fn rc_ring_wait_copy(ring: *mut Raw, slot: c_int) -> c_int;
    fn rc_ring_wait_slot(ring: *mut Raw, slot: c_int) -> c_int;
    fn rc_ring_sync(ring: *mut Raw) -> c_int;
    fn rc_ring_error_name(code: c_int) -> *const c_char;
    fn rc_ring_stats(
        ring: *mut Raw,
        copies: *mut u64,
        computes: *mut u64,
        bytes: *mut u64,
        failed: *mut c_int,
    ) -> c_int;
    fn rc_ring_launch_seg_id(ring: *mut Raw, ctx: *mut c_void, start: c_int, len: c_int) -> c_int;
    fn rc_ring_launch_feature(
        ring: *mut Raw,
        ctx: *mut c_void,
        kind: c_int,
        k: c_int,
        which: c_int,
        start: c_int,
        len: c_int,
        out: *mut f64,
    ) -> c_int;
    fn rc_ring_overlap_probe(
        bytes: usize,
        iters: c_int,
        cycles: u64,
        copy_ms: *mut f64,
        compute_ms: *mut f64,
        overlapped: *mut c_int,
        copies: *mut u64,
        failed: *mut c_int,
    ) -> c_int;
}

/// `what` plus the CUDA error the `rc_ring_*` call returned, if it named one.
fn cuda_failure(what: &str, rc: c_int) -> String {
    if rc > 0 {
        let name = unsafe { rc_ring_error_name(rc) };
        if !name.is_null() {
            let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
            return format!("projected ring: {} ({})", what, name);
        }
    }
    format!("projected ring: {}", what)
}

/// One filled group, in ring order. It owns its pinned slot: `upload` hands
/// the slot back to the reader, and dropping the group unused does too.
pub struct RingGroup {
    pub group: usize,
    pub slot: usize,
    pub len: usize,
    pub stats: ProjectedReadStats,
    lease: Option<Arc<Shared>>,
}

impl RingGroup {
    /// The slot never returns to the reader: a failed upload may still have
    /// copies in flight from its pinned bytes.
    fn forget_slot(&mut self) {
        if let Some(shared) = self.lease.take() {
            let mut st = shared.lock();
            st.held = st.held.saturating_sub(1);
        }
    }
}

impl Drop for RingGroup {
    fn drop(&mut self) {
        if let Some(shared) = self.lease.take() {
            shared.release(self.slot);
        }
    }
}

impl std::fmt::Debug for RingGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RingGroup")
            .field("group", &self.group)
            .field("slot", &self.slot)
            .field("len", &self.len)
            .field("stats", &self.stats)
            .finish()
    }
}

/// Cumulative transport counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct RingStats {
    pub copies: u64,
    pub computes: u64,
    pub ring_bytes: u64,
    pub groups: u64,
    pub bytes_read: u64,
    pub failed: bool,
}

/// Bounds and test hooks for the ring's blocking waits.
#[derive(Clone, Debug, Default)]
pub struct RingOptions {
    /// Longest wait for the reader in `next_group`, and for the reader to exit
    /// after `finish`/`Drop` cancel it. `None` uses [`default_wait`].
    pub wait: Option<Duration>,
    /// Test hook: the reader takes a slot for this group and then stalls until
    /// it is cancelled, without delivering it.
    pub stall_before_group: Option<usize>,
    /// Test hook: the stalled reader ignores cancellation for this long.
    pub stall_ignores_cancel: Option<Duration>,
}

impl RingOptions {
    /// `RCSR_RING_WAIT_MS` overrides the wait bound; the test hook
    /// `RCSR_TEST_RING_STALL_GROUP` stalls the reader before that group.
    pub fn from_env() -> Self {
        let wait = std::env::var("RCSR_RING_WAIT_MS")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .filter(|ms| *ms > 0)
            .map(Duration::from_millis);
        let stall_before_group = std::env::var("RCSR_TEST_RING_STALL_GROUP")
            .ok()
            .and_then(|v| v.trim().parse::<usize>().ok());
        RingOptions {
            wait,
            stall_before_group,
            stall_ignores_cancel: None,
        }
    }
}

/// Default bound on one wait for the reader of a ring with `slot_bytes` slots.
pub fn default_wait(slot_bytes: usize) -> Duration {
    WAIT_FLOOR.max(Duration::from_secs(slot_bytes as u64 / WAIT_MIN_BYTES_PER_S + 1))
}

/// Per-tile feature kind understood by `rc_ring_launch_feature`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TileFeature {
    Ret(usize),
    Lag(usize),
    Range,
    Clv,
    Calendar(usize),
}

impl TileFeature {
    fn kind(self) -> (c_int, c_int) {
        match self {
            TileFeature::Ret(k) => (0, k as c_int),
            TileFeature::Lag(k) => (1, k as c_int),
            TileFeature::Range => (2, 0),
            TileFeature::Clv => (3, 0),
            TileFeature::Calendar(w) => (4, w as c_int),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Phase {
    WaitSlot,
    WaitCopy,
    Read,
    Stalled,
}

struct Filled {
    group: usize,
    slot: usize,
    len: usize,
    stats: ProjectedReadStats,
}

struct State {
    /// Slots the reader may fill.
    free: Vec<usize>,
    /// Filled groups (or the reader's error) not yet taken by the consumer.
    ready: VecDeque<Result<Filled, String>>,
    cancelled: bool,
    /// The reader delivered every group of the store.
    completed: bool,
    /// The reader left its loop, for any reason (including a panic).
    exited: bool,
    phase: Phase,
    group: usize,
    sent: u64,
    received: u64,
    /// Groups the consumer holds and has neither uploaded nor dropped.
    held: usize,
}

struct Shared {
    state: Mutex<State>,
    wake: Condvar,
    slots: usize,
    groups: usize,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn wait<'a>(&self, st: MutexGuard<'a, State>) -> MutexGuard<'a, State> {
        self.wake.wait(st).unwrap_or_else(|e| e.into_inner())
    }

    fn wait_until<'a>(
        &self,
        st: MutexGuard<'a, State>,
        deadline: Instant,
    ) -> MutexGuard<'a, State> {
        let left = deadline.saturating_duration_since(Instant::now());
        match self.wake.wait_timeout(st, left) {
            Ok((g, _)) => g,
            Err(e) => e.into_inner().0,
        }
    }

    fn release(&self, slot: usize) {
        let mut st = self.lock();
        st.held = st.held.saturating_sub(1);
        st.free.push(slot);
        self.wake.notify_all();
    }

    /// Record the reader's next step; `false` once it has been cancelled.
    fn enter(&self, phase: Phase, group: usize) -> bool {
        let mut st = self.lock();
        st.phase = phase;
        st.group = group;
        !st.cancelled
    }

    /// Hand a filled group or an error to the consumer; `false` once cancelled.
    fn deliver(&self, item: Result<Filled, String>) -> bool {
        let mut st = self.lock();
        if st.cancelled {
            return false;
        }
        if item.is_ok() {
            st.sent += 1;
        }
        st.ready.push_back(item);
        self.wake.notify_all();
        true
    }

    fn describe(&self, st: &State) -> String {
        let reader = if st.exited {
            format!("reader exited at group {}", st.group)
        } else {
            let doing = match st.phase {
                Phase::WaitSlot => "waiting for a free slot for",
                Phase::WaitCopy => "waiting for the previous copy out of its slot before",
                Phase::Read => "reading",
                Phase::Stalled => "stalled by the test hook before",
            };
            format!("reader {} group {}", doing, st.group)
        };
        format!(
            "{}; slots free {} of {}, held by the consumer {}; groups sent {}, received {} of {}",
            reader,
            st.free.len(),
            self.slots,
            st.held,
            st.sent,
            st.received,
            self.groups
        )
    }
}

/// Marks the reader as gone however its loop ends, so no wait outlives it.
struct ReaderExit(Arc<Shared>);

impl Drop for ReaderExit {
    fn drop(&mut self) {
        let mut st = self.0.lock();
        st.exited = true;
        self.0.wake.notify_all();
    }
}

/// A bounded pinned ring plus its one reader thread.
pub struct ProjectedRing {
    handle: *mut Raw,
    slots: usize,
    slot_bytes: usize,
    projection: Projection,
    shared: Arc<Shared>,
    reader: Option<JoinHandle<()>>,
    wait: Duration,
    stats: RingStats,
    synced: bool,
    /// The reader outlived its bound: the pinned slots stay allocated because
    /// it may still write into them.
    leaked: bool,
}

impl ProjectedRing {
    /// Create the ring for `projection`, with bounds from [`RingOptions::from_env`].
    ///
    /// `slots` is clamped to 2..=3 and the whole ring must fit 64 MB. A single
    /// group larger than the cap is refused with its byte count: the transport
    /// never silently shrinks the projection or falls back to a full table.
    pub fn create(
        store: &FileStore,
        projection: Projection,
        slots: usize,
    ) -> Result<Self, String> {
        Self::create_with(store, projection, slots, RingOptions::from_env())
    }

    /// [`ProjectedRing::create`] with explicit bounds and test hooks.
    pub fn create_with(
        store: &FileStore,
        projection: Projection,
        slots: usize,
        options: RingOptions,
    ) -> Result<Self, String> {
        let max_len = if store.n_groups() == 0 {
            0
        } else {
            store.header().group_rows.min(store.n_rows())
        };
        let slot_bytes = projection.group_bytes(max_len) as usize;
        if slot_bytes == 0 {
            return Err("projected ring: the store has no rows".to_string());
        }
        let max_slots = (RING_CAP_BYTES as usize) / slot_bytes;
        if max_slots < 2 {
            return Err(format!(
                "projected ring: one group is {} bytes, so two pinned slots need {} bytes, \
                 above the {} byte cap; reduce --group-rows or the projection",
                slot_bytes,
                2 * slot_bytes,
                RING_CAP_BYTES
            ));
        }
        let slots = slots.clamp(2, 3).min(max_slots);
        let handle = unsafe { rc_ring_create(slots as c_int, slot_bytes) };
        if handle.is_null() {
            return Err("projected ring: cudaHostAlloc/cudaStreamCreate failed".to_string());
        }
        let slot_ptrs: Vec<usize> = (0..slots)
            .map(|i| unsafe { rc_ring_host(handle, i as c_int) as usize })
            .collect();
        if slot_ptrs.iter().any(|p| *p == 0) {
            unsafe { rc_ring_destroy(handle) };
            return Err("projected ring: a pinned slot was not allocated".to_string());
        }
        let store = match store.try_clone() {
            Ok(s) => s,
            Err(e) => {
                unsafe { rc_ring_destroy(handle) };
                return Err(e);
            }
        };

        let wait = options.wait.unwrap_or_else(|| default_wait(slot_bytes));
        let shared = Arc::new(Shared {
            state: Mutex::new(State {
                free: (0..slots).rev().collect(),
                ready: VecDeque::new(),
                cancelled: false,
                completed: false,
                exited: false,
                phase: Phase::WaitSlot,
                group: 0,
                sent: 0,
                received: 0,
                held: 0,
            }),
            wake: Condvar::new(),
            slots,
            groups: store.n_groups(),
        });
        let proj = projection.clone();
        let handle_addr = handle as usize;
        let reader_shared = Arc::clone(&shared);
        let spawned = std::thread::Builder::new()
            .name("rcsr-projected-reader".to_string())
            .spawn(move || {
                let shared = reader_shared;
                let _exit = ReaderExit(Arc::clone(&shared));
                let ring = handle_addr as *mut Raw;
                for group in 0..store.n_groups() {
                    let slot = {
                        let mut st = shared.lock();
                        st.phase = Phase::WaitSlot;
                        st.group = group;
                        loop {
                            if st.cancelled {
                                return;
                            }
                            if let Some(s) = st.free.pop() {
                                break s;
                            }
                            st = shared.wait(st);
                        }
                    };
                    if options.stall_before_group == Some(group) {
                        let mut st = shared.lock();
                        st.phase = Phase::Stalled;
                        match options.stall_ignores_cancel {
                            Some(d) => {
                                drop(st);
                                std::thread::sleep(d);
                            }
                            None => {
                                while !st.cancelled {
                                    st = shared.wait(st);
                                }
                            }
                        }
                        return;
                    }
                    // The previous consumer of this slot may still have an
                    // in-flight H2D copy reading the pinned bytes. Wait on
                    // that slot's copy event in the reader thread, so the main
                    // thread can keep enqueuing the next group's copy.
                    if !shared.enter(Phase::WaitCopy, group) {
                        return;
                    }
                    let rc = unsafe { rc_ring_wait_copy(ring, slot as c_int) };
                    if rc != 0 {
                        shared.deliver(Err(cuda_failure("copy event wait failed", rc)));
                        return;
                    }
                    if !shared.enter(Phase::Read, group) {
                        return;
                    }
                    let len = store.header().group_len(group);
                    let need = proj.group_bytes(len) as usize;
                    let dst = unsafe {
                        std::slice::from_raw_parts_mut(slot_ptrs[slot] as *mut u8, slot_bytes)
                    };
                    match store.read_group_projected_raw_into(group, &proj, &mut dst[..need]) {
                        Ok(mut stats) => {
                            stats.groups = 1;
                            let filled = Filled {
                                group,
                                slot,
                                len,
                                stats,
                            };
                            if !shared.deliver(Ok(filled)) {
                                return;
                            }
                        }
                        Err(e) => {
                            shared.deliver(Err(e));
                            return;
                        }
                    }
                }
                shared.lock().completed = true;
            });
        let join = match spawned {
            Ok(j) => j,
            Err(e) => {
                unsafe { rc_ring_destroy(handle) };
                return Err(format!("projected ring: reader thread: {}", e));
            }
        };

        Ok(ProjectedRing {
            handle,
            slots,
            slot_bytes,
            projection,
            shared,
            reader: Some(join),
            wait,
            stats: RingStats::default(),
            synced: false,
            leaked: false,
        })
    }

    pub fn projection(&self) -> &Projection {
        &self.projection
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// The bound on one wait for the reader.
    pub fn wait_limit(&self) -> Duration {
        self.wait
    }

    /// Constant pinned byte count for the life of the ring.
    pub fn ring_bytes(&self) -> u64 {
        (self.slots as u64).saturating_mul(self.slot_bytes as u64)
    }

    /// The next group the reader has filled, or `None` at end of store.
    ///
    /// Waits at most [`ProjectedRing::wait_limit`]; on expiry, or if the
    /// reader exits before the end of the store, the error names the ring
    /// state.
    pub fn next_group(&mut self) -> Option<Result<RingGroup, String>> {
        self.reader.as_ref()?;
        let shared = Arc::clone(&self.shared);
        let deadline = Instant::now() + self.wait;
        let mut st = shared.lock();
        loop {
            if let Some(item) = st.ready.pop_front() {
                let f = match item {
                    Ok(f) => f,
                    Err(e) => return Some(Err(e)),
                };
                st.received += 1;
                st.held += 1;
                drop(st);
                self.stats.groups += 1;
                self.stats.bytes_read = self.stats.bytes_read.saturating_add(f.stats.value_bytes);
                return Some(Ok(RingGroup {
                    group: f.group,
                    slot: f.slot,
                    len: f.len,
                    stats: f.stats,
                    lease: Some(Arc::clone(&shared)),
                }));
            }
            if st.completed {
                return None;
            }
            if st.exited {
                return Some(Err(format!(
                    "projected ring: the reader stopped before the end of the store ({})",
                    shared.describe(&st)
                )));
            }
            if Instant::now() >= deadline {
                return Some(Err(format!(
                    "projected ring: no group from the reader within {:.1} s ({})",
                    self.wait.as_secs_f64(),
                    shared.describe(&st)
                )));
            }
            st = shared.wait_until(st, deadline);
        }
    }

    /// Enqueue the asynchronous upload of `group`'s slot into the destination
    /// columns, then hand the slot back to the reader.
    ///
    /// `dests` is one `(device base pointer, device byte offset)` per projected
    /// column; the host offset is derived from the projection order and `len`.
    pub fn upload(
        &mut self,
        mut group: RingGroup,
        dests: &[(*mut c_void, usize)],
    ) -> Result<(), String> {
        match group.lease.as_ref() {
            Some(lease) if Arc::ptr_eq(lease, &self.shared) => {}
            _ => return Err("projected ring: the group does not belong to this ring".to_string()),
        }
        if self.reader.is_none() {
            return Err("projected ring: upload after finish".to_string());
        }
        if dests.len() != self.projection.len() {
            return Err(format!(
                "projected ring: {} destinations for {} projected columns",
                dests.len(),
                self.projection.len()
            ));
        }
        let mut dev_ptrs: Vec<*mut c_void> = Vec::with_capacity(dests.len());
        let mut dev_offsets: Vec<usize> = Vec::with_capacity(dests.len());
        let mut host_offsets: Vec<usize> = Vec::with_capacity(dests.len());
        let mut bytes: Vec<usize> = Vec::with_capacity(dests.len());
        let mut at = 0usize;
        for (i, col) in self.projection.columns().iter().enumerate() {
            let n = group.len * col.width as usize;
            dev_ptrs.push(dests[i].0);
            dev_offsets.push(dests[i].1);
            host_offsets.push(at);
            bytes.push(n);
            at += n;
        }
        let rc = unsafe {
            rc_ring_upload(
                self.handle,
                group.slot as c_int,
                dev_ptrs.as_ptr(),
                dev_offsets.as_ptr(),
                host_offsets.as_ptr(),
                bytes.as_ptr(),
                dev_ptrs.len() as c_int,
            )
        };
        if rc != 0 {
            self.stats.failed = true;
            group.forget_slot();
            return Err(cuda_failure("cudaMemcpyAsync failed", rc));
        }
        // Dropping the group returns the slot. The reader waits on this slot's
        // copy event before overwriting the pinned bytes; the main thread does
        // not block here.
        drop(group);
        Ok(())
    }

    /// Make the compute stream wait for this slot's copy.
    pub fn compute_begin(&mut self, slot: usize) -> Result<(), String> {
        let rc = unsafe { rc_ring_compute_begin(self.handle, slot as c_int) };
        if rc != 0 {
            self.stats.failed = true;
            return Err(cuda_failure("compute stream wait failed", rc));
        }
        Ok(())
    }

    /// Record that the compute stream finished with this slot.
    pub fn compute_end(&mut self, slot: usize) -> Result<(), String> {
        let rc = unsafe { rc_ring_compute_end(self.handle, slot as c_int) };
        if rc != 0 {
            self.stats.failed = true;
            return Err(cuda_failure("compute event record failed", rc));
        }
        Ok(())
    }

    /// Launch the segment-id bookkeeping kernel on the compute stream.
    pub fn launch_seg_id(
        &mut self,
        ctx: *mut c_void,
        start: usize,
        len: usize,
    ) -> Result<(), String> {
        let rc = unsafe {
            rc_ring_launch_seg_id(self.handle, ctx, start as c_int, len as c_int)
        };
        if rc != 0 {
            self.stats.failed = true;
            return Err(cuda_failure("segment-id kernel failed", rc));
        }
        Ok(())
    }

    /// Launch one tiled feature kernel on the compute stream.
    pub fn launch_feature(
        &mut self,
        ctx: *mut c_void,
        feature: TileFeature,
        start: usize,
        len: usize,
        out: *mut f64,
    ) -> Result<(), String> {
        let (kind, arg) = feature.kind();
        let which = arg;
        let k = arg;
        let rc = unsafe {
            rc_ring_launch_feature(
                self.handle,
                ctx,
                kind,
                k,
                which,
                start as c_int,
                len as c_int,
                out,
            )
        };
        if rc != 0 {
            self.stats.failed = true;
            return Err(cuda_failure("tiled feature kernel failed", rc));
        }
        Ok(())
    }

    /// Cancel the reader and wait at most the wait bound for it to exit. A
    /// reader that outlives the bound is detached and its pinned slots leaked.
    fn stop_reader(&mut self) -> Option<String> {
        let join = self.reader.take()?;
        let shared = Arc::clone(&self.shared);
        let deadline = Instant::now() + self.wait;
        let mut st = shared.lock();
        st.cancelled = true;
        st.ready.clear();
        shared.wake.notify_all();
        while !st.exited {
            if Instant::now() >= deadline {
                let msg = format!(
                    "projected ring: the reader did not exit within {:.1} s of cancellation ({}); \
                     its {} pinned bytes stay allocated",
                    self.wait.as_secs_f64(),
                    shared.describe(&st),
                    self.ring_bytes()
                );
                drop(st);
                self.leaked = true;
                drop(join);
                return Some(msg);
            }
            st = shared.wait_until(st, deadline);
        }
        drop(st);
        match join.join() {
            Ok(()) => None,
            Err(_) => Some("projected ring: the reader thread panicked".to_string()),
        }
    }

    /// Final join: cancel the reader, then one stream synchronisation.
    pub fn finish(&mut self) -> Result<(), String> {
        let stopped = self.stop_reader();
        let rc = unsafe { rc_ring_sync(self.handle) };
        self.synced = true;
        let mut copies = 0u64;
        let mut computes = 0u64;
        let mut bytes = 0u64;
        let mut failed = 0 as c_int;
        unsafe {
            rc_ring_stats(
                self.handle,
                &mut copies,
                &mut computes,
                &mut bytes,
                &mut failed,
            )
        };
        self.stats.copies = copies;
        self.stats.computes = computes;
        self.stats.ring_bytes = bytes;
        self.stats.failed |= failed != 0;
        if let Some(e) = stopped {
            self.stats.failed = true;
            return Err(e);
        }
        if rc != 0 || self.stats.failed {
            Err("projected ring: final synchronisation reported a failure".to_string())
        } else {
            Ok(())
        }
    }

    pub fn stats(&self) -> RingStats {
        self.stats
    }

    /// Wait until the compute stream has consumed `slot`. Used by tests.
    pub fn wait_slot(&mut self, slot: usize) -> Result<(), String> {
        if unsafe { rc_ring_wait_slot(self.handle, slot as c_int) } != 0 {
            Err("projected ring: slot event wait failed".to_string())
        } else {
            Ok(())
        }
    }
}

impl Drop for ProjectedRing {
    fn drop(&mut self) {
        if let Some(e) = self.stop_reader() {
            eprintln!("# {}", e);
        }
        if self.handle.is_null() {
            return;
        }
        if !self.synced {
            unsafe { rc_ring_sync(self.handle) };
        }
        if !self.leaked {
            unsafe { rc_ring_destroy(self.handle) };
        }
        self.handle = std::ptr::null_mut();
    }
}

/// Result of the event-ordering probe, in milliseconds.
#[derive(Clone, Copy, Debug, Default)]
pub struct OverlapProbe {
    pub copy_ms: f64,
    pub compute_ms: f64,
    pub overlapped: bool,
    pub copies: u64,
    pub failed: bool,
}

/// Test/benchmark hook: run copies and a busy compute kernel through the same
/// ring and report whether the two stream windows overlap.
pub fn overlap_probe(bytes: usize, iters: usize, cycles: u64) -> Option<OverlapProbe> {
    let mut copy_ms = 0.0f64;
    let mut compute_ms = 0.0f64;
    let mut overlapped = 0 as c_int;
    let mut copies = 0u64;
    let mut failed = 0 as c_int;
    let rc = unsafe {
        rc_ring_overlap_probe(
            bytes,
            iters as c_int,
            cycles,
            &mut copy_ms,
            &mut compute_ms,
            &mut overlapped,
            &mut copies,
            &mut failed,
        )
    };
    if rc != 0 {
        return None;
    }
    Some(OverlapProbe {
        copy_ms,
        compute_ms,
        overlapped: overlapped != 0,
        copies,
        failed: failed != 0,
    })
}
