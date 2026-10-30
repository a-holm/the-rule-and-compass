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

#![cfg(feature = "cuda")]

use crate::memory::RING_CAP_BYTES;
use crate::store::{FileStore, ProjectedReadStats, Projection};
use std::os::raw::{c_int, c_void};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::thread::JoinHandle;

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

/// One uploaded group, in ring order.
#[derive(Clone, Copy, Debug)]
pub struct RingGroup {
    pub group: usize,
    pub slot: usize,
    pub len: usize,
    pub stats: ProjectedReadStats,
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

struct Reader {
    join: JoinHandle<()>,
    free: SyncSender<usize>,
    jobs: Receiver<Result<RingGroup, String>>,
}

/// A bounded pinned ring plus its one reader thread.
pub struct ProjectedRing {
    handle: *mut Raw,
    slots: usize,
    slot_bytes: usize,
    projection: Projection,
    reader: Option<Reader>,
    stats: RingStats,
}

impl ProjectedRing {
    /// Create the ring for `projection`.
    ///
    /// `slots` is clamped to 2..=3 and the whole ring must fit 64 MB. A single
    /// group larger than the cap is refused with its byte count: the transport
    /// never silently shrinks the projection or falls back to a full table.
    pub fn create(
        store: &FileStore,
        projection: Projection,
        slots: usize,
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

        let store = store.try_clone()?;
        let proj = projection.clone();
        let handle_addr = handle as usize;
        let (free_tx, free_rx) = sync_channel::<usize>(slots);
        let (job_tx, job_rx) = sync_channel::<Result<RingGroup, String>>(1);
        for i in 0..slots {
            let _ = free_tx.send(i);
        }
        let join = std::thread::Builder::new()
            .name("rcsr-projected-reader".to_string())
            .spawn(move || {
                for group in 0..store.n_groups() {
                    let slot = match free_rx.recv() {
                        Ok(s) => s,
                        Err(_) => return,
                    };
                    // The previous consumer of this slot may still have an
                    // in-flight H2D copy reading the pinned bytes. Wait on
                    // that slot's copy event in the reader thread, so the main
                    // thread can keep enqueuing the next group's copy.
                    let ring = handle_addr as *mut Raw;
                    if unsafe { rc_ring_wait_copy(ring, slot as c_int) } != 0 {
                        let _ = job_tx.send(Err("projected ring: copy event wait failed".to_string()));
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
                            let job = RingGroup {
                                group,
                                slot,
                                len,
                                stats,
                            };
                            if job_tx.send(Ok(job)).is_err() {
                                return;
                            }
                        }
                        Err(e) => {
                            let _ = job_tx.send(Err(e));
                            return;
                        }
                    }
                }
            })
            .map_err(|e| format!("projected ring: reader thread: {}", e))?;

        Ok(ProjectedRing {
            handle,
            slots,
            slot_bytes,
            projection,
            reader: Some(Reader {
                join,
                free: free_tx,
                jobs: job_rx,
            }),
            stats: RingStats::default(),
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

    /// Constant pinned byte count for the life of the ring.
    pub fn ring_bytes(&self) -> u64 {
        (self.slots as u64).saturating_mul(self.slot_bytes as u64)
    }

    /// The next group the reader has filled, or `None` at end of store.
    pub fn next_group(&mut self) -> Option<Result<RingGroup, String>> {
        let reader = self.reader.as_ref()?;
        match reader.jobs.recv() {
            Ok(Ok(g)) => {
                self.stats.groups += 1;
                self.stats.bytes_read = self.stats.bytes_read.saturating_add(g.stats.value_bytes);
                Some(Ok(g))
            }
            Ok(Err(e)) => Some(Err(e)),
            Err(_) => None,
        }
    }

    /// Enqueue the asynchronous upload of `slot` into the destination columns.
    ///
    /// `dests` is one `(device base pointer, device byte offset)` per projected
    /// column; the host offset is derived from the projection order and `len`.
    pub fn upload(
        &mut self,
        group: &RingGroup,
        dests: &[(*mut c_void, usize)],
    ) -> Result<(), String> {
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
            return Err("projected ring: cudaMemcpyAsync failed".to_string());
        }
        // The reader waits on this slot's copy event before overwriting the
        // pinned bytes; the main thread does not block here.
        if let Some(reader) = self.reader.as_ref() {
            let _ = reader.free.send(group.slot);
        }
        Ok(())
    }

    /// Make the compute stream wait for this slot's copy.
    pub fn compute_begin(&mut self, slot: usize) -> Result<(), String> {
        if unsafe { rc_ring_compute_begin(self.handle, slot as c_int) } != 0 {
            self.stats.failed = true;
            return Err("projected ring: compute stream wait failed".to_string());
        }
        Ok(())
    }

    /// Record that the compute stream finished with this slot.
    pub fn compute_end(&mut self, slot: usize) -> Result<(), String> {
        if unsafe { rc_ring_compute_end(self.handle, slot as c_int) } != 0 {
            self.stats.failed = true;
            return Err("projected ring: compute event record failed".to_string());
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
            return Err("projected ring: segment-id kernel failed".to_string());
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
            return Err("projected ring: tiled feature kernel failed".to_string());
        }
        Ok(())
    }

    /// Final join: one stream synchronisation after every group.
    pub fn finish(&mut self) -> Result<(), String> {
        // Drop the free sender so the reader exits after its last group.
        if let Some(reader) = self.reader.as_mut() {
            let (dead_tx, _dead_rx) = sync_channel(0);
            let old = std::mem::replace(&mut reader.free, dead_tx);
            drop(old);
        }
        let reader = self.reader.take();
        if let Some(reader) = reader {
            // Drain any group the reader managed to fill after the caller's
            // last `next_group`, so it cannot block on a full job channel.
            while reader.jobs.try_recv().is_ok() {}
            let _ = reader.join.join();
        }
        let rc = unsafe { rc_ring_sync(self.handle) };
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
        if let Some(mut reader) = self.reader.take() {
            let (dead_tx, _dead_rx) = sync_channel(0);
            let old = std::mem::replace(&mut reader.free, dead_tx);
            drop(old);
            while reader.jobs.try_recv().is_ok() {}
            let _ = reader.join.join();
        }
        if !self.handle.is_null() {
            unsafe { rc_ring_destroy(self.handle) };
            self.handle = std::ptr::null_mut();
        }
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
