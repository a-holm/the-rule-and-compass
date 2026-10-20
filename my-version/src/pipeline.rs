//! #119: a bounded, ordered job queue for overlapping production and scoring.
//!
//! The search's CPU side (variation, simplification, fitting, de-duplication)
//! and the GPU evaluator are independent: a candidate is scored from its tree
//! alone, and selection only reads the completed errors at the generation
//! barrier. This module supplies the bounded channel between them. Producers
//! send candidate jobs in their own fixed order; the consumer takes batches of
//! at most `batch` jobs, scores them, and keeps draining even after a scoring
//! error so a producer can never block on a full queue.
//!
//! The memory bound is `capacity + batch` live jobs: `capacity` queued in the
//! channel plus at most `batch` in the consumer's hands. Nothing here can grow
//! with the population or the generation count.

use std::sync::mpsc::{sync_channel, Receiver, SyncSender};

/// #119: how one generation is split across the overlap queue.
///
/// `depth` chunks are produced; the scorer consumes them in `batch`-sized
/// pieces. The queue never holds more than `capacity` jobs, which is at most
/// one whole generation (and exactly `depth * batch` when `pop` is a multiple
/// of `depth`). Sizing is a pure function of `(pop, depth)` so the memory bound
/// is testable without a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkPlan {
    /// Candidates per scoring call.
    pub batch: usize,
    /// Jobs the channel may hold before a producer blocks.
    pub capacity: usize,
}

/// Split `pop` candidates into `depth` chunks.
///
/// `depth = 0` is never passed here (the caller disables the queue), but is
/// treated as one chunk for safety. A generation smaller than `depth` gets
/// one-candidate chunks rather than a zero batch, and the channel is capped at
/// the generation size so an idle queue cannot reserve more slots than the
/// search can ever produce.
pub fn chunk_plan(pop: usize, depth: usize) -> ChunkPlan {
    let depth = depth.max(1);
    let batch = pop.div_ceil(depth).max(1);
    ChunkPlan {
        batch,
        capacity: depth.saturating_mul(batch).min(pop.max(1)),
    }
}

/// A bounded multi-producer, single-consumer job queue.
pub struct JobQueue<T> {
    tx: SyncSender<T>,
    rx: Receiver<T>,
}

impl<T> JobQueue<T> {
    /// A queue that holds at most `capacity` jobs (clamped to at least one).
    pub fn new(capacity: usize) -> Self {
        let (tx, rx) = sync_channel(capacity.max(1));
        JobQueue { tx, rx }
    }

    /// A handle for one producer. Every handle delivers its own jobs in the
    /// order that producer sent them (the channel is FIFO).
    pub fn sender(&self) -> SyncSender<T> {
        self.tx.clone()
    }

    /// Drain the queue in batches, calling `consume` with at most `batch` jobs
    /// at a time. Returns the first consumer error, if any. The loop always
    /// runs until every sender is dropped; after an error it discards the rest
    /// instead of scoring it, but still drains so no producer can block.
    pub fn drain<F>(self, batch: usize, mut consume: F) -> Option<String>
    where
        F: FnMut(Vec<T>) -> Result<(), String>,
    {
        // Drop the queue's own sender immediately: otherwise the receiver
        // never disconnects after the producers finish and `recv` blocks
        // forever.
        let JobQueue { tx, rx } = self;
        drop(tx);
        let batch = batch.max(1);
        let mut buf: Vec<T> = Vec::with_capacity(batch);
        let mut error: Option<String> = None;
        loop {
            match rx.recv() {
                Ok(job) => {
                    buf.push(job);
                    if buf.len() >= batch {
                        let jobs = std::mem::replace(&mut buf, Vec::with_capacity(batch));
                        if error.is_none() {
                            if let Err(e) = consume(jobs) {
                                error = Some(e);
                            }
                        }
                    }
                }
                Err(_) => {
                    if !buf.is_empty() && error.is_none() {
                        if let Err(e) = consume(buf) {
                            error = Some(e);
                        }
                    }
                    break;
                }
            }
        }
        error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;
    use std::time::Duration;

    /// A fast producer and a deliberately slow consumer: every job is delivered
    /// in FIFO order, and the number of live jobs never exceeds capacity +
    /// batch. The consumer is slow enough that the producer would run ahead
    /// without the bound.
    #[test]
    fn slow_consumer_keeps_order_and_the_memory_bound() {
        const JOBS: usize = 200;
        const CAPACITY: usize = 4;
        const BATCH: usize = 3;

        let queue = JobQueue::new(CAPACITY);
        let sender = queue.sender();
        let live = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));

        let producer = {
            let live = Arc::clone(&live);
            let peak = Arc::clone(&peak);
            std::thread::spawn(move || {
                for i in 0..JOBS {
                    let now = live.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    sender.send(i).expect("consumer is alive");
                }
            })
        };

        let mut seen = Vec::new();
        let error = queue.drain(BATCH, |jobs| {
            assert!(jobs.len() <= BATCH, "batch of {} jobs", jobs.len());
            // A slow scorer: this is what gives the producer a chance to run
            // ahead, so the bound is exercised rather than incidental.
            std::thread::sleep(Duration::from_micros(200));
            for j in jobs {
                seen.push(j);
                live.fetch_sub(1, Ordering::SeqCst);
            }
            Ok(())
        });
        producer.join().expect("producer");
        assert!(error.is_none());

        assert_eq!(seen, (0..JOBS).collect::<Vec<_>>(), "FIFO order was lost");
        assert_eq!(live.load(Ordering::SeqCst), 0);
        // `capacity` queued + `batch` held by the consumer + one job in a
        // producer's hand, incremented just before a `send` that may block.
        assert!(
            peak.load(Ordering::SeqCst) <= CAPACITY + BATCH + 1,
            "peak live jobs {} exceeded capacity + batch + 1 {}",
            peak.load(Ordering::SeqCst),
            CAPACITY + BATCH + 1
        );
        assert!(
            peak.load(Ordering::SeqCst) > CAPACITY,
            "the producer never ran ahead; the test did not exercise the bound"
        );
    }

    /// A consumer error is returned to the caller and every producer still
    /// finishes: no deadlock, and the remaining jobs are discarded.
    #[test]
    fn consumer_error_propagates_without_blocking_the_producer() {
        let queue = JobQueue::new(2);
        let sender = queue.sender();
        let jobs = 50usize;
        let producer = std::thread::spawn(move || {
            for i in 0..jobs {
                sender.send(i).expect("consumer is draining");
            }
        });
        let mut calls = 0;
        let error = queue.drain(5, |_jobs| {
            calls += 1;
            Err("slow scorer failed".to_string())
        });
        producer.join().expect("producer finishes");
        assert_eq!(error.as_deref(), Some("slow scorer failed"));
        assert_eq!(calls, 1, "the consumer must stop after the first error");
    }

    /// #119: the chunk plan is total, covers the population and never reserves
    /// more slots than a generation can fill. The batch shrinks as the depth
    /// grows, which is what lets the scorer start before the generation ends.
    #[test]
    fn chunk_plan_is_bounded_and_covers_the_population() {
        for pop in [1usize, 3, 24, 400, 2000] {
            for depth in 1..=64 {
                let plan = chunk_plan(pop, depth);
                assert!(plan.batch >= 1, "pop {pop} depth {depth}");
                assert!(
                    plan.capacity >= plan.batch,
                    "a full batch must fit: pop {pop} depth {depth}"
                );
                assert!(
                    plan.capacity <= pop.max(1),
                    "the queue must not exceed one generation: pop {pop} depth {depth}"
                );
                assert!(
                    depth * plan.batch >= pop,
                    "depth chunks must cover the population: pop {pop} depth {depth}"
                );
            }
        }
        // The default: two chunks of 200 for pop 400.
        assert_eq!(
            chunk_plan(400, 2),
            ChunkPlan {
                batch: 200,
                capacity: 400
            }
        );
        // A deep queue trades launch size for an earlier start.
        assert_eq!(
            chunk_plan(400, 8),
            ChunkPlan {
                batch: 50,
                capacity: 400
            }
        );
        // A generation smaller than the requested depth still makes progress.
        assert_eq!(
            chunk_plan(3, 64),
            ChunkPlan {
                batch: 1,
                capacity: 3
            }
        );
    }

    /// #119: several producers (one per island) each keep their own send order
    /// even though the merged arrival order is a scheduling artefact. Results
    /// are keyed by `(island, slot)`, so only per-producer order matters.
    #[test]
    fn every_producer_keeps_its_own_order() {
        const PRODUCERS: usize = 4;
        const PER: usize = 25;
        let queue = JobQueue::new(8);
        let producers: Vec<_> = (0..PRODUCERS)
            .map(|p| {
                let tx = queue.sender();
                std::thread::spawn(move || {
                    for slot in 0..PER {
                        tx.send((p, slot)).expect("consumer is draining");
                    }
                })
            })
            .collect();
        let mut last = [None; PRODUCERS];
        let error = queue.drain(3, |jobs| {
            for (p, slot) in jobs {
                if let Some(prev) = last[p] {
                    assert_eq!(slot, prev + 1, "producer {p} was reordered");
                }
                last[p] = Some(slot);
            }
            Ok(())
        });
        for p in producers {
            p.join().expect("producer");
        }
        assert!(error.is_none());
        assert_eq!(last, [Some(PER - 1); PRODUCERS]);
    }
}
