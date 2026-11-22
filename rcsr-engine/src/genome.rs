//! Fixed-capacity postfix genome pool and counter-based RNG (#122).
//!
//! The evolutionary loop's hot data are short postfix programs. Historically
//! every generation built a fresh packed `Vec<u32>` and every CUDA batch called
//! `cudaMalloc`/`cudaFree`. This module supplies the representation the device
//! path needs instead:
//!
//! * **fixed slots** — every genome occupies exactly [`width_for`] words inside
//!   an arena allocated once, so a slot never moves and no free list is needed
//!   (Nordin 1994; Langdon & Banzhaf 2008);
//! * **double buffering** — [`GenomePool::begin_generation`] writes into the
//!   back arena while the front arena still describes the generation the device
//!   is evaluating; [`GenomePool::commit`] swaps them;
//! * **SoA metadata** — length, island, tag, hash and fitness live beside the
//!   programs, not inside them;
//! * **a canonical postfix hash** — FNV-1a over the words up to and including
//!   `HALT`, mirrored bit for bit by the CUDA kernel, so a host and a device
//!   dedup pass see the same key;
//! * **Philox4x32-10** — a counter-based generator whose output is a pure
//!   function of `(key, counter)`, so a child can be regenerated from its
//!   `(seed, island, generation, slot, draw)` position (Salmon et al. 2011).
//!
//! The pool is deliberately independent of `cuda`: it is compiled (and tested)
//! in every build, and the CUDA backend merely mirrors its layout.

/// `HALT` word that terminates every program, matching `src/cuda/eval.cu`.
pub const OP_HALT: u32 = 255;

/// Fixed slot width for a configured maximum complexity.
///
/// A tree of `c` nodes compiles to at most `2*c + 2` words (each leaf is two
/// words and each operator one, plus the terminator), which is the bound the
/// GPU plan uses (`W = 2*max_complexity + 4`).
pub fn width_for(max_complexity: usize) -> usize {
    (2 * max_complexity + 4).max(8)
}

/// FNV-1a over the postfix words, including the terminating `HALT`.
///
/// This is canonical for the *program*: postfix has no ambiguity, and the
/// constants are compared by their `f32` bits exactly as the kernel evaluates
/// them. It matches `hash_kernel` in `src/cuda/eval.cu` word for word.
pub fn postfix_hash(words: &[u32]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &w in words {
        h ^= w as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
        if w == OP_HALT {
            break;
        }
    }
    h
}

/// SoA metadata for one slot.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GenomeMeta {
    /// Program length in words, including `HALT`.
    pub len: u16,
    /// Logical island that owns the genome.
    pub island: u16,
    /// `(generation << 20) | slot`, for provenance and checkpoints.
    pub tag: u32,
    /// Canonical [`postfix_hash`] of the program.
    pub hash: u64,
    /// Reserved flags: bit 0 marks a duplicate within the generation.
    pub flag: u8,
}

/// A fixed-capacity, double-buffered postfix genome arena.
///
/// At `max_complexity = 30` the width is 64 words, i.e. 256 B per genome
/// (2.6 MB per 10 000 genomes). Capacity and width are fixed at construction;
/// [`GenomePool::push`] never reallocates and rejects a program that does not
/// fit, so peak memory cannot grow across generations.
#[derive(Clone, Debug)]
pub struct GenomePool {
    cap: usize,
    width: usize,
    arenas: [Vec<u32>; 2],
    meta: Vec<GenomeMeta>,
    /// Index of the most recently committed (readable) arena.
    front: usize,
    /// Index of the arena currently being written.
    back: usize,
    len: usize,
}

impl GenomePool {
    /// Allocate both arenas once. Capacity and width are immutable.
    pub fn new(cap: usize, width: usize) -> Self {
        assert!(cap > 0, "a genome pool needs at least one slot");
        assert!(width >= 2, "a genome slot must hold a program and HALT");
        GenomePool {
            cap,
            width,
            arenas: [vec![0; cap * width], vec![0; cap * width]],
            meta: vec![GenomeMeta::default(); cap],
            front: 1,
            back: 0,
            len: 0,
        }
    }

    /// Start filling the back arena. The front arena is untouched, so a
    /// consumer may still read the previous generation while this one is built.
    pub fn begin_generation(&mut self) {
        self.back = self.front ^ 1;
        self.len = 0;
    }

    /// Make the back arena the readable front and return its genome count.
    pub fn commit(&mut self) -> usize {
        self.front = self.back;
        self.len
    }

    /// The arena that is currently being written (or was just written).
    pub fn active(&self) -> &[u32] {
        &self.arenas[self.back]
    }

    /// The arena from the previously committed generation.
    pub fn previous(&self) -> &[u32] {
        &self.arenas[self.front]
    }

    pub fn capacity(&self) -> usize {
        self.cap
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Words of host memory reserved by both arenas.
    pub fn bytes(&self) -> usize {
        2 * self.cap * self.width * std::mem::size_of::<u32>()
            + self.cap * std::mem::size_of::<GenomeMeta>()
    }

    pub fn meta(&self) -> &[GenomeMeta] {
        &self.meta[..self.len]
    }

    pub fn meta_mut(&mut self, slot: usize) -> &mut GenomeMeta {
        &mut self.meta[slot]
    }

    /// The committed program in `slot`, `HALT`-terminated.
    pub fn program(&self, slot: usize) -> &[u32] {
        assert!(slot < self.len, "slot {} is outside the pool", slot);
        let base = slot * self.width;
        // The slot is HALT-padded, so this slice is the whole fixed slot.
        &self.arenas[self.back][base..base + self.width]
    }

    /// Copy `words` into the next fixed slot, `HALT`-padded. Returns `false`
    /// (and changes nothing) when the pool is full or the program is too wide.
    pub fn push(&mut self, words: &[u32], meta: GenomeMeta) -> bool {
        if self.len >= self.cap || words.len() > self.width {
            return false;
        }
        let slot = self.len;
        let base = slot * self.width;
        let arena = &mut self.arenas[self.back];
        arena[base..base + self.width].fill(OP_HALT);
        arena[base..base + words.len()].copy_from_slice(words);
        let mut meta = meta;
        meta.len = words.len() as u16;
        meta.hash = postfix_hash(&arena[base..base + words.len()]);
        self.meta[slot] = meta;
        self.len += 1;
        true
    }
}

/// The canonical `(hash, slot)` order used by dedup and compaction.
///
/// Sorting by hash alone is not enough: equal hashes are compared by slot, so
/// the order is stable and independent of thread or block count.
pub fn canonical_order(pool: &GenomePool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..pool.len()).collect();
    order.sort_by_key(|&i| (pool.meta[i].hash, i as u64));
    order
}

/// Mark slots whose `(hash, program)` is a repeat of an earlier slot.
///
/// A hash collision is only a candidate: `push` hashed the full program, so two
/// equal hashes are compared word for word before the duplicate flag is set.
/// The first slot in slot order always survives, which makes the result
/// deterministic regardless of how a parallel scan visits slots.
pub fn mark_duplicates(pool: &mut GenomePool) {
    let order = canonical_order(pool);
    for pair in order.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if pool.meta[a].hash == pool.meta[b].hash && pool.program(a) == pool.program(b) {
            pool.meta[b].flag |= 1;
        }
    }
}

/* ------------------------------------------------------------------------- */
/* Philox4x32-10                                                             */
/* ------------------------------------------------------------------------- */

const PHILOX_M0: u32 = 0xD251_1F53;
const PHILOX_M1: u32 = 0xCD9E_8D57;
const PHILOX_W0: u32 = 0x9E37_79B9;
const PHILOX_W1: u32 = 0xBB67_AE85;

/// The high 32 bits of `a * b` (an exact 32x32→64 multiply).
fn mulhi(a: u32, b: u32) -> u32 {
    ((a as u64 * b as u64) >> 32) as u32
}

/// Philox4x32-10: a deterministic counter-based generator.
///
/// The state is only `(key, counter)`, so the `draw`-th number for a slot is
/// [`Philox::next`] at counter `(generation, slot, draw, 0)`. It is bit-for-bit
/// reproducible on the CPU and, in the plan, on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Philox {
    key: [u32; 2],
    counter: [u32; 4],
}

impl Philox {
    /// Seed the key with `(seed, island)`, as the GPU plan specifies.
    pub fn new(seed: u32, island: u32) -> Self {
        Philox {
            key: [seed, island],
            counter: [0; 4],
        }
    }

    /// A generator whose position is fixed by `(seed, island, generation, slot)`.
    pub fn at(seed: u32, island: u32, generation: u32, slot: u32) -> Self {
        Philox {
            key: [seed, island],
            counter: [generation, slot, 0, 0],
        }
    }

    pub fn key(&self) -> [u32; 2] {
        self.key
    }

    pub fn counter(&self) -> [u32; 4] {
        self.counter
    }

    /// Advance the draw index within the current slot.
    pub fn set_draw(&mut self, draw: u32) {
        self.counter[2] = draw;
    }

    /// The ten-round Philox permutation of the current counter and key.
    pub fn next(&mut self) -> [u32; 4] {
        let mut c = self.counter;
        let mut k = self.key;
        for _ in 0..10 {
            let lo0 = PHILOX_M0.wrapping_mul(c[0]);
            let hi0 = mulhi(PHILOX_M0, c[0]);
            let lo1 = PHILOX_M1.wrapping_mul(c[2]);
            let hi1 = mulhi(PHILOX_M1, c[2]);
            c = [
                hi1 ^ c[1] ^ k[0],
                lo1,
                hi0 ^ c[3] ^ k[1],
                lo0,
            ];
            k[0] = k[0].wrapping_add(PHILOX_W0);
            k[1] = k[1].wrapping_add(PHILOX_W1);
        }
        self.counter[3] = self.counter[3].wrapping_add(1);
        c
    }

    /// Uniform in `[0, 1)` from one 32-bit draw.
    pub fn unit(&mut self) -> f64 {
        (self.next()[0] as f64) / 4_294_967_296.0
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next()[0] as usize) % n
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn words(v: &[u32]) -> Vec<u32> {
        let mut out = v.to_vec();
        out.push(OP_HALT);
        out
    }

    #[test]
    fn width_is_the_documented_fixed_slot() {
        assert_eq!(width_for(30), 64);
        assert_eq!(width_for(0), 8);
    }

    #[test]
    fn pool_is_bounded_and_never_reallocates() {
        let mut pool = GenomePool::new(2, 4);
        let before = pool.active().as_ptr();
        assert!(pool.push(&words(&[1]), GenomeMeta::default()));
        assert!(pool.push(&words(&[2]), GenomeMeta::default()));
        assert!(!pool.push(&words(&[3]), GenomeMeta::default()), "capacity is fixed");
        assert_eq!(pool.len(), 2);
        // A program wider than a slot is rejected rather than truncated.
        assert!(!pool.push(&words(&[1, 2, 3, 4]), GenomeMeta::default()));
        assert_eq!(pool.active().as_ptr(), before, "the arena is never reallocated");
        assert_eq!(pool.bytes(), 2 * 2 * 4 * 4 + 2 * std::mem::size_of::<GenomeMeta>());
    }

    #[test]
    fn slots_are_halt_padded_and_independent() {
        let mut pool = GenomePool::new(3, 8);
        pool.push(&words(&[1, 2]), GenomeMeta::default());
        pool.push(&words(&[3]), GenomeMeta::default());
        assert_eq!(&pool.program(0)[..3], &[1, 2, OP_HALT]);
        assert!(pool.program(0)[3..].iter().all(|&w| w == OP_HALT));
        assert_eq!(&pool.program(1)[..2], &[3, OP_HALT]);
        assert_eq!(pool.meta()[1].len, 2);
        assert_eq!(pool.meta()[0].hash, postfix_hash(&[1, 2, OP_HALT]));
    }

    #[test]
    fn double_buffer_keeps_the_previous_generation_readable() {
        let mut pool = GenomePool::new(2, 4);
        pool.push(&words(&[10]), GenomeMeta::default());
        assert_eq!(pool.commit(), 1);
        pool.begin_generation();
        pool.push(&words(&[20]), GenomeMeta::default());
        // Before commit the front (previous) still holds generation g.
        assert_eq!(&pool.previous()[..2], &[10, OP_HALT]);
        assert_eq!(&pool.active()[..2], &[20, OP_HALT]);
        assert_eq!(pool.commit(), 1);
        assert_eq!(&pool.active()[..2], &[20, OP_HALT]);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn ten_thousand_genomes_fit_the_documented_budget() {
        let pool = GenomePool::new(10_000, width_for(30));
        // Two 2.6 MB program arenas plus 24 B of metadata per slot.
        assert_eq!(10_000 * 64 * 4, 2_560_000);
        assert_eq!(pool.bytes(), 2 * 2_560_000 + 10_000 * 24);
    }

    #[test]
    fn duplicate_marking_is_stable_and_slot_ordered() {
        let mut pool = GenomePool::new(4, 4);
        pool.push(&words(&[7, 1]), GenomeMeta::default());
        pool.push(&words(&[7, 1]), GenomeMeta::default());
        pool.push(&words(&[7, 2]), GenomeMeta::default());
        mark_duplicates(&mut pool);
        assert_eq!(pool.meta()[0].flag, 0);
        assert_eq!(pool.meta()[1].flag, 1, "the later slot is the duplicate");
        assert_eq!(pool.meta()[2].flag, 0, "a different program is not a duplicate");
        // Same result when re-run: no atomic decides the winner.
        let mut again = pool.clone();
        for m in again.meta.iter_mut() {
            m.flag = 0;
        }
        mark_duplicates(&mut again);
        assert_eq!(again.meta()[1].flag, 1);
    }

    #[test]
    fn philox_matches_the_random123_known_answer() {
        // Philox4x32-10, counter = 0, key = 0.
        let mut p = Philox::new(0, 0);
        let out = p.next();
        assert_eq!(
            out,
            [0x6627_e8d5, 0xe169_c58d, 0xbc57_ac4c, 0x9b00_dbd8]
        );
    }

    #[test]
    fn philox_is_position_determined() {
        // The draw is a pure function of (key, counter): two generators at the
        // same position agree, and island/generation/slot change the stream.
        let a = Philox::at(9, 3, 7, 11).next();
        let b = Philox::at(9, 3, 7, 11).next();
        assert_eq!(a, b);
        assert_ne!(a, Philox::at(9, 4, 7, 11).next());
        assert_ne!(a, Philox::at(9, 3, 8, 11).next());
        assert_ne!(a, Philox::at(9, 3, 7, 12).next());
        // A later draw is not the first draw.
        let mut c = Philox::at(9, 3, 7, 11);
        let first = c.next();
        let second = c.next();
        assert_ne!(first, second);
    }
}
