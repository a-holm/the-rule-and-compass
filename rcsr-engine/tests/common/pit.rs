//! Shared fixtures for the #302 PIT as-of join tests (CPU and device).
#![allow(dead_code)]

use rcsr::panel_pit::{PitBlock, PitOutput, PitSegment};

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// Random membership-style block: `nseg` symbols, 15 m rows with outages and
/// equal-time ties, sparse decisions with `member` flips and migration chain
/// changes.
pub fn membership_block(seed: u64, nseg: usize, rows_per_seg: usize) -> PitBlock {
    let mut rng = Lcg(seed);
    let mut b = PitBlock::default();
    for _ in 0..nseg {
        let start = b.times.len() as u32;
        let leaf_start = b.keys.len() as u32;
        let t0 = 1_600_000_000i64 + (rng.below(50) as i64) * 900;
        let mut t = t0;
        let mut chain = 0i64;
        for _ in 0..rows_per_seg {
            let r = rng.below(100);
            if r < 8 {
                t += 900 * (1 + rng.below(40) as i64);
            } else if r >= 97 {
                // equal-time tie
            } else {
                t += 900;
            }
            if rng.below(2000) == 0 {
                chain += 1;
            }
            b.times.push(t);
            b.chain.push(chain);
        }
        let end = *b.times.last().unwrap();
        let mut k = t0 - 3 * 86_400 + (rng.below(5) as i64) * 3600;
        let mut member = 1u8;
        while k < end + 86_400 {
            // Some keys land exactly on a row time to exercise `<=`.
            let key = if rng.below(4) == 0 {
                let idx = start as usize + rng.below(rows_per_seg as u64) as usize;
                b.times[idx].max(k)
            } else {
                k
            };
            if b.keys.len() as u32 > leaf_start && key < *b.keys.last().unwrap() {
                k += 86_400;
                continue;
            }
            if rng.below(3) == 0 {
                member ^= 1;
            }
            b.keys.push(key);
            b.member.push(member);
            k += 86_400 * (1 + rng.below(6) as i64);
        }
        b.segments.push(PitSegment {
            start,
            len: rows_per_seg as u32,
            leaf_start,
            leaf_len: b.keys.len() as u32 - leaf_start,
        });
    }
    b
}

/// The sub-block with rows `ts <= cut` and decisions not yet visible
/// (`key > cut`) removed. The leaves removed lie in the future of every
/// remaining row.
pub fn prefix(block: &PitBlock, cut: i64) -> PitBlock {
    let mut out = PitBlock::default();
    for s in &block.segments {
        let start = out.times.len() as u32;
        let leaf_start = out.keys.len() as u32;
        for r in s.start as usize..(s.start + s.len) as usize {
            if block.times[r] <= cut {
                out.times.push(block.times[r]);
                out.chain.push(block.chain[r]);
            }
        }
        for j in s.leaf_start as usize..(s.leaf_start + s.leaf_len) as usize {
            if block.keys[j] <= cut {
                out.keys.push(block.keys[j]);
                out.member.push(block.member[j]);
            }
        }
        out.segments.push(PitSegment {
            start,
            len: out.times.len() as u32 - start,
            leaf_start,
            leaf_len: out.keys.len() as u32 - leaf_start,
        });
    }
    out
}

/// Bit-level comparison of one prefix result against the full result
/// truncated to the same rows. Leaf indices are compared through their keys.
pub fn assert_prefix_equal(
    label: &str,
    full_block: &PitBlock,
    full: &PitOutput,
    pre_block: &PitBlock,
    pre: &PitOutput,
) {
    let mut p = 0usize;
    for s in &full_block.segments {
        let cut_len = pre_block
            .segments
            .iter()
            .nth(full_block.segments.iter().position(|x| x == s).unwrap())
            .unwrap()
            .len as usize;
        for off in 0..cut_len {
            let fr = s.start as usize + off;
            assert_eq!(full_block.times[fr], pre_block.times[p], "{label}: time at {p}");
            let fk = (full.leaf[fr] >= 0).then(|| full_block.keys[full.leaf[fr] as usize]);
            let pk = (pre.leaf[p] >= 0).then(|| pre_block.keys[pre.leaf[p] as usize]);
            assert_eq!(fk, pk, "{label}: visible key at row {p}");
            assert_eq!(full.keep[fr], pre.keep[p], "{label}: keep at {p}");
            assert_eq!(full.as_of[fr], pre.as_of[p], "{label}: as_of at {p}");
            assert_eq!(full.gap[fr], pre.gap[p], "{label}: gap at {p}");
            assert_eq!(full.chain_id[fr], pre.chain_id[p], "{label}: chain_id at {p}");
            p += 1;
        }
    }
    assert_eq!(p, pre_block.times.len(), "{label}: prefix row count");
}

pub fn assert_same(label: &str, a: &PitOutput, b: &PitOutput) {
    assert_eq!(a.leaf, b.leaf, "{label}: leaf");
    assert_eq!(a.keep, b.keep, "{label}: keep");
    assert_eq!(a.found, b.found, "{label}: found");
    assert_eq!(a.as_of, b.as_of, "{label}: as_of");
    assert_eq!(a.gap, b.gap, "{label}: gap");
    assert_eq!(a.chain_id, b.chain_id, "{label}: chain_id");
    assert_eq!(a.kept_rows, b.kept_rows, "{label}: kept_rows");
    assert_eq!(a.lookahead_violations, b.lookahead_violations, "{label}: violations");
}

/// FNV-1a over the canonical little-endian bytes of every output column.
pub fn digest(o: &PitOutput) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut eat = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    };
    for v in &o.leaf {
        eat(&v.to_le_bytes());
    }
    eat(&o.keep);
    eat(&o.found);
    for v in &o.as_of {
        eat(&v.to_le_bytes());
    }
    for v in &o.gap {
        eat(&v.to_le_bytes());
    }
    for v in &o.chain_id {
        eat(&v.to_le_bytes());
    }
    for v in &o.kept_rows {
        eat(&v.to_le_bytes());
    }
    h
}
