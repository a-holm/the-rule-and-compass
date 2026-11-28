//! Input and output vectors of the functions that have a Lean model under `formal/`.
//!
//! Each generator calls the real library function. `tests/formal_sync.rs` compares
//! [`golden`] with the committed files under `formal/vectors/golden/`, and the Lean
//! program `rcsr_model_check` replays the same files against the Lean model.
//! The text format is `rcsr-vec 1`, described in `docs/formal-verification.md`.

use std::cmp::Ordering;
use std::fmt::Write;

/// Lines of a committed golden file.
pub const GOLDEN_COUNT: usize = 4096;
/// Seed of the committed golden files.
pub const GOLDEN_SEED: u64 = 0x5eed_0001;
/// Models with a generator, in file order.
pub const MODELS: &[&str] = &["cmp_error", "ulps_step"];

const SIGN: u64 = 0x8000_0000_0000_0000;
const TOP: u64 = 0x7ff0_0000_0000_0000;

struct SplitMix64(u64);

impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn pick(&mut self, items: &[u64]) -> u64 {
        items[(self.next() % items.len() as u64) as usize]
    }
}

/// Bit patterns every float input is tried on: zeros, subnormal edges, 1, `MAX`, infinities,
/// quiet, signalling and all-ones NaN, each with both signs.
pub fn base_values() -> Vec<u64> {
    let magnitudes: [u64; 11] = [
        0,
        1,
        0x000f_ffff_ffff_ffff,
        0x0010_0000_0000_0000,
        0x3ff0_0000_0000_0000,
        0x7fef_ffff_ffff_ffff,
        TOP,
        TOP + 1,
        0x7ff8_0000_0000_0000,
        0x7ff8_0000_0000_0001,
        0x7fff_ffff_ffff_ffff,
    ];
    let mut out = Vec::new();
    for m in magnitudes {
        out.push(m);
        out.push(m | SIGN);
    }
    out
}

/// [`base_values`] plus the 64 neighbours of zero and of `+inf` on each side, with both signs.
pub fn directed_values() -> Vec<u64> {
    let mut out = base_values();
    for d in 1..=64u64 {
        for m in [d, TOP - d, TOP + d] {
            out.push(m);
            out.push(m | SIGN);
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn ordering_code(o: Ordering) -> u64 {
    match o {
        Ordering::Less => 0,
        Ordering::Equal => 1,
        Ordering::Greater => 2,
    }
}

fn cmp_error_row(a: u64, b: u64) -> [u64; 3] {
    [a, b, ordering_code(crate::error_order::cmp_error(f64::from_bits(a), f64::from_bits(b)))]
}

fn cmp_error_rows(count: usize, seed: u64) -> Vec<[u64; 3]> {
    let base = base_values();
    let directed = directed_values();
    let mut rows = Vec::new();
    for &a in &base {
        for &b in &base {
            rows.push(cmp_error_row(a, b));
        }
    }
    for &v in &directed {
        for anchor in [0, TOP] {
            rows.push(cmp_error_row(v, anchor));
            rows.push(cmp_error_row(anchor, v));
        }
    }
    let mut rng = SplitMix64(seed);
    while rows.len() < count {
        let a = if rng.next() % 3 == 0 { rng.pick(&directed) } else { rng.next() };
        let b = match rng.next() % 8 {
            0 => a ^ (1 << (rng.next() % 64)),
            1 | 2 => rng.pick(&directed),
            _ => rng.next(),
        };
        rows.push(cmp_error_row(a, b));
    }
    rows
}

/// Steps every `ulps_step` input is tried with.
pub fn directed_steps() -> Vec<i64> {
    let top = TOP as i64;
    vec![
        0,
        1,
        -1,
        2,
        -2,
        63,
        -63,
        64,
        -64,
        65,
        -65,
        i32::MAX as i64,
        i32::MIN as i64,
        i64::MAX,
        i64::MIN,
        i64::MIN + 1,
        top,
        -top,
        top - 1,
        1 << 62,
        -(1 << 62),
    ]
}

fn ulps_step_row(v: u64, k: i64) -> [u64; 3] {
    [v, k as u64, crate::flat_bound::rows::ulps_step(f64::from_bits(v), k).to_bits()]
}

fn ulps_step_rows(count: usize, seed: u64) -> Vec<[u64; 3]> {
    let base = base_values();
    let directed = directed_values();
    let steps = directed_steps();
    let mut rows = Vec::new();
    for &v in &base {
        for &k in &steps {
            rows.push(ulps_step_row(v, k));
        }
    }
    for &v in &directed {
        for k in [64, -64, 1, -1] {
            rows.push(ulps_step_row(v, k));
        }
    }
    let mut rng = SplitMix64(seed);
    while rows.len() < count {
        let v = if rng.next() % 4 == 0 { rng.pick(&directed) } else { rng.next() };
        let k = match rng.next() % 4 {
            0 => steps[(rng.next() % steps.len() as u64) as usize],
            1 => (rng.next() % 257) as i64 - 128,
            2 => (rng.next() as i64) >> (rng.next() % 64),
            _ => rng.next() as i64,
        };
        rows.push(ulps_step_row(v, k));
    }
    rows
}

/// Rows a file of `model` always holds, whatever the count asked for.
pub fn directed_len(model: &str) -> Option<usize> {
    let (b, d) = (base_values().len(), directed_values().len());
    match model {
        "cmp_error" => Some(b * b + 4 * d),
        "ulps_step" => Some(b * directed_steps().len() + 4 * d),
        _ => None,
    }
}

/// The text of a vector file with exactly `count` vector lines. `None` for an unknown model
/// or a `count` below [`directed_len`].
pub fn vectors(model: &str, count: usize, seed: u64) -> Option<String> {
    if count < directed_len(model)? {
        return None;
    }
    let rows = match model {
        "cmp_error" => cmp_error_rows(count, seed),
        "ulps_step" => ulps_step_rows(count, seed),
        _ => return None,
    };
    let mut text = format!("rcsr-vec 1 {model} sig=w,w->w\n");
    for r in &rows {
        writeln!(text, "{:016x} {:016x} {:016x}", r[0], r[1], r[2]).expect("write to a String");
    }
    writeln!(text, "end {}", rows.len()).expect("write to a String");
    Some(text)
}

/// The text of the committed golden file of `model`.
pub fn golden(model: &str) -> Option<String> {
    vectors(model, GOLDEN_COUNT, GOLDEN_SEED)
}
