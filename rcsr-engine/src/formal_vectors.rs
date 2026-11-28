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
pub const MODELS: &[&str] = &["cmp_error", "f64_add", "f64_sub"];

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

/// A NaN result is written as this pattern: Rust does not fix the sign or the payload of a NaN result.
const QNAN: u64 = 0x7ff8_0000_0000_0000;

fn canon(v: f64) -> u64 {
    if v.is_nan() {
        QNAN
    } else {
        v.to_bits()
    }
}

fn f64_add_row(a: u64, b: u64) -> [u64; 3] {
    [a, b, canon(f64::from_bits(a) + f64::from_bits(b))]
}

fn f64_sub_row(a: u64, b: u64) -> [u64; 3] {
    [a, b, canon(f64::from_bits(a) - f64::from_bits(b))]
}

/// Second operands that make alignment shifts, cancellation and rounding ties likely for `a`.
fn near_operand(rng: &mut SplitMix64, a: u64, directed: &[u64]) -> u64 {
    match rng.next() % 8 {
        0 | 1 | 2 => {
            let ea = (a >> 52) & 0x7ff;
            let eb = (ea + rng.next() % 120 + 2047 - 60) % 2047;
            (rng.next() & 0x800f_ffff_ffff_ffff) | (eb << 52)
        }
        3 => a ^ (1 << (rng.next() % 64)),
        4 => a ^ SIGN,
        5 => rng.pick(directed),
        _ => rng.next(),
    }
}

fn f64_arith_rows(row: fn(u64, u64) -> [u64; 3], count: usize, seed: u64) -> Vec<[u64; 3]> {
    let base = base_values();
    let directed = directed_values();
    let mut rows = Vec::new();
    for &a in &base {
        for &b in &base {
            rows.push(row(a, b));
        }
    }
    for &v in &directed {
        for anchor in [0x3ff0_0000_0000_0000, 0x7fef_ffff_ffff_ffff] {
            rows.push(row(v, anchor));
            rows.push(row(anchor, v));
        }
    }
    let mut rng = SplitMix64(seed);
    while rows.len() < count {
        let a = if rng.next() % 4 == 0 { rng.pick(&directed) } else { rng.next() };
        let b = near_operand(&mut rng, a, &directed);
        rows.push(row(a, b));
    }
    rows
}

/// Rows a file of `model` always holds, whatever the count asked for.
pub fn directed_len(model: &str) -> Option<usize> {
    let (b, d) = (base_values().len(), directed_values().len());
    match model {
        "cmp_error" => Some(b * b + 4 * d),
        "f64_add" | "f64_sub" => Some(b * b + 4 * d),
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
        "f64_add" => f64_arith_rows(f64_add_row, count, seed),
        "f64_sub" => f64_arith_rows(f64_sub_row, count, seed),
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
