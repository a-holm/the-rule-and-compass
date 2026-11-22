// Streaming SHA-256. Included by `manifest.rs` and by `build.rs` (the build
// source digest), so the build script needs no dependency.

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn round(a: u32, b: u32, c: u32, d: &mut u32, e: u32, f: u32, g: u32, h: &mut u32, k: u32, w: u32) {
    let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
    let ch = (e & f) ^ (!e & g);
    let t1 = h
        .wrapping_add(s1)
        .wrapping_add(ch)
        .wrapping_add(k)
        .wrapping_add(w);
    let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
    let maj = (a & b) ^ (a & c) ^ (b & c);
    *d = d.wrapping_add(t1);
    *h = t1.wrapping_add(s0.wrapping_add(maj));
}

/// Rolling 16-word message schedule: `w[i % 16]` becomes `W[i]` for `i >= 16`.
#[inline(always)]
fn schedule(w: &mut [u32; 16], i: usize) -> u32 {
    let w15 = w[(i + 1) % 16];
    let w2 = w[(i + 14) % 16];
    let s0 = w15.rotate_right(7) ^ w15.rotate_right(18) ^ (w15 >> 3);
    let s1 = w2.rotate_right(17) ^ w2.rotate_right(19) ^ (w2 >> 10);
    w[i % 16] = w[i % 16]
        .wrapping_add(s0)
        .wrapping_add(w[(i + 9) % 16])
        .wrapping_add(s1);
    w[i % 16]
}

/// Streaming SHA-256 state. The file and in-memory entry points share it, so a
/// multi-gigabyte store is hashed without reading or copying it into RAM.
struct Sha256 {
    h: [u32; 8],
    buf: [u8; 64],
    buf_len: usize,
    total_len: u64,
}

impl Sha256 {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    fn new() -> Self {
        Sha256 {
            h: [
                0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
            ],
            buf: [0u8; 64],
            buf_len: 0,
            total_len: 0,
        }
    }

    fn update(&mut self, mut data: &[u8]) {
        self.total_len = self.total_len.wrapping_add(data.len() as u64);
        if self.buf_len > 0 {
            let take = (64 - self.buf_len).min(data.len());
            self.buf[self.buf_len..self.buf_len + take].copy_from_slice(&data[..take]);
            self.buf_len += take;
            data = &data[take..];
            if self.buf_len == 64 {
                let chunk = self.buf;
                self.compress(&chunk);
                self.buf_len = 0;
            }
        }
        let mut blocks = data.chunks_exact(64);
        for block in &mut blocks {
            let block: &[u8; 64] = block.try_into().expect("64-byte block");
            self.compress(block);
        }
        let rest = blocks.remainder();
        if !rest.is_empty() {
            self.buf[..rest.len()].copy_from_slice(rest);
            self.buf_len = rest.len();
        }
    }

    fn compress(&mut self, chunk: &[u8; 64]) {
        let mut w = [0u32; 16];
        for (word, bytes) in w.iter_mut().zip(chunk.chunks_exact(4)) {
            *word = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = self.h;
        macro_rules! r8 {
            ($i:expr, $wf:expr) => {{
                round(a, b, c, &mut d, e, f, g, &mut h, Self::K[$i], $wf($i));
                round(h, a, b, &mut c, d, e, f, &mut g, Self::K[$i + 1], $wf($i + 1));
                round(g, h, a, &mut b, c, d, e, &mut f, Self::K[$i + 2], $wf($i + 2));
                round(f, g, h, &mut a, b, c, d, &mut e, Self::K[$i + 3], $wf($i + 3));
                round(e, f, g, &mut h, a, b, c, &mut d, Self::K[$i + 4], $wf($i + 4));
                round(d, e, f, &mut g, h, a, b, &mut c, Self::K[$i + 5], $wf($i + 5));
                round(c, d, e, &mut f, g, h, a, &mut b, Self::K[$i + 6], $wf($i + 6));
                round(b, c, d, &mut e, f, g, h, &mut a, Self::K[$i + 7], $wf($i + 7));
            }};
        }
        r8!(0, |i: usize| w[i]);
        r8!(8, |i: usize| w[i]);
        r8!(16, |i: usize| schedule(&mut w, i));
        r8!(24, |i: usize| schedule(&mut w, i));
        r8!(32, |i: usize| schedule(&mut w, i));
        r8!(40, |i: usize| schedule(&mut w, i));
        r8!(48, |i: usize| schedule(&mut w, i));
        r8!(56, |i: usize| schedule(&mut w, i));
        for (state, v) in self.h.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *state = state.wrapping_add(v);
        }
    }

    fn finalize(mut self) -> String {
        let bits = self.total_len.wrapping_mul(8);
        let rem = self.buf_len;
        let pad_len = if rem < 56 { 56 - rem } else { 120 - rem };
        let mut pad = [0u8; 64];
        pad[0] = 0x80;
        self.update(&pad[..pad_len]);
        self.update(&bits.to_be_bytes());
        let mut out = String::with_capacity(64);
        for word in self.h {
            out.push_str(&format!("{:08x}", word));
        }
        out
    }
}
