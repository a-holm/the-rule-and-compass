//! MT19937 (32-bit Mersenne Twister) with the standard std::mt19937 parameters.
//!
//! A deterministic RNG so that runs are exactly reproducible from a seed.

#[derive(Clone)]
pub struct Mt {
    mt: [u32; 624],
    idx: usize,
}

impl Mt {
    pub fn new(seed: u32) -> Self {
        let mut mt = [0u32; 624];
        mt[0] = seed;
        for i in 1..624 {
            mt[i] = 1812433253u32
                .wrapping_mul(mt[i - 1] ^ (mt[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Mt { mt, idx: 624 }
    }

    /// The exact MT19937 state: the 624-word state array and the next index.
    ///
    /// This is the entire state of the generator, so a value obtained here and
    /// passed to [`Mt::from_state`] reproduces the stream bit for bit, from the
    /// next draw onwards. Used by the checkpoint format to resume a search.
    pub fn state(&self) -> (&[u32; 624], usize) {
        (&self.mt, self.idx)
    }

    /// Rebuild a generator from an exact MT19937 state.
    ///
    /// `idx` is the index of the next word to temper, in `0..=624`; `624` means
    /// the state array is due to be twisted before the next draw (the value
    /// [`Mt::new`] starts with). A restored generator produces the identical
    /// stream to the one it was copied from.
    pub fn from_state(mt: [u32; 624], idx: usize) -> Self {
        Mt { mt, idx }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 624 {
            for i in 0..624 {
                let y = (self.mt[i] & 0x8000_0000) | (self.mt[(i + 1) % 624] & 0x7fff_ffff);
                self.mt[i] = self.mt[(i + 397) % 624]
                    ^ (y >> 1)
                    ^ if y & 1 != 0 { 0x9908_b0df } else { 0 };
            }
            self.idx = 0;
        }
        let mut y = self.mt[self.idx];
        self.idx += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }

    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u32() as f64) / 4_294_967_296.0
    }

    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }

    pub fn usize_below(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u32() as usize) % n
        }
    }

    pub fn pick<'a, T>(&mut self, s: &'a [T]) -> &'a T {
        let i = self.usize_below(s.len());
        &s[i]
    }

    pub fn bool(&mut self, p: f64) -> bool {
        self.next_f64() < p
    }
}
