//! Rolling window statistics from block-local compensated prefix sums (#364).
//!
//! Every stream is a Sum2 (for `x`) or Dot2 (for `x*y`) accumulation that
//! restarts at the segment start, at every gap row and at every `BLOCK`
//! segment-relative rows. A window sum is an exact difference of two prefix
//! pairs plus the compensation, held as a double-word. Only IEEE `+ - * / sqrt`
//! are used and no fused multiply-add, so the device kernels reproduce every
//! bit. Window validity (`contig`, finite counts) is decided by the callers.

use std::cell::OnceCell;

/// Mean policies shared with `RC_WIN_MEAN_*` in `win_stats.h`.
pub const MEAN_FINITE: i32 = 0;
pub const MEAN_STRICT: i32 = 1;

/// Rows per accumulation. The device kernels use the same constant.
pub const BLOCK: usize = 1024;

#[derive(Clone, Copy, Debug)]
pub struct Dw {
    pub h: f64,
    pub l: f64,
}

const DZ: Dw = Dw { h: 0.0, l: 0.0 };

#[inline(always)]
fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    let bb = s - a;
    (s, (a - (s - bb)) + (b - bb))
}

#[inline(always)]
fn fast_two_sum(a: f64, b: f64) -> (f64, f64) {
    let s = a + b;
    (s, b - (s - a))
}

#[inline(always)]
fn split(a: f64) -> (f64, f64) {
    let t = 134_217_729.0 * a;
    let hi = t - (t - a);
    (hi, a - hi)
}

#[inline(always)]
pub fn two_prod(a: f64, b: f64) -> (f64, f64) {
    let p = a * b;
    let (ah, al) = split(a);
    let (bh, bl) = split(b);
    (p, al * bl - (((p - ah * bh) - al * bh) - ah * bl))
}

#[inline(always)]
pub fn dw_add(x: Dw, y: Dw) -> Dw {
    let (sh, sl) = two_sum(x.h, y.h);
    let (th, tl) = two_sum(x.l, y.l);
    let c = sl + th;
    let (vh, vl) = fast_two_sum(sh, c);
    let w = tl + vl;
    let (h, l) = fast_two_sum(vh, w);
    Dw { h, l }
}

#[inline(always)]
pub fn dw_neg(x: Dw) -> Dw {
    Dw { h: -x.h, l: -x.l }
}

#[inline(always)]
pub fn dw_mul_f(x: Dw, y: f64) -> Dw {
    let (ch, cl1) = two_prod(x.h, y);
    let cl2 = x.l * y;
    let (th, tl1) = fast_two_sum(ch, cl2);
    let tl2 = tl1 + cl1;
    let (h, l) = fast_two_sum(th, tl2);
    Dw { h, l }
}

#[inline(always)]
pub fn dw_mul(x: Dw, y: Dw) -> Dw {
    let (ch, cl1) = two_prod(x.h, y.h);
    let tl1 = x.h * y.l;
    let tl2 = x.l * y.h;
    let cl2 = tl1 + tl2;
    let cl3 = cl1 + cl2;
    let (h, l) = fast_two_sum(ch, cl3);
    Dw { h, l }
}

/// Double-word sum divided by an exact count, rounded once with one
/// correction step.
#[inline(always)]
pub fn div_count(s: Dw, n: f64) -> f64 {
    let q1 = s.h / n;
    let (p, e) = two_prod(q1, n);
    let r = ((s.h - p) - e) + s.l;
    q1 + r / n
}

const TAU: f64 = 8.673617379884035e-19;

/// Variance numerator, exactly zero when every finite cell of the window is
/// equal. The scan runs only when the numerator is inside the double-word
/// error bound.
#[inline(always)]
pub fn guarded(num: Dw, s2: Dw, n: f64, x: &[f64], lo: usize, hi: usize) -> f64 {
    if s2.h == 0.0 {
        return 0.0;
    }
    if num.h <= TAU * (n * s2.h) {
        let mut first = f64::NAN;
        let mut equal = true;
        for &v in &x[lo..=hi] {
            if v.is_finite() {
                if first.is_nan() {
                    first = v;
                } else if v != first {
                    equal = false;
                    break;
                }
            }
        }
        if equal {
            return 0.0;
        }
    }
    num.h
}

/// First row of the accumulation holding each row.
pub fn acc0_of(times: &[i64], step: i64) -> Vec<u32> {
    let mut a = vec![0u32; times.len()];
    for j in 0..times.len() {
        let restart = j == 0 || j % BLOCK == 0 || times[j] - times[j - 1] != step;
        a[j] = if restart { j as u32 } else { a[j - 1] };
    }
    a
}

/// Prefix counts of finite cells (`out[i + 1] - out[lo]` counts `lo..=i`).
pub fn finite_counts(x: &[f64]) -> Vec<u32> {
    let mut c = Vec::with_capacity(x.len() + 1);
    c.push(0u32);
    let mut k = 0u32;
    for &v in x {
        if v.is_finite() {
            k += 1;
        }
        c.push(k);
    }
    c
}

pub fn pair_counts(x: &[f64], y: &[f64]) -> Vec<u32> {
    let mut c = Vec::with_capacity(x.len() + 1);
    c.push(0u32);
    let mut k = 0u32;
    for (a, b) in x.iter().zip(y) {
        if a.is_finite() && b.is_finite() {
            k += 1;
        }
        c.push(k);
    }
    c
}

/// Block-local compensated running sums: `s` is the running sum, `c` the
/// running compensation.
pub struct Stream {
    s: Vec<f64>,
    c: Vec<f64>,
}

impl Stream {
    /// Sum2 of `x`.
    pub fn sum(x: &[f64], acc0: &[u32]) -> Stream {
        let n = x.len();
        let (mut ps, mut pc) = (vec![0.0; n], vec![0.0; n]);
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for j in 0..n {
            if acc0[j] as usize == j {
                s = 0.0;
                c = 0.0;
            }
            if x[j].is_finite() {
                let (t, e) = two_sum(s, x[j]);
                s = t;
                c += e;
            }
            ps[j] = s;
            pc[j] = c;
        }
        Stream { s: ps, c: pc }
    }

    /// Dot2 of `x * y`.
    pub fn dot(x: &[f64], y: &[f64], acc0: &[u32]) -> Stream {
        let n = x.len();
        let (mut ps, mut pc) = (vec![0.0; n], vec![0.0; n]);
        let (mut s, mut c) = (0.0f64, 0.0f64);
        for j in 0..n {
            if acc0[j] as usize == j {
                s = 0.0;
                c = 0.0;
            }
            if x[j].is_finite() && y[j].is_finite() {
                let (p, pe) = two_prod(x[j], y[j]);
                let (t, e) = two_sum(s, p);
                s = t;
                c += e + pe;
            }
            ps[j] = s;
            pc[j] = c;
        }
        Stream { s: ps, c: pc }
    }

    /// Sum over rows `lo..=hi` as a double-word, pieces added from `hi` back.
    #[inline(always)]
    pub fn win(&self, acc0: &[u32], lo: usize, hi: usize) -> Dw {
        let mut e = hi;
        let mut acc = DZ;
        let mut first = true;
        loop {
            let a = acc0[e] as usize;
            let (h, l) = if a >= lo {
                two_sum(self.s[e], self.c[e])
            } else {
                let (d, r) = two_sum(self.s[e], -self.s[lo - 1]);
                let t = r + (self.c[e] - self.c[lo - 1]);
                two_sum(d, t)
            };
            let piece = Dw { h, l };
            acc = if first { piece } else { dw_add(acc, piece) };
            first = false;
            if a <= lo {
                return acc;
            }
            e = a - 1;
        }
    }
}

/// Window statistics of one input array. The caller supplies the window end
/// `i` and length `w` after its own `contig` test.
pub struct WinStats<'a> {
    x: &'a [f64],
    acc0: Vec<u32>,
    cnt: Vec<u32>,
    p1: Stream,
    p2: OnceCell<Stream>,
}

impl<'a> WinStats<'a> {
    pub fn new(x: &'a [f64], times: &[i64], step: i64) -> Self {
        let acc0 = acc0_of(times, step);
        let p1 = Stream::sum(x, &acc0);
        WinStats { x, cnt: finite_counts(x), acc0, p1, p2: OnceCell::new() }
    }

    /// Finite cells in `[i + 1 - w, i]`.
    #[inline(always)]
    pub fn count(&self, i: usize, w: usize) -> usize {
        (self.cnt[i + 1] - self.cnt[i + 1 - w]) as usize
    }

    /// Mean of the finite cells, `None` when there are none.
    pub fn mean_finite(&self, i: usize, w: usize) -> Option<f64> {
        let k = self.count(i, w);
        if k == 0 {
            return None;
        }
        Some(div_count(self.p1.win(&self.acc0, i + 1 - w, i), k as f64))
    }

    /// Mean of a window whose cells are all finite.
    pub fn mean_strict(&self, i: usize, w: usize) -> Option<f64> {
        if i + 1 < w || self.count(i, w) != w {
            return None;
        }
        Some(div_count(self.p1.win(&self.acc0, i + 1 - w, i), w as f64))
    }

    /// Population standard deviation of a window whose cells are all finite.
    pub fn std_pop(&self, i: usize, w: usize) -> Option<f64> {
        if i + 1 < w || self.count(i, w) != w {
            return None;
        }
        let lo = i + 1 - w;
        let p2 = self.p2.get_or_init(|| Stream::dot(self.x, self.x, &self.acc0));
        let s1 = self.p1.win(&self.acc0, lo, i);
        let s2 = p2.win(&self.acc0, lo, i);
        let wf = w as f64;
        let num = dw_add(dw_mul_f(s2, wf), dw_neg(dw_mul(s1, s1)));
        let v = guarded(num, s2, wf, self.x, lo, i) / (wf * wf);
        Some(v.max(0.0).sqrt())
    }
}

/// `ln(1 + r)` from `+ - * /` only for `|r| <= 0.25`, so the device reproduces it
/// bit for bit: `2 atanh(z)` with `z = r / (2 + r)`. Larger arguments use the
/// platform `ln_1p`. The AR(1) coefficient behind `ou_hl96` is a difference of
/// near-equal sums whose output amplifies a last-bit logarithm difference by
/// up to 1e3, so its input must not depend on the platform logarithm.
pub fn log1p_det(r: f64) -> f64 {
    if !(r.abs() <= 0.25) {
        return r.ln_1p();
    }
    let z = r / (2.0 + r);
    let z2 = z * z;
    let mut s = 1.0 / 21.0;
    let mut k = 10;
    while k > 0 {
        k -= 1;
        s = s * z2 + 1.0 / (2 * k + 1) as f64;
    }
    (2.0 * z) * s
}

/// `cov / var` of `x` against its lag `xl` over each `w`-row window, from
/// double-word co-moments with the finite-count sets A (both finite), B (`x`
/// finite) and C (`xl` finite). `NaN` where a set is empty, the window is not
/// contiguous, or the variance is not positive.
pub fn comoment_ratio(x: &[f64], xl: &[f64], w: usize, times: &[i64], step: i64) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![f64::NAN; n];
    if w == 0 || n < w {
        return out;
    }
    let acc0 = acc0_of(times, step);
    let sx = Stream::sum(x, &acc0);
    let sl = Stream::sum(xl, &acc0);
    let sxl = Stream::dot(x, xl, &acc0);
    let sl2 = Stream::dot(xl, xl, &acc0);
    let (cb, cc, ca) = (finite_counts(x), finite_counts(xl), pair_counts(x, xl));
    for i in w - 1..n {
        if times[i + 1 - w] != times[i] - (w as i64 - 1) * step {
            continue;
        }
        let lo = i + 1 - w;
        let na = (ca[i + 1] - ca[lo]) as f64;
        let nb = (cb[i + 1] - cb[lo]) as f64;
        let nc = (cc[i + 1] - cc[lo]) as f64;
        if !(na > 0.0 && nb > 0.0 && nc > 0.0) {
            continue;
        }
        let (a, b, c, d) = (sx.win(&acc0, lo, i), sl.win(&acc0, lo, i), sxl.win(&acc0, lo, i), sl2.win(&acc0, lo, i));
        let numc = dw_add(dw_mul_f(c, nb * nc), dw_neg(dw_mul_f(dw_mul(a, b), na)));
        let cov = numc.h / (na * nb * nc);
        let numv = dw_add(dw_mul_f(d, nc), dw_neg(dw_mul(b, b)));
        let var = guarded(numv, d, nc, xl, lo, i) / (nc * nc);
        if var > 0.0 {
            out[i] = cov / var;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const STEP: i64 = 900;
    const SCALE: f64 = 1_048_576.0;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn times_with_gaps(n: usize, gaps: &[usize]) -> Vec<i64> {
        let mut t = Vec::with_capacity(n);
        let mut cur = 1_700_000_000i64;
        for j in 0..n {
            if gaps.contains(&j) {
                cur += 7 * STEP;
            }
            t.push(cur);
            cur += STEP;
        }
        t
    }

    fn contig(t: &[i64], i: usize, w: usize) -> bool {
        i + 1 >= w && t[i + 1 - w] == t[i] - (w as i64 - 1) * STEP
    }

    /// Values `k / 2^20` with `|k| < 2^52`, so every sum is exact in i128.
    fn lattice(rng: &mut Rng, n: usize, offset: i64, spread: u64, nan_every: u64) -> (Vec<f64>, Vec<i64>) {
        let mut x = Vec::with_capacity(n);
        let mut k = Vec::with_capacity(n);
        for _ in 0..n {
            if nan_every > 0 && rng.below(nan_every) == 0 {
                x.push(f64::NAN);
                k.push(0);
            } else {
                let v = offset + rng.below(spread) as i64 - (spread / 2) as i64;
                x.push(v as f64 / SCALE);
                k.push(v);
            }
        }
        (x, k)
    }

    fn ulp_diff(a: f64, b: f64) -> u64 {
        if a == b {
            return 0;
        }
        (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs()
    }

    fn check_exact(x: &[f64], k: &[i64], t: &[i64], ws: &[usize], what: &str) {
        let st = WinStats::new(x, t, STEP);
        for &w in ws {
            for i in w - 1..x.len() {
                if !contig(t, i, w) {
                    continue;
                }
                let r = i + 1 - w..=i;
                let fin: Vec<i64> = r.clone().filter(|&j| x[j].is_finite()).map(|j| k[j]).collect();
                let n = fin.len();
                let s1: i128 = fin.iter().map(|&v| v as i128).sum();
                let got = st.mean_finite(i, w);
                if n == 0 {
                    assert!(got.is_none(), "{what} w{w} i{i}");
                } else {
                    let want = (s1 as f64) / (n as f64) / SCALE;
                    assert!(
                        ulp_diff(got.unwrap(), want) <= 1,
                        "{what} mean w{w} i{i}: {} vs {}",
                        got.unwrap(),
                        want
                    );
                }
                if n == w {
                    let s2: i128 = fin.iter().map(|&v| v as i128 * v as i128).sum();
                    let num = w as i128 * s2 - s1 * s1;
                    assert!(num >= 0);
                    let want = if num == 0 { 0.0 } else { (num as f64).sqrt() / (w as f64) / SCALE };
                    let got = st.std_pop(i, w).unwrap();
                    if num == 0 {
                        assert_eq!(got, 0.0, "{what} constant window w{w} i{i}");
                    } else {
                        assert!(ulp_diff(got, want) <= 2, "{what} std w{w} i{i}: {got} vs {want}");
                    }
                    assert_eq!(st.mean_strict(i, w).is_some(), true);
                } else {
                    assert!(st.std_pop(i, w).is_none() && st.mean_strict(i, w).is_none());
                }
            }
        }
    }

    const WS: [usize; 5] = [3, 16, 96, 672, 2880];

    #[test]
    fn windows_match_an_exact_reference_across_blocks_and_gaps() {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
        let n = 5_200;
        let cases: [(i64, u64, u64, &str); 5] = [
            (1 << 30, 1 << 18, 0, "offset 1e3, kappa 2e8"),
            (1 << 34, 1 << 18, 0, "offset 2e4, kappa 5e10"),
            (1 << 20, 1 << 22, 0, "ordinary"),
            (0, 1 << 30, 40, "zero mean with holes"),
            (3 << 45, 1, 0, "constant"),
        ];
        for (off, spread, holes, what) in cases {
            let t = times_with_gaps(n, &[777, 1030, 3100]);
            let (x, k) = lattice(&mut rng, n, off, spread, holes);
            check_exact(&x, &k, &t, &WS, what);
            let t = times_with_gaps(n, &[]);
            check_exact(&x, &k, &t, &WS, what);
        }
    }

    #[test]
    fn constant_runs_give_exact_zero_and_flat_neighbours_do_not() {
        let mut rng = Rng(5);
        let n = 3_000;
        let t = times_with_gaps(n, &[]);
        let (mut x, _) = lattice(&mut rng, n, 1 << 24, 1 << 10, 0);
        for v in x[500..1500].iter_mut() {
            *v = 1.0000000001;
        }
        let st = WinStats::new(&x, &t, STEP);
        assert_eq!(st.std_pop(1400, 96), Some(0.0));
        assert_eq!(st.std_pop(1499, 672), Some(0.0));
        assert!(st.std_pop(1550, 96).unwrap() > 0.0);
    }

    fn bits(v: &[Option<f64>]) -> Vec<u64> {
        v.iter().map(|o| o.map_or(u64::MAX, f64::to_bits)).collect()
    }

    #[test]
    fn a_window_does_not_depend_on_rows_before_a_gap() {
        let mut rng = Rng(11);
        let n = 4_000;
        let (x, _) = lattice(&mut rng, n, 1 << 22, 1 << 12, 0);
        let t = times_with_gaps(n, &[]);
        for junk in [0usize, 1, 1023, 1024, 1500, 2051] {
            let (jx, _) = lattice(&mut rng, junk, 1 << 30, 1 << 28, 0);
            let mut xx = jx.clone();
            xx.extend_from_slice(&x);
            let mut tt: Vec<i64> = (0..junk as i64).map(|j| 1_600_000_000 + j * STEP).collect();
            let last = t[0] - 5 * STEP;
            for (j, v) in tt.iter_mut().enumerate() {
                *v = last - (junk as i64 - j as i64) * STEP - 10 * STEP;
            }
            tt.extend_from_slice(&t);
            let a = WinStats::new(&x, &t, STEP);
            let b = WinStats::new(&xx, &tt, STEP);
            for w in [16usize, 96, 672] {
                let ma: Vec<Option<f64>> = (w - 1..n).map(|i| a.mean_finite(i, w)).collect();
                let mb: Vec<Option<f64>> = (w - 1..n).map(|i| b.mean_finite(i + junk, w)).collect();
                assert_eq!(bits(&ma), bits(&mb), "mean w{w} junk{junk}");
                let sa: Vec<Option<f64>> = (w - 1..n).map(|i| a.std_pop(i, w)).collect();
                let sb: Vec<Option<f64>> = (w - 1..n).map(|i| b.std_pop(i + junk, w)).collect();
                assert_eq!(bits(&sa), bits(&sb), "std w{w} junk{junk}");
            }
        }
    }

    /// The old helpers' count logic, kept as the mask reference.
    fn old_masks(x: &[f64], w: usize, t: &[i64]) -> (Vec<bool>, Vec<bool>) {
        let mut cnt = vec![0usize];
        for v in x {
            cnt.push(cnt.last().unwrap() + v.is_finite() as usize);
        }
        let n = x.len();
        let (mut fin, mut strict) = (vec![false; n], vec![false; n]);
        for i in w - 1..n {
            if !contig(t, i, w) {
                continue;
            }
            let c = cnt[i + 1] - cnt[i + 1 - w];
            fin[i] = c > 0;
            strict[i] = c == w;
        }
        (fin, strict)
    }

    #[test]
    fn nan_masks_equal_the_count_logic() {
        let mut rng = Rng(23);
        let n = 4_000;
        let t = times_with_gaps(n, &[900, 2500]);
        let (x, _) = lattice(&mut rng, n, 5, 1 << 20, 30);
        let st = WinStats::new(&x, &t, STEP);
        for w in [4usize, 16, 96, 384] {
            let (fin, strict) = old_masks(&x, w, &t);
            for i in w - 1..n {
                if !contig(&t, i, w) {
                    continue;
                }
                assert_eq!(st.mean_finite(i, w).is_some(), fin[i], "finite w{w} i{i}");
                assert_eq!(st.mean_strict(i, w).is_some(), strict[i], "strict w{w} i{i}");
                assert_eq!(st.std_pop(i, w).is_some(), strict[i], "std w{w} i{i}");
            }
        }
    }

    #[test]
    fn log1p_det_is_accurate_and_continuous_at_its_cutoff() {
        let mut worst = 0.0f64;
        for k in -2500..=2500 {
            let r = k as f64 * 1e-4 + 3.3e-9;
            let want = r.ln_1p();
            let got = log1p_det(r);
            if want != 0.0 {
                worst = worst.max(((got - want) / want).abs());
            }
        }
        assert!(worst < 1e-15, "worst relative error {worst:e}");
        assert_eq!(log1p_det(0.0), 0.0);
        assert!(log1p_det(f64::NAN).is_nan());
        assert_eq!(log1p_det(0.3), 0.3f64.ln_1p());
    }

    #[test]
    fn the_co_moment_ratio_matches_an_exact_ar1_estimate() {
        let mut rng = Rng(31);
        let n = 3_000;
        let t = times_with_gaps(n, &[1200]);
        let (x, k) = lattice(&mut rng, n, 1 << 17, 1 << 18, 0);
        let mut xl = vec![f64::NAN; n];
        let mut kl = vec![0i64; n];
        for j in 1..n {
            if t[j] - t[j - 1] == STEP {
                xl[j] = x[j - 1];
                kl[j] = k[j - 1];
            }
        }
        let w = 96;
        let got = comoment_ratio(&x, &xl, w, &t, STEP);
        for i in w - 1..n {
            if !contig(&t, i, w) || (i + 1 - w..=i).any(|j| !xl[j].is_finite()) {
                continue;
            }
            let nn = w as i128;
            let (mut sx, mut sl, mut sxl, mut sl2) = (0i128, 0i128, 0i128, 0i128);
            for j in i + 1 - w..=i {
                let (a, b) = (k[j] as i128, kl[j] as i128);
                sx += a;
                sl += b;
                sxl += a * b;
                sl2 += b * b;
            }
            let cov = (nn * sxl - sx * sl) as f64;
            let var = (nn * sl2 - sl * sl) as f64;
            let want = cov / var;
            assert!((got[i] - want).abs() <= 4.0 * f64::EPSILON * want.abs().max(1.0), "i{i}: {} vs {}", got[i], want);
        }
    }
}
