/*
 * Rolling window statistics from block-local compensated prefix sums (#364).
 *
 * This header is the device twin of `src/window_stats.rs`. Every expression
 * keeps the order and parenthesization of the Rust code and uses the IEEE
 * `__d*_rn` intrinsics, so no compiler setting can contract a product into a
 * fused multiply-add and the CPU and device results are bit-identical.
 *
 * A "view" `V` exposes slot-indexed access to one input:
 *   double V::s(int k, int q)   running sum of stream k at slot q
 *   double V::c(int k, int q)   running compensation of stream k at slot q
 *   int    V::bk(int q)         rows from slot q back to its accumulation start
 *   int    V::cn(int q)         finite cells up to and including slot q
 *   double V::x(int q)          the input value at slot q
 * Stream 0 is Sum2 of x, stream 1 is Dot2 of x*x. A window is the slot range
 * [lo, hi]; slot lo-1 must be readable (zero before the first row).
 */
#ifndef RCSR_WIN_STATS_H
#define RCSR_WIN_STATS_H

#define RW_BLOCK 1024
#define RW_TAU 8.673617379884035e-19

/* NaN policies of the window mean (explicit, not derived from the length). */
#define RC_WIN_MEAN_FINITE 0 /* mean of the finite cells, any count > 0 */
#define RC_WIN_MEAN_STRICT 1 /* every cell of the window must be finite */

struct RwDw {
    double h;
    double l;
};

__device__ __forceinline__ void rw_two_sum(double a, double b, double *s, double *e)
{
    double ss = __dadd_rn(a, b);
    double bb = __dsub_rn(ss, a);
    *e = __dadd_rn(__dsub_rn(a, __dsub_rn(ss, bb)), __dsub_rn(b, bb));
    *s = ss;
}

__device__ __forceinline__ void rw_fast_two_sum(double a, double b, double *s, double *e)
{
    double ss = __dadd_rn(a, b);
    *e = __dsub_rn(b, __dsub_rn(ss, a));
    *s = ss;
}

__device__ __forceinline__ void rw_split(double a, double *hi, double *lo)
{
    double t = __dmul_rn(134217729.0, a);
    double h = __dsub_rn(t, __dsub_rn(t, a));
    *hi = h;
    *lo = __dsub_rn(a, h);
}

__device__ __forceinline__ void rw_two_prod(double a, double b, double *p, double *e)
{
    double pp = __dmul_rn(a, b);
    double ah, al, bh, bl;
    rw_split(a, &ah, &al);
    rw_split(b, &bh, &bl);
    *e = __dsub_rn(__dmul_rn(al, bl),
                   __dsub_rn(__dsub_rn(__dsub_rn(pp, __dmul_rn(ah, bh)), __dmul_rn(al, bh)),
                             __dmul_rn(ah, bl)));
    *p = pp;
}

__device__ __forceinline__ RwDw rw_dw_add(RwDw x, RwDw y)
{
    double sh, sl, th, tl, vh, vl, zh, zl;
    rw_two_sum(x.h, y.h, &sh, &sl);
    rw_two_sum(x.l, y.l, &th, &tl);
    double c = __dadd_rn(sl, th);
    rw_fast_two_sum(sh, c, &vh, &vl);
    double w = __dadd_rn(tl, vl);
    rw_fast_two_sum(vh, w, &zh, &zl);
    RwDw r = {zh, zl};
    return r;
}

__device__ __forceinline__ RwDw rw_dw_neg(RwDw x)
{
    RwDw r = {-x.h, -x.l};
    return r;
}

__device__ __forceinline__ RwDw rw_dw_mul_f(RwDw x, double y)
{
    double ch, cl1, th, tl1, zh, zl;
    rw_two_prod(x.h, y, &ch, &cl1);
    double cl2 = __dmul_rn(x.l, y);
    rw_fast_two_sum(ch, cl2, &th, &tl1);
    double tl2 = __dadd_rn(tl1, cl1);
    rw_fast_two_sum(th, tl2, &zh, &zl);
    RwDw r = {zh, zl};
    return r;
}

__device__ __forceinline__ RwDw rw_dw_mul(RwDw x, RwDw y)
{
    double ch, cl1, zh, zl;
    rw_two_prod(x.h, y.h, &ch, &cl1);
    double tl1 = __dmul_rn(x.h, y.l);
    double tl2 = __dmul_rn(x.l, y.h);
    double cl2 = __dadd_rn(tl1, tl2);
    double cl3 = __dadd_rn(cl1, cl2);
    rw_fast_two_sum(ch, cl3, &zh, &zl);
    RwDw r = {zh, zl};
    return r;
}

/* Double-word sum over an exact count, rounded once with one correction. */
__device__ __forceinline__ double rw_div_count(RwDw s, double n)
{
    double q1 = __ddiv_rn(s.h, n);
    double p, e;
    rw_two_prod(q1, n, &p, &e);
    double r = __dadd_rn(__dsub_rn(__dsub_rn(s.h, p), e), s.l);
    return __dadd_rn(q1, __ddiv_rn(r, n));
}

/* Variance numerator, exactly zero when all finite cells of the window are
   equal. The scan runs only inside the double-word error bound. */
template <class V>
__device__ __forceinline__ double rw_guarded(const V &v, RwDw num, RwDw s2, double n, int lo, int hi)
{
    if (s2.h == 0.0) return 0.0;
    if (num.h <= __dmul_rn(RW_TAU, __dmul_rn(n, s2.h))) {
        double first = NAN;
        bool equal = true;
        for (int j = lo; j <= hi; ++j) {
            double xv = v.x(j);
            if (isfinite(xv)) {
                if (isnan(first)) {
                    first = xv;
                } else if (xv != first) {
                    equal = false;
                    break;
                }
            }
        }
        if (equal) return 0.0;
    }
    return num.h;
}

/* Sum over slots [lo, hi] of stream k as a double-word, pieces from hi back. */
template <class V>
__device__ __forceinline__ RwDw rw_win(const V &v, int k, int lo, int hi)
{
    int e = hi;
    RwDw acc = {0.0, 0.0};
    bool first = true;
    for (;;) {
        int a = e - v.bk(e);
        double h, l;
        if (a >= lo) {
            rw_two_sum(v.s(k, e), v.c(k, e), &h, &l);
        } else {
            double d, r, t;
            rw_two_sum(v.s(k, e), -v.s(k, lo - 1), &d, &r);
            t = __dadd_rn(r, __dsub_rn(v.c(k, e), v.c(k, lo - 1)));
            rw_two_sum(d, t, &h, &l);
        }
        RwDw piece = {h, l};
        acc = first ? piece : rw_dw_add(acc, piece);
        first = false;
        if (a <= lo) return acc;
        e = a - 1;
    }
}

/* `rmean_py`: mean of the finite cells, NaN when there are none. */
template <class V>
__device__ __forceinline__ double rw_mean_finite(const V &v, int hi, int w)
{
    int lo = hi + 1 - w;
    int k = v.cn(hi) - v.cn(lo - 1);
    if (k <= 0) return NAN;
    return rw_div_count(rw_win(v, 0, lo, hi), (double)k);
}

/* Strict mean: every cell of the window finite. */
template <class V>
__device__ __forceinline__ double rw_mean_strict(const V &v, int hi, int w)
{
    int lo = hi + 1 - w;
    if (v.cn(hi) - v.cn(lo - 1) != w) return NAN;
    return rw_div_count(rw_win(v, 0, lo, hi), (double)w);
}

/* Population standard deviation of a fully finite window. */
template <class V>
__device__ __forceinline__ double rw_std_pop(const V &v, int hi, int w)
{
    int lo = hi + 1 - w;
    if (v.cn(hi) - v.cn(lo - 1) != w) return NAN;
    RwDw s1 = rw_win(v, 0, lo, hi);
    RwDw s2 = rw_win(v, 1, lo, hi);
    double wf = (double)w;
    RwDw num = rw_dw_add(rw_dw_mul_f(s2, wf), rw_dw_neg(rw_dw_mul(s1, s1)));
    double vv = __ddiv_rn(rw_guarded(v, num, s2, wf, lo, hi), __dmul_rn(wf, wf));
    return __dsqrt_rn(fmax(vv, 0.0));
}

/* `window_stats::log1p_det`: ln(1 + r) from + - * / only for |r| <= 0.25. */
__device__ __forceinline__ double rw_log1p_det(double r)
{
    if (!(fabs(r) <= 0.25)) return log1p(r);
    double z = __ddiv_rn(r, __dadd_rn(2.0, r));
    double z2 = __dmul_rn(z, z);
    double s = __ddiv_rn(1.0, 21.0);
    for (int k = 9; k >= 0; --k) {
        s = __dadd_rn(__dmul_rn(s, z2), __ddiv_rn(1.0, (double)(2 * k + 1)));
    }
    return __dmul_rn(__dmul_rn(2.0, z), s);
}

/* One running accumulation step. `restart` zeroes the accumulation first. */
struct RwAcc {
    double s1, c1, s2, c2;
    int cnt;
    int start; /* row (slot) index where the accumulation began */
};

__device__ __forceinline__ void rw_acc_init(RwAcc *a)
{
    a->s1 = 0.0;
    a->c1 = 0.0;
    a->s2 = 0.0;
    a->c2 = 0.0;
    a->cnt = 0;
    a->start = 0;
}

/* Sum2 of x, plus Dot2 of x*x when `sq`. `q` is the slot of this row. */
__device__ __forceinline__ void rw_acc_push(RwAcc *a, double x, int q, bool restart, bool sq)
{
    if (restart) {
        a->s1 = 0.0;
        a->c1 = 0.0;
        a->s2 = 0.0;
        a->c2 = 0.0;
        a->start = q;
    }
    if (isfinite(x)) {
        double t, e;
        rw_two_sum(a->s1, x, &t, &e);
        a->s1 = t;
        a->c1 = __dadd_rn(a->c1, e);
        if (sq) {
            double p, pe;
            rw_two_prod(x, x, &p, &pe);
            rw_two_sum(a->s2, p, &t, &e);
            a->s2 = t;
            a->c2 = __dadd_rn(a->c2, __dadd_rn(e, pe));
        }
        a->cnt += 1;
    }
}

/* Dot2 of x*y into stream 1 position (used as the only stream of a pair ring). */
__device__ __forceinline__ void rw_acc_push_dot(RwAcc *a, double x, double y, int q, bool restart)
{
    if (restart) {
        a->s1 = 0.0;
        a->c1 = 0.0;
        a->s2 = 0.0;
        a->c2 = 0.0;
        a->start = q;
    }
    if (isfinite(x) && isfinite(y)) {
        double p, pe, t, e;
        rw_two_prod(x, y, &p, &pe);
        rw_two_sum(a->s1, p, &t, &e);
        a->s1 = t;
        a->c1 = __dadd_rn(a->c1, __dadd_rn(e, pe));
        a->cnt += 1;
    }
}

#endif /* RCSR_WIN_STATS_H */
