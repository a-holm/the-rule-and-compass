/* #1051: exact row decisions of the device book check.
 *
 * Included at the end of eval.cu after enumerate.h. The device mirror of
 * src/flat_bound/rows.rs: the same postfix words, the same interval and NaN
 * rules, the same integer-step widening. Every rule below names the Rust
 * function it follows; the two must stay equal.
 *
 * One block per (slot, 8192-row chunk). A warp handles 32 aligned rows and
 * writes one word of each plane: bit r of word w is row 32 w + r. `enter` holds
 * the rows whose f64 value is certainly finite and above the enter threshold,
 * `stay` those above the stay threshold. Rows the intervals leave undecided
 * go to a per-slot list (cap RC_CHECK_UND_CAP) and the host resolves them with
 * the CPU scalar evaluator. A row that is certainly non-finite sets
 * RC_CHECK_NONFIN, which stops the rest of the slot.
 *
 * Build flags: -fmad=false, no fast math, so + - * / are single IEEE
 * operations as on the host. */
#ifndef RCSR_CHECK_ROWS_H
#define RCSR_CHECK_ROWS_H

#define RC_CHECK_STACK 16
#define RC_CHECK_ULPS 64
#define RC_CHECK_UND_CAP 4096
#define RC_CHECK_SLOT_CAP 64
#define RC_CHECK_ROWS_PER_BLOCK 8192
#define RC_CHECK_THREADS 256

#define RC_CHECK_UND 1u
#define RC_CHECK_NONFIN 2u

enum {
    CK_CONST = 0, CK_VAR = 1, CK_ADD = 2, CK_SUB = 3, CK_MUL = 4, CK_DIV = 5,
    CK_EXP = 9, CK_ABS = 12, CK_TANH = 13, CK_LOGISTIC = 14, CK_SQUARE = 15,
    CK_NEG = 17, CK_GAUSS = 18, CK_MIN = 30, CK_MAX = 31, CK_IF = 32,
    CK_GREATER = 37, CK_HALT = 255
};
#define RC_CK_SAME(a, b) ((int)(a) == (int)(b))
static_assert(RC_CK_SAME(CK_EXP, OP_EXP) && RC_CK_SAME(CK_TANH, OP_TANH) && RC_CK_SAME(CK_LOGISTIC, OP_SIG)
              && RC_CK_SAME(CK_GAUSS, OP_GAUSS) && RC_CK_SAME(CK_MIN, OP_MIN) && RC_CK_SAME(CK_MAX, OP_MAX)
              && RC_CK_SAME(CK_IF, OP_IF) && RC_CK_SAME(CK_GREATER, OP_GREATER) && RC_CK_SAME(CK_NEG, OP_NEG)
              && RC_CK_SAME(CK_ABS, OP_ABS) && RC_CK_SAME(CK_SQUARE, OP_SQUARE) && RC_CK_SAME(CK_ADD, OP_ADD)
              && RC_CK_SAME(CK_SUB, OP_SUB) && RC_CK_SAME(CK_MUL, OP_MUL) && RC_CK_SAME(CK_DIV, OP_DIV)
              && RC_CK_SAME(CK_CONST, OP_CONST) && RC_CK_SAME(CK_VAR, OP_VAR) && RC_CK_SAME(CK_HALT, OP_HALT),
              "check opcodes equal the interpreter opcodes");

#define RC_NAN_NO 0
#define RC_NAN_MAYBE 1
#define RC_NAN_CERTAIN 2

/* `RowIv` of rows.rs. */
struct RcIv {
    double lo, hi;
    int nan;
};

#define RC_INF (__longlong_as_double(0x7ff0000000000000LL))

__device__ __forceinline__ RcIv rc_iv_nan_row()
{
    RcIv r;
    r.lo = -RC_INF;
    r.hi = RC_INF;
    r.nan = RC_NAN_CERTAIN;
    return r;
}

__device__ __forceinline__ RcIv rc_iv(double lo, double hi, int nan)
{
    RcIv r;
    r.lo = lo;
    r.hi = hi;
    r.nan = nan;
    return r;
}

/* `point` */
__device__ __forceinline__ RcIv rc_iv_point(double v)
{
    return isnan(v) ? rc_iv_nan_row() : rc_iv(v, v, RC_NAN_NO);
}

/* `ulps_step`: integer step on the ordered bit pattern, saturating at the
   infinities; infinities and NaN unchanged. */
__device__ __forceinline__ double rc_ulps(double v, int k)
{
    if (!isfinite(v)) return v;
    long long i = __double_as_longlong(v);
    if (i < 0) i = (long long)(0x8000000000000000ULL - (unsigned long long)i);
    i += k;
    const long long top = 0x7ff0000000000000LL;
    if (i > top) i = top;
    if (i < -top) i = -top;
    if (i < 0) i = (long long)(0x8000000000000000ULL - (unsigned long long)i);
    return __longlong_as_double(i);
}

__device__ __forceinline__ double rc_dn(double v) { return rc_ulps(v, -RC_CHECK_ULPS); }
__device__ __forceinline__ double rc_up(double v) { return rc_ulps(v, RC_CHECK_ULPS); }

__device__ __forceinline__ int rc_join_nan(int a, int b)
{
    if (a == RC_NAN_CERTAIN || b == RC_NAN_CERTAIN) return RC_NAN_CERTAIN;
    return a > b ? a : b;
}

__device__ __forceinline__ bool rc_has_inf(RcIv a) { return a.lo == -RC_INF || a.hi == RC_INF; }
__device__ __forceinline__ bool rc_contains_zero(RcIv a) { return a.lo <= 0.0 && a.hi >= 0.0; }
__device__ __forceinline__ bool rc_is_point(RcIv a) { return a.lo == a.hi; }

/* `clean`: a NaN endpoint becomes the matching infinity. */
__device__ __forceinline__ double rc_clean_lo(double v) { return isnan(v) ? -RC_INF : v; }
__device__ __forceinline__ double rc_clean_hi(double v) { return isnan(v) ? RC_INF : v; }

/* `finish_nan` */
__device__ __forceinline__ RcIv rc_finish_nan(RcIv out, RcIv a, RcIv b, bool possible)
{
    const int inherited = rc_join_nan(a.nan, b.nan);
    if (inherited == RC_NAN_CERTAIN) return rc_iv_nan_row();
    if (possible) {
        if (rc_is_point(a) && rc_is_point(b)) return rc_iv_nan_row();
        return rc_iv(-RC_INF, RC_INF, RC_NAN_MAYBE);
    }
    out.nan = inherited;
    return out;
}

__device__ __forceinline__ RcIv rc_add(RcIv a, RcIv b)
{
    const double lo = rc_clean_lo(a.lo + b.lo), hi = rc_clean_hi(a.hi + b.hi);
    const bool possible = (a.hi == RC_INF && b.lo == -RC_INF) || (a.lo == -RC_INF && b.hi == RC_INF);
    return rc_finish_nan(rc_iv(lo, hi, RC_NAN_NO), a, b, possible);
}

__device__ __forceinline__ RcIv rc_sub(RcIv a, RcIv b)
{
    const double lo = rc_clean_lo(a.lo - b.hi), hi = rc_clean_hi(a.hi - b.lo);
    const bool possible = (a.hi == RC_INF && b.hi == RC_INF) || (a.lo == -RC_INF && b.lo == -RC_INF);
    return rc_finish_nan(rc_iv(lo, hi, RC_NAN_NO), a, b, possible);
}

/* `corners` of a product (div = 0) or a quotient (div = 1). */
__device__ __forceinline__ void rc_corners(RcIv a, RcIv b, int div, double *lo, double *hi)
{
    double c[4];
    c[0] = div ? a.lo / b.lo : a.lo * b.lo;
    c[1] = div ? a.lo / b.hi : a.lo * b.hi;
    c[2] = div ? a.hi / b.lo : a.hi * b.lo;
    c[3] = div ? a.hi / b.hi : a.hi * b.hi;
    if (isnan(c[0]) || isnan(c[1]) || isnan(c[2]) || isnan(c[3])) {
        *lo = -RC_INF;
        *hi = RC_INF;
        return;
    }
    double l = RC_INF, h = -RC_INF;
    for (int i = 0; i < 4; ++i) {
        l = fmin(l, c[i]);
        h = fmax(h, c[i]);
    }
    *lo = l;
    *hi = h;
}

__device__ __forceinline__ RcIv rc_mul(RcIv a, RcIv b)
{
    const bool possible = (rc_contains_zero(a) && rc_has_inf(b)) || (rc_contains_zero(b) && rc_has_inf(a));
    double lo, hi;
    rc_corners(a, b, 0, &lo, &hi);
    return rc_finish_nan(rc_iv(lo, hi, RC_NAN_NO), a, b, possible);
}

#define RC_CHECK_GUARD 1e-12
#define RC_CHECK_EXP_LIMIT 700.0

__device__ __forceinline__ RcIv rc_div(RcIv a, RcIv b)
{
    const int inherited = rc_join_nan(a.nan, b.nan);
    if (inherited == RC_NAN_CERTAIN) return rc_iv_nan_row();
    if (b.lo > -RC_CHECK_GUARD && b.hi < RC_CHECK_GUARD) return rc_iv_nan_row();
    const bool zone = b.lo < RC_CHECK_GUARD && b.hi > -RC_CHECK_GUARD;
    double lo = RC_INF, hi = -RC_INF;
    bool nan_possible = zone;
    if (b.lo <= -RC_CHECK_GUARD) {
        const RcIv part = rc_iv(b.lo, fmin(b.hi, -RC_CHECK_GUARD), RC_NAN_NO);
        nan_possible = nan_possible || (rc_has_inf(a) && rc_has_inf(part));
        double l, h;
        rc_corners(a, part, 1, &l, &h);
        lo = fmin(lo, l);
        hi = fmax(hi, h);
    }
    if (b.hi >= RC_CHECK_GUARD) {
        const RcIv part = rc_iv(fmax(b.lo, RC_CHECK_GUARD), b.hi, RC_NAN_NO);
        nan_possible = nan_possible || (rc_has_inf(a) && rc_has_inf(part));
        double l, h;
        rc_corners(a, part, 1, &l, &h);
        lo = fmin(lo, l);
        hi = fmax(hi, h);
    }
    if (rc_is_point(a) && rc_is_point(b) && nan_possible && !zone) return rc_iv_nan_row();
    return rc_iv(lo, hi, nan_possible ? (inherited > RC_NAN_MAYBE ? inherited : RC_NAN_MAYBE) : inherited);
}

/* `square_iv` */
__device__ __forceinline__ void rc_square_iv(double lo, double hi, double *olo, double *ohi)
{
    if (lo >= 0.0) {
        *olo = lo * lo;
        *ohi = hi * hi;
    } else if (hi <= 0.0) {
        *olo = hi * hi;
        *ohi = lo * lo;
    } else {
        *olo = 0.0;
        *ohi = fmax(lo * lo, hi * hi);
    }
}

/* `exp_raw` */
__device__ __forceinline__ void rc_exp_raw(double lo, double hi, double *olo, double *ohi)
{
    *olo = fmax(rc_dn(exp(lo)), 0.0);
    *ohi = rc_up(exp(hi));
}

__device__ __forceinline__ RcIv rc_exp_guarded(RcIv a)
{
    return rc_iv(a.lo > RC_CHECK_EXP_LIMIT ? RC_INF : fmax(rc_dn(exp(a.lo)), 0.0),
                 a.hi > RC_CHECK_EXP_LIMIT ? RC_INF : rc_up(exp(a.hi)), RC_NAN_NO);
}

__device__ __forceinline__ RcIv rc_tanh(RcIv a)
{
    return rc_iv(fmax(rc_dn(tanh(a.lo)), -1.0), fmin(rc_up(tanh(a.hi)), 1.0), RC_NAN_NO);
}

__device__ __forceinline__ RcIv rc_logistic(RcIv a)
{
    double elo, ehi;
    rc_exp_raw(-a.hi, -a.lo, &elo, &ehi);
    return rc_iv(1.0 / (1.0 + ehi), 1.0 / (1.0 + elo), RC_NAN_NO);
}

__device__ __forceinline__ RcIv rc_gauss(RcIv a)
{
    double slo, shi, elo, ehi;
    rc_square_iv(a.lo, a.hi, &slo, &shi);
    rc_exp_raw(-shi, -slo, &elo, &ehi);
    return rc_iv(fmax(elo, 0.0), fmin(ehi, 1.0), RC_NAN_NO);
}

__device__ __forceinline__ RcIv rc_min_max(RcIv a, RcIv b, bool is_min)
{
    const int inherited = rc_join_nan(a.nan, b.nan);
    if (inherited == RC_NAN_CERTAIN) return rc_iv_nan_row();
    return is_min ? rc_iv(fmin(a.lo, b.lo), fmin(a.hi, b.hi), inherited)
                  : rc_iv(fmax(a.lo, b.lo), fmax(a.hi, b.hi), inherited);
}

__device__ __forceinline__ RcIv rc_greater(RcIv a, RcIv b)
{
    const int inherited = rc_join_nan(a.nan, b.nan);
    if (inherited == RC_NAN_CERTAIN) return rc_iv_nan_row();
    if (a.lo > b.hi) return rc_iv(1.0, 1.0, inherited);
    if (a.hi <= b.lo) return rc_iv(0.0, 0.0, inherited);
    return rc_iv(0.0, 1.0, inherited);
}

/* `select`: a NaN condition counts as "not > 0". */
__device__ __forceinline__ RcIv rc_select(RcIv c, RcIv t, RcIv e)
{
    if (c.nan == RC_NAN_CERTAIN || c.hi <= 0.0) return e;
    if (c.nan == RC_NAN_NO && c.lo > 0.0) return t;
    if (t.nan == RC_NAN_CERTAIN && e.nan == RC_NAN_CERTAIN) return rc_iv_nan_row();
    if (t.nan == RC_NAN_CERTAIN) {
        e.nan = RC_NAN_MAYBE;
        return e;
    }
    if (e.nan == RC_NAN_CERTAIN) {
        t.nan = RC_NAN_MAYBE;
        return t;
    }
    return rc_iv(fmin(t.lo, e.lo), fmax(t.hi, e.hi), t.nan > e.nan ? t.nan : e.nan);
}

/* unary operators: `unary_nan` keeps the operand's NaN state */
__device__ __forceinline__ RcIv rc_unary(unsigned int w, RcIv a)
{
    if (a.nan == RC_NAN_CERTAIN) return rc_iv_nan_row();
    RcIv r;
    switch (w) {
        case CK_NEG: r = rc_iv(-a.hi, -a.lo, RC_NAN_NO); break;
        case CK_ABS:
            if (a.lo >= 0.0) r = a;
            else if (a.hi <= 0.0) r = rc_iv(-a.hi, -a.lo, RC_NAN_NO);
            else r = rc_iv(0.0, fmax(-a.lo, a.hi), RC_NAN_NO);
            break;
        case CK_SQUARE: {
            double lo, hi;
            rc_square_iv(a.lo, a.hi, &lo, &hi);
            r = rc_iv(lo, hi, RC_NAN_NO);
            break;
        }
        case CK_EXP: r = rc_exp_guarded(a); break;
        case CK_TANH: r = rc_tanh(a); break;
        case CK_LOGISTIC: r = rc_logistic(a); break;
        default: r = rc_gauss(a); break;
    }
    r.nan = a.nan;
    return r;
}

/* `row_iv`. `code` is one slot's words; `cols` holds column k at k * n. */
__device__ RcIv rc_check_row_iv(const unsigned int *code, const double *cols, int n, int row)
{
    RcIv st[RC_CHECK_STACK + 1];
    int sp = 0;
    int i = 0;
    for (;;) {
        const unsigned int w = code[i++];
        if (w == CK_HALT) break;
        if (w == CK_CONST) {
            const unsigned long long b = (unsigned long long)code[i] | ((unsigned long long)code[i + 1] << 32);
            i += 2;
            st[sp++] = rc_iv_point(__longlong_as_double((long long)b));
        } else if (w == CK_VAR) {
            st[sp++] = rc_iv_point(cols[(size_t)code[i] * (size_t)n + (size_t)row]);
            i += 1;
        } else if (w == CK_NEG || w == CK_ABS || w == CK_SQUARE || w == CK_EXP || w == CK_TANH
                   || w == CK_LOGISTIC || w == CK_GAUSS) {
            st[sp - 1] = rc_unary(w, st[sp - 1]);
        } else if (w == CK_IF) {
            const RcIv c = st[sp - 3], t = st[sp - 2], e = st[sp - 1];
            sp -= 2;
            st[sp - 1] = rc_select(c, t, e);
        } else {
            const RcIv a = st[sp - 2], b = st[sp - 1];
            sp -= 1;
            RcIv r;
            switch (w) {
                case CK_ADD: r = rc_add(a, b); break;
                case CK_SUB: r = rc_sub(a, b); break;
                case CK_MUL: r = rc_mul(a, b); break;
                case CK_DIV: r = rc_div(a, b); break;
                case CK_MIN: r = rc_min_max(a, b, true); break;
                case CK_MAX: r = rc_min_max(a, b, false); break;
                default: r = rc_greater(a, b); break;
            }
            st[sp - 1] = r;
        }
    }
    return st[0];
}

/* `classify` codes */
#define RC_ROW_NONFINITE 0
#define RC_ROW_ENTER 1
#define RC_ROW_STAY 2
#define RC_ROW_NEITHER 3
#define RC_ROW_UNDECIDED 4

__device__ __forceinline__ int rc_check_classify(RcIv v, double t_enter, double t_stay)
{
    if (v.nan == RC_NAN_CERTAIN || v.lo == RC_INF || v.hi == -RC_INF) return RC_ROW_NONFINITE;
    if (v.nan != RC_NAN_NO || v.lo == -RC_INF || v.hi == RC_INF) return RC_ROW_UNDECIDED;
    if (v.lo > t_enter) return RC_ROW_ENTER;
    if (v.hi <= t_stay) return RC_ROW_NEITHER;
    if (v.lo > t_stay && v.hi <= t_enter) return RC_ROW_STAY;
    return RC_ROW_UNDECIDED;
}

/* The device `row_iv` and `classify` of every row of a test input: one thread
   per (slot, row). Used by the soundness test only. */
__global__ void check_iv_kernel(const double *cols, int n, const unsigned int *prog, const unsigned int *off,
                                double t_enter, double t_stay, double *lo, double *hi, int *nan, int *cls)
{
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    if (row >= n) return;
    const RcIv v = rc_check_row_iv(prog + off[slot], cols, n, row);
    const size_t o = (size_t)slot * (size_t)n + (size_t)row;
    lo[o] = v.lo;
    hi[o] = v.hi;
    nan[o] = v.nan;
    cls[o] = rc_check_classify(v, t_enter, t_stay);
}

/* Grid (ceil(n / 8192), slots), 256 threads. Word w of slot s is
   bits[s * words + w]. Rows past n are Neither. */
__global__ void check_rows_kernel(const double *cols, int n, const unsigned int *prog, const unsigned int *off,
                                  double t_enter, double t_stay, int words, unsigned int *enter_bits,
                                  unsigned int *stay_bits, unsigned int *slot_flags, unsigned int *und_count,
                                  unsigned int *und_rows)
{
    const int slot = blockIdx.y, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int *code = prog + off[slot];
    const int chunk = blockIdx.x * RC_CHECK_ROWS_PER_BLOCK;
    const int warps = RC_CHECK_THREADS / 32;
    for (int i = 0; i < RC_CHECK_ROWS_PER_BLOCK / RC_CHECK_THREADS; ++i) {
        const int base = chunk + (i * warps + warp) * 32;
        if (base >= n) return;
        unsigned int seen = 0;
        if (lane == 0) seen = *(volatile const unsigned int *)&slot_flags[slot];
        seen = __shfl_sync(0xffffffffu, seen, 0);
        if (seen & RC_CHECK_NONFIN) return;
        const int row = base + lane;
        int cls = RC_ROW_NEITHER;
        if (row < n) cls = rc_check_classify(rc_check_row_iv(code, cols, n, row), t_enter, t_stay);
        const unsigned int ent = __ballot_sync(0xffffffffu, cls == RC_ROW_ENTER);
        const unsigned int sty = __ballot_sync(0xffffffffu, cls == RC_ROW_ENTER || cls == RC_ROW_STAY);
        const unsigned int und = __ballot_sync(0xffffffffu, cls == RC_ROW_UNDECIDED);
        const unsigned int bad = __ballot_sync(0xffffffffu, cls == RC_ROW_NONFINITE);
        unsigned int first = 0;
        if (lane == 0) {
            enter_bits[(size_t)slot * (size_t)words + (size_t)(base >> 5)] = ent;
            stay_bits[(size_t)slot * (size_t)words + (size_t)(base >> 5)] = sty;
            if (und) first = atomicAdd(&und_count[slot], (unsigned int)__popc(und));
            if (und | bad) atomicOr(&slot_flags[slot], (und ? RC_CHECK_UND : 0u) | (bad ? RC_CHECK_NONFIN : 0u));
        }
        if (und) {
            first = __shfl_sync(0xffffffffu, first, 0);
            if ((und >> lane) & 1u) {
                const unsigned int k = first + (unsigned int)__popc(und & ((1u << lane) - 1u));
                if (k < RC_CHECK_UND_CAP) und_rows[(size_t)slot * RC_CHECK_UND_CAP + k] = (unsigned int)row;
            }
        }
    }
}

/* Test kernel: device `exp` and `tanh` of `n` points (`mode` 0 exp, 1 tanh). */
__global__ void check_libm_kernel(const double *x, int n, int mode, double *y)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = mode == 0 ? exp(x[i]) : tanh(x[i]);
}

/* The kernels trust their program words: every slot is checked on the host
   first (known opcodes, operand depth, column index below `nvars`, stack at most
   RC_CHECK_STACK, one result, HALT as the last word of the slot). */
static int rc_check_validate(const unsigned int *prog, const unsigned int *off, int nprog, int nvars)
{
    for (int s = 0; s < nprog; ++s) {
        const unsigned int end = off[s + 1];
        unsigned int i = off[s];
        int depth = 0;
        int halted = 0;
        while (i < end) {
            const unsigned int w = prog[i++];
            int arity = -1;
            switch (w) {
                case CK_HALT: halted = 1; break;
                case CK_CONST:
                    if (i + 2 > end) return -1;
                    i += 2;
                    arity = 0;
                    break;
                case CK_VAR:
                    if (i + 1 > end || prog[i] >= (unsigned int)nvars) return -1;
                    i += 1;
                    arity = 0;
                    break;
                case CK_NEG: case CK_ABS: case CK_SQUARE: case CK_EXP: case CK_TANH: case CK_LOGISTIC: case CK_GAUSS:
                    arity = 1;
                    break;
                case CK_ADD: case CK_SUB: case CK_MUL: case CK_DIV: case CK_MIN: case CK_MAX: case CK_GREATER:
                    arity = 2;
                    break;
                case CK_IF:
                    arity = 3;
                    break;
                default: return -1;
            }
            if (halted) break;
            if (depth < arity) return -1;
            depth = depth - arity + 1;
            if (depth > RC_CHECK_STACK) return -1;
        }
        if (!halted || i != end || depth != 1) return -1;
    }
    return 0;
}

struct RcCheck {
    RcCtx *c;
    int n, nvars, words, slot_cap, word_cap;
    double *d_cols;
    unsigned int *d_prog, *d_off;
    unsigned int *d_enter, *d_stay, *d_flags, *d_und_count, *d_und_rows;
    int cols_set;
};

static char g_check_err[160];

extern "C" {

const char *rc_check_last_error(void)
{
    return g_check_err;
}

/* One handle per context: the exact f64 columns (`check_f64`), the bit planes
   of one group of RC_CHECK_SLOT_CAP slots (`check_bits`) and the slot buffers. */
void *rc_check_create(void *ctx, int nvars, int slot_cap, int word_cap)
{
    g_check_err[0] = 0;
    rc_alloc_fail_clear();
    RcCtx *c = (RcCtx *)ctx;
    if (!c || c->tiled || nvars < 1 || slot_cap < 1 || word_cap < 1 || c->n < 1) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_create: invalid arguments");
        return NULL;
    }
    RcCheck *h = (RcCheck *)calloc(1, sizeof(RcCheck));
    if (!h) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_create: host allocation failed");
        return NULL;
    }
    h->c = c;
    h->n = c->n;
    h->nvars = nvars;
    h->words = (c->n + 31) / 32;
    h->slot_cap = slot_cap > RC_CHECK_SLOT_CAP ? RC_CHECK_SLOT_CAP : slot_cap;
    h->word_cap = word_cap;
    const size_t g = (size_t)h->slot_cap;
    const size_t plane = g * (size_t)h->words * sizeof(unsigned int);
    int rc = rc_dev_alloc(c, "check_f64", (void **)&h->d_cols, sizeof(double) * (size_t)c->n * (size_t)nvars);
    if (rc == 0) rc = rc_dev_alloc(c, "check_bits", (void **)&h->d_enter, plane);
    if (rc == 0) rc = rc_dev_alloc(c, "check_bits", (void **)&h->d_stay, plane);
    if (rc == 0) rc = rc_dev_alloc(c, "check_prog", (void **)&h->d_prog, sizeof(unsigned int) * (size_t)word_cap);
    if (rc == 0) rc = rc_dev_alloc(c, "check_prog", (void **)&h->d_off, sizeof(unsigned int) * (g + 1));
    if (rc == 0) rc = rc_dev_alloc(c, "check_flags", (void **)&h->d_flags, sizeof(unsigned int) * g);
    if (rc == 0) rc = rc_dev_alloc(c, "check_flags", (void **)&h->d_und_count, sizeof(unsigned int) * g);
    if (rc == 0) rc = rc_dev_alloc(c, "check_und", (void **)&h->d_und_rows, sizeof(unsigned int) * g * RC_CHECK_UND_CAP);
    if (rc != 0) {
        void *ps[] = {h->d_cols, h->d_enter, h->d_stay, h->d_prog, h->d_off, h->d_flags, h->d_und_count, h->d_und_rows};
        for (size_t i = 0; i < sizeof(ps) / sizeof(ps[0]); ++i) rc_dev_free(c, ps[i]);
        free(h);
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_create: device allocation refused");
        return NULL;
    }
    return h;
}

void rc_check_destroy(void *hv)
{
    RcCheck *h = (RcCheck *)hv;
    if (!h) return;
    void *ps[] = {h->d_cols, h->d_enter, h->d_stay, h->d_prog, h->d_off, h->d_flags, h->d_und_count, h->d_und_rows};
    for (size_t i = 0; i < sizeof(ps) / sizeof(ps[0]); ++i) rc_dev_free(h->c, ps[i]);
    free(h);
}

/* Column k, n doubles. The host never builds a flat copy. */
int rc_check_set_col(void *hv, int k, const double *col)
{
    RcCheck *h = (RcCheck *)hv;
    if (!h || !col || k < 0 || k >= h->nvars) return -1;
    rc_upload(h->d_cols + (size_t)k * (size_t)h->n, col, sizeof(double) * (size_t)h->n);
    if (cudaGetLastError() != cudaSuccess) return -1;
    if (k == h->nvars - 1) h->cols_set = 1;
    return 0;
}

/* Runs `nprog` slots (words `prog`, offsets `off`, `nprog + 1` entries). Per
   slot: flags (bit 0 undecided rows exist, bit 1 certainly non-finite row), the
   count of undecided rows, up to RC_CHECK_UND_CAP of them, and the two planes
   (`words` words per slot). An output pointer may be null. */
int rc_check_batch(void *hv, const unsigned int *prog, const unsigned int *off, int nprog, double t_enter,
                   double t_stay, unsigned int *out_flags, unsigned int *out_und_count, unsigned int *out_und_rows,
                   unsigned int *out_enter, unsigned int *out_stay)
{
    RcCheck *h = (RcCheck *)hv;
    g_check_err[0] = 0;
    if (!h || !prog || !off || nprog < 1 || off[0] != 0 || (int)off[nprog] > h->word_cap) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: invalid arguments");
        return -1;
    }
    if (!h->cols_set) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: the columns are not uploaded");
        return -1;
    }
    for (int s = 0; s < nprog; ++s) {
        if (off[s + 1] <= off[s] || off[s + 1] > off[nprog]) {
            snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: offsets are not increasing");
            return -1;
        }
    }
    if (rc_check_validate(prog, off, nprog, h->nvars) != 0) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: a program is malformed");
        return -1;
    }
    const size_t words = (size_t)h->words;
    for (int g0 = 0; g0 < nprog; g0 += h->slot_cap) {
        const int g = nprog - g0 < h->slot_cap ? nprog - g0 : h->slot_cap;
        const unsigned int base = off[g0];
        unsigned int offs[RC_CHECK_SLOT_CAP + 1];
        for (int s = 0; s <= g; ++s) offs[s] = off[g0 + s] - base;
        rc_upload(h->d_prog, prog + base, sizeof(unsigned int) * (size_t)offs[g]);
        rc_upload(h->d_off, offs, sizeof(unsigned int) * (size_t)(g + 1));
        if (cudaMemset(h->d_flags, 0, sizeof(unsigned int) * (size_t)g) != cudaSuccess ||
            cudaMemset(h->d_und_count, 0, sizeof(unsigned int) * (size_t)g) != cudaSuccess ||
            cudaMemset(h->d_enter, 0, sizeof(unsigned int) * (size_t)g * words) != cudaSuccess ||
            cudaMemset(h->d_stay, 0, sizeof(unsigned int) * (size_t)g * words) != cudaSuccess) {
            snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: clearing the outputs failed");
            return -1;
        }
        const dim3 grid((unsigned int)((h->n + RC_CHECK_ROWS_PER_BLOCK - 1) / RC_CHECK_ROWS_PER_BLOCK), (unsigned int)g);
        cudaError_t e = rc_launch("check_rows_kernel", check_rows_kernel, grid, dim3(RC_CHECK_THREADS), 0, (cudaStream_t)0,
                                  (const double *)h->d_cols, h->n, (const unsigned int *)h->d_prog,
                                  (const unsigned int *)h->d_off, t_enter, t_stay, h->words, h->d_enter, h->d_stay,
                                  h->d_flags, h->d_und_count, h->d_und_rows);
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            snprintf(g_check_err, sizeof(g_check_err), "rc_check_batch: check_rows_kernel failed: %s", cudaGetErrorString(e));
            return -1;
        }
        if (out_flags) rc_download(out_flags + g0, h->d_flags, sizeof(unsigned int) * (size_t)g);
        if (out_und_count) rc_download(out_und_count + g0, h->d_und_count, sizeof(unsigned int) * (size_t)g);
        if (out_und_rows) rc_download(out_und_rows + (size_t)g0 * RC_CHECK_UND_CAP, h->d_und_rows, sizeof(unsigned int) * (size_t)g * RC_CHECK_UND_CAP);
        if (out_enter) rc_download(out_enter + (size_t)g0 * words, h->d_enter, sizeof(unsigned int) * (size_t)g * words);
        if (out_stay) rc_download(out_stay + (size_t)g0 * words, h->d_stay, sizeof(unsigned int) * (size_t)g * words);
    }
    return 0;
}

/* Test entry: the device interval, NaN state and class of every row of every
   slot (`nprog * n` entries each), through one thread per row. */
int rc_check_iv_for_tests(void *hv, const unsigned int *prog, const unsigned int *off, int nprog, double t_enter,
                          double t_stay, double *lo, double *hi, int *nan, int *cls)
{
    RcCheck *h = (RcCheck *)hv;
    g_check_err[0] = 0;
    if (!h || !h->cols_set || nprog < 1 || nprog > h->slot_cap || off[0] != 0 || (int)off[nprog] > h->word_cap) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_iv_for_tests: invalid arguments");
        return -1;
    }
    if (rc_check_validate(prog, off, nprog, h->nvars) != 0) {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_iv_for_tests: a program is malformed");
        return -1;
    }
    const size_t cnt = (size_t)nprog * (size_t)h->n;
    double *d_lo = 0, *d_hi = 0;
    int *d_nan = 0, *d_cls = 0;
    int rc = rc_dev_alloc(h->c, "check_test", (void **)&d_lo, sizeof(double) * cnt);
    if (rc == 0) rc = rc_dev_alloc(h->c, "check_test", (void **)&d_hi, sizeof(double) * cnt);
    if (rc == 0) rc = rc_dev_alloc(h->c, "check_test", (void **)&d_nan, sizeof(int) * cnt);
    if (rc == 0) rc = rc_dev_alloc(h->c, "check_test", (void **)&d_cls, sizeof(int) * cnt);
    int ret = -1;
    if (rc == 0) {
        rc_upload(h->d_prog, prog, sizeof(unsigned int) * (size_t)off[nprog]);
        rc_upload(h->d_off, off, sizeof(unsigned int) * (size_t)(nprog + 1));
        const dim3 grid((unsigned int)((h->n + 255) / 256), (unsigned int)nprog);
        cudaError_t e = rc_launch("check_iv_kernel", check_iv_kernel, grid, dim3(256), 0, (cudaStream_t)0,
                                  (const double *)h->d_cols, h->n, (const unsigned int *)h->d_prog,
                                  (const unsigned int *)h->d_off, t_enter, t_stay, d_lo, d_hi, d_nan, d_cls);
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e == cudaSuccess) {
            rc_download(lo, d_lo, sizeof(double) * cnt);
            rc_download(hi, d_hi, sizeof(double) * cnt);
            rc_download(nan, d_nan, sizeof(int) * cnt);
            rc_download(cls, d_cls, sizeof(int) * cnt);
            ret = 0;
        } else {
            snprintf(g_check_err, sizeof(g_check_err), "check_iv_kernel failed: %s", cudaGetErrorString(e));
        }
    } else {
        snprintf(g_check_err, sizeof(g_check_err), "rc_check_iv_for_tests: device allocation refused");
    }
    rc_dev_free(h->c, d_lo);
    rc_dev_free(h->c, d_hi);
    rc_dev_free(h->c, d_nan);
    rc_dev_free(h->c, d_cls);
    return ret;
}

int rc_check_libm_for_tests(void *ctx, const double *x, int n, int mode, double *y)
{
    RcCtx *c = (RcCtx *)ctx;
    g_check_err[0] = 0;
    if (!c || !x || !y || n < 1) return -1;
    double *dx = 0, *dy = 0;
    int ret = -1;
    if (rc_dev_alloc(c, "check_test", (void **)&dx, sizeof(double) * (size_t)n) == 0 &&
        rc_dev_alloc(c, "check_test", (void **)&dy, sizeof(double) * (size_t)n) == 0) {
        rc_upload(dx, x, sizeof(double) * (size_t)n);
        cudaError_t e = rc_launch("check_libm_kernel", check_libm_kernel, dim3((unsigned int)((n + 255) / 256)), dim3(256), 0,
                                  (cudaStream_t)0, (const double *)dx, n, mode, dy);
        if (e == cudaSuccess) e = cudaDeviceSynchronize();
        if (e == cudaSuccess) {
            rc_download(y, dy, sizeof(double) * (size_t)n);
            ret = 0;
        }
    }
    rc_dev_free(c, dx);
    rc_dev_free(c, dy);
    return ret;
}

} /* extern "C" */

#endif
