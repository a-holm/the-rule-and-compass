/*
 * Shared device-context layout for the enrichment kernels.
 *
 * `enrich.cu` owns the resident OHLCV context and the original feature
 * families. `features.cu` (#297) adds the fused flow / price-action producers
 * and reuses the same context, so both translation units must agree on the
 * struct layout and on the optional-column presence bits.
 */
#ifndef RCSR_ENRICH_COMMON_H
#define RCSR_ENRICH_COMMON_H

#include <stdint.h>

/* Optional raw columns. A set bit means the column was present in the store and
 * was uploaded; a clear bit means the device holds an all-NaN buffer and every
 * consumer must emit NaN instead of a zero. */
#define RC_PRESENT_VOLUME        (1u << 0)
#define RC_PRESENT_QUOTE_VOLUME  (1u << 1)
#define RC_PRESENT_TRADES        (1u << 2)
#define RC_PRESENT_TAKER_BUY     (1u << 3)
#define RC_PRESENT_TAKER_BUY_QV  (1u << 4)

typedef struct RcEnrichStats {
    double kernel_ms;
    double launch_ms;
    double sync_ms;
    double alloc_ms;
    double h2d_ms;
    double d2h_ms;
    double wall_ms;
    unsigned long long h2d_bytes;
    unsigned long long d2h_bytes;
    int batch;
    int failed;
} RcEnrichStats;

typedef struct RcEnrich {
    int n;
    int nseg;
    int step;
    long long *d_times;
    double *d_open;
    double *d_high;
    double *d_low;
    double *d_close;
    double *d_volume;
    double *d_qvolume;
    /* Optional flow columns (#297). Absent columns are NaN buffers. */
    double *d_trades;
    double *d_tb;
    double *d_tbq;
    unsigned int present;
    /* Gap-run id per row: increments at a symbol boundary and at every
       non-contiguous time step. `run_id` equality replaces ad-hoc gap tests. */
    int *d_run_id;
    int *d_seg_start;
    int *d_seg_len;
    int *d_seg_id;
    /* Scratch arena, reused by every family. */
    double *d_a;
    double *d_b;
    double *d_c;
    double *d_d;
    /* #364: compensation plane and integer planes of the window-statistic arena
       (`d_a`, `d_b`, `d_c` hold the sums, `d_e` the squares compensation). */
    double *d_e;
    int *d_cn;
    int *d_bk;
    double *d_out;
    double *d_out2;
    double *d_out3;
    /* #300: one output arena per family block plus its pinned host mirror. */
    double *d_batch;
    double *h_pinned;
    int batch_cap;
    /* Fused flow / price-action base planes (#297), lazily allocated. */
    double *d_feat;
    int feat_cap;
    int base_reads;
    int base_passes;
    /* Cross-section permutation (#295), built once and reused by every rank
       column: the (time, symbol, row) order, group ids and segment offsets. */
    unsigned int *d_xs_order;
    unsigned int *d_xs_order_tmp;
    unsigned int *d_xs_rows_out;
    unsigned int *d_xs_gid;
    unsigned int *d_xs_counts;
    unsigned int *d_xs_sym;
    int *d_xs_off;
    unsigned long long *d_xs_keys;
    unsigned long long *d_xs_keys_out;
    unsigned char *d_xs_member;
    double *d_xs_src;
    void *d_xs_temp;
    size_t xs_temp_bytes;
    int xs_groups;
    int xs_ready;
    RcEnrichStats last;
    double batch_start_ms;
} RcEnrich;

/* Test-only device byte budget for the allocations made after context
   creation (tile outputs and fused planes). Zero means unlimited. */
#ifdef __cplusplus
extern "C" {
#endif
int rc_budget_take(size_t bytes);
void rc_budget_give(size_t bytes);
#ifdef __cplusplus
}
#endif

#endif /* RCSR_ENRICH_COMMON_H */
