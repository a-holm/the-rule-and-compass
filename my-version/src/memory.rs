//! Header-derived resident-memory accounting for `.rcsb` stores (issue #112).
//!
//! The old store guard used one constant, `300 B/cell`, which was a six-column
//! per-row measurement divided by six. That over-charged wide stores by an order
//! of magnitude and made the 3,943,856 x 56 panel look like 61.71 GiB. This
//! module replaces it with an additive model whose terms come from the store
//! header:
//!
//! ```text
//! storage      = resident Table: values only for numeric columns,
//!                an extra i64 copy, 4-byte codes, dictionaries
//! directory    = row-group offsets, header text, Vec/Table headers
//! group buffer = one decode group + decode scratch
//! search       = rows x used columns x (row-major + SoA + f32 [+ mutable])
//!                + rows x 240 B of fixed preparation
//! fit copy     = optional exact 0.75 x rows x used columns x 8 B
//! estimate     = ceil(1.25 x (storage + directory + buffer + search + fit))
//! ```
//!
//! Every product is computed in `u128` and clamped to `u64::MAX`; a clamped
//! result carries `saturated = true` so reporting can say "at least".

/// Fixed per-row preparation cost. This is the same calibrated term used by the
/// CSV guard: row headers, selectors, train/test indices and per-row bookkeeping.
/// It is deliberately **not** multiplied by the column count.
pub const FIXED_BYTES_PER_ROW: u64 = 240;

/// One `f64` in the row-major `Sample.x` and a second in the SoA `EvalSample`.
pub const SEARCH_BYTES_PER_CELL: u64 = 8;
pub const SOA_BYTES_PER_CELL: u64 = 8;
/// GPU `f32` staging copy per used cell (training side).
pub const GPU_F32_BYTES_PER_CELL: u64 = 4;
/// Extra row-major `cells` copy when preparation rewrites values.
pub const MUTABLE_PREP_BYTES_PER_CELL: u64 = 8;
/// Exact `f64` fit image per used cell, amortised over the default 75 % train
/// fraction: `0.75 * 8 = 6` in the sample cap.
pub const GPU_FIT_BYTES_PER_CELL: u64 = 8;

/// 25 % allocator slack, applied once to the summed base.
pub const SLACK_NUM: u64 = 5;
pub const SLACK_DEN: u64 = 4;

/// #300: the per-family GPU output arena is capped at 64 MiB. A block whose
/// columns do not fit falls back to per-column staging, so this bound holds for
/// every column count.
pub const GPU_OUTPUT_ARENA_BYTES: u64 = 64 * 1024 * 1024;
/// #300: the pinned host mirror has the same bound as the device arena.
pub const GPU_PINNED_STAGE_BYTES: u64 = GPU_OUTPUT_ARENA_BYTES;
/// #300: `TempColumn` stages a feature file in 64 KiB blocks; two buffers can
/// be live while a batch block is written.
pub const ENRICH_TEMP_WRITE_CHUNK_BYTES: u64 = 64 * 1024;

/// #300: resident bytes charged for the batched enrichment output path.
/// `rows × columns × 8` is bounded by the arena, so the result never grows
/// without limit when a 358-column catalogue is enriched.
pub fn enrich_output_stage_bytes(rows: u64, columns: u64) -> u64 {
    rows.saturating_mul(columns)
        .saturating_mul(8)
        .min(GPU_OUTPUT_ARENA_BYTES)
        .saturating_add(ENRICH_TEMP_WRITE_CHUNK_BYTES)
}

/// The sample path may use at most 60 % of the budget.
pub const SAMPLE_BUDGET_NUM: u64 = 6;
pub const SAMPLE_BUDGET_DEN: u64 = 10;

/// Projected transport ring (issue #294): three pinned slots, capped at 64 MB.
pub const RING_SLOTS: usize = 3;
pub const RING_CAP_BYTES: u64 = 64 * 1024 * 1024;

/// Resident cost of one column as the loader actually allocates it.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ColumnShape {
    pub name: String,
    /// 8 B per row when the column has a resident `values` block.
    pub value_width: u64,
    /// 8 B per row for the exact `i64` copy (`Column::ints`).
    pub int_width: u64,
    /// 4 B per row when the column keeps dictionary codes.
    pub code_width: u64,
    /// Level strings plus their `String` headers and the dictionary `Vec`.
    pub dictionary_bytes: u64,
}

impl ColumnShape {
    pub fn per_row(&self) -> u64 {
        self.value_width
            .saturating_add(self.int_width)
            .saturating_add(self.code_width)
    }
}

/// Header-derived shape of a store. Pure data: no file handle, no I/O.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct StoreShape {
    pub rows: usize,
    pub group_rows: usize,
    pub header_len: u64,
    pub columns: Vec<ColumnShape>,
}

impl StoreShape {
    pub fn cols(&self) -> usize {
        self.columns.len()
    }

    pub fn n_groups(&self) -> usize {
        if self.rows == 0 {
            0
        } else {
            let g = self.group_rows.max(1);
            (self.rows + g - 1) / g
        }
    }

    /// Exact resident table bytes: values, i64 copies, codes and dictionaries.
    pub fn storage_bytes(&self) -> u64 {
        let rows = self.rows as u128;
        let mut total: u128 = 0;
        for c in &self.columns {
            total = total.saturating_add(rows.saturating_mul(c.value_width as u128));
            total = total.saturating_add(rows.saturating_mul(c.int_width as u128));
            total = total.saturating_add(rows.saturating_mul(c.code_width as u128));
            total = total.saturating_add(c.dictionary_bytes as u128);
        }
        clamp(total).0
    }

    /// Resident bytes per row of a sampled table (the sample loader currently
    /// materialises every column).
    pub fn table_bytes_per_row(&self) -> u64 {
        self.columns
            .iter()
            .fold(0u64, |a, c| a.saturating_add(c.per_row()))
    }

    /// Resident bytes per row for a resolved projection (#294).
    pub fn projected_bytes_per_row(&self, projection: &[usize]) -> u64 {
        projection.iter().fold(0u64, |a, &c| {
            a.saturating_add(self.columns.get(c).map(|c| c.per_row()).unwrap_or(0))
        })
    }

    pub fn any_i64(&self) -> bool {
        self.columns.iter().any(|c| c.int_width > 0)
    }

    pub fn any_codes(&self) -> bool {
        self.columns.iter().any(|c| c.code_width > 0)
    }

    pub fn n_code_columns(&self) -> u64 {
        self.columns.iter().filter(|c| c.code_width > 0).count() as u64
    }

    /// Estimate the resident and prepared footprint under `opts`.
    pub fn estimate(&self, opts: &EstimateOptions) -> MemoryEstimate {
        let rows = self.rows as u128;
        let cols = self.cols();
        let used = opts.resolved_used_cols(cols);
        let scan = opts.scan_cols.min(cols.saturating_sub(used.min(cols)));
        let projection = opts.resolved_projection(cols);
        let g = self.n_groups() as u128;
        let group_rows = self.group_rows.max(1) as u128;

        // Exact resident table. A projection charges only the selected blocks
        // (issue #294); dictionary/directory metadata is still header-sized.
        let mut storage: u128 = 0;
        let mut projected_per_row: u128 = 0;
        for &c in &projection {
            let col = &self.columns[c];
            storage = storage.saturating_add(rows.saturating_mul(col.value_width as u128));
            storage = storage.saturating_add(rows.saturating_mul(col.int_width as u128));
            storage = storage.saturating_add(rows.saturating_mul(col.code_width as u128));
            storage = storage.saturating_add(col.dictionary_bytes as u128);
            projected_per_row = projected_per_row
                .saturating_add(col.value_width as u128)
                .saturating_add(col.int_width as u128)
                .saturating_add(col.code_width as u128);
        }

        // Row-group directory (`offsets` and `code_offsets`), raw header text
        // and the Vec/Table/String headers that survive `open`.
        let directory = g
            .saturating_mul(8)
            .saturating_mul((cols as u128).saturating_add(self.n_code_columns() as u128))
            .saturating_add(self.header_len as u128)
            .saturating_add((cols as u128).saturating_mul(64))
            .saturating_add(256);

        // One `GroupBuf` (one f64 slice per column) plus decode scratch. A
        // projected read never allocates the unprojected group buffer.
        let buffer_cols = if opts.projection.is_some() {
            projected_per_row
        } else {
            (cols as u128).saturating_mul(8)
        };
        let mut group_buffer = group_rows
            .saturating_mul(buffer_cols)
            .saturating_add(group_rows.saturating_mul(8));
        if self.any_i64() {
            group_buffer = group_buffer.saturating_add(group_rows.saturating_mul(8));
        }
        if self.any_codes() {
            group_buffer = group_buffer.saturating_add(group_rows.saturating_mul(4));
        }

        // Search copies: row-major Sample, SoA transpose, optional GPU f32 and
        // optional mutable preparation copy, all per used cell.
        let mut stride = SEARCH_BYTES_PER_CELL.saturating_add(SOA_BYTES_PER_CELL);
        if opts.simd {
            // SOA_BYTES_PER_CELL already covers the default SIMD transpose; a
            // scalar run still keeps the row-major sample.
        } else {
            stride = stride.saturating_sub(SOA_BYTES_PER_CELL);
        }
        if opts.gpu {
            stride = stride.saturating_add(GPU_F32_BYTES_PER_CELL);
        }
        if opts.mutable_prep {
            stride = stride.saturating_add(MUTABLE_PREP_BYTES_PER_CELL);
        }
        let search_cells = rows
            .saturating_mul(used as u128)
            .saturating_mul(stride as u128);
        // Time and group are read during preparation but are not copied into
        // the search sample; charge one f64 per scan cell (codes are already
        // resident in the table).
        let scan_cells = rows
            .saturating_mul(scan as u128)
            .saturating_mul(SEARCH_BYTES_PER_CELL as u128);
        let fixed = rows.saturating_mul(FIXED_BYTES_PER_ROW as u128);
        let search = search_cells.saturating_add(scan_cells);

        // Optional exact f64 fit image. The default split is 75 % train; the
        // 25 % slack covers an explicit full-train selection.
        let fit_rows = if opts.gpu_fit {
            rows.saturating_mul(3) / 4
        } else {
            0
        };
        let fit_copy = fit_rows
            .saturating_mul(used as u128)
            .saturating_mul(GPU_FIT_BYTES_PER_CELL as u128);

        // #294: the projected device transport pins a bounded ring and keeps
        // one device tile per group. Charge the ring on the host side; the
        // device tile is reported separately (it is not host RSS).
        let group_bytes = group_rows.saturating_mul(projected_per_row);
        let device_tile = if opts.gpu { group_bytes } else { 0 };
        let ring = if opts.gpu {
            group_bytes
                .saturating_mul(RING_SLOTS as u128)
                .min(RING_CAP_BYTES as u128)
        } else {
            0
        };

        let base = storage
            .saturating_add(directory)
            .saturating_add(group_buffer)
            .saturating_add(search)
            .saturating_add(fit_copy)
            .saturating_add(ring)
            .saturating_add(fixed);
        let total = base
            .saturating_mul(SLACK_NUM as u128)
            .saturating_add(SLACK_DEN as u128 - 1)
            / SLACK_DEN as u128;
        let slack = total.saturating_sub(base);

        let (storage_u, mut saturated) = clamp(storage);
        let (directory_u, s) = clamp(directory);
        saturated |= s;
        let (group_buffer_u, s) = clamp(group_buffer);
        saturated |= s;
        let (search_u, s) = clamp(search);
        saturated |= s;
        let (fit_u, s) = clamp(fit_copy);
        saturated |= s;
        let (ring_u, s) = clamp(ring);
        saturated |= s;
        let (tile_u, s) = clamp(device_tile);
        saturated |= s;
        let (base_u, s) = clamp(base);
        saturated |= s;
        let (total_u, s) = clamp(total);
        saturated |= s;
        let (slack_u, s) = clamp(slack);
        saturated |= s;

        MemoryEstimate {
            rows: self.rows,
            cols,
            used_cols: used,
            scan_cols: scan,
            projected_cols: projection.len(),
            storage_bytes: storage_u,
            directory_bytes: directory_u,
            group_buffer_bytes: group_buffer_u,
            search_bytes: search_u,
            fit_copy_bytes: fit_u,
            ring_bytes: ring_u,
            device_tile_bytes: tile_u,
            fixed_bytes: clamp(fixed).0,
            slack_bytes: slack_u,
            total_bytes: total_u,
            base_bytes: base_u,
            saturated,
        }
    }

    /// Largest row count whose sampled estimate fits 60 % of `budget`, using
    /// the same per-row terms as [`StoreShape::estimate`]. Returns `rows` when
    /// the full store fits, so callers never invent a sample that is not
    /// smaller than the data.
    pub fn sample_row_cap(&self, budget: u64, opts: &EstimateOptions) -> usize {
        let cols = self.cols();
        let used = opts.resolved_used_cols(cols);
        let mut stride = SEARCH_BYTES_PER_CELL.saturating_add(SOA_BYTES_PER_CELL);
        if !opts.simd {
            stride = stride.saturating_sub(SOA_BYTES_PER_CELL);
        }
        if opts.gpu {
            stride = stride.saturating_add(GPU_F32_BYTES_PER_CELL);
        }
        if opts.mutable_prep {
            stride = stride.saturating_add(MUTABLE_PREP_BYTES_PER_CELL);
        }
        let search_per_row = (used as u64)
            .saturating_mul(stride)
            .saturating_add(FIXED_BYTES_PER_ROW);
        let table_per_row = self.projected_bytes_per_row(&opts.resolved_projection(cols));
        let fit_per_row = if opts.gpu_fit {
            (used as u64).saturating_mul(6)
        } else {
            0
        };
        let per_row = search_per_row
            .saturating_add(table_per_row)
            .saturating_add(fit_per_row)
            .max(1);
        let usable = budget
            .saturating_mul(SAMPLE_BUDGET_NUM)
            / SAMPLE_BUDGET_DEN;
        let cap = (usable / per_row) as usize;
        cap.min(self.rows)
    }
}

/// Options that turn a [`StoreShape`] into an estimate.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EstimateOptions {
    /// Columns copied into the search sample: variables, targets and weight.
    /// `0` means the whole header (the projection is not known yet).
    pub used_cols: usize,
    /// Time/group columns read during preparation but not copied into the
    /// search sample.
    pub scan_cols: usize,
    /// Resolved read projection (header indices). `None` means the whole
    /// header. This is what the resident table and the ring actually touch,
    /// so storage is charged over these columns only (issue #294).
    pub projection: Option<Vec<usize>>,
    pub gpu: bool,
    /// The GPU arm also materialises an exact `f64` fit image.
    pub gpu_fit: bool,
    pub mutable_prep: bool,
    pub simd: bool,
}

impl Default for EstimateOptions {
    fn default() -> Self {
        EstimateOptions {
            used_cols: 0,
            scan_cols: 0,
            projection: None,
            gpu: false,
            gpu_fit: false,
            mutable_prep: false,
            simd: true,
        }
    }
}

impl EstimateOptions {
    pub fn resolved_used_cols(&self, cols: usize) -> usize {
        if self.used_cols == 0 {
            cols
        } else {
            self.used_cols.min(cols)
        }
    }

    /// The resolved projection as header indices, in bounds and deduplicated.
    pub fn resolved_projection(&self, cols: usize) -> Vec<usize> {
        match &self.projection {
            None => (0..cols).collect(),
            Some(v) => {
                let mut out: Vec<usize> = v.iter().copied().filter(|&c| c < cols).collect();
                out.sort_unstable();
                out.dedup();
                if out.is_empty() {
                    (0..cols).collect()
                } else {
                    out
                }
            }
        }
    }
}

/// One estimate and every term, so text/JSON can report the basis.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MemoryEstimate {
    pub rows: usize,
    pub cols: usize,
    pub used_cols: usize,
    pub scan_cols: usize,
    /// Header columns actually read by the projected path.
    pub projected_cols: usize,
    pub storage_bytes: u64,
    pub directory_bytes: u64,
    pub group_buffer_bytes: u64,
    pub search_bytes: u64,
    pub fit_copy_bytes: u64,
    /// Pinned host ring bytes (#294).
    pub ring_bytes: u64,
    /// One device tile per group (#294), not part of host RSS.
    pub device_tile_bytes: u64,
    pub fixed_bytes: u64,
    pub base_bytes: u64,
    pub slack_bytes: u64,
    pub total_bytes: u64,
    /// True when any term hit `u64::MAX`.
    pub saturated: bool,
}

impl MemoryEstimate {
    pub fn projection_cols(&self) -> usize {
        self.projected_cols
    }

    /// The estimate with an explicit sampled table footprint (used by the
    /// `--sample-rows` path).
    pub fn with_sampled_rows(mut self, rows: usize, shape: &StoreShape, opts: &EstimateOptions) -> Self {
        let rows = rows.min(self.rows);
        let used = opts.resolved_used_cols(self.cols);
        let mut stride = SEARCH_BYTES_PER_CELL.saturating_add(if opts.simd {
            SOA_BYTES_PER_CELL
        } else {
            0
        });
        if opts.gpu {
            stride = stride.saturating_add(GPU_F32_BYTES_PER_CELL);
        }
        if opts.mutable_prep {
            stride = stride.saturating_add(MUTABLE_PREP_BYTES_PER_CELL);
        }
        let projection = opts.resolved_projection(self.cols);
        let table = (rows as u128)
            .saturating_mul(shape.projected_bytes_per_row(&projection) as u128);
        let fixed = (rows as u128).saturating_mul(FIXED_BYTES_PER_ROW as u128);
        let search = (rows as u128)
            .saturating_mul(used as u128)
            .saturating_mul(stride as u128);
        let fit = if opts.gpu_fit {
            let fit_rows = (rows as u128).saturating_mul(3) / 4;
            fit_rows
                .saturating_mul(used as u128)
                .saturating_mul(GPU_FIT_BYTES_PER_CELL as u128)
        } else {
            0
        };
        let directory = self.directory_bytes as u128;
        let buffer = self.group_buffer_bytes as u128;
        let ring = self.ring_bytes as u128;
        let base = table
            .saturating_add(directory)
            .saturating_add(buffer)
            .saturating_add(ring)
            .saturating_add(search)
            .saturating_add(fit)
            .saturating_add(fixed);
        let total = base
            .saturating_mul(SLACK_NUM as u128)
            .saturating_add(SLACK_DEN as u128 - 1)
            / SLACK_DEN as u128;
        self.rows = rows;
        self.storage_bytes = clamp(table).0;
        self.search_bytes = clamp(search).0;
        self.fit_copy_bytes = clamp(fit).0;
        self.fixed_bytes = clamp(fixed).0;
        self.base_bytes = clamp(base).0;
        self.total_bytes = clamp(total).0;
        self.slack_bytes = self.total_bytes.saturating_sub(self.base_bytes);
        self.saturated |= self.total_bytes == u64::MAX;
        self
    }

    /// Human-readable terms; used by the preparation block and refusal.
    pub fn text_line(&self, path: &str) -> String {
        format!(
            "store estimate for '{}': {} rows x {} columns ({} search + {} scan used); \
             storage {} + directory/group {} + search {} + fit {} + fixed {} + slack {} = {}",
            path,
            self.rows,
            self.cols,
            self.used_cols,
            self.scan_cols,
            human(self.storage_bytes),
            human(self.directory_bytes.saturating_add(self.group_buffer_bytes)),
            human(self.search_bytes),
            human(self.fit_copy_bytes),
            human(self.fixed_bytes),
            human(self.slack_bytes),
            human(self.total_bytes)
        )
    }
}

fn clamp(v: u128) -> (u64, bool) {
    if v > u64::MAX as u128 {
        (u64::MAX, true)
    } else {
        (v as u64, false)
    }
}

/// `123.4 MiB`-style formatting shared with the CLI guard.
pub fn human(bytes: u64) -> String {
    const UNITS: [&str; 6] = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{} {}", bytes, UNITS[u])
    } else {
        format!("{:.2} {}", v, UNITS[u])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(rows: usize, group_rows: usize) -> StoreShape {
        StoreShape {
            rows,
            group_rows,
            header_len: 100,
            columns: vec![
                ColumnShape {
                    name: "x".into(),
                    value_width: 8,
                    int_width: 0,
                    code_width: 0,
                    dictionary_bytes: 0,
                },
                ColumnShape {
                    name: "t".into(),
                    value_width: 8,
                    int_width: 8,
                    code_width: 0,
                    dictionary_bytes: 0,
                },
                ColumnShape {
                    name: "g".into(),
                    value_width: 0,
                    int_width: 0,
                    code_width: 4,
                    dictionary_bytes: 100,
                },
            ],
        }
    }

    #[test]
    fn storage_is_exact_for_dtypes_and_codes() {
        let s = shape(20, 7);
        // x 20*8 + t (20*8 + 20*8) + g 20*4 + levels 100.
        assert_eq!(s.storage_bytes(), 20 * 8 + (20 * 8 + 20 * 8) + 20 * 4 + 100);
    }

    #[test]
    fn saturation_is_reported_not_wrapped() {
        let s = StoreShape {
            rows: usize::MAX,
            group_rows: 65_536,
            header_len: u64::MAX,
            columns: vec![ColumnShape {
                name: "x".into(),
                value_width: 8,
                int_width: 8,
                code_width: 4,
                dictionary_bytes: u64::MAX,
            }],
        };
        let e = s.estimate(&EstimateOptions::default());
        assert_eq!(e.total_bytes, u64::MAX);
        assert!(e.saturated);
    }

    #[test]
    fn projection_changes_storage_and_search_terms() {
        let s = shape(1_000, 7);
        let mut opts = EstimateOptions::default();
        let full = s.estimate(&opts);
        opts.used_cols = 1;
        opts.projection = Some(vec![0]);
        opts.gpu = true;
        let projected = s.estimate(&opts);
        assert!(projected.storage_bytes < full.storage_bytes);
        assert_eq!(projected.projected_cols, 1);
        assert!(projected.ring_bytes > 0 && projected.ring_bytes <= RING_CAP_BYTES);
        assert!(projected.search_bytes < full.search_bytes);
    }
}
