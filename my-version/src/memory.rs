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

// ---------------------------------------------------------------------------
// #293: device memory plan
// ---------------------------------------------------------------------------

/// Resident launch width of the score kernels: lanes per candidate.
pub const DEVICE_LANES: usize = 256;
/// Largest share of device memory a run may plan on.
pub const DEVICE_CAP_BYTES: u64 = 11_274_289_152;
/// Free device memory kept back for context growth and fragmentation.
pub const DEVICE_RESERVE_BYTES: u64 = 1 << 30;
/// Cap of the per-candidate tile partials.
pub const DEVICE_PARTIALS_CAP_BYTES: u64 = 64 << 20;
/// Cap of one exact rank or order-statistic wave.
pub const DEVICE_STAT_WAVE_BYTES: u64 = 512 << 20;
const DEVICE_BUFFER_BASE_BYTES: u128 = 1 << 20;
const DEVICE_STAT_ROW_BYTES: u128 = 64;
const DEVICE_STAT_WAVE_TRIALS: u128 = 6;
const DEVICE_TILE_MIN_ROWS: u64 = 512;
const DEVICE_TILE_MAX_ROWS: u64 = 8192;
/// Per-candidate lane carry: 16 B per lane plus 16 B of header.
const DEVICE_CARRY_BYTES: u128 = DEVICE_LANES as u128 * 16 + 16;

/// How the training columns live on the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    Resident,
    Tiled,
    Refused,
}

impl Residency {
    pub fn name(self) -> &'static str {
        match self {
            Residency::Resident => "resident",
            Residency::Tiled => "tiled",
            Residency::Refused => "refused",
        }
    }
}

/// Where one stage runs under the plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageRoute {
    Device,
    Cpu(&'static str),
    Refused,
}

impl StageRoute {
    pub fn name(self) -> &'static str {
        match self {
            StageRoute::Device => "device",
            StageRoute::Cpu(_) => "cpu",
            StageRoute::Refused => "refused",
        }
    }

    pub fn reason(self) -> &'static str {
        match self {
            StageRoute::Device => "",
            StageRoute::Cpu(r) => r,
            StageRoute::Refused => "refused",
        }
    }
}

/// What the device reports once its context exists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    pub free: u64,
    pub total: u64,
    pub l2_bytes: u64,
}

/// Limits the plan works within.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceCaps {
    pub cap: u64,
    pub reserve: u64,
    /// `--gpu-memory`.
    pub user: Option<u64>,
    pub partials_cap: u64,
    /// `RCSR_TEST_GPU_TILE_ROWS`.
    pub tile_rows: Option<usize>,
}

impl DeviceCaps {
    pub fn new(user: Option<u64>) -> Self {
        DeviceCaps {
            cap: DEVICE_CAP_BYTES,
            reserve: DEVICE_RESERVE_BYTES,
            user,
            partials_cap: DEVICE_PARTIALS_CAP_BYTES,
            tile_rows: None,
        }
    }

    /// The caps with the test hooks `RCSR_TEST_GPU_TILE_ROWS` and
    /// `RCSR_TEST_GPU_PARTIALS_CAP_BYTES` applied.
    pub fn from_env(user: Option<u64>) -> Self {
        let mut caps = Self::new(user);
        let read = |name: &str| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.trim().parse::<u64>().ok())
                .filter(|v| *v > 0)
        };
        caps.tile_rows = read("RCSR_TEST_GPU_TILE_ROWS").map(|v| v as usize);
        if let Some(v) = read("RCSR_TEST_GPU_PARTIALS_CAP_BYTES") {
            caps.partials_cap = v;
        }
        caps
    }
}

/// The run as the plan sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceWorkload {
    pub rows: usize,
    pub vars: usize,
    pub candidates: usize,
    pub program_words: usize,
    /// An explicit `--fit-strategy gpu` with `fit_iters > 0`.
    pub device_fit: bool,
    /// The exact rank or order-statistic scorer.
    pub stat: bool,
    pub weights: bool,
    pub label_sign: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceTerms {
    pub data_f32: u64,
    pub side_vectors: u64,
    pub fit_twin: u64,
    pub buffers: u64,
    pub stat_wave: u64,
    pub tile_arena: u64,
    pub tile_pinned: u64,
    pub partials: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceRefusal {
    pub stage: &'static str,
    pub reason: String,
    /// The terms that add up to `required`.
    pub terms: DeviceTerms,
    pub required: u64,
    pub budget: u64,
    pub fixes: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePlan {
    pub workload: DeviceWorkload,
    pub mode: Residency,
    pub budget: u64,
    pub budget_source: &'static str,
    pub cap: u64,
    pub free: u64,
    pub total: u64,
    pub reserve: u64,
    pub required: u64,
    pub terms: DeviceTerms,
    pub l2_bytes: u64,
    pub tile_rows: usize,
    pub waves: usize,
    pub wave_candidates: usize,
    pub fit_route: StageRoute,
    pub stat_route: StageRoute,
    pub refusal: Option<DeviceRefusal>,
    /// A term hit the `u64` ceiling.
    pub saturated: bool,
}

/// Tile height for the tiled mode: half of L2 holds two tiles of `vars` f32
/// columns, rounded down to a power of two inside `[512, 8192]` and to a
/// multiple of `lanes`.
pub fn tile_rows_for(vars: usize, l2_bytes: u64, lanes: usize) -> usize {
    let per_row = 2 * vars.max(1) as u64 * 4;
    let raw = (l2_bytes / per_row).max(1);
    let pow2 = 1u64 << (63 - raw.leading_zeros());
    let mut rows = pow2.clamp(DEVICE_TILE_MIN_ROWS, DEVICE_TILE_MAX_ROWS);
    if lanes > 1 {
        let lanes = lanes as u64;
        rows = (rows / lanes * lanes).max(lanes);
    }
    rows as usize
}

fn clamp_u64(v: u128, saturated: &mut bool) -> u64 {
    if v > u64::MAX as u128 {
        *saturated = true;
        u64::MAX
    } else {
        v as u64
    }
}

fn device_buffer_bytes(w: &DeviceWorkload) -> u128 {
    let p = w.candidates as u128;
    let width = w.program_words as u128;
    let slots = 2 * (2 * p * width * 4);
    let score_and_hash = (12 + 8) * p;
    let fit_region = 5 * p * (4 * width + 96) / 4;
    DEVICE_BUFFER_BASE_BYTES + slots + score_and_hash + fit_region
}

fn plan_core(w: &DeviceWorkload, d: &DeviceInfo, caps: &DeviceCaps) -> DevicePlan {
    let mut sat = false;
    let n = w.rows as u128;
    let v = w.vars as u128;
    let data_f32 = clamp_u64(n * (v + 1) * 4, &mut sat);
    let mut side: u128 = 0;
    if w.label_sign {
        side += n * 12;
    }
    if w.weights {
        side += n * 8;
    }
    let side_vectors = clamp_u64(side, &mut sat);
    let twin_all = clamp_u64(n * (v + 1) * 8, &mut sat);
    let buffers = clamp_u64(device_buffer_bytes(w), &mut sat);
    let stat_wave = if w.stat {
        let min_wave = DEVICE_STAT_WAVE_TRIALS * (n * DEVICE_STAT_ROW_BYTES + (1 << 20));
        clamp_u64(min_wave.max(DEVICE_STAT_WAVE_BYTES as u128), &mut sat)
    } else {
        0
    };

    let free_budget = d.free.saturating_sub(caps.reserve);
    let mut budget = caps.cap;
    let mut source = "cap";
    if free_budget < budget {
        budget = free_budget;
        source = "free";
    }
    if let Some(user) = caps.user {
        if user <= budget {
            budget = user;
            source = "flag";
        }
    }

    let tile_rows = caps
        .tile_rows
        .unwrap_or_else(|| tile_rows_for(w.vars, d.l2_bytes, DEVICE_LANES));
    let sum = |parts: &[u64], sat: &mut bool| -> u64 {
        clamp_u64(parts.iter().map(|p| *p as u128).sum::<u128>(), sat)
    };
    let core = sum(&[data_f32, side_vectors, buffers], &mut sat);
    let core_terms = DeviceTerms {
        data_f32,
        side_vectors,
        buffers,
        ..DeviceTerms::default()
    };
    let twin_terms = DeviceTerms {
        fit_twin: twin_all,
        ..core_terms
    };
    let twin = if w.device_fit || w.stat { twin_all } else { 0 };
    let core_twin = sum(&[core, twin], &mut sat);
    let core_twin_wave = sum(&[core, twin, stat_wave], &mut sat);

    let mut plan = DevicePlan {
        workload: *w,
        mode: Residency::Resident,
        budget,
        budget_source: source,
        cap: caps.cap,
        free: d.free,
        total: d.total,
        reserve: caps.reserve,
        required: core,
        terms: core_terms,
        l2_bytes: d.l2_bytes,
        tile_rows,
        waves: 0,
        wave_candidates: 0,
        fit_route: StageRoute::Cpu("not_requested"),
        stat_route: StageRoute::Cpu("not_requested"),
        refusal: None,
        saturated: false,
    };
    let refusal = |stage: &'static str, reason: &str, required: u64, terms: DeviceTerms| {
        Some(DeviceRefusal {
            stage,
            reason: reason.to_string(),
            terms,
            required,
            budget,
            fixes: Vec::new(),
        })
    };

    if w.rows > i32::MAX as usize {
        plan.mode = Residency::Refused;
        if w.device_fit {
            plan.fit_route = StageRoute::Refused;
        }
        plan.refusal = refusal(
            "eval",
            "rows exceed the kernel's 32-bit row index",
            core,
            core_terms,
        );
        plan.saturated = sat;
        return plan;
    }

    if core <= budget {
        let twin_ok = core_twin <= budget;
        let wave_ok = core_twin_wave <= budget;
        if w.device_fit {
            plan.fit_route = if twin_ok { StageRoute::Device } else { StageRoute::Refused };
        }
        if w.stat {
            plan.stat_route = if wave_ok {
                StageRoute::Device
            } else {
                StageRoute::Cpu("device_memory")
            };
        }
        if w.stat && wave_ok {
            plan.required = core_twin_wave;
            plan.terms.fit_twin = twin_all;
            plan.terms.stat_wave = stat_wave;
        } else if w.device_fit && twin_ok {
            plan.required = core_twin;
            plan.terms.fit_twin = twin_all;
        }
        if w.device_fit && !twin_ok {
            plan.required = core_twin;
            plan.refusal = refusal(
                "fit",
                "the f64 fit twin does not fit beside the resident rows",
                core_twin,
                twin_terms,
            );
        }
        plan.saturated = sat;
        return plan;
    }

    // The resident rows do not fit: plan tiles.
    let arena = clamp_u64(2 * tile_rows as u128 * v * 4, &mut sat);
    let fixed = sum(&[side_vectors, buffers, arena], &mut sat);
    let room = (caps.partials_cap.min(budget.saturating_sub(fixed)) as u128 / DEVICE_CARRY_BYTES) as usize;
    let wave = room.min(w.candidates);
    let tiled_terms = DeviceTerms {
        side_vectors,
        buffers,
        tile_arena: arena,
        tile_pinned: arena,
        ..DeviceTerms::default()
    };
    if budget >= fixed && wave >= 1 {
        let partials = clamp_u64(wave as u128 * DEVICE_CARRY_BYTES, &mut sat);
        plan.mode = Residency::Tiled;
        plan.terms = DeviceTerms { partials, ..tiled_terms };
        plan.required = sum(&[fixed, partials], &mut sat);
        plan.wave_candidates = wave;
        plan.waves = w.candidates.div_ceil(wave);
        if w.stat {
            plan.stat_route = StageRoute::Cpu("device_memory");
        }
        plan.refusal = if w.device_fit {
            plan.fit_route = StageRoute::Refused;
            refusal("fit", "tiled residency has no device fit", core_twin, twin_terms)
        } else {
            refusal("eval", "tiled residency is not available in this build", core, core_terms)
        };
        plan.saturated = sat;
        return plan;
    }

    plan.mode = Residency::Refused;
    if w.device_fit {
        plan.fit_route = StageRoute::Refused;
    }
    let tiled_min = sum(&[fixed, DEVICE_CARRY_BYTES as u64], &mut sat);
    let (required, terms) = if tiled_min < core {
        (tiled_min, tiled_terms)
    } else {
        (core, core_terms)
    };
    plan.required = required;
    plan.refusal = refusal("eval", "the rows fit neither resident nor as tiles", required, terms);
    plan.saturated = sat;
    plan
}

/// Largest `k` in `[1, hi)` for which `fits(k)` holds, or `None`. `fits` must
/// be monotone: true up to some `k` and false after it.
fn largest_fitting(hi: usize, fits: impl Fn(usize) -> bool) -> Option<usize> {
    if hi < 2 || !fits(1) {
        return None;
    }
    let (mut lo, mut top) = (1usize, hi - 1);
    while lo < top {
        let mid = lo + (top - lo).div_ceil(2);
        if fits(mid) {
            lo = mid;
        } else {
            top = mid - 1;
        }
    }
    Some(lo)
}

/// Largest `--vars` count below the workload's that the plan accepts.
pub fn fix_vars(w: &DeviceWorkload, d: &DeviceInfo, caps: &DeviceCaps) -> Option<usize> {
    largest_fitting(w.vars, |k| {
        plan_core(&DeviceWorkload { vars: k, ..*w }, d, caps).refusal.is_none()
    })
}

/// Largest training row count below the workload's that the plan accepts.
pub fn fix_rows(w: &DeviceWorkload, d: &DeviceInfo, caps: &DeviceCaps) -> Option<usize> {
    largest_fitting(w.rows, |r| {
        plan_core(&DeviceWorkload { rows: r, ..*w }, d, caps).refusal.is_none()
    })
}

/// The device plan for a run. Pure: no CUDA, no allocation. A refused stage
/// carries its computed fixes, smallest change first.
pub fn device_plan(w: &DeviceWorkload, d: &DeviceInfo, caps: &DeviceCaps) -> DevicePlan {
    let mut plan = plan_core(w, d, caps);
    let Some(stage) = plan.refusal.as_ref().map(|r| r.stage) else {
        return plan;
    };
    let mut fixes = Vec::new();
    if let Some(k) = fix_vars(w, d, caps) {
        fixes.push(format!("--vars {}", k));
    }
    if let Some(r) = fix_rows(w, d, caps) {
        fixes.push(format!(
            "at most {} training rows (--sample-rows on store input, scaled by the train fraction)",
            r
        ));
    }
    if stage == "fit" {
        fixes.push("--fit-strategy exact".to_string());
    }
    if plan.budget_source == "flag" {
        fixes.push("raise --gpu-memory".to_string());
    } else if d.total.saturating_sub(d.free) > (1 << 30) {
        fixes.push(format!(
            "free device memory: other processes hold {}",
            human(d.total - d.free)
        ));
    }
    fixes.push("--cpu".to_string());
    if let Some(r) = plan.refusal.as_mut() {
        r.fixes = fixes;
    }
    plan
}

fn grouped(v: u64) -> String {
    let digits = v.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(c);
    }
    out
}

fn json_str(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push(' '),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

impl DevicePlan {
    /// The refusal text for stderr and for `backend_reason`.
    pub fn refusal_text(&self) -> Option<String> {
        let r = self.refusal.as_ref()?;
        let w = &self.workload;
        let t = &r.terms;
        let mut parts = vec![format!("rows f32 {}", human(t.data_f32))];
        if t.fit_twin > 0 {
            parts.push(format!("f64 fit twin {}", human(t.fit_twin)));
        }
        if t.side_vectors > 0 {
            parts.push(format!("side vectors {}", human(t.side_vectors)));
        }
        if t.tile_arena > 0 {
            parts.push(format!("tile arena {}", human(t.tile_arena)));
        }
        parts.push(format!("buffers {}", human(t.buffers)));
        let binding = match self.budget_source {
            "flag" => format!("--gpu-memory {}", human(self.budget)),
            "free" => format!(
                "min(cap {}, free {} - reserve {})",
                human(self.cap),
                human(self.free),
                human(self.reserve)
            ),
            _ => format!("cap {}", human(self.cap)),
        };
        Some(format!(
            "GPU memory plan refused the {} stage: {}\n  need {} ({} bytes): {} ({} rows x {} vars, P={})\n  budget {} ({} bytes) = {}\n  smallest fix: {}",
            r.stage,
            r.reason,
            human(r.required),
            grouped(r.required),
            parts.join(" + "),
            grouped(w.rows as u64),
            w.vars,
            grouped(w.candidates as u64),
            human(r.budget),
            grouped(r.budget),
            binding,
            r.fixes.join("; or ")
        ))
    }

    /// One line for `backend_reason`: starts with `device_memory`.
    pub fn refusal_reason(&self) -> Option<String> {
        let r = self.refusal.as_ref()?;
        Some(format!(
            "device_memory: {} stage needs {} bytes, budget {} bytes ({}); smallest fix: {}",
            r.stage,
            r.required,
            r.budget,
            r.reason,
            r.fixes.join("; or ")
        ))
    }

    /// `gpu.memory_plan` in the JSON output.
    pub fn to_json(&self) -> String {
        let t = &self.terms;
        let refusal = match &self.refusal {
            None => "null".to_string(),
            Some(r) => format!(
                "{{\"stage\":{},\"reason\":{},\"required_bytes\":{},\"budget_bytes\":{},\"fixes\":[{}]}}",
                json_str(r.stage),
                json_str(&r.reason),
                r.required,
                r.budget,
                r.fixes.iter().map(|f| json_str(f)).collect::<Vec<_>>().join(",")
            ),
        };
        format!(
            "{{\"mode\":{},\"budget_bytes\":{},\"budget_source\":{},\"free_bytes\":{},\"total_bytes\":{},\"reserve_bytes\":{},\"cap_bytes\":{},\"required_bytes\":{},\"saturated\":{},\"terms\":{{\"data_f32\":{},\"side_vectors\":{},\"fit_twin\":{},\"buffers\":{},\"stat_wave\":{},\"tile_arena\":{},\"tile_pinned\":{},\"partials\":{}}},\"rows\":{},\"vars\":{},\"candidates\":{},\"l2_bytes\":{},\"tile_rows\":{},\"waves\":{},\"wave_candidates\":{},\"fit_route\":{},\"fit_reason\":{},\"stat_route\":{},\"stat_reason\":{},\"refusal\":{}}}",
            json_str(self.mode.name()),
            self.budget,
            json_str(self.budget_source),
            self.free,
            self.total,
            self.reserve,
            self.cap,
            self.required,
            self.saturated,
            t.data_f32,
            t.side_vectors,
            t.fit_twin,
            t.buffers,
            t.stat_wave,
            t.tile_arena,
            t.tile_pinned,
            t.partials,
            self.workload.rows,
            self.workload.vars,
            self.workload.candidates,
            self.l2_bytes,
            self.tile_rows,
            self.waves,
            self.wave_candidates,
            json_str(self.fit_route.name()),
            json_str(self.fit_route.reason()),
            json_str(self.stat_route.name()),
            json_str(self.stat_route.reason()),
            refusal
        )
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

    const MIB: u64 = 1 << 20;

    fn rtx3060(free_mib: u64) -> DeviceInfo {
        DeviceInfo {
            free: free_mib * MIB,
            total: 12_288 * MIB,
            l2_bytes: 3 * MIB,
        }
    }

    fn workload(vars: usize, candidates: usize, device_fit: bool) -> DeviceWorkload {
        DeviceWorkload {
            rows: 4_775_757,
            vars,
            candidates,
            program_words: crate::genome::width_for(30),
            device_fit,
            stat: false,
            weights: false,
            label_sign: false,
        }
    }

    #[test]
    fn device_plan_matrix() {
        let d = rtx3060(11_353);
        let caps = DeviceCaps::new(None);
        for vars in [20usize, 56, 364] {
            for candidates in [1_000usize, 10_000] {
                for fit in [false, true] {
                    let w = workload(vars, candidates, fit);
                    let p = device_plan(&w, &d, &caps);
                    let cell = format!("vars={} P={} fit={}", vars, candidates, fit);
                    assert_eq!(p.mode, Residency::Resident, "{}", cell);
                    assert_eq!(p.budget_source, "free", "{}", cell);
                    assert_eq!(p.budget, 11_353 * MIB - 1024 * MIB, "{}", cell);
                    if fit && vars == 364 {
                        assert_eq!(p.fit_route, StageRoute::Refused, "{}", cell);
                        let r = p.refusal.as_ref().expect(&cell);
                        assert_eq!(r.stage, "fit", "{}", cell);
                        assert!(r.required > r.budget, "{}", cell);
                        assert!(r.fixes.len() >= 2, "{}", cell);
                        assert!(r.fixes.iter().any(|f| f.starts_with("--vars ")), "{}", cell);
                        assert!(r.fixes.iter().any(|f| f == "--fit-strategy exact"), "{}", cell);
                    } else {
                        assert!(p.refusal.is_none(), "{}", cell);
                        assert!(p.required <= p.budget, "{}", cell);
                        let expect = if fit { StageRoute::Device } else { StageRoute::Cpu("not_requested") };
                        assert_eq!(p.fit_route, expect, "{}", cell);
                    }
                }
            }
        }
    }

    #[test]
    fn device_plan_fix_is_tight() {
        let caps = DeviceCaps::new(None);
        let d = rtx3060(11_353);
        let refused = vec![
            (workload(364, 10_000, true), d, caps),
            (workload(364, 1_000, true), d, DeviceCaps::new(Some(2 * 1024 * MIB))),
            (workload(56, 10_000, false), rtx3060(1_100), caps),
            (workload(56, 10_000, true), rtx3060(2_048), caps),
            (
                DeviceWorkload { rows: 1 << 31, ..workload(20, 10_000, false) },
                d,
                caps,
            ),
        ];
        let mut checked = 0;
        for (w, d, caps) in refused {
            let p = device_plan(&w, &d, &caps);
            let r = p.refusal.clone().expect("every case in the list refuses");
            assert!(r.required > r.budget);
            assert!(r.fixes.len() >= 2);
            checked += 1;
            if let Some(k) = fix_vars(&w, &d, &caps) {
                let fit = device_plan(&DeviceWorkload { vars: k, ..w }, &d, &caps);
                assert!(fit.refusal.is_none(), "vars {} must fit", k);
                if k + 1 < w.vars {
                    let over = device_plan(&DeviceWorkload { vars: k + 1, ..w }, &d, &caps);
                    assert!(over.refusal.is_some(), "vars {} must not fit", k + 1);
                }
            }
            if let Some(rows) = fix_rows(&w, &d, &caps) {
                let fit = device_plan(&DeviceWorkload { rows, ..w }, &d, &caps);
                assert!(fit.refusal.is_none(), "rows {} must fit", rows);
                if rows + 1 < w.rows {
                    let over = device_plan(&DeviceWorkload { rows: rows + 1, ..w }, &d, &caps);
                    assert!(over.refusal.is_some(), "rows {} must not fit", rows + 1);
                }
            }
        }
        assert!(checked >= 5, "only {} refusals exercised", checked);
    }

    #[test]
    fn device_plan_fix_names_the_largest_fitting_vars() {
        let p = device_plan(&workload(364, 10_000, true), &rtx3060(11_353), &DeviceCaps::new(None));
        let fixes = p.refusal.unwrap().fixes;
        assert_eq!(fixes[0], "--vars 187");
        assert!(fixes[1].starts_with("at most 24"), "{:?}", fixes);
    }

    #[test]
    fn device_plan_refuses_i32_rows() {
        let d = rtx3060(11_353);
        let caps = DeviceCaps::new(None);
        let w = DeviceWorkload { rows: 1 << 31, ..workload(20, 1_000, false) };
        let p = device_plan(&w, &d, &caps);
        assert_eq!(p.mode, Residency::Refused);
        let r = p.refusal.as_ref().unwrap();
        assert_eq!(r.stage, "eval");
        assert!(r.reason.contains("32-bit"));
        assert!(!p.saturated);
        assert!(r.fixes.len() >= 2);
        let ok = DeviceWorkload { rows: i32::MAX as usize, ..w };
        assert!(device_plan(&ok, &d, &caps).refusal.as_ref().map_or(true, |r| !r.reason.contains("32-bit")));

        let huge = DeviceWorkload { rows: usize::MAX, vars: 364, ..w };
        let p = device_plan(&huge, &d, &caps);
        assert!(p.saturated);
        assert_eq!(p.mode, Residency::Refused);
    }

    #[test]
    fn tile_rows_follow_l2() {
        assert_eq!(tile_rows_for(20, 3 * MIB, 256), 8192);
        assert_eq!(tile_rows_for(56, 3 * MIB, 256), 4096);
        assert_eq!(tile_rows_for(364, 3 * MIB, 256), 1024);
        for vars in [1usize, 7, 20, 56, 200, 364, 5000] {
            for l2 in [MIB, 3 * MIB, 6 * MIB, 40 * MIB] {
                let t = tile_rows_for(vars, l2, 256);
                assert_eq!(t % 256, 0);
                assert!((512..=8192).contains(&t), "{} {} -> {}", vars, l2, t);
            }
        }
    }

    #[test]
    fn device_plan_tiles_under_pressure() {
        let w = workload(56, 10_000, false);
        let p = device_plan(&w, &rtx3060(2_048), &DeviceCaps::new(None));
        assert_eq!(p.mode, Residency::Tiled);
        assert_eq!(p.tile_rows, 4096);
        assert_eq!(p.terms.tile_arena, 2 * 4096 * 56 * 4);
        assert_eq!(p.terms.data_f32, 0);
        assert_eq!(p.wave_candidates, 10_000);
        assert_eq!(p.waves, 1);
        assert!(p.required <= p.budget);
        let capped = DeviceCaps { partials_cap: 1 << 20, ..DeviceCaps::new(None) };
        let p = device_plan(&w, &rtx3060(2_048), &capped);
        assert_eq!(p.wave_candidates, (1 << 20) / 4112);
        assert_eq!(p.waves, 10_000usize.div_ceil(p.wave_candidates));
    }

    #[test]
    fn device_plan_stat_wave_falls_back_to_the_cpu() {
        let mut w = workload(20, 10_000, false);
        w.stat = true;
        let wide = device_plan(&w, &rtx3060(11_353), &DeviceCaps::new(None));
        assert_eq!(wide.stat_route, StageRoute::Device);
        assert!(wide.terms.stat_wave >= DEVICE_STAT_WAVE_BYTES);
        let narrow = device_plan(&w, &rtx3060(2_048), &DeviceCaps::new(None));
        assert_eq!(narrow.mode, Residency::Resident);
        assert_eq!(narrow.stat_route, StageRoute::Cpu("device_memory"));
    }

    #[test]
    fn device_plan_budget_names_the_binding_term() {
        let w = workload(20, 1_000, false);
        let d = rtx3060(11_353);
        assert_eq!(device_plan(&w, &d, &DeviceCaps::new(None)).budget_source, "free");
        assert_eq!(device_plan(&w, &rtx3060(20_000), &DeviceCaps::new(None)).budget_source, "cap");
        let p = device_plan(&w, &d, &DeviceCaps::new(Some(5 * 1024 * MIB)));
        assert_eq!((p.budget_source, p.budget), ("flag", 5 * 1024 * MIB));
    }

    #[test]
    fn device_plan_texts_name_bytes_and_fixes() {
        let p = device_plan(&workload(364, 10_000, true), &rtx3060(11_353), &DeviceCaps::new(None));
        let text = p.refusal_text().unwrap();
        let r = p.refusal.as_ref().unwrap();
        assert!(text.contains("GPU memory plan refused the fit stage"));
        assert!(text.contains(&grouped(r.required)));
        assert!(text.contains("budget"));
        assert!(text.contains("smallest fix: --vars 187"));
        assert!(p.refusal_reason().unwrap().starts_with("device_memory: fit stage"));
        let json = p.to_json();
        assert!(json.starts_with("{\"mode\":\"resident\""));
        assert!(json.contains("\"refusal\":{\"stage\":\"fit\""));
    }
}
