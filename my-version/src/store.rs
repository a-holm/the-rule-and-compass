//! Versioned, out-of-core column store (issue #71).
//!
//! A store is a small binary container around the same column-major image the
//! resident [`crate::data::Table`] uses:
//!
//! ```text
//! magic "RCSRBIN\0" | u32 version | u32 header_len | JSON header | data blocks
//! ```
//!
//! The JSON header is self-describing (names, dtypes, `n_rows`, `group_rows`,
//! categorical level tables and per-column, per-group block offsets), and the
//! blocks are plain little-endian `f64` values. Lower-cardinality numeric and
//! text columns additionally carry a `u32` code block per group, encoded as
//! `f64` so the hot representation has one element type.
//!
//! # Backing policy
//!
//! The measured rule from #70 is a **capacity** rule, not a speed rule: a
//! resident `Vec<f64>` is never slower than a file. [`FileStore`] is therefore
//! only used when the resident estimate does not fit the budget.
//!
//! When it is used, group scans are explicit `pread`s (`read_exact_at`) into
//! reusable [`GroupBuf`]s. That keeps RSS at one group and avoids the
//! `mmap`-RSS trap where touched pages accumulate (and needs
//! `madvise(MADV_DONTNEED)` per group). Sparse gathers never issue a syscall
//! per element: the caller reads the containing group sequentially and filters
//! it. This is the measured "sorted/coalesced and streamed groups: `pread`"
//! branch of the policy. No `libc` dependency is needed.
//!
//! [`RowSelector`](crate::column::RowSelector) is still the selector layer; a
//! sampled search materialises at most a budgeted number of selected rows, and
//! the hash predicate is evaluated during the scan (never stored as an index
//! vector).

use crate::column::GroupBuf;
use crate::data::{Column, DictColumn, Table};
use std::fs::File;
use std::io::{Read, Write};

/// Eight-byte magic; the trailing NUL makes a truncated file easy to reject.
pub const MAGIC: &[u8; 8] = b"RCSRBIN\0";
/// Format version. A mismatch is refused with an actionable message.
///
/// Version 1 is the original all-`f64` layout. Issue #127 adds an optional
/// per-column `"dtype"` field *without* bumping the version: a v1 header
/// without `dtype` is the old all-`f64` store, and a header with
/// `"dtype":"i64"` stores that column's blocks as little-endian `i64`. This
/// keeps every existing store readable while new stores can be
/// nanosecond-safe.
pub const VERSION: u32 = 1;

/// Storage type of one column's value blocks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    I64,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::I64 => "i64",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "f64" => Some(DType::F64),
            "i64" => Some(DType::I64),
            _ => None,
        }
    }
}
/// Default row group, sized for 512 KiB per `f64` column (the measured I/O
/// block). It is a streaming unit, not an L2 tile.
pub const DEFAULT_GROUP_ROWS: usize = 65_536;

/// Measured resident cost per cell when the engine prepares a search sample
/// (1.75 KiB/row for six columns in #70), used to turn a byte budget into a row
/// cap. It is deliberately pessimistic: the cost is copies, not the cells.
pub const RESIDENT_BYTES_PER_CELL: u64 = 300;

#[derive(Clone, Debug, PartialEq)]
struct ColumnMeta {
    name: String,
    numeric: bool,
    /// Value-block type. `i64` is used for the time column only.
    dtype: DType,
    levels: Vec<String>,
    /// `true` when the column has a dictionary and therefore a code block.
    has_codes: bool,
    /// Offsets relative to the start of the data section, one per group.
    offsets: Vec<u64>,
    code_offsets: Vec<u64>,
}

/// Parsed header of a store. Public so callers can budget before reading data.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub n_rows: usize,
    pub n_cols: usize,
    pub group_rows: usize,
    columns: Vec<ColumnMeta>,
}

impl Header {
    pub fn n_groups(&self) -> usize {
        if self.n_rows == 0 {
            0
        } else {
            (self.n_rows + self.group_rows - 1) / self.group_rows
        }
    }

    pub fn names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    pub fn numeric(&self) -> Vec<bool> {
        self.columns.iter().map(|c| c.numeric).collect()
    }

    /// Value-block type per column, in header order.
    pub fn dtypes(&self) -> Vec<DType> {
        self.columns.iter().map(|c| c.dtype).collect()
    }

    /// Byte width of one value cell in `col`.
    fn value_width(&self, col: usize) -> u64 {
        match self.columns[col].dtype {
            DType::F64 | DType::I64 => 8,
        }
    }

    pub fn group_len(&self, group: usize) -> usize {
        let lo = group.saturating_mul(self.group_rows);
        (self.n_rows.saturating_sub(lo)).min(self.group_rows)
    }
}

/// A healthy store on disk: header plus an open file handle.
#[derive(Debug)]
pub struct FileStore {
    file: File,
    header: Header,
    data_start: u64,
    path: String,
}

impl FileStore {
    /// Open and validate the magic, version, header and every block offset.
    pub fn open(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("{}: {}", path, e))?;
        let file_len = file
            .metadata()
            .map_err(|e| format!("{}: {}", path, e))?
            .len();
        let mut magic = [0u8; 8];
        file.read_exact(&mut magic)
            .map_err(|_| format!("{}: not an rcsr column store (file is too short)", path))?;
        if &magic != MAGIC {
            return Err(format!(
                "{}: not an rcsr column store (bad magic; run `rcsr import` first)",
                path
            ));
        }
        let version = read_u32(&mut file, path)?;
        if version != VERSION {
            return Err(format!(
                "{}: column store version {} is not supported by this build (reads version {}); \
                 re-run `rcsr import` to rewrite the file",
                path, version, VERSION
            ));
        }
        let header_len = read_u32(&mut file, path)? as u64;
        if header_len == 0 || header_len > 64 * 1024 * 1024 {
            return Err(format!(
                "{}: corrupt column store header length {}",
                path, header_len
            ));
        }
        let mut raw = vec![0u8; header_len as usize];
        file.read_exact(&mut raw)
            .map_err(|_| format!("{}: truncated column store header", path))?;
        let text = std::str::from_utf8(&raw)
            .map_err(|_| format!("{}: column store header is not UTF-8", path))?;
        let header = parse_header(text).map_err(|e| format!("{}: {}", path, e))?;
        let data_start = 16 + header_len;
        let mut store = FileStore {
            file,
            header,
            data_start,
            path: path.to_string(),
        };
        store.validate_blocks(file_len)?;
        Ok(store)
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    pub fn path(&self) -> &str {
        &self.path
    }

    pub fn n_rows(&self) -> usize {
        self.header.n_rows
    }

    pub fn n_cols(&self) -> usize {
        self.header.n_cols
    }

    pub fn group_rows(&self) -> usize {
        self.header.group_rows
    }

    pub fn n_groups(&self) -> usize {
        self.header.n_groups()
    }

    /// Bytes a resident `Table` plus its search preparation would need. The
    /// estimate is intentionally conservative (see [`RESIDENT_BYTES_PER_CELL`]).
    pub fn estimated_resident_bytes(&self) -> u64 {
        resident_estimate(self.header.n_rows, self.header.n_cols)
    }

    fn validate_blocks(&mut self, file_len: u64) -> Result<(), String> {
        let n_groups = self.header.n_groups();
        for col in &self.header.columns {
            if col.offsets.len() != n_groups {
                return Err(format!(
                    "{}: column '{}' has {} group offsets but the header says {} groups",
                    self.path,
                    col.name,
                    col.offsets.len(),
                    n_groups
                ));
            }
            if col.has_codes && col.code_offsets.len() != n_groups {
                return Err(format!(
                    "{}: column '{}' has {} code offsets but the header says {} groups",
                    self.path,
                    col.name,
                    col.code_offsets.len(),
                    n_groups
                ));
            }
        }
        for g in 0..n_groups {
            let len = self.header.group_len(g) as u64;
            for (c, col) in self.header.columns.iter().enumerate() {
                let values_end =
                    self.data_start + col.offsets[g] + len * self.header.value_width(c);
                if values_end > file_len {
                    return Err(format!(
                        "{}: truncated data: group {} column '{}' ends at byte {} but the file is {} bytes",
                        self.path, g, col.name, values_end, file_len
                    ));
                }
                if col.has_codes {
                    let codes_end = self.data_start + col.code_offsets[g] + len * 8;
                    if codes_end > file_len {
                        return Err(format!(
                            "{}: truncated codes: group {} column '{}' ends at byte {} but the file is {} bytes",
                            self.path, g, col.name, codes_end, file_len
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Read one group into a caller-owned buffer, reusing its allocation.
    ///
    /// Values only. Dictionary codes are read separately, because only text or
    /// low-cardinality columns have them.
    pub fn load_group<'b>(
        &self,
        group: usize,
        buf: &'b mut GroupBuf,
    ) -> Result<crate::column::GroupView<'b>, String> {
        let mut scratch = Vec::new();
        self.load_group_with(group, buf, &mut scratch)
    }

    /// [`FileStore::load_group`] with a reusable decode scratch buffer.
    pub fn load_group_with<'b>(
        &self,
        group: usize,
        buf: &'b mut GroupBuf,
        scratch: &mut Vec<u8>,
    ) -> Result<crate::column::GroupView<'b>, String> {
        if group >= self.header.n_groups() {
            return Err(format!(
                "{}: group {} is out of range ({} groups)",
                self.path,
                group,
                self.header.n_groups()
            ));
        }
        let len = self.header.group_len(group);
        buf.prepare(len, self.header.n_cols);
        for (c, col) in self.header.columns.iter().enumerate() {
            let dst = buf.col_mut(c);
            let off = self.data_start + col.offsets[group];
            match col.dtype {
                DType::F64 => read_f64_block(&self.file, &self.path, off, dst, scratch)?,
                DType::I64 => {
                    let mut ints = vec![0i64; len];
                    read_i64_block(&self.file, &self.path, off, &mut ints, scratch)?;
                    for (dst, v) in dst.iter_mut().zip(ints) {
                        *dst = if v == i64::MIN { f64::NAN } else { v as f64 };
                    }
                }
            }
        }
        Ok(buf.view_ref())
    }

    /// Read one group's exact `i64` values for `col` (the time column), or an
    /// empty vector when the column is stored as `f64`.
    pub fn read_i64(&self, group: usize, col: usize) -> Result<Vec<i64>, String> {
        let meta = self
            .header
            .columns
            .get(col)
            .ok_or_else(|| format!("{}: column {} is out of range", self.path, col))?;
        if meta.dtype != DType::I64 {
            return Ok(Vec::new());
        }
        let len = self.header.group_len(group);
        let mut out = vec![0i64; len];
        let mut scratch = Vec::new();
        read_i64_block(
            &self.file,
            &self.path,
            self.data_start + meta.offsets[group],
            &mut out,
            &mut scratch,
        )?;
        Ok(out)
    }

    /// Decode a group's dictionary codes for `col`, as `u32` values in a fresh
    /// vector. Empty when the column has no dictionary.
    pub fn read_codes(&self, group: usize, col: usize) -> Result<Vec<u32>, String> {
        let meta = self
            .header
            .columns
            .get(col)
            .ok_or_else(|| format!("{}: column {} is out of range", self.path, col))?;
        if !meta.has_codes {
            return Ok(Vec::new());
        }
        let len = self.header.group_len(group);
        let mut vals = vec![0.0f64; len];
        let mut scratch = Vec::new();
        read_f64_block(
            &self.file,
            &self.path,
            self.data_start + meta.code_offsets[group],
            &mut vals,
            &mut scratch,
        )?;
        Ok(vals
            .into_iter()
            .map(|v| {
                if v.is_finite() && v >= 0.0 {
                    v as u32
                } else {
                    0
                }
            })
            .collect())
    }

    /// Load the whole store into the resident column-major image. This is the
    /// exact bridge for fixtures that fit: from here every path is identical to
    /// the CSV path.
    pub fn to_table(&self) -> Result<Table, String> {
        let n = self.header.n_rows;
        let mut values: Vec<Vec<f64>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.numeric {
                    Vec::with_capacity(n)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut codes: Vec<Option<Vec<u32>>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.has_codes {
                    Some(Vec::with_capacity(n))
                } else {
                    None
                }
            })
            .collect();
        let mut ints: Vec<Option<Vec<i64>>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.dtype == DType::I64 {
                    Some(Vec::with_capacity(n))
                } else {
                    None
                }
            })
            .collect();
        let mut buf = GroupBuf::new();
        let mut scratch = Vec::new();
        for g in 0..self.header.n_groups() {
            let view = self.load_group_with(g, &mut buf, &mut scratch)?;
            for (c, col) in self.header.columns.iter().enumerate() {
                if col.numeric {
                    values[c].extend_from_slice(view.col(c));
                }
                if col.dtype == DType::I64 {
                    ints[c].as_mut().unwrap().extend(self.read_i64(g, c)?);
                }
                if col.has_codes {
                    let part = self.read_codes(g, c)?;
                    codes[c].as_mut().unwrap().extend(part);
                }
            }
        }
        let columns = self
            .header
            .columns
            .iter()
            .enumerate()
            .map(|(c, col)| Column {
                values: std::mem::take(&mut values[c]),
                ints: ints[c].take(),
                dict: codes[c].take().map(|codes| DictColumn {
                    codes,
                    levels: col.levels.clone(),
                }),
            })
            .collect();
        Ok(Table::from_parts(
            self.header.names(),
            self.header.numeric(),
            columns,
            n,
            0,
        ))
    }

    /// Materialise at most `max_rows` rows into a resident [`Table`].
    ///
    /// Rows are chosen by the pinned one-multiply hash (the measured cheap
    /// selector) during one group-wise scan; no row-id vector is built. Rows
    /// stay in file order, so `Range`/temporal slices keep their meaning on the
    /// sample. The result is a *sample*, which the caller must report: it is
    /// not a full-data pass.
    pub fn sample_table(&self, max_rows: usize, seed: u64) -> Result<Table, String> {
        if max_rows >= self.header.n_rows {
            return self.to_table();
        }
        let max_rows = max_rows.max(1);
        let pct = max_rows as f64 / self.header.n_rows as f64;
        let threshold = (pct.clamp(0.0, 1.0) * u64::MAX as f64) as u64;
        let mut values: Vec<Vec<f64>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.numeric {
                    Vec::with_capacity(max_rows)
                } else {
                    Vec::new()
                }
            })
            .collect();
        let mut codes: Vec<Option<Vec<u32>>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.has_codes {
                    Some(Vec::with_capacity(max_rows))
                } else {
                    None
                }
            })
            .collect();
        let mut ints: Vec<Option<Vec<i64>>> = self
            .header
            .columns
            .iter()
            .map(|c| {
                if c.dtype == DType::I64 {
                    Some(Vec::with_capacity(max_rows))
                } else {
                    None
                }
            })
            .collect();
        let mut buf = GroupBuf::new();
        let mut scratch = Vec::new();
        let mut selected_rows = 0usize;
        for g in 0..self.header.n_groups() {
            let view = self.load_group_with(g, &mut buf, &mut scratch)?;
            let base = g * self.header.group_rows;
            let keep: Vec<usize> = (0..view.len())
                .filter(|&r| {
                    crate::column::hash_row((base + r) as u64, seed) < threshold
                })
                .collect();
            selected_rows += keep.len();
            for (c, col) in self.header.columns.iter().enumerate() {
                if col.numeric {
                    for &r in &keep {
                        values[c].push(view.col(c)[r]);
                    }
                }
                if col.dtype == DType::I64 {
                    let part = self.read_i64(g, c)?;
                    let dst = ints[c].as_mut().unwrap();
                    for &r in &keep {
                        dst.push(part[r]);
                    }
                }
            }
            for (c, col) in self.header.columns.iter().enumerate() {
                if !col.has_codes {
                    continue;
                }
                let part = self.read_codes(g, c)?;
                let dst = codes[c].as_mut().unwrap();
                for &r in &keep {
                    dst.push(part[r]);
                }
            }
        }
        let columns = self
            .header
            .columns
            .iter()
            .enumerate()
            .map(|(c, col)| Column {
                values: std::mem::take(&mut values[c]),
                ints: ints[c].take(),
                dict: codes[c].take().map(|codes| DictColumn {
                    codes,
                    levels: col.levels.clone(),
                }),
            })
            .collect();
        Ok(Table::from_parts(
            self.header.names(),
            self.header.numeric(),
            columns,
            selected_rows,
            0,
        ))
    }
}

/// True when `path` is meant to be a column store (by extension or magic).
pub fn is_store_path(path: &str) -> bool {
    if path.to_ascii_lowercase().ends_with(".rcsb") {
        return true;
    }
    let mut magic = [0u8; 8];
    File::open(path)
        .and_then(|mut f| f.read_exact(&mut magic))
        .map(|_| &magic == MAGIC)
        .unwrap_or(false)
}

/// Estimated resident footprint of a table plus search preparation.
pub fn resident_estimate(n_rows: usize, n_cols: usize) -> u64 {
    (n_rows as u64)
        .saturating_mul(n_cols.max(1) as u64)
        .saturating_mul(RESIDENT_BYTES_PER_CELL)
}

/// How many rows fit in a memory budget for a sampled search. The search holds
/// several copies while preparing, so only ~60 % of the budget is offered.
pub fn sample_row_cap(budget: u64, n_cols: usize) -> usize {
    let usable = budget.saturating_mul(6) / 10;
    let per_row = (n_cols.max(1) as u64).saturating_mul(RESIDENT_BYTES_PER_CELL);
    (usable / per_row).max(1_000) as usize
}

/// Detected memory budget from the kernel, in bytes:
/// `min(MemAvailable, address-space limit, cgroup limit) * 3/4`.
///
/// The address-space limit is what `ulimit -v` sets, and the cgroup limit is
/// what a container or `systemd-run --scope -p MemoryMax=` sets. Both are
/// honoured even when `--max-memory` is not passed, which is what keeps an
/// out-of-core store from silently becoming an OOM kill.
pub fn detected_memory_budget() -> Option<u64> {
    let available = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let rest = line.strip_prefix("MemAvailable:")?;
                let kb = rest.split_whitespace().next()?.parse::<u64>().ok()?;
                Some(kb.saturating_mul(1024))
            })
        });
    let hard = detected_hard_budget();
    match (available, hard) {
        (Some(a), Some(h)) => Some(a.min(h).saturating_mul(3) / 4),
        (Some(a), None) => Some(a.saturating_mul(3) / 4),
        (None, Some(h)) => Some(h.saturating_mul(3) / 4),
        (None, None) => None,
    }
}

/// A hard cap that the process cannot exceed without being killed: `ulimit -v`
/// (`Max address space`) or a cgroup memory limit. Unlike
/// [`detected_memory_budget`] this ignores the soft `MemAvailable` figure, so
/// it can be used to auto-refuse a CSV only when there really is a wall.
pub fn detected_hard_budget() -> Option<u64> {
    let mut budget: Option<u64> = None;
    let mut take = |v: u64| {
        if v > 0 {
            budget = Some(budget.map_or(v, |b| b.min(v)));
        }
    };
    if let Ok(text) = std::fs::read_to_string("/proc/self/limits") {
        for line in text.lines() {
            if line.starts_with("Max address space") {
                if let Some(v) = line.split_whitespace().nth(3) {
                    if v != "unlimited" {
                        if let Ok(bytes) = v.parse::<u64>() {
                            take(bytes);
                        }
                    }
                }
            }
        }
    }
    for path in [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ] {
        if let Ok(text) = std::fs::read_to_string(path) {
            let text = text.trim();
            if text != "max" {
                if let Ok(bytes) = text.parse::<u64>() {
                    take(bytes);
                }
            }
        }
    }
    budget
}

// ---------------------------------------------------------------------------
// Writing
// ---------------------------------------------------------------------------

/// Write `table` as a version-1 store at `path`.
///
/// A column that carries an exact `i64` time view (see
/// [`crate::data::Table::ints`]) is written as an `i64` block, so nanosecond
/// timestamps round-trip exactly. Everything else is `f64` as before.
pub fn write_store(path: &str, table: &Table, group_rows: usize) -> Result<(), String> {
    write_store_with_time(path, table, group_rows, None)
}

/// [`write_store`] with an explicit time column (`rcsr import --time <col>`).
/// When `time_col` is `None` the first column with an exact `i64` view is used;
/// with no such column the store is the historical all-`f64` layout.
pub fn write_store_with_time(
    path: &str,
    table: &Table,
    group_rows: usize,
    time_col: Option<usize>,
) -> Result<(), String> {
    let group_rows = group_rows.max(1);
    let n_rows = table.len();
    let n_cols = table.n_cols();
    if n_rows == 0 {
        return Err("cannot write a column store with no rows".to_string());
    }
    let time_col = match time_col {
        Some(c) => {
            if c >= n_cols {
                return Err(format!(
                    "time column {} is out of range ({} columns)",
                    c, n_cols
                ));
            }
            if table.ints(c).is_none() {
                return Err(format!(
                    "time column '{}' is not an integer timestamp; i64 storage needs whole-number cells",
                    table.names[c]
                ));
            }
            Some(c)
        }
        None => (0..n_cols).find(|&c| table.ints(c).is_some()),
    };
    let n_groups = (n_rows + group_rows - 1) / group_rows;
    let mut columns: Vec<ColumnMeta> = (0..n_cols)
        .map(|c| {
            let dict = table.dict(c);
            ColumnMeta {
                name: table.names[c].clone(),
                numeric: table.numeric[c],
                dtype: if Some(c) == time_col {
                    DType::I64
                } else {
                    DType::F64
                },
                levels: dict.map(|d| d.levels.clone()).unwrap_or_default(),
                has_codes: dict.is_some(),
                offsets: Vec::with_capacity(n_groups),
                code_offsets: Vec::with_capacity(n_groups),
            }
        })
        .collect();
    // Offsets are relative to the data section, so the header length does not
    // feed back into the offsets (no fixed-point iteration needed).
    let mut cursor = 0u64;
    for g in 0..n_groups {
        let len = group_len(n_rows, group_rows, g) as u64;
        for col in &mut columns {
            col.offsets.push(cursor);
            cursor = cursor.saturating_add(len * 8);
            if col.has_codes {
                col.code_offsets.push(cursor);
                cursor = cursor.saturating_add(len * 8);
            }
        }
    }
    let dtypes: Vec<DType> = columns.iter().map(|c| c.dtype).collect();
    let header = Header {
        n_rows,
        n_cols,
        group_rows,
        columns,
    };
    let json = header_json(&header);
    if json.len() > u32::MAX as usize {
        return Err("column store header is too large".to_string());
    }
    let mut file = File::create(path).map_err(|e| format!("{}: {}", path, e))?;
    file.write_all(MAGIC)
        .and_then(|_| file.write_all(&VERSION.to_le_bytes()))
        .and_then(|_| file.write_all(&(json.len() as u32).to_le_bytes()))
        .and_then(|_| file.write_all(json.as_bytes()))
        .map_err(|e| format!("{}: {}", path, e))?;

    // Values and codes are emitted group by group, column by column, in the
    // order the offsets describe.
    let mut bytes: Vec<u8> = Vec::new();
    for g in 0..n_groups {
        let lo = g * group_rows;
        let hi = (lo + group_rows).min(n_rows);
        for c in 0..n_cols {
            bytes.clear();
            bytes.reserve((hi - lo) * 8);
            match dtypes[c] {
                DType::F64 => {
                    let values = table.col(c);
                    for r in lo..hi {
                        let v = values.get(r).copied().unwrap_or(f64::NAN);
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                DType::I64 => {
                    let ints = table.ints(c).ok_or_else(|| {
                        format!("column '{}' has no i64 view", table.names[c])
                    })?;
                    for r in lo..hi {
                        bytes.extend_from_slice(&ints.get(r).copied().unwrap_or(i64::MIN).to_le_bytes());
                    }
                }
            }
            file.write_all(&bytes)
                .map_err(|e| format!("{}: {}", path, e))?;
            let Some(dict) = table.dict(c) else {
                continue;
            };
            bytes.clear();
            bytes.reserve((hi - lo) * 8);
            for r in lo..hi {
                bytes.extend_from_slice(&(dict.codes[r] as f64).to_le_bytes());
            }
            file.write_all(&bytes)
                .map_err(|e| format!("{}: {}", path, e))?;
        }
    }
    file.flush().map_err(|e| format!("{}: {}", path, e))?;
    Ok(())
}

fn group_len(n_rows: usize, group_rows: usize, group: usize) -> usize {
    let lo = group.saturating_mul(group_rows);
    (n_rows.saturating_sub(lo)).min(group_rows)
}

fn header_json(h: &Header) -> String {
    let mut cols = String::new();
    for (i, c) in h.columns.iter().enumerate() {
        if i > 0 {
            cols.push(',');
        }
        let levels = c
            .levels
            .iter()
            .map(|s| jstr(s))
            .collect::<Vec<_>>()
            .join(",");
        let offsets = c
            .offsets
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        let code_offsets = c
            .code_offsets
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(",");
        cols.push_str(&format!(
            "{{\"name\":{},\"numeric\":{},\"dtype\":\"{}\",\"levels\":[{}],\"has_codes\":{},\"offsets\":[{}],\"code_offsets\":[{}]}}",
            jstr(&c.name),
            c.numeric,
            c.dtype.name(),
            levels,
            c.has_codes,
            offsets,
            code_offsets
        ));
    }
    format!(
        "{{\"format\":\"rcsr-column-store\",\"version\":{},\"n_rows\":{},\"n_cols\":{},\"group_rows\":{},\"columns\":[{}]}}",
        VERSION, h.n_rows, h.n_cols, h.group_rows, cols
    )
}

fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for ch in s.chars() {
        match ch {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ---------------------------------------------------------------------------
// Reading helpers
// ---------------------------------------------------------------------------

fn read_u32(file: &mut File, path: &str) -> Result<u32, String> {
    let mut b = [0u8; 4];
    file.read_exact(&mut b)
        .map_err(|_| format!("{}: truncated column store header", path))?;
    Ok(u32::from_le_bytes(b))
}

fn read_f64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [f64],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    scratch.resize(bytes, 0);
    read_exact_at(file, &mut scratch[..], offset).map_err(|e| {
        format!(
            "{}: reading {} bytes at offset {}: {}",
            path, bytes, offset, e
        )
    })?;
    for (dst, src) in out.iter_mut().zip(scratch.chunks_exact(8)) {
        let mut b = [0u8; 8];
        b.copy_from_slice(src);
        *dst = f64::from_le_bytes(b);
    }
    Ok(())
}

fn read_i64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [i64],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    scratch.resize(bytes, 0);
    read_exact_at(file, &mut scratch[..], offset).map_err(|e| {
        format!(
            "{}: reading {} bytes at offset {}: {}",
            path, bytes, offset, e
        )
    })?;
    for (dst, src) in out.iter_mut().zip(scratch.chunks_exact(8)) {
        let mut b = [0u8; 8];
        b.copy_from_slice(src);
        *dst = i64::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {    use std::os::unix::fs::FileExt;
    file.read_exact_at(buf, offset)
}

#[cfg(windows)]
fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::os::windows::fs::FileExt;
    let mut done = 0usize;
    while done < buf.len() {
        let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::UnexpectedEof,
                "short read",
            ));
        }
        done += n;
    }
    Ok(())
}

#[cfg(not(any(unix, windows)))]
fn read_exact_at(mut file: &File, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
    use std::io::{Seek, SeekFrom};
    file.seek(SeekFrom::Start(offset))?;
    file.read_exact(buf)
}

// ---------------------------------------------------------------------------
// A deliberately small JSON parser for our own header
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
enum Json {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    Arr(Vec<Json>),
    Obj(Vec<(String, Json)>),
}

impl Json {
    fn get(&self, key: &str) -> Option<&Json> {
        match self {
            Json::Obj(items) => items.iter().find(|(k, _)| k == key).map(|(_, v)| v),
            _ => None,
        }
    }
    fn as_usize(&self) -> Option<usize> {
        match self {
            Json::Num(v) if *v >= 0.0 && v.fract() == 0.0 => Some(*v as usize),
            _ => None,
        }
    }
    fn as_str(&self) -> Option<&str> {
        match self {
            Json::Str(s) => Some(s),
            _ => None,
        }
    }
    fn as_bool(&self) -> Option<bool> {
        match self {
            Json::Bool(b) => Some(*b),
            _ => None,
        }
    }
    fn as_arr(&self) -> Option<&[Json]> {
        match self {
            Json::Arr(v) => Some(v),
            _ => None,
        }
    }
}

fn parse_header(text: &str) -> Result<Header, String> {
    let mut p = JsonParser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let root = p.value()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err("trailing bytes in column store header".to_string());
    }
    let n_rows = root
        .get("n_rows")
        .and_then(Json::as_usize)
        .ok_or_else(|| "header is missing n_rows".to_string())?;
    let n_cols = root
        .get("n_cols")
        .and_then(Json::as_usize)
        .ok_or_else(|| "header is missing n_cols".to_string())?;
    let group_rows = root
        .get("group_rows")
        .and_then(Json::as_usize)
        .filter(|&g| g > 0)
        .ok_or_else(|| "header is missing a positive group_rows".to_string())?;
    let cols = root
        .get("columns")
        .and_then(Json::as_arr)
        .ok_or_else(|| "header is missing columns".to_string())?;
    if cols.len() != n_cols {
        return Err(format!(
            "header says {} columns but lists {}",
            n_cols,
            cols.len()
        ));
    }
    let mut columns = Vec::with_capacity(n_cols);
    for c in cols {
        let name = c
            .get("name")
            .and_then(Json::as_str)
            .ok_or_else(|| "column is missing name".to_string())?
            .to_string();
        let numeric = c
            .get("numeric")
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("column '{}' is missing numeric", name))?;
        // Absent in the original v1 header, which means all-`f64`.
        let dtype = match c.get("dtype") {
            None => DType::F64,
            Some(v) => {
                let s = v
                    .as_str()
                    .ok_or_else(|| format!("column '{}' has a non-string dtype", name))?;
                DType::parse(s).ok_or_else(|| {
                    format!(
                        "column '{}' has unknown dtype '{}' (expected f64 or i64)",
                        name, s
                    )
                })?
            }
        };
        if dtype == DType::I64 && !numeric {
            return Err(format!(
                "column '{}' has dtype i64 but numeric:false",
                name
            ));
        }
        let levels = c
            .get("levels")
            .and_then(Json::as_arr)
            .ok_or_else(|| format!("column '{}' is missing levels", name))?
            .iter()
            .map(|v| {
                v.as_str()
                    .map(|s| s.to_string())
                    .ok_or_else(|| format!("column '{}' has a non-string level", name))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let has_codes = c
            .get("has_codes")
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("column '{}' is missing has_codes", name))?;
        let offsets = parse_offsets(c.get("offsets"), &name)?;
        let code_offsets = parse_offsets(c.get("code_offsets"), &name)?;
        columns.push(ColumnMeta {
            name,
            numeric,
            dtype,
            levels,
            has_codes,
            offsets,
            code_offsets,
        });
    }
    if n_rows == 0 {
        return Err("column store has no rows".to_string());
    }
    Ok(Header {
        n_rows,
        n_cols,
        group_rows,
        columns,
    })
}

fn parse_offsets(v: Option<&Json>, name: &str) -> Result<Vec<u64>, String> {
    v.and_then(Json::as_arr)
        .ok_or_else(|| format!("column '{}' is missing offsets", name))?
        .iter()
        .map(|x| match x {
            Json::Num(n) if *n >= 0.0 && n.fract() == 0.0 => Ok(*n as u64),
            _ => Err(format!("column '{}' has a non-integer offset", name)),
        })
        .collect()
}

struct JsonParser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> JsonParser<'a> {
    fn ws(&mut self) {
        while self.pos < self.bytes.len() && self.bytes[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn value(&mut self) -> Result<Json, String> {
        self.ws();
        let b = *self
            .bytes
            .get(self.pos)
            .ok_or_else(|| "unexpected end of JSON".to_string())?;
        match b {
            b'{' => self.object(),
            b'[' => self.array(),
            b'"' => self.string().map(Json::Str),
            b't' => {
                self.expect(b"true")?;
                Ok(Json::Bool(true))
            }
            b'f' => {
                self.expect(b"false")?;
                Ok(Json::Bool(false))
            }
            b'n' => {
                self.expect(b"null")?;
                Ok(Json::Null)
            }
            b'-' | b'0'..=b'9' => self.number(),
            other => Err(format!("unexpected byte '{}' in JSON header", other as char)),
        }
    }

    fn expect(&mut self, lit: &[u8]) -> Result<(), String> {
        if self.bytes[self.pos..].starts_with(lit) {
            self.pos += lit.len();
            Ok(())
        } else {
            Err("invalid JSON literal".to_string())
        }
    }

    fn object(&mut self) -> Result<Json, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.bytes.get(self.pos) == Some(&b'}') {
            self.pos += 1;
            return Ok(Json::Obj(items));
        }
        loop {
            self.ws();
            let key = self.string()?;
            self.ws();
            if self.bytes.get(self.pos) != Some(&b':') {
                return Err("expected ':' in JSON header".to_string());
            }
            self.pos += 1;
            let val = self.value()?;
            items.push((key, val));
            self.ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Json::Obj(items));
                }
                _ => return Err("expected ',' or '}' in JSON header".to_string()),
            }
        }
    }

    fn array(&mut self) -> Result<Json, String> {
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.bytes.get(self.pos) == Some(&b']') {
            self.pos += 1;
            return Ok(Json::Arr(items));
        }
        loop {
            items.push(self.value()?);
            self.ws();
            match self.bytes.get(self.pos) {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Json::Arr(items));
                }
                _ => return Err("expected ',' or ']' in JSON header".to_string()),
            }
        }
    }

    fn string(&mut self) -> Result<String, String> {
        if self.bytes.get(self.pos) != Some(&b'"') {
            return Err("expected a JSON string".to_string());
        }
        self.pos += 1;
        let mut out = String::new();
        while let Some(&b) = self.bytes.get(self.pos) {
            self.pos += 1;
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let esc = *self
                        .bytes
                        .get(self.pos)
                        .ok_or_else(|| "truncated JSON escape".to_string())?;
                    self.pos += 1;
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000c}'),
                        b'u' => {
                            let hex = self
                                .bytes
                                .get(self.pos..self.pos + 4)
                                .ok_or_else(|| "truncated \\u escape".to_string())?;
                            self.pos += 4;
                            let s = std::str::from_utf8(hex)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            let code = u32::from_str_radix(s, 16)
                                .map_err(|_| "bad \\u escape".to_string())?;
                            out.push(
                                char::from_u32(code)
                                    .ok_or_else(|| "invalid \\u code point".to_string())?,
                            );
                        }
                        other => {
                            return Err(format!("unknown JSON escape '\\{}'", other as char))
                        }
                    }
                }
                b if b < 0x20 => {
                    return Err("control character in JSON string".to_string())
                }
                _ => {
                    // Copy the raw UTF-8 bytes (a multi-byte char arrives one
                    // byte at a time here, so defer validation until the end).
                    let start = self.pos - 1;
                    let mut end = start + 1;
                    while end < self.bytes.len()
                        && self.bytes[end] != b'"'
                        && self.bytes[end] != b'\\'
                        && self.bytes[end] >= 0x20
                    {
                        end += 1;
                    }
                    let s = std::str::from_utf8(&self.bytes[start..end])
                        .map_err(|_| "invalid UTF-8 in JSON string".to_string())?;
                    out.push_str(s);
                    self.pos = end;
                }
            }
        }
        Err("unterminated JSON string".to_string())
    }

    fn number(&mut self) -> Result<Json, String> {
        let start = self.pos;
        if self.bytes.get(self.pos) == Some(&b'-') {
            self.pos += 1;
        }
        while let Some(b) = self.bytes.get(self.pos) {
            if b.is_ascii_digit() || matches!(b, b'.' | b'e' | b'E' | b'+' | b'-') {
                self.pos += 1;
            } else {
                break;
            }
        }
        let s = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| "bad JSON number".to_string())?;
        s.parse::<f64>()
            .map(Json::Num)
            .map_err(|_| format!("bad JSON number '{}'", s))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_json_round_trips() {
        let h = Header {
            n_rows: 10,
            n_cols: 2,
            group_rows: 4,
            columns: vec![
                ColumnMeta {
                    name: "a\"b".to_string(),
                    numeric: true,
                    dtype: DType::F64,
                    levels: vec![],
                    has_codes: false,
                    offsets: vec![0, 64, 128],
                    code_offsets: vec![],
                },
                ColumnMeta {
                    name: "label".to_string(),
                    numeric: false,
                    dtype: DType::F64,
                    levels: vec!["x".to_string(), "y".to_string()],
                    has_codes: true,
                    offsets: vec![0, 32, 64],
                    code_offsets: vec![1, 33, 65],
                },
            ],
        };
        let text = header_json(&h);
        let back = parse_header(&text).unwrap();
        assert_eq!(h, back);
    }
}
