//! Versioned, out-of-core column store (issues #71, #127, #128).
//!
//! A store is a small binary container around the same column-major image the
//! resident [`crate::data::Table`] uses:
//!
//! ```text
//! magic "RCSRBIN\0" | u32 version | u32 header_len | JSON header | data blocks
//! ```
//!
//! Version 2 is self-describing per column: value blocks are `f64`
//! (features/targets), `i64` (the time column, nanosecond-safe) or absent for
//! a pure dictionary column (`u32`), and dictionary code blocks are `u32`
//! (4 bytes) instead of the version-1 `f64` codes (8 bytes). Version 1 remains
//! readable: it is the original all-`f64` layout, and a v1 header with an
//! explicit `"dtype":"i64"` (the #127 extension) is read as an i64 time block.
//! Both versions use the same group structure and relative offsets.
//!
//! Dictionary levels are the evaluator's view; codes are kept in a separate
//! [`DictColumn`](crate::data::DictColumn) and never enter arithmetic.
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
use std::path::PathBuf;

/// Eight-byte magic; the trailing NUL makes a truncated file easy to reject.
pub const MAGIC: &[u8; 8] = b"RCSRBIN\0";
/// Current format version. Writers emit v2; readers accept v1 and v2.
pub const VERSION: u32 = 2;
/// Original all-`f64` layout, still readable.
pub const LEGACY_VERSION: u32 = 1;

/// Storage type of one column.
///
/// * `F64`: numeric value block (8 B); an optional code block is `u32`.
/// * `I64`: the time column's value block (8 B), nanosecond-safe.
/// * `U32`: pure dictionary/text column; no value block, only a `u32` code
///   block. The evaluator sees levels, never codes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F64,
    I64,
    U32,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::F64 => "f64",
            DType::I64 => "i64",
            DType::U32 => "u32",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        match s {
            "f64" => Some(DType::F64),
            "i64" => Some(DType::I64),
            "u32" => Some(DType::U32),
            _ => None,
        }
    }
}
/// Default row group, sized for 512 KiB per `f64` column (the measured I/O
/// block). It is a streaming unit, not an L2 tile.
pub const DEFAULT_GROUP_ROWS: usize = 65_536;

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
    pub version: u32,
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

    /// Byte width of one value cell in `col`. A pure dictionary column has no
    /// value block in v2.
    fn value_width(&self, col: usize) -> u64 {
        match self.columns[col].dtype {
            DType::F64 | DType::I64 => 8,
            DType::U32 => {
                if self.version >= 2 {
                    0
                } else {
                    8
                }
            }
        }
    }

    /// Byte width of one dictionary code: v1 stored codes as `f64`, v2 as
    /// `u32`.
    fn code_width(&self) -> u64 {
        if self.version >= 2 {
            4
        } else {
            8
        }
    }

    pub fn group_len(&self, group: usize) -> usize {
        let lo = group.saturating_mul(self.group_rows);
        (self.n_rows.saturating_sub(lo)).min(self.group_rows)
    }

    /// Column index by exact name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    /// Whether column `col` is numeric (text columns only have dictionary codes).
    pub fn is_numeric(&self, col: usize) -> Option<bool> {
        self.columns.get(col).map(|c| c.numeric)
    }

    /// Dictionary levels for a text (or low-cardinality numeric) column.
    ///
    /// The codes returned by [`FileStore::read_codes`] index this slice. `None`
    /// means the column does not carry a dictionary.
    pub fn levels(&self, col: usize) -> Option<&[String]> {
        self.columns
            .get(col)
            .filter(|c| c.has_codes)
            .map(|c| c.levels.as_slice())
    }

    /// Metadata of one column, in header order.
    pub fn column(&self, col: usize) -> Option<ColumnInfo> {
        self.columns.get(col).map(|c| ColumnInfo {
            name: c.name.clone(),
            numeric: c.numeric,
            dtype: c.dtype,
            has_codes: c.has_codes,
            levels: c.levels.clone(),
        })
    }

    /// All column metadata, in header order.
    pub fn columns(&self) -> Vec<ColumnInfo> {
        (0..self.columns.len()).filter_map(|c| self.column(c)).collect()
    }

    /// Header-derived resident shape (issue #112): exactly what `to_table` and
    /// the search allocate, without opening or reading the data blocks.
    ///
    /// Sizes come from the *resident* representation, not the on-disk width:
    /// values only for `numeric` columns, an extra `i64` copy, always-`u32`
    /// codes, dictionary levels, row-group directories and the group buffer.
    pub fn memory_shape(&self) -> crate::memory::StoreShape {
        let columns = self
            .columns
            .iter()
            .map(|c| {
                let dictionary_bytes = if c.has_codes {
                    c.levels
                        .iter()
                        .fold(0u64, |a, level| a.saturating_add(level.len() as u64 + 24))
                        .saturating_add(24)
                } else {
                    0
                };
                crate::memory::ColumnShape {
                    name: c.name.clone(),
                    value_width: if c.numeric { 8 } else { 0 },
                    int_width: if c.dtype == DType::I64 { 8 } else { 0 },
                    code_width: if c.has_codes { 4 } else { 0 },
                    dictionary_bytes,
                }
            })
            .collect();
        crate::memory::StoreShape {
            rows: self.n_rows,
            group_rows: self.group_rows,
            // The header JSON is parsed and dropped, but its offsets survive as
            // `Vec<u64>` directories; the estimator charges those plus the
            // per-column Vec/String headers.
            header_len: 0,
            columns,
        }
    }

    /// Estimated peak resident bytes for loading this store and preparing a
    /// search sample (issue #112). `opts` carries the used-column projection
    /// and the backend copies.
    pub fn estimate_resident(
        &self,
        opts: &crate::memory::EstimateOptions,
    ) -> crate::memory::MemoryEstimate {
        self.memory_shape().estimate(opts)
    }
}

/// Public, read-only description of one stored column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColumnInfo {
    pub name: String,
    pub numeric: bool,
    pub dtype: DType,
    pub has_codes: bool,
    pub levels: Vec<String>,
}

/// One resolved column of a [`Projection`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionColumn {
    /// Header index in the store (the identity the file uses).
    pub index: usize,
    pub name: String,
    pub dtype: DType,
    /// Bytes per value cell. Both readable value dtypes are 8 bytes.
    pub width: u64,
}

/// A resolved *read* projection (issue #294, policy #260/#265).
///
/// The projection is the only thing the projected reader and the device
/// transport consult. It is resolved by exact column name against the store
/// header, so a typo is a hard error rather than a silently dropped column.
/// The canonical order is header order; the hash is FNV-1a 64 over
/// `index:name:dtype;` in that order, which makes two requests for the same
/// columns hash identically regardless of the order they were named in.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Projection {
    columns: Vec<ProjectionColumn>,
    hash: u64,
}

impl Projection {
    /// Resolve `names` against `store`'s header.
    ///
    /// Every name must exist exactly once and must carry a numeric value block
    /// (`f64` or the exact `i64` time column). A dictionary-only column has no
    /// value block, so it cannot feed a numeric feature kernel; that is refused
    /// here instead of surfacing as a NaN column later.
    pub fn resolve(store: &FileStore, names: &[&str]) -> Result<Self, String> {
        let header = store.header();
        let mut chosen: Vec<usize> = Vec::with_capacity(names.len());
        for name in names {
            let index = header.column_index(name).ok_or_else(|| {
                format!(
                    "{}: projection column '{}' is not in the store ({} columns); \
                     refusing to drop it silently",
                    store.path(),
                    name,
                    header.n_cols
                )
            })?;
            if !chosen.contains(&index) {
                chosen.push(index);
            }
        }
        if chosen.is_empty() {
            return Err(format!("{}: empty projection", store.path()));
        }
        Self::from_indices(store, &chosen)
    }

    /// Resolve names that must all carry a numeric value block. The device
    /// transport uses this form: dictionary/text columns cannot feed arithmetic.
    pub fn resolve_numeric(store: &FileStore, names: &[&str]) -> Result<Self, String> {
        let p = Self::resolve(store, names)?;
        for c in p.columns() {
            if c.width == 0 {
                return Err(format!(
                    "{}: projection column '{}' is a dictionary/text column with no numeric \
                     value block; it cannot feed a numeric kernel",
                    store.path(),
                    c.name
                ));
            }
        }
        Ok(p)
    }

    /// Resolve owned names; convenience for call sites that build a `Vec<String>`.
    pub fn resolve_owned(store: &FileStore, names: &[String]) -> Result<Self, String> {
        let refs: Vec<&str> = names.iter().map(|s| s.as_str()).collect();
        Self::resolve(store, &refs)
    }

    /// Resolve an explicit index vector (the form #260 resolves once per run).
    pub fn from_indices(store: &FileStore, indices: &[usize]) -> Result<Self, String> {
        let header = store.header();
        let mut columns: Vec<ProjectionColumn> = Vec::with_capacity(indices.len());
        for &index in indices {
            let info = header.column(index).ok_or_else(|| {
                format!(
                    "{}: projection index {} is out of range ({} columns)",
                    store.path(),
                    index,
                    header.n_cols
                )
            })?;
            if !columns.iter().any(|c| c.index == index) {
                columns.push(ProjectionColumn {
                    index,
                    name: info.name,
                    dtype: info.dtype,
                    width: if info.dtype == DType::U32 { 0 } else { 8 },
                });
            }
        }
        if columns.is_empty() {
            return Err(format!("{}: empty projection", store.path()));
        }
        columns.sort_by_key(|c| c.index);
        let mut text = String::new();
        for c in &columns {
            text.push_str(&format!("{}:{}:{};", c.index, c.name, c.dtype.name()));
        }
        Ok(Projection {
            columns,
            hash: fnv1a_64(text.as_bytes()),
        })
    }

    pub fn len(&self) -> usize {
        self.columns.len()
    }

    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }

    pub fn columns(&self) -> &[ProjectionColumn] {
        &self.columns
    }

    pub fn indices(&self) -> Vec<usize> {
        self.columns.iter().map(|c| c.index).collect()
    }

    pub fn names(&self) -> Vec<String> {
        self.columns.iter().map(|c| c.name.clone()).collect()
    }

    /// FNV-1a 64 over the canonical `index:name:dtype;` text.
    pub fn hash(&self) -> u64 {
        self.hash
    }

    pub fn hash_hex(&self) -> String {
        format!("{:016x}", self.hash)
    }

    /// Bytes per row of the projected layout (`f64`/`i64` values only).
    pub fn row_bytes(&self) -> u64 {
        self.columns.iter().map(|c| c.width).sum()
    }

    /// Bytes for one group of `len` rows.
    pub fn group_bytes(&self, len: usize) -> u64 {
        self.row_bytes().saturating_mul(len as u64)
    }

    /// Canonical text; used in reports/manifests.
    pub fn canonical_text(&self) -> String {
        self.columns
            .iter()
            .map(|c| format!("{}:{}:{}", c.index, c.name, c.dtype.name()))
            .collect::<Vec<_>>()
            .join(";")
    }
}

fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// The destination row ranges of one group in the table columns, one entry per
/// projection column. A column that has no such block holds `None`.
struct GroupDst<'a> {
    group: usize,
    f64s: Vec<Option<&'a mut [f64]>>,
    ints: Vec<Option<&'a mut [i64]>>,
    codes: Vec<Option<&'a mut [u32]>>,
}

/// Statistics from one projected group read. Incremented at the `pread` site.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProjectedReadStats {
    pub groups: usize,
    pub columns: usize,
    pub value_bytes: u64,
}

/// Reusable column-major buffer for [`FileStore::read_group_projected`].
#[derive(Debug, Default)]
pub struct ProjectedGroup {
    len: usize,
    columns: usize,
    values: Vec<f64>,
    int_scratch: Vec<i64>,
    byte_scratch: Vec<u8>,
}

impl ProjectedGroup {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn n_columns(&self) -> usize {
        self.columns
    }

    /// One column of the group, in projection order.
    pub fn col(&self, c: usize) -> &[f64] {
        let lo = c * self.len;
        &self.values[lo..lo + self.len]
    }

    pub fn col_mut(&mut self, c: usize) -> &mut [f64] {
        let lo = c * self.len;
        let hi = lo + self.len;
        &mut self.values[lo..hi]
    }

    fn prepare(&mut self, len: usize, columns: usize) {
        self.len = len;
        self.columns = columns;
        self.values.resize(len * columns, f64::NAN);
        self.int_scratch.resize(len, 0);
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
        if version != LEGACY_VERSION && version != VERSION {
            return Err(format!(
                "{}: column store version {} is not supported by this build (reads versions {} and {}); \
                 re-run `rcsr import` to rewrite the file",
                path, version, LEGACY_VERSION, VERSION
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
        let header = parse_header(text, version).map_err(|e| format!("{}: {}", path, e))?;
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

    /// A second handle to the same store, for the projected reader thread.
    pub fn try_clone(&self) -> Result<Self, String> {
        let file = self
            .file
            .try_clone()
            .map_err(|e| format!("{}: {}", self.path, e))?;
        Ok(FileStore {
            file,
            header: self.header.clone(),
            data_start: self.data_start,
            path: self.path.clone(),
        })
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

    /// Header-derived resident shape (issue #112). Sizes are the resident
    /// representation, not the compact on-disk widths.
    pub fn memory_shape(&self) -> crate::memory::StoreShape {
        self.header.memory_shape()
    }

    /// Estimated peak resident bytes for this store plus search preparation.
    pub fn estimate_resident(
        &self,
        opts: &crate::memory::EstimateOptions,
    ) -> crate::memory::MemoryEstimate {
        self.header.estimate_resident(opts)
    }

    /// Largest sample that fits the same model. Explicit `--sample-rows` uses
    /// this only to warn/refuse; the automatic path never samples.
    pub fn sample_row_cap(
        &self,
        budget: u64,
        opts: &crate::memory::EstimateOptions,
    ) -> usize {
        self.memory_shape().sample_row_cap(budget, opts)
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
                    let codes_end = self.data_start
                        + col.code_offsets[g]
                        + len * self.header.code_width();
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
                // A pure dictionary column has no value block; the buffer keeps
                // NaNs and callers read the codes instead.
                DType::U32 => {}
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
        let mut scratch = Vec::new();
        let off = self.data_start + meta.code_offsets[group];
        if self.header.version >= 2 {
            let mut codes = vec![0u32; len];
            read_u32_block(&self.file, &self.path, off, &mut codes, &mut scratch)?;
            return Ok(codes);
        }
        let mut vals = vec![0.0f64; len];
        read_f64_block(&self.file, &self.path, off, &mut vals, &mut scratch)?;
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

    /// Read one group of one numeric column into a fresh `f64` vector.
    ///
    /// This is the columnar I/O seam used by `rcsr enrich`: it decodes only the
    /// column a caller asks for, instead of every column in the group. An
    /// `i64` time column is converted exactly (its sentinel becomes `NaN`);
    /// a dictionary-only column has no value block and returns `NaN`s.
    pub fn read_f64_group(&self, group: usize, col: usize) -> Result<Vec<f64>, String> {
        let meta = self
            .header
            .columns
            .get(col)
            .ok_or_else(|| format!("{}: column {} is out of range", self.path, col))?;
        if group >= self.header.n_groups() {
            return Err(format!(
                "{}: group {} is out of range ({} groups)",
                self.path,
                group,
                self.header.n_groups()
            ));
        }
        let len = self.header.group_len(group);
        let mut out = vec![f64::NAN; len];
        let mut scratch = Vec::new();
        match meta.dtype {
            DType::F64 => read_f64_block(
                &self.file,
                &self.path,
                self.data_start + meta.offsets[group],
                &mut out,
                &mut scratch,
            )?,
            DType::I64 => {
                let mut ints = vec![0i64; len];
                read_i64_block(
                    &self.file,
                    &self.path,
                    self.data_start + meta.offsets[group],
                    &mut ints,
                    &mut scratch,
                )?;
                for (dst, v) in out.iter_mut().zip(ints) {
                    *dst = if v == i64::MIN { f64::NAN } else { v as f64 };
                }
            }
            DType::U32 => {}
        }
        Ok(out)
    }

    /// Read one group of only the projected columns into reusable storage.
    ///
    /// The buffer is column-major in projection order with `len` values per
    /// column: `values[c * len + row]`. Only the selected blocks are `pread`;
    /// no other column of the store is touched.
    pub fn read_group_projected(
        &self,
        group: usize,
        projection: &Projection,
        out: &mut ProjectedGroup,
    ) -> Result<ProjectedReadStats, String> {
        let len = self.header.group_len(group);
        out.prepare(len, projection.len());
        let mut stats = self.read_group_projected_into(
            group,
            projection,
            &mut out.values,
            &mut out.int_scratch,
            &mut out.byte_scratch,
        )?;
        stats.groups = 1;
        Ok(stats)
    }

    /// [`FileStore::read_group_projected`] into a caller-owned `f64` slice.
    ///
    /// The slice must hold at least `projection.len() * group_len` values and
    /// be 8-byte aligned. This is the pinned-ring seam: the CUDA transport
    /// passes its pinned slot to this function and uploads the same bytes
    /// asynchronously, so there is never a second host copy.
    pub fn read_group_projected_into(
        &self,
        group: usize,
        projection: &Projection,
        out: &mut [f64],
        int_scratch: &mut Vec<i64>,
        byte_scratch: &mut Vec<u8>,
    ) -> Result<ProjectedReadStats, String> {
        if group >= self.header.n_groups() {
            return Err(format!(
                "{}: group {} is out of range ({} groups)",
                self.path,
                group,
                self.header.n_groups()
            ));
        }
        let len = self.header.group_len(group);
        let need = projection.len().saturating_mul(len);
        if out.len() < need {
            return Err(format!(
                "{}: projected group {} needs {} values but the buffer holds {}",
                self.path,
                group,
                need,
                out.len()
            ));
        }
        int_scratch.resize(len, 0);
        for (c, col) in projection.columns().iter().enumerate() {
            let lo = c * len;
            let hi = lo + len;
            let meta = &self.header.columns[col.index];
            let off = self.data_start + meta.offsets[group];
            match meta.dtype {
                DType::F64 => {
                    read_f64_block(&self.file, &self.path, off, &mut out[lo..hi], byte_scratch)?
                }
                DType::I64 => {
                    read_i64_block(&self.file, &self.path, off, int_scratch, byte_scratch)?;
                    for (dst, &v) in out[lo..hi].iter_mut().zip(int_scratch.iter()) {
                        *dst = if v == i64::MIN { f64::NAN } else { v as f64 };
                    }
                }
                DType::U32 => {
                    // A dictionary column has no value block: the resident
                    // Table reads its codes separately (`to_table_projected`).
                    for v in &mut out[lo..hi] {
                        *v = f64::NAN;
                    }
                }
            }
        }
        Ok(ProjectedReadStats {
            groups: 1,
            columns: projection.len(),
            value_bytes: (need as u64).saturating_mul(8),
        })
    }

    /// Raw-byte projected group read, for the pinned upload ring.
    ///
    /// The bytes are copied straight from the store into `out` with one
    /// `pread` per column; `f64` stays `f64` and the time column stays `i64`.
    /// The caller must provide `out.len() >= projection.group_bytes(len)`.
    pub fn read_group_projected_raw_into(
        &self,
        group: usize,
        projection: &Projection,
        out: &mut [u8],
    ) -> Result<ProjectedReadStats, String> {
        use std::os::unix::fs::FileExt;
        if group >= self.header.n_groups() {
            return Err(format!(
                "{}: group {} is out of range ({} groups)",
                self.path,
                group,
                self.header.n_groups()
            ));
        }
        let len = self.header.group_len(group);
        let need = projection.group_bytes(len) as usize;
        if out.len() < need {
            return Err(format!(
                "{}: projected group {} needs {} bytes but the buffer holds {}",
                self.path,
                group,
                need,
                out.len()
            ));
        }
        let mut at = 0usize;
        for col in projection.columns() {
            let bytes = len * col.width as usize;
            if bytes == 0 {
                // Dictionary/text block: no value bytes to upload.
                continue;
            }
            let meta = &self.header.columns[col.index];
            let off = self.data_start + meta.offsets[group];
            self.file
                .read_exact_at(&mut out[at..at + bytes], off)
                .map_err(|e| {
                    format!(
                        "{}: reading {} bytes of '{}' at offset {}: {}",
                        self.path, bytes, col.name, off, e
                    )
                })?;
            at += bytes;
        }
        Ok(ProjectedReadStats {
            groups: 1,
            columns: projection.len(),
            value_bytes: need as u64,
        })
    }

    /// Materialise only the projected columns into a resident [`Table`].
    ///
    /// The result has exactly `projection.len()` columns in projection order.
    /// It is the narrow counterpart of [`FileStore::to_table`] used by the
    /// 20-variable path; it never allocates a full-table image. Groups are read
    /// by [`store_readers`] concurrent readers (`RCSR_STORE_READERS`).
    pub fn to_table_projected(&self, projection: &Projection) -> Result<Table, String> {
        self.to_table_projected_with_readers(projection, store_readers())
    }

    /// [`FileStore::to_table_projected`] with an explicit reader count. One
    /// reader is the serial reference path; every count gives the same table.
    pub fn to_table_projected_with_readers(
        &self,
        projection: &Projection,
        readers: usize,
    ) -> Result<Table, String> {
        let n = self.header.n_rows;
        let n_groups = self.header.n_groups();
        let readers = readers.min(n_groups).max(1);
        // Every element is overwritten by the group reads below, so the zeroed
        // allocation is never filled a second time.
        let mut values: Vec<Vec<f64>> = projection
            .columns()
            .iter()
            .map(|c| if c.width == 0 { Vec::new() } else { vec![0.0f64; n] })
            .collect();
        let mut ints: Vec<Option<Vec<i64>>> = projection
            .columns()
            .iter()
            .map(|c| (c.dtype == DType::I64).then(|| vec![i64::MIN; n]))
            .collect();
        let mut codes: Vec<Option<Vec<u32>>> = projection
            .columns()
            .iter()
            .map(|c| (c.dtype == DType::U32).then(|| vec![0u32; n]))
            .collect();
        if n_groups > 0 {
            let mut work = self.group_destinations(projection, &mut values, &mut ints, &mut codes);
            if readers == 1 {
                for dst in &mut work {
                    self.read_group_into(projection, dst)?;
                }
            } else {
                self.read_groups_concurrently(projection, readers, work)?;
            }
        }
        let names = projection.names();
        let numeric: Vec<bool> = projection
            .columns()
            .iter()
            .map(|c| c.dtype != DType::U32)
            .collect();
        let columns = names
            .iter()
            .enumerate()
            .map(|(c, _)| Column {
                values: std::mem::take(&mut values[c]),
                ints: ints[c].take(),
                dict: codes[c].take().map(|codes| DictColumn {
                    codes,
                    levels: self
                        .header
                        .levels(projection.columns()[c].index)
                        .map(|l| l.to_vec())
                        .unwrap_or_default(),
                }),
            })
            .collect();
        Ok(Table::from_parts(names, numeric, columns, n, 0))
    }

    /// Split every projected column into its per-group row ranges and collect
    /// the ranges of one group into one work item, in group order.
    fn group_destinations<'a>(
        &self,
        projection: &Projection,
        values: &'a mut [Vec<f64>],
        ints: &'a mut [Option<Vec<i64>>],
        codes: &'a mut [Option<Vec<u32>>],
    ) -> Vec<GroupDst<'a>> {
        let rows = self.header.group_rows.max(1);
        let n_cols = projection.len();
        let mut work: Vec<GroupDst<'a>> = (0..self.header.n_groups())
            .map(|group| GroupDst {
                group,
                f64s: Vec::with_capacity(n_cols),
                ints: Vec::with_capacity(n_cols),
                codes: Vec::with_capacity(n_cols),
            })
            .collect();
        for v in values.iter_mut() {
            if v.is_empty() {
                work.iter_mut().for_each(|w| w.f64s.push(None));
            } else {
                for (w, chunk) in work.iter_mut().zip(v.chunks_mut(rows)) {
                    w.f64s.push(Some(chunk));
                }
            }
        }
        for v in ints.iter_mut() {
            match v {
                Some(v) => {
                    for (w, chunk) in work.iter_mut().zip(v.chunks_mut(rows)) {
                        w.ints.push(Some(chunk));
                    }
                }
                None => work.iter_mut().for_each(|w| w.ints.push(None)),
            }
        }
        for v in codes.iter_mut() {
            match v {
                Some(v) => {
                    for (w, chunk) in work.iter_mut().zip(v.chunks_mut(rows)) {
                        w.codes.push(Some(chunk));
                    }
                }
                None => work.iter_mut().for_each(|w| w.codes.push(None)),
            }
        }
        work
    }

    /// Read the projected blocks of one group straight into its destination
    /// slices. An `i64` column is read once into `ints` and converted into the
    /// `f64` view from there.
    fn read_group_into(
        &self,
        projection: &Projection,
        dst: &mut GroupDst<'_>,
    ) -> Result<(), String> {
        let mut scratch = Vec::new();
        for (c, col) in projection.columns().iter().enumerate() {
            let meta = &self.header.columns[col.index];
            let off = self.data_start + meta.offsets[dst.group];
            match meta.dtype {
                DType::F64 => read_f64_block(
                    &self.file,
                    &self.path,
                    off,
                    dst.f64s[c].as_deref_mut().ok_or("missing f64 destination")?,
                    &mut scratch,
                )?,
                DType::I64 => {
                    let exact = dst.ints[c].as_deref_mut().ok_or("missing i64 destination")?;
                    read_i64_block(&self.file, &self.path, off, exact, &mut scratch)?;
                    let vals = dst.f64s[c].as_deref_mut().ok_or("missing f64 destination")?;
                    for (v, &i) in vals.iter_mut().zip(exact.iter()) {
                        *v = if i == i64::MIN { f64::NAN } else { i as f64 };
                    }
                }
                DType::U32 => read_u32_block(
                    &self.file,
                    &self.path,
                    self.data_start + meta.code_offsets[dst.group],
                    dst.codes[c].as_deref_mut().ok_or("missing code destination")?,
                    &mut scratch,
                )?,
            }
        }
        Ok(())
    }

    /// Reader pool of [`FileStore::to_table_projected_with_readers`]: workers
    /// take the next group's destination slices, read through their own handle
    /// and write the blocks into them. The error of the lowest failing group
    /// is returned: groups are claimed in order and a claimed group is always
    /// finished, so every group below a failing one has been read.
    fn read_groups_concurrently(
        &self,
        projection: &Projection,
        readers: usize,
        work: Vec<GroupDst<'_>>,
    ) -> Result<(), String> {
        use std::sync::atomic::{AtomicBool, Ordering};
        use std::sync::Mutex;

        let handles = (0..readers)
            .map(|_| self.try_clone())
            .collect::<Result<Vec<FileStore>, String>>()?;
        let queue = Mutex::new(work.into_iter());
        let failed: Mutex<Option<(usize, String)>> = Mutex::new(None);
        let stop = AtomicBool::new(false);
        std::thread::scope(|s| {
            for fs in &handles {
                let queue = &queue;
                let failed = &failed;
                let stop = &stop;
                s.spawn(move || {
                    while !stop.load(Ordering::Relaxed) {
                        let next = queue.lock().map(|mut q| q.next()).unwrap_or(None);
                        let Some(mut dst) = next else { break };
                        if let Err(e) = fs.read_group_into(projection, &mut dst) {
                            if let Ok(mut slot) = failed.lock() {
                                if slot.as_ref().map_or(true, |(g, _)| dst.group < *g) {
                                    *slot = Some((dst.group, e));
                                }
                            }
                            stop.store(true, Ordering::Relaxed);
                        }
                    }
                });
            }
        });
        match failed.into_inner() {
            Ok(Some((_, e))) => Err(e),
            Ok(None) => Ok(()),
            Err(_) => Err(format!("{}: a reader thread panicked", self.path)),
        }
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

/// Default and cap of the projected reader pool.
const STORE_READERS_DEFAULT: usize = 4;
const STORE_READERS_MAX: usize = 64;

/// `RCSR_STORE_READERS` value: default 4, clamped to 1..=64; 1 is the serial
/// path. An unparsable value is the default.
pub fn parse_store_readers(value: Option<&str>) -> usize {
    match value.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) => n.clamp(1, STORE_READERS_MAX),
        None => STORE_READERS_DEFAULT,
    }
}

/// Concurrent group readers of a projected read (`RCSR_STORE_READERS`).
pub fn store_readers() -> usize {
    parse_store_readers(std::env::var("RCSR_STORE_READERS").ok().as_deref())
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
    let plans: Vec<ColumnPlan> = (0..n_cols)
        .map(|c| {
            let dict = table.dict(c);
            ColumnPlan {
                name: table.names[c].clone(),
                numeric: table.numeric[c],
                dtype: if Some(c) == time_col {
                    DType::I64
                } else if !table.numeric[c] && dict.is_some() {
                    DType::U32
                } else {
                    DType::F64
                },
                levels: dict.map(|d| d.levels.clone()).unwrap_or_default(),
                has_codes: dict.is_some(),
            }
        })
        .collect();
    write_store_grouped(path, &plans, n_rows, group_rows, |lo, hi, out| {
        for (c, plan) in plans.iter().enumerate() {
            let slot = &mut out[c];
            match plan.dtype {
                DType::F64 => {
                    let values = table.col(c);
                    slot.f64s
                        .extend((lo..hi).map(|r| values.get(r).copied().unwrap_or(f64::NAN)));
                }
                DType::I64 => {
                    let ints = table.ints(c).ok_or_else(|| {
                        format!("column '{}' has no i64 view", table.names[c])
                    })?;
                    slot.i64s
                        .extend((lo..hi).map(|r| ints.get(r).copied().unwrap_or(i64::MIN)));
                }
                DType::U32 => {}
            }
            if let Some(dict) = table.dict(c) {
                slot.codes.extend_from_slice(&dict.codes[lo..hi]);
            }
        }
        Ok(())
    })
}

/// Declared layout of one column of a store written group by group.
#[derive(Clone, Debug)]
pub struct ColumnPlan {
    pub name: String,
    pub numeric: bool,
    pub dtype: DType,
    /// Dictionary levels, known before the first group is produced.
    pub levels: Vec<String>,
    pub has_codes: bool,
}

/// One group of one column. Only the vectors the plan's dtype and `has_codes`
/// call for are filled; the writer clears them before every group.
#[derive(Default)]
pub struct GroupColumn {
    pub f64s: Vec<f64>,
    pub i64s: Vec<i64>,
    pub codes: Vec<u32>,
}

/// Write a version-1 store without holding the table: `produce(lo, hi, cols)`
/// fills one group of rows `lo..hi` at a time, so peak memory is one group of
/// every column. The bytes equal [`write_store_with_time`] for the same data.
pub fn write_store_grouped<F>(
    path: &str,
    plans: &[ColumnPlan],
    n_rows: usize,
    group_rows: usize,
    mut produce: F,
) -> Result<(), String>
where
    F: FnMut(usize, usize, &mut [GroupColumn]) -> Result<(), String>,
{
    let group_rows = group_rows.max(1);
    let n_cols = plans.len();
    if n_rows == 0 {
        return Err("cannot write a column store with no rows".to_string());
    }
    let n_groups = (n_rows + group_rows - 1) / group_rows;
    let mut columns: Vec<ColumnMeta> = plans
        .iter()
        .map(|p| ColumnMeta {
            name: p.name.clone(),
            numeric: p.numeric,
            dtype: p.dtype,
            levels: p.levels.clone(),
            has_codes: p.has_codes,
            offsets: Vec::with_capacity(n_groups),
            code_offsets: Vec::with_capacity(n_groups),
        })
        .collect();
    // Offsets are relative to the data section, so the header length does not
    // feed back into the offsets (no fixed-point iteration needed).
    let mut cursor = 0u64;
    for g in 0..n_groups {
        let len = group_len(n_rows, group_rows, g) as u64;
        for col in &mut columns {
            col.offsets.push(cursor);
            cursor = cursor.saturating_add(len * col_value_width(col.dtype));
            if col.has_codes {
                col.code_offsets.push(cursor);
                cursor = cursor.saturating_add(len * 4);
            }
        }
    }
    let header = Header {
        version: VERSION,
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
    let mut cols: Vec<GroupColumn> = (0..n_cols).map(|_| GroupColumn::default()).collect();
    let mut bytes: Vec<u8> = Vec::new();
    for g in 0..n_groups {
        let lo = g * group_rows;
        let hi = (lo + group_rows).min(n_rows);
        for c in cols.iter_mut() {
            c.f64s.clear();
            c.i64s.clear();
            c.codes.clear();
        }
        produce(lo, hi, &mut cols)?;
        for (c, plan) in plans.iter().enumerate() {
            bytes.clear();
            match plan.dtype {
                DType::F64 => {
                    check_group_len(plan, cols[c].f64s.len(), hi - lo)?;
                    bytes.reserve((hi - lo) * 8);
                    for v in &cols[c].f64s {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                DType::I64 => {
                    check_group_len(plan, cols[c].i64s.len(), hi - lo)?;
                    bytes.reserve((hi - lo) * 8);
                    for v in &cols[c].i64s {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                // Text/categorical: no value block, codes below.
                DType::U32 => {}
            }
            file.write_all(&bytes)
                .map_err(|e| format!("{}: {}", path, e))?;
            if !plan.has_codes {
                continue;
            }
            check_group_len(plan, cols[c].codes.len(), hi - lo)?;
            bytes.clear();
            bytes.reserve((hi - lo) * 4);
            for v in &cols[c].codes {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            file.write_all(&bytes)
                .map_err(|e| format!("{}: {}", path, e))?;
        }
    }
    file.flush().map_err(|e| format!("{}: {}", path, e))?;
    Ok(())
}

fn check_group_len(plan: &ColumnPlan, got: usize, want: usize) -> Result<(), String> {
    if got == want {
        Ok(())
    } else {
        Err(format!(
            "column '{}': producer returned {} rows for a group of {}",
            plan.name, got, want
        ))
    }
}

fn group_len(n_rows: usize, group_rows: usize, group: usize) -> usize {
    let lo = group.saturating_mul(group_rows);
    (n_rows.saturating_sub(lo)).min(group_rows)
}

/// One pre-computed column to append to an existing store.
///
/// `file` is a raw little-endian `f64` column in row order (one value per row,
/// no header), normally produced group-by-group by `rcsr enrich`. Keeping the
/// values in a temporary column file is what lets the writer hold one group in
/// memory instead of all new columns.
#[derive(Clone, Debug)]
pub struct AppendColumn {
    pub name: String,
    pub file: PathBuf,
    /// Replace an existing `f64` column of the same name in place. Without
    /// this, a duplicate name is an error and existing columns are preserved.
    pub replace: bool,
}

/// Write `source` to `path` with `extras` appended, copying the existing
/// value/code blocks verbatim.
///
/// This is the append-only enrichment writer. It does **not** decode and
/// re-encode existing rows: every untouched column keeps its exact bytes,
/// dtype, dictionary, levels and position. Only the header offsets change
/// because new columns are laid out after the existing ones in each group.
///
/// A `replace` extra keeps the existing column's position (and therefore its
/// place in the schema) but writes new `f64` values. Replacing a time,
/// dictionary, text or non-numeric column is refused: those carry meaning that
/// a feature column cannot reproduce.
pub fn write_store_appending(
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
) -> Result<(), String> {
    let mut file = File::create(path).map_err(|e| format!("{}: {}", path, e))?;
    let batch = append_batch_groups(source, extras.len());
    let mut out = std::io::BufWriter::with_capacity(APPEND_WRITE_BUFFER, &mut file);
    write_store_appending_to(&mut out, path, source, extras, batch, &mut AppendStats::default())?;
    out.flush().map_err(|e| format!("{}: {}", path, e))
}

/// Like [`write_store_appending`], but also return the SHA-256 of the bytes it
/// wrote.
///
/// The hash runs on its own thread while the blocks stream to disk, so the
/// caller does not have to read a multi-gigabyte output a second time just to
/// checksum it and the writer never waits for the digest.
///
/// The bytes go to `<path>.tmp-<pid>` and are renamed to `path` on success, so
/// a failed write leaves no file at `path` and no temporary file.
pub fn write_store_appending_hashed(
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
) -> Result<String, String> {
    write_store_appending_hashed_stats(path, source, extras).map(|(sha, _)| sha)
}

/// [`write_store_appending_hashed`] that also returns where the time went.
pub fn write_store_appending_hashed_stats(
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
) -> Result<(String, AppendStats), String> {
    let batch = append_batch_groups(source, extras.len());
    write_store_appending_hashed_with(path, source, extras, batch)
}

/// Where the append pass spends its time. The three timers partition the pass:
/// temp-column and source reads, file writes, and time the writer is blocked
/// on the hash thread (its queue is full, or the digest is still draining at
/// the end). The call counts exist in unit tests only.
#[derive(Debug, Default, Clone, Copy)]
pub struct AppendStats {
    pub read_s: f64,
    pub write_s: f64,
    pub hash_wait_s: f64,
    #[cfg(test)]
    pub extra_reads: u64,
    #[cfg(test)]
    pub read_calls: u64,
    #[cfg(test)]
    pub read_bytes: u64,
    #[cfg(test)]
    pub write_calls: u64,
    #[cfg(test)]
    pub write_bytes: u64,
}

/// Size of the buffer in front of the output file.
const APPEND_WRITE_BUFFER: usize = 4 << 20;
/// Memory the staged temp-column batch may use by default.
const APPEND_BUDGET_MIB_DEFAULT: usize = 1024;
/// Bounds of `RCSR_APPEND_BUDGET_MIB`.
const APPEND_BUDGET_MIB_RANGE: std::ops::RangeInclusive<usize> = 64..=65536;
/// Most groups one batch stages.
const APPEND_BATCH_MAX: usize = 16;
/// Blocks queued for the hash thread.
const APPEND_HASH_QUEUE: usize = 8;

/// Groups staged per temp-column read: `RCSR_APPEND_BATCH_GROUPS` when set
/// (1 reads one group at a time), else as many as fit the budget. Either way
/// the result is at most 16, at most the group count, and at most what
/// `RCSR_APPEND_BUDGET_MIB` (default 1024) holds, with 1 as the floor.
fn append_batch_groups(source: &FileStore, n_extras: usize) -> usize {
    let (batch, note) = parse_append_batch(
        std::env::var("RCSR_APPEND_BATCH_GROUPS").ok().as_deref(),
        std::env::var("RCSR_APPEND_BUDGET_MIB").ok().as_deref(),
        source.header.group_rows,
        source.header.n_groups(),
        n_extras,
    );
    if let Some(note) = note {
        eprintln!("{}", note);
    }
    batch
}

/// The batch size and, when the budget or the maximum lowered it, the stderr
/// note that says so.
fn parse_append_batch(
    value: Option<&str>,
    budget_mib: Option<&str>,
    group_rows: usize,
    n_groups: usize,
    n_extras: usize,
) -> (usize, Option<String>) {
    let n_groups = n_groups.max(1);
    let mib = budget_mib
        .and_then(|v| v.trim().parse::<usize>().ok())
        .map_or(APPEND_BUDGET_MIB_DEFAULT, |m| {
            m.clamp(*APPEND_BUDGET_MIB_RANGE.start(), *APPEND_BUDGET_MIB_RANGE.end())
        });
    let budget = mib << 20;
    let per_group = n_extras.max(1).saturating_mul(group_rows.max(1)).saturating_mul(8);
    let by_budget = budget / per_group;
    let cap = by_budget.clamp(1, APPEND_BATCH_MAX).min(n_groups);
    let floor_note = (by_budget == 0).then(|| {
        format!(
            "one group ({} MiB) exceeds the append budget ({} MiB); staging 1 group",
            per_group.div_ceil(1 << 20),
            mib
        )
    });
    match value.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(n) => {
            let want = n.clamp(1, n_groups);
            let got = want.min(cap);
            let note = if got >= want {
                None
            } else if by_budget == 0 {
                floor_note
            } else if by_budget < APPEND_BATCH_MAX {
                Some(format!(
                    "RCSR_APPEND_BATCH_GROUPS={} clamped to {} by the {} MiB append budget",
                    n, got, mib
                ))
            } else {
                Some(format!(
                    "RCSR_APPEND_BATCH_GROUPS={} clamped to {} by the maximum batch",
                    n, got
                ))
            };
            (got, note)
        }
        None => (cap, floor_note),
    }
}

pub(crate) fn write_store_appending_hashed_with(
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
    batch: usize,
) -> Result<(String, AppendStats), String> {
    write_store_appending_hashed_over(path, source, extras, batch, |file| file)
}

/// [`write_store_appending_hashed_with`] with `wrap` applied to the temporary
/// file; the bytes go through the wrapper.
fn write_store_appending_hashed_over<W: Write>(
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
    batch: usize,
    wrap: impl FnOnce(File) -> W,
) -> Result<(String, AppendStats), String> {
    let mut stats = AppendStats::default();
    let tmp = format!("{}.tmp-{}", path, std::process::id());
    let guard = TempGuard(tmp.clone());
    let mut file = wrap(File::create(&tmp).map_err(|e| format!("{}: {}", tmp, e))?);
    let (sha, tail) = with_hash_thread(&mut file, path, |out| {
        write_store_appending_to(out, path, source, extras, batch, &mut stats)
    })?;
    stats.write_s += tail.write_s;
    stats.hash_wait_s += tail.hash_wait_s;
    #[cfg(test)]
    {
        stats.write_calls += tail.calls;
        stats.write_bytes += tail.bytes;
    }
    drop(file);
    std::fs::rename(&tmp, path).map_err(|e| format!("{}: {}", path, e))?;
    std::mem::forget(guard);
    Ok((sha, stats))
}

/// Test hook: the hash thread of the append to this path panics at this block.
#[cfg(test)]
static APPEND_HASH_PANIC: std::sync::Mutex<Option<(String, usize)>> = std::sync::Mutex::new(None);

#[cfg(test)]
fn append_hash_hook(path: &str, block: usize) {
    let hit = matches!(
        &*APPEND_HASH_PANIC.lock().unwrap_or_else(|e| e.into_inner()),
        Some((p, n)) if p == path && *n == block
    );
    if hit {
        panic!("injected hash thread panic at block {}", block);
    }
}

/// Removes its file on drop; forgotten once the file has been renamed away.
struct TempGuard(String);

impl Drop for TempGuard {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

/// Write-side counters of [`HashingWriter`].
#[derive(Default)]
struct WriterStats {
    write_s: f64,
    hash_wait_s: f64,
    #[cfg(test)]
    calls: u64,
    #[cfg(test)]
    bytes: u64,
}

/// `Write` adapter: writes to the file and hands a copy of every block to the
/// hash thread. Emptied blocks come back through `free` for reuse.
struct HashingWriter<'a, W: Write> {
    inner: &'a mut W,
    tx: std::sync::mpsc::SyncSender<Vec<u8>>,
    free: std::sync::mpsc::Receiver<Vec<u8>>,
    stats: WriterStats,
}

impl<W: Write> Write for HashingWriter<'_, W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let t = std::time::Instant::now();
        let n = self.inner.write(buf)?;
        self.stats.write_s += t.elapsed().as_secs_f64();
        #[cfg(test)]
        {
            self.stats.calls += 1;
            self.stats.bytes += n as u64;
        }
        let mut block = self.free.try_recv().unwrap_or_default();
        block.clear();
        block.extend_from_slice(&buf[..n]);
        let t = std::time::Instant::now();
        self.tx
            .send(block)
            .map_err(|_| std::io::Error::other("hash thread ended"))?;
        self.stats.hash_wait_s += t.elapsed().as_secs_f64();
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        self.inner.flush()
    }
}

/// Runs `body` with a buffered writer over `file` whose bytes are also hashed
/// on a second thread. Returns the digest of everything written and the
/// writer's own counters (time still waiting for the digest included).
fn with_hash_thread<W: Write, R>(
    file: &mut W,
    path: &str,
    body: impl FnOnce(&mut std::io::BufWriter<HashingWriter<'_, W>>) -> Result<R, String>,
) -> Result<(String, WriterStats), String> {
    std::thread::scope(|scope| {
        let (tx, rx) = std::sync::mpsc::sync_channel::<Vec<u8>>(APPEND_HASH_QUEUE);
        let (free_tx, free) = std::sync::mpsc::channel::<Vec<u8>>();
        let digest = scope.spawn(move || {
            let mut hasher = crate::manifest::Sha256Hasher::new();
            for (_i, block) in rx.into_iter().enumerate() {
                #[cfg(test)]
                append_hash_hook(path, _i);
                hasher.update(&block);
                let _ = free_tx.send(block);
            }
            hasher.finalize()
        });
        let mut out = std::io::BufWriter::with_capacity(
            APPEND_WRITE_BUFFER,
            HashingWriter { inner: file, tx, free, stats: WriterStats::default() },
        );
        let run = body(&mut out).and_then(|_| out.flush().map_err(|e| format!("{}: {}", path, e)));
        let mut stats = match out.into_inner() {
            Ok(w) => w.stats,
            Err(_) => WriterStats::default(),
        };
        let t = std::time::Instant::now();
        let sha = digest.join().map_err(|_| "hash thread panicked".to_string());
        stats.hash_wait_s += t.elapsed().as_secs_f64();
        run?;
        Ok((sha?, stats))
    })
}

fn write_store_appending_to<W: Write + ?Sized>(
    out: &mut W,
    path: &str,
    source: &FileStore,
    extras: &[AppendColumn],
    batch: usize,
    stats: &mut AppendStats,
) -> Result<(), String> {
    if extras.is_empty() {
        return Err("append needs at least one column".to_string());
    }
    if source.path == path {
        return Err(format!(
            "{}: refusing to append in place; write to a different path and rename",
            path
        ));
    }
    if source.header.version != VERSION {
        return Err(format!(
            "{}: appending needs a version-{} store (this one is version {}); \
             re-run `rcsr import` to rewrite it",
            source.path, VERSION, source.header.version
        ));
    }
    let n_rows = source.header.n_rows;
    let group_rows = source.header.group_rows;
    let n_groups = source.header.n_groups();

    let mut seen = std::collections::BTreeSet::new();
    for e in extras {
        if e.name.is_empty() {
            return Err("append column has an empty name".to_string());
        }
        if !seen.insert(e.name.clone()) {
            return Err(format!("feature '{}' appears twice", e.name));
        }
        let len = std::fs::metadata(&e.file)
            .map_err(|err| format!("{}: {}", e.file.display(), err))?
            .len();
        let want = (n_rows as u64).saturating_mul(8);
        if len != want {
            return Err(format!(
                "{}: temporary column for '{}' has {} bytes, expected {} ({} rows x 8)",
                e.file.display(),
                e.name,
                len,
                want,
                n_rows
            ));
        }
    }

    let mut columns = source.header.columns.clone();
    // `extra_of[output column] = Some(index into extras)`.
    let mut extra_of: Vec<Option<usize>> = vec![None; columns.len()];
    for (ei, e) in extras.iter().enumerate() {
        if let Some(ci) = source.header.column_index(&e.name) {
            if !e.replace {
                return Err(format!(
                    "column '{}' already exists",
                    e.name
                ));
            }
            let c = &mut columns[ci];
            if c.dtype != DType::F64 || !c.numeric {
                return Err(format!(
                    "column '{}' is {} and cannot be replaced by a numeric feature",
                    e.name,
                    if c.has_codes { "a dictionary column" } else { c.dtype.name() }
                ));
            }
            // A numeric column may carry a low-cardinality dictionary; the
            // replacement feature has no levels, so drop the stale codes.
            c.has_codes = false;
            c.levels.clear();
            extra_of[ci] = Some(ei);
        } else {
            columns.push(ColumnMeta {
                name: e.name.clone(),
                numeric: true,
                dtype: DType::F64,
                levels: Vec::new(),
                has_codes: false,
                offsets: Vec::with_capacity(n_groups),
                code_offsets: Vec::with_capacity(n_groups),
            });
            extra_of.push(Some(ei));
        }
    }

    // Recompute offsets for the new schema. Existing widths are unchanged, so
    // the copy loop below can hand the source block to the file untouched.
    // The cloned source metadata already carries the old offsets; clear them
    // before the group-major layout pass appends the new ones.
    for col in &mut columns {
        col.offsets.clear();
        col.code_offsets.clear();
    }
    let mut cursor = 0u64;
    for g in 0..n_groups {
        let len = group_len(n_rows, group_rows, g) as u64;
        for col in &mut columns {
            col.offsets.push(cursor);
            cursor = cursor.saturating_add(len * col_value_width(col.dtype));
            if col.has_codes {
                col.code_offsets.push(cursor);
                cursor = cursor.saturating_add(len * 4);
            }
        }
    }
    let header = Header {
        version: VERSION,
        n_rows,
        n_cols: columns.len(),
        group_rows,
        columns,
    };
    let json = header_json(&header);
    if json.len() > u32::MAX as usize {
        return Err("column store header is too large".to_string());
    }

    let extras_files: Vec<File> = extras
        .iter()
        .map(|e| File::open(&e.file).map_err(|err| format!("{}: {}", e.file.display(), err)))
        .collect::<Result<_, _>>()?;

    let file = out;
    file.write_all(MAGIC)
        .and_then(|_| file.write_all(&VERSION.to_le_bytes()))
        .and_then(|_| file.write_all(&(json.len() as u32).to_le_bytes()))
        .and_then(|_| file.write_all(json.as_bytes()))
        .map_err(|e| format!("{}: {}", path, e))?;

    let batch = batch.clamp(1, n_groups.max(1));
    let mut stage: Vec<Vec<u8>> = vec![Vec::new(); extras.len()];
    let mut buf: Vec<u8> = Vec::new();
    for g0 in (0..n_groups).step_by(batch) {
        let g1 = (g0 + batch).min(n_groups);
        let batch_lo = g0 * group_rows;
        let batch_hi = (g1 * group_rows).min(n_rows);
        for (ei, block) in stage.iter_mut().enumerate() {
            block.resize((batch_hi - batch_lo) * 8, 0);
            let t = std::time::Instant::now();
            read_exact_at(&extras_files[ei], block, (batch_lo as u64) * 8).map_err(|e| {
                format!("{}: reading appended column '{}': {}", path, extras[ei].name, e)
            })?;
            stats.read_s += t.elapsed().as_secs_f64();
            #[cfg(test)]
            {
                stats.extra_reads += 1;
                stats.read_calls += 1;
                stats.read_bytes += block.len() as u64;
            }
        }
        for g in g0..g1 {
            let lo = g * group_rows;
            let len = group_len(n_rows, group_rows, g);
            for (ci, _) in header.columns.iter().enumerate() {
                if let Some(ei) = extra_of[ci] {
                    let from = (lo - batch_lo) * 8;
                    file.write_all(&stage[ei][from..from + len * 8])
                        .map_err(|e| format!("{}: {}", path, e))?;
                    continue;
                }
                let src = &source.header.columns[ci];
                let bytes = len as u64 * source.header.value_width(ci);
                if bytes > 0 {
                    buf.resize(bytes as usize, 0);
                    let t = std::time::Instant::now();
                    read_exact_at(&source.file, &mut buf, source.data_start + src.offsets[g])
                        .map_err(|e| format!("{}: copying column '{}': {}", path, src.name, e))?;
                    stats.read_s += t.elapsed().as_secs_f64();
                    #[cfg(test)]
                    {
                        stats.read_calls += 1;
                        stats.read_bytes += bytes;
                    }
                    file.write_all(&buf).map_err(|e| format!("{}: {}", path, e))?;
                }
                if src.has_codes {
                    let bytes = len * 4;
                    buf.resize(bytes, 0);
                    let t = std::time::Instant::now();
                    read_exact_at(&source.file, &mut buf, source.data_start + src.code_offsets[g])
                        .map_err(|e| {
                            format!("{}: copying codes for '{}': {}", path, src.name, e)
                        })?;
                    stats.read_s += t.elapsed().as_secs_f64();
                    #[cfg(test)]
                    {
                        stats.read_calls += 1;
                        stats.read_bytes += bytes as u64;
                    }
                    file.write_all(&buf).map_err(|e| format!("{}: {}", path, e))?;
                }
            }
        }
    }
    file.flush().map_err(|e| format!("{}: {}", path, e))?;
    Ok(())
}

/// v2 value-block width. A pure dictionary column stores no values.
fn col_value_width(dtype: DType) -> u64 {
    match dtype {
        DType::F64 | DType::I64 => 8,
        DType::U32 => 0,
    }
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
        h.version, h.n_rows, h.n_cols, h.group_rows, cols
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

/// Types with no invalid bit pattern and no padding; private, so only this module can add one.
#[cfg(target_endian = "little")]
trait PlainBytes: Copy {}
#[cfg(target_endian = "little")]
impl PlainBytes for f64 {}
#[cfg(target_endian = "little")]
impl PlainBytes for i64 {}
#[cfg(target_endian = "little")]
impl PlainBytes for u32 {}

/// The bytes of a value slice, for a positional read straight into it.
#[cfg(target_endian = "little")]
fn bytes_of_mut<T: PlainBytes>(out: &mut [T]) -> &mut [u8] {
    // SAFETY: `PlainBytes` is implemented only for f64, i64 and u32, which have no invalid bit
    // patterns and no padding; u8 has alignment 1; the view covers exactly the bytes of `out`
    // and borrows it.
    unsafe {
        std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, std::mem::size_of_val(out))
    }
}

fn block_error(path: &str, bytes: usize, offset: u64, e: std::io::Error) -> String {
    format!(
        "{}: reading {} bytes at offset {}: {}",
        path, bytes, offset, e
    )
}

#[cfg(target_endian = "little")]
fn read_f64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [f64],
    _scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    read_exact_at(file, bytes_of_mut(out), offset).map_err(|e| block_error(path, bytes, offset, e))
}

#[cfg(not(target_endian = "little"))]
fn read_f64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [f64],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    scratch.resize(bytes, 0);
    read_exact_at(file, &mut scratch[..], offset)
        .map_err(|e| block_error(path, bytes, offset, e))?;
    for (dst, src) in out.iter_mut().zip(scratch.chunks_exact(8)) {
        let mut b = [0u8; 8];
        b.copy_from_slice(src);
        *dst = f64::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(target_endian = "little")]
fn read_i64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [i64],
    _scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    read_exact_at(file, bytes_of_mut(out), offset).map_err(|e| block_error(path, bytes, offset, e))
}

#[cfg(not(target_endian = "little"))]
fn read_i64_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [i64],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 8;
    scratch.resize(bytes, 0);
    read_exact_at(file, &mut scratch[..], offset)
        .map_err(|e| block_error(path, bytes, offset, e))?;
    for (dst, src) in out.iter_mut().zip(scratch.chunks_exact(8)) {
        let mut b = [0u8; 8];
        b.copy_from_slice(src);
        *dst = i64::from_le_bytes(b);
    }
    Ok(())
}

#[cfg(target_endian = "little")]
fn read_u32_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [u32],
    _scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 4;
    read_exact_at(file, bytes_of_mut(out), offset).map_err(|e| block_error(path, bytes, offset, e))
}

#[cfg(not(target_endian = "little"))]
fn read_u32_block(
    file: &File,
    path: &str,
    offset: u64,
    out: &mut [u32],
    scratch: &mut Vec<u8>,
) -> Result<(), String> {
    let bytes = out.len() * 4;
    scratch.resize(bytes, 0);
    read_exact_at(file, &mut scratch[..], offset)
        .map_err(|e| block_error(path, bytes, offset, e))?;
    for (dst, src) in out.iter_mut().zip(scratch.chunks_exact(4)) {
        let mut b = [0u8; 4];
        b.copy_from_slice(src);
        *dst = u32::from_le_bytes(b);
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

fn parse_header(text: &str, outer_version: u32) -> Result<Header, String> {
    let mut p = JsonParser {
        bytes: text.as_bytes(),
        pos: 0,
    };
    let root = p.value()?;
    p.ws();
    if p.pos != p.bytes.len() {
        return Err("trailing bytes in column store header".to_string());
    }
    let header_version = root
        .get("version")
        .and_then(Json::as_usize)
        .ok_or_else(|| "header is missing version".to_string())? as u32;
    if header_version != outer_version {
        return Err(format!(
            "header version {} disagrees with the file version {}",
            header_version, outer_version
        ));
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
        let has_codes = c
            .get("has_codes")
            .and_then(Json::as_bool)
            .ok_or_else(|| format!("column '{}' is missing has_codes", name))?;
        if dtype == DType::U32 {
            if outer_version < 2 {
                return Err(format!(
                    "column '{}' has dtype u32, which needs store version 2 or later",
                    name
                ));
            }
            if numeric || !has_codes {
                return Err(format!(
                    "column '{}' has dtype u32 but is not a dictionary-only column",
                    name
                ));
            }
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
        version: outer_version,
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
            version: VERSION,
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
                    dtype: DType::U32,
                    levels: vec!["x".to_string(), "y".to_string()],
                    has_codes: true,
                    offsets: vec![0, 32, 64],
                    code_offsets: vec![1, 33, 65],
                },
            ],
        };
        let text = header_json(&h);
        let back = parse_header(&text, VERSION).unwrap();
        assert_eq!(h, back);
    }

    fn write_raw(tag: &str, bytes: &[u8]) -> (std::path::PathBuf, File) {
        let path = std::env::temp_dir().join(format!(
            "rcsr-1076-{}-{}.bin",
            tag,
            std::process::id()
        ));
        std::fs::write(&path, bytes).unwrap();
        let file = File::open(&path).unwrap();
        (path, file)
    }

    #[test]
    fn direct_block_reads_match_the_decode_loop() {
        let f64s = [
            f64::NAN,
            f64::from_bits(0x7ff8_0000_dead_beef),
            f64::from_bits(0xfff4_0000_0000_0001),
            -0.0,
            0.0,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE / 8.0,
            1.5e300,
            -3.25,
        ];
        let i64s = [i64::MIN, i64::MAX, 0, -1, 1_700_000_000_000_000_000, 42];
        let u32s = [0u32, 1, u32::MAX, 7, 0x8000_0000];
        let mut raw: Vec<u8> = vec![0xAA; 3];
        for v in f64s {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let i_off = raw.len() as u64;
        for v in i64s {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let u_off = raw.len() as u64;
        for v in u32s {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        let (path, file) = write_raw("blocks", &raw);
        let p = path.to_string_lossy().into_owned();
        let mut scratch = Vec::new();

        let mut got = vec![0.0f64; f64s.len()];
        read_f64_block(&file, &p, 3, &mut got, &mut scratch).unwrap();
        let want: Vec<u64> = raw[3..]
            .chunks_exact(8)
            .take(f64s.len())
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()).to_bits())
            .collect();
        assert_eq!(got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), want);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            f64s.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );

        let mut got = vec![0i64; i64s.len()];
        read_i64_block(&file, &p, i_off, &mut got, &mut scratch).unwrap();
        let want: Vec<i64> = raw[i_off as usize..]
            .chunks_exact(8)
            .take(i64s.len())
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, want);
        assert_eq!(got, i64s);

        let mut got = vec![0u32; u32s.len()];
        read_u32_block(&file, &p, u_off, &mut got, &mut scratch).unwrap();
        let want: Vec<u32> = raw[u_off as usize..]
            .chunks_exact(4)
            .take(u32s.len())
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(got, want);
        assert_eq!(got, u32s);

        let mut short = vec![0.0f64; f64s.len() + 40];
        let e = read_f64_block(&file, &p, 3, &mut short, &mut scratch).unwrap_err();
        assert!(e.contains("reading 400 bytes at offset 3"), "{}", e);
        let _ = std::fs::remove_file(path);
    }

    #[test]
    fn a_failed_group_read_reports_the_lowest_group() {
        let rows = 5 * 16usize;
        let mut text = String::from("ts,sym,a,b\n");
        for r in 0..rows {
            text.push_str(&format!(
                "{},{},{},{}\n",
                1_700_000_000i64 + r as i64,
                ["P", "Q"][r % 2],
                r as f64 * 0.5,
                r as f64 - 3.0
            ));
        }
        let table = crate::data::parse_table_with_time(&text, false, Some("ts")).unwrap();
        let path = std::env::temp_dir()
            .join(format!("rcsr-1076-trunc-{}.rcsb", std::process::id()))
            .to_string_lossy()
            .into_owned();
        write_store_with_time(&path, &table, 16, Some(0)).unwrap();
        let fs = FileStore::open(&path).unwrap();
        assert_eq!(fs.n_groups(), 5);
        let proj = Projection::resolve(&fs, &["ts", "sym", "a", "b"]).unwrap();
        let cut = fs.data_start + fs.header.columns[2].offsets[3] + 8;
        std::fs::OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(cut)
            .unwrap();
        let expect = format!(
            "reading 128 bytes at offset {}",
            fs.data_start + fs.header.columns[2].offsets[3]
        );
        for readers in [1usize, 2, 4, 64] {
            for _ in 0..8 {
                let e = fs
                    .to_table_projected_with_readers(&proj, readers)
                    .err()
                    .expect("truncated store");
                assert!(e.contains(&expect), "{} readers: {}", readers, e);
            }
        }
        let _ = std::fs::remove_file(path);
    }

    const APPEND_GOLDEN_SHA256: &str = "68d592c2c632cf7e9d8ff04485425fd7e648ab0a294cb7dbeaab5ab7ee355ff6";
    const APPEND_GROUPS: usize = 7;
    const APPEND_EXTRAS: usize = 40;

    fn splitmix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e37_79b9_7f4a_7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        x ^ (x >> 31)
    }

    struct AppendFixture {
        groups: usize,
        rows: usize,
        dir: std::path::PathBuf,
        source: FileStore,
        extras: Vec<AppendColumn>,
    }

    impl Drop for AppendFixture {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// 24 numeric columns, a time column and a dictionary column over seven
    /// groups of 16 rows (the last one short), plus 40 temp columns of
    /// arbitrary bit patterns, one of which replaces `a3`.
    fn append_fixture(
        tag: &str,
        groups: usize,
        group_rows: usize,
        a_cols: usize,
        n_extras: usize,
    ) -> AppendFixture {
        let rows = groups * group_rows - group_rows / 3;
        let mut text = String::from("ts,sym");
        for c in 0..a_cols {
            text.push_str(&format!(",a{}", c));
        }
        text.push('\n');
        for r in 0..rows {
            text.push_str(&format!("{},{}", 1_700_000_000i64 + r as i64, ["P", "Q", "R"][r % 3]));
            for c in 0..a_cols {
                text.push_str(&format!(",{}", r as f64 * 0.25 - c as f64));
            }
            text.push('\n');
        }
        let table = crate::data::parse_table_with_time(&text, false, Some("ts")).unwrap();
        let dir = std::env::temp_dir().join(format!("rcsr-1037-{}-{}", tag, std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let src = dir.join("src.rcsb").to_string_lossy().into_owned();
        write_store_with_time(&src, &table, group_rows, Some(0)).unwrap();
        let source = FileStore::open(&src).unwrap();
        assert_eq!(source.n_groups(), groups);
        let mut extras = Vec::new();
        for e in 0..n_extras {
            let file = dir.join(format!("extra{}.bin", e));
            let mut bytes = Vec::with_capacity(rows * 8);
            for r in 0..rows {
                bytes.extend_from_slice(&splitmix((e * 1_000_003 + r) as u64).to_le_bytes());
            }
            std::fs::write(&file, bytes).unwrap();
            extras.push(AppendColumn {
                name: if e == 3 { "a3".to_string() } else { format!("f{}", e) },
                file,
                replace: e == 3,
            });
        }
        AppendFixture { groups, rows, dir, source, extras }
    }

    fn run_append(fx: &AppendFixture, name: &str, batch: Option<usize>) -> (Vec<u8>, String, AppendStats) {
        let batch = batch.unwrap_or_else(|| append_batch_groups(&fx.source, fx.extras.len()));
        let out = fx.dir.join(name).to_string_lossy().into_owned();
        let (sha, stats) =
            write_store_appending_hashed_with(&out, &fx.source, &fx.extras, batch).unwrap();
        (std::fs::read(&out).unwrap(), sha, stats)
    }

    #[test]
    fn append_output_matches_the_golden_digest() {
        let fx = append_fixture("golden", APPEND_GROUPS, 16, 24, APPEND_EXTRAS);
        let (bytes, sha, _) = run_append(&fx, "out.rcsb", Some(1));
        assert_eq!(sha, crate::manifest::sha256_hex(&bytes));
        assert_eq!(sha, APPEND_GOLDEN_SHA256);
    }

    #[test]
    fn append_batched_reads_are_byte_identical() {
        let fx = append_fixture("batches", APPEND_GROUPS, 16, 24, APPEND_EXTRAS);
        let (b1, s1, _) = run_append(&fx, "b1.rcsb", Some(1));
        for (i, batch) in [Some(2), Some(APPEND_GROUPS), Some(100), None].into_iter().enumerate() {
            let (b, s, _) = run_append(&fx, &format!("b{}.rcsb", i + 2), batch);
            assert!(b == b1, "batch {:?} changed the bytes", batch);
            assert_eq!(s, s1);
        }
        assert_eq!(s1, crate::manifest::sha256_hex(&b1));
    }


    #[test]
    fn append_batching_cuts_the_temp_column_reads() {
        let fx = append_fixture("reads", APPEND_GROUPS, 16, 24, APPEND_EXTRAS);
        let (_, _, one) = run_append(&fx, "b1.rcsb", Some(1));
        let (_, _, two) = run_append(&fx, "b2.rcsb", Some(2));
        let (_, _, dflt) = run_append(&fx, "bd.rcsb", None);
        assert_eq!(one.extra_reads as usize, APPEND_EXTRAS * APPEND_GROUPS);
        assert_eq!(two.extra_reads as usize, APPEND_EXTRAS * APPEND_GROUPS.div_ceil(2));
        assert_eq!(dflt.extra_reads as usize, APPEND_EXTRAS);
    }

    #[test]
    fn append_batch_default_follows_the_budget() {
        assert_eq!(batch_of(None, None, 65_536, 292, 427), 4);
        assert_eq!(batch_of(None, None, 65_536, 292, 1), 16);
        assert_eq!(batch_of(None, None, 65_536, 3, 1), 3);
        assert_eq!(batch_of(None, None, 65_536, 292, 100_000), 1);
        assert_eq!(batch_of(Some("1"), None, 65_536, 292, 427), 1);
        assert_eq!(batch_of(Some("1000"), None, 65_536, 292, 427), 4);
        assert_eq!(batch_of(Some("0"), None, 65_536, 292, 427), 1);
        assert_eq!(batch_of(Some("x"), None, 65_536, 292, 427), 4);
    }

    /// Deterministic I/O proxy on a store shaped like the C2 append: 17 source
    /// columns, 427 temp columns, 292 groups. Rows per group are tiny; the
    /// call counts depend only on the shape.
    #[test]
    fn append_io_proxy_on_the_c2_shape() {
        let fx = append_fixture("proxy", 292, 4, 15, 427);
        let src_cols = fx.source.n_cols();
        assert_eq!(src_cols, 17);
        let mut counts = Vec::new();
        let mut first: Option<Vec<u8>> = None;
        for batch in [1usize, 4, 16] {
            let (bytes, _, st) = run_append(&fx, &format!("p{}.rcsb", batch), Some(batch));
            assert_eq!(st.extra_reads as usize, 427 * 292usize.div_ceil(batch));
            assert_eq!(st.write_bytes, bytes.len() as u64);
            assert_eq!(st.write_calls, (bytes.len() as u64).div_ceil(APPEND_WRITE_BUFFER as u64));
            let out_len = bytes.len();
            match &first {
                None => first = Some(bytes),
                Some(f) => assert!(*f == bytes),
            }
            counts.push((batch, st, out_len));
        }
        let old_write_calls = 292 * (src_cols + 427 + 1);
        for (batch, st, out_len) in &counts {
            eprintln!(
                "proxy c2-shape batch={} read_calls={} read_bytes={} write_calls={} (per-block writer: {}) out_bytes={} read_calls_per_MiB_out={:.2}",
                batch,
                st.read_calls,
                st.read_bytes,
                st.write_calls,
                old_write_calls,
                out_len,
                st.read_calls as f64 / (*out_len as f64 / 1048576.0)
            );
        }
        assert_eq!(fx.groups, 292);
        assert!(fx.rows > 0);
    }

    #[test]
    fn hashing_writer_digest_matches_in_memory() {
        let total = 20usize << 20;
        let data: Vec<u8> = (0..total).map(|i| (splitmix(i as u64 / 7) >> 5) as u8 ^ i as u8).collect();
        let dir = std::env::temp_dir().join(format!("rcsr-1037-hw-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.bin");
        let mut file = File::create(&path).unwrap();
        let (sha, _) = with_hash_thread(&mut file, "out.bin", |out| {
            let mut at = 0;
            let mut sizes = [1usize, 4095, 4096, 65_537, 5_000_000].iter().cycle();
            while at < data.len() {
                let n = (*sizes.next().unwrap()).min(data.len() - at);
                out.write_all(&data[at..at + n]).map_err(|e| e.to_string())?;
                at += n;
            }
            Ok(())
        })
        .unwrap();
        drop(file);
        assert_eq!(sha, crate::manifest::sha256_hex(&data));
        assert!(std::fs::read(&path).unwrap() == data);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_failed_append_leaves_no_output_and_joins_the_hash_thread() {
        let fx = append_fixture("fail", 3, 16, 4, 2);
        std::fs::write(&fx.extras[1].file, [0u8; 8]).unwrap();
        let out = fx.dir.join("never.rcsb").to_string_lossy().into_owned();
        let err = write_store_appending_hashed_stats(&out, &fx.source, &fx.extras).unwrap_err();
        assert!(err.contains("temporary column"), "{}", err);
        assert!(!std::path::Path::new(&out).exists());
        assert!(!std::path::Path::new(&format!("{}.tmp-{}", out, std::process::id())).exists());
    }

    fn batch_of(value: Option<&str>, mib: Option<&str>, rows: usize, groups: usize, cols: usize) -> usize {
        parse_append_batch(value, mib, rows, groups, cols).0
    }

    #[test]
    fn append_batch_override_is_clamped_to_the_budget() {
        for v in [Some("16"), Some("292"), None] {
            assert_eq!(batch_of(v, None, 65_536, 292, 427), 4, "{:?}", v);
        }
        let (n, note) = parse_append_batch(Some("16"), None, 65_536, 292, 427);
        assert_eq!(n, 4);
        assert_eq!(
            note.as_deref(),
            Some("RCSR_APPEND_BATCH_GROUPS=16 clamped to 4 by the 1024 MiB append budget")
        );
        assert_eq!(parse_append_batch(None, None, 65_536, 292, 427).1, None);
        assert_eq!(batch_of(Some("1"), None, 65_536, 292, 427), 1);
        assert_eq!(batch_of(Some("0"), None, 65_536, 292, 427), 1);
        assert_eq!(batch_of(Some("x"), None, 65_536, 292, 427), 4);
        assert_eq!(batch_of(Some("500"), None, 16, 7, 24), 7);
        assert_eq!(batch_of(Some("500"), None, 16, 700, 24), APPEND_BATCH_MAX);
        assert_eq!(batch_of(None, None, 0, 5, 0), 5);
        assert_eq!(batch_of(None, None, 16, 0, 24), 1);
    }

    #[test]
    fn append_batch_floor_is_one_group() {
        let (n, note) = parse_append_batch(None, None, 65_536, 292, 3_000);
        assert_eq!(n, 1);
        assert_eq!(
            note.as_deref(),
            Some("one group (1500 MiB) exceeds the append budget (1024 MiB); staging 1 group")
        );
        let (n, note) = parse_append_batch(Some("16"), None, 65_536, 292, 3_000);
        assert_eq!(n, 1);
        assert!(note.unwrap().contains("exceeds the append budget"));
    }

    #[test]
    fn append_budget_env_is_parsed_and_clamped() {
        assert_eq!(batch_of(None, Some("64"), 65_536, 292, 427), 1);
        assert_eq!(batch_of(None, Some("1"), 65_536, 292, 427), 1);
        assert_eq!(batch_of(Some("16"), Some("65536"), 65_536, 292, 427), 16);
        assert_eq!(batch_of(Some("16"), Some("999999999"), 65_536, 292, 427), 16);
        for bad in ["", "garbage", "-5", "1.5"] {
            assert_eq!(batch_of(Some("16"), Some(bad), 65_536, 292, 427), 4, "{:?}", bad);
        }
        let (n, note) = parse_append_batch(Some("16"), Some("64"), 65_536, 292, 427);
        assert_eq!(n, 1);
        assert!(note.unwrap().contains("exceeds the append budget (64 MiB)"));
    }

    /// Passes `limit` bytes through, then fails every write.
    struct FailAfter<W: Write> {
        inner: W,
        limit: usize,
    }

    impl<W: Write> Write for FailAfter<W> {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            if self.limit == 0 {
                return Err(std::io::Error::other("injected write failure"));
            }
            let n = buf.len().min(self.limit);
            let n = self.inner.write(&buf[..n])?;
            self.limit -= n;
            Ok(n)
        }
        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    fn assert_no_output(out: &str) {
        assert!(!std::path::Path::new(out).exists(), "{} exists", out);
        let tmp = format!("{}.tmp-{}", out, std::process::id());
        assert!(!std::path::Path::new(&tmp).exists(), "{} remains", tmp);
    }

    #[test]
    fn append_mid_stream_write_error_leaves_no_output() {
        for batch in [1, 3] {
            let fx = append_fixture(&format!("wfail{}", batch), 4, 16, 4, 2);
            let out = fx.dir.join("never.rcsb").to_string_lossy().into_owned();
            let err = write_store_appending_hashed_over(&out, &fx.source, &fx.extras, batch, |file| {
                FailAfter { inner: file, limit: 700 }
            })
            .map(|_| ())
            .unwrap_err();
            assert_eq!(err, format!("{}: injected write failure", out));
            assert_no_output(&out);
            let _ = std::fs::remove_dir_all(&fx.dir);
        }
    }

    #[test]
    fn append_mid_stream_read_error_leaves_no_output() {
        for batch in [1, 3] {
            let fx = append_fixture(&format!("rfail{}", batch), 4, 16, 4, 2);
            let keep = fx.source.data_start + fx.source.header.columns[0].offsets[3];
            std::fs::OpenOptions::new()
                .write(true)
                .open(&fx.source.path)
                .unwrap()
                .set_len(keep)
                .unwrap();
            let out = fx.dir.join("never.rcsb").to_string_lossy().into_owned();
            let err = write_store_appending_hashed_with(&out, &fx.source, &fx.extras, batch)
                .map(|_| ())
                .unwrap_err();
            assert!(
                err.starts_with(&format!("{}: copying column 'ts': ", out)),
                "{}",
                err
            );
            assert_no_output(&out);
            let _ = std::fs::remove_dir_all(&fx.dir);
        }
    }

    #[test]
    fn append_hash_thread_panic_leaves_no_output() {
        for batch in [1, 3] {
            let fx = append_fixture(&format!("hpanic{}", batch), 4, 16, 4, 2);
            let out = fx.dir.join("never.rcsb").to_string_lossy().into_owned();
            *APPEND_HASH_PANIC.lock().unwrap() = Some((out.clone(), 0));
            let res = write_store_appending_hashed_with(&out, &fx.source, &fx.extras, batch);
            *APPEND_HASH_PANIC.lock().unwrap() = None;
            assert_eq!(res.map(|_| ()).unwrap_err(), "hash thread panicked");
            assert_no_output(&out);
            let _ = std::fs::remove_dir_all(&fx.dir);
        }
    }
}
