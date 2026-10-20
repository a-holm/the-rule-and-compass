//! Data loading, splitting and normalisation.
//!
//! Rows are split into train/test in one of four ways:
//!
//! * `random` — shuffled i.i.d. holdout (the default; only correct when rows are
//!   exchangeable);
//! * `temporal` — chronological holdout: train on the past, test on the future,
//!   optionally with an **embargo** ([`SplitSpec::gap`] rows) between the two so
//!   that autocorrelation cannot leak the future into the training set;
//! * `group` — whole groups (a stock, a subject, a sensor) are assigned to one
//!   side only, so the test error measures generalisation to *unseen* groups;
//! * `temporal_group` — a chronological holdout performed inside every group:
//!   train on the past of every stock and test on its future, which is the
//!   realistic setting for panel data (OHLC for many symbols, over one period).
//!
//! Normalisation statistics are always computed from the **training rows only**.
//! Using the whole table would leak the test set into the model's input scaling.
//!
//! Time columns may be numbers or dates (`2024-01-31`, `2024/01/31`,
//! `2024-01-31T09:30:00`); anything unparseable falls back to a lexicographic
//! ordering, so the split is still well defined. Group columns may be text
//! (ticker symbols) — they are never used as model inputs, only for splitting.

use crate::column::RowSelector;
use crate::expr::{self, Node};
use crate::rng::Mt;
use std::collections::{BTreeSet, HashMap, HashSet};

/// Tokens that stand for a missing cell. They are accepted in numeric columns;
/// anything else that does not parse as a number still makes the column
/// non-numeric (see [`parse_table`]).
pub fn is_missing_token(cell: &str) -> bool {
    let c = cell.trim();
    c.is_empty()
        || c.eq_ignore_ascii_case("na")
        || c.eq_ignore_ascii_case("nan")
        || c.eq_ignore_ascii_case("null")
}

#[derive(Clone, Default)]
pub struct Sample {
    /// x[row][variable]
    pub x: Vec<Vec<f64>>,
    pub y: Vec<f64>,
    /// Per-row weights. **Empty means every row weighs 1** (the behaviour
    /// before issue #1); otherwise `w[i]` is the weight of row `i`. Weights are
    /// finite, non-negative and relative: in the weighted mean
    /// `Σ wᵢ·lossᵢ / Σ wᵢ` only their ratios matter.
    pub w: Vec<f64>,
    /// Multi-output targets (issue #15). Empty for the historical single-output
    /// case, where `y` is the only target. When non-empty this holds **one
    /// column per target**, including the first one; `y` is a copy of
    /// `ys[0]` so every single-target diagnostic keeps working unchanged.
    pub ys: Vec<Vec<f64>>,
}

impl Sample {
    /// An unweighted sample (every row weighs 1).
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Self {
        Sample {
            x,
            y,
            w: Vec::new(),
            ys: Vec::new(),
        }
    }

    /// A sample with one weight per row.
    ///
    /// The caller is responsible for `w.len() == y.len()` and for the values
    /// being finite and non-negative; `prepare_full_weighted_with` validates a
    /// weight column before it gets this far.
    pub fn weighted(x: Vec<Vec<f64>>, y: Vec<f64>, w: Vec<f64>) -> Self {
        Sample {
            x,
            y,
            w,
            ys: Vec::new(),
        }
    }

    // ---- multi-output helpers (issue #15) -----------------------------
    /// True when more than one target column was requested.
    pub fn is_multi(&self) -> bool {
        self.ys.len() > 1
    }

    /// Number of targets: one for the historical single-output sample.
    pub fn target_count(&self) -> usize {
        if self.ys.is_empty() {
            1
        } else {
            self.ys.len()
        }
    }

    /// Target column `j` (0-based). Target 0 is `y` for a multi sample too.
    pub fn target(&self, j: usize) -> &[f64] {
        if self.ys.is_empty() {
            &self.y
        } else {
            &self.ys[j]
        }
    }

    pub fn len(&self) -> usize {
        self.y.len()
    }
    pub fn is_empty(&self) -> bool {
        self.y.is_empty()
    }
    /// True when an explicit weight vector was supplied.
    pub fn has_weights(&self) -> bool {
        !self.w.is_empty()
    }
    /// Weight of row `i`; `1.0` when the sample is unweighted.
    pub fn weight(&self, i: usize) -> f64 {
        self.w.get(i).copied().unwrap_or(1.0)
    }
    /// Sum of the row weights (`len` for an unweighted sample).
    pub fn weight_sum(&self) -> f64 {
        if self.w.is_empty() {
            self.len() as f64
        } else {
            self.w.iter().sum()
        }
    }
    /// Number of rows with a strictly positive weight; the effective `n` for
    /// the information criteria and the residual degrees of freedom.
    pub fn positive_weight_rows(&self) -> usize {
        if self.w.is_empty() {
            self.len()
        } else {
            self.w.iter().filter(|&&v| v > 0.0).count()
        }
    }

    /// #14: turn an explicit sample into an implicit one.
    ///
    /// Every row gains the (normalised) target as one more input, and the target
    /// itself becomes the constant `0`: the search then fits
    /// `f(x1..xn, y) = 0`, the level set that the data should lie on. `mean` and
    /// `std` are the training target's normalisation constants (see
    /// [`implicit_target_stats`]), applied to both sides so the appended input
    /// lives on the same scale as the other inputs and denormalises back to the
    /// original units.
    pub fn with_target_as_input(&self, mean: f64, std: f64) -> Sample {
        let mut out = Sample::default();
        out.x = self
            .x
            .iter()
            .zip(&self.y)
            .map(|(row, &y)| {
                let mut r = row.clone();
                r.push((y - mean) / std);
                r
            })
            .collect();
        out.y = vec![0.0; self.len()];
        out.w = self.w.clone();
        out
    }
}

/// #14: the training-row mean and standard deviation of the target, used to
/// normalise it when it becomes an implicit input. A degenerate (zero-variance)
/// target gets `std = 1`, exactly like [`prepare_full_weighted_with`] for an
/// input column.
pub fn implicit_target_stats(train: &Sample) -> (f64, f64) {
    if train.y.is_empty() {
        return (0.0, 1.0);
    }
    let n = train.y.len() as f64;
    let mean = train.y.iter().sum::<f64>() / n;
    let var = train.y.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    (mean, if std.abs() >= 1e-12 { std } else { 1.0 })
}

/// #14: rewrite prepared train/test/folds in place for `--implicit`.
///
/// The target column is appended as an extra input to every sample and every
/// target becomes `0`; `means`/`stds` grow by the target's normalisation
/// constants so the printed formula is in original units. Returns
/// `(mean, std)` for the report. Nothing else in the pipeline changes: the
/// search, the tree machinery and the archive see an ordinary sample with one
/// more input.
pub fn make_implicit(p: &mut Prepared) -> (f64, f64) {
    let (mean, std) = implicit_target_stats(&p.train);
    p.train = p.train.with_target_as_input(mean, std);
    p.test = p.test.with_target_as_input(mean, std);
    p.fold_samples = p
        .fold_samples
        .iter()
        .map(|f| f.with_target_as_input(mean, std))
        .collect();
    p.means.push(mean);
    p.stds.push(std);
    (mean, std)
}

/// Dictionary encoding for a text/categorical column: `u32` codes into one
/// level table. The per-row cost is 4 bytes, independent of the label length.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct DictColumn {
    pub codes: Vec<u32>,
    pub levels: Vec<String>,
}

impl DictColumn {
    pub fn text(&self, row: usize) -> &str {
        &self.levels[self.codes[row] as usize]
    }

    pub fn len(&self) -> usize {
        self.codes.len()
    }
}

/// One column of a [`Table`].
///
/// Numeric columns keep one contiguous `f64` block (`values`). Text/categorical
/// columns keep `u32` codes and one label table (`dict`) and do **not** keep a
/// per-row `String`. A low-cardinality numeric column keeps both: the values
/// are the search's view, the dictionary preserves the exact source text for
/// dates, group labels and categorical levels.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Column {
    pub values: Vec<f64>,
    /// Exact 64-bit integer cells, for columns whose source text is integral
    /// and whose name marks them as a time axis (`ts`, `timestamp`, ...).
    /// Nanosecond timestamps exceed binary64's 2^53 exact range, so ordering
    /// and round-tripping use this view, never `values`. `i64::MIN` marks a
    /// missing cell; the evaluator still sees `values` (and so never performs
    /// arithmetic on a time column by accident).
    pub ints: Option<Vec<i64>>,
    pub dict: Option<DictColumn>,
}

impl Column {
    pub fn text(&self, row: usize) -> Option<&str> {
        self.dict.as_ref().map(|d| d.text(row))
    }
}

/// A parsed table, stored column-major.
///
/// There is no `raw: Vec<Vec<String>>` copy: non-numeric cells are dictionary
/// codes, and numeric cells keep the source text only when the column has few
/// distinct values (where it is needed for dates/labels and costs 4 B/row).
#[derive(Clone, Debug, Default)]
pub struct Table {
    pub names: Vec<String>,
    /// `numeric[j]` is true when every non-empty cell in column `j` parses as a
    /// number.
    pub numeric: Vec<bool>,
    columns: Vec<Column>,
    n_rows: usize,
    /// Data rows dropped because their field count differed from the header.
    /// Non-zero only when the loader was explicitly asked to skip them
    /// (`--skip-malformed`); the default refuses the file instead.
    pub skipped_malformed: usize,
}

/// A numeric column keeps its exact source text in the dictionary up to this
/// many distinct values. Above it the column is high-cardinality and the text
/// is dropped: nothing in the pipeline needs it, and it would dominate memory.
pub const NUMERIC_DICT_MAX: usize = 4096;

impl Table {
    pub fn index_of(&self, name: &str) -> Result<usize, String> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("column '{}' not found in {:?}", name, self.names))
    }

    pub fn len(&self) -> usize {
        self.n_rows
    }

    pub fn is_empty(&self) -> bool {
        self.n_rows == 0
    }

    pub fn n_cols(&self) -> usize {
        self.columns.len()
    }

    /// The whole contiguous `f64` column (empty for a text column).
    pub fn col(&self, j: usize) -> &[f64] {
        &self.columns[j].values
    }

    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        &mut self.columns[j].values
    }

    pub fn value(&self, row: usize, col: usize) -> f64 {
        self.columns
            .get(col)
            .and_then(|c| c.values.get(row))
            .copied()
            .unwrap_or(f64::NAN)
    }

    /// The exact source text of a cell, when the column retained it.
    pub fn text(&self, row: usize, col: usize) -> Option<&str> {
        self.columns.get(col).and_then(|c| c.text(row))
    }

    /// Source text as an owned string, formatting a high-cardinality numeric
    /// column when it no longer carries a dictionary. Only reporting and dtype
    /// validation use this; hot paths read [`Table::col`].
    pub fn cell_text(&self, row: usize, col: usize) -> String {
        if let Some(s) = self.text(row, col) {
            return s.to_string();
        }
        let v = self.value(row, col);
        if v.is_nan() {
            String::new()
        } else {
            format!("{}", v)
        }
    }

    pub fn dict(&self, col: usize) -> Option<&DictColumn> {
        self.columns.get(col).and_then(|c| c.dict.as_ref())
    }

    /// Exact integer cells for a time column (see [`Column::ints`]).
    pub fn ints(&self, col: usize) -> Option<&[i64]> {
        self.columns.get(col).and_then(|c| c.ints.as_deref())
    }

    /// Copy one row out. Hot loops use [`Table::col`] instead.
    pub fn row(&self, row: usize) -> Vec<f64> {
        self.columns.iter().map(|c| c.values.get(row).copied().unwrap_or(f64::NAN)).collect()
    }

    pub(crate) fn from_parts(
        names: Vec<String>,
        numeric: Vec<bool>,
        columns: Vec<Column>,
        n_rows: usize,
        skipped_malformed: usize,
    ) -> Self {
        Table {
            names,
            numeric,
            columns,
            n_rows,
            skipped_malformed,
        }
    }

    pub fn push_column(&mut self, name: String, numeric: bool, column: Column) {
        self.names.push(name);
        self.numeric.push(numeric);
        self.columns.push(column);
    }

    /// A new table with only the rows where `keep[r]` is true, preserving
    /// order. Columns stay contiguous; nothing is transposed.
    pub fn select(&self, keep: &[bool]) -> Table {
        let idx: Vec<usize> = (0..self.n_rows).filter(|&r| keep[r]).collect();
        self.select_indices(&idx)
    }

    pub fn select_indices(&self, idx: &[usize]) -> Table {
        let columns = self
            .columns
            .iter()
            .map(|c| Column {
                values: if c.values.is_empty() {
                    Vec::new()
                } else {
                    idx.iter().map(|&r| c.values[r]).collect()
                },
                ints: c.ints.as_ref().map(|v| idx.iter().map(|&r| v[r]).collect()),
                dict: c.dict.as_ref().map(|d| DictColumn {
                    codes: idx.iter().map(|&r| d.codes[r]).collect(),
                    levels: d.levels.clone(),
                }),
            })
            .collect();
        Table {
            names: self.names.clone(),
            numeric: self.numeric.clone(),
            columns,
            n_rows: idx.len(),
            skipped_malformed: self.skipped_malformed,
        }
    }

    /// Number of distinct values in a column (used by auto-detection).
    pub fn distinct_count(&self, col: usize) -> usize {
        if let Some(d) = self.dict(col) {
            return d.levels.len();
        }
        if self.numeric.get(col).copied().unwrap_or(false) {
            let mut seen: Vec<u64> = self
                .col(col)
                .iter()
                .map(|v| v.to_bits())
                .collect();
            seen.sort_unstable();
            seen.dedup();
            return seen.len();
        }
        0
    }
}

/// The datatype of one search input (issue #5).
///
/// `Continuous` is the default and is standardised like every column before
/// issue #5. `Boolean` holds `0`/`1` and is **never** centred or scaled, so the
/// evaluator receives exactly the values from the file. `Categorical` is not a
/// value the search can use directly: [`apply_dtypes`] expands it into one-hot
/// `Boolean` columns named `column=value`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum DType {
    #[default]
    Continuous,
    Boolean,
    Categorical,
}

impl DType {
    pub fn name(self) -> &'static str {
        match self {
            DType::Continuous => "continuous",
            DType::Boolean => "boolean",
            DType::Categorical => "categorical",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "continuous" | "numeric" | "float" => Ok(DType::Continuous),
            "boolean" | "bool" | "binary" => Ok(DType::Boolean),
            "categorical" | "category" | "factor" => Ok(DType::Categorical),
            other => Err(format!(
                "unknown --dtype kind '{}' (expected continuous, boolean or categorical)",
                other
            )),
        }
    }
}

/// One row of the datatype report: an input variable as the search sees it.
#[derive(Clone, Debug)]
pub struct VariableDType {
    /// Name of the input (for a one-hot indicator, `column=value`).
    pub name: String,
    pub dtype: DType,
    /// Original column, when `name` is a one-hot indicator.
    pub source: Option<String>,
}

/// A table after [`apply_dtypes`]: the expanded table plus the input list the
/// rest of the pipeline must use.
#[derive(Clone)]
pub struct TypedTable {
    pub table: Table,
    pub vars: Vec<String>,
    /// Datatype per entry of `vars`; categorical never appears here.
    pub dtypes: Vec<DType>,
    /// Human-readable datatype of every input, for the text report and JSON.
    pub variables: Vec<VariableDType>,
}

/// Why a datatype request failed. `usage` separates a bad flag (exit 2) from
/// bad data (exit 3); both carry a complete message.
#[derive(Clone, Debug)]
pub struct DTypeError {
    pub usage: bool,
    pub message: String,
}

impl DTypeError {
    fn usage(message: impl Into<String>) -> Self {
        DTypeError {
            usage: true,
            message: message.into(),
        }
    }

    fn input(message: impl Into<String>) -> Self {
        DTypeError {
            usage: false,
            message: message.into(),
        }
    }
}

impl std::fmt::Display for DTypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for DTypeError {}

/// Level name used for a missing/empty cell in a categorical column.
pub const MISSING_LEVEL: &str = "<missing>";

/// Parse `--dtype <col>=<kind>[,...]`. Kinds are validated here; column names
/// are validated against the table by [`apply_dtypes`].
pub fn parse_dtype_spec(s: &str) -> Result<Vec<(String, DType)>, String> {
    let mut out: Vec<(String, DType)> = Vec::new();
    for item in s.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (name, kind) = item.split_once('=').ok_or_else(|| {
            format!(
                "--dtype entry '{}' must be <column>=<kind>, e.g. flag=boolean",
                item
            )
        })?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("--dtype entry '{}' has an empty column name", item));
        }
        let kind = DType::parse(kind)?;
        if out.iter().any(|(n, _)| n == name) {
            return Err(format!("--dtype names column '{}' more than once", name));
        }
        out.push((name.to_string(), kind));
    }
    if out.is_empty() {
        return Err("--dtype needs at least one <column>=<kind> entry".to_string());
    }
    Ok(out)
}

/// Apply the per-variable datatype plan to a table.
///
/// * `continuous` columns pass through unchanged;
/// * `boolean` columns are validated to contain only `0`/`1` (missing cells are
///   left for `--fill`) and pass through unchanged;
/// * `categorical` columns are expanded into one `0`/`1` indicator per distinct
///   value, appended to the table in sorted level order and named
///   `column=value`. Missing/empty cells become the level
///   [`MISSING_LEVEL`]. All levels are kept: the search selects terms, it does
///   not invert a design matrix, so dropping a reference level would only throw
///   information away.
///
/// `max_levels` bounds the number of distinct values in a categorical column;
/// exceeding it is a usage error naming the column, the count and the flag.
pub fn apply_dtypes(
    t: &Table,
    vars: &[String],
    spec: &[(String, DType)],
    max_levels: usize,
) -> Result<TypedTable, DTypeError> {
    let mut per_col: HashMap<usize, DType> = HashMap::new();
    for (name, kind) in spec {
        let j = t.names.iter().position(|n| n == name).ok_or_else(|| {
            DTypeError::usage(format!(
                "unknown --dtype column '{}' (columns: {:?})",
                name, t.names
            ))
        })?;
        if !vars.iter().any(|v| v == name) {
            return Err(DTypeError::usage(format!(
                "column '{}' is not an input variable; add it to --vars or drop it from --dtype",
                name
            )));
        }
        if per_col.insert(j, *kind).is_some() {
            return Err(DTypeError::usage(format!(
                "column '{}' has more than one --dtype entry",
                name
            )));
        }
    }

    let mut table = t.clone();
    let mut new_vars: Vec<String> = Vec::with_capacity(vars.len());
    let mut dtypes: Vec<DType> = Vec::with_capacity(vars.len());
    let mut variables: Vec<VariableDType> = Vec::new();

    for v in vars {
        let j = t.index_of(v).map_err(DTypeError::usage)?;
        match per_col.get(&j).copied().unwrap_or(DType::Continuous) {
            DType::Continuous => {
                new_vars.push(v.clone());
                dtypes.push(DType::Continuous);
                variables.push(VariableDType {
                    name: v.clone(),
                    dtype: DType::Continuous,
                    source: None,
                });
            }
            DType::Boolean => {
                validate_boolean(t, j)?;
                new_vars.push(v.clone());
                dtypes.push(DType::Boolean);
                variables.push(VariableDType {
                    name: v.clone(),
                    dtype: DType::Boolean,
                    source: None,
                });
            }
            DType::Categorical => {
                let levels = categorical_levels(t, j, max_levels)?;
                for level in levels {
                    let name = format!("{}={}", v, level);
                    if table.names.iter().any(|n| n == &name) {
                        return Err(DTypeError::usage(format!(
                            "categorical column '{}' would create '{}', which already exists; rename the column or the level",
                            v, name
                        )));
                    }
                    let mut values = Vec::with_capacity(t.len());
                    let mut codes = Vec::with_capacity(t.len());
                    for r in 0..t.len() {
                        let cell = t.cell_text(r, j);
                        let cell = cell.trim();
                        let hit = if is_missing_token(cell) {
                            level == MISSING_LEVEL
                        } else {
                            cell == level
                        };
                        values.push(if hit { 1.0 } else { 0.0 });
                        codes.push(if hit { 0 } else { 1 });
                    }
                    table.push_column(
                        name.clone(),
                        true,
                        Column {
                            values,
                            ints: None,
                            dict: Some(DictColumn {
                                codes,
                                levels: vec!["1".to_string(), "0".to_string()],
                            }),
                        },
                    );
                    new_vars.push(name.clone());
                    dtypes.push(DType::Boolean);
                    variables.push(VariableDType {
                        name,
                        dtype: DType::Boolean,
                        source: Some(v.clone()),
                    });
                }
            }
        }
    }

    Ok(TypedTable {
        table,
        vars: new_vars,
        dtypes,
        variables,
    })
}

/// A boolean column may contain only numeric `0` and `1`. Missing cells are
/// skipped here because `--fill` owns the decision about them (and rejects a
/// fill that would produce a fraction later).
fn validate_boolean(t: &Table, j: usize) -> Result<(), DTypeError> {
    let name = &t.names[j];
    for r in 0..t.len() {
        let cell = t.cell_text(r, j);
        let cell = cell.trim();
        if is_missing_token(cell) {
            continue;
        }
        match cell.parse::<f64>() {
            Ok(v) if v == 0.0 || v == 1.0 => {}
            _ => {
                return Err(DTypeError::input(format!(
                    "boolean column '{}' must contain only 0 and 1; row {} (1-based, header excluded) has '{}'",
                    name,
                    r + 1,
                    cell
                )))
            }
        }
    }
    Ok(())
}

/// Distinct values of a categorical column, sorted and independent of row
/// order. All-numeric levels sort by value (`1, 2, 10`); otherwise the level
/// text sorts lexicographically. A missing/empty cell becomes the final
/// [`MISSING_LEVEL`] level.
fn categorical_levels(
    t: &Table,
    j: usize,
    max_levels: usize,
) -> Result<Vec<String>, DTypeError> {
    let mut set: BTreeSet<String> = BTreeSet::new();
    for r in 0..t.len() {
        let cell = t.cell_text(r, j);
        let cell = cell.trim();
        if is_missing_token(cell) {
            set.insert(MISSING_LEVEL.to_string());
        } else if cell == MISSING_LEVEL {
            return Err(DTypeError::usage(format!(
                "categorical column '{}' contains the literal '{}', which is reserved for missing cells; rename that value",
                t.names[j], MISSING_LEVEL
            )));
        } else {
            set.insert(cell.to_string());
        }
    }
    let mut levels: Vec<String> = set.into_iter().collect();
    let has_missing = levels.iter().any(|l| l == MISSING_LEVEL);
    levels.retain(|l| l != MISSING_LEVEL);
    let count = levels.len() + if has_missing { 1 } else { 0 };
    if count > max_levels {
        return Err(DTypeError::usage(format!(
            "categorical column '{}' has {} distinct values, more than --max-levels {}; raise the limit (e.g. --max-levels {})",
            t.names[j],
            count,
            max_levels,
            count
        )));
    }
    let numeric = !levels.is_empty() && levels.iter().all(|l| l.parse::<f64>().is_ok());
    if numeric {
        levels.sort_by(|a, b| {
            let av = a.parse::<f64>().unwrap_or(f64::NAN);
            let bv = b.parse::<f64>().unwrap_or(f64::NAN);
            av.partial_cmp(&bv)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(b))
        });
    }
    if has_missing {
        levels.push(MISSING_LEVEL.to_string());
    }
    Ok(levels)
}

/// Shape of a delimited text table, measured without building the `Table`.
///
/// `parse_table_with` already scans the whole text twice (header plus column
/// typing); this scan counts header fields and data rows so the memory guard
/// can estimate from the table's shape rather than from file size (issue #112).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvShape {
    /// Data rows (blank and `#` lines excluded).
    pub rows: usize,
    /// Header fields.
    pub cols: usize,
    /// Bytes of source text read (a resident `String` only when measured with
    /// [`csv_shape`]; the CLI measures the same field from a stream).
    pub bytes: u64,
}

/// Measure a delimited text buffer: delimiter, header width and data-row count.
///
/// The delimiter follows the loader (`\t` if the header contains a tab, else
/// `,`, else whitespace). The scan allocates nothing per line.
pub fn csv_shape(text: &str) -> CsvShape {
    let mut cols = 0usize;
    let mut rows = 0usize;
    let mut header_seen = false;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if !header_seen {
            header_seen = true;
            let d = if line.contains('\t') {
                '\t'
            } else if line.contains(',') {
                ','
            } else {
                ' '
            };
            cols = if d == ' ' {
                line.split_whitespace().count()
            } else {
                line.bytes().filter(|&b| b == d as u8).count() + 1
            };
            continue;
        }
        rows += 1;
    }
    CsvShape {
        rows,
        cols,
        bytes: text.len() as u64,
    }
}

/// Measure a delimited source **while reading it**, with the same quote-aware
/// record reader that builds the table (issue #126).
///
/// The scan holds one logical record at a time. It returns the table shape and
/// the number of bytes consumed, so the memory guard no longer needs a
/// whole-file `String` (or a whole-file `read_to_string`) to estimate a run.
pub fn csv_shape_stream<'a, F>(mut open: F) -> Result<CsvShape, String>
where
    F: FnMut() -> Result<Box<dyn std::io::BufRead + 'a>, String>,
{
    let mut rd = CsvRecords::new(open()?);
    let header = match rd.next() {
        Some(Ok(record)) => record,
        Some(Err(e)) => return Err(e),
        None => {
            return Ok(CsvShape {
                rows: 0,
                cols: 0,
                bytes: 0,
            })
        }
    };
    let cols = header.fields.len();
    let mut rows = 0usize;
    for record in &mut rd {
        let record = record?;
        // A ragged row is still a record the source contains; counting it keeps
        // the fallback estimate an upper bound before the parser refuses it.
        if !record.fields.is_empty() {
            rows += 1;
        }
    }
    Ok(CsvShape {
        rows,
        cols,
        bytes: rd.bytes,
    })
}

/// [`csv_shape_stream`] for a path, opening the file a second time (the parser
/// reopens it for its typing and building passes as well).
pub fn csv_shape_path(path: &str) -> Result<CsvShape, String> {
    csv_shape_stream(|| {
        std::fs::File::open(path)
            .map(|f| Box::new(std::io::BufReader::new(f)) as Box<dyn std::io::BufRead>)
            .map_err(|e| format!("{}: {}", path, e))
    })
}

/// Resident bytes per table cell: one `f64` value plus the `u32` code that a
/// low-cardinality numeric or text column keeps alongside it.
pub const CSV_TABLE_BYTES_PER_CELL: u64 = 12;

/// Per-row preparation cost: one row header per sample row, selector
/// bookkeeping, and the per-row copies the search and diagnostics keep.
///
/// This was 72 B before the #112 calibration. The 1M-row x 6-column run peaked
/// at 552 MiB while the old model predicted 378 MiB, so the fixed term was
/// raised to 240 B/row. At that value both the 6-column and 90-column measured
/// peaks are covered without exceeding 1.5x (see `docs/data-format.md`).
pub const CSV_PREP_FIXED_BYTES_PER_ROW: u64 = 240;

/// Per-row, per-used-column preparation cost: one `f64` in each of the
/// `cells`, train and test copies, plus each row's `Vec<f64>` header.
pub const CSV_PREP_BYTES_PER_USED_CELL: u64 = 24;

/// One measured memory estimate and its terms (issue #112). Reporting the
/// basis, rather than only the total, lets the preparation block show that the
/// guard used `rows x columns` and not a multiple of the file size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CsvEstimate {
    pub rows: usize,
    pub cols: usize,
    /// Columns used by the search (`--vars` + targets + weight); `cols` when no
    /// projection was supplied.
    pub used_cols: usize,
    pub text_bytes: u64,
    pub table_bytes: u64,
    pub prep_bytes: u64,
    /// The 25 % allocator slack included in `total_bytes`.
    pub slack_bytes: u64,
    pub total_bytes: u64,
}

impl CsvEstimate {
    /// Drop the resident text term. A streaming caller (issue #126) uses this;
    /// the terms stay the same so calibration still applies.
    pub fn without_text(mut self) -> Self {
        let base = self
            .table_bytes
            .saturating_add(self.prep_bytes);
        self.text_bytes = 0;
        self.total_bytes = base.saturating_mul(5) / 4;
        self.slack_bytes = self.total_bytes.saturating_sub(base);
        self
    }
}

/// Estimated peak resident bytes for loading and preparing a CSV (issue #112),
/// with the terms exposed.
///
/// The model is `text + rows x columns x 12 + rows x used_columns x 24` plus
/// `rows x 240 B` of fixed preparation overhead and 25 % allocator slack.
/// `used_cols` is the search's projection (variables plus targets); pass `0`
/// when it is not known yet and the whole header width is used. The text term
/// matters for a caller that holds the whole file, such as the current
/// `read_to_string` path; a streaming caller uses [`CsvEstimate::without_text`].
pub fn csv_estimate(shape: CsvShape, used_cols: usize) -> CsvEstimate {
    let (text, table, rows) = csv_terms(shape);
    let cols = shape.cols.max(1);
    let used = if used_cols == 0 {
        cols
    } else {
        used_cols.min(cols)
    };
    let prep = rows.saturating_mul(
        (used as u64)
            .saturating_mul(CSV_PREP_BYTES_PER_USED_CELL)
            .saturating_add(CSV_PREP_FIXED_BYTES_PER_ROW),
    );
    let base = text.saturating_add(table).saturating_add(prep);
    let total = base.saturating_mul(5) / 4;
    CsvEstimate {
        rows: shape.rows,
        cols: shape.cols,
        used_cols: used,
        text_bytes: text,
        table_bytes: table,
        prep_bytes: prep,
        slack_bytes: total.saturating_sub(base),
        total_bytes: total,
    }
}

/// Estimated peak resident bytes for the **streaming** CSV path (issue #126).
///
/// The CLI no longer keeps the file text in memory, so the text term is zero;
/// the shape is measured from the stream and the table/preparation terms are
/// the same as [`estimate_csv_bytes`]. `bytes` on the shape is still the number
/// of bytes read, for reporting, not a resident term.
pub fn estimate_csv_stream_bytes(shape: CsvShape, used_cols: usize) -> u64 {
    let (_text, table, rows) = csv_terms(shape);
    let cols = shape.cols.max(1) as u64;
    let used = if used_cols == 0 {
        cols
    } else {
        (used_cols as u64).min(cols)
    };
    let prep = rows.saturating_mul(
        used.saturating_mul(CSV_PREP_BYTES_PER_USED_CELL)
            .saturating_add(CSV_PREP_FIXED_BYTES_PER_ROW),
    );
    table.saturating_add(prep).saturating_mul(5) / 4
}

/// Estimated peak resident bytes for a streaming `rcsr import`: the resident
/// table plus allocator slack, with no whole-file text and no search sample.
pub fn estimate_csv_load_stream_bytes(shape: CsvShape) -> u64 {
    let (_text, table, _rows) = csv_terms(shape);
    table.saturating_mul(5) / 4
}

/// Estimated peak resident bytes for reading a CSV without preparing a search
/// sample (`rcsr import`): the text plus the resident table.
pub fn csv_load_estimate(shape: CsvShape) -> CsvEstimate {
    let (text, table, _) = csv_terms(shape);
    let base = text.saturating_add(table);
    let total = base.saturating_mul(5) / 4;
    CsvEstimate {
        rows: shape.rows,
        cols: shape.cols,
        used_cols: shape.cols,
        text_bytes: text,
        table_bytes: table,
        prep_bytes: 0,
        slack_bytes: total.saturating_sub(base),
        total_bytes: total,
    }
}

/// [`csv_estimate`]'s total, kept as the concise guard entry point.
pub fn estimate_csv_bytes(shape: CsvShape, used_cols: usize) -> u64 {
    csv_estimate(shape, used_cols).total_bytes
}

/// [`csv_load_estimate`]'s total, kept as the concise import guard entry point.
pub fn estimate_csv_load_bytes(shape: CsvShape) -> u64 {
    csv_load_estimate(shape).total_bytes
}

/// `(text, table, rows)`; `text` is the whole-file buffer, so an unmeasured
/// shape falls back to eight resident bytes per text byte.
fn csv_terms(shape: CsvShape) -> (u64, u64, u64) {
    let rows = shape.rows as u64;
    if rows == 0 || shape.cols == 0 {
        return (shape.bytes.saturating_mul(8), 0, 0);
    }
    let cols = shape.cols as u64;
    let table = rows
        .saturating_mul(cols)
        .saturating_mul(CSV_TABLE_BYTES_PER_CELL);
    (shape.bytes, table, rows)
}

pub fn read_table(path: &str) -> Result<Table, String> {
    read_table_with(path, false)
}

/// Like [`read_table`], but `skip_malformed` drops ragged rows instead of
/// refusing the file. The number dropped is kept on [`Table`].
pub fn read_table_with(path: &str, skip_malformed: bool) -> Result<Table, String> {
    parse_stream_with(
        || {
            std::fs::File::open(path)
                .map(|f| Box::new(std::io::BufReader::new(f)) as Box<dyn std::io::BufRead>)
                .map_err(|e| format!("{}: {}", path, e))
        },
        skip_malformed,
        None,
    )
}

/// Like [`read_table_with`], with an explicit time column name whose integral
/// cells are kept as an exact `i64` view even when the name is not time-like.
pub fn read_table_with_time(
    path: &str,
    skip_malformed: bool,
    time_col_name: Option<&str>,
) -> Result<Table, String> {
    parse_stream_with(
        || {
            std::fs::File::open(path)
                .map(|f| Box::new(std::io::BufReader::new(f)) as Box<dyn std::io::BufRead>)
                .map_err(|e| format!("{}: {}", path, e))
        },
        skip_malformed,
        time_col_name,
    )
}

/// One logical CSV record: the fields after quote-aware splitting and the
/// 1-based physical line on which the record starts.
struct Record {
    fields: Vec<String>,
    line: usize,
}

/// True when every `"` in `s` opens a quoted region that is closed again.
/// `""` inside a quoted region is an escaped quote, not a close.
fn quotes_balanced(s: &str) -> bool {
    let mut in_quotes = false;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes && c == '"' {
            if chars.peek() == Some(&'"') {
                chars.next();
            } else {
                in_quotes = false;
            }
        } else if !in_quotes && c == '"' {
            in_quotes = true;
        }
    }
    !in_quotes
}

/// Split one logical record into fields, honouring RFC 4180 quoting.
///
/// * a delimiter inside quotes is literal;
/// * CR/LF inside quotes is literal (the caller joins physical lines with
///   `\n`, so a CRLF file and an LF file produce the same value);
/// * `""` inside a quoted field is one literal `"`;
/// * unquoted fields keep the historical trim; quoted fields are exact;
/// * whitespace-delimited input still collapses runs of spaces.
fn split_record(record: &str, delim: char) -> Vec<String> {
    if delim == ' ' {
        return record.split_whitespace().map(|s| s.to_string()).collect();
    }
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    let mut after_quote = false;
    let mut quoted = false;
    let mut chars = record.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    field.push('"');
                } else {
                    in_quotes = false;
                    after_quote = true;
                }
            } else {
                field.push(c);
            }
        } else if after_quote {
            if c == delim {
                fields.push(std::mem::take(&mut field));
                after_quote = false;
                quoted = false;
            } else if !c.is_whitespace() {
                // Lenient with junk after a closing quote: keep it rather than
                // silently dropping data. (RFC 4180 calls this invalid.)
                field.push(c);
            }
        } else if c == '"' && field.trim().is_empty() {
            field.clear();
            in_quotes = true;
            quoted = true;
        } else if c == delim {
            fields.push(if quoted {
                std::mem::take(&mut field)
            } else {
                field.trim().to_string()
            });
            field.clear();
            quoted = false;
        } else {
            field.push(c);
        }
    }
    fields.push(if quoted {
        field
    } else {
        field.trim().to_string()
    });
    fields
}

/// Streaming record reader: one logical record in memory at a time.
struct CsvRecords<R: std::io::BufRead> {
    reader: R,
    /// 1-based number of the last physical line read.
    line: usize,
    /// Bytes consumed from the source, including record separators.
    bytes: u64,
    /// Chosen delimiter; `None` until the header has been sniffed.
    delim: Option<char>,
    /// A record read during sniffing that must be returned next.
    pending: Option<Record>,
}

impl<R: std::io::BufRead> CsvRecords<R> {
    fn new(reader: R) -> Self {
        CsvRecords {
            reader,
            line: 0,
            bytes: 0,
            delim: None,
            pending: None,
        }
    }

    /// Read one physical line without its `\n`/`\r\n`, advancing the line count.
    fn physical(&mut self) -> Option<Result<String, String>> {
        let mut bytes = Vec::new();
        match self.reader.read_until(b'\n', &mut bytes) {
            Ok(0) => None,
            Ok(n) => {
                self.bytes = self.bytes.saturating_add(n as u64);
                self.line += 1;
                while matches!(bytes.last(), Some(b'\n') | Some(b'\r')) {
                    bytes.pop();
                }
                match String::from_utf8(bytes) {
                    Ok(s) => Some(Ok(s)),
                    Err(_) => Some(Err(format!("line {}: input is not valid UTF-8", self.line))),
                }
            }
            Err(e) => Some(Err(format!("line {}: read error: {}", self.line + 1, e))),
        }
    }

    /// Read one logical record as raw text, joining physical lines while inside
    /// quotes. Blank lines and `#` comments between records are skipped.
    fn raw_record(&mut self) -> Option<Result<(String, usize), String>> {
        loop {
            let first = match self.physical()? {
                Ok(l) => l,
                Err(e) => return Some(Err(e)),
            };
            let trimmed = first.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let start = self.line;
            let mut raw = first;
            while !quotes_balanced(&raw) {
                match self.physical() {
                    Some(Ok(l)) => {
                        raw.push('\n');
                        raw.push_str(&l);
                    }
                    Some(Err(e)) => return Some(Err(e)),
                    None => {
                        return Some(Err(format!(
                            "line {}: unterminated quoted field (end of file inside quotes)",
                            start
                        )))
                    }
                }
            }
            return Some(Ok((raw, start)));
        }
    }

    /// Pick the delimiter from the first record (and, when present, the second).
    ///
    /// Candidates are tab, comma and semicolon; whitespace is the fallback. A
    /// delimiter that appears inside quotes does not count, and the second
    /// record breaks ties (a comma file whose header quotes a tab still gets a
    /// comma, because the counts agree for the comma and not for the tab).
    fn sniff(first: &str, second: Option<&str>) -> char {
        let mut best: Option<(char, usize)> = None;
        for &cand in &['\t', ',', ';'] {
            let n1 = split_record(first, cand).len();
            if n1 <= 1 {
                continue;
            }
            if let Some(second) = second {
                if split_record(second, cand).len() != n1 {
                    continue;
                }
            }
            match best {
                // `\t`, `,`, `;` in that order: ties keep the earlier one.
                Some((_, n)) if n >= n1 => {}
                _ => best = Some((cand, n1)),
            }
        }
        best.map(|(c, _)| c).unwrap_or(' ')
    }
}

impl<R: std::io::BufRead> Iterator for CsvRecords<R> {
    type Item = Result<Record, String>;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(rec) = self.pending.take() {
            return Some(Ok(rec));
        }
        if self.delim.is_none() {
            let (first, line) = match self.raw_record()? {
                Ok(v) => v,
                Err(e) => return Some(Err(e)),
            };
            // Sniff with a one-record look-ahead. The look-ahead is at most one
            // record, so this stays streaming.
            let second = match self.raw_record() {
                Some(Ok((raw, line))) => Some((raw, line)),
                Some(Err(e)) => return Some(Err(e)),
                None => None,
            };
            let delim = CsvRecords::<std::io::Empty>::sniff(
                &first,
                second.as_ref().map(|(raw, _)| raw.as_str()),
            );
            self.delim = Some(delim);
            if let Some((raw, line)) = second {
                self.pending = Some(Record {
                    fields: split_record(&raw, delim),
                    line,
                });
            }
            return Some(Ok(Record {
                fields: split_record(&first, delim),
                line,
            }));
        }
        let delim = self.delim.unwrap();
        match self.raw_record()? {
            Ok((raw, line)) => Some(Ok(Record {
                fields: split_record(&raw, delim),
                line,
            })),
            Err(e) => Some(Err(e)),
        }
    }
}

/// Parse a delimited table (header row required, `,`/tab/whitespace detected).
///
/// A data row whose field count differs from the header is an error naming its
/// 1-based file line. Blank lines and lines whose first non-space character is
/// `#` are not data rows. Missing *cells* in a correct-width row are still
/// handled by `--fill`; this is about the wrong number of fields.
pub fn parse_table(text: &str) -> Result<Table, String> {
    parse_table_with(text, false)
}

/// [`parse_table`] with an explicit choice for ragged rows. With
/// `skip_malformed`, rows whose field count differs from the header are
/// dropped and counted (`Table::skipped_malformed`) instead of refused.
pub fn parse_table_with(text: &str, skip_malformed: bool) -> Result<Table, String> {
    parse_stream_with(
        || {
            Ok(Box::new(std::io::Cursor::new(text.as_bytes()))
                as Box<dyn std::io::BufRead + '_>)
        },
        skip_malformed,
        None,
    )
}

/// [`parse_table_with`] with an explicit time column name (for
/// `rcsr import --time <col>`). Integral cells in that column are retained as
/// exact `i64` values; a non-integral cell falls back to the f64-only layout.
pub fn parse_table_with_time(
    text: &str,
    skip_malformed: bool,
    time_col_name: Option<&str>,
) -> Result<Table, String> {
    parse_stream_with(
        || {
            Ok(Box::new(std::io::Cursor::new(text.as_bytes()))
                as Box<dyn std::io::BufRead + '_>)
        },
        skip_malformed,
        time_col_name,
    )
}

/// Two streaming passes over the same source: the first fixes the width,
/// counts ragged rows and decides each column's type; the second builds the
/// columns. The source is reopened instead of being held in memory.
fn parse_stream_with<'a, F>(
    mut open: F,
    skip_malformed: bool,
    time_col_name: Option<&str>,
) -> Result<Table, String>
where
    F: FnMut() -> Result<Box<dyn std::io::BufRead + 'a>, String>,
{
    // Pass 1: header, ragged rows, and numeric-vs-categorical.
    let mut rd = CsvRecords::new(open()?);
    let header = match rd.next() {
        Some(Ok(r)) => r,
        Some(Err(e)) => return Err(e),
        None => return Err("empty input".to_string()),
    };
    let names = header.fields;
    let nfields = names.len();
    let mut skipped_malformed = 0usize;
    let mut data_rows = 0usize;
    let mut any = vec![false; nfields];
    let mut all = vec![true; nfields];
    for rec in rd {
        let rec = rec?;
        if rec.fields.len() != nfields {
            if skip_malformed {
                skipped_malformed += 1;
                continue;
            }
            return Err(format!(
                "line {}: expected {} field(s), found {} (ragged row; pass \
                 --skip-malformed to drop such rows instead)",
                rec.line,
                nfields,
                rec.fields.len()
            ));
        }
        data_rows += 1;
        for (j, cell) in rec.fields.iter().enumerate() {
            let cell = cell.trim();
            if is_missing_token(cell) {
                continue;
            }
            any[j] = true;
            if cell.parse::<f64>().is_err() {
                all[j] = false;
            }
        }
    }
    if data_rows == 0 {
        return Err("no data rows".to_string());
    }
    // A column is numeric when every non-missing cell parses as a number.
    // Missing tokens (`''`, `NA`, `NaN`, `null`, case-insensitively) are skipped
    // and become `NaN` in the numeric view.
    let numeric: Vec<bool> = any.iter().zip(&all).map(|(&a, &b)| a && b).collect();

    // Pass 2: one contiguous `f64` block per numeric column and, where the
    // source text is worth keeping, one `u32` code per row plus a level table.
    let mut rd = CsvRecords::new(open()?);
    match rd.next() {
        Some(Ok(_header)) => {}
        Some(Err(e)) => return Err(e),
        None => return Err("empty input".to_string()),
    }
    let mut columns: Vec<Column> = (0..nfields).map(|_| Column::default()).collect();
    for (j, col) in columns.iter_mut().enumerate() {
        if numeric[j] {
            col.values.reserve(data_rows);
        }
    }
    let mut maps: Vec<HashMap<String, u32>> = (0..nfields).map(|_| HashMap::new()).collect();
    let mut level_lists: Vec<Vec<String>> = (0..nfields).map(|_| Vec::new()).collect();
    let mut codes: Vec<Vec<u32>> = (0..nfields).map(|_| Vec::with_capacity(data_rows)).collect();
    // A numeric column only keeps its dictionary while it stays low-cardinality.
    // Once it crosses the cap the map is dropped immediately, so a large random
    // column never builds one `String` key per distinct value.
    let mut dict_live: Vec<bool> = (0..nfields).map(|_| true).collect();
    // A time-like column whose cells are all integers keeps an exact `i64`
    // view alongside the `f64` evaluator view. Names are the only signal the
    // CSV reader has before `--time` is chosen, and a false negative is safe:
    // the column simply falls back to binary64 (the old behaviour).
    let mut int_candidate: Vec<bool> = (0..nfields)
        .map(|j| {
            numeric[j]
                && (time_name(&names[j])
                    || time_col_name.is_some_and(|t| t == names[j]))
        })
        .collect();
    let mut ints: Vec<Vec<i64>> = (0..nfields)
        .map(|j| {
            if int_candidate[j] {
                Vec::with_capacity(data_rows)
            } else {
                Vec::new()
            }
        })
        .collect();
    for rec in rd {
        let rec = rec?;
        if rec.fields.len() != nfields {
            continue; // accepted or skipped by pass 1
        }
        for (j, cell) in rec.fields.iter().enumerate() {
            let cell = cell.trim();
            if numeric[j] {
                columns[j].values.push(if is_missing_token(cell) {
                    f64::NAN
                } else {
                    cell.parse::<f64>().unwrap_or(f64::NAN)
                });
                if int_candidate[j] {
                    if is_missing_token(cell) {
                        ints[j].push(i64::MIN);
                    } else {
                        match cell.parse::<i64>() {
                            Ok(v) => ints[j].push(v),
                            Err(_) => {
                                int_candidate[j] = false;
                                ints[j] = Vec::new();
                            }
                        }
                    }
                }
            }
            if !dict_live[j] {
                continue;
            }
            let code = match maps[j].get(cell) {
                Some(&c) => c,
                None => {
                    if maps[j].len() >= NUMERIC_DICT_MAX && numeric[j] {
                        dict_live[j] = false;
                        maps[j] = HashMap::new();
                        level_lists[j] = Vec::new();
                        codes[j] = Vec::new();
                        continue;
                    }
                    let c = level_lists[j].len() as u32;
                    maps[j].insert(cell.to_string(), c);
                    level_lists[j].push(cell.to_string());
                    c
                }
            };
            codes[j].push(code);
        }
    }

    for (j, col) in columns.iter_mut().enumerate() {
        if int_candidate[j] {
            col.ints = Some(std::mem::take(&mut ints[j]));
        }
        if dict_live[j] {
            col.dict = Some(DictColumn {
                codes: std::mem::take(&mut codes[j]),
                levels: std::mem::take(&mut level_lists[j]),
            });
        }
    }

    Ok(Table::from_parts(
        names,
        numeric,
        columns,
        data_rows,
        skipped_malformed,
    ))
}

// ---------------------------------------------------------------------------
// Splitting
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum SplitMode {
    Random,
    Temporal,
    Group,
    TemporalGroup,
}

impl SplitMode {
    pub fn name(self) -> &'static str {
        match self {
            SplitMode::Random => "random",
            SplitMode::Temporal => "temporal",
            SplitMode::Group => "group",
            SplitMode::TemporalGroup => "temporal-group",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "random" => Ok(SplitMode::Random),
            "temporal" | "time" => Ok(SplitMode::Temporal),
            "group" | "grouped" => Ok(SplitMode::Group),
            "temporal-group" | "temporal_group" | "panel" => Ok(SplitMode::TemporalGroup),
            other => Err(format!(
                "unknown split mode '{}' (expected random, temporal, group or temporal-group)",
                other
            )),
        }
    }
}

#[derive(Clone)]
pub struct SplitSpec {
    pub mode: SplitMode,
    /// Index of the time/ordering column (`temporal*` modes).
    pub time_col: Option<usize>,
    /// Index of the group column (`group` and `temporal_group` modes).
    pub group_col: Option<usize>,
    pub test_frac: f64,
    /// Rows dropped immediately after the train/test boundary (embargo).
    pub gap: usize,
    /// Fraction of the training period held out as a **validation** slice,
    /// taken chronologically from its end (issue #111). `0` disables the slice.
    /// Model selection may read train and validation; the test block stays a
    /// lockbox.
    pub validation_frac: f64,
    /// Number of contiguous evaluation blocks in the test region.
    pub folds: usize,
    /// Rows where this expression is true are assigned to training (issue #3).
    /// When either selection is present it replaces the positional mode.
    pub train_select: Option<Filter>,
    /// Rows where this expression is true are assigned to test (issue #3).
    pub test_select: Option<Filter>,
    /// Rows where this expression is true are dropped before splitting (#3).
    pub exclude: Option<Filter>,
    /// #130: how the time column is parsed.
    pub time: TimeSpec,
}

impl Default for SplitSpec {
    fn default() -> Self {
        SplitSpec {
            mode: SplitMode::Random,
            time_col: None,
            group_col: None,
            test_frac: 0.25,
            gap: 0,
            validation_frac: 0.0,
            folds: 1,
            train_select: None,
            test_select: None,
            exclude: None,
            time: TimeSpec::default(),
        }
    }
}

/// What the split actually did, for reporting.
#[derive(Clone, Default)]
pub struct SplitReport {
    pub mode: &'static str,
    pub rows_total: usize,
    pub train_rows: usize,
    pub test_rows: usize,
    /// Rows carved out of the training period as the validation slice (#111).
    pub validation_rows: usize,
    /// Training rows dropped as a purge/embargo between fit-train and
    /// validation (#111).
    pub purged_validation_rows: usize,
    pub folds: usize,
    pub gap: usize,
    pub groups_total: usize,
    pub groups_train: usize,
    pub groups_test: usize,
    pub train_time_max: Option<f64>,
    pub test_time_min: Option<f64>,
    /// Rows dropped by `--exclude` before splitting (issue #3).
    pub excluded_rows: usize,
    /// Rows matching both selections; they are assigned to training (#3).
    pub overlap_rows: usize,
    /// Rows matching neither selection; unused (#3).
    pub unassigned_rows: usize,
    /// Which [`RowSelector`] representation the train/fold rows use (#70):
    /// `all`, `range`, `hash` or `set`. Reported so a reader can see whether
    /// evaluation is a scan or a gather.
    pub selector: String,
    pub notes: Vec<String>,
}

pub struct Prepared {
    pub train: Sample,
    /// Chronological validation slice carved from the end of the training
    /// period (#111), when `SplitSpec::validation_frac > 0`. Selection may use
    /// it; the search never fits on it.
    pub validation: Option<Sample>,
    /// All test folds concatenated, in fold order (what the search evaluates on).
    pub test: Sample,
    /// One sample per evaluation block when there is more than one fold. With
    /// `folds == 1` this is empty and `test` is the single block (issue #133:
    /// the pooled block is not built twice).
    pub fold_samples: Vec<Sample>,
    /// Affine denormalisation constants: original = normalised * stds + means.
    /// With the default centre/scale these are the training mean and standard
    /// deviation, exactly as before.
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
    pub report: SplitReport,
    /// What data preparation did (missing values, outliers, smoothing,
    /// normalisation, row filter).
    pub prep: PrepReport,
}

// ---------------------------------------------------------------------------
// Data preparation (issue #2)
// ---------------------------------------------------------------------------

/// How to handle a missing cell in a used column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FillMethod {
    /// Leave it missing (today's behaviour; the default).
    None,
    /// Drop any row with a missing value in a used column.
    Drop,
    /// Column mean, computed from the training rows.
    Mean,
    /// Column median, computed from the training rows.
    Median,
    /// A literal zero.
    Zero,
    /// The previous non-missing value in the same side, in row order.
    Previous,
    /// Linear interpolation between the neighbouring non-missing values in the
    /// same side, in row order.
    Interpolate,
}

impl FillMethod {
    pub fn name(self) -> &'static str {
        match self {
            FillMethod::None => "none",
            FillMethod::Drop => "drop",
            FillMethod::Mean => "mean",
            FillMethod::Median => "median",
            FillMethod::Zero => "zero",
            FillMethod::Previous => "previous",
            FillMethod::Interpolate => "interpolate",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" => Ok(FillMethod::None),
            "drop" | "ignore" => Ok(FillMethod::Drop),
            "mean" => Ok(FillMethod::Mean),
            "median" => Ok(FillMethod::Median),
            "zero" | "0" => Ok(FillMethod::Zero),
            "previous" | "prev" | "forward" => Ok(FillMethod::Previous),
            "interpolate" | "linear" | "interp" => Ok(FillMethod::Interpolate),
            other => Err(format!(
                "unknown --fill method '{}' (expected none, drop, mean, median, zero, previous or interpolate)",
                other
            )),
        }
    }
}

/// What to subtract before the search (the `--center` option).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Center {
    Mean,
    Median,
    Min,
    Max,
    None,
}

impl Center {
    pub fn name(self) -> &'static str {
        match self {
            Center::Mean => "mean",
            Center::Median => "median",
            Center::Min => "min",
            Center::Max => "max",
            Center::None => "none",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "mean" => Ok(Center::Mean),
            "median" => Ok(Center::Median),
            "min" | "lower" => Ok(Center::Min),
            "max" | "upper" => Ok(Center::Max),
            "none" | "off" => Ok(Center::None),
            other => Err(format!(
                "unknown --center '{}' (expected mean, median, min, max or none)",
                other
            )),
        }
    }
}

/// What to divide by before the search (the `--scale` option).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Scale {
    Std,
    Iqr,
    Range,
    None,
}

impl Scale {
    pub fn name(self) -> &'static str {
        match self {
            Scale::Std => "std",
            Scale::Iqr => "iqr",
            Scale::Range => "range",
            Scale::None => "none",
        }
    }

    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "std" | "sd" | "standard-deviation" | "standard_deviation" => Ok(Scale::Std),
            "iqr" => Ok(Scale::Iqr),
            "range" | "minmax" | "min-max" => Ok(Scale::Range),
            "none" | "off" => Ok(Scale::None),
            other => Err(format!(
                "unknown --scale '{}' (expected std, iqr, range or none)",
                other
            )),
        }
    }
}

/// A parsed row filter. The tree was parsed against `vars`, so input indices in
/// the tree refer to positions in this slice.
#[derive(Clone)]
pub struct Filter {
    pub tree: Node,
    pub vars: Vec<String>,
    /// Human-readable form, for the report.
    pub expression: String,
}

/// Opt-in data preparation. Every default reproduces the behaviour before
/// issue #2: no filling, no outlier removal, no smoothing, mean/std
/// normalisation and no row filter.
#[derive(Clone)]
pub struct PrepSpec {
    pub fill: FillMethod,
    /// Drop rows more than this many interquartile ranges from the column
    /// median (training rows define the bounds).
    pub outlier_iqr: Option<f64>,
    /// Moving-average window in rows (`0` disables it). Trailing by default.
    pub smooth: usize,
    /// #110: use the centred window (reads future bars). Explicit opt-in only.
    pub smooth_centred: bool,
    /// #110: allow `--fill interpolate` on time-ordered data (it reads the
    /// next bar). Off by default.
    pub allow_interpolate: bool,
    pub center: Center,
    pub scale: Scale,
    pub filter: Option<Filter>,
    /// Datatype per input variable, aligned with `vars` after [`apply_dtypes`]
    /// expansion. Empty means every input is continuous. A `Boolean` column is
    /// never centred or scaled (see `docs/variable-types.md`).
    pub dtypes: Vec<DType>,
}

impl Default for PrepSpec {
    fn default() -> Self {
        PrepSpec {
            fill: FillMethod::None,
            outlier_iqr: None,
            smooth: 0,
            smooth_centred: false,
            allow_interpolate: false,
            center: Center::Mean,
            scale: Scale::Std,
            filter: None,
            dtypes: Vec::new(),
        }
    }
}

/// Counts and methods actually used during preparation, for the report and the
/// `preparation` object in `--json` / `--ndjson`.
#[derive(Clone)]
pub struct PrepReport {
    pub fill: &'static str,
    /// Rows with at least one cell filled.
    pub rows_filled: usize,
    pub rows_dropped_missing: usize,
    pub rows_dropped_outlier: usize,
    /// Number of cells replaced by a smoothing window.
    pub smoothed: usize,
    pub center: &'static str,
    pub scale: &'static str,
    /// Rows removed by `--filter`.
    pub filtered: usize,
    /// Data rows dropped by `--skip-malformed` because their field count did
    /// not match the header.
    pub rows_skipped_malformed: usize,
    /// #110: smoothing window orientation (`trailing` or `centred`).
    pub smooth_mode: &'static str,
    /// #110: time-column hazards. Empty unit means no `--time` column.
    pub time_unit: String,
    pub time_rows: usize,
    /// Rows whose numeric instant repeats an earlier row (after unit scaling).
    pub time_duplicates: usize,
    /// Rows where the time value decreases relative to the previous row.
    pub time_non_monotonic: usize,
    /// Numeric rows rescaled from ms/us/ns to seconds.
    pub time_converted: usize,
    /// #112: the memory guard's estimate for this CSV and its basis. `None`
    /// for a `.rcsb` store or when the source did not go through the CSV guard.
    pub memory_estimate: Option<CsvEstimate>,
}

impl Default for PrepReport {
    fn default() -> Self {
        PrepReport {
            fill: "none",
            rows_filled: 0,
            rows_dropped_missing: 0,
            rows_dropped_outlier: 0,
            smoothed: 0,
            center: "mean",
            scale: "std",
            filtered: 0,
            rows_skipped_malformed: 0,
            smooth_mode: "trailing",
            time_unit: String::new(),
            time_rows: 0,
            time_duplicates: 0,
            time_non_monotonic: 0,
            time_converted: 0,
            memory_estimate: None,
        }
    }
}

impl PrepReport {
    /// True when at least one non-default transformation ran.
    pub fn active(&self) -> bool {
        self.fill != "none"
            || self.rows_dropped_missing > 0
            || self.rows_dropped_outlier > 0
            || self.smoothed > 0
            || self.center != "mean"
            || self.scale != "std"
            || self.filtered > 0
            || self.rows_skipped_malformed > 0
            || self.smoothed > 0
    }

    /// One short, commented line per active transformation. `params` carries
    /// the raw option values that are not part of the counts.
    pub fn text_lines(&self, outlier_iqr: Option<f64>, smooth: usize) -> Vec<String> {
        let mut out = Vec::new();
        if let Some(m) = self.memory_estimate {
            out.push(format!(
                "# preparation: memory estimate {} B from {} rows x {} columns ({} used); \
                 basis: text {} B + table {} B + preparation {} B + slack {} B",
                m.total_bytes,
                m.rows,
                m.cols,
                m.used_cols,
                m.text_bytes,
                m.table_bytes,
                m.prep_bytes,
                m.slack_bytes
            ));
        }
        if self.fill != "none" || self.rows_dropped_missing > 0 {
            out.push(format!(
                "# preparation: fill={} rows_filled={} rows_dropped_missing={}",
                self.fill, self.rows_filled, self.rows_dropped_missing
            ));
        }
        if let Some(k) = outlier_iqr {
            if k > 0.0 {
                out.push(format!(
                    "# preparation: outlier-iqr={} rows_dropped_outlier={}",
                    k, self.rows_dropped_outlier
                ));
            }
        }
        if smooth > 0 {
            out.push(format!(
                "# preparation: smooth={} mode={} cells_smoothed={}",
                smooth, self.smooth_mode, self.smoothed
            ));
            if self.smooth_mode == "centred" {
                out.push(
                    "# preparation: warning: centred smoothing reads future bars (look-ahead)"
                        .to_string(),
                );
            }
        }
        if self.center != "mean" || self.scale != "std" {
            out.push(format!(
                "# preparation: center={} scale={}",
                self.center, self.scale
            ));
        }
        if self.filtered > 0 {
            out.push(format!(
                "# preparation: filter removed {} row(s)",
                self.filtered
            ));
        }
        if self.rows_skipped_malformed > 0 {
            out.push(format!(
                "# preparation: --skip-malformed dropped {} ragged row(s)",
                self.rows_skipped_malformed
            ));
        }
        if !self.time_unit.is_empty() {
            out.push(format!(
                "# preparation: time unit={} rows={} duplicates={} non_monotonic={} converted={}",
                self.time_unit,
                self.time_rows,
                self.time_duplicates,
                self.time_non_monotonic,
                self.time_converted
            ));
            if self.time_duplicates > 0 {
                out.push(format!(
                    "# preparation: warning: {} duplicate time instant(s); de-duplicate by \
                     instant before a temporal split",
                    self.time_duplicates
                ));
            }
            if self.time_non_monotonic > 0 {
                out.push(format!(
                    "# preparation: warning: time is not monotonic in row order ({} step(s)); \
                     rows were ordered for the split, check the source",
                    self.time_non_monotonic
                ));
            }
        }
        out
    }

    /// The `preparation` object shared by `--json` and `--ndjson`.
    pub fn json(&self) -> String {
        let memory = match self.memory_estimate {
            Some(m) => format!(
                "{{\"estimate_bytes\":{},\"rows\":{},\"columns\":{},\"used_columns\":{},\
                 \"text_bytes\":{},\"table_bytes\":{},\"preparation_bytes\":{},\
                 \"slack_bytes\":{},\"basis\":\"rows x columns\"}}",
                m.total_bytes,
                m.rows,
                m.cols,
                m.used_cols,
                m.text_bytes,
                m.table_bytes,
                m.prep_bytes,
                m.slack_bytes
            ),
            None => "null".to_string(),
        };
        format!(
            "{{\"fill\":\"{}\",\"rows_filled\":{},\"rows_dropped_missing\":{},\"rows_dropped_outlier\":{},\"smoothed\":{},\"smooth_mode\":\"{}\",\"center\":\"{}\",\"scale\":\"{}\",\"filtered\":{},\"rows_skipped_malformed\":{},\"time_unit\":\"{}\",\"time_rows\":{},\"time_duplicates\":{},\"time_non_monotonic\":{},\"time_converted\":{},\"memory\":{}}}",
            self.fill,
            self.rows_filled,
            self.rows_dropped_missing,
            self.rows_dropped_outlier,
            self.smoothed,
            self.smooth_mode,
            self.center,
            self.scale,
            self.filtered,
            self.rows_skipped_malformed,
            self.time_unit,
            self.time_rows,
            self.time_duplicates,
            self.time_non_monotonic,
            self.time_converted,
            memory
        )
    }
}

/// #110: describe the time column without changing it: detected unit,
/// duplicate instants (by numeric instant, so `1673343780` and
/// `1673343780.0` collide) and non-monotonic steps in row order.
struct TimeHazards {
    unit: String,
    rows: usize,
    duplicates: usize,
    non_monotonic: usize,
    converted: usize,
}

fn time_hazards(t: &Table, spec: &SplitSpec) -> TimeHazards {
    let mut out = TimeHazards {
        unit: String::new(),
        rows: 0,
        duplicates: 0,
        non_monotonic: 0,
        converted: 0,
    };
    let Some(col) = spec.time_col else {
        return out;
    };
    if col >= t.n_cols() {
        return out;
    }
    let n = t.len();
    let numeric = t.numeric[col];
    let mut vals: Vec<f64> = if numeric {
        t.col(col).to_vec()
    } else if let Some(d) = t.dict(col) {
        (0..n).map(|r| parse_time(d.text(r)).unwrap_or(f64::NAN)).collect()
    } else {
        t.col(col).to_vec()
    };
    out.rows = n;
    let mut unit = "s";
    let mut factor = 1.0;
    if numeric {
        let (lo, hi) = vals
            .iter()
            .filter(|v| v.is_finite())
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        // Same disjoint epoch windows as the strict parser in #130. Small
        // columns are a plain index/seconds axis and keep their meaning.
        let units = [
            ("s", 1.0, 1.0e8, 4.2e9),
            ("ms", 1e-3, 1.0e11, 4.2e12),
            ("us", 1e-6, 1.0e14, 4.2e15),
            ("ns", 1e-9, 1.0e17, 4.2e18),
        ];
        if lo >= 0.0 && hi < 1.0e8 {
            // index-like
        } else {
            for (name, f, a, b) in units {
                if lo >= a && hi <= b {
                    unit = name;
                    factor = f;
                    break;
                }
            }
        }
        if unit != "s" {
            out.converted = vals.iter().filter(|v| v.is_finite()).count();
        }
    }
    if factor != 1.0 {
        for v in vals.iter_mut() {
            if v.is_finite() {
                *v *= factor;
            }
        }
    }
    let mut seen: HashSet<u64> = HashSet::new();
    let mut prev: Option<f64> = None;
    for &v in &vals {
        if !v.is_finite() {
            continue;
        }
        if !seen.insert(v.to_bits()) {
            out.duplicates += 1;
        }
        if let Some(p) = prev {
            if v < p {
                out.non_monotonic += 1;
            }
        }
        prev = Some(v);
    }
    out.unit = unit.to_string();
    out
}

/// Build normalised train/test samples. Returns (train, test, means, stds).
///
/// Backwards-compatible wrapper around [`prepare_full`] using a random split.
pub fn prepare(
    t: &Table,
    vars: &[String],
    target: &str,
    test_frac: f64,
    rng: &mut Mt,
) -> Result<(Sample, Sample, Vec<f64>, Vec<f64>), String> {
    let spec = SplitSpec {
        test_frac,
        ..Default::default()
    };
    let p = prepare_full(t, vars, target, &spec, rng)?;
    Ok((p.train, p.test, p.means, p.stds))
}

/// Resolve the input columns and the target column, validating that they are
/// numeric and distinct. Shared by [`prepare_full`] and [`cv_folds`].
fn resolve_columns(t: &Table, vars: &[String], target: &str) -> Result<(Vec<usize>, usize), String> {
    let var_idx: Vec<usize> = vars
        .iter()
        .map(|v| {
            let j = t.index_of(v)?;
            if !t.numeric[j] {
                return Err(format!(
                    "column '{}' is not numeric and cannot be used as an input variable",
                    v
                ));
            }
            if v == target {
                return Err(format!("column '{}' is both the target and an input", v));
            }
            Ok(j)
        })
        .collect::<Result<_, _>>()?;
    let tgt_idx = t.index_of(target)?;
    if !t.numeric[tgt_idx] {
        return Err(format!("target column '{}' is not numeric", target));
    }
    Ok((var_idx, tgt_idx))
}

/// Multi-output variant of [`resolve_columns`] (issue #15): every named target
/// must be numeric, distinct and none may also be an input.
fn resolve_columns_multi(
    t: &Table,
    vars: &[String],
    targets: &[&str],
) -> Result<(Vec<usize>, Vec<usize>), String> {
    if targets.is_empty() {
        return Err("no target column was given".to_string());
    }
    let mut var_idx = Vec::with_capacity(vars.len());
    for v in vars {
        let j = t.index_of(v)?;
        if !t.numeric[j] {
            return Err(format!(
                "column '{}' is not numeric and cannot be used as an input variable",
                v
            ));
        }
        if targets.iter().any(|t| *t == v.as_str()) {
            return Err(format!("column '{}' is both a target and an input", v));
        }
        var_idx.push(j);
    }
    let mut tgt_idx = Vec::with_capacity(targets.len());
    for target in targets {
        let j = t.index_of(target)?;
        if !t.numeric[j] {
            return Err(format!("target column '{}' is not numeric", target));
        }
        if tgt_idx.contains(&j) {
            return Err(format!("target column '{}' was named more than once", target));
        }
        tgt_idx.push(j);
    }
    Ok((var_idx, tgt_idx))
}

/// Resolve and validate the `--weight` column. Returns its index, or `None`
/// when no weight column was requested.
fn resolve_weight_column(t: &Table, weight: Option<&str>) -> Result<Option<usize>, String> {
    let Some(name) = weight else {
        return Ok(None);
    };
    let j = match t.names.iter().position(|n| n == name) {
        Some(j) => j,
        None => {
            return Err(format!(
                "weight column '{}' not found (columns: {:?})",
                name, t.names
            ))
        }
    };
    if !t.numeric[j] {
        return Err(format!(
            "weight column '{}' is not numeric; weights must be finite and >= 0",
            name
        ));
    }
    Ok(Some(j))
}

/// Check the weight values as a whole: every value finite and `>= 0`, and at
/// least one strictly positive. The error names the column and the number of
/// offending rows.
fn validate_weights(name: &str, w: &[f64]) -> Result<(), String> {
    let non_finite = w.iter().filter(|v| !v.is_finite()).count();
    if non_finite > 0 {
        return Err(format!(
            "weight column '{}' has {} non-finite value(s); weights must be finite and >= 0",
            name, non_finite
        ));
    }
    let negative = w.iter().filter(|&&v| v < 0.0).count();
    if negative > 0 {
        return Err(format!(
            "weight column '{}' has {} negative value(s); weights must be finite and >= 0",
            name, negative
        ));
    }
    if w.iter().all(|&v| v == 0.0) {
        return Err(format!(
            "weight column '{}' is all zero; at least one row must have a positive weight",
            name
        ));
    }
    Ok(())
}

// #133: test-only count of row-major `cells` copies built by preparation.
#[cfg(test)]
thread_local! {
    static CELL_COPIES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

pub fn prepare_full(
    t: &Table,
    vars: &[String],
    target: &str,
    spec: &SplitSpec,
    rng: &mut Mt,
) -> Result<Prepared, String> {
    prepare_full_with(t, vars, target, spec, rng, &PrepSpec::default())
}

/// Build normalised train/test samples after applying the opt-in preparation
/// in `prep`.
///
/// Order is what keeps the split honest:
///
/// 1. the row filter selects rows, then the train/test split is drawn;
/// 2. missing values are filled or dropped using statistics from the **training
///    rows only** (a `previous`/`interpolate` fill never crosses the boundary);
/// 3. outlier bounds are the training median and IQR;
/// 4. smoothing runs inside each side, so a training window never reads a test
///    value;
/// 5. centre/scale are computed from the surviving training rows and applied to
///    both sides. `means`/`stds` remain the affine denormalisation constants, so
///    a printed formula is always in original units.
pub fn prepare_full_with(
    t: &Table,
    vars: &[String],
    target: &str,
    spec: &SplitSpec,
    rng: &mut Mt,
    prep: &PrepSpec,
) -> Result<Prepared, String> {
    prepare_full_weighted_with(t, vars, target, None, spec, rng, prep)
}

/// Like [`prepare_full_with`], with an optional per-row weight column.
///
/// `weight` names a numeric column whose values become [`Sample::w`] on the
/// training sample, every test fold and the pooled test sample. The column is
/// never an input variable. Its values are validated **after** `--exclude` and
/// `--filter` have selected the population and must be finite, non-negative and
/// not all zero; the error names the column and the number of offending rows.
/// When `weight` is `None` the samples carry an empty `w` and every calculation
/// is bit-for-bit the unweighted one.
pub fn prepare_full_weighted_with(
    t: &Table,
    vars: &[String],
    target: &str,
    weight: Option<&str>,
    spec: &SplitSpec,
    rng: &mut Mt,
    prep: &PrepSpec,
) -> Result<Prepared, String> {
    prepare_full_targets_weighted_with(t, vars, &[target], weight, spec, rng, prep)
}

/// Multi-target preparation (issue #15). The split, missing-value handling,
/// outlier bounds, smoothing and input normalisation are computed **once** over
/// every target, so all outputs stay row-aligned by construction. The single
/// target wrapper above is the identical code path with a one-element slice.
pub fn prepare_full_targets_weighted_with(
    t: &Table,
    vars: &[String],
    targets: &[&str],
    weight: Option<&str>,
    spec: &SplitSpec,
    rng: &mut Mt,
    prep: &PrepSpec,
) -> Result<Prepared, String> {
    // With an explicit --fill method, a junk token in an otherwise numeric
    // column is treated as a missing value (and then filled or dropped).
    // Without it the column stays non-numeric, exactly as before. #133: only
    // clone/scan the table when a named input or target is actually non-numeric;
    // an all-numeric store must not pay for a second full columnar image.
    let needs_coerce = prep.fill != FillMethod::None
        && (targets.iter().any(|name| {
            t.index_of(name)
                .map(|j| !t.numeric[j])
                .unwrap_or(false)
        }) || vars.iter().any(|name| {
            t.index_of(name)
                .map(|j| !t.numeric[j])
                .unwrap_or(false)
        }));
    let coerced = if needs_coerce {
        Some(coerce_numeric(t))
    } else {
        None
    };
    let t = coerced.as_ref().unwrap_or(t);
    // Keep the historical single-target resolver (and its error text) exactly;
    // the multi resolver only runs when there really are several targets.
    let (var_idx, tgt_idx) = if targets.len() == 1 {
        let (v, tgt) = resolve_columns(t, vars, targets[0])?;
        (v, vec![tgt])
    } else {
        resolve_columns_multi(t, vars, targets)?
    };
    let nt = tgt_idx.len();
    let weight_idx = resolve_weight_column(t, weight)?;
    let nv = var_idx.len();
    if !prep.dtypes.is_empty() && prep.dtypes.len() != nv {
        return Err(format!(
            "internal datatype plan has {} entries for {} input variables",
            prep.dtypes.len(),
            nv
        ));
    }
    let dtype_at = |j: usize| prep.dtypes.get(j).copied().unwrap_or(DType::Continuous);

    // 0. Explicit exclusions remove rows from the population (issue #3). They
    // run before the row filter, so `--exclude` can never be undone by it.
    // The population is a boolean view over the column-major table: no second
    // copy of the cells (and no text copy) is made while filtering.
    let mut excluded_count = 0usize;
    let mut keep_row = vec![true; t.len()];
    if let Some(x) = &spec.exclude {
        let cols: Vec<usize> = x
            .vars
            .iter()
            .map(|v| t.index_of(v))
            .collect::<Result<_, _>>()?;
        for r in 0..t.len() {
            let vals: Vec<f64> = cols.iter().map(|&c| t.value(r, c)).collect();
            if expr::eval(&x.tree, &vals) > 0.0 {
                excluded_count += 1;
                keep_row[r] = false;
            }
        }
        if !keep_row.iter().any(|&b| b) {
            return Err("--exclude removed every row".to_string());
        }
    }

    // 1. The row filter defines the population the split is drawn from.
    let mut filtered_count = 0usize;
    if let Some(f) = &prep.filter {
        let cols: Vec<usize> = f
            .vars
            .iter()
            .map(|v| t.index_of(v))
            .collect::<Result<_, _>>()?;
        for r in 0..t.len() {
            if !keep_row[r] {
                continue;
            }
            let vals: Vec<f64> = cols.iter().map(|&c| t.value(r, c)).collect();
            if expr::eval(&f.tree, &vals) > 0.0 {
                // kept
            } else {
                keep_row[r] = false;
                filtered_count += 1;
            }
        }
        if !keep_row.iter().any(|&b| b) {
            return Err("--filter kept no rows".to_string());
        }
    }
    // With no `--exclude`/`--filter` the population is the whole table: borrow
    // it rather than copy a second columnar image.
    let filtered_owned;
    let filtered: &Table = if keep_row.iter().all(|&b| b) {
        t
    } else {
        filtered_owned = t.select(&keep_row);
        &filtered_owned
    };

    // Weights follow the rows through the split; validate them on the surviving
    // population (after --exclude / --filter) rather than on discarded rows.
    let weights_raw: Option<Vec<f64>> = match weight_idx {
        Some(j) => {
            let name = weight.unwrap_or("");
            let w: Vec<f64> = filtered.col(j).to_vec();
            validate_weights(name, &w)?;
            Some(w)
        }
        None => None,
    };
    let (mut train_idx, mut fold_idx, mut report) = build_split(filtered, spec, rng)?;
    report.excluded_rows = excluded_count;
    if train_idx.is_empty() {
        return Err("training split is empty (reduce --test-frac)".to_string());
    }

    // #111: carve a chronological validation slice from the **end** of the
    // training period. Selection may use train/validation; the test block is a
    // lockbox and never influences which candidate wins. A purge of `gap` rows
    // stands between fit-train and validation, mirroring the test embargo.
    let mut validation_idx: Vec<usize> = Vec::new();
    if spec.validation_frac > 0.0 {
        if !(spec.validation_frac < 1.0) {
            return Err("--validation-frac must be in (0, 1)".to_string());
        }
        let order: Vec<usize> = if let Some(tc) = spec.time_col {
            let keys = time_keys(filtered, tc);
            let mut v = train_idx.clone();
            v.sort_by(|&a, &b| keys.cmp(a, b).then(a.cmp(&b)));
            v
        } else {
            // Without a time column the file order is the chronological order.
            let mut v = train_idx.clone();
            v.sort_unstable();
            v
        };
        let m = order.len();
        if m >= 2 {
            let want = ((m as f64) * spec.validation_frac).round() as usize;
            let vn = want.clamp(1, m - 1);
            let cut = m - vn;
            let gap = spec.gap.min(cut.saturating_sub(1));
            let fit_cut = cut - gap;
            validation_idx = order[cut..].to_vec();
            train_idx = order[..fit_cut].to_vec();
            report.purged_validation_rows = gap;
            report.notes.push(format!(
                "validation: held out the last {} of {} training row(s){}",
                vn,
                m,
                if gap > 0 {
                    format!(" after a {}-row purge between fit-train and validation", gap)
                } else {
                    String::new()
                }
            ));
        } else {
            report.notes.push(
                "validation slice requested but the training period has fewer than 2 rows; \
                 all training rows are used to fit"
                    .to_string(),
            );
        }
        if train_idx.is_empty() {
            return Err("--validation-frac left no rows to fit on".to_string());
        }
    }

    let n = filtered.len();
    // Used columns: every input, then every target (one target in the
    // single-output case; issue #15 keeps all of them row-aligned here).
    let used_cols: Vec<usize> = var_idx.iter().copied().chain(tgt_idx.iter().copied()).collect();
    let nc = used_cols.len();
    // #133: a full row-major `cells` copy is only needed when a preparation
    // stage mutates a cell (fill / outlier / smoothing). The default path
    // (no fill, no outlier rule, no smoothing) reads the table directly, so the
    // preparation no longer duplicates every used column before sampling.
    // `Drop` only reads cells to set the row mask; it never mutates one, so it
    // must not trigger the row-major copy either.
    let mutating = !matches!(prep.fill, FillMethod::None | FillMethod::Drop)
        || prep.outlier_iqr.is_some_and(|k| k > 0.0)
        || prep.smooth > 1;
    #[cfg(test)]
    if mutating {
        CELL_COPIES.with(|n| n.set(n.get() + 1));
    }
    let mut cells: Option<Vec<Vec<f64>>> = if mutating {
        Some(
            (0..filtered.len())
                .map(|r| used_cols.iter().map(|&c| filtered.value(r, c)).collect())
                .collect(),
        )
    } else {
        None
    };
    // #133: when no stage needs a mutable copy, keep direct slices of the
    // columnar table. Reads then avoid the per-cell `Table::value` bounds and
    // `Option` chain on every row of every column.
    let direct_cols: Vec<&[f64]> = if cells.is_none() {
        used_cols.iter().map(|&c| filtered.col(c)).collect()
    } else {
        Vec::new()
    };
    let cell = |cells: &Option<Vec<Vec<f64>>>, r: usize, k: usize| -> f64 {
        match cells {
            Some(c) => c[r][k],
            None => direct_cols[k][r],
        }
    };
    let mut keep = vec![true; n];
    let mut row_filled = vec![false; n];
    let mut rows_dropped_missing = 0usize;

    // #110: interpolation reads the next bar in each side. On a time-ordered
    // table that is look-ahead, so it needs an explicit opt-in.
    if prep.fill == FillMethod::Interpolate
        && !prep.allow_interpolate
        && (spec.time_col.is_some()
            || matches!(spec.mode, SplitMode::Temporal | SplitMode::TemporalGroup))
    {
        return Err(
            "--fill interpolate reads the next bar, which is look-ahead on time-ordered \
             data; use --fill previous or --fill drop, or pass --allow-interpolate if the \
             rows are not a time series"
                .to_string(),
        );
    }

    // 2. Missing values.
    match prep.fill {
        FillMethod::None => {
            // Leaving a gap in a used column would poison every evaluation: the
            // search would return an empty front and report `inf`/`NaN`
            // baselines while still exiting 0. That reads as "no structure
            // found" when the real answer is "you have gaps". Fail loudly and
            // name the columns and the fix.
            let mut detail: Vec<String> = Vec::new();
            let mut total = 0usize;
            for (k, &c) in used_cols.iter().enumerate() {
                let missing = (0..n).filter(|&r| cell(&cells, r, k).is_nan()).count();
                if missing > 0 {
                    total += missing;
                    let name = t
                        .names
                        .get(c)
                        .cloned()
                        .unwrap_or_else(|| format!("column {}", c));
                    detail.push(format!("{}: {}", name, missing));
                }
            }
            if total > 0 {
                return Err(format!(
                    "{} missing cell(s) in used columns ({}); pass --fill \
                     drop|mean|median|zero|previous|interpolate",
                    total,
                    detail.join(", ")
                ));
            }
        }
        FillMethod::Drop => {
            for r in 0..n {
                if (0..nc).any(|k| cell(&cells, r, k).is_nan()) {
                    keep[r] = false;
                    rows_dropped_missing += 1;
                }
            }
        }
        method => {
            // Training-only summary statistics.
            let stats: Vec<f64> = (0..nc)
                .map(|k| {
                    let mut vals: Vec<f64> = train_idx
                        .iter()
                        .filter_map(|&r| {
                            let v = cell(&cells, r, k);
                            if v.is_finite() {
                                Some(v)
                            } else {
                                None
                            }
                        })
                        .collect();
                    match method {
                        FillMethod::Mean => {
                            if vals.is_empty() {
                                0.0
                            } else {
                                vals.iter().sum::<f64>() / vals.len() as f64
                            }
                        }
                        FillMethod::Median => {
                            vals.sort_by(|a, b| {
                                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
                            });
                            if vals.is_empty() {
                                0.0
                            } else {
                                quantile(&vals, 0.5)
                            }
                        }
                        _ => 0.0,
                    }
                })
                .collect();

            match method {
                FillMethod::Mean | FillMethod::Median | FillMethod::Zero => {
                    let cells = cells.as_mut().expect("fill requires a mutable copy");
                    for r in 0..n {
                        for k in 0..nc {
                            if cells[r][k].is_nan() {
                                cells[r][k] = stats[k];
                                row_filled[r] = true;
                            }
                        }
                    }
                }
                FillMethod::Previous => {
                    let cells = cells.as_mut().expect("fill requires a mutable copy");
                    for side in split_sides(&train_idx, &fold_idx, &validation_idx) {
                        let mut last: Vec<Option<f64>> = vec![None; nc];
                        for &r in &side {
                            for k in 0..nc {
                                if cells[r][k].is_nan() {
                                    if let Some(v) = last[k] {
                                        cells[r][k] = v;
                                        row_filled[r] = true;
                                    }
                                } else {
                                    last[k] = Some(cells[r][k]);
                                }
                            }
                        }
                    }
                }
                FillMethod::Interpolate => {
                    let cells = cells.as_mut().expect("fill requires a mutable copy");
                    for side in split_sides(&train_idx, &fold_idx, &validation_idx) {
                        for k in 0..nc {
                            let mut p = 0usize;
                            while p < side.len() {
                                if !cells[side[p]][k].is_nan() {
                                    p += 1;
                                    continue;
                                }
                                let start = p;
                                while p < side.len() && cells[side[p]][k].is_nan() {
                                    p += 1;
                                }
                                let end = p;
                                if start > 0 && end < side.len() {
                                    let a = cells[side[start - 1]][k];
                                    let b = cells[side[end]][k];
                                    let span = (end - start + 1) as f64;
                                    for (q, &r) in side.iter().enumerate().take(end).skip(start) {
                                        let t = (q - start + 1) as f64 / span;
                                        cells[r][k] = a + (b - a) * t;
                                        row_filled[r] = true;
                                    }
                                }
                            }
                        }
                    }
                }
                FillMethod::None | FillMethod::Drop => {}
            }
        }
    }

    // A requested fill that could not reach an edge (previous at the start of a
    // side, interpolation outside its two neighbours) leaves the row missing;
    // it is dropped rather than silently passed to the search.
    if !matches!(prep.fill, FillMethod::None | FillMethod::Drop) {
        for r in 0..n {
            if keep[r] && (0..nc).any(|k| cell(&cells, r, k).is_nan()) {
                keep[r] = false;
                rows_dropped_missing += 1;
            }
        }
    }
    let rows_filled = row_filled.iter().filter(|&&b| b).count();

    // A boolean input must still be exactly 0/1 after preparation. Fill methods
    // such as mean/median/interpolate can average two truth values into a
    // fraction, which would silently change the semantics of every logical
    // operator (and of `if`), so refuse it and name the column and the fix.
    for j in 0..nv {
        if dtype_at(j) != DType::Boolean {
            continue;
        }
        for r in 0..n {
            if !keep[r] {
                continue;
            }
            let v = cell(&cells, r, j);
            if !(v == 0.0 || v == 1.0) {
                let name = t
                    .names
                    .get(var_idx[j])
                    .cloned()
                    .unwrap_or_else(|| format!("column {}", var_idx[j]));
                return Err(format!(
                    "--fill {} produced value {} in boolean column '{}' (row {}); \
                     use --fill zero|drop|previous so the column stays 0/1",
                    prep.fill.name(),
                    v,
                    name,
                    r + 1
                ));
            }
        }
    }

    // 3. Outlier removal. Bounds come from the training rows only.
    let mut rows_dropped_outlier = 0usize;
    if let Some(k) = prep.outlier_iqr {
        if k > 0.0 {
            let mut bounds = vec![(f64::NEG_INFINITY, f64::INFINITY); nc];
            for (col, bound) in bounds.iter_mut().enumerate() {
                let mut vals: Vec<f64> = train_idx
                    .iter()
                    .filter_map(|&r| {
                        let v = cell(&cells, r, col);
                        if keep[r] && v.is_finite() {
                            Some(v)
                        } else {
                            None
                        }
                    })
                    .collect();
                if vals.len() < 2 {
                    continue;
                }
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                let med = quantile(&vals, 0.5);
                let iqr = quantile(&vals, 0.75) - quantile(&vals, 0.25);
                *bound = (med - k * iqr, med + k * iqr);
            }
            for r in 0..n {
                if !keep[r] {
                    continue;
                }
                let out = (0..nc).any(|c| {
                    let v = cell(&cells, r, c);
                    v.is_finite() && (v < bounds[c].0 || v > bounds[c].1)
                });
                if out {
                    keep[r] = false;
                    rows_dropped_outlier += 1;
                }
            }
        }
    }

    // 4. Smoothing, one side at a time so a training window cannot read a test
    // value (and vice versa).
    let mut smoothed = 0usize;
    if prep.smooth > 1 {
        for side in split_sides(&train_idx, &fold_idx, &validation_idx) {
            let side: Vec<usize> = side.into_iter().filter(|&r| keep[r]).collect();
            for col in 0..nc {
                // Smoothing a 0/1 column would create fractions and break the
                // truthiness contract; it applies to continuous inputs and the
                // target only.
                if col < nv && dtype_at(col) == DType::Boolean {
                    continue;
                }
                let cells_mut = cells.as_mut().expect("smoothing requires a mutable copy");
                smoothed += smooth_cells(
                    cells_mut,
                    &side,
                    col,
                    prep.smooth,
                    prep.smooth_centred,
                );
            }
        }
    }

    // Rows dropped by preparation leave both the training and test sets.
    train_idx.retain(|&i| keep[i]);
    validation_idx.retain(|&i| keep[i]);
    for f in fold_idx.iter_mut() {
        f.retain(|&i| keep[i]);
    }
    fold_idx.retain(|f| !f.is_empty());
    if train_idx.is_empty() {
        return Err("preparation removed every training row".to_string());
    }
    report.train_rows = train_idx.len();
    report.validation_rows = validation_idx.len();
    report.test_rows = fold_idx.iter().map(|f| f.len()).sum();
    report.folds = fold_idx.len().max(1);

    // `cells` is the single prepared copy: inputs in `0..nv`, targets after.
    // The samples below normalise out of it directly instead of keeping another
    // row-major copy (the old `x_vals`/`y_vals`).

    // 5. Centre/scale from the surviving training rows, applied to both sides.
    // The default (mean/std) is the same arithmetic as before issue #2.
    let mut means = vec![0.0; nv];
    let mut stds = vec![1.0; nv];
    for j in 0..nv {
        // Boolean inputs pass through unchanged: mean 0 / std 1 are the affine
        // identity, so `make` below hands the evaluator exactly the 0/1 from the
        // file and a denormalised formula is still in original units.
        if dtype_at(j) == DType::Boolean {
            means[j] = 0.0;
            stds[j] = 1.0;
            continue;
        }
        // #133: sum in the same training-row order without materialising a
        // per-column copy. The arithmetic (two passes, mean then squared
        // deviations) is unchanged, so selected rows and scores stay identical.
        let mut sum = 0.0f64;
        for &i in &train_idx {
            sum += cell(&cells, i, j);
        }
        let m = sum / train_idx.len() as f64;
        let mut var = 0.0f64;
        for &i in &train_idx {
            let d = cell(&cells, i, j) - m;
            var += d.powi(2);
        }
        var /= train_idx.len() as f64;
        let s = var.sqrt();
        // #133: median/min/max/IQR/range are the only modes that read an
        // ordered copy. The default mean/std path never clones or sorts.
        let need_sorted = !matches!(prep.center, Center::Mean | Center::None)
            || !matches!(prep.scale, Scale::Std | Scale::None);
        let sorted = if need_sorted {
            let mut sorted: Vec<f64> = train_idx.iter().map(|&i| cell(&cells, i, j)).collect();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            sorted
        } else {
            Vec::new()
        };
        means[j] = match prep.center {
            Center::Mean => m,
            Center::Median => quantile(&sorted, 0.5),
            Center::Min => sorted[0],
            Center::Max => sorted[sorted.len() - 1],
            Center::None => 0.0,
        };
        let factor = match prep.scale {
            Scale::Std => s,
            Scale::Iqr => quantile(&sorted, 0.75) - quantile(&sorted, 0.25),
            Scale::Range => sorted[sorted.len() - 1] - sorted[0],
            Scale::None => 1.0,
        };
        stds[j] = if !(factor.abs() >= 1e-12) { 1.0 } else { factor };
    }

    let make = |idx: &[usize]| -> Sample {
        let mut s = Sample::default();
        s.x.reserve(idx.len());
        s.y.reserve(idx.len());
        if cells.is_none() {
            // Default path: allocate the rows once, then fill them in
            // row-blocks. Reading 53 columns row-by-row streams from 53
            // distant cache lines; a 256-row block makes each column read
            // sequential and keeps the written rows in L2.
            const ROW_BLOCK: usize = 256;
            s.x = (0..idx.len())
                .map(|_| vec![0.0; nv])
                .collect();
            for rb in (0..idx.len()).step_by(ROW_BLOCK) {
                let re = (rb + ROW_BLOCK).min(idx.len());
                for j in 0..nv {
                    let src = direct_cols[j];
                    let m = means[j];
                    let sd = stds[j];
                    for r in rb..re {
                        s.x[r][j] = (src[idx[r]] - m) / sd;
                    }
                }
            }
            s.y = idx.iter().map(|&i| direct_cols[nv][i]).collect();
            if nt > 1 {
                s.ys = (0..nt)
                    .map(|j| idx.iter().map(|&i| direct_cols[nv + j][i]).collect())
                    .collect();
            }
        } else {
            for &i in idx {
                let mut row = Vec::with_capacity(nv);
                for j in 0..nv {
                    row.push((cell(&cells, i, j) - means[j]) / stds[j]);
                }
                s.x.push(row);
                s.y.push(cell(&cells, i, nv));
            }
            if nt > 1 {
                s.ys = (0..nt)
                    .map(|j| idx.iter().map(|&i| cell(&cells, i, nv + j)).collect())
                    .collect();
            }
        }
        if let Some(w) = &weights_raw {
            s.w = idx.iter().map(|&i| w[i]).collect();
        }
        s
    };

    let train = make(&train_idx);
    let validation = if validation_idx.is_empty() {
        None
    } else {
        Some(make(&validation_idx))
    };
    if let (Some(w), Some(name)) = (&weights_raw, weight) {
        let positive = train_idx.iter().filter(|&&i| w[i] > 0.0).count();
        if positive == 0 {
            return Err(format!(
                "weight column '{}' has no positive weight in the training rows after preparation",
                name
            ));
        }
    }
    // #133: build the pooled test block once from the concatenated fold
    // indices. The old path built every fold sample and then cloned all of its
    // rows into `test`, doubling the test-side allocations. With a single fold
    // (the default) `fold_samples` stays empty and every consumer falls back to
    // `test`; with several folds the per-fold samples are still materialised.
    let test_idx: Vec<usize> = fold_idx.iter().flatten().copied().collect();
    let test = make(&test_idx);
    let fold_samples: Vec<Sample> = if fold_idx.len() > 1 {
        fold_idx.iter().map(|f| make(f)).collect()
    } else {
        Vec::new()
    };

    let hazards = time_hazards(t, spec);
    Ok(Prepared {
        train,
        validation,
        test,
        fold_samples,
        means,
        stds,
        report,
        prep: PrepReport {
            fill: prep.fill.name(),
            rows_filled,
            rows_dropped_missing,
            rows_dropped_outlier,
            smoothed,
            center: prep.center.name(),
            scale: prep.scale.name(),
            filtered: filtered_count,
            rows_skipped_malformed: t.skipped_malformed,
            smooth_mode: if prep.smooth_centred { "centred" } else { "trailing" },
            time_unit: hazards.unit,
            time_rows: hazards.rows,
            time_duplicates: hazards.duplicates,
            time_non_monotonic: hazards.non_monotonic,
            time_converted: hazards.converted,
            memory_estimate: None,
        },
    })
}

/// Mark columns numeric when they contain at least one number and every other
/// non-missing cell is either a number or a junk token. Junk tokens become
/// missing. Used only when `--fill` was given explicitly, so the default
/// loader keeps rejecting typo-ridden columns.
fn coerce_numeric(t: &Table) -> Table {
    let mut out = t.clone();
    for j in 0..t.names.len() {
        if out.numeric[j] {
            continue;
        }
        let mut any_number = false;
        for r in 0..t.len() {
            let cell = t.cell_text(r, j);
            let cell = cell.trim();
            if !is_missing_token(cell) && cell.parse::<f64>().is_ok() {
                any_number = true;
                break;
            }
        }
        if !any_number {
            continue;
        }
        out.numeric[j] = true;
        let values = &mut out.columns[j].values;
        values.reserve(t.len());
        for r in 0..t.len() {
            let cell = t.cell_text(r, j);
            let cell = cell.trim();
            values.push(if is_missing_token(cell) {
                f64::NAN
            } else {
                cell.parse::<f64>().unwrap_or(f64::NAN)
            });
        }
    }
    out
}

/// Linear-interpolation quantile (Hyndman–Fan type 7), on sorted values.
fn quantile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let h = (sorted.len() - 1) as f64 * p.clamp(0.0, 1.0);
    let lo = h.floor() as usize;
    let hi = h.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        sorted[lo] + (h - lo as f64) * (sorted[hi] - sorted[lo])
    }
}

/// The training and test rows in table order, for operations that walk a
/// column sequentially. Working per side is what keeps them leak-free.
fn split_sides(
    train: &[usize],
    folds: &[Vec<usize>],
    validation: &[usize],
) -> Vec<Vec<usize>> {
    let mut sides = Vec::new();
    let mut tr = train.to_vec();
    tr.sort_unstable();
    sides.push(tr);
    // Validation is its own side: a forward fill or smoother must not read
    // across the fit-train/validation boundary either (#111).
    let mut va = validation.to_vec();
    va.sort_unstable();
    if !va.is_empty() {
        sides.push(va);
    }
    let mut te: Vec<usize> = folds.iter().flatten().copied().collect();
    te.sort_unstable();
    te.dedup();
    sides.push(te);
    sides
}

/// Moving average over a `k`-wide window, shrinking at the edges of `side`.
///
/// #110: the default window is **trailing** (`p-k+1 ..= p`), so a row only
/// reads its own and earlier bars. `centred` keeps the old `p-k/2 ..= p+k/2`
/// behaviour and therefore reads future bars; it is an explicit opt-in.
/// Missing cells are skipped; a cell with no observed neighbours is left
/// missing. Returns the number of cells replaced.
fn smooth_cells(
    cells: &mut [Vec<f64>],
    side: &[usize],
    col: usize,
    k: usize,
    centred: bool,
) -> usize {
    if side.is_empty() || k < 2 {
        return 0;
    }
    let before = if centred { k / 2 } else { k - 1 };
    let after = if centred { k - 1 - k / 2 } else { 0 };
    let orig: Vec<f64> = side.iter().map(|&r| cells[r][col]).collect();
    let mut count = 0usize;
    for p in 0..side.len() {
        let a = p.saturating_sub(before);
        let b = (p + after + 1).min(side.len());
        let mut sum = 0.0;
        let mut used = 0usize;
        for v in &orig[a..b] {
            if v.is_finite() {
                sum += *v;
                used += 1;
            }
        }
        if used > 0 {
            cells[side[p]][col] = sum / used as f64;
            count += 1;
        }
    }
    count
}

/// Cross-validation scheme (see `docs/overfitting.md`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CvMode {
    /// A shuffled partition into `k` equal parts: i.i.d. data.
    KFold,
    /// Forward chaining: an expanding training window, the next contiguous
    /// block held out. The honest scheme for time-ordered data.
    Rolling,
    /// `k` contiguous parts with `gap` rows embargoed either side of each test
    /// block, so leaking neighbours (autocorrelation, look-ahead) cannot help.
    Purged,
}

impl CvMode {
    pub fn parse(s: &str) -> Result<CvMode, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "k-fold" | "kfold" | "k_fold" | "fold" | "k" => Ok(CvMode::KFold),
            "rolling" | "roll" | "forward" | "expanding" => Ok(CvMode::Rolling),
            "purged" | "purge" | "embargo" => Ok(CvMode::Purged),
            other => Err(format!(
                "unknown --cv mode '{}' (expected k-fold, rolling or purged)",
                other
            )),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            CvMode::KFold => "k-fold",
            CvMode::Rolling => "rolling",
            CvMode::Purged => "purged",
        }
    }
}

/// One cross-validation fold, normalised on its **own** training rows.
pub struct CvFold {
    pub train: Sample,
    pub test: Sample,
    /// Training-only normalisation statistics, so a fitted model can be
    /// denormalised back to the original units for display.
    pub means: Vec<f64>,
    pub stds: Vec<f64>,
}

/// Build `k` cross-validation folds from the raw table.
///
/// Normalisation statistics are computed from each fold's training rows only,
/// so nothing leaks from the held-out block. Folds that would be empty are
/// dropped; an error is returned if no usable fold remains.
pub fn cv_folds(
    t: &Table,
    vars: &[String],
    target: &str,
    mode: CvMode,
    k: usize,
    gap: usize,
    seed: u32,
) -> Result<Vec<CvFold>, String> {
    cv_folds_weighted(t, vars, target, None, mode, k, gap, seed)
}

/// Like [`cv_folds`], with an optional per-row weight column carried into each
/// fold's training and test samples. The column is validated once on the whole
/// table with the same rules as [`prepare_full_weighted_with`].
pub fn cv_folds_weighted(
    t: &Table,
    vars: &[String],
    target: &str,
    weight: Option<&str>,
    mode: CvMode,
    k: usize,
    gap: usize,
    seed: u32,
) -> Result<Vec<CvFold>, String> {
    cv_folds_typed(t, vars, target, weight, mode, k, gap, seed, &[])
}

/// Like [`cv_folds_weighted`], with the per-input datatypes produced by
/// [`apply_dtypes`]. A `Boolean` input is never centred or scaled, exactly as in
/// [`prepare_full_weighted_with`]; an empty slice means every input is
/// continuous.
#[allow(clippy::too_many_arguments)]
pub fn cv_folds_typed(
    t: &Table,
    vars: &[String],
    target: &str,
    weight: Option<&str>,
    mode: CvMode,
    k: usize,
    gap: usize,
    seed: u32,
    dtypes: &[DType],
) -> Result<Vec<CvFold>, String> {
    let (var_idx, tgt_idx) = resolve_columns(t, vars, target)?;
    let nv = var_idx.len();
    if !dtypes.is_empty() && dtypes.len() != nv {
        return Err(format!(
            "internal datatype plan has {} entries for {} input variables",
            dtypes.len(),
            nv
        ));
    }
    let dtype_at = |j: usize| dtypes.get(j).copied().unwrap_or(DType::Continuous);
    let weight_idx = resolve_weight_column(t, weight)?;
    let weights_raw: Option<Vec<f64>> = match weight_idx {
        Some(j) => {
            let name = weight.unwrap_or("");
            let w: Vec<f64> = t.col(j).to_vec();
            validate_weights(name, &w)?;
            Some(w)
        }
        None => None,
    };
    let n = t.len();
    let k = k.max(2);
    if n < k + 1 {
        return Err(format!(
            "cross-validation needs more than {} rows, got {}",
            k, n
        ));
    }
    let x_raw: Vec<Vec<f64>> = (0..n)
        .map(|r| var_idx.iter().map(|&i| t.value(r, i)).collect())
        .collect();
    let y: Vec<f64> = t.col(tgt_idx).to_vec();

    // (train indices, test indices) per fold.
    let splits: Vec<(Vec<usize>, Vec<usize>)> = match mode {
        CvMode::KFold => {
            let mut order: Vec<usize> = (0..n).collect();
            let mut rng = Mt::new(seed);
            for i in (1..n).rev() {
                let j = rng.usize_below(i + 1);
                order.swap(i, j);
            }
            (0..k)
                .map(|f| {
                    let lo = f * n / k;
                    let hi = (f + 1) * n / k;
                    let test = order[lo..hi].to_vec();
                    let mut train = order[..lo].to_vec();
                    train.extend_from_slice(&order[hi..]);
                    (train, test)
                })
                .collect()
        }
        CvMode::Purged => (0..k)
            .map(|f| {
                let lo = f * n / k;
                let hi = (f + 1) * n / k;
                let test: Vec<usize> = (lo..hi).collect();
                // Embargo `gap` rows on both sides of the test block.
                let a = lo.saturating_sub(gap);
                let b = (hi + gap).min(n);
                let train: Vec<usize> = (0..n).filter(|&i| i < a || i >= b).collect();
                (train, test)
            })
            .collect(),
        CvMode::Rolling => (0..k)
            .map(|f| {
                // `k + 1` contiguous blocks; fold `f` trains on 0..=f and tests
                // on f+1 (an expanding window).
                let train_end = (f + 1) * n / (k + 1);
                let test_end = (f + 2) * n / (k + 1);
                ((0..train_end).collect(), (train_end..test_end).collect())
            })
            .collect(),
    };

    let nv = var_idx.len();
    let mut folds = Vec::new();
    for (train_idx, test_idx) in splits {
        if train_idx.is_empty() || test_idx.is_empty() {
            continue;
        }
        let mut means = vec![0.0; nv];
        let mut stds = vec![1.0; nv];
        for j in 0..nv {
            if dtype_at(j) == DType::Boolean {
                means[j] = 0.0;
                stds[j] = 1.0;
                continue;
            }
            let m = train_idx.iter().map(|&i| x_raw[i][j]).sum::<f64>() / train_idx.len() as f64;
            let var = train_idx
                .iter()
                .map(|&i| (x_raw[i][j] - m).powi(2))
                .sum::<f64>()
                / train_idx.len() as f64;
            let s = var.sqrt();
            means[j] = m;
            stds[j] = if s < 1e-12 { 1.0 } else { s };
        }
        if let (Some(w), Some(name)) = (&weights_raw, weight) {
            if train_idx.iter().all(|&i| w[i] <= 0.0) {
                return Err(format!(
                    "weight column '{}' has no positive weight in a cross-validation training fold",
                    name
                ));
            }
        }
        let make = |idx: &[usize]| -> Sample {
            let mut s = Sample::default();
            for &i in idx {
                s.x.push(
                    (0..nv)
                        .map(|j| (x_raw[i][j] - means[j]) / stds[j])
                        .collect(),
                );
                s.y.push(y[i]);
            }
            if let Some(w) = &weights_raw {
                s.w = idx.iter().map(|&i| w[i]).collect();
            }
            s
        };
        folds.push(CvFold {
            train: make(&train_idx),
            test: make(&test_idx),
            means,
            stds,
        });
    }
    if folds.is_empty() {
        return Err("cross-validation produced no usable folds (data too small)".to_string());
    }
    Ok(folds)
}

/// A compact description of the selectors a split is stored as (#70).
fn describe_selector(n: usize, train: &[usize], test: &[usize]) -> String {
    let train_ids: Vec<u32> = train.iter().map(|&i| i as u32).collect();
    let test_ids: Vec<u32> = test.iter().map(|&i| i as u32).collect();
    let tr = RowSelector::from_rows(n, &train_ids);
    let te = RowSelector::from_rows(n, &test_ids);
    format!(
        "train {} | test {}",
        tr.describe(n),
        te.describe(n)
    )
}

/// Decide where each row goes. Returns `(train indices, test fold indices, report)`.
fn build_split(
    t: &Table,
    spec: &SplitSpec,
    rng: &mut Mt,
) -> Result<(Vec<usize>, Vec<Vec<usize>>, SplitReport), String> {
    let n = t.len();
    let folds = spec.folds.max(1);
    let mut notes: Vec<String> = Vec::new();
    let mut report = SplitReport {
        mode: spec.mode.name(),
        rows_total: n,
        folds,
        gap: spec.gap,
        ..Default::default()
    };

    // Explicit row selection (issue #3). The expressions assign rows to a side;
    // because they say exactly which rows belong where, they replace the
    // positional modes rather than competing with them.
    if spec.train_select.is_some() || spec.test_select.is_some() {
        let evaluate = |f: &Filter, t: &Table| -> Result<Vec<bool>, String> {
            let cols: Vec<usize> = f
                .vars
                .iter()
                .map(|v| t.index_of(v))
                .collect::<Result<_, _>>()?;
            Ok((0..n)
                .map(|r| {
                    let vals: Vec<f64> = cols.iter().map(|&c| t.value(r, c)).collect();
                    expr::eval(&f.tree, &vals) > 0.0
                })
                .collect())
        };
        let tr_mask = spec
            .train_select
            .as_ref()
            .map(|f| evaluate(f, t))
            .transpose()?;
        let te_mask = spec
            .test_select
            .as_ref()
            .map(|f| evaluate(f, t))
            .transpose()?;

        let mut train = Vec::new();
        let mut test = Vec::new();
        let mut overlap = 0usize;
        let mut unused = 0usize;
        for r in 0..n {
            let want_train = tr_mask.as_ref().map(|m| m[r]);
            let want_test = te_mask.as_ref().map(|m| m[r]);
            let (to_train, to_test) = match (want_train, want_test) {
                (Some(a), Some(b)) => match (a, b) {
                    // Both sides claim the row: training wins, and the overlap
                    // is reported rather than silently resolved.
                    (true, true) => {
                        overlap += 1;
                        (true, false)
                    }
                    (true, false) => (true, false),
                    (false, true) => (false, true),
                    (false, false) => {
                        unused += 1;
                        (false, false)
                    }
                },
                // Only one side was given: the other side is the complement.
                (Some(a), None) => (a, !a),
                (None, Some(b)) => (!b, b),
                (None, None) => unreachable!(),
            };
            if to_train {
                train.push(r);
            }
            if to_test {
                test.push(r);
            }
        }

        report.mode = "select";
        report.overlap_rows = overlap;
        report.unassigned_rows = unused;
        notes.push(format!(
            "row selection assigned {} train / {} test row(s)",
            train.len(),
            test.len()
        ));
        if overlap > 0 {
            notes.push(format!(
                "{} row(s) matched both selections; they were assigned to training",
                overlap
            ));
        }
        if unused > 0 {
            notes.push(format!(
                "{} row(s) matched neither selection and were left unused",
                unused
            ));
        }
        notes.push(format!(
            "row selection replaced the {} split",
            spec.mode.name()
        ));
        if spec.gap > 0 {
            notes.push("the embargo gap does not apply to a row selection".to_string());
        }
        report.notes = notes;
        report.selector = describe_selector(n, &train, &test);
        return Ok((train, split_blocks(&test, folds), report));
    }

    let (train, fold_idx) = match spec.mode {
        SplitMode::Random => {
            let mut idx: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = rng.usize_below(i + 1);
                idx.swap(i, j);
            }
            let test_n = ((n as f64) * spec.test_frac).round() as usize;
            let test_n = test_n.clamp(if spec.test_frac > 0.0 { 1 } else { 0 }, n.saturating_sub(1));
            let test = idx[..test_n].to_vec();
            let train = idx[test_n..].to_vec();
            (train, split_blocks(&test, folds))
        }
        SplitMode::Temporal => {
            let time_col = spec
                .time_col
                .ok_or("temporal split needs a time column (--time <col>)")?;
            let (keys_f64, trep) = time_keys_with(t, time_col, &spec.time)?;
            let keys = TimeKeys::F64(keys_f64);
            report.notes.extend(trep.text_lines());
            let order = sorted_by(&keys);
            match split_ordered(&order, spec, folds) {
                Some((tr, folds_idx)) => {
                    report.train_time_max = order
                        .get(tr.len().saturating_sub(1))
                        .map(|&i| keys.value(i));
                    report.test_time_min = folds_idx
                        .iter()
                        .flatten()
                        .map(|&i| keys.value(i))
                        .fold(None, |acc: Option<f64>, v| {
                            Some(acc.map_or(v, |a| a.min(v)))
                        });
                    (tr, folds_idx)
                }
                None => {
                    notes.push(format!(
                        "too few rows for a temporal split{}; all rows used for training",
                        if spec.gap > 0 {
                            " with this gap"
                        } else {
                            ""
                        }
                    ));
                    (order, Vec::new())
                }
            }
        }
        SplitMode::Group => {
            let group_col = spec
                .group_col
                .ok_or("group split needs a group column (--group-by <col>)")?;
            let (ids, names) = group_ids(t, group_col);
            report.groups_total = names.len();
            // Deterministic, balanced assignment: order groups by a hash of the
            // name, then move whole groups to the test side until it is big
            // enough. No group ever appears on both sides.
            let mut order: Vec<usize> = (0..names.len()).collect();
            let hashes: Vec<u64> = names.iter().map(|s| hash_str(s, 0x5eed_5eed)).collect();
            order.sort_by_key(|&g| hashes[g]);
            let target_test = ((n as f64) * spec.test_frac).round() as usize;
            let mut is_test = vec![false; names.len()];
            let mut acc = 0usize;
            let counts: Vec<usize> = (0..names.len())
                .map(|g| ids.iter().filter(|&&i| i == g).count())
                .collect();
            for &g in &order {
                if acc >= target_test {
                    break;
                }
                is_test[g] = true;
                acc += counts[g];
            }
            if is_test.iter().all(|&b| b) {
                if let Some(&g) = order.first() {
                    is_test[g] = false;
                    notes.push("only one group; kept it in training".to_string());
                }
            }
            if is_test.iter().all(|&b| !b) {
                if let Some(&g) = order.last() {
                    is_test[g] = true;
                    notes.push("kept at least one group for testing".to_string());
                }
            }
            let train: Vec<usize> = (0..n).filter(|&i| !is_test[ids[i]]).collect();
            let test_groups: Vec<usize> = order.iter().copied().filter(|&g| is_test[g]).collect();
            let ids = &ids;
            let test: Vec<usize> = test_groups
                .iter()
                .flat_map(|&g| (0..n).filter(move |&i| ids[i] == g))
                .collect();
            report.groups_train = names.len() - test_groups.len();
            report.groups_test = test_groups.len();
            (train, split_blocks(&test, folds))
        }
        SplitMode::TemporalGroup => {
            let time_col = spec
                .time_col
                .ok_or("temporal-group split needs a time column (--time <col>)")?;
            let group_col = spec
                .group_col
                .ok_or("temporal-group split needs a group column (--group-by <col>)")?;
            let (keys_f64, trep) = time_keys_with(t, time_col, &spec.time)?;
            let keys = TimeKeys::F64(keys_f64);
            report.notes.extend(trep.text_lines());
            let (ids, names) = group_ids(t, group_col);
            report.groups_total = names.len();
            let mut train: Vec<usize> = Vec::new();
            let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); folds];
            let mut groups_train = 0usize;
            for g in 0..names.len() {
                let mut rows: Vec<usize> = (0..n).filter(|&i| ids[i] == g).collect();
                rows.sort_by(|&a, &b| {
                    keys.cmp(a, b).then(a.cmp(&b))
                });
                match split_ordered(&rows, spec, folds) {
                    Some((tr, folds_idx)) => {
                        if !tr.is_empty() {
                            groups_train += 1;
                        }
                        train.extend(tr);
                        for (k, blk) in folds_idx.into_iter().enumerate() {
                            if k < blocks.len() {
                                blocks[k].extend(blk);
                            }
                        }
                    }
                    None => {
                        notes.push(format!(
                            "group '{}' has too few rows for a temporal split; all of it is in training",
                            names[g]
                        ));
                        train.extend(rows);
                        groups_train += 1;
                    }
                }
            }
            let blocks: Vec<Vec<usize>> = blocks.into_iter().filter(|b| !b.is_empty()).collect();
            report.groups_train = groups_train;
            report.groups_test = names.len() - groups_train;
            report.folds = blocks.len().max(1);
            (train, blocks)
        }
    };

    report.train_rows = train.len();
    report.test_rows = fold_idx.iter().map(|f| f.len()).sum();
    report.folds = fold_idx.len();
    report.notes = notes;
    let all_folds: Vec<usize> = fold_idx.iter().flatten().copied().collect();
    report.selector = describe_selector(n, &train, &all_folds);
    Ok((train, fold_idx, report))
}

/// Split one time-ordered sequence into a training prefix and `folds`
/// contiguous test blocks, leaving `gap` rows as an embargo after the cut.
fn split_ordered(order: &[usize], spec: &SplitSpec, folds: usize) -> Option<(Vec<usize>, Vec<Vec<usize>>)> {
    let m = order.len();
    if m < 2 {
        return None;
    }
    let test_n = ((m as f64) * spec.test_frac).round() as usize;
    let test_n = test_n.clamp(1, m - 1);
    let train_n = m - test_n;
    let from = (train_n + spec.gap).min(m - 1);
    let tail = &order[from..];
    if tail.is_empty() {
        return None;
    }
    Some((order[..train_n].to_vec(), split_blocks(tail, folds)))
}

/// Chop a sequence into `folds` contiguous, near-equal, non-empty blocks.
fn split_blocks(idx: &[usize], folds: usize) -> Vec<Vec<usize>> {
    let folds = folds.max(1).min(idx.len().max(1));
    let mut out = Vec::with_capacity(folds);
    for k in 0..folds {
        let a = idx.len() * k / folds;
        let b = idx.len() * (k + 1) / folds;
        if b > a {
            out.push(idx[a..b].to_vec());
        }
    }
    if out.is_empty() {
        out.push(idx.to_vec());
    }
    out
}

/// Comparable ordering keys for a time column: exact `i64` when the table
/// carries one (nanosecond-safe), otherwise the historical `f64` seconds.
#[derive(Clone, Debug)]
pub enum TimeKeys {
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl TimeKeys {
    pub fn len(&self) -> usize {
        match self {
            TimeKeys::I64(v) => v.len(),
            TimeKeys::F64(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Total order with a row-stable tie break applied by the caller. A gap
    /// (`i64::MIN`) sorts last, matching the old `NaN -> +inf` rule.
    fn cmp(&self, a: usize, b: usize) -> std::cmp::Ordering {
        match self {
            TimeKeys::I64(v) => {
                let key = |i: i64| if i == i64::MIN { i64::MAX } else { i };
                key(v[a]).cmp(&key(v[b]))
            }
            TimeKeys::F64(v) => v[a]
                .partial_cmp(&v[b])
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    }

    /// Display form for split reports (seconds or raw ticks, whichever unit the
    /// source used). Precision beyond binary64 is not claimed here.
    pub fn value(&self, i: usize) -> f64 {
        match self {
            TimeKeys::I64(v) => {
                if v[i] == i64::MIN {
                    f64::INFINITY
                } else {
                    v[i] as f64
                }
            }
            TimeKeys::F64(v) => v[i],
        }
    }

    /// Materialise the ordering key as f64 for consumers that only need a
    /// monotone time axis (values beyond 2^53 are approximate, as documented).
    pub fn to_f64(&self) -> Vec<f64> {
        (0..self.len()).map(|i| self.value(i)).collect()
    }

    fn difference(&self, a: usize, b: usize) -> f64 {
        match self {
            TimeKeys::I64(v) => {
                if v[a] == i64::MIN || v[b] == i64::MIN {
                    f64::INFINITY
                } else {
                    v[b].saturating_sub(v[a]) as f64
                }
            }
            TimeKeys::F64(v) => v[b] - v[a],
        }
    }
}

fn sorted_by(keys: &TimeKeys) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| keys.cmp(a, b).then(a.cmp(&b)));
    order
}

/// Comparable ordering key for every row of a time column.
///
/// Integers are used exactly when the table has an `i64` view (a time-like
/// column parsed from integral text or loaded from an i64 store block), so
/// two nanosecond prints 1 ns apart stay distinct. Otherwise numbers and dates
/// are parsed to seconds; if any value cannot be parsed the whole column falls
/// back to lexicographic order, which is still a valid total order (and the
/// right one for zero-padded `YYYY-MM-DD`).
pub fn time_keys(t: &Table, col: usize) -> TimeKeys {
    let n = t.len();
    if let Some(ints) = t.ints(col) {
        return TimeKeys::I64(ints.to_vec());
    }
    // A numeric column without gaps is already in seconds: parsing its text
    // would only round-trip the same `f64`.
    if t.numeric[col] && !t.col(col).iter().any(|v| v.is_nan()) {
        return TimeKeys::F64(t.col(col).to_vec());
    }
    if let Some(d) = t.dict(col) {
        let parsed: Option<Vec<f64>> = (0..n).map(|r| parse_time(d.text(r))).collect();
        if let Some(v) = parsed {
            return TimeKeys::F64(v);
        }
        let mut pairs: Vec<(String, usize)> = (0..n)
            .map(|r| (d.text(r).trim().to_string(), r))
            .collect();
        pairs.sort();
        let mut key = vec![0.0; n];
        for (rank, (_s, r)) in pairs.into_iter().enumerate() {
            key[r] = rank as f64;
        }
        return TimeKeys::F64(key);
    }
    // High-cardinality numeric column with missing cells: compare the values
    // directly and sort gaps last.
    TimeKeys::F64(
        t.col(col)
            .iter()
            .map(|&v| if v.is_nan() { f64::INFINITY } else { v })
            .collect(),
    )
}

// ---------------------------------------------------------------------------
// #130: strict time parsing
// ---------------------------------------------------------------------------

/// What to do with a date-only value (`2024-01-02`, no clock).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DateOnlyPolicy {
    /// Refuse the file: the instant is under-specified. The default.
    Error,
    /// The value means 00:00:00 UTC.
    UtcMidnight,
    /// The value means the session close, 21:00 UTC (16:00 New York, EST).
    SessionClose,
}

impl DateOnlyPolicy {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "error" => Ok(DateOnlyPolicy::Error),
            "utc-midnight" => Ok(DateOnlyPolicy::UtcMidnight),
            "session-close" => Ok(DateOnlyPolicy::SessionClose),
            other => Err(format!(
                "unknown --date-only policy '{}' (expected error, utc-midnight or session-close)",
                other
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            DateOnlyPolicy::Error => "error",
            DateOnlyPolicy::UtcMidnight => "utc-midnight",
            DateOnlyPolicy::SessionClose => "session-close",
        }
    }
}

/// Unit of a numeric time column.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeUnit {
    Seconds,
    Millis,
    Micros,
    Nanos,
}

impl TimeUnit {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "s" | "sec" | "seconds" => Ok(TimeUnit::Seconds),
            "ms" | "millis" | "milliseconds" => Ok(TimeUnit::Millis),
            "us" | "µs" | "micros" | "microseconds" => Ok(TimeUnit::Micros),
            "ns" | "nanos" | "nanoseconds" => Ok(TimeUnit::Nanos),
            other => Err(format!(
                "unknown --time-unit '{}' (expected s, ms, us or ns)",
                other
            )),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            TimeUnit::Seconds => "s",
            TimeUnit::Millis => "ms",
            TimeUnit::Micros => "us",
            TimeUnit::Nanos => "ns",
        }
    }
    /// Multiplier from this unit to seconds.
    fn to_seconds(self) -> f64 {
        match self {
            TimeUnit::Seconds => 1.0,
            TimeUnit::Millis => 1e-3,
            TimeUnit::Micros => 1e-6,
            TimeUnit::Nanos => 1e-9,
        }
    }
}

/// How to interpret a time column. The defaults are deliberately strict:
/// date-only values are an error and bare clock times are reported as UTC.
#[derive(Clone, Copy, Debug)]
pub struct TimeSpec {
    pub date_only: DateOnlyPolicy,
    /// Explicit unit for numeric columns; `None` auto-detects.
    pub unit: Option<TimeUnit>,
    /// Fixed UTC offset in seconds for timestamps without their own offset.
    /// `None` means UTC (and is reported, never silent).
    pub tz: Option<i64>,
}

impl Default for TimeSpec {
    fn default() -> Self {
        TimeSpec {
            date_only: DateOnlyPolicy::Error,
            unit: None,
            tz: None,
        }
    }
}

impl TimeSpec {
    /// Parse `--tz`: `UTC`/`Z` or a fixed offset such as `+02:00`, `-0500`.
    ///
    /// Named zones need a tz database (and DST rules); std has none, so they
    /// are refused with a suggestion rather than guessed.
    pub fn parse_tz(s: &str) -> Result<i64, String> {
        let t = s.trim();
        if t.eq_ignore_ascii_case("utc") || t.eq_ignore_ascii_case("z") || t == "+00:00" {
            return Ok(0);
        }
        let (sign, rest) = match t.as_bytes().first() {
            Some(b'+') => (1i64, &t[1..]),
            Some(b'-') => (-1i64, &t[1..]),
            _ => {
                return Err(format!(
                    "--tz '{}': named time zones need a tz database; pass a fixed offset \
                     such as --tz -05:00 (or UTC)",
                    s
                ))
            }
        };
        let digits: String = rest.chars().filter(|c| *c != ':').collect();
        if digits.len() != 4 || !digits.chars().all(|c| c.is_ascii_digit()) {
            return Err(format!(
                "--tz '{}': expected a fixed offset such as -05:00 or +0530",
                s
            ));
        }
        let hh: i64 = digits[0..2].parse().unwrap_or(99);
        let mm: i64 = digits[2..4].parse().unwrap_or(99);
        if hh > 23 || mm > 59 {
            return Err(format!("--tz '{}': offset out of range", s));
        }
        Ok(sign * (hh * 3600 + mm * 60))
    }
}

/// What a time column contained and how it was converted, for the preparation
/// block and the split notes.
#[derive(Clone, Debug, Default)]
pub struct TimeReport {
    pub format: String,
    pub unit: String,
    pub date_only: String,
    pub rows: usize,
    /// Rows that carried their own UTC offset.
    pub offset_rows: usize,
    /// Numeric rows rescaled from ms/us/ns to seconds.
    pub scaled_rows: usize,
    pub notes: Vec<String>,
}

impl TimeReport {
    pub fn text_lines(&self) -> Vec<String> {
        let mut out = vec![format!(
            "# time: format={} unit={} date-only={} offset_rows={} scaled_rows={}",
            self.format, self.unit, self.date_only, self.offset_rows, self.scaled_rows
        )];
        out.extend(self.notes.iter().map(|n| format!("# time: {}", n)));
        out
    }

    pub fn json(&self) -> String {
        format!(
            "{{\"format\":\"{}\",\"unit\":\"{}\",\"date_only\":\"{}\",\"rows\":{},\"offset_rows\":{},\"scaled_rows\":{},\"notes\":{}}}",
            self.format,
            self.unit,
            self.date_only,
            self.rows,
            self.offset_rows,
            self.scaled_rows,
            json_note_array(&self.notes)
        )
    }
}

fn json_note_array(notes: &[String]) -> String {
    let items: Vec<String> = notes
        .iter()
        .map(|n| format!("\"{}\"", n.replace('\\', "\\\\").replace('"', "\\\"")))
        .collect();
    format!("[{}]", items.join(","))
}

/// Numeric unit windows: the converted instant must land between 1973 and
/// 2103. The windows are disjoint by a factor of 1000, so realistic data is
/// unambiguous; tiny values (row indices) are handled separately.
const EPOCH_MIN_SECS: f64 = 1.0e8;
const EPOCH_MAX_SECS: f64 = 4.2e9;

fn unit_fits(vmin: f64, vmax: f64, unit: TimeUnit) -> bool {
    let inv = 1.0 / unit.to_seconds();
    let lo = EPOCH_MIN_SECS * inv;
    let hi = EPOCH_MAX_SECS * inv;
    vmin >= lo && vmax <= hi
}

fn detect_time_unit(values: &[f64], spec: &TimeSpec) -> Result<(TimeUnit, usize), String> {
    let mut vmin = f64::INFINITY;
    let mut vmax = f64::NEG_INFINITY;
    let mut finite = 0usize;
    for &v in values {
        if v.is_finite() {
            finite += 1;
            vmin = vmin.min(v);
            vmax = vmax.max(v);
        }
    }
    if finite == 0 {
        return Err("time column has no finite values".to_string());
    }
    if let Some(u) = spec.unit {
        return Ok((u, finite));
    }
    // Row-index-like columns: too small to be an epoch in any unit. Keep the
    // historical meaning (seconds) so synthetic integer time axes are
    // unchanged.
    if vmin >= 0.0 && vmax < EPOCH_MIN_SECS {
        return Ok((TimeUnit::Seconds, 0));
    }
    let fits: Vec<TimeUnit> = [TimeUnit::Seconds, TimeUnit::Millis, TimeUnit::Micros, TimeUnit::Nanos]
        .into_iter()
        .filter(|&u| unit_fits(vmin, vmax, u))
        .collect();
    match fits.as_slice() {
        [one] => Ok((*one, if *one == TimeUnit::Seconds { 0 } else { finite })),
        [] => Err(format!(
            "numeric time column spans [{} ..= {}] and fits no single epoch unit \
             (s/ms/us/ns); pass --time-unit s|ms|us|ns if you know the unit",
            vmin, vmax
        )),
        many => Err(format!(
            "numeric time column is ambiguous: it fits {} units ({}); pass --time-unit \
             s|ms|us|ns",
            many.len(),
            many.iter().map(|u| u.name()).collect::<Vec<_>>().join(", ")
        )),
    }
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 if (y % 4 == 0 && y % 100 != 0) || y % 400 == 0 => 29,
        2 => 28,
        _ => 0,
    }
}

struct ParsedTime {
    secs: f64,
    /// Comparable signature of the syntactic form: separator, date-only vs
    /// date-time, and whether the value carried an offset.
    signature: String,
    has_offset: bool,
    date_only: bool,
}

/// Parse one strict ISO 8601 / RFC 3339 value.
fn parse_iso(s: &str, spec: &TimeSpec) -> Result<ParsedTime, String> {
    let (date, clock) = match s.split_once(['T', ' ']) {
        Some((d, c)) => (d, Some(c.trim())),
        None => (s, None),
    };
    if date.is_empty() {
        return Err(format!("'{}' has no date", s));
    }
    let sep = if date.contains('/') {
        '/'
    } else {
        '-'
    };
    let parts: Vec<&str> = date.split(sep).collect();
    if parts.len() != 3 || parts.iter().any(|p| p.is_empty()) {
        return Err(format!(
            "'{}': expected YYYY-MM-DD (or YYYY/MM/DD){}",
            s,
            if clock.is_some() { " with an optional clock" } else { "" }
        ));
    }
    let y: i64 = parts[0]
        .parse()
        .map_err(|_| format!("'{}': bad year '{}'", s, parts[0]))?;
    let mo: i64 = parts[1]
        .parse()
        .map_err(|_| format!("'{}': bad month '{}'", s, parts[1]))?;
    let d: i64 = parts[2]
        .parse()
        .map_err(|_| format!("'{}': bad day '{}'", s, parts[2]))?;
    if !(1..=12).contains(&mo) {
        return Err(format!("'{}': month {} is out of range", s, mo));
    }
    if d < 1 || d > days_in_month(y, mo) {
        return Err(format!("'{}': day {} is out of range for {}-{:02}", s, d, y, mo));
    }
    let mut secs = days_from_civil(y, mo, d) as f64 * 86400.0;
    let mut has_offset = false;
    let mut has_clock = false;
    if let Some(mut c) = clock {
        if c.is_empty() {
            return Err(format!("'{}': empty clock after T/space", s));
        }
        has_clock = true;
        let mut offset = 0i64;
        if c.ends_with('Z') || c.ends_with('z') {
            has_offset = true;
            c = &c[..c.len() - 1];
        } else if let Some(pos) = c.rfind(['+', '-']) {
            if pos > 0 {
                has_offset = true;
                let sign = if c.as_bytes()[pos] == b'-' { -1 } else { 1 };
                let off = &c[pos + 1..];
                let digits: String = off.chars().filter(|ch| *ch != ':').collect();
                if digits.len() != 4 || !digits.chars().all(|ch| ch.is_ascii_digit()) {
                    return Err(format!("'{}': bad UTC offset '+{}'", s, &c[pos..]));
                }
                let oh: i64 = digits[0..2].parse().unwrap_or(99);
                let om: i64 = digits[2..4].parse().unwrap_or(99);
                if oh > 23 || om > 59 {
                    return Err(format!("'{}': UTC offset out of range", s));
                }
                offset = sign * (oh * 3600 + om * 60);
                c = &c[..pos];
            }
        }
        let tp: Vec<&str> = c.split(':').collect();
        if tp.len() < 2 || tp.len() > 3 {
            return Err(format!("'{}': expected HH:MM[:SS[.fff]]", s));
        }
        let hh: i64 = tp[0]
            .trim()
            .parse()
            .map_err(|_| format!("'{}': bad hour '{}'", s, tp[0]))?;
        let mm: i64 = tp[1]
            .trim()
            .parse()
            .map_err(|_| format!("'{}': bad minute '{}'", s, tp[1]))?;
        let ss: f64 = if tp.len() == 3 {
            tp[2]
                .trim()
                .parse()
                .map_err(|_| format!("'{}': bad second '{}'", s, tp[2]))?
        } else {
            0.0
        };
        if !(0..=23).contains(&hh) || !(0..=59).contains(&mm) || !(0.0..61.0).contains(&ss) {
            return Err(format!("'{}': clock component out of range", s));
        }
        secs += hh as f64 * 3600.0 + mm as f64 * 60.0 + ss;
        // An explicit offset converts to UTC. Without one, use --tz or UTC;
        // the caller records which assumption was used.
        let applied = if has_offset {
            offset
        } else {
            spec.tz.unwrap_or(0)
        };
        secs -= applied as f64;
    } else {
        match spec.date_only {
            DateOnlyPolicy::Error => {
                return Err(format!(
                    "'{}' is date-only: its instant is under-specified. Pass \
                     --date-only utc-midnight or --date-only session-close",
                    s
                ))
            }
            DateOnlyPolicy::UtcMidnight => {}
            DateOnlyPolicy::SessionClose => secs += 21.0 * 3600.0,
        }
    }
    // Widths catch the `2024-1-2` vs `2024-01-02` mix that a purely
    // lexicographic fallback would order wrongly. Offsets are compared
    // separately: with --tz a mixed offset/plain column is allowed.
    let signature = format!(
        "{}{}{}{}",
        sep,
        has_clock,
        parts[1].len(),
        parts[2].len()
    );
    Ok(ParsedTime {
        secs,
        signature,
        has_offset,
        date_only: !has_clock,
    })
}

/// Parse a timestamp to seconds with the strictest defaults.
///
/// Accepts plain numbers (returned unchanged), `YYYY-MM-DD` (as UTC midnight,
/// for backwards compatibility of this helper), `YYYY/MM/DD` and an optional
/// `T`/space separated `HH:MM[:SS[.fff]]` with `Z` or `±HH:MM` offsets.
/// [`time_keys_with`] is the column-level entry point and enforces the
/// date-only policy.
pub fn parse_time(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let spec = TimeSpec {
        date_only: DateOnlyPolicy::UtcMidnight,
        ..Default::default()
    };
    parse_iso(s, &spec).ok().map(|p| p.secs)
}

/// Comparable ordering keys for a time column, with an explicit [`TimeSpec`].
///
/// Numeric columns are unit-detected (s/ms/us/ns) and normalised to seconds;
/// ISO 8601 columns are offset-aware and must share one format (and one
/// offset convention unless `--tz` overrides). Every conversion and assumption
/// is returned in [`TimeReport`].
pub fn time_keys_with(
    t: &Table,
    col: usize,
    spec: &TimeSpec,
) -> Result<(Vec<f64>, TimeReport), String> {
    let n = t.len();
    let mut report = TimeReport {
        date_only: spec.date_only.name().to_string(),
        ..Default::default()
    };
    // A full numeric column: detect and normalise its unit. A high-cardinality
    // numeric column with missing cells takes the same path (NaN is a gap).
    if t.numeric[col] || t.dict(col).is_none() {
        let values = t.col(col);
        let (unit, scaled) = detect_time_unit(values, spec)?;
        let factor = unit.to_seconds();
        let keys: Vec<f64> = values
            .iter()
            .map(|&v| {
                if v.is_nan() {
                    f64::INFINITY
                } else {
                    v * factor
                }
            })
            .collect();
        report.format = "numeric".to_string();
        report.unit = unit.name().to_string();
        report.rows = n;
        report.scaled_rows = scaled;
        if scaled > 0 {
            report.notes.push(format!(
                "numeric time column read as {} and scaled to seconds ({} row(s))",
                unit.name(),
                scaled
            ));
        } else {
            report
                .notes
                .push(format!("numeric time column read as {}", unit.name()));
        }
        return Ok((keys, report));
    }
    let dict = t.dict(col).expect("dictionary checked above");
    let mut keys = Vec::with_capacity(n);
    let mut signature: Option<String> = None;
    let mut first_has_offset: Option<bool> = None;
    let mut offset_rows = 0usize;
    let mut plain_rows = 0usize;
    let mut date_only_rows = 0usize;
    let mut assumed_utc = false;
    for r in 0..n {
        let raw = dict.text(r).trim();
        if raw.is_empty() || is_missing_token(raw) {
            keys.push(f64::INFINITY);
            continue;
        }
        let parsed = parse_iso(raw, spec)
            .map_err(|e| format!("data row {} (time column '{}'): {}", r + 1, t.names[col], e))?;
        if let Some(sig) = &signature {
            // Mixed formats are refused; an offset/plain mix is refused unless
            // --tz tells us what the plain values mean.
            if *sig != parsed.signature {
                return Err(format!(
                    "data row {} (time column '{}'): mixed time formats ('{}' after an earlier \
                     '{}'); normalise the column, or pass --tz/--date-only with one convention",
                    r + 1,
                    t.names[col],
                    raw,
                    dict.text(0)
                ));
            }
            if first_has_offset != Some(parsed.has_offset) && spec.tz.is_none() {
                return Err(format!(
                    "data row {} (time column '{}'): mixes offset-bearing and offset-less \
                     timestamps ('{}'); pass --tz UTC or a fixed offset to state what the \
                     offset-less values mean",
                    r + 1,
                    t.names[col],
                    raw
                ));
            }
        } else {
            signature = Some(parsed.signature.clone());
            first_has_offset = Some(parsed.has_offset);
        }
        if parsed.has_offset {
            offset_rows += 1;
        } else {
            plain_rows += 1;
        }
        if parsed.date_only {
            date_only_rows += 1;
        }
        if !parsed.has_offset && spec.tz.is_none() && !assumed_utc && !parsed.date_only {
            assumed_utc = true;
        }
        keys.push(parsed.secs);
    }
    let sig = signature.unwrap_or_else(|| "-".to_string());
    report.format = if sig.contains('/') { "iso8601-slash" } else { "iso8601" }.to_string();
    report.unit = "s".to_string();
    report.rows = n;
    report.offset_rows = offset_rows;
    if date_only_rows > 0 {
        report.notes.push(format!(
            "{} date-only value(s) read as {}",
            date_only_rows,
            spec.date_only.name()
        ));
    }
    if assumed_utc {
        report
            .notes
            .push("offset-less timestamps interpreted as UTC (no --tz given)".to_string());
    }
    if offset_rows > 0 {
        report
            .notes
            .push(format!("{} row(s) converted from their ISO 8601 offset to UTC", offset_rows));
    }
    if plain_rows > 0 && offset_rows > 0 {
        report
            .notes
            .push("mixed offset/offset-less input accepted because --tz was given".to_string());
    }
    Ok((keys, report))
}

/// Days since the Unix epoch (Howard Hinnant's `days_from_civil`).
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = if m <= 2 { y - 1 } else { y };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Stable group ids (0..g) from a label or numeric column, plus the group names.
pub fn group_ids(t: &Table, col: usize) -> (Vec<usize>, Vec<String>) {
    let mut names: Vec<String> = Vec::new();
    let mut ids = Vec::with_capacity(t.len());
    if let Some(d) = t.dict(col) {
        let mut map: HashMap<&str, usize> = HashMap::new();
        for r in 0..t.len() {
            let key = d.text(r).trim();
            let id = match map.get(key) {
                Some(&i) => i,
                None => {
                    let i = names.len();
                    names.push(key.to_string());
                    map.insert(key, i);
                    i
                }
            };
            ids.push(id);
        }
        return (ids, names);
    }
    // A high-cardinality numeric group column: the value itself is the key.
    let mut map: HashMap<u64, usize> = HashMap::new();
    for r in 0..t.len() {
        let v = t.value(r, col);
        let key = v.to_bits();
        let id = match map.get(&key) {
            Some(&i) => i,
            None => {
                let i = names.len();
                names.push(format!("{}", v));
                map.insert(key, i);
                i
            }
        };
        ids.push(id);
    }
    (ids, names)
}

// ---------------------------------------------------------------------------
// History and delayed variables (time-delay embedding)
// ---------------------------------------------------------------------------

/// Options for turning a table into a time-delay embedding.
#[derive(Clone, Default)]
pub struct LagSpec {
    /// Number of delayed copies added per input variable (`x_lag1 .. x_lagN`).
    pub lags: usize,
    /// Fraction of each series kept as *history*: its leading rows are dropped
    /// from the scored data but still supply the past for later rows.
    pub history_frac: f64,
    /// Start a new series when the time step exceeds this multiple of the
    /// series' median step (`0` disables gap splitting).
    pub gap_break: f64,
    /// #130: how the time column is parsed.
    pub time: TimeSpec,
}

#[derive(Clone, Default)]
pub struct LagReport {
    pub lags: usize,
    pub series: usize,
    pub rows_in: usize,
    pub rows_kept: usize,
    pub history_rows: usize,
    pub breaks: usize,
    pub notes: Vec<String>,
}

/// Append `x_lag1 .. x_lagK` columns for every input variable.
///
/// The past of a row is its predecessor **inside the same series**, where a
/// series is a group (when `--group-by` is set) broken further at time gaps.
/// The first `max(K, history_frac · series_length)` rows of every series are
/// dropped: they cannot be scored without looking before the start of the
/// series, and the rest of the history reserve is exactly the "Maximum History
/// Data" idea — rows that only exist to give later rows a past.
pub fn add_lags(
    t: &Table,
    vars: &[String],
    lag: &LagSpec,
    time_col: Option<usize>,
    group_col: Option<usize>,
) -> Result<(Table, Vec<String>, LagReport), String> {
    let n = t.len();
    let mut report = LagReport {
        lags: lag.lags,
        rows_in: n,
        ..Default::default()
    };
    if lag.lags == 0 && lag.history_frac <= 0.0 {
        report.rows_kept = n;
        return Ok((t.clone(), vars.to_vec(), report));
    }
    let var_idx: Vec<usize> = vars
        .iter()
        .map(|v| t.index_of(v))
        .collect::<Result<_, _>>()?;

    // Series id per row, and the ordering key used inside a series.
    let (gid, group_count) = match group_col {
        Some(c) => {
            let (ids, names) = group_ids(t, c);
            (ids, names.len())
        }
        None => (vec![0usize; n], 1),
    };
    let keys: TimeKeys = match time_col {
        Some(c) => time_keys(t, c),
        None => TimeKeys::F64((0..n).map(|i| i as f64).collect()),
    };

    let mut by_series: Vec<Vec<usize>> = vec![Vec::new(); group_count];
    for i in 0..n {
        by_series[gid[i]].push(i);
    }

    // Walk each series in time order, cutting it into segments at big steps,
    // and record each row's segment and position inside it.
    let mut seg_id = vec![usize::MAX; n];
    let mut seg_pos = vec![0usize; n];
    let mut seg_rows: Vec<Vec<usize>> = Vec::new();
    for rows in by_series.iter_mut() {
        rows.sort_by(|&a, &b| keys.cmp(a, b).then(a.cmp(&b)));
        if rows.is_empty() {
            continue;
        }
        let dts: Vec<f64> = if lag.gap_break > 0.0 && rows.len() > 1 {
            rows.windows(2).map(|w| keys.difference(w[0], w[1])).collect()
        } else {
            Vec::new()
        };
        let threshold = {
            let mut sorted = dts.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            match sorted.get(sorted.len() / 2) {
                Some(&med) if med > 0.0 => Some(lag.gap_break * med),
                _ => None,
            }
        };
        let mut cur = usize::MAX;
        for (k, &r) in rows.iter().enumerate() {
            let starts = k == 0
                || match threshold {
                    Some(th) if dts[k - 1] > th => {
                        report.breaks += 1;
                        true
                    }
                    _ => false,
                };
            if starts {
                cur = seg_rows.len();
                seg_rows.push(Vec::new());
            }
            seg_id[r] = cur;
            seg_pos[r] = seg_rows[cur].len();
            seg_rows[cur].push(r);
        }
    }
    report.series = seg_rows.len();

    // Keep a row only once its segment has enough past for every lag and the
    // requested history reserve.
    let mut keep = vec![false; n];
    for i in 0..n {
        let len = seg_rows[seg_id[i]].len() as f64;
        let reserve = lag.lags.max((lag.history_frac * len).ceil() as usize);
        if seg_pos[i] >= reserve {
            keep[i] = true;
        } else {
            report.history_rows += 1;
        }
    }

    // Names for the new columns, in the same variable order.
    let mut new_vars = vars.to_vec();
    let mut lag_names = Vec::new();
    for v in vars {
        for k in 1..=lag.lags {
            let name = format!("{}_lag{}", v, k);
            lag_names.push(name.clone());
            new_vars.push(name);
        }
    }

    let kept: Vec<usize> = (0..n).filter(|&i| keep[i]).collect();
    let mut out = t.select_indices(&kept);
    report.rows_kept = kept.len();
    if report.rows_kept == 0 {
        return Err("history/delays removed every row (reduce --lags/--history-frac)".to_string());
    }
    // Delayed copies are numeric and high-cardinality: values only, no text.
    let mut lag_values: Vec<Vec<f64>> = vec![Vec::with_capacity(kept.len()); lag_names.len()];
    for (r, &i) in kept.iter().enumerate() {
        let mut c = 0usize;
        for &j in &var_idx {
            for k in 1..=lag.lags {
                let v = if seg_pos[i] >= k {
                    let src = seg_rows[seg_id[i]][seg_pos[i] - k];
                    t.value(src, j)
                } else {
                    f64::NAN
                };
                lag_values[c].push(v);
                c += 1;
            }
        }
        let _ = r;
    }
    for (name, values) in lag_names.into_iter().zip(lag_values) {
        out.push_column(
            name,
            true,
            Column {
                values,
                ints: None,
                dict: None,
            },
        );
    }
    if report.breaks > 0 {
        report.notes.push(format!(
            "{} time gap(s) treated as series boundaries",
            report.breaks
        ));
    }
    report.notes.push(format!(
        "{} series, {} history row(s) dropped, {} delayed column(s) added",
        report.series,
        report.history_rows,
        lag.lags * vars.len()
    ));

    Ok((out, new_vars, report))
}

fn hash_str(s: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// True when a column name marks it as a time/ordering axis.
fn time_name(name: &str) -> bool {
    let name = name.to_lowercase();
    ["time", "date", "timestamp", "datetime", "ts", "day", "dt"]
        .iter()
        .any(|k| name == *k || name.contains(k))
}

/// True when column `j` looks like a time/ordering axis (by name, or because it
/// is monotone non-decreasing and almost all distinct).
pub fn is_time_like(t: &Table, j: usize) -> bool {
    if time_name(&t.names[j]) {
        return true;
    }
    if !t.numeric[j] {
        return false;
    }
    let col: Vec<f64> = t.col(j).to_vec();
    let mono = col.windows(2).all(|w| w[0] <= w[1]);
    mono && t.distinct_count(j) * 5 >= col.len() * 4
}

/// True when column `j` looks like a group label: text, or a low-cardinality
/// numeric column.
pub fn is_group_like(t: &Table, j: usize) -> bool {
    let d = t.distinct_count(j);
    if !t.numeric[j] {
        return d >= 2;
    }
    d >= 2 && d * 5 <= t.len().max(5) + 1
}

/// First column (excluding `exclude`) that looks like a time axis.
pub fn find_time_col(t: &Table, exclude: &[usize]) -> Option<usize> {
    (0..t.names.len())
        .find(|&j| !exclude.contains(&j) && is_time_like(t, j))
}

/// First column (excluding `exclude`) that looks like a group label.
pub fn find_group_col(t: &Table, exclude: &[usize]) -> Option<usize> {
    (0..t.names.len())
        .find(|&j| !exclude.contains(&j) && is_group_like(t, j))
}

/// Pick a split mode from the shape of the table.
///
/// Returns `(mode, time column, group column, notes)`. The caller decides
/// whether to accept it; nothing is changed silently.
pub fn suggest_split(
    t: &Table,
    target_idx: usize,
    var_idx: &[usize],
) -> (SplitMode, Option<usize>, Option<usize>, Vec<String>) {
    let mut notes: Vec<String> = Vec::new();
    let mut exclude: Vec<usize> = vec![target_idx];
    exclude.extend_from_slice(var_idx);

    let time_col = find_time_col(t, &exclude);
    if let Some(j) = time_col {
        exclude.push(j);
    }
    let group_col = find_group_col(t, &exclude);

    // Notes describe the *observation*, never the decision: the caller may be
    // the `--split auto` path (which adopts the mode) or the warning path
    // (which does not). Asserting "using a chronological split" here made the
    // warning contradict the `split=random` line it accompanied.
    let mode = match (time_col, group_col) {
        (Some(j), Some(g)) => {
            notes.push(format!(
                "found a time-like column '{}' and a group-like column '{}'",
                t.names[j], t.names[g]
            ));
            SplitMode::TemporalGroup
        }
        (Some(j), None) => {
            notes.push(format!("found a time-like column '{}'", t.names[j]));
            SplitMode::Temporal
        }
        (None, Some(g)) => {
            notes.push(format!("found a group-like column '{}'", t.names[g]));
            SplitMode::Group
        }
        (None, None) => {
            notes.push("found no time-like or group-like column".to_string());
            SplitMode::Random
        }
    };
    (mode, time_col, group_col, notes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(csv: &str) -> Table {
        parse_table(csv).unwrap()
    }

    fn temporal() -> SplitSpec {
        SplitSpec {
            mode: SplitMode::Temporal,
            time_col: Some(0),
            test_frac: 0.3,
            ..Default::default()
        }
    }

    #[test]
    fn missing_markers_are_parsed_as_gaps() {
        let t = table("x,y\n1,2\nNA,3\n,4\nnull,5\nNaN,6\n");
        assert!(t.numeric[0] && t.numeric[1]);
        assert_eq!(t.value(0, 0), 1.0);
        for r in 1..5 {
            assert!(t.value(r, 0).is_nan());
        }
        // A genuinely unparseable cell still makes the column non-numeric.
        let bad = table("x\n1\nnot-a-number\n");
        assert!(!bad.numeric[0]);
    }

    #[test]
    fn dtype_spec_parses_kinds_and_rejects_bad_entries() {
        let spec = parse_dtype_spec("flag=boolean, temp = continuous,sym=categorical").unwrap();
        assert_eq!(spec.len(), 3);
        assert_eq!(spec[0].0, "flag");
        assert_eq!(spec[0].1, DType::Boolean);
        assert_eq!(spec[2].1, DType::Categorical);
        assert!(parse_dtype_spec("flag=ordinal").unwrap_err().contains("boolean"));
        assert!(parse_dtype_spec("flag").unwrap_err().contains("<column>=<kind>"));
        assert!(parse_dtype_spec("flag=boolean,flag=continuous")
            .unwrap_err()
            .contains("more than once"));
    }

    #[test]
    fn apply_dtypes_expands_levels_and_keeps_boolean_raw() {
        let t = table("symbol,flag,y\nB,0,0\nA,1,1\nB,1,0\n");
        let typed = apply_dtypes(
            &t,
            &["symbol".into(), "flag".into()],
            &[
                ("symbol".into(), DType::Categorical),
                ("flag".into(), DType::Boolean),
            ],
            16,
        )
        .unwrap();
        assert_eq!(typed.vars, vec!["symbol=A", "symbol=B", "flag"]);
        assert_eq!(
            typed.dtypes,
            vec![DType::Boolean, DType::Boolean, DType::Boolean]
        );
        assert_eq!(typed.variables[1].source.as_deref(), Some("symbol"));
        let a = typed.table.index_of("symbol=A").unwrap();
        assert_eq!(
            typed.table.col(a).to_vec(),
            vec![0.0, 1.0, 0.0]
        );
    }

    #[test]
    fn quantile_interpolates_between_ranks() {
        let v = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(quantile(&v, 0.0), 0.0);
        assert_eq!(quantile(&v, 1.0), 3.0);
        assert_eq!(quantile(&v, 0.5), 1.5);
        assert_eq!(quantile(&v, 0.25), 0.75);
        assert_eq!(quantile(&v, 0.75), 2.25);
        assert_eq!(quantile(&[7.0], 0.5), 7.0);
        assert!(quantile(&[], 0.5).is_nan());
    }

    /// `previous` walks each side in table order; its first missing row cannot
    /// look into the other side and is dropped instead.
    #[test]
    fn previous_fill_stays_inside_the_side() {
        let t = table("t,x,y\n0,1,0\n1,NA,0\n2,3,0\n3,NA,0\n4,NA,0\n5,6,0\n");
        let p = prepare_full_with(
            &t,
            &["x".into()],
            "y",
            &temporal(),
            &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Previous,
                center: Center::None,
                scale: Scale::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.train.x.iter().map(|r| r[0]).collect::<Vec<_>>(), vec![1.0, 1.0, 3.0, 3.0]);
        assert_eq!(p.prep.rows_filled, 2);
        // Test row 4 is the first row of its side and has no previous value.
        assert_eq!(p.prep.rows_dropped_missing, 1);
        assert_eq!(p.test.x.iter().map(|r| r[0]).collect::<Vec<_>>(), vec![6.0]);
    }

    /// `zero` is the literal fill, with no training statistic involved.
    #[test]
    fn zero_fill_sets_gaps_to_zero() {
        let t = table("t,x,y\n0,1,0\n1,NA,0\n2,3,0\n3,NA,0\n4,5,0\n5,6,0\n");
        let p = prepare_full_with(
            &t,
            &["x".into()],
            "y",
            &temporal(),
            &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Zero,
                center: Center::None,
                scale: Scale::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            p.train.x.iter().map(|r| r[0]).collect::<Vec<_>>(),
            vec![1.0, 0.0, 3.0, 0.0]
        );
        assert_eq!(
            p.test.x.iter().map(|r| r[0]).collect::<Vec<_>>(),
            vec![5.0, 6.0]
        );
        assert_eq!(p.prep.rows_filled, 2);
        assert_eq!(p.prep.rows_dropped_missing, 0);
    }

    /// `interpolate` uses the two observed neighbours inside a side and drops a
    /// gap at an edge rather than extrapolating.
    #[test]
    fn interpolate_uses_neighbours_and_drops_edges() {
        let t = table("t,x,y\n0,1,0\n1,NA,0\n2,3,0\n3,NA,0\n4,NA,0\n5,6,0\n");
        let p = prepare_full_with(
            &t,
            &["x".into()],
            "y",
            &temporal(),
            &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Interpolate,
                allow_interpolate: true,
                center: Center::None,
                scale: Scale::None,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(p.train.x.iter().map(|r| r[0]).collect::<Vec<_>>(), vec![1.0, 2.0, 3.0]);
        assert_eq!(p.prep.rows_filled, 1);
        // The trailing train gap and the leading test gap cannot interpolate.
        assert_eq!(p.prep.rows_dropped_missing, 2);
        assert_eq!(p.test.x.iter().map(|r| r[0]).collect::<Vec<_>>(), vec![6.0]);
    }

    /// #133: the default path (no fill, no outlier rule, no smoothing) reads
    /// the table directly. Only a mutating preparation copies the used columns.
    #[test]
    fn the_default_path_does_not_copy_the_used_columns() {
        let t = table("t,x,y\n0,1,2\n1,2,4\n2,3,6\n3,4,8\n4,5,10\n");
        let vars = ["x".to_string()];

        CELL_COPIES.with(|n| n.set(0));
        let p = prepare_full_with(
            &t,
            &vars,
            "y",
            &SplitSpec::default(),
            &mut Mt::new(1),
            &PrepSpec::default(),
        )
        .unwrap();
        assert_eq!(p.train.len() + p.test.len(), 5);
        assert_eq!(
            CELL_COPIES.with(|n| n.get()),
            0,
            "the default mean/std path must not materialise a row-major copy"
        );

        // #133: `drop` only reads cells to build the keep mask; it must not
        // trigger the row-major copy that mutating fills need.
        CELL_COPIES.with(|n| n.set(0));
        let _ = prepare_full_with(
            &t,
            &vars,
            "y",
            &SplitSpec::default(),
            &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Drop,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            CELL_COPIES.with(|n| n.get()),
            0,
            "a drop-only fill must not copy the used columns"
        );

        CELL_COPIES.with(|n| n.set(0));
        let _ = prepare_full_with(
            &t,
            &vars,
            "y",
            &SplitSpec::default(),
            &mut Mt::new(1),
            &PrepSpec {
                fill: FillMethod::Mean,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(
            CELL_COPIES.with(|n| n.get()),
            1,
            "a mutating fill needs one mutable copy"
        );
    }
}
