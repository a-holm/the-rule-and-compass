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

use crate::expr::{self, Node};
use crate::rng::Mt;
use std::collections::{BTreeSet, HashMap};

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
}

impl Sample {
    /// An unweighted sample (every row weighs 1).
    pub fn new(x: Vec<Vec<f64>>, y: Vec<f64>) -> Self {
        Sample {
            x,
            y,
            w: Vec::new(),
        }
    }

    /// A sample with one weight per row.
    ///
    /// The caller is responsible for `w.len() == y.len()` and for the values
    /// being finite and non-negative; `prepare_full_weighted_with` validates a
    /// weight column before it gets this far.
    pub fn weighted(x: Vec<Vec<f64>>, y: Vec<f64>, w: Vec<f64>) -> Self {
        Sample { x, y, w }
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
}

/// A parsed table. Numeric cells appear in `rows`; the original text of every
/// cell is kept in `raw` so date and label columns remain usable.
#[derive(Clone)]
pub struct Table {
    pub names: Vec<String>,
    /// `numeric[j]` is true when every non-empty cell in column `j` parses as a
    /// number.
    pub numeric: Vec<bool>,
    /// Numeric view; non-numeric cells are `NaN`.
    pub rows: Vec<Vec<f64>>,
    /// Text view of every cell.
    pub raw: Vec<Vec<String>>,
}

impl Table {
    pub fn index_of(&self, name: &str) -> Result<usize, String> {
        self.names
            .iter()
            .position(|n| n == name)
            .ok_or_else(|| format!("column '{}' not found in {:?}", name, self.names))
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Number of distinct values in a column (used by auto-detection).
    pub fn distinct_count(&self, col: usize) -> usize {
        let mut seen: Vec<&str> = Vec::new();
        for r in &self.raw {
            let s = r[col].trim();
            if !seen.contains(&s) {
                seen.push(s);
            }
        }
        seen.len()
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
                    for (r, raw) in t.raw.iter().enumerate() {
                        let cell = raw[j].trim();
                        let hit = if is_missing_token(cell) {
                            level == MISSING_LEVEL
                        } else {
                            cell == level
                        };
                        table.rows[r].push(if hit { 1.0 } else { 0.0 });
                        table.raw[r].push(if hit { "1" } else { "0" }.to_string());
                    }
                    table.names.push(name.clone());
                    table.numeric.push(true);
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
    for (r, raw) in t.raw.iter().enumerate() {
        let cell = raw[j].trim();
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
    for raw in &t.raw {
        let cell = raw[j].trim();
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

pub fn read_table(path: &str) -> Result<Table, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    parse_table(&text)
}

/// Parse a delimited table (header row required, `,`/tab/whitespace detected).
pub fn parse_table(text: &str) -> Result<Table, String> {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header = lines.next().ok_or_else(|| "empty input".to_string())?;
    let delim = if header.contains('\t') {
        '\t'
    } else if header.contains(',') {
        ','
    } else {
        ' '
    };
    let split = |l: &str| -> Vec<String> {
        if delim == ' ' {
            l.split_whitespace().map(|s| s.to_string()).collect()
        } else {
            l.split(delim).map(|s| s.trim().to_string()).collect()
        }
    };
    let names = split(header);
    let mut raw: Vec<Vec<String>> = Vec::new();
    for l in lines {
        let parts = split(l);
        if parts.len() != names.len() {
            continue;
        }
        raw.push(parts);
    }
    if raw.is_empty() {
        return Err("no data rows".to_string());
    }

    // A column is numeric when every non-missing cell parses as a number.
    // Missing tokens (`''`, `NA`, `NaN`, `null`, case-insensitively) are skipped
    // here and become `NaN` in the numeric view.
    let mut numeric = vec![false; names.len()];
    for (j, slot) in numeric.iter_mut().enumerate() {
        let mut any = false;
        let mut all = true;
        for r in &raw {
            let cell = r[j].trim();
            if is_missing_token(cell) {
                continue;
            }
            any = true;
            if cell.parse::<f64>().is_err() {
                all = false;
                break;
            }
        }
        *slot = any && all;
    }

    let rows: Vec<Vec<f64>> = raw
        .iter()
        .map(|r| {
            r.iter()
                .enumerate()
                .map(|(j, c)| {
                    if numeric[j] && !is_missing_token(c) {
                        c.trim().parse::<f64>().unwrap_or(f64::NAN)
                    } else {
                        f64::NAN
                    }
                })
                .collect()
        })
        .collect();

    Ok(Table {
        names,
        numeric,
        rows,
        raw,
    })
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
    /// Number of contiguous evaluation blocks in the test region.
    pub folds: usize,
    /// Rows where this expression is true are assigned to training (issue #3).
    /// When either selection is present it replaces the positional mode.
    pub train_select: Option<Filter>,
    /// Rows where this expression is true are assigned to test (issue #3).
    pub test_select: Option<Filter>,
    /// Rows where this expression is true are dropped before splitting (#3).
    pub exclude: Option<Filter>,
}

impl Default for SplitSpec {
    fn default() -> Self {
        SplitSpec {
            mode: SplitMode::Random,
            time_col: None,
            group_col: None,
            test_frac: 0.25,
            gap: 0,
            folds: 1,
            train_select: None,
            test_select: None,
            exclude: None,
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
    pub notes: Vec<String>,
}

pub struct Prepared {
    pub train: Sample,
    /// All test folds concatenated, in fold order (what the search evaluates on).
    pub test: Sample,
    /// One sample per evaluation block. With `folds == 1` this is a single
    /// sample identical to `test`.
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
    /// Moving-average window in rows (`0` disables it).
    pub smooth: usize,
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
    }

    /// One short, commented line per active transformation. `params` carries
    /// the raw option values that are not part of the counts.
    pub fn text_lines(&self, outlier_iqr: Option<f64>, smooth: usize) -> Vec<String> {
        let mut out = Vec::new();
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
                "# preparation: smooth={} cells_smoothed={}",
                smooth, self.smoothed
            ));
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
        out
    }

    /// The `preparation` object shared by `--json` and `--ndjson`.
    pub fn json(&self) -> String {
        format!(
            "{{\"fill\":\"{}\",\"rows_filled\":{},\"rows_dropped_missing\":{},\"rows_dropped_outlier\":{},\"smoothed\":{},\"center\":\"{}\",\"scale\":\"{}\",\"filtered\":{}}}",
            self.fill,
            self.rows_filled,
            self.rows_dropped_missing,
            self.rows_dropped_outlier,
            self.smoothed,
            self.center,
            self.scale,
            self.filtered
        )
    }
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
    // With an explicit --fill method, a junk token in an otherwise numeric
    // column is treated as a missing value (and then filled or dropped).
    // Without it the column stays non-numeric, exactly as before.
    let coerced = if prep.fill != FillMethod::None {
        Some(coerce_numeric(t))
    } else {
        None
    };
    let t = coerced.as_ref().unwrap_or(t);
    let (var_idx, tgt_idx) = resolve_columns(t, vars, target)?;
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
    let mut excluded_count = 0usize;
    let mut filtered = t.clone();
    if let Some(x) = &spec.exclude {
        let cols: Vec<usize> = x
            .vars
            .iter()
            .map(|v| filtered.index_of(v))
            .collect::<Result<_, _>>()?;
        let mut rows = Vec::new();
        let mut raw = Vec::new();
        for r in 0..filtered.rows.len() {
            let vals: Vec<f64> = cols.iter().map(|&c| filtered.rows[r][c]).collect();
            if expr::eval(&x.tree, &vals) > 0.0 {
                excluded_count += 1;
            } else {
                rows.push(filtered.rows[r].clone());
                raw.push(filtered.raw[r].clone());
            }
        }
        if rows.is_empty() {
            return Err("--exclude removed every row".to_string());
        }
        filtered.rows = rows;
        filtered.raw = raw;
    }

    // 1. The row filter defines the population the split is drawn from.
    let mut filtered_count = 0usize;
    if let Some(f) = &prep.filter {
        let cols: Vec<usize> = f
            .vars
            .iter()
            .map(|v| filtered.index_of(v))
            .collect::<Result<_, _>>()?;
        let mut rows = Vec::new();
        let mut raw = Vec::new();
        for r in 0..filtered.rows.len() {
            let vals: Vec<f64> = cols.iter().map(|&c| filtered.rows[r][c]).collect();
            if expr::eval(&f.tree, &vals) > 0.0 {
                rows.push(filtered.rows[r].clone());
                raw.push(filtered.raw[r].clone());
            } else {
                filtered_count += 1;
            }
        }
        if rows.is_empty() {
            return Err("--filter kept no rows".to_string());
        }
        filtered.rows = rows;
        filtered.raw = raw;
    }

    let x_raw: Vec<Vec<f64>> = filtered
        .rows
        .iter()
        .map(|r| var_idx.iter().map(|&i| r[i]).collect())
        .collect();
    // Weights follow the rows through the split; validate them on the surviving
    // population (after --exclude / --filter) rather than on discarded rows.
    let weights_raw: Option<Vec<f64>> = match weight_idx {
        Some(j) => {
            let name = weight.unwrap_or("");
            let w: Vec<f64> = filtered.rows.iter().map(|r| r[j]).collect();
            validate_weights(name, &w)?;
            Some(w)
        }
        None => None,
    };
    let (mut train_idx, mut fold_idx, mut report) = build_split(&filtered, spec, rng, &x_raw)?;
    report.excluded_rows = excluded_count;
    if train_idx.is_empty() {
        return Err("training split is empty (reduce --test-frac)".to_string());
    }

    let n = filtered.rows.len();
    // Used columns: every input, then the target.
    let used_cols: Vec<usize> = var_idx.iter().copied().chain(std::iter::once(tgt_idx)).collect();
    let nc = used_cols.len();
    let mut cells: Vec<Vec<f64>> = filtered
        .rows
        .iter()
        .map(|r| used_cols.iter().map(|&c| r[c]).collect())
        .collect();
    let mut keep = vec![true; n];
    let mut row_filled = vec![false; n];
    let mut rows_dropped_missing = 0usize;

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
                let missing = (0..n).filter(|&r| cells[r][k].is_nan()).count();
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
                if (0..nc).any(|k| cells[r][k].is_nan()) {
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
                            let v = cells[r][k];
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
                    for side in split_sides(&train_idx, &fold_idx) {
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
                    for side in split_sides(&train_idx, &fold_idx) {
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
            if keep[r] && (0..nc).any(|k| cells[r][k].is_nan()) {
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
            let v = cells[r][j];
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
                        let v = cells[r][col];
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
                    let v = cells[r][c];
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
        for side in split_sides(&train_idx, &fold_idx) {
            let side: Vec<usize> = side.into_iter().filter(|&r| keep[r]).collect();
            for col in 0..nc {
                // Smoothing a 0/1 column would create fractions and break the
                // truthiness contract; it applies to continuous inputs and the
                // target only.
                if col < nv && dtype_at(col) == DType::Boolean {
                    continue;
                }
                smoothed += smooth_cells(&mut cells, &side, col, prep.smooth);
            }
        }
    }

    // Rows dropped by preparation leave both the training and test sets.
    train_idx.retain(|&i| keep[i]);
    for f in fold_idx.iter_mut() {
        f.retain(|&i| keep[i]);
    }
    fold_idx.retain(|f| !f.is_empty());
    if train_idx.is_empty() {
        return Err("preparation removed every training row".to_string());
    }
    report.train_rows = train_idx.len();
    report.test_rows = fold_idx.iter().map(|f| f.len()).sum();
    report.folds = fold_idx.len().max(1);

    let x_vals: Vec<Vec<f64>> = cells
        .iter()
        .map(|r| r[..nv].to_vec())
        .collect();
    let y_vals: Vec<f64> = cells.iter().map(|r| r[nv]).collect();

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
        let vals: Vec<f64> = train_idx.iter().map(|&i| x_vals[i][j]).collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - m).powi(2)).sum::<f64>() / vals.len() as f64;
        let s = var.sqrt();
        let mut sorted = vals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
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
        for &i in idx {
            s.x.push(
                (0..nv)
                    .map(|j| (x_vals[i][j] - means[j]) / stds[j])
                    .collect(),
            );
            s.y.push(y_vals[i]);
        }
        if let Some(w) = &weights_raw {
            s.w = idx.iter().map(|&i| w[i]).collect();
        }
        s
    };

    let train = make(&train_idx);
    if let (Some(w), Some(name)) = (&weights_raw, weight) {
        let positive = train_idx.iter().filter(|&&i| w[i] > 0.0).count();
        if positive == 0 {
            return Err(format!(
                "weight column '{}' has no positive weight in the training rows after preparation",
                name
            ));
        }
    }
    let fold_samples: Vec<Sample> = fold_idx.iter().map(|f| make(f)).collect();
    let mut test = Sample::default();
    for f in &fold_samples {
        test.x.extend(f.x.iter().cloned());
        test.y.extend(f.y.iter().copied());
        test.w.extend(f.w.iter().copied());
    }

    Ok(Prepared {
        train,
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
        for r in &t.raw {
            let cell = r[j].trim();
            if !is_missing_token(cell) && cell.parse::<f64>().is_ok() {
                any_number = true;
                break;
            }
        }
        if !any_number {
            continue;
        }
        out.numeric[j] = true;
        for (r, row) in out.rows.iter_mut().enumerate() {
            let cell = t.raw[r][j].trim();
            row[j] = if is_missing_token(cell) {
                f64::NAN
            } else {
                cell.parse::<f64>().unwrap_or(f64::NAN)
            };
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
fn split_sides(train: &[usize], folds: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut sides = Vec::new();
    let mut tr = train.to_vec();
    tr.sort_unstable();
    sides.push(tr);
    let mut te: Vec<usize> = folds.iter().flatten().copied().collect();
    te.sort_unstable();
    te.dedup();
    sides.push(te);
    sides
}

/// Moving average over a `k`-wide window, centred on each value and shrinking at
/// the edges of `side`. Missing cells are skipped; a cell with no observed
/// neighbours is left missing. Returns the number of cells replaced.
fn smooth_cells(cells: &mut [Vec<f64>], side: &[usize], col: usize, k: usize) -> usize {
    if side.is_empty() || k < 2 {
        return 0;
    }
    let before = k / 2;
    let after = k - before - 1;
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
            let w: Vec<f64> = t.rows.iter().map(|r| r[j]).collect();
            validate_weights(name, &w)?;
            Some(w)
        }
        None => None,
    };
    let n = t.rows.len();
    let k = k.max(2);
    if n < k + 1 {
        return Err(format!(
            "cross-validation needs more than {} rows, got {}",
            k, n
        ));
    }
    let x_raw: Vec<Vec<f64>> = t
        .rows
        .iter()
        .map(|r| var_idx.iter().map(|&i| r[i]).collect())
        .collect();
    let y: Vec<f64> = t.rows.iter().map(|r| r[tgt_idx]).collect();

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

/// Decide where each row goes. Returns `(train indices, test fold indices, report)`.
fn build_split(
    t: &Table,
    spec: &SplitSpec,
    rng: &mut Mt,
    x_raw: &[Vec<f64>],
) -> Result<(Vec<usize>, Vec<Vec<usize>>, SplitReport), String> {
    let n = t.rows.len();
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
                    let vals: Vec<f64> = cols.iter().map(|&c| t.rows[r][c]).collect();
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
            let keys = time_keys(t, time_col);
            let order = sorted_by(&keys);
            match split_ordered(&order, spec, folds) {
                Some((tr, folds_idx)) => {
                    report.train_time_max = order
                        .get(tr.len().saturating_sub(1))
                        .map(|&i| keys[i]);
                    report.test_time_min = folds_idx
                        .iter()
                        .flatten()
                        .map(|&i| keys[i])
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
            let keys = time_keys(t, time_col);
            let (ids, names) = group_ids(t, group_col);
            report.groups_total = names.len();
            let mut train: Vec<usize> = Vec::new();
            let mut blocks: Vec<Vec<usize>> = vec![Vec::new(); folds];
            let mut groups_train = 0usize;
            for g in 0..names.len() {
                let mut rows: Vec<usize> = (0..n).filter(|&i| ids[i] == g).collect();
                rows.sort_by(|&a, &b| {
                    keys[a]
                        .partial_cmp(&keys[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
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
    let _ = x_raw;
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

fn sorted_by(keys: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..keys.len()).collect();
    order.sort_by(|&a, &b| {
        keys[a]
            .partial_cmp(&keys[b])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    order
}

/// Comparable ordering key for every row of a time column.
///
/// Numbers and dates are parsed to seconds; if any value cannot be parsed the
/// whole column falls back to lexicographic order, which is still a valid total
/// order (and the right one for zero-padded `YYYY-MM-DD`).
pub fn time_keys(t: &Table, col: usize) -> Vec<f64> {
    let n = t.rows.len();
    let parsed: Option<Vec<f64>> = (0..n).map(|r| parse_time(&t.raw[r][col])).collect();
    if let Some(v) = parsed {
        return v;
    }
    let mut pairs: Vec<(String, usize)> = (0..n)
        .map(|r| (t.raw[r][col].trim().to_string(), r))
        .collect();
    pairs.sort();
    let mut key = vec![0.0; n];
    for (rank, (_s, r)) in pairs.into_iter().enumerate() {
        key[r] = rank as f64;
    }
    key
}

/// Parse a timestamp to seconds: accepts plain numbers, `YYYY-MM-DD`,
/// `YYYY/MM/DD` and an optional `T`/space separated `HH:MM[:SS]`.
pub fn parse_time(s: &str) -> Option<f64> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Ok(v) = s.parse::<f64>() {
        return Some(v);
    }
    let (date, clock) = match s.split_once(['T', ' ']) {
        Some((d, c)) => (d, Some(c)),
        None => (s, None),
    };
    let norm: String = date
        .chars()
        .map(|c| if c == '/' { '-' } else { c })
        .collect();
    let parts: Vec<&str> = norm.split('-').collect();
    if parts.len() != 3 {
        return None;
    }
    let y: i64 = parts[0].parse().ok()?;
    let mo: i64 = parts[1].parse().ok()?;
    let d: i64 = parts[2].parse().ok()?;
    if !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let mut secs = days_from_civil(y, mo, d) as f64 * 86400.0;
    if let Some(c) = clock {
        let tp: Vec<&str> = c.split(':').collect();
        let hh: f64 = tp.first().and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        let mm: f64 = tp.get(1).and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        let ss: f64 = tp.get(2).and_then(|x| x.trim().parse().ok()).unwrap_or(0.0);
        secs += hh * 3600.0 + mm * 60.0 + ss;
    }
    Some(secs)
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
    let mut map: HashMap<&str, usize> = HashMap::new();
    let mut names: Vec<String> = Vec::new();
    let mut ids = Vec::with_capacity(t.rows.len());
    for r in &t.raw {
        let key = r[col].trim();
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
    let n = t.rows.len();
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
    let keys: Vec<f64> = match time_col {
        Some(c) => time_keys(t, c),
        None => (0..n).map(|i| i as f64).collect(),
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
        rows.sort_by(|&a, &b| {
            keys[a]
                .partial_cmp(&keys[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        if rows.is_empty() {
            continue;
        }
        let dts: Vec<f64> = if lag.gap_break > 0.0 && rows.len() > 1 {
            rows.windows(2).map(|w| keys[w[1]] - keys[w[0]]).collect()
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
    let mut names = t.names.clone();
    let mut new_vars = vars.to_vec();
    for v in vars {
        for k in 1..=lag.lags {
            let name = format!("{}_lag{}", v, k);
            names.push(name.clone());
            new_vars.push(name);
        }
    }
    let mut numeric = t.numeric.clone();
    numeric.extend(std::iter::repeat(true).take(names.len() - numeric.len()));

    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut raw: Vec<Vec<String>> = Vec::new();
    for i in 0..n {
        if !keep[i] {
            continue;
        }
        let mut r = t.rows[i].clone();
        let mut rr = t.raw[i].clone();
        for &j in &var_idx {
            for k in 1..=lag.lags {
                let v = if seg_pos[i] >= k {
                    let src = seg_rows[seg_id[i]][seg_pos[i] - k];
                    t.rows[src][j]
                } else {
                    f64::NAN
                };
                r.push(v);
                rr.push(format!("{}", v));
            }
        }
        rows.push(r);
        raw.push(rr);
    }

    report.rows_kept = rows.len();
    if report.rows_kept == 0 {
        return Err("history/delays removed every row (reduce --lags/--history-frac)".to_string());
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

    Ok((
        Table {
            names,
            numeric,
            rows,
            raw,
        },
        new_vars,
        report,
    ))
}

fn hash_str(s: &str, seed: u64) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ seed;
    for b in s.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// True when column `j` looks like a time/ordering axis (by name, or because it
/// is monotone non-decreasing and almost all distinct).
pub fn is_time_like(t: &Table, j: usize) -> bool {
    let name = t.names[j].to_lowercase();
    if ["time", "date", "timestamp", "datetime", "ts", "day", "dt"]
        .iter()
        .any(|k| name == *k || name.contains(k))
    {
        return true;
    }
    if !t.numeric[j] {
        return false;
    }
    let col: Vec<f64> = t.rows.iter().map(|r| r[j]).collect();
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
    d >= 2 && d * 5 <= t.rows.len().max(5) + 1
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
        assert_eq!(t.rows[0][0], 1.0);
        for r in 1..5 {
            assert!(t.rows[r][0].is_nan());
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
            typed.table.rows.iter().map(|r| r[a]).collect::<Vec<_>>(),
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
}
