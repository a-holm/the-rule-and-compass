//! #1272: `rcsr panel permute-labels`, the store of the search-capacity null.
//!
//! The value columns of one label family (`lbl_<label>_*` without `t1` and
//! `weight`) move together, per symbol, by one circular-block order over the
//! rows whose `lbl_<label>_y` and return are finite (the label pool). Rows with
//! a NaN `y` and a finite return (the warm-up of the label's volatility window,
//! gaps inside it) move the same way in a pool of their own, because the book
//! prices every row with a finite return. Time, symbol, `t1`, `weight`, every
//! feature column and every row without a finite return keep their bytes, so the
//! priced rows and the search, walk-forward and trials flags of the source store
//! stay the same.
//!
//! The output is an ordinary store with another data hash. Its store header (the
//! `null_store` member, #1304) and its manifest mark it as a null store, so a copy of
//! the store bytes stays marked; a walk-forward report priced on it carries the marker
//! as `null_store`, CPCV paths of such reports carry it on, `walkforward --ledger`
//! (like `ladder --locked-test` and `protocol regimes --after-read`) refuses it, and
//! `rcsr trials` reads a marked report only through `--table --null-wf`.
//!
//! A source with a row inside a locked range is refused. A row in the label
//! purge before a range or in its embargo after it neither moves nor gives a
//! value; the declaration and the lock record carry over to the output.

use crate::ablation::{block_permutation, mix_seed, SegmentKind, Segments};
use crate::devlog::Json;
use crate::labels;
use crate::lockbox::{self, Lockbox, Source};
use crate::lockrec::LockRecord;
use crate::manifest;
use crate::rng::Mt;
use crate::store::{self, ColumnPlan, DType, FileStore};
use crate::target::{self, TargetSpec};
use crate::walkforward::jstr;
use std::collections::BTreeSet;
use std::path::Path;

/// Schema of the manifest of a null store.
pub const SCHEMA: &str = "rcsr-null-store/1";
/// The `order` of the marker: one order per symbol.
pub const ORDER: &str = "per_symbol";
const SEED_TAG: u64 = 0x4E55_4C4C_5045_524D;
/// Mixed into the seed of a gap pool, so its order is independent of the label pool's.
const GAP_TAG: u64 = 0x4741_5050_4F4F_4C31;
const WRITER: &str = "rcsr panel permute-labels";

type Fail = (i32, String);

fn usage<T>(m: impl Into<String>) -> Result<T, Fail> {
    Err((2, m.into()))
}

fn input<T>(m: impl Into<String>) -> Result<T, Fail> {
    Err((3, m.into()))
}

pub struct Options {
    /// Source column store.
    pub data: String,
    pub target_spec: String,
    pub label: String,
    /// Block length in rows of one pool of one symbol.
    pub block: usize,
    pub seed: u64,
    /// Feature columns to keep; `None` keeps every column.
    pub columns: Option<Vec<String>>,
    pub out: String,
    pub time: String,
    pub group_by: String,
    pub lockbox: Option<String>,
    pub lock_record: Option<String>,
}

#[derive(Debug)]
pub struct Summary {
    pub output_sha256: String,
    pub source_sha256: String,
    /// The permuted columns, in store order.
    pub columns: Vec<String>,
    pub symbols: usize,
    /// Rows that take part in the permutation, both pools.
    pub moved_rows: usize,
    /// Of `moved_rows`, the rows of the gap pools.
    pub gap_rows: usize,
    /// Rows of either pool in a label purge or an embargo; they keep their values.
    pub guarded_rows: usize,
    pub manifest: String,
    pub targets: String,
    pub schema: Option<String>,
}

/// The pool of a moving row: a finite `y` and return (`Label`), or a NaN `y` and a finite
/// return (`Gap`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Pool {
    Label,
    Gap,
}

/// The order key of a symbol: the first 64 bits of the SHA-256 of its name, so the
/// order of a symbol does not depend on the other symbols of the store.
pub fn symbol_key(name: &str) -> u64 {
    u64::from_str_radix(&manifest::sha256_hex(name.as_bytes())[..16], 16).unwrap_or(0)
}

/// The source row of every output row. The rows of one symbol (`symbols[i]`, key
/// `keys[symbols[i]]`), one pool and one segment (`slot[i]`) form one series in time
/// order (ties by row); [`block_permutation`] seeded from `(seed, key, segment)`, and
/// the gap tag for a gap pool, reorders it. A row with slot `None` keeps its own values.
pub fn source_rows(symbols: &[u32], keys: &[u64], time: &[i64], slot: &[Option<(Pool, usize)>], block: usize, seed: u64) -> Vec<usize> {
    let mut src: Vec<usize> = (0..symbols.len()).collect();
    let mut rows: Vec<usize> = (0..symbols.len()).filter(|&i| slot[i].is_some()).collect();
    rows.sort_by_key(|&i| (keys[symbols[i] as usize], symbols[i], slot[i], time[i], i));
    let mut lo = 0;
    while lo < rows.len() {
        let (sym, at) = (symbols[rows[lo]], slot[rows[lo]]);
        let mut hi = lo + 1;
        while hi < rows.len() && symbols[rows[hi]] == sym && slot[rows[hi]] == at {
            hi += 1;
        }
        let series = &rows[lo..hi];
        let (pool, seg) = at.expect("moving rows have a slot");
        let mut parts = vec![seed, keys[sym as usize], seg as u64, SEED_TAG];
        if pool == Pool::Gap {
            parts.push(GAP_TAG);
        }
        let mut rng = Mt::new(mix_seed(&parts));
        for (k, &o) in block_permutation(series.len(), block, &mut rng).iter().enumerate() {
            src[series[k]] = series[o];
        }
        lo = hi;
    }
    src
}

/// The label of a family column: `lbl_<label>_<kind>` of [`labels::COLUMN_KINDS`], else `lbl_<label>_xs`.
fn family_label(name: &str) -> Option<&str> {
    labels::parse_column(name)
        .map(|(l, _)| l)
        .or_else(|| labels::is_xs_column(name).then(|| &name[labels::COLUMN_PREFIX.len()..name.len() - 3]))
}

/// Whether a target spec entry belongs to `label`: its column or its return column is in the family.
fn names_label(s: &TargetSpec, label: &str) -> bool {
    s.name == label || family_label(&s.column) == Some(label) || s.return_column.as_deref().and_then(family_label) == Some(label)
}

/// Every column a target spec entry names.
fn spec_columns(s: &TargetSpec) -> Vec<String> {
    let mut v = vec![s.column.clone()];
    v.extend(s.return_column.clone());
    v.extend(s.construction_columns.iter().cloned());
    v.extend(s.scale_column().map(str::to_string));
    v
}

/// Whether column `name` is in the family of `label`: a standard kind of the label, its
/// `_xs` column, or a `lbl_<label>_` column the target spec names that is no other label's.
fn in_family(name: &str, label: &str, named: &BTreeSet<String>) -> bool {
    match family_label(name) {
        Some(l) => l == label,
        None => name.starts_with(&format!("{}{}_", labels::COLUMN_PREFIX, label)) && named.contains(name),
    }
}

fn read_f64(src: &FileStore, col: usize) -> Result<Vec<f64>, Fail> {
    let mut out = Vec::with_capacity(src.n_rows());
    for g in 0..src.n_groups() {
        out.extend(src.read_f64_group(g, col).map_err(|e| (3, e))?);
    }
    Ok(out)
}

fn read_time(src: &FileStore, col: usize) -> Result<Vec<i64>, Fail> {
    let h = src.header();
    let mut out = Vec::with_capacity(src.n_rows());
    for g in 0..src.n_groups() {
        match h.dtypes()[col] {
            DType::I64 => out.extend(src.read_i64(g, col).map_err(|e| (3, e))?),
            DType::F64 => {
                for v in src.read_f64_group(g, col).map_err(|e| (3, e))? {
                    if !v.is_finite() || v.fract() != 0.0 {
                        return input(format!("{}: time column '{}' holds a value that is not whole seconds", src.path(), h.names()[col]));
                    }
                    out.push(v as i64);
                }
            }
            DType::U32 => return usage(format!("{}: time column '{}' is text; it needs epoch seconds", src.path(), h.names()[col])),
        }
    }
    Ok(out)
}

/// Symbol id per row and the symbol names by id.
fn read_symbols(src: &FileStore, col: usize) -> Result<(Vec<u32>, Vec<String>), Fail> {
    let h = src.header();
    if let Some(levels) = h.levels(col) {
        let levels = levels.to_vec();
        let mut ids = Vec::with_capacity(src.n_rows());
        for g in 0..src.n_groups() {
            for c in src.read_codes(g, col).map_err(|e| (3, e))? {
                if c as usize >= levels.len() {
                    return input(format!("{}: symbol code {} has no level", src.path(), c));
                }
                ids.push(c);
            }
        }
        return Ok((ids, levels));
    }
    let values: Vec<String> = match h.dtypes()[col] {
        DType::I64 => (0..src.n_groups())
            .map(|g| src.read_i64(g, col).map_err(|e| (3, e)))
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .map(|v| v.to_string())
            .collect(),
        _ => read_f64(src, col)?.into_iter().map(|v| format!("{}", v)).collect(),
    };
    let mut names: Vec<String> = values.clone();
    names.sort();
    names.dedup();
    let ids = values.iter().map(|v| names.binary_search(v).unwrap_or(0) as u32).collect();
    Ok((ids, names))
}

/// The `lock_record` declaration a source sidecar carries, as a record.
fn carried_record(data: &str) -> Result<Option<LockRecord>, Fail> {
    for ext in ["schema.json", "manifest.json", "enrich.json", "lockbox.json"] {
        let file = format!("{}.{}", data, ext);
        let Ok(text) = std::fs::read_to_string(&file) else { continue };
        let Some(obj) = manifest::object(&text, "lock_record") else { continue };
        let (start, end) = (lockbox::int_field(obj, "start"), lockbox::int_field(obj, "end"));
        let sha = manifest::string_field(obj, "record_sha256");
        return match (start, end, sha) {
            (Some(start), Some(end), Some(sha256)) if end > start => Ok(Some(LockRecord { range: Lockbox { start, end }, sha256 })),
            _ => input(format!("{}: unreadable lock_record declaration (needs integer start < end and record_sha256)", file)),
        };
    }
    Ok(None)
}

fn exists(p: &str) -> bool {
    Path::new(p).exists()
}

/// Build the null store of `o`.
pub fn permute(o: &Options) -> Result<Summary, Fail> {
    if o.block == 0 {
        return usage("--block must be at least 1 label row");
    }
    if o.label.is_empty() {
        return usage("--label must name a label (the <name> of lbl_<name>_y)");
    }
    if Path::new(&o.out) == Path::new(&o.data) {
        return usage("--out must differ from --data");
    }
    if !store::is_store_path(&o.data) {
        return usage(format!("--data {}: needs a column store (.rcsb)", o.data));
    }
    let targets = target::sidecar_path(&o.out).ok_or((2, "--out needs a file path".to_string()))?;
    let out_files = [o.out.clone(), targets.clone(), format!("{}.manifest.json", o.out), format!("{}.schema.json", o.out), format!("{}.lockbox.json", o.out), format!("{}.enrich.json", o.out), format!("{}.labels.json", o.out)];
    if let Some(f) = out_files.iter().find(|f| exists(f)) {
        return usage(format!("{} exists; a null store is written to a fresh path (remove the old store and its sidecars first)", f));
    }

    let spec_text = std::fs::read_to_string(&o.target_spec).map_err(|e| (3, format!("{}: {}", o.target_spec, e)))?;
    let specs = target::parse_file(&spec_text).map_err(|e| (3, format!("{}: {}", o.target_spec, e)))?;
    let entries: Vec<&TargetSpec> = specs.iter().filter(|s| names_label(s, &o.label)).collect();
    if entries.is_empty() {
        return usage(format!("{}: no target spec entry names the label '{}' (lbl_{}_*)", o.target_spec, o.label, o.label));
    }
    let purge = entries.iter().map(|s| s.required_purge_seconds()).fold(0.0, f64::max).ceil() as i64;
    let named: BTreeSet<String> = entries.iter().flat_map(|s| spec_columns(s)).collect();

    let src = FileStore::open(&o.data).map_err(|e| (3, e))?;
    let h = src.header();
    if h.version != store::VERSION {
        return input(format!("{}: permute-labels needs a version-{} store (this one is version {}); rewrite it with `rcsr import`", o.data, store::VERSION, h.version));
    }
    let names = h.names();
    let col = |n: &str| names.iter().position(|x| x == n);
    let t_col = col(&o.time).ok_or_else(|| (2, format!("{}: no time column '{}' (--time)", o.data, o.time)))?;
    let s_col = col(&o.group_by).ok_or_else(|| (2, format!("{}: no symbol column '{}' (--group-by)", o.data, o.group_by)))?;
    let y_name = labels::column_name(&o.label, "y");
    let y_col = col(&y_name).ok_or_else(|| (2, format!("{}: no column '{}': the rows that take part are the rows with a finite y", o.data, y_name)))?;
    let family: Vec<usize> = (0..names.len()).filter(|&c| in_family(&names[c], &o.label, &named)).collect();
    let fixed = [labels::column_name(&o.label, "t1"), labels::column_name(&o.label, "weight")];
    let moved: Vec<usize> = family.iter().copied().filter(|&c| !fixed.contains(&names[c])).collect();
    if let Some(&c) = moved.iter().find(|&&c| h.dtypes()[c] != DType::F64 || !h.numeric()[c]) {
        return input(format!("{}: label column '{}' is not numeric f64", o.data, names[c]));
    }
    let keep: Vec<usize> = match &o.columns {
        None => (0..names.len()).collect(),
        Some(list) => {
            let missing: Vec<&str> = list.iter().filter(|c| col(c).is_none()).map(String::as_str).collect();
            if !missing.is_empty() {
                return usage(format!("--columns: {} has no column {}", o.data, missing.join(", ")));
            }
            let want: BTreeSet<usize> = [t_col, s_col]
                .into_iter()
                .chain(list.iter().filter_map(|c| col(c)))
                .chain(family.iter().copied())
                .chain(named.iter().filter_map(|c| col(c)))
                .collect();
            want.into_iter().collect()
        }
    };

    let time = read_time(&src, t_col)?;
    let (symbols, levels) = read_symbols(&src, s_col)?;
    let y = read_f64(&src, y_col)?;
    // The book prices every row with a finite return: the return columns of the label's
    // entries that move, else lbl_<label>_fwd.
    let mut ret_cols: Vec<usize> = entries.iter().filter_map(|s| s.return_column.as_deref().and_then(|r| col(r))).filter(|c| moved.contains(c)).collect();
    if ret_cols.is_empty() {
        ret_cols.extend(col(&labels::column_name(&o.label, "fwd")).filter(|c| moved.contains(c)));
    }
    ret_cols.sort_unstable();
    ret_cols.dedup();
    let rets: Vec<Vec<f64>> = ret_cols.iter().map(|&c| read_f64(&src, c)).collect::<Result<_, _>>()?;
    let priced = |i: usize| rets.iter().all(|r| r[i].is_finite());

    let resolved = lockbox::resolve(o.lockbox.as_deref(), &o.data).map_err(|e| (e.exit, e.message))?;
    let carried = carried_record(&o.data)?;
    let record = match o.lock_record.as_deref() {
        Some(p) => {
            let r = LockRecord::load(p).map_err(|e| (e.exit, e.message))?;
            if let Some(c) = carried.as_ref().filter(|c| c.sha256 != r.sha256) {
                return usage(format!(
                    "--lock-record {} (sha256 {}) is not the lock record {} declares (sha256 {})",
                    p, r.sha256, o.data, c.sha256
                ));
            }
            Some(r)
        }
        None => carried,
    };
    let (lo, hi) = (time.iter().min().copied().unwrap_or(0), time.iter().max().copied().unwrap_or(0));
    let epoch = lockbox::plausible_epoch_seconds(lo as f64, hi as f64);
    let mut ranges: Vec<(Lockbox, String)> = Vec::new();
    if resolved.source != Source::ProtocolDefault || epoch {
        ranges.extend(resolved.ranges().into_iter().map(|r| (r, resolved.source.as_str().to_string())));
    }
    if let Some(r) = &record {
        ranges.push((r.range, "lock record".to_string()));
    }
    for (r, origin) in &ranges {
        let inside = time.iter().filter(|&&t| r.contains(t)).count();
        if inside > 0 {
            return input(format!(
                "{}: {} row(s) lie inside the locked range {} ({}); a null store never holds a locked row: build it from a development store that ends before the range",
                o.data,
                inside,
                lockbox::span(r),
                origin
            ));
        }
    }
    let embargo = if resolved.range.is_some() { resolved.embargo.seconds } else { crate::protocol::DEFAULT_LOCKBOX_EMBARGO_SECONDS };
    let bounds: Vec<Lockbox> = ranges.iter().map(|(r, _)| *r).collect();
    let segs = Segments::with_embargo(&bounds, purge, embargo);
    let mut guarded_rows = 0usize;
    let slot: Vec<Option<(Pool, usize)>> = (0..time.len())
        .map(|i| {
            let pool = match (y[i].is_finite(), priced(i), rets.is_empty()) {
                (true, true, _) => Pool::Label,
                (false, true, false) => Pool::Gap,
                _ => return None,
            };
            let raw = segs.raw(time[i] as f64);
            if segs.kind(raw) == SegmentKind::Development {
                Some((pool, raw))
            } else {
                guarded_rows += 1;
                None
            }
        })
        .collect();
    let mut pools = vec![[0usize; 2]; levels.len()];
    for (i, s) in slot.iter().enumerate() {
        if let Some((pool, _)) = s {
            pools[symbols[i] as usize][(*pool == Pool::Gap) as usize] += 1;
        }
    }
    let moved_rows: usize = pools.iter().map(|p| p[0] + p[1]).sum();
    let gap_rows: usize = pools.iter().map(|p| p[1]).sum();
    let keys: Vec<u64> = levels.iter().map(|s| symbol_key(s)).collect();
    let from = source_rows(&symbols, &keys, &time, &slot, o.block, o.seed);
    drop((time, y, slot, rets));

    let permuted: Vec<Vec<f64>> = moved
        .iter()
        .map(|&c| read_f64(&src, c).map(|v| from.iter().map(|&s| v[s]).collect()))
        .collect::<Result<_, _>>()?;
    let codes: Vec<Option<Vec<u32>>> = moved
        .iter()
        .map(|&c| -> Result<Option<Vec<u32>>, Fail> {
            if h.levels(c).is_none() {
                return Ok(None);
            }
            let mut all = Vec::with_capacity(src.n_rows());
            for g in 0..src.n_groups() {
                all.extend(src.read_codes(g, c).map_err(|e| (3, e))?);
            }
            Ok(Some(from.iter().map(|&s| all[s]).collect()))
        })
        .collect::<Result<_, Fail>>()?;
    let source_sha256 = manifest::sha256_file(&o.data).map_err(|e| (3, e))?;
    let columns: Vec<String> = moved.iter().map(|&c| names[c].clone()).collect();
    let null = format!(
        "{{\"label\":{},\"block\":{},\"seed\":{},\"order\":\"{}\",\"source_sha256\":{},\"target_spec_sha256\":{},\"columns\":[{}],\"return_columns\":[{}],\"symbols\":{},\"moved_rows\":{},\"gap_rows\":{},\"guarded_rows\":{},\"pools\":[{}],\"purge_seconds\":{},\"embargo_seconds\":{}}}",
        jstr(&o.label),
        o.block,
        o.seed,
        ORDER,
        jstr(&source_sha256),
        jstr(&manifest::sha256_hex(spec_text.as_bytes())),
        columns.iter().map(|c| jstr(c)).collect::<Vec<_>>().join(","),
        ret_cols.iter().map(|&c| jstr(&names[c])).collect::<Vec<_>>().join(","),
        levels.len(),
        moved_rows,
        gap_rows,
        guarded_rows,
        levels
            .iter()
            .zip(&pools)
            .map(|(s, p)| format!("{{\"symbol\":{},\"label_rows\":{},\"gap_rows\":{}}}", jstr(s), p[0], p[1]))
            .collect::<Vec<_>>()
            .join(","),
        purge,
        embargo
    );
    if let Some(dir) = Path::new(&o.out).parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir).map_err(|e| (3, format!("{}: {}", dir.display(), e)))?;
    }
    let output_sha256 = write_store(&o.out, &src, &keep, &moved, &permuted, &codes, &null)?;

    let (declared, also) = match resolved.source {
        Source::Store | Source::Flag | Source::FlagAndDefault => (resolved.range, resolved.also),
        Source::ProtocolDefault | Source::None => (None, None),
    };
    let (declared, also) = match &record {
        Some(r) => crate::lockrec::with_record(declared, also, &r.range).map_err(|e| (e.exit, e.message))?,
        None => (declared, also),
    };
    let floor = lockbox::declared_embargo(&o.data).map_err(|e| (e.exit, e.message))?.map(|(_, s)| s);
    let mut text = format!(
        "{{\"schema\":\"{}\",\"tool\":\"{}\",\"version\":{},\"source\":{},\"target_spec\":{},\"null\":{},\"output_sha256\":{}}}\n",
        SCHEMA,
        WRITER,
        jstr(env!("CARGO_PKG_VERSION")),
        jstr(&o.data),
        jstr(&o.target_spec),
        null,
        jstr(&output_sha256)
    );
    let embargo_text = floor.map(|s| lockbox::embargo_json(s, "(input)", crate::enrich::EMBARGO_RULE, 0));
    let origin = format!("{} ({})", WRITER, resolved.source.as_str());
    match (&declared, &embargo_text) {
        (Some(d), _) => text = lockbox::insert_declaration(&text, d, also.as_ref(), embargo_text.as_deref(), &origin),
        (None, Some(e)) => text = lockbox::insert_embargo(&text, e),
        (None, None) => {}
    }
    if let Some(r) = &record {
        text = r.insert_declaration(&text, 0);
    }
    let manifest_path = format!("{}.manifest.json", o.out);
    crate::atomic::write_text_atomic(&manifest_path, &text).map_err(|e| (3, e))?;
    crate::atomic::write_text_atomic(&targets, &spec_text).map_err(|e| (3, e))?;
    let source_schema = format!("{}.schema.json", o.data);
    let schema = if exists(&source_schema) {
        let old = std::fs::read_to_string(&source_schema).map_err(|e| (3, format!("{}: {}", source_schema, e)))?;
        let version = manifest::string_field(&old, "panel_version").unwrap_or_else(|| crate::registry::DEFAULT_PANEL_VERSION.to_string());
        let kept: Vec<String> = keep.iter().map(|&c| names[c].clone()).collect();
        let floors = crate::cross_floor::resolved_columns(&o.data, &kept);
        Some(
            crate::registry::write_schema_floors(&o.out, &version, declared.as_ref(), also.as_ref(), floor, None, record.as_ref(), &floors)
                .map_err(|e| (3, e))?,
        )
    } else {
        None
    };
    Ok(Summary {
        output_sha256,
        source_sha256,
        columns,
        symbols: levels.len(),
        moved_rows,
        gap_rows,
        guarded_rows,
        manifest: manifest_path,
        targets,
        schema,
    })
}

/// Copy the `keep` columns of `src` to `out` group by group; the `moved` columns take
/// `values` (and `codes` when they carry a dictionary), and the header carries `null` as
/// `null_store`. Returns the SHA-256 of the file.
fn write_store(out: &str, src: &FileStore, keep: &[usize], moved: &[usize], values: &[Vec<f64>], codes: &[Option<Vec<u32>>], null: &str) -> Result<String, Fail> {
    let h = src.header();
    let plans: Vec<ColumnPlan> = keep
        .iter()
        .map(|&c| {
            let info = h.column(c).expect("column index from the header");
            ColumnPlan { name: info.name, numeric: info.numeric, dtype: info.dtype, levels: info.levels, has_codes: info.has_codes }
        })
        .collect();
    let tmp = format!("{}.tmp-{}", out, std::process::id());
    let group_rows = h.group_rows;
    let written = store::write_store_grouped_marked(&tmp, &plans, h.n_rows, group_rows, Some(null), |lo, hi, cols| {
        let g = lo / group_rows;
        for (k, &c) in keep.iter().enumerate() {
            if let Some(m) = moved.iter().position(|&x| x == c) {
                cols[k].f64s.extend_from_slice(&values[m][lo..hi]);
                if let Some(cd) = &codes[m] {
                    cols[k].codes.extend_from_slice(&cd[lo..hi]);
                }
                continue;
            }
            match plans[k].dtype {
                DType::F64 => cols[k].f64s.extend(src.read_f64_group(g, c)?),
                DType::I64 => cols[k].i64s.extend(src.read_i64(g, c)?),
                DType::U32 => {}
            }
            if plans[k].has_codes {
                cols[k].codes.extend(src.read_codes(g, c)?);
            }
        }
        Ok(())
    });
    let done = written.and_then(|_| std::fs::rename(&tmp, out).map_err(|e| format!("{}: {}", out, e)));
    if let Err(e) = done {
        let _ = std::fs::remove_file(&tmp);
        return input(e);
    }
    manifest::sha256_file(out).map_err(|e| (3, e))
}

/// The `null` object of a null store at `data`, as JSON text: the `null_store` member of
/// its store header (#1304), else the `null` object of `<data>.manifest.json` (a null
/// store written before #1304); `None` for any other data.
pub fn marker(data: &str) -> Option<String> {
    if data.is_empty() || data == "-" {
        return None;
    }
    if let Some(m) = FileStore::open(data).ok().and_then(|s| s.header().null_store().map(str::to_string)) {
        return Some(m);
    }
    let text = std::fs::read_to_string(format!("{}.manifest.json", data)).ok()?;
    let j = Json::parse(text.trim())?;
    if j.get("schema").and_then(Json::as_str) != Some(SCHEMA) {
        return None;
    }
    j.get("null").map(crate::cpcv::to_json)
}

/// `null_store label L, seed S` of a parsed marker.
pub fn describe(m: &Json) -> String {
    format!(
        "null_store label {}, seed {}",
        m.get("label").and_then(Json::as_str).unwrap_or("?"),
        m.get("seed").and_then(Json::as_f64).map_or("?".to_string(), crate::walkforward::jnum)
    )
}

/// #1304: why `walkforward --ledger` (and so `--locked-test`) refuses the null store `data`
/// with marker `m`.
pub fn ledger_refusal(data: &str, m: &str, locked_test: bool) -> String {
    refusal(
        data,
        m,
        if locked_test {
            "a null store is never a protocol candidate, and a sealed read of it would spend a lockbox read"
        } else {
            "a null store is never a protocol candidate: drop --ledger"
        },
    )
}

/// `<data> is a null store (null_store label L, seed S); <why>` for the null store `data`
/// with marker `m`.
pub fn refusal(data: &str, m: &str, why: &str) -> String {
    let what = Json::parse(m).map_or_else(|| "null_store".to_string(), |j| describe(&j));
    format!("{} is a null store ({}); {}", data, what, why)
}

/// A report JSON text with a `null_store` member holding `marker`; unchanged without one.
pub fn mark_report(text: String, marker: Option<&str>) -> String {
    let (Some(m), Some(end)) = (marker, text.rfind('}')) else { return text };
    format!("{},\"null_store\":{}{}", &text[..end], m, &text[end..])
}

pub fn usage_text() -> &'static str {
    "rcsr panel permute-labels --data panel.rcsb --target-spec panel.targets.json --label NAME --block ROWS --seed N --out null.rcsb
                [--columns a,b,...] [--time ts] [--group-by symbol] [--lockbox S,E|none] [--lock-record R]
  The search-capacity null store (#1272). Every value column of the label family lbl_NAME_* (y, fwd,
  long_net, action, exit_ret, m_net, m_barrier and any lbl_NAME_ column the spec names, never t1 or
  weight) moves per symbol by one seeded circular-block order (blocks of --block rows; a random rotation,
  then the blocks shuffled) over the rows with a finite lbl_NAME_y and a finite return (the label pool).
  Rows with a NaN y and a finite return (the label's volatility warm-up, gaps inside its window) move the
  same way in a gap pool of their own with an independent order, so the null book prices the rows of the
  real book with permuted returns. Time, symbol, t1, weight, every feature column and every row without
  a finite return keep their bytes. The order of a pool
  comes from the seed, the symbol name and the pool only. --columns keeps only the named features plus
  time, symbol, the family and the columns the label's spec entries name. <out stem>.targets.json is the
  spec file unchanged; <out>.manifest.json (schema rcsr-null-store/1) records null {label, block, seed,
  order per_symbol, source_sha256, columns, pools per symbol: label_rows, gap_rows} and carries the
  locked-test declaration and the lock record of the source (--lock-record, or the lock_record object of
  a source sidecar). The store header carries the same null object as null_store, so a copy of the store
  bytes stays marked, as does a store appended from it (rcsr enrich, rcsr panel append and rank); every
  other store writer starts unmarked. A source row inside a locked range exits 3;
  rows in a label purge or an embargo neither move nor give a value. A walkforward report priced on the
  output carries null_store; walkforward --ledger (and so --locked-test), ladder --locked-test and
  protocol regimes --after-read on the output exit 2, and rcsr trials reads the report only with --table
  --null-wf. --out and its sidecars must not exist (exit 2).
"
}

fn fail(msg: &str, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code);
}

pub fn cli(args: &[String]) -> ! {
    let mut opts = Options {
        data: String::new(),
        target_spec: String::new(),
        label: String::new(),
        block: 0,
        seed: 0,
        columns: None,
        out: String::new(),
        time: "ts".to_string(),
        group_by: "symbol".to_string(),
        lockbox: None,
        lock_record: None,
    };
    let (mut block, mut seed) = (None, None);
    let mut i = 0usize;
    while i < args.len() {
        let flag = args[i].as_str();
        if matches!(flag, "-h" | "--help") {
            print!("{}", usage_text());
            std::process::exit(0);
        }
        let v = args.get(i + 1).cloned().unwrap_or_else(|| fail(&format!("{} requires a value", flag), 2));
        match flag {
            "--data" => opts.data = v,
            "--target-spec" => opts.target_spec = v,
            "--label" => opts.label = v,
            "--block" => block = Some(v.parse::<usize>().unwrap_or_else(|_| fail(&format!("--block: '{}' is not a whole number of rows", v), 2))),
            "--seed" => seed = Some(v.parse::<u64>().unwrap_or_else(|_| fail(&format!("--seed: '{}' is not a whole number", v), 2))),
            "--columns" => opts.columns = Some(v.split(',').map(|c| c.trim().to_string()).filter(|c| !c.is_empty()).collect()),
            "--out" => opts.out = v,
            "--time" => opts.time = v,
            "--group-by" => opts.group_by = v,
            "--lockbox" => opts.lockbox = Some(v),
            "--lock-record" => opts.lock_record = Some(v),
            other => fail(&format!("unknown option '{}' for `rcsr panel permute-labels`\n\n{}", other, usage_text()), 2),
        }
        i += 2;
    }
    for (flag, missing) in [("--data", opts.data.is_empty()), ("--target-spec", opts.target_spec.is_empty()), ("--label", opts.label.is_empty()), ("--out", opts.out.is_empty())] {
        if missing {
            fail(&format!("`rcsr panel permute-labels` needs {}", flag), 2);
        }
    }
    opts.block = block.unwrap_or_else(|| fail("`rcsr panel permute-labels` needs --block: the block length depends on the horizon and is pre-registered", 2));
    opts.seed = seed.unwrap_or_else(|| fail("`rcsr panel permute-labels` needs --seed", 2));
    match permute(&opts) {
        Ok(s) => {
            eprintln!(
                "permute-labels: {} column(s) [{}] over {} symbol(s), {} row(s) moved ({} in gap pools), {} row(s) kept in a purge or embargo band; source sha256 {}; output sha256 {}; manifest {}; targets {}{}",
                s.columns.len(),
                s.columns.join(","),
                s.symbols,
                s.moved_rows,
                s.gap_rows,
                s.guarded_rows,
                s.source_sha256,
                s.output_sha256,
                s.manifest,
                s.targets,
                s.schema.map(|p| format!("; schema {}", p)).unwrap_or_default()
            );
            std::process::exit(0);
        }
        Err((code, e)) => fail(&e, code),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_rows_permute_each_symbol_pool_and_segment_alone() {
        // Two symbols interleaved in time order, a guarded row, a row of another segment and gap rows.
        let symbols = [0u32, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1];
        let time: Vec<i64> = (0..14).map(|i| 100 + (i / 2) as i64).collect();
        let (l, g) = (Pool::Label, Pool::Gap);
        let slot = [
            Some((g, 0)),
            Some((g, 0)),
            Some((g, 0)),
            Some((g, 0)),
            Some((l, 0)),
            Some((l, 0)),
            None,
            Some((l, 0)),
            Some((l, 2)),
            Some((l, 0)),
            Some((l, 0)),
            Some((l, 0)),
            Some((l, 0)),
            Some((l, 0)),
        ];
        let keys = [symbol_key("A"), symbol_key("B")];
        let src = source_rows(&symbols, &keys, &time, &slot, 2, 7);
        assert_eq!(src[6], 6);
        assert_eq!(src[8], 8);
        for (i, &s) in src.iter().enumerate() {
            assert_eq!(symbols[s], symbols[i]);
            assert_eq!(slot[s], slot[i]);
        }
        let mut sorted = src.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..14).collect::<Vec<_>>());
    }

    #[test]
    fn a_symbol_order_does_not_depend_on_the_other_symbols() {
        let time: Vec<i64> = (0..40).collect();
        let label = Some((Pool::Label, 0));
        let alone = source_rows(&[0; 40], &[symbol_key("A")], &time, &[label; 40], 4, 3);
        let symbols: Vec<u32> = (0..80).map(|i| (i % 2) as u32).collect();
        let t2: Vec<i64> = (0..80).map(|i| (i / 2) as i64).collect();
        let mixed = source_rows(&symbols, &[symbol_key("Z"), symbol_key("A")], &t2, &[label; 80], 4, 3);
        let a_rows: Vec<usize> = (0..80).filter(|i| i % 2 == 1).collect();
        let got: Vec<usize> = a_rows.iter().map(|&i| a_rows.iter().position(|&r| r == mixed[i]).unwrap()).collect();
        assert_eq!(got, alone);
    }

    #[test]
    fn the_gap_pool_order_does_not_follow_the_label_pool_order() {
        let time: Vec<i64> = (0..40).collect();
        let label = source_rows(&[0; 40], &[symbol_key("A")], &time, &[Some((Pool::Label, 0)); 40], 4, 3);
        let gap = source_rows(&[0; 40], &[symbol_key("A")], &time, &[Some((Pool::Gap, 0)); 40], 4, 3);
        assert_ne!(label, gap);
    }

    /// #1304: the header member first, the manifest of a null store written before #1304
    /// second, nothing for real data or stdin.
    #[test]
    fn the_marker_comes_from_the_header_then_the_manifest() {
        let dir = std::env::temp_dir().join(format!("rcsr-1304-marker-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let plans = [ColumnPlan { name: "x".to_string(), numeric: true, dtype: DType::F64, levels: vec![], has_codes: false }];
        let write = |name: &str, m: Option<&str>| -> String {
            let p = dir.join(name).to_string_lossy().into_owned();
            store::write_store_grouped_marked(&p, &plans, 3, 2, m, |lo, hi, cols| {
                cols[0].f64s.extend((lo..hi).map(|r| r as f64));
                Ok(())
            })
            .unwrap();
            p
        };
        let header = "{\"label\":\"h24\",\"seed\":3}";
        let (marked, plain) = (write("marked.rcsb", Some(header)), write("plain.rcsb", None));
        assert_eq!(marker(&marked).as_deref(), Some(header));
        assert_eq!(marker(&plain), None);
        let legacy = format!("{{\"schema\":\"{}\",\"null\":{{\"label\":\"h1\",\"seed\":9}}}}\n", SCHEMA);
        std::fs::write(format!("{}.manifest.json", plain), &legacy).unwrap();
        assert_eq!(marker(&plain).as_deref(), Some("{\"label\":\"h1\",\"seed\":9}"));
        std::fs::write(format!("{}.manifest.json", marked), &legacy).unwrap();
        assert_eq!(marker(&marked).as_deref(), Some(header), "the header member wins: it is in the hashed bytes");
        std::fs::write(format!("{}.manifest.json", plain), "{\"schema\":\"rcsr-panel/1\",\"null\":{}}\n").unwrap();
        assert_eq!(marker(&plain), None, "a manifest of another schema marks nothing");
        assert_eq!(marker("-"), None);
        assert_eq!(marker(""), None);
        let e = ledger_refusal("n.rcsb", header, true);
        assert!(e.contains("n.rcsb is a null store (null_store label h24, seed 3)") && e.contains("lockbox read"), "{e}");
        assert!(ledger_refusal("n.rcsb", header, false).ends_with("drop --ledger"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_report_marker_goes_inside_the_last_brace() {
        assert_eq!(mark_report("{\"a\":1}\n".to_string(), Some("{\"seed\":1}")), "{\"a\":1,\"null_store\":{\"seed\":1}}\n");
        assert_eq!(mark_report("{\"a\":1}".to_string(), None), "{\"a\":1}");
    }
}
