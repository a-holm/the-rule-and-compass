//! #493: `rcsr panel rank`, the cross-sectional rank target of a time-major panel.
//!
//! Per timestamp, the rank of a column among the panel rows (the PIT members at
//! that timestamp), mapped to `[-1, 1]`. A rank reads the values of one
//! timestamp only. The panel's existing blocks are copied unchanged, the input
//! is never opened for writing, and the target spec entry
//! (`kind = "xs_rank"`) joins `<out stem>.targets.json`.

use crate::manifest;
use crate::registry;
use crate::store::{self, AppendColumn, DType, FileStore};
use crate::target::{self, Scale, TargetSpec};
use std::io::Write;
use std::path::{Path, PathBuf};

pub const DEFAULT_MIN_MEMBERS: usize = 20;

pub struct RankOptions {
    /// Time-major panel store.
    pub data: String,
    /// Column to rank (the forward return).
    pub column: String,
    /// Name of the appended column, `lbl_<label>_xs`.
    pub name: String,
    pub out: String,
    pub min_members: usize,
    /// Spec file that holds the entry of `column`; default `<data stem>.targets.json`.
    pub target_spec: Option<String>,
    pub panel_version: String,
    pub scratch: Option<String>,
    pub lockbox: Option<String>,
}

#[derive(Debug)]
pub struct RankReport {
    pub input_sha256_before: String,
    pub input_sha256_after: String,
    pub output_sha256: String,
    pub timestamps: usize,
    /// Timestamps with fewer than `min_members` finite values.
    pub thin_timestamps: usize,
    pub ranked_rows: usize,
    pub schema: String,
    pub targets: String,
}

/// Rank of each value among the finite values with the same time stamp,
/// mapped to `[-1, 1]` by `r = 2 * (rank - 1) / (n - 1) - 1`. Ties take the
/// average rank. A row is `NaN` when its value is not finite or when its time
/// stamp has fewer than `max(min_members, 2)` finite values. Returns the ranks
/// and the number of timestamps below `min_members`.
pub fn xs_rank(values: &[f64], time: &[i64], min_members: usize) -> (Vec<f64>, usize) {
    let mut out = vec![f64::NAN; values.len()];
    let mut order: Vec<usize> = (0..values.len().min(time.len())).filter(|&i| values[i].is_finite()).collect();
    order.sort_by(|&a, &b| time[a].cmp(&time[b]).then_with(|| values[a].total_cmp(&values[b])).then_with(|| a.cmp(&b)));
    let need = 2;
    let mut thin = 0usize;
    let mut start = 0usize;
    while start < order.len() {
        let mut end = start + 1;
        while end < order.len() && time[order[end]] == time[order[start]] {
            end += 1;
        }
        let n = end - start;
        if n < need {
            thin += 1;
        } else {
            let mut i = start;
            while i < end {
                let mut j = i;
                while j + 1 < end && values[order[j + 1]] == values[order[i]] {
                    j += 1;
                }
                let rank = ((i - start) + (j - start)) as f64 / 2.0 + 1.0;
                let r = 2.0 * (rank - 1.0) / (n as f64 - 1.0) - 1.0;
                for &k in &order[i..=j] {
                    out[k] = r;
                }
                i = j + 1;
            }
        }
        start = end;
    }
    (out, thin)
}

fn column_index(store: &FileStore, name: &str) -> Result<usize, String> {
    store
        .header()
        .column_index(name)
        .ok_or_else(|| format!("{}: no column '{}'", store.path(), name))
}

fn read_ts(store: &FileStore) -> Result<Vec<i64>, String> {
    let t = column_index(store, "ts")?;
    let h = store.header();
    let mut ts = Vec::with_capacity(store.n_rows());
    for g in 0..h.n_groups() {
        if h.dtypes()[t] == DType::I64 {
            ts.extend(store.read_i64(g, t)?);
        } else {
            ts.extend(store.read_f64_group(g, t)?.into_iter().map(|v| v as i64));
        }
    }
    Ok(ts)
}

fn read_column(store: &FileStore, name: &str) -> Result<Vec<f64>, String> {
    let c = column_index(store, name)?;
    if store.header().dtypes()[c] != DType::F64 {
        return Err(format!("{}: column '{}' is not numeric f64", store.path(), name));
    }
    let mut out = Vec::with_capacity(store.n_rows());
    for g in 0..store.header().n_groups() {
        out.extend(store.read_f64_group(g, c)?);
    }
    Ok(out)
}

fn read_specs(path: &str) -> Result<Vec<TargetSpec>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {}", path, e))?;
    target::parse_file(&text).map_err(|e| format!("{}: {}", path, e))
}

/// The xs_rank spec of `name`, derived from the forward-return entry of `column`.
fn xs_spec(name: &str, column: &str, source: &TargetSpec) -> TargetSpec {
    TargetSpec {
        column: name.to_string(),
        name: name.strip_prefix(crate::labels::COLUMN_PREFIX).unwrap_or(name).to_string(),
        kind: target::KIND_XS_RANK.to_string(),
        deprecated: false,
        return_column: Some(column.to_string()),
        return_units: source.return_units.clone(),
        horizon_seconds: source.horizon_seconds,
        entry_offset_seconds: source.entry_offset_seconds,
        delay_seconds: source.delay_seconds,
        entry_price: source.entry_price.clone(),
        exit_price: source.exit_price.clone(),
        offset: 0.0,
        scale: Scale::Constant { value: 1.0 },
        clip: None,
        units: "rank_in_[-1,1]".to_string(),
        cost_bps: source.cost_bps,
        cost_in_label: false,
        construction_columns: vec![column.to_string()],
        note: Some(format!("per-timestamp rank of {} over the panel members", column)),
        long_only: true,
    }
}

pub fn rank_panel(opts: &RankOptions) -> Result<RankReport, String> {
    if opts.min_members < 2 {
        return Err("--min-members must be at least 2".to_string());
    }
    if !crate::labels::is_xs_column(&opts.name) {
        return Err(format!(
            "--name '{}' must look like lbl_<label>_xs: the registry knows the rank column as a label-family lead column",
            opts.name
        ));
    }
    if Path::new(&opts.out) == Path::new(&opts.data) {
        return Err("--out must differ from --data".to_string());
    }
    let resolved = crate::lockbox::resolve(opts.lockbox.as_deref(), &opts.data).map_err(|e| e.message)?;

    let specs_path = match &opts.target_spec {
        Some(p) => p.clone(),
        None => target::sidecar_path(&opts.data).ok_or("rank needs a file panel")?,
    };
    let source_specs = read_specs(&specs_path)?;
    let source = source_specs
        .iter()
        .find(|s| s.kind == "forward_return" && s.return_column.as_deref() == Some(opts.column.as_str()))
        .or_else(|| source_specs.iter().find(|s| s.kind == "forward_return" && s.column == opts.column))
        .ok_or_else(|| {
            format!(
                "{}: no forward_return entry whose return_column or column is '{}'",
                specs_path, opts.column
            )
        })?;
    let spec = xs_spec(&opts.name, &opts.column, source);
    let out_targets = target::sidecar_path(&opts.out).ok_or("rank needs a file output")?;
    let mut specs: Vec<TargetSpec> = if Path::new(&out_targets).is_file() {
        read_specs(&out_targets)?
    } else if Path::new(&target::sidecar_path(&opts.data).unwrap_or_default()).is_file() {
        read_specs(&target::sidecar_path(&opts.data).unwrap_or_default())?
    } else {
        source_specs.clone()
    };
    specs.retain(|s| s.column != spec.column && s.name != spec.name);
    specs.push(spec);

    let before = manifest::sha256_file(&opts.data)?;
    let panel = FileStore::open(&opts.data)?;
    let ts = read_ts(&panel)?;
    let values = read_column(&panel, &opts.column)?;
    let (ranks, thin) = xs_rank(&values, &ts, opts.min_members);
    let mut stamps = ts.clone();
    stamps.sort_unstable();
    stamps.dedup();
    let ranked_rows = ranks.iter().filter(|r| r.is_finite()).count();

    let scratch: PathBuf = match &opts.scratch {
        Some(s) => PathBuf::from(s),
        None => Path::new(&opts.out).parent().map(|p| p.to_path_buf()).unwrap_or_default(),
    };
    std::fs::create_dir_all(&scratch).map_err(|e| format!("{}: {}", scratch.display(), e))?;
    let tag = format!("{}-{}", std::process::id(), &manifest::sha256_hex(opts.out.as_bytes())[..8]);
    let file = scratch.join(format!("rcsr-rank-{}-{}.f64", tag, opts.name));
    let result = (|| -> Result<String, String> {
        let mut w = std::io::BufWriter::new(
            std::fs::File::create(&file).map_err(|e| format!("{}: {}", file.display(), e))?,
        );
        for v in &ranks {
            w.write_all(&v.to_le_bytes()).map_err(|e| format!("{}: {}", file.display(), e))?;
        }
        w.flush().map_err(|e| format!("{}: {}", file.display(), e))?;
        drop(w);
        let extra = AppendColumn { name: opts.name.clone(), file: file.clone(), replace: false };
        store::write_store_appending_hashed(&opts.out, &panel, &[extra])
    })();
    let _ = std::fs::remove_file(&file);
    let output_sha256 = result?;

    let schema = registry::write_schema_declared(&opts.out, &opts.panel_version, resolved.sidecar_range().as_ref())?;
    let body = format!(
        "{{\"schema\":{},\"targets\":[{}]}}\n",
        crate::devlog::text(target::SCHEMA),
        specs.iter().map(|s| format!("{{{}}}", s.file_json())).collect::<Vec<_>>().join(",")
    );
    crate::atomic::write_text_atomic(&out_targets, &body)?;
    let after = manifest::sha256_file(&opts.data)?;
    Ok(RankReport {
        input_sha256_before: before,
        input_sha256_after: after,
        output_sha256,
        timestamps: stamps.len(),
        thin_timestamps: thin,
        ranked_rows,
        schema,
        targets: out_targets,
    })
}

pub fn usage_text() -> &'static str {
    "rcsr panel rank --data panel.rcsb --column lbl_h1_fwd --name lbl_h1_xs --out out.rcsb [--min-members 20]
                [--target-spec FILE] [--lockbox S,E] [--panel-version V] [--scratch dir]
  Per timestamp, the rank of --column among the panel rows (PIT members), mapped to [-1, 1]
  (ties take the average rank). NaN when fewer than --min-members finite values.
  --name must look like lbl_<label>_xs. The target spec entry (kind xs_rank, return_column
  --column) is added to <out stem>.targets.json. --target-spec names the file with the
  forward_return entry of --column (default <data stem>.targets.json).
"
}

fn fail(msg: &str, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code);
}

fn value(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| fail(&format!("{} requires a value", args[*i - 1]), 2))
}

pub fn cli(args: &[String]) -> ! {
    let mut data: Option<String> = None;
    let mut column: Option<String> = None;
    let mut name: Option<String> = None;
    let mut out: Option<String> = None;
    let mut min_members = DEFAULT_MIN_MEMBERS;
    let mut target_spec: Option<String> = None;
    let mut panel_version = registry::DEFAULT_PANEL_VERSION.to_string();
    let mut scratch: Option<String> = None;
    let mut lockbox: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => data = Some(value(args, &mut i)),
            "--column" => column = Some(value(args, &mut i)),
            "--name" => name = Some(value(args, &mut i)),
            "--out" => out = Some(value(args, &mut i)),
            "--min-members" => {
                let v = value(args, &mut i);
                min_members = v.parse().unwrap_or_else(|_| fail(&format!("--min-members: cannot parse '{}'", v), 2));
            }
            "--target-spec" => target_spec = Some(value(args, &mut i)),
            "--panel-version" => panel_version = value(args, &mut i),
            "--scratch" => scratch = Some(value(args, &mut i)),
            "--lockbox" => lockbox = Some(value(args, &mut i)),
            "-h" | "--help" => {
                print!("{}", usage_text());
                std::process::exit(0);
            }
            other => fail(&format!("unknown option '{}' for `rcsr panel rank`", other), 2),
        }
        i += 1;
    }
    let opts = RankOptions {
        data: data.unwrap_or_else(|| fail("`rcsr panel rank` needs --data", 2)),
        column: column.unwrap_or_else(|| fail("`rcsr panel rank` needs --column", 2)),
        name: name.unwrap_or_else(|| fail("`rcsr panel rank` needs --name", 2)),
        out: out.unwrap_or_else(|| fail("`rcsr panel rank` needs --out", 2)),
        min_members,
        target_spec,
        panel_version,
        scratch,
        lockbox,
    };
    match rank_panel(&opts) {
        Ok(r) => {
            eprintln!(
                "ranked {} row(s) over {} timestamp(s) ({} below --min-members); input sha256 {} (unchanged: {}); output sha256 {}; schema {}; targets {}",
                r.ranked_rows,
                r.timestamps,
                r.thin_timestamps,
                r.input_sha256_before,
                r.input_sha256_before == r.input_sha256_after,
                r.output_sha256,
                r.schema,
                r.targets
            );
            std::process::exit(0);
        }
        Err(e) => fail(&e, 3),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64) {
        assert!((a - b).abs() <= 1e-12, "{a} != {b}");
    }

    #[test]
    fn rank_maps_to_unit_interval_with_average_ties() {
        // t=1: 0.3 0.1 0.2 0.4 -> ranks 3 1 2 4
        // t=2: tie of the two middle values -> rank 2.5 each
        // t=3: one NaN leaves 3 finite values
        let time = [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3];
        let v = [0.3, 0.1, 0.2, 0.4, 1.0, 5.0, 5.0, 9.0, 1.0, f64::NAN, 3.0, 2.0];
        let (r, thin) = xs_rank(&v, &time, 3);
        assert_eq!(thin, 0);
        for (got, want) in r[..4].iter().zip([1.0 / 3.0, -1.0, -1.0 / 3.0, 1.0]) {
            close(*got, want);
        }
        for (got, want) in r[4..8].iter().zip([-1.0, 0.0, 0.0, 1.0]) {
            close(*got, want);
        }
        close(r[8], -1.0);
        assert!(r[9].is_nan());
        close(r[10], 1.0);
        close(r[11], 0.0);
    }

    #[test]
    fn rank_is_nan_below_min_members() {
        let time = [1, 1, 1, 2, 2];
        let v = [0.1, 0.2, 0.3, 0.1, 0.2];
        let (r, thin) = xs_rank(&v, &time, 3);
        assert_eq!(thin, 1);
        assert!(r[3].is_nan() && r[4].is_nan());
        close(r[0], -1.0);
        let (r, _) = xs_rank(&[1.0], &[1], 2);
        assert!(r[0].is_nan());
    }

    #[test]
    fn rank_ignores_row_order_and_other_timestamps() {
        let (a, _) = xs_rank(&[0.2, 0.9, 0.1, 5.0, 4.0], &[7, 7, 7, 3, 3], 2);
        let (b, _) = xs_rank(&[4.0, 5.0, 0.1, 0.9, 0.2], &[3, 3, 7, 7, 7], 2);
        close(a[0], b[4]);
        close(a[1], b[3]);
        close(a[2], b[2]);
        close(a[3], b[1]);
        close(a[4], b[0]);
    }
}
