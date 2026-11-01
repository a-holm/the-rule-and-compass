//! #260: append a registered feature family to a time-major panel and write
//! `rcsr panel schema`.
//!
//! `rcsr enrich` needs one contiguous run per symbol, so a family is computed on
//! a symbol-major store (the hourly base store) and carried into the time-major
//! panel by `(symbol, ts)` key. The panel's existing value and code blocks are
//! copied unchanged ([`store::write_store_appending_hashed`]), the input file is
//! never opened for writing, and the output schema records every column hash.

use crate::manifest;
use crate::registry;
use crate::store::{self, AppendColumn, DType, FileStore};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

pub struct AppendOptions {
    /// Time-major panel store (the existing blocks).
    pub target: String,
    /// Symbol-major store that carries the family columns.
    pub family: String,
    pub columns: Vec<String>,
    pub out: String,
    pub panel_version: String,
    /// Directory for the temporary raw columns (default: next to `out`).
    pub scratch: Option<String>,
}

#[derive(Debug)]
pub struct AppendReport {
    pub input_sha256_before: String,
    pub input_sha256_after: String,
    pub output_sha256: String,
    pub appended: Vec<String>,
    /// Panel rows with no row in the family store; their values are NaN.
    pub unmatched_rows: usize,
    pub schema: String,
}

struct Keys {
    ts: Vec<i64>,
    sym: Vec<String>,
}

fn read_keys(store: &FileStore) -> Result<Keys, String> {
    let h = store.header();
    let names = h.names();
    let t = names
        .iter()
        .position(|n| n == "ts")
        .ok_or_else(|| format!("{}: no 'ts' column", store.path()))?;
    let s = names
        .iter()
        .position(|n| n == "symbol")
        .ok_or_else(|| format!("{}: no 'symbol' column", store.path()))?;
    let levels = h.levels(s).map(|l| l.to_vec()).unwrap_or_default();
    let dtypes = h.dtypes();
    let mut ts = Vec::with_capacity(store.n_rows());
    let mut sym = Vec::with_capacity(store.n_rows());
    for g in 0..h.n_groups() {
        if dtypes[t] == DType::I64 {
            ts.extend(store.read_i64(g, t)?);
        } else {
            ts.extend(store.read_f64_group(g, t)?.into_iter().map(|v| v as i64));
        }
        for code in store.read_codes(g, s)? {
            sym.push(levels.get(code as usize).cloned().unwrap_or_default());
        }
    }
    Ok(Keys { ts, sym })
}

fn column_all(store: &FileStore, name: &str) -> Result<Vec<f64>, String> {
    let c = store
        .header()
        .column_index(name)
        .ok_or_else(|| format!("{}: no column '{}'", store.path(), name))?;
    if store.header().dtypes()[c] != DType::F64 {
        return Err(format!("{}: column '{}' is not numeric f64", store.path(), name));
    }
    let mut out = Vec::with_capacity(store.n_rows());
    for g in 0..store.header().n_groups() {
        out.extend(store.read_f64_group(g, c)?);
    }
    Ok(out)
}

/// Append `opts.columns` from the family store to the panel.
pub fn append_family(opts: &AppendOptions) -> Result<AppendReport, String> {
    if opts.columns.is_empty() {
        return Err("append needs at least one --columns name".to_string());
    }
    for c in &opts.columns {
        match registry::lookup(c) {
            Some(r) if r.available => {}
            Some(r) => {
                return Err(format!(
                    "column '{}' is registered in family '{}' but unavailable: {}",
                    c,
                    r.family,
                    r.reason.unwrap_or_default()
                ))
            }
            None => {
                let near = registry::nearest(c)
                    .map(|(n, f)| format!(" (nearest registry name '{}', family '{}')", n, f))
                    .unwrap_or_default();
                return Err(format!("unknown column '{}': not in the feature registry{}", c, near));
            }
        }
    }
    let before = manifest::sha256_file(&opts.target)?;
    let target = FileStore::open(&opts.target)?;
    let family = FileStore::open(&opts.family)?;
    let tk = read_keys(&target)?;
    let fk = read_keys(&family)?;

    let mut index: HashMap<(&str, i64), u32> = HashMap::with_capacity(fk.ts.len());
    for (i, (s, t)) in fk.sym.iter().zip(&fk.ts).enumerate() {
        if index.insert((s.as_str(), *t), i as u32).is_some() {
            return Err(format!(
                "{}: duplicate row for ({}, {}); the family store must have one row per key",
                opts.family, s, t
            ));
        }
    }
    let rows: Vec<Option<u32>> = tk
        .sym
        .iter()
        .zip(&tk.ts)
        .map(|(s, t)| index.get(&(s.as_str(), *t)).copied())
        .collect();
    let unmatched_rows = rows.iter().filter(|r| r.is_none()).count();

    let scratch: PathBuf = match &opts.scratch {
        Some(s) => PathBuf::from(s),
        None => Path::new(&opts.out)
            .parent()
            .map(|p| p.to_path_buf())
            .unwrap_or_default(),
    };
    std::fs::create_dir_all(&scratch).map_err(|e| format!("{}: {}", scratch.display(), e))?;
    let tag = format!("{}-{}", std::process::id(), manifest::sha256_hex(opts.out.as_bytes())[..8].to_string());
    let mut extras: Vec<AppendColumn> = Vec::new();
    let mut files: Vec<PathBuf> = Vec::new();
    let result = (|| -> Result<String, String> {
        for name in &opts.columns {
            let values = column_all(&family, name)?;
            let file = scratch.join(format!("rcsr-append-{}-{}.f64", tag, name));
            let mut w = std::io::BufWriter::new(
                std::fs::File::create(&file).map_err(|e| format!("{}: {}", file.display(), e))?,
            );
            files.push(file.clone());
            for r in &rows {
                let v = r.map(|i| values[i as usize]).unwrap_or(f64::NAN);
                w.write_all(&v.to_le_bytes())
                    .map_err(|e| format!("{}: {}", file.display(), e))?;
            }
            w.flush().map_err(|e| format!("{}: {}", file.display(), e))?;
            extras.push(AppendColumn {
                name: name.clone(),
                file,
                replace: false,
            });
        }
        store::write_store_appending_hashed(&opts.out, &target, &extras)
    })();
    for f in &files {
        let _ = std::fs::remove_file(f);
    }
    let output_sha256 = result?;
    let declared = crate::lockbox::declared_for(&opts.target);
    let schema = registry::write_schema_declared(&opts.out, &opts.panel_version, declared.as_ref())?;
    let after = manifest::sha256_file(&opts.target)?;
    Ok(AppendReport {
        input_sha256_before: before,
        input_sha256_after: after,
        output_sha256,
        appended: opts.columns.clone(),
        unmatched_rows,
        schema,
    })
}

fn usage_text() -> &'static str {
    "rcsr panel schema --data panel.rcsb [--panel-version V] [--verify]
rcsr panel append --data panel.rcsb --family symbol_major.rcsb --columns a,b --out out.rcsb [--panel-version V] [--scratch dir]
"
}

fn fail(msg: &str, code: i32) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(code);
}

/// `rcsr panel schema | append`.
pub fn cli(cmd: &str, args: &[String]) -> ! {
    let mut data: Option<String> = None;
    let mut family: Option<String> = None;
    let mut columns: Vec<String> = Vec::new();
    let mut out: Option<String> = None;
    let mut panel_version = registry::DEFAULT_PANEL_VERSION.to_string();
    let mut scratch: Option<String> = None;
    let mut verify = false;
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--data" => data = Some(value(args, &mut i)),
            "--family" => family = Some(value(args, &mut i)),
            "--columns" => {
                columns = value(args, &mut i)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--out" => out = Some(value(args, &mut i)),
            "--panel-version" => panel_version = value(args, &mut i),
            "--scratch" => scratch = Some(value(args, &mut i)),
            "--verify" => verify = true,
            "-h" | "--help" => {
                print!("{}", usage_text());
                std::process::exit(0);
            }
            other => fail(&format!("unknown option '{}' for `rcsr panel {}`", other, cmd), 2),
        }
        i += 1;
    }
    let data = data.unwrap_or_else(|| fail(&format!("`rcsr panel {}` needs --data", cmd), 2));
    match cmd {
        "schema" => {
            if verify {
                match registry::verify_schema(&data) {
                    Ok(bad) if bad.is_empty() => eprintln!("schema ok: {}", registry::schema_path(&data)),
                    Ok(bad) => fail(&format!("schema hash mismatch for: {}", bad.join(", ")), 3),
                    Err(e) => fail(&e, 3),
                }
            } else {
                match registry::write_schema(&data, &panel_version) {
                    Ok(p) => eprintln!("schema written: {}", p),
                    Err(e) => fail(&e, 3),
                }
            }
        }
        "append" => {
            let opts = AppendOptions {
                target: data,
                family: family.unwrap_or_else(|| fail("append needs --family", 2)),
                columns,
                out: out.unwrap_or_else(|| fail("append needs --out", 2)),
                panel_version,
                scratch,
            };
            match append_family(&opts) {
                Ok(r) => {
                    eprintln!(
                        "appended {} column(s); unmatched rows {}; input sha256 {} (unchanged: {}); output sha256 {}; schema {}",
                        r.appended.len(),
                        r.unmatched_rows,
                        r.input_sha256_before,
                        r.input_sha256_before == r.input_sha256_after,
                        r.output_sha256,
                        r.schema
                    );
                }
                Err(e) => fail(&e, 3),
            }
        }
        _ => fail(usage_text(), 2),
    }
    std::process::exit(0);
}

fn value(args: &[String], i: &mut usize) -> String {
    *i += 1;
    args.get(*i)
        .cloned()
        .unwrap_or_else(|| fail(&format!("{} requires a value", args[*i - 1]), 2))
}
