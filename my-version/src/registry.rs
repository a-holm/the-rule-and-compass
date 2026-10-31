//! #260: versioned feature registry contract for column stores.
//!
//! The registry itself lives in [`crate::enrich::default_registry`]. This
//! module turns it into the `<panel>.schema.json` sidecar (one entry per
//! stored column, with a SHA-256 of the column's stored values), checks
//! `--vars` names against that sidecar, and selects the deterministic
//! family-stratified `--project` views. It is metadata only: the expression
//! engine stays the search.

use crate::enrich::{self, RegistryEntry};
use crate::manifest;
use crate::store::{DType, FileStore};
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::OnceLock;

pub const SCHEMA_VERSION: u32 = 1;
pub const FEATURE_REGISTRY_VERSION: u32 = 1;
pub const DEFAULT_PANEL_VERSION: &str = "2";

/// Compact 52-feature parity is owned by #269; this module registers and
/// hashes columns and never certifies a formula.
pub const PARITY_OWNER_ISSUE: u32 = 269;

pub const DEFAULT_VIEW_SIZE: usize = 36;
pub const VIEW_SIZE_MIN: usize = 24;
pub const VIEW_SIZE_MAX: usize = 48;
pub const RESEARCH_SIZE_MIN: usize = 96;
pub const RESEARCH_SIZE_MAX: usize = 128;

/// The lag menu used by `--lag-set default`.
pub const DEFAULT_LAG_MENU: [usize; 4] = [1, 4, 24, 96];

/// Panel-store columns that no feature group emits.
const QUALITY_COLUMNS: [&str; 6] = ["gap", "cov15m", "flags", "traded", "dup_rows", "archive_unit"];

fn registry_map() -> &'static BTreeMap<String, RegistryEntry> {
    static MAP: OnceLock<BTreeMap<String, RegistryEntry>> = OnceLock::new();
    MAP.get_or_init(|| {
        enrich::default_registry()
            .into_iter()
            .map(|e| (e.name.clone(), e))
            .collect()
    })
}

/// A column resolved against the registry.
#[derive(Clone, Debug)]
pub struct Resolved {
    pub family: String,
    pub timeframe: String,
    pub warmup: usize,
    pub transform: String,
    pub pit_rule: String,
    pub available: bool,
    pub reason: Option<String>,
    pub causal_offset: i64,
    pub input: bool,
    pub sources: Vec<String>,
    pub unit: String,
    pub semantic_cost: u32,
}

fn split_lag(name: &str) -> Option<(&str, usize)> {
    let (stem, k) = name.rsplit_once("_lag")?;
    let k: usize = k.parse().ok()?;
    if k == 0 || !registry_map().contains_key(stem) {
        return None;
    }
    Some((stem, k))
}

/// Resolve a stored column name to its registry entry. A `<name>_lag<k>`
/// column resolves to its base entry delayed by `k` bars.
pub fn lookup(name: &str) -> Option<Resolved> {
    if let Some(e) = registry_map().get(name) {
        return Some(resolve_entry(e, 0));
    }
    if QUALITY_COLUMNS.contains(&name) {
        return Some(Resolved {
            family: "quality".to_string(),
            timeframe: "1h".to_string(),
            warmup: 0,
            transform: "panel builder quality column".to_string(),
            pit_rule: "bar t".to_string(),
            available: true,
            reason: None,
            causal_offset: 0,
            input: true,
            sources: Vec::new(),
            unit: match name {
                "flags" | "traded" => "flag",
                "archive_unit" => "code",
                _ => "count",
            }
            .to_string(),
            semantic_cost: 0,
        });
    }
    let (stem, k) = split_lag(name)?;
    let mut r = resolve_entry(registry_map().get(stem)?, k);
    r.family = "lag".to_string();
    r.transform = format!("delay({}, {})", stem, k);
    r.input = false;
    Some(r)
}

fn resolve_entry(e: &RegistryEntry, lag: usize) -> Resolved {
    let input = e.source == "input";
    let family = e.family.to_string();
    let sources: Vec<&str> = if input {
        Vec::new()
    } else {
        match e.family {
            "calendar" => vec!["ts"],
            "cross" | "xs" | "btc" => vec!["close", "volume", "symbol", "member_asof"],
            "flow" => vec!["volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"],
            "returns" | "ma" | "vol" | "bollinger" | "rsi" => vec!["close"],
            "atr" | "range" | "candle" | "donchian" | "price_action" => {
                vec!["open", "high", "low", "close"]
            }
            _ => vec!["open", "high", "low", "close", "volume"],
        }
    };
    let semantic_cost = if input {
        0
    } else if matches!(e.family, "mtf" | "cross" | "xs" | "btc" | "exotic" | "py15m") {
        3
    } else if e.warmup > 0 {
        2
    } else {
        1
    };
    Resolved {
        unit: unit_of(&e.name, &family),
        family,
        timeframe: e.timeframe.to_string(),
        warmup: e.warmup + lag,
        transform: e.formula.clone(),
        pit_rule: e.pit.to_string(),
        available: e.available,
        reason: e.reason.clone(),
        causal_offset: -(lag as i64),
        input,
        sources: sources.into_iter().map(String::from).collect(),
        semantic_cost: if lag > 0 { semantic_cost + 1 } else { semantic_cost },
    }
}

fn unit_of(name: &str, family: &str) -> String {
    let u = match name {
        "ts" | "member_asof" => "unix_seconds",
        "symbol" | "listing_id" => "id",
        "open" | "high" | "low" | "close" | "vwap" => "quote_price",
        "volume" | "taker_buy_base" => "base_units",
        "quote_volume" | "taker_buy_quote" => "quote_currency",
        "trades" => "count",
        _ if name.contains("rank") => "rank_unit_interval",
        _ => match family {
            "calendar" => "unit_circle",
            "returns" => "log_return",
            "vol" => "log_return_std",
            _ => "dimensionless",
        },
    };
    u.to_string()
}

fn esc(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let sub = prev[j - 1] + usize::from(a[i - 1] != b[j - 1]);
            cur[j] = sub.min(prev[j] + 1).min(cur[j - 1] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

/// Registry name closest to `name`, with its family.
pub fn nearest(name: &str) -> Option<(String, String)> {
    registry_map()
        .values()
        .map(|e| (levenshtein(name, &e.name), e))
        .min_by(|a, b| a.0.cmp(&b.0).then(a.1.name.cmp(&b.1.name)))
        .map(|(_, e)| (e.name.clone(), e.family.to_string()))
}

/// `<store>.schema.json`.
pub fn schema_path(store_path: &str) -> String {
    format!("{}.schema.json", store_path)
}

/// One stored column with its registry metadata and data hash.
#[derive(Clone, Debug)]
pub struct ColumnSchema {
    pub name: String,
    pub resolved: Resolved,
    pub dtype: DType,
    pub availability_start: Option<i64>,
    pub availability_end: Option<i64>,
    pub sha256: String,
}

/// Build the schema of an open store. Every column must be registered; the
/// first unknown name is a hard error that names the nearest registry entry.
///
/// The store is read one group at a time, so memory is one group of one
/// column plus the time column.
pub fn build_schema(store: &FileStore) -> Result<Vec<ColumnSchema>, String> {
    let header = store.header();
    let names = header.names();
    let dtypes = header.dtypes();
    let mut resolved = Vec::with_capacity(names.len());
    for n in &names {
        match lookup(n) {
            Some(r) if r.available => resolved.push(r),
            Some(r) => {
                return Err(format!(
                    "column '{}' is registered in family '{}' but unavailable: {}",
                    n,
                    r.family,
                    r.reason.unwrap_or_else(|| "reserved".to_string())
                ))
            }
            None => return Err(unknown_message(n, &schema_path(store.path()), None)),
        }
    }
    let time_col = names
        .iter()
        .position(|n| n == "ts")
        .or_else(|| dtypes.iter().position(|d| *d == DType::I64));
    let mut hashers: Vec<manifest::Sha256Hasher> =
        names.iter().map(|_| manifest::Sha256Hasher::new()).collect();
    let mut first: Vec<Option<i64>> = vec![None; names.len()];
    let mut last: Vec<Option<i64>> = vec![None; names.len()];
    let mut bytes: Vec<u8> = Vec::new();
    for g in 0..header.n_groups() {
        let times: Vec<i64> = match time_col {
            Some(c) if dtypes[c] == DType::I64 => store.read_i64(g, c)?,
            Some(c) => store
                .read_f64_group(g, c)?
                .into_iter()
                .map(|v| v as i64)
                .collect(),
            None => Vec::new(),
        };
        for (c, dtype) in dtypes.iter().enumerate() {
            bytes.clear();
            match dtype {
                DType::I64 => {
                    for v in store.read_i64(g, c)? {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
                DType::F64 => {
                    let values = store.read_f64_group(g, c)?;
                    for (r, v) in values.iter().enumerate() {
                        bytes.extend_from_slice(&v.to_le_bytes());
                        if v.is_finite() {
                            if let Some(&t) = times.get(r) {
                                first[c].get_or_insert(t);
                                last[c] = Some(t);
                            }
                        }
                    }
                }
                DType::U32 => {
                    for v in store.read_codes(g, c)? {
                        bytes.extend_from_slice(&v.to_le_bytes());
                    }
                }
            }
            hashers[c].update(&bytes);
        }
    }
    let mut out = Vec::with_capacity(names.len());
    for (c, (hasher, r)) in hashers.into_iter().zip(resolved).enumerate() {
        out.push(ColumnSchema {
            name: names[c].clone(),
            resolved: r,
            dtype: dtypes[c],
            availability_start: first[c],
            availability_end: last[c],
            sha256: hasher.finalize(),
        });
    }
    Ok(out)
}

fn opt_i64(v: Option<i64>) -> String {
    v.map(|x| x.to_string()).unwrap_or_else(|| "null".to_string())
}

/// Serialise the schema: a header line, one line per column.
pub fn schema_json(columns: &[ColumnSchema], rows: usize, panel_version: &str) -> String {
    let mut s = format!(
        "{{\"panel_version\":{},\"schema_version\":{},\"feature_registry_version\":{},\
         \"compact_parity_owner\":\"#{}\",\"rows\":{},\"columns\":[\n",
        esc(panel_version),
        SCHEMA_VERSION,
        FEATURE_REGISTRY_VERSION,
        PARITY_OWNER_ISSUE,
        rows
    );
    for (i, c) in columns.iter().enumerate() {
        let r = &c.resolved;
        let sources: Vec<String> = r.sources.iter().map(|x| esc(x)).collect();
        s.push_str(&format!(
            "{{\"name\":{},\"family\":{},\"timeframe\":{},\"sources\":[{}],\"transform\":{},\
             \"causal_offset\":{},\"warmup_bars\":{},\"dtype\":{},\"unit\":{},\
             \"availability_start\":{},\"availability_end\":{},\"pit_rule\":{},\
             \"semantic_cost\":{},\"status\":{},\"sha256\":{}}}{}\n",
            esc(&c.name),
            esc(&r.family),
            esc(&r.timeframe),
            sources.join(","),
            esc(&r.transform),
            r.causal_offset,
            r.warmup,
            esc(c.dtype.name()),
            esc(&r.unit),
            opt_i64(c.availability_start),
            opt_i64(c.availability_end),
            esc(&r.pit_rule),
            r.semantic_cost,
            esc(if r.available { "available" } else { "unavailable" }),
            esc(&c.sha256),
            if i + 1 < columns.len() { "," } else { "" }
        ));
    }
    s.push_str("]}\n");
    s
}

/// Hash every column of `store_path` and write `<store_path>.schema.json`.
pub fn write_schema(store_path: &str, panel_version: &str) -> Result<String, String> {
    let store = FileStore::open(store_path)?;
    let columns = build_schema(&store)?;
    let text = schema_json(&columns, store.n_rows(), panel_version);
    let out = schema_path(store_path);
    std::fs::write(&out, text).map_err(|e| format!("{}: {}", out, e))?;
    Ok(out)
}

/// Column name to SHA-256 as recorded in a schema file.
pub fn read_hashes(schema_file: &str) -> Result<BTreeMap<String, String>, String> {
    let text = std::fs::read_to_string(schema_file).map_err(|e| format!("{}: {}", schema_file, e))?;
    let mut out = BTreeMap::new();
    for line in text.lines().filter(|l| l.starts_with("{\"name\":")) {
        let name = manifest::string_field(line, "name");
        let sha = manifest::string_field(line, "sha256");
        if let (Some(n), Some(h)) = (name, sha) {
            out.insert(n, h);
        }
    }
    if out.is_empty() {
        return Err(format!("{}: no column entries", schema_file));
    }
    Ok(out)
}

/// Re-hash the store and compare with its schema file. Returns the names whose
/// hash differs or that are missing from either side.
pub fn verify_schema(store_path: &str) -> Result<Vec<String>, String> {
    let recorded = read_hashes(&schema_path(store_path))?;
    let store = FileStore::open(store_path)?;
    let actual = build_schema(&store)?;
    let mut bad = Vec::new();
    for c in &actual {
        if recorded.get(&c.name) != Some(&c.sha256) {
            bad.push(c.name.clone());
        }
    }
    for n in recorded.keys() {
        if !actual.iter().any(|c| &c.name == n) {
            bad.push(n.clone());
        }
    }
    Ok(bad)
}

fn unknown_message(name: &str, schema_file: &str, panel_has: Option<&str>) -> String {
    if let Some(r) = lookup(name) {
        return format!(
            "column '{}' is registered (family '{}', timeframe {}) but absent from {}{}",
            name,
            r.family,
            r.timeframe,
            schema_file,
            panel_has.map(|p| format!(" ({})", p)).unwrap_or_default()
        );
    }
    match nearest(name) {
        Some((near, family)) => format!(
            "unknown column '{}': not in the feature registry (registry name '{}', family '{}')",
            name, near, family
        ),
        None => format!("unknown column '{}': not in the feature registry", name),
    }
}

/// Validate `--vars` against the store's schema sidecar. A store without a
/// sidecar is not a registry panel and passes. An unknown name is an error.
pub fn check_vars(store_path: &str, vars: &[String]) -> Result<(), String> {
    let schema_file = schema_path(store_path);
    if !Path::new(&schema_file).is_file() {
        return Ok(());
    }
    let known = read_hashes(&schema_file)?;
    for v in vars {
        if !known.contains_key(v) {
            return Err(unknown_message(v, &schema_file, Some("not a column of this panel")));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Family-stratified views
// ---------------------------------------------------------------------------

/// A deterministic view over the registered feature columns of a panel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct View {
    pub id: String,
    pub seed: u64,
    pub view: usize,
    pub size: usize,
    pub columns: Vec<String>,
    pub families: Vec<(String, Vec<String>)>,
}

impl View {
    pub fn json(&self) -> String {
        let cols: Vec<String> = self.columns.iter().map(|c| esc(c)).collect();
        let fams: Vec<String> = self
            .families
            .iter()
            .map(|(f, names)| {
                let n: Vec<String> = names.iter().map(|x| esc(x)).collect();
                format!("{}:[{}]", esc(f), n.join(","))
            })
            .collect();
        format!(
            "{{\"view_id\":{},\"seed\":{},\"view\":{},\"size\":{},\"feature_registry_version\":{},\
             \"columns\":[{}],\"families\":{{{}}}}}",
            esc(&self.id),
            self.seed,
            self.view,
            self.size,
            FEATURE_REGISTRY_VERSION,
            cols.join(","),
            fams.join(",")
        )
    }
}

fn hash64(text: &str) -> u64 {
    let hex = manifest::sha256_hex(text.as_bytes());
    u64::from_str_radix(&hex[..16], 16).unwrap_or(0)
}

/// Check a requested view size: 24 to 48, or 96 to 128 for a research cell.
pub fn check_view_size(size: usize) -> Result<(), String> {
    if (VIEW_SIZE_MIN..=VIEW_SIZE_MAX).contains(&size)
        || (RESEARCH_SIZE_MIN..=RESEARCH_SIZE_MAX).contains(&size)
    {
        Ok(())
    } else {
        Err(format!(
            "--project {}: view size must be {}-{} (research cell {}-{})",
            size, VIEW_SIZE_MIN, VIEW_SIZE_MAX, RESEARCH_SIZE_MIN, RESEARCH_SIZE_MAX
        ))
    }
}

/// Select a view from `columns` (stored column names in header order).
///
/// Candidates are the available, derived (non-input) registry columns. Each
/// family contributes one column, or two when `size` allows, and the choice
/// rotates with `(seed, view)`: consecutive views walk different families and
/// different names. The result depends only on the column names, the registry
/// and the arguments, never on row data.
pub fn select_view(columns: &[String], size: usize, seed: u64, view: usize) -> Result<View, String> {
    check_view_size(size)?;
    let mut fams: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for c in columns {
        if let Some(r) = lookup(c) {
            if r.available && !r.input && r.causal_offset <= 0 {
                fams.entry(r.family).or_default().push(c.clone());
            }
        }
    }
    if fams.is_empty() {
        return Err("--project: the store has no registered feature columns".to_string());
    }
    for names in fams.values_mut() {
        names.sort();
    }
    let family_names: Vec<String> = fams.keys().cloned().collect();
    let nf = family_names.len();
    let first_round = size.min(nf);
    let start = (hash64(&format!("family|{}", seed)) as usize).wrapping_add(view.wrapping_mul(first_round)) % nf;
    let order: Vec<&String> = (0..nf).map(|i| &family_names[(start + i) % nf]).collect();
    let mut picked: BTreeMap<String, Vec<String>> = BTreeMap::new();
    let mut count = 0usize;
    let rounds = if size <= VIEW_SIZE_MAX { 2 } else { usize::MAX };
    let per = size.div_ceil(nf).max(1);
    for round in 0..rounds {
        if count >= size || round > fams.values().map(|n| n.len()).max().unwrap_or(0) {
            break;
        }
        for fam in &order {
            if count >= size {
                break;
            }
            let names = &fams[*fam];
            if round >= names.len() {
                continue;
            }
            let base = hash64(&format!("name|{}|{}", seed, fam)) as usize;
            let idx = (base.wrapping_add(view.wrapping_mul(per)).wrapping_add(round)) % names.len();
            picked.entry((*fam).clone()).or_default().push(names[idx].clone());
            count += 1;
        }
    }
    let chosen: std::collections::BTreeSet<&String> = picked.values().flatten().collect();
    let cols: Vec<String> = columns.iter().filter(|c| chosen.contains(c)).cloned().collect();
    let families: Vec<(String, Vec<String>)> = picked.into_iter().collect();
    let id_text = format!("{}|{}|{}|{}|{}", FEATURE_REGISTRY_VERSION, seed, view, size, cols.join(","));
    let id = manifest::sha256_hex(id_text.as_bytes())[..12].to_string();
    Ok(View {
        id,
        seed,
        view,
        size,
        columns: cols,
        families,
    })
}

/// Parse a lag menu: `default` or a comma list of positive bar counts.
pub fn parse_lag_menu(s: &str) -> Result<Vec<usize>, String> {
    if s.trim() == "default" {
        return Ok(DEFAULT_LAG_MENU.to_vec());
    }
    let mut out: Vec<usize> = Vec::new();
    for part in s.split(',') {
        let k: usize = part
            .trim()
            .parse()
            .map_err(|_| format!("--lag-set: '{}' is not a bar count", part))?;
        if k == 0 {
            return Err("--lag-set: a lag of 0 bars is the variable itself".to_string());
        }
        if !out.contains(&k) {
            out.push(k);
        }
    }
    if out.is_empty() {
        return Err("--lag-set: empty menu".to_string());
    }
    out.sort_unstable();
    Ok(out)
}
