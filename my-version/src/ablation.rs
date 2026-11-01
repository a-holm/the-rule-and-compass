//! Add-one-family ablation: arm definitions, manifests, null columns, scoring
//! of a finished search run, and the pre-committed ship/drop verdict.
//!
//! The definition file (`protocol/ablation.json`) is committed before any run.
//! It names the base columns `B0`, the column groups, the arms `A0` to `A5`,
//! the generation budget, the seeds and the decision rule. The commands are:
//!
//! * `rcsr ablation plan`: one manifest per arm (arm name, ordered feature
//!   list, its SHA-256, generation budget, seeds) and `commands.sh` with every
//!   search, scoring and verdict command;
//! * `rcsr ablation prepare`: a projection CSV with the placebo columns
//!   (matched on marginal distribution, hence prevalence, and lag-1
//!   autocorrelation) and the circular-block-permuted copies of the family
//!   columns;
//! * `rcsr ablation score`: one search `run.json` into a result file (return
//!   space MSE, per-period net series, net per fold and threshold);
//! * `rcsr ablation verdict`: the ship/drop verdict per family.
//!
//! # Decision rule
//!
//! A family is `ship` only if every check passes against every control arm,
//! and `drop` otherwise, with every failed check listed:
//!
//! * `gain_exceeds_seed_range`: the mean return-space MSE gain (control minus
//!   treatment) is larger than the across-seed range of either arm;
//! * `spa_p_below_alpha`: the SPA p-value of the seed-mean net difference is
//!   below `spa_alpha`;
//! * `outside_null_band`: the gain is larger than the largest gain the
//!   permuted-feature arm reaches against its control;
//! * `no_sign_flip`: the net difference keeps one sign across seeds, across
//!   folds and across the threshold grid;
//! * `fdr_survives`: the Benjamini-Hochberg step-up at `fdr_q` over the SPA
//!   p-values of all families keeps the family;
//! * `dsr_above_min`: the deflated Sharpe ratio of the treatment, deflated by
//!   `N_trials`, is at least `min_dsr`.
//! * `pbo`: the PBO of the arms' seed-mean series, by CSCV on the raw series, is
//!   below `max_pbo`. It is a property of the arm set, so it applies to every
//!   family; a failure is listed on each family as `pbo`.
//!
//! The ablation is a selection on training data repeated inside every fold:
//! each run selects its finalist on a validation slice and is scored on the
//! fold's test slice, so the verdict never reads a slice a run trained on.

use crate::data::{self, Table};
use crate::devlog::Json;
use crate::inference::{self, Dsr, Pbo, Spa};
use crate::ladder::{self, jnum, jstr, Fold, Input, Options, Panel};
use crate::lockbox::Lockbox;
use crate::manifest::sha256_hex;
use crate::protocol;
use crate::rng::Mt;
use std::collections::BTreeMap;
use std::path::Path;

pub const DEFINITION_SCHEMA: &str = "rcsr-ablation/1";
pub const MANIFEST_SCHEMA: &str = "rcsr-ablation-arm/1";
pub const RESULT_SCHEMA: &str = "rcsr-ablation-result/1";
pub const VERDICT_SCHEMA: &str = "rcsr-ablation-verdict/1";

/// SHA-256 of the ordered variable list, one name per line.
pub fn feature_list_sha256(vars: &[String]) -> String {
    sha256_hex(vars.join("\n").as_bytes())
}

// ---------------------------------------------------------------------------
// Definition
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct ArmDef {
    pub name: String,
    /// Column groups added as real columns.
    pub add: Vec<String>,
    /// Number of placebo columns.
    pub placebo: usize,
    /// Column groups added as circular-block-permuted copies.
    pub permute: Vec<String>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FamilyDef {
    pub name: String,
    pub treatment: String,
    pub controls: Vec<String>,
    /// Permuted-feature arm and the arm it is compared with.
    pub null_arm: Option<(String, String)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Rule {
    pub min_seeds: usize,
    pub finalist_seeds: usize,
    pub spa_alpha: f64,
    pub fdr_q: f64,
    pub min_dsr: f64,
    pub draws: usize,
    pub block_len: usize,
    pub seed: u32,
    pub pbo_blocks: usize,
    /// A family ships only when the PBO of the arms is below this.
    pub max_pbo: f64,
    pub families: Vec<FamilyDef>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Definition {
    pub b0: Vec<String>,
    pub groups: Vec<(String, Vec<String>)>,
    /// Group whose columns are the templates of the placebo columns.
    pub placebo_template_group: String,
    pub placebo_prefix: String,
    pub permute_prefix: String,
    pub permute_block_len: usize,
    pub generation_budget: u64,
    pub screen_seeds: usize,
    pub finalist_seeds: usize,
    pub windows: u64,
    pub horizons: u64,
    pub arms: Vec<ArmDef>,
    pub rule: Rule,
    pub sha256: String,
}

fn str_list(j: &Json, key: &str) -> Result<Vec<String>, String> {
    match j.get(key) {
        Some(Json::Arr(a)) => a
            .iter()
            .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("'{}' must hold strings", key)))
            .collect(),
        None => Ok(Vec::new()),
        _ => Err(format!("'{}' must be an array", key)),
    }
}

fn req_f64(j: &Json, key: &str) -> Result<f64, String> {
    j.get(key).and_then(Json::as_f64).ok_or_else(|| format!("missing number '{}'", key))
}

fn opt_f64(j: &Json, key: &str, default: f64) -> f64 {
    j.get(key).and_then(Json::as_f64).unwrap_or(default)
}

fn req_str(j: &Json, key: &str) -> Result<String, String> {
    j.get(key).and_then(Json::as_str).map(str::to_string).ok_or_else(|| format!("missing string '{}'", key))
}

pub fn parse_definition(text: &str) -> Result<Definition, String> {
    let doc = Json::parse(text.trim()).ok_or_else(|| "not valid JSON".to_string())?;
    match doc.get("schema").and_then(Json::as_str) {
        Some(DEFINITION_SCHEMA) => {}
        other => return Err(format!("schema must be {}, found {:?}", DEFINITION_SCHEMA, other)),
    }
    let b0 = str_list(&doc, "b0")?;
    if b0.is_empty() {
        return Err("b0 needs at least one column".to_string());
    }
    let mut groups = Vec::new();
    match doc.get("groups") {
        Some(Json::Obj(fields)) => {
            for (k, v) in fields {
                let cols = match v {
                    Json::Arr(a) => a
                        .iter()
                        .map(|x| x.as_str().map(str::to_string).ok_or_else(|| format!("group '{}' must hold strings", k)))
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => return Err(format!("group '{}' must be an array", k)),
                };
                groups.push((k.clone(), cols));
            }
        }
        _ => return Err("missing object 'groups'".to_string()),
    }
    let mut arms = Vec::new();
    match doc.get("arms") {
        Some(Json::Arr(a)) => {
            for x in a {
                arms.push(ArmDef {
                    name: req_str(x, "name")?,
                    add: str_list(x, "add")?,
                    placebo: opt_f64(x, "placebo", 0.0) as usize,
                    permute: str_list(x, "permute")?,
                });
            }
        }
        _ => return Err("missing array 'arms'".to_string()),
    }
    let r = doc.get("rule").ok_or_else(|| "missing object 'rule'".to_string())?;
    let mut families = Vec::new();
    match r.get("families") {
        Some(Json::Arr(a)) => {
            for x in a {
                let null_arm = match x.get("null_arm").and_then(Json::as_str) {
                    Some(n) => Some((n.to_string(), req_str(x, "null_control")?)),
                    None => None,
                };
                families.push(FamilyDef {
                    name: req_str(x, "name")?,
                    treatment: req_str(x, "treatment")?,
                    controls: str_list(x, "controls")?,
                    null_arm,
                });
            }
        }
        _ => return Err("rule needs a 'families' array".to_string()),
    }
    let trials = doc.get("trials");
    let def = Definition {
        b0,
        groups,
        placebo_template_group: req_str(&doc, "placebo_template_group")?,
        placebo_prefix: doc.get("placebo_prefix").and_then(Json::as_str).unwrap_or("placebo_").to_string(),
        permute_prefix: doc.get("permute_prefix").and_then(Json::as_str).unwrap_or("perm_").to_string(),
        permute_block_len: opt_f64(&doc, "permute_block_len", 96.0) as usize,
        generation_budget: req_f64(&doc, "generation_budget")? as u64,
        screen_seeds: opt_f64(&doc, "screen_seeds", 3.0) as usize,
        finalist_seeds: opt_f64(&doc, "finalist_seeds", 5.0) as usize,
        windows: trials.map_or(1.0, |t| opt_f64(t, "windows", 1.0)) as u64,
        horizons: trials.map_or(1.0, |t| opt_f64(t, "horizons", 1.0)) as u64,
        arms,
        rule: Rule {
            min_seeds: opt_f64(r, "min_seeds", 3.0) as usize,
            finalist_seeds: opt_f64(&doc, "finalist_seeds", 5.0) as usize,
            spa_alpha: opt_f64(r, "spa_alpha", 0.05),
            fdr_q: opt_f64(r, "fdr_q", 0.10),
            min_dsr: opt_f64(r, "min_dsr", 0.5),
            draws: opt_f64(r, "draws", 2000.0) as usize,
            block_len: opt_f64(r, "block_len", 0.0) as usize,
            seed: opt_f64(r, "seed", 0.0) as u32,
            pbo_blocks: opt_f64(r, "pbo_blocks", 8.0) as usize,
            max_pbo: opt_f64(r, "max_pbo", 0.5),
            families,
        },
        sha256: sha256_hex(text.as_bytes()),
    };
    validate_definition(&def)?;
    Ok(def)
}

fn validate_definition(d: &Definition) -> Result<(), String> {
    let group = |n: &str| d.groups.iter().any(|(k, _)| k == n);
    if !group(&d.placebo_template_group) {
        return Err(format!("placebo_template_group '{}' is not a group", d.placebo_template_group));
    }
    for a in &d.arms {
        for g in a.add.iter().chain(&a.permute) {
            if !group(g) {
                return Err(format!("arm {}: unknown group '{}'", a.name, g));
            }
        }
    }
    let arm = |n: &str| d.arms.iter().any(|a| a.name == n);
    for f in &d.rule.families {
        for n in std::iter::once(&f.treatment).chain(&f.controls).chain(f.null_arm.iter().flat_map(|(a, b)| [a, b])) {
            if !arm(n) {
                return Err(format!("family {}: unknown arm '{}'", f.name, n));
            }
        }
        if f.controls.is_empty() {
            return Err(format!("family {}: needs at least one control arm", f.name));
        }
    }
    if d.generation_budget == 0 {
        return Err("generation_budget must be positive".to_string());
    }
    if d.screen_seeds < 3 || d.finalist_seeds < d.screen_seeds {
        return Err("screen_seeds must be at least 3 and finalist_seeds at least screen_seeds".to_string());
    }
    if d.rule.pbo_blocks % 2 != 0 || d.rule.pbo_blocks < 2 {
        return Err("rule.pbo_blocks must be even and at least 2".to_string());
    }
    if !(d.rule.max_pbo > 0.0 && d.rule.max_pbo <= 1.0) {
        return Err("rule.max_pbo must be in (0, 1]".to_string());
    }
    Ok(())
}

impl Definition {
    pub fn group(&self, name: &str) -> &[String] {
        self.groups.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_slice()).unwrap_or(&[])
    }

    pub fn placebo_names(&self, n: usize) -> Vec<String> {
        (1..=n).map(|k| format!("{}{}", self.placebo_prefix, k)).collect()
    }

    pub fn permuted_name(&self, col: &str) -> String {
        format!("{}{}", self.permute_prefix, col)
    }

    /// Ordered feature list of an arm: base columns, real groups, placebo
    /// columns, then permuted groups.
    pub fn features(&self, arm: &ArmDef) -> Vec<String> {
        let mut out = self.b0.clone();
        for g in &arm.add {
            out.extend(self.group(g).iter().cloned());
        }
        out.extend(self.placebo_names(arm.placebo));
        for g in &arm.permute {
            out.extend(self.group(g).iter().map(|c| self.permuted_name(c)));
        }
        out
    }

    pub fn arm(&self, name: &str) -> Result<&ArmDef, String> {
        self.arms.iter().find(|a| a.name == name).ok_or_else(|| format!("unknown arm '{}'", name))
    }

    /// Real columns every arm may need.
    pub fn real_columns(&self) -> Vec<String> {
        let mut v = self.b0.clone();
        for (_, cols) in &self.groups {
            v.extend(cols.iter().cloned());
        }
        v.sort();
        v.dedup();
        v
    }

    /// `families x columns x windows x seeds x folds x horizons`.
    pub fn n_trials(&self, seeds: usize, folds: usize) -> u64 {
        let mut cols: Vec<String> = Vec::new();
        for a in &self.arms {
            cols.extend(self.features(a));
        }
        cols.sort();
        cols.dedup();
        self.rule.families.len() as u64 * cols.len() as u64 * self.windows * seeds as u64 * folds as u64 * self.horizons
    }
}

// ---------------------------------------------------------------------------
// Null columns
// ---------------------------------------------------------------------------

fn gaussian(rng: &mut Mt) -> f64 {
    let u1 = rng.next_f64().max(1e-300);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

fn mix_seed(parts: &[u64]) -> u32 {
    let mut h = 0x9E37_79B9_7F4A_7C15u64;
    for p in parts {
        h ^= p.wrapping_add(0x9E37_79B9_7F4A_7C15).wrapping_add(h << 6).wrapping_add(h >> 2);
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        h ^= h >> 29;
    }
    (h ^ (h >> 32)) as u32
}

/// Lag-1 autocorrelation of `x` inside each group, pooled; `groups` must be
/// contiguous.
pub fn lag1_autocorr(x: &[f64], groups: &[u64]) -> f64 {
    let finite: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() < 3 {
        return 0.0;
    }
    let m = finite.iter().sum::<f64>() / finite.len() as f64;
    let (mut num, mut den) = (0.0, 0.0);
    for i in 0..x.len() {
        if x[i].is_finite() {
            den += (x[i] - m).powi(2);
            if i > 0 && groups[i] == groups[i - 1] && x[i - 1].is_finite() {
                num += (x[i] - m) * (x[i - 1] - m);
            }
        }
    }
    if den > 0.0 {
        (num / den).clamp(-0.99, 0.99)
    } else {
        0.0
    }
}

/// A placebo column for `template`: the template's lag-1 autocorrelation and
/// exact marginal distribution (so the prevalence of a binary or sparse
/// template), with no relation to anything else. A Gaussian AR(1) path per
/// group is mapped by rank onto the sorted finite template values; the latent
/// correlation is bisected until the mapped column reaches the template's
/// lag-1 autocorrelation. The NaN positions of the template are kept.
pub fn placebo_column(template: &[f64], groups: &[u64], seed: u32, k: u64) -> Vec<f64> {
    let target = lag1_autocorr(template, groups);
    let n = template.len();
    let mut sorted: Vec<f64> = template.iter().copied().filter(|v| v.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let finite: Vec<usize> = (0..n).filter(|&i| template[i].is_finite()).collect();
    let build = |rho: f64| -> Vec<f64> {
        let innov = (1.0 - rho * rho).sqrt();
        let mut z = vec![0.0f64; n];
        let mut rng = Mt::new(mix_seed(&[seed as u64, k, 0xA11]));
        let mut prev = 0.0;
        for i in 0..n {
            prev = if i == 0 || groups[i] != groups[i - 1] {
                gaussian(&mut rng)
            } else {
                rho * prev + innov * gaussian(&mut rng)
            };
            z[i] = prev;
        }
        let mut keep = finite.clone();
        keep.sort_by(|&a, &b| z[a].total_cmp(&z[b]).then_with(|| a.cmp(&b)));
        let mut out = vec![f64::NAN; n];
        for (rank, &i) in keep.iter().enumerate() {
            out[i] = sorted[rank];
        }
        out
    };
    let (mut lo, mut hi) = (-0.995f64, 0.995f64);
    let mut out = build(0.0);
    for _ in 0..10 {
        let mid = 0.5 * (lo + hi);
        out = build(mid);
        if lag1_autocorr(&out, groups) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out
}

/// Circular-block permutation order for a series of length `n`: a random
/// circular shift, blocks of `block` rows, the blocks shuffled. The result is a
/// permutation of `0..n`.
pub fn block_permutation(n: usize, block: usize, rng: &mut Mt) -> Vec<usize> {
    if n == 0 {
        return Vec::new();
    }
    let block = block.clamp(1, n);
    let shift = rng.usize_below(n);
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut at = 0;
    while at < n {
        let len = block.min(n - at);
        blocks.push((at..at + len).map(|i| (i + shift) % n).collect());
        at += len;
    }
    for i in (1..blocks.len()).rev() {
        let j = rng.usize_below(i + 1);
        blocks.swap(i, j);
    }
    blocks.into_iter().flatten().collect()
}

/// Permute every column of `cols` with one per-group circular-block order, so
/// the columns keep their joint structure and lose their alignment with the
/// target. `groups` must be contiguous.
pub fn permute_columns(cols: &[&[f64]], groups: &[u64], block: usize, seed: u32) -> Vec<Vec<f64>> {
    let n = groups.len();
    let mut out: Vec<Vec<f64>> = cols.iter().map(|_| vec![f64::NAN; n]).collect();
    let mut lo = 0;
    while lo < n {
        let mut hi = lo + 1;
        while hi < n && groups[hi] == groups[lo] {
            hi += 1;
        }
        let mut rng = Mt::new(mix_seed(&[seed as u64, groups[lo], 0xB10C]));
        let order = block_permutation(hi - lo, block, &mut rng);
        for (c, col) in cols.iter().enumerate() {
            for (k, &src) in order.iter().enumerate() {
                out[c][lo + k] = col[lo + src];
            }
        }
        lo = hi;
    }
    out
}

fn csv_cell(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "NaN".to_string()
    }
}

fn csv_field(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub struct PrepareInput<'a> {
    pub table: &'a Table,
    pub def: &'a Definition,
    pub time: &'a str,
    pub group: &'a str,
    /// Extra columns copied through (target, return, scale).
    pub keep: &'a [String],
    pub seed: u32,
    /// Declared locked test: null columns are built per segment (before,
    /// inside, after) so no locked value reaches a development row.
    pub locked: Option<Lockbox>,
}

/// Segment of a time stamp relative to the locked test: 0 before, 1 inside, 2 after.
pub fn segment_of(t: f64, locked: Option<&Lockbox>) -> u8 {
    match locked {
        Some(l) if t >= l.end as f64 => 2,
        Some(l) if t >= l.start as f64 => 1,
        _ => 0,
    }
}

/// Group key per row for contiguous per-(group, segment) runs. Segment 0 keeps
/// the plain group id.
pub fn segment_keys(groups: &[u64], segs: &[u8]) -> Vec<u64> {
    groups.iter().zip(segs).map(|(&g, &s)| g + ((s as u64) << 32)).collect()
}

/// `placebo_column` built independently inside each segment: the marginal and
/// the lag-1 target of a segment come from that segment's template values only.
pub fn placebo_column_segmented(template: &[f64], groups: &[u64], segs: &[u8], seed: u32, k: u64) -> Vec<f64> {
    if segs.iter().all(|&s| s == segs[0]) {
        return placebo_column(template, groups, seed, k);
    }
    let mut out = vec![f64::NAN; template.len()];
    for s in 0..3u8 {
        let idx: Vec<usize> = (0..template.len()).filter(|&i| segs[i] == s).collect();
        if idx.is_empty() {
            continue;
        }
        let tpl: Vec<f64> = idx.iter().map(|&i| template[i]).collect();
        let g: Vec<u64> = idx.iter().map(|&i| groups[i]).collect();
        for (&j, v) in idx.iter().zip(placebo_column(&tpl, &g, seed, k)) {
            out[j] = v;
        }
    }
    out
}

/// The projection CSV: key columns, kept columns, real columns, placebo and
/// permuted columns, rows sorted by `(group, time)`.
pub fn prepare_csv(inp: &PrepareInput) -> Result<String, String> {
    prepare_csv_segments(inp).map(|(csv, _)| csv)
}

/// `prepare_csv` plus the row count of each segment (before, locked, after).
pub fn prepare_csv_segments(inp: &PrepareInput) -> Result<(String, [usize; 3]), String> {
    let t = inp.table;
    let def = inp.def;
    let n = t.len();
    let time_idx = t.index_of(inp.time)?;
    let group_idx = t.index_of(inp.group)?;
    let time: Vec<f64> = if t.numeric[time_idx] {
        t.col(time_idx).to_vec()
    } else {
        let d = t.dict(time_idx).ok_or_else(|| "time column is neither numeric nor a date".to_string())?;
        (0..n)
            .map(|r| data::parse_time(d.text(r)).ok_or_else(|| format!("row {}: cannot parse time", r + 1)))
            .collect::<Result<_, _>>()?
    };
    let mut gid: Vec<u64> = Vec::with_capacity(n);
    let mut labels: Vec<String> = Vec::new();
    for r in 0..n {
        let s = t.cell_text(r, group_idx);
        let k = match labels.iter().position(|x| *x == s) {
            Some(k) => k,
            None => {
                labels.push(s);
                labels.len() - 1
            }
        };
        gid.push(k as u64);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| gid[a].cmp(&gid[b]).then_with(|| time[a].total_cmp(&time[b])).then_with(|| a.cmp(&b)));
    let groups: Vec<u64> = order.iter().map(|&i| gid[i]).collect();
    let segs: Vec<u8> = order.iter().map(|&i| segment_of(time[i], inp.locked.as_ref())).collect();
    let mut seg_rows = [0usize; 3];
    for &s in &segs {
        seg_rows[s as usize] += 1;
    }
    let seg_groups = segment_keys(&groups, &segs);

    let real = def.real_columns();
    let mut copied: Vec<String> = vec![inp.time.to_string(), inp.group.to_string()];
    for k in inp.keep {
        if !copied.contains(k) {
            copied.push(k.clone());
        }
    }
    for c in &real {
        if !copied.contains(c) {
            copied.push(c.clone());
        }
    }
    let numeric_col = |name: &str| -> Result<Vec<f64>, String> {
        let j = t.index_of(name)?;
        if !t.numeric[j] {
            return Err(format!("column '{}' must be numeric", name));
        }
        let c = t.col(j);
        Ok(order.iter().map(|&i| c[i]).collect())
    };

    let max_placebo = def.arms.iter().map(|a| a.placebo).max().unwrap_or(0);
    let templates = def.group(&def.placebo_template_group).to_vec();
    if max_placebo > 0 && templates.is_empty() {
        return Err("the placebo template group is empty".to_string());
    }
    let mut new_names: Vec<String> = Vec::new();
    let mut new_cols: Vec<Vec<f64>> = Vec::new();
    for (k, name) in def.placebo_names(max_placebo).into_iter().enumerate() {
        let tpl = numeric_col(&templates[k % templates.len()])?;
        new_cols.push(placebo_column_segmented(&tpl, &groups, &segs, inp.seed, k as u64 + 1));
        new_names.push(name);
    }
    let mut perm_groups: Vec<&String> = Vec::new();
    for a in &def.arms {
        for g in &a.permute {
            if !perm_groups.contains(&g) {
                perm_groups.push(g);
            }
        }
    }
    for g in perm_groups {
        let names = def.group(g);
        let cols: Vec<Vec<f64>> = names.iter().map(|c| numeric_col(c)).collect::<Result<_, _>>()?;
        let refs: Vec<&[f64]> = cols.iter().map(|c| c.as_slice()).collect();
        let permuted = permute_columns(&refs, &seg_groups, def.permute_block_len, inp.seed);
        for (c, v) in names.iter().zip(permuted) {
            new_names.push(def.permuted_name(c));
            new_cols.push(v);
        }
    }

    let mut out = String::new();
    let header: Vec<String> = copied.iter().chain(&new_names).map(|s| csv_field(s)).collect();
    out.push_str(&header.join(","));
    out.push('\n');
    let src_idx: Vec<usize> = copied.iter().map(|c| t.index_of(c)).collect::<Result<_, _>>()?;
    for (row, &i) in order.iter().enumerate() {
        let mut cells: Vec<String> = Vec::with_capacity(src_idx.len() + new_cols.len());
        for &j in &src_idx {
            if t.numeric[j] {
                match t.ints(j) {
                    Some(ints) => cells.push(ints[i].to_string()),
                    None => cells.push(csv_cell(t.col(j)[i])),
                }
            } else {
                cells.push(csv_field(&t.cell_text(i, j)));
            }
        }
        for c in &new_cols {
            cells.push(csv_cell(c[row]));
        }
        out.push_str(&cells.join(","));
        out.push('\n');
    }
    Ok((out, seg_rows))
}

// ---------------------------------------------------------------------------
// Plan and manifests
// ---------------------------------------------------------------------------

pub struct PlanInput<'a> {
    pub def: &'a Definition,
    pub data: &'a str,
    pub target: &'a str,
    pub time: &'a str,
    pub group: Option<&'a str>,
    pub return_col: Option<&'a str>,
    pub windows: &'a str,
    pub scheme: &'a str,
    pub out_dir: &'a str,
    pub finalist: bool,
    pub search_args: &'a str,
    pub score_args: &'a str,
    pub verdict_args: &'a str,
}

fn shell_quote(s: &str) -> String {
    if !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || "/._-=:,".contains(c)) {
        s.to_string()
    } else {
        format!("'{}'", s.replace('\'', "'\\''"))
    }
}

pub fn seeds_for(def: &Definition, finalist: bool) -> Vec<u32> {
    let n = if finalist { def.finalist_seeds } else { def.screen_seeds };
    (1..=n as u32).collect()
}

/// `(file name, content)` pairs: one manifest per arm, `plan.json`, `commands.sh`.
pub fn plan_files(inp: &PlanInput) -> Vec<(String, String)> {
    let def = inp.def;
    let seeds = seeds_for(def, inp.finalist);
    let mut files = Vec::new();
    let mut script = String::new();
    script.push_str("#!/usr/bin/env bash\nset -euo pipefail\n");
    let dir = inp.out_dir;
    script.push_str(&format!(
        "# prepare: placebo and permuted columns (projection CSV; import it with `rcsr import` for a column store)\n\
rcsr ablation prepare --definition {} --data {} --time {} --group {} --keep {} --out {}/prepared.csv\n",
        shell_quote(&format!("{}/definition.json", dir)),
        shell_quote(inp.data),
        shell_quote(inp.time),
        shell_quote(inp.group.unwrap_or("")),
        shell_quote(&inp.target.to_string()),
        dir
    ));
    let prepared = format!("{}/prepared.csv", dir);
    let mut arms_json = Vec::new();
    for arm in &def.arms {
        let feats = def.features(arm);
        let hash = feature_list_sha256(&feats);
        let mut cmds = Vec::new();
        for &s in &seeds {
            let run = format!("{}/{}-s{}.run.json", dir, arm.name, s);
            let res = format!("{}/{}-s{}.result.json", dir, arm.name, s);
            let search = format!(
                "rcsr --data {} --target {} --vars {} --generations {} --seed {} --arm {} --json-out {} --run-manifest {}/{}-s{}.run-manifest.json {}",
                shell_quote(&prepared),
                shell_quote(inp.target),
                shell_quote(&feats.join(",")),
                def.generation_budget,
                s,
                arm.name,
                shell_quote(&run),
                dir,
                arm.name,
                s,
                inp.search_args
            );
            let score = format!(
                "rcsr ablation score --run {} --data {} --target {} --time {}{}{} --windows {} --scheme {} --out {} {}",
                shell_quote(&run),
                shell_quote(&prepared),
                shell_quote(inp.target),
                shell_quote(inp.time),
                inp.group.map_or(String::new(), |g| format!(" --group {}", shell_quote(g))),
                inp.return_col.map_or(String::new(), |g| format!(" --return-col {}", shell_quote(g))),
                shell_quote(inp.windows),
                inp.scheme,
                shell_quote(&res),
                inp.score_args
            );
            script.push_str(&format!("{}\n{}\n", search.trim_end(), score.trim_end()));
            cmds.push(search);
        }
        let seeds_json = seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(",");
        let manifest = format!(
            "{{\"schema\":{},\"arm\":{},\"features\":[{}],\"feature_count\":{},\"feature_list_sha256\":{},\
\"generation_budget\":{},\"seeds\":[{}],\"stage\":{},\"placebo_columns\":{},\"permuted_groups\":[{}],\
\"definition_sha256\":{}}}",
            jstr(MANIFEST_SCHEMA),
            jstr(&arm.name),
            feats.iter().map(|f| jstr(f)).collect::<Vec<_>>().join(","),
            feats.len(),
            jstr(&hash),
            def.generation_budget,
            seeds_json,
            jstr(if inp.finalist { "finalist" } else { "screen" }),
            arm.placebo,
            arm.permute.iter().map(|g| jstr(g)).collect::<Vec<_>>().join(","),
            jstr(&def.sha256)
        );
        files.push((format!("{}.arm.json", arm.name), manifest));
        arms_json.push(format!(
            "{{\"arm\":{},\"feature_list_sha256\":{},\"manifest\":{}}}",
            jstr(&arm.name),
            jstr(&hash),
            jstr(&format!("{}.arm.json", arm.name))
        ));
    }
    script.push_str(&format!(
        "rcsr ablation verdict --definition {} --results {} --plan {} --out {}/verdict.json --markdown {}/verdict.md {}\n",
        shell_quote(&format!("{}/definition.json", dir)),
        dir,
        dir,
        dir,
        dir,
        inp.verdict_args
    ));
    let plan = format!(
        "{{\"schema\":\"rcsr-ablation-plan/1\",\"definition_sha256\":{},\"generation_budget\":{},\"seeds\":[{}],\"stage\":{},\"arms\":[{}]}}",
        jstr(&def.sha256),
        def.generation_budget,
        seeds.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(","),
        jstr(if inp.finalist { "finalist" } else { "screen" }),
        arms_json.join(",")
    );
    files.push(("plan.json".to_string(), plan));
    files.push(("commands.sh".to_string(), script));
    files
}

// ---------------------------------------------------------------------------
// Scoring one run
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub struct SeedResult {
    pub arm: String,
    pub seed: u32,
    pub feature_list_sha256: String,
    pub generation_budget: u64,
    pub generations_run: u64,
    pub features: Vec<String>,
    pub features_used: Vec<String>,
    pub return_mse: f64,
    pub folds: Vec<String>,
    pub thresholds: Vec<f64>,
    /// `[fold][threshold]`: sum of equal-weight period net returns.
    pub net_by_fold_threshold: Vec<Vec<f64>>,
    /// Equal-weight period net returns at the headline threshold.
    pub net_period: Vec<f64>,
}

impl SeedResult {
    pub fn json(&self) -> String {
        let arr = |v: &[f64]| format!("[{}]", v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(","));
        let strs = |v: &[String]| format!("[{}]", v.iter().map(|x| jstr(x)).collect::<Vec<_>>().join(","));
        format!(
            "{{\"schema\":{},\"arm\":{},\"seed\":{},\"feature_list_sha256\":{},\"generation_budget\":{},\"generations_run\":{},\
\"features\":{},\"features_used\":{},\"return_mse\":{},\"folds\":{},\"thresholds\":{},\"net_by_fold_threshold\":[{}],\"net_period\":{}}}",
            jstr(RESULT_SCHEMA),
            jstr(&self.arm),
            self.seed,
            jstr(&self.feature_list_sha256),
            self.generation_budget,
            self.generations_run,
            strs(&self.features),
            strs(&self.features_used),
            jnum(self.return_mse),
            strs(&self.folds),
            arr(&self.thresholds),
            self.net_by_fold_threshold.iter().map(|r| arr(r)).collect::<Vec<_>>().join(","),
            arr(&self.net_period)
        )
    }

    pub fn parse(text: &str) -> Result<SeedResult, String> {
        let j = Json::parse(text.trim()).ok_or_else(|| "not valid JSON".to_string())?;
        if j.get("schema").and_then(Json::as_str) != Some(RESULT_SCHEMA) {
            return Err(format!("schema must be {}", RESULT_SCHEMA));
        }
        let nums = |j: &Json| -> Result<Vec<f64>, String> {
            match j {
                Json::Arr(a) => Ok(a.iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()),
                _ => Err("expected an array".to_string()),
            }
        };
        let field = |k: &str| j.get(k).ok_or_else(|| format!("missing '{}'", k));
        Ok(SeedResult {
            arm: req_str(&j, "arm")?,
            seed: req_f64(&j, "seed")? as u32,
            feature_list_sha256: req_str(&j, "feature_list_sha256")?,
            generation_budget: req_f64(&j, "generation_budget")? as u64,
            generations_run: req_f64(&j, "generations_run")? as u64,
            features: str_list(&j, "features")?,
            features_used: str_list(&j, "features_used")?,
            return_mse: j.get("return_mse").and_then(Json::as_f64).unwrap_or(f64::NAN),
            folds: str_list(&j, "folds")?,
            thresholds: nums(field("thresholds")?)?,
            net_by_fold_threshold: match field("net_by_fold_threshold")? {
                Json::Arr(a) => a.iter().map(nums).collect::<Result<_, _>>()?,
                _ => return Err("net_by_fold_threshold must be an array".to_string()),
            },
            net_period: nums(field("net_period")?)?,
        })
    }
}

/// Variables of the run, its arm facts and the finalist expression, read from
/// a search `run.json`.
#[derive(Clone, Debug)]
pub struct RunFacts {
    pub arm: String,
    pub feature_list_sha256: String,
    pub generation_budget: u64,
    pub generations_run: u64,
    pub variables: Vec<String>,
    pub expression: String,
    pub seed: u32,
}

pub fn parse_run(text: &str) -> Result<RunFacts, String> {
    let j = Json::parse(text.trim()).ok_or_else(|| "run.json is not valid JSON".to_string())?;
    let ab = j.get("ablation").ok_or_else(|| "run.json has no 'ablation' object: run with --arm".to_string())?;
    let variables = str_list(&j, "variables")?;
    let hash = req_str(ab, "feature_list_sha256")?;
    if hash != feature_list_sha256(&variables) {
        return Err("run.json feature_list_sha256 does not match its variables".to_string());
    }
    let expression = j
        .get("objective")
        .and_then(|o| o.get("finalist"))
        .and_then(|f| f.get("expression"))
        .and_then(Json::as_str)
        .ok_or_else(|| {
            "run.json has no objective.finalist: run with --objective and a validation slice so the finalist is chosen without the test block"
                .to_string()
        })?
        .to_string();
    Ok(RunFacts {
        arm: req_str(ab, "arm")?,
        feature_list_sha256: hash,
        generation_budget: req_f64(ab, "generation_budget")? as u64,
        generations_run: req_f64(ab, "generations_run")? as u64,
        variables,
        expression,
        seed: req_f64(&j, "seed")? as u32,
    })
}

fn expr_uses(tree: &crate::expr::Node, out: &mut Vec<usize>) {
    if tree.op == crate::expr::Op::Input {
        let i = tree.val as usize;
        if !out.contains(&i) {
            out.push(i);
        }
    }
    for k in &tree.kids {
        expr_uses(k, out);
    }
}

/// Score the finalist expression of a run on every fold. The threshold of a
/// grid entry is the type-7 quantile of the finalist's fit-row scores; the
/// grid is fixed in advance, never chosen on the test slice.
pub fn score_run(
    facts: &RunFacts,
    panel: &Panel,
    folds: &[Fold],
    thresholds: &[f64],
    headline: usize,
    opts: &Options,
) -> Result<SeedResult, String> {
    score_core(facts, panel, folds, thresholds, headline, None, opts)
}

/// [`score_run`] under the book the search selected: one fixed threshold, no quantile.
pub fn score_run_fixed(
    facts: &RunFacts,
    panel: &Panel,
    folds: &[Fold],
    theta: f64,
    opts: &Options,
) -> Result<SeedResult, String> {
    score_core(facts, panel, folds, &[theta], 0, Some(theta), opts)
}

fn score_core(
    facts: &RunFacts,
    panel: &Panel,
    folds: &[Fold],
    thresholds: &[f64],
    headline: usize,
    fixed: Option<f64>,
    opts: &Options,
) -> Result<SeedResult, String> {
    let tree = crate::expr::parse(&facts.expression, &facts.variables)
        .map_err(|e| format!("cannot parse '{}': {}", facts.expression, e))?;
    let cols: Vec<&[f64]> = facts
        .variables
        .iter()
        .map(|v| panel.col(v).ok_or_else(|| format!("variable '{}' is not in the data", v)))
        .collect::<Result<_, _>>()?;
    let mut row = vec![0.0f64; cols.len()];
    let mut score = |i: usize| -> f64 {
        for (k, c) in cols.iter().enumerate() {
            row[k] = c[i];
        }
        crate::expr::eval(&tree, &row)
    };
    let side = opts.cost_bps * 1e-4;
    let mut net_by_fold_threshold = Vec::new();
    let mut net_period: Vec<f64> = Vec::new();
    let mut pooled_score: Vec<f64> = Vec::new();
    let mut pooled_rows: Vec<usize> = Vec::new();
    for f in folds {
        let fit_s: Vec<f64> = f.fit.iter().map(|&i| score(i)).collect();
        let test_s: Vec<f64> = f.test.iter().map(|&i| score(i)).collect();
        let mut per_thr = Vec::new();
        for (ti, &q) in thresholds.iter().enumerate() {
            let theta = fixed.unwrap_or_else(|| crate::walkforward::quantile(&fit_s, q));
            let pos: Vec<f64> = test_s.iter().map(|&s| if s > theta { 1.0 } else { 0.0 }).collect();
            let series = ladder::period_net(panel, &f.test, &pos, side);
            per_thr.push(series.iter().sum::<f64>());
            if ti == headline {
                net_period.extend(series);
            }
        }
        net_by_fold_threshold.push(per_thr);
        pooled_score.extend_from_slice(&test_s);
        pooled_rows.extend_from_slice(&f.test);
    }
    let mut used = Vec::new();
    expr_uses(&tree, &mut used);
    used.sort_unstable();
    Ok(SeedResult {
        arm: facts.arm.clone(),
        seed: facts.seed,
        feature_list_sha256: facts.feature_list_sha256.clone(),
        generation_budget: facts.generation_budget,
        generations_run: facts.generations_run,
        features: facts.variables.clone(),
        features_used: used.iter().filter_map(|&i| facts.variables.get(i).cloned()).collect(),
        return_mse: ladder::return_space_mse_rows(panel, &pooled_rows, &pooled_score, opts),
        folds: folds.iter().map(|f| f.label.clone()).collect(),
        thresholds: thresholds.to_vec(),
        net_by_fold_threshold,
        net_period,
    })
}

// ---------------------------------------------------------------------------
// Verdict
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Check {
    pub name: &'static str,
    pub passed: bool,
    pub detail: String,
}

#[derive(Clone, Debug)]
pub struct ControlComparison {
    pub control: String,
    pub gain_mse: f64,
    pub seed_range: f64,
    pub net_diff_mean: f64,
    pub net_diff_ci: (f64, f64),
    pub spa: Option<Spa>,
    pub spa_error: Option<String>,
    pub sign_by_seed: Vec<f64>,
    pub sign_by_fold: Vec<f64>,
    pub sign_by_threshold: Vec<f64>,
    pub checks: Vec<Check>,
}

#[derive(Clone, Debug)]
pub struct FamilyVerdict {
    pub name: String,
    pub treatment: String,
    pub verdict: &'static str,
    pub comparisons: Vec<ControlComparison>,
    pub null_band: Option<(f64, f64)>,
    pub spa_p: f64,
    pub bh_adjusted: f64,
    pub bh_rejected: bool,
    pub dsr: Option<Dsr>,
    pub failed: Vec<String>,
    pub features_used_share: Vec<(String, f64)>,
}

#[derive(Clone, Debug)]
pub struct ArmSummary {
    pub arm: String,
    pub seeds: usize,
    pub mse_mean: f64,
    pub mse_min: f64,
    pub mse_max: f64,
    pub generation_budget: u64,
    pub generations_min: u64,
    pub generations_max: u64,
    pub feature_list_sha256: String,
    pub net_mean: f64,
}

#[derive(Clone, Debug)]
pub struct Verdict {
    pub definition_sha256: String,
    pub arms: Vec<ArmSummary>,
    pub families: Vec<FamilyVerdict>,
    pub n_trials: u64,
    pub seeds: usize,
    pub folds: usize,
    pub bh_q: f64,
    pub pbo: Option<Pbo>,
    pub pbo_error: Option<String>,
    pub notes: Vec<String>,
}

fn range(v: &[f64]) -> f64 {
    let lo = v.iter().copied().fold(f64::INFINITY, f64::min);
    let hi = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    hi - lo
}

fn mean(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len().max(1) as f64
}

fn mean_series(rs: &[&SeedResult]) -> Vec<f64> {
    let n = rs.iter().map(|r| r.net_period.len()).min().unwrap_or(0);
    (0..n).map(|t| mean(&rs.iter().map(|r| r.net_period[t]).collect::<Vec<_>>())).collect()
}

fn flips(v: &[f64]) -> bool {
    v.iter().any(|x| *x > 0.0) && v.iter().any(|x| *x < 0.0)
}

type ByArm<'a> = BTreeMap<String, Vec<&'a SeedResult>>;

fn compare(def: &Definition, by: &ByArm, treat: &str, ctrl: &str, n_trials: u64) -> Result<ControlComparison, String> {
    let t = by.get(treat).ok_or_else(|| format!("no results for arm {}", treat))?;
    let c = by.get(ctrl).ok_or_else(|| format!("no results for arm {}", ctrl))?;
    let ts: Vec<u32> = t.iter().map(|r| r.seed).collect();
    let cs: Vec<u32> = c.iter().map(|r| r.seed).collect();
    if ts != cs {
        return Err(format!("arms {} and {} ran different seeds ({:?} vs {:?})", treat, ctrl, ts, cs));
    }
    let tm: Vec<f64> = t.iter().map(|r| r.return_mse).collect();
    let cm: Vec<f64> = c.iter().map(|r| r.return_mse).collect();
    let gain = mean(&cm) - mean(&tm);
    let seed_range = range(&tm).max(range(&cm));
    let (ta, ca) = (mean_series(t), mean_series(c));
    if ta.len() != ca.len() || ta.len() < 3 {
        return Err(format!("arms {} and {} have different or too few periods", treat, ctrl));
    }
    let d: Vec<f64> = ta.iter().zip(&ca).map(|(a, b)| a - b).collect();
    let block = if def.rule.block_len > 0 {
        def.rule.block_len
    } else {
        inference::auto_block_len(d.len(), 1, 1)
    };
    let bs = inference::block_bootstrap(&d, block, def.rule.draws, def.rule.seed, 1.0, 0.95);
    let (spa, spa_error) = match inference::spa_test(&[d.clone()], block, def.rule.draws, def.rule.seed) {
        Ok(s) => (Some(s), None),
        Err(e) => (None, Some(e)),
    };
    // Signs of the net difference by seed, fold and threshold.
    let (nf, nt) = (t[0].net_by_fold_threshold.len(), t[0].thresholds.len());
    for r in t.iter().chain(c.iter()) {
        if r.net_by_fold_threshold.len() != nf || r.thresholds.len() != nt || r.net_by_fold_threshold.iter().any(|x| x.len() != nt) {
            return Err("arms differ in folds or threshold grid".to_string());
        }
    }
    let cell = |s: usize, f: usize, q: usize| t[s].net_by_fold_threshold[f][q] - c[s].net_by_fold_threshold[f][q];
    let sign_by_seed: Vec<f64> = (0..t.len())
        .map(|s| mean(&(0..nf).flat_map(|f| (0..nt).map(move |q| (f, q))).map(|(f, q)| cell(s, f, q)).collect::<Vec<_>>()))
        .collect();
    let sign_by_fold: Vec<f64> = (0..nf)
        .map(|f| mean(&(0..t.len()).flat_map(|s| (0..nt).map(move |q| (s, q))).map(|(s, q)| cell(s, f, q)).collect::<Vec<_>>()))
        .collect();
    let sign_by_threshold: Vec<f64> = (0..nt)
        .map(|q| mean(&(0..t.len()).flat_map(|s| (0..nf).map(move |f| (s, f))).map(|(s, f)| cell(s, f, q)).collect::<Vec<_>>()))
        .collect();
    let _ = n_trials;
    Ok(ControlComparison {
        control: ctrl.to_string(),
        gain_mse: gain,
        seed_range,
        net_diff_mean: bs.mean,
        net_diff_ci: (bs.mean_ci.lo, bs.mean_ci.hi),
        spa,
        spa_error,
        sign_by_seed,
        sign_by_fold,
        sign_by_threshold,
        checks: Vec::new(),
    })
}

pub fn verdict(def: &Definition, results: &[SeedResult], expected_hashes: &BTreeMap<String, String>) -> Result<Verdict, String> {
    let mut by: ByArm = BTreeMap::new();
    for r in results {
        by.entry(r.arm.clone()).or_default().push(r);
    }
    for v in by.values_mut() {
        v.sort_by_key(|r| r.seed);
    }
    let mut notes = Vec::new();
    let mut budget: Option<u64> = None;
    let mut arms = Vec::new();
    for (name, rs) in &by {
        let arm = def.arm(name)?;
        let want = feature_list_sha256(&def.features(arm));
        for r in rs {
            if r.feature_list_sha256 != want {
                return Err(format!("arm {} seed {}: feature_list_sha256 differs from the definition", name, r.seed));
            }
            if let Some(h) = expected_hashes.get(name) {
                if *h != r.feature_list_sha256 {
                    return Err(format!("arm {} seed {}: feature_list_sha256 differs from the arm manifest", name, r.seed));
                }
            }
            if r.generation_budget != def.generation_budget {
                return Err(format!(
                    "arm {} seed {}: generation budget {} differs from the definition's {}",
                    name, r.seed, r.generation_budget, def.generation_budget
                ));
            }
            budget = Some(r.generation_budget);
        }
        if rs.len() < def.rule.min_seeds {
            return Err(format!("arm {}: {} seeds, the rule needs {}", name, rs.len(), def.rule.min_seeds));
        }
        let mses: Vec<f64> = rs.iter().map(|r| r.return_mse).collect();
        let nets: Vec<f64> = rs.iter().map(|r| r.net_period.iter().sum::<f64>()).collect();
        let gens: Vec<u64> = rs.iter().map(|r| r.generations_run).collect();
        arms.push(ArmSummary {
            arm: name.clone(),
            seeds: rs.len(),
            mse_mean: mean(&mses),
            mse_min: mses.iter().copied().fold(f64::INFINITY, f64::min),
            mse_max: mses.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            generation_budget: def.generation_budget,
            generations_min: *gens.iter().min().unwrap_or(&0),
            generations_max: *gens.iter().max().unwrap_or(&0),
            feature_list_sha256: want,
            net_mean: mean(&nets),
        });
        if gens.iter().any(|g| *g < def.generation_budget) {
            notes.push(format!(
                "arm {}: a run reached fewer generations than the budget ({} to {} of {})",
                name,
                gens.iter().min().unwrap(),
                gens.iter().max().unwrap(),
                def.generation_budget
            ));
        }
    }
    let _ = budget;
    let seeds = by.values().map(|v| v.len()).min().unwrap_or(0);
    let folds = by.values().flat_map(|v| v.first()).map(|r| r.folds.len()).max().unwrap_or(0);
    let n_trials = def.n_trials(seeds, folds);

    // PBO over the arms' seed-mean series.
    let cols: Vec<Vec<f64>> = by.values().map(|rs| mean_series(rs)).collect();
    let minlen = cols.iter().map(|c| c.len()).min().unwrap_or(0);
    let cols: Vec<Vec<f64>> = cols.into_iter().map(|c| c[..minlen].to_vec()).collect();
    let (pbo, pbo_error) = if cols.len() < 2 {
        (None, Some("PBO needs at least two arms".to_string()))
    } else {
        match inference::pbo_cscv(&cols, def.rule.pbo_blocks) {
            Ok(p) => (Some(p), None),
            Err(e) => (None, Some(e)),
        }
    };

    let mut families: Vec<FamilyVerdict> = Vec::new();
    for fam in &def.rule.families {
        let mut comparisons = Vec::new();
        for ctrl in &fam.controls {
            comparisons.push(compare(def, &by, &fam.treatment, ctrl, n_trials)?);
        }
        let null_band = match &fam.null_arm {
            Some((n, c)) => {
                let (nr, cr) = (by.get(n), by.get(c));
                match (nr, cr) {
                    (Some(nr), Some(cr)) => {
                        let gains: Vec<f64> = nr.iter().zip(cr.iter()).map(|(a, b)| b.return_mse - a.return_mse).collect();
                        Some((
                            gains.iter().copied().fold(f64::INFINITY, f64::min),
                            gains.iter().copied().fold(f64::NEG_INFINITY, f64::max),
                        ))
                    }
                    _ => return Err(format!("family {}: no results for null arm {} or {}", fam.name, n, c)),
                }
            }
            None => None,
        };
        let spa_p = comparisons
            .iter()
            .map(|c| c.spa.as_ref().map_or(1.0, |s| s.p_consistent))
            .fold(0.0f64, f64::max);
        let t = &by[&fam.treatment];
        let ta = mean_series(t);
        let sharpes: Vec<f64> = t.iter().map(|r| inference::sharpe_per_period(&r.net_period)).collect();
        let finite: Vec<f64> = sharpes.iter().copied().filter(|s| s.is_finite()).collect();
        let var_sr = {
            let sd = inference::sample_sd(&finite);
            sd * sd
        };
        let dsr = if ta.len() >= 4 {
            Some(inference::deflated_sharpe(
                inference::sharpe_per_period(&ta),
                ta.len() as f64,
                inference::skewness(&ta),
                inference::kurtosis(&ta),
                n_trials as f64,
                var_sr,
            ))
        } else {
            None
        };
        let mut usage: BTreeMap<String, usize> = BTreeMap::new();
        for r in t {
            for f in &r.features_used {
                *usage.entry(f.clone()).or_default() += 1;
            }
        }
        families.push(FamilyVerdict {
            name: fam.name.clone(),
            treatment: fam.treatment.clone(),
            verdict: "drop",
            comparisons,
            null_band,
            spa_p,
            bh_adjusted: f64::NAN,
            bh_rejected: false,
            dsr,
            failed: Vec::new(),
            features_used_share: usage.into_iter().map(|(k, v)| (k, v as f64 / t.len() as f64)).collect(),
        });
    }
    let ps: Vec<f64> = families.iter().map(|f| f.spa_p).collect();
    let bh = inference::bh_fdr(&ps, def.rule.fdr_q);
    for (k, f) in families.iter_mut().enumerate() {
        f.bh_adjusted = bh.adjusted[k];
        f.bh_rejected = bh.rejected[k];
        let mut failed: Vec<String> = Vec::new();
        for c in f.comparisons.iter_mut() {
            c.checks.push(Check {
                name: "gain_exceeds_seed_range",
                passed: c.gain_mse > c.seed_range && c.gain_mse > 0.0,
                detail: format!("gain {} against seed range {}", jnum(c.gain_mse), jnum(c.seed_range)),
            });
            let p = c.spa.as_ref().map_or(1.0, |s| s.p_consistent);
            c.checks.push(Check {
                name: "spa_p_below_alpha",
                passed: p < def.rule.spa_alpha,
                detail: format!("SPA p {} against alpha {}", jnum(p), jnum(def.rule.spa_alpha)),
            });
            let flip = flips(&c.sign_by_seed) || flips(&c.sign_by_fold) || flips(&c.sign_by_threshold);
            c.checks.push(Check {
                name: "no_sign_flip",
                passed: !flip,
                detail: format!(
                    "sign flip by seed {}, fold {}, threshold {}",
                    flips(&c.sign_by_seed),
                    flips(&c.sign_by_fold),
                    flips(&c.sign_by_threshold)
                ),
            });
            if let Some((_, hi)) = f.null_band {
                c.checks.push(Check {
                    name: "outside_null_band",
                    passed: c.gain_mse > hi,
                    detail: format!("gain {} against the null band upper edge {}", jnum(c.gain_mse), jnum(hi)),
                });
            }
            for ch in &c.checks {
                if !ch.passed {
                    failed.push(format!("{} vs {}: {}", ch.name, c.control, ch.detail));
                }
            }
        }
        if !f.bh_rejected {
            failed.push(format!("fdr_survives: adjusted p {} at q {}", jnum(f.bh_adjusted), jnum(def.rule.fdr_q)));
        }
        match &f.dsr {
            Some(d) if d.dsr >= def.rule.min_dsr => {}
            Some(d) => failed.push(format!("dsr_above_min: DSR {} below {}", jnum(d.dsr), jnum(def.rule.min_dsr))),
            None => failed.push("dsr_above_min: too few periods".to_string()),
        }
        match &pbo {
            Some(p) if p.pbo < def.rule.max_pbo => {}
            Some(p) => failed.push(format!("pbo: PBO {} not below {}", jnum(p.pbo), jnum(def.rule.max_pbo))),
            None => failed.push(format!("pbo: unavailable ({})", pbo_error.as_deref().unwrap_or("no result"))),
        }
        f.verdict = if failed.is_empty() { "ship" } else { "drop" };
        f.failed = failed;
    }
    notes.push(
        "the ablation is a selection on training data repeated inside every fold: each run selects its finalist on a validation slice and is scored on the fold's test slice"
            .to_string(),
    );
    notes.push(format!(
        "N_trials = families x columns x windows x seeds x folds x horizons = {} x {} x {} x {} x {} x {}",
        def.rule.families.len(),
        {
            let mut cols: Vec<String> = Vec::new();
            for a in &def.arms {
                cols.extend(def.features(a));
            }
            cols.sort();
            cols.dedup();
            cols.len()
        },
        def.windows,
        seeds,
        folds,
        def.horizons
    ));
    Ok(Verdict {
        definition_sha256: def.sha256.clone(),
        arms,
        families,
        n_trials,
        seeds,
        folds,
        bh_q: def.rule.fdr_q,
        pbo,
        pbo_error,
        notes,
    })
}

impl Verdict {
    pub fn json(&self) -> String {
        let arms: Vec<String> = self
            .arms
            .iter()
            .map(|a| {
                format!(
                    "{{\"arm\":{},\"seeds\":{},\"return_mse_mean\":{},\"return_mse_min\":{},\"return_mse_max\":{},\
\"generation_budget\":{},\"generations_reached_min\":{},\"generations_reached_max\":{},\"feature_list_sha256\":{},\"net_mean\":{}}}",
                    jstr(&a.arm),
                    a.seeds,
                    jnum(a.mse_mean),
                    jnum(a.mse_min),
                    jnum(a.mse_max),
                    a.generation_budget,
                    a.generations_min,
                    a.generations_max,
                    jstr(&a.feature_list_sha256),
                    jnum(a.net_mean)
                )
            })
            .collect();
        let fams: Vec<String> = self
            .families
            .iter()
            .map(|f| {
                let cmps: Vec<String> = f
                    .comparisons
                    .iter()
                    .map(|c| {
                        let checks: Vec<String> = c
                            .checks
                            .iter()
                            .map(|k| format!("{{\"check\":{},\"passed\":{},\"detail\":{}}}", jstr(k.name), k.passed, jstr(&k.detail)))
                            .collect();
                        let spa = match &c.spa {
                            Some(s) => format!(
                                "{{\"p_consistent\":{},\"p_lower\":{},\"p_upper\":{},\"statistic\":{},\"block_len\":{},\"draws\":{}}}",
                                jnum(s.p_consistent),
                                jnum(s.p_lower),
                                jnum(s.p_upper),
                                jnum(s.statistic),
                                s.block_len,
                                s.draws
                            ),
                            None => format!("{{\"error\":{}}}", jstr(c.spa_error.as_deref().unwrap_or(""))),
                        };
                        let arr = |v: &[f64]| format!("[{}]", v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(","));
                        format!(
                            "{{\"control\":{},\"return_mse_gain\":{},\"seed_range\":{},\"net_diff_mean\":{},\"net_diff_ci\":[{},{}],\
\"spa\":{},\"net_diff_by_seed\":{},\"net_diff_by_fold\":{},\"net_diff_by_threshold\":{},\"checks\":[{}]}}",
                            jstr(&c.control),
                            jnum(c.gain_mse),
                            jnum(c.seed_range),
                            jnum(c.net_diff_mean),
                            jnum(c.net_diff_ci.0),
                            jnum(c.net_diff_ci.1),
                            spa,
                            arr(&c.sign_by_seed),
                            arr(&c.sign_by_fold),
                            arr(&c.sign_by_threshold),
                            checks.join(",")
                        )
                    })
                    .collect();
                let dsr = match &f.dsr {
                    Some(d) => format!(
                        "{{\"sharpe_per_period\":{},\"periods\":{},\"n_trials\":{},\"var_sr\":{},\"sr0\":{},\"dsr\":{}}}",
                        jnum(d.sharpe),
                        jnum(d.t),
                        jnum(d.n_trials),
                        jnum(d.var_sr),
                        jnum(d.sr0),
                        jnum(d.dsr)
                    ),
                    None => "null".to_string(),
                };
                format!(
                    "{{\"family\":{},\"treatment\":{},\"verdict\":{},\"failed\":[{}],\"null_band\":{},\"spa_p\":{},\
\"bh_adjusted\":{},\"bh_rejected\":{},\"dsr\":{},\"comparisons\":[{}],\"feature_usage_share\":{{{}}}}}",
                    jstr(&f.name),
                    jstr(&f.treatment),
                    jstr(f.verdict),
                    f.failed.iter().map(|x| jstr(x)).collect::<Vec<_>>().join(","),
                    f.null_band.map_or("null".to_string(), |(a, b)| format!("[{},{}]", jnum(a), jnum(b))),
                    jnum(f.spa_p),
                    jnum(f.bh_adjusted),
                    f.bh_rejected,
                    dsr,
                    cmps.join(","),
                    f.features_used_share.iter().map(|(k, v)| format!("{}:{}", jstr(k), jnum(*v))).collect::<Vec<_>>().join(",")
                )
            })
            .collect();
        let pbo = match &self.pbo {
            Some(p) => format!(
                "{{\"candidates\":{},\"blocks\":{},\"combinations\":{},\"pbo\":{}}}",
                p.candidates,
                p.blocks,
                p.combinations,
                jnum(p.pbo)
            ),
            None => format!("{{\"error\":{}}}", jstr(self.pbo_error.as_deref().unwrap_or(""))),
        };
        format!(
            "{{\"schema\":{},\"definition_sha256\":{},\"n_trials\":{},\"seeds\":{},\"folds\":{},\"bh_q\":{},\"pbo\":{},\
\"notes\":[{}],\"arms\":[{}],\"families\":[{}]}}",
            jstr(VERDICT_SCHEMA),
            jstr(&self.definition_sha256),
            self.n_trials,
            self.seeds,
            self.folds,
            jnum(self.bh_q),
            pbo,
            self.notes.iter().map(|x| jstr(x)).collect::<Vec<_>>().join(","),
            arms.join(","),
            fams.join(",")
        )
    }

    pub fn markdown(&self) -> String {
        let f = |v: f64| if v.is_finite() { format!("{:.6}", v) } else { "-".to_string() };
        let mut s = String::from("# Ablation verdict\n\n");
        s.push_str(&format!(
            "N_trials {} ({} seeds, {} folds). Definition SHA-256 {}.\n\n",
            self.n_trials, self.seeds, self.folds, self.definition_sha256
        ));
        s.push_str("| arm | seeds | return MSE mean [min, max] | generations reached / budget | feature list SHA-256 |\n|---|---|---|---|---|\n");
        for a in &self.arms {
            s.push_str(&format!(
                "| {} | {} | {} [{}, {}] | {}..{} / {} | {} |\n",
                a.arm,
                a.seeds,
                f(a.mse_mean),
                f(a.mse_min),
                f(a.mse_max),
                a.generations_min,
                a.generations_max,
                a.generation_budget,
                &a.feature_list_sha256[..16.min(a.feature_list_sha256.len())]
            ));
        }
        s.push_str("\n| family | treatment | verdict | SPA p | BH adjusted p | DSR | null band | failed checks |\n|---|---|---|---|---|---|---|---|\n");
        for x in &self.families {
            s.push_str(&format!(
                "| {} | {} | {} | {} | {} | {} | {} | {} |\n",
                x.name,
                x.treatment,
                x.verdict,
                f(x.spa_p),
                f(x.bh_adjusted),
                x.dsr.as_ref().map_or("-".to_string(), |d| f(d.dsr)),
                x.null_band.map_or("-".to_string(), |(a, b)| format!("[{}, {}]", f(a), f(b))),
                if x.failed.is_empty() { "none".to_string() } else { x.failed.join("; ") }
            ));
        }
        for x in &self.families {
            s.push_str(&format!("\n## {}: {}\n\n", x.name, x.verdict));
            for c in &x.comparisons {
                s.push_str(&format!(
                    "* {} against {}: MSE gain {} (seed range {}), net difference {} [{}, {}].\n",
                    x.treatment,
                    c.control,
                    f(c.gain_mse),
                    f(c.seed_range),
                    f(c.net_diff_mean),
                    f(c.net_diff_ci.0),
                    f(c.net_diff_ci.1)
                ));
            }
        }
        if let Some(p) = &self.pbo {
            s.push_str(&format!("\nPBO over {} arms: {}.\n", p.candidates, f(p.pbo)));
        }
        for n in &self.notes {
            s.push_str(&format!("\n{}.\n", n));
        }
        s
    }
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

fn usage() -> &'static str {
    "usage: rcsr ablation <plan|prepare|score|verdict> [options]\n\
\n\
  plan      --definition <file> --data <path> --target <col> --time <col> --windows <file> --out <dir>\n\
            [--group <col>] [--return-col <col>] [--scheme expanding] [--finalist]\n\
            [--search-args \"...\"] [--score-args \"...\"] [--verdict-args \"...\"]\n\
            writes <arm>.arm.json (arm, feature list, SHA-256, generation budget, seeds), plan.json,\n\
            a copy of the definition and commands.sh\n\
  prepare   --definition <file> --data <csv|rcsb> --time <col> --group <col> --out <csv> [--lockbox START,END]\n\
            writes <csv>.lockbox.json (with the row count of each segment) when the data (or --lockbox) declares a locked test;\n\
            placebo and permuted columns are built separately before, inside and after the locked test\n\
            [--keep a,b] [--seed 0]\n\
            writes the projection CSV with placebo and circular-block-permuted columns\n\
  score     (--run <run.json> | --from-run <run.json>) --data <csv|rcsb> --target <col> --time <col> --windows <file> --out <file>\n\
            [--lockbox START,END|none]  a fold ending after START minus --purge-seconds (same resolution) is refused (exit 3)\n\
            [--allow-in-sample]  a fold whose test starts before the run's last seen row + purge is refused (exit 3); with the flag it is labelled oof:false and left out of every pooled series\n\
            [--group <col>] [--return-col <col>] [--scheme expanding] [--thresholds 0.6,0.7,0.8]\n\
            [--headline-threshold 0.7] [--cost-bps 12 | --cost-override <bp>] [--purge-seconds 219600] [--target-spec <file>]\n\
            scores the run's finalist (objective.finalist.expression) on every fold. --target-spec sets the per-side cost\n\
            (cost_bps / 2), the return units (log is converted with exp_m1) and the return column; a --cost-bps that differs is refused.\n\
            --from-run prices the book the search selected: the run's target spec, per-side cost and fixed threshold\n\
            (objective.spec.threshold); it refuses --thresholds, --headline-threshold and --target-spec, and a run without an objective block\n
  verdict   --definition <file> --results <dir> [--plan <dir>] [--out <file>] [--markdown <file|->]\n\
            ship or drop per family under the committed decision rule\n"
}

fn io_err(p: &str, e: std::io::Error) -> (i32, String) {
    (3, format!("{}: {}", p, e))
}

pub fn run(args: &[String]) -> ! {
    match run_inner(args) {
        Ok(()) => std::process::exit(0),
        Err((code, msg)) => {
            eprintln!("error: {}", msg);
            std::process::exit(code)
        }
    }
}

struct Flags {
    map: BTreeMap<String, String>,
    switches: Vec<String>,
}

fn parse_flags(args: &[String], switches: &[&str]) -> Result<Flags, (i32, String)> {
    let mut map = BTreeMap::new();
    let mut sw = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if switches.contains(&a.as_str()) {
            sw.push(a.clone());
        } else if a.starts_with("--") {
            i += 1;
            let v = args.get(i).ok_or_else(|| (2, format!("{} requires a value", a)))?;
            map.insert(a.clone(), v.clone());
        } else {
            return Err((2, format!("unknown argument '{}'", a)));
        }
        i += 1;
    }
    Ok(Flags { map, switches: sw })
}

impl Flags {
    fn need(&self, k: &str) -> Result<&str, (i32, String)> {
        self.map.get(k).map(|s| s.as_str()).ok_or_else(|| (2, format!("{} is required", k)))
    }
    fn opt(&self, k: &str) -> Option<&str> {
        self.map.get(k).map(|s| s.as_str())
    }
    fn has(&self, k: &str) -> bool {
        self.switches.iter().any(|s| s == k)
    }
}

fn load_definition(path: &str) -> Result<Definition, (i32, String)> {
    let text = std::fs::read_to_string(path).map_err(|e| io_err(path, e))?;
    parse_definition(&text).map_err(|e| (3, format!("{}: {}", path, e)))
}

fn run_inner(args: &[String]) -> Result<(), (i32, String)> {
    if args.is_empty() || args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", usage());
        return if args.is_empty() { Err((2, "a subcommand is required".to_string())) } else { Ok(()) };
    }
    let rest = &args[1..];
    match args[0].as_str() {
        "plan" => {
            let fl = parse_flags(rest, &["--finalist"])?;
            let def = load_definition(fl.need("--definition")?)?;
            let out = fl.need("--out")?;
            std::fs::create_dir_all(out).map_err(|e| io_err(out, e))?;
            let data = fl.need("--data")?;
            check_columns(&def, data)?;
            let files = plan_files(&PlanInput {
                def: &def,
                data,
                target: fl.need("--target")?,
                time: fl.need("--time")?,
                group: fl.opt("--group"),
                return_col: fl.opt("--return-col"),
                windows: fl.need("--windows")?,
                scheme: fl.opt("--scheme").unwrap_or("expanding"),
                out_dir: out,
                finalist: fl.has("--finalist"),
                search_args: fl.opt("--search-args").unwrap_or(""),
                score_args: fl.opt("--score-args").unwrap_or(""),
                verdict_args: fl.opt("--verdict-args").unwrap_or(""),
            });
            for (name, body) in files {
                let p = Path::new(out).join(&name);
                std::fs::write(&p, body + "\n").map_err(|e| io_err(&p.display().to_string(), e))?;
            }
            let def_text = std::fs::read_to_string(fl.need("--definition")?).map_err(|e| io_err("definition", e))?;
            let p = Path::new(out).join("definition.json");
            std::fs::write(&p, def_text).map_err(|e| io_err(&p.display().to_string(), e))?;
            println!("wrote {} arm manifests, plan.json and commands.sh to {}", def.arms.len(), out);
            Ok(())
        }
        "prepare" => {
            let fl = parse_flags(rest, &[])?;
            let def = load_definition(fl.need("--definition")?)?;
            let data = fl.need("--data")?;
            let keep: Vec<String> = fl
                .opt("--keep")
                .map(|s| s.split(',').map(|x| x.trim().to_string()).filter(|x| !x.is_empty()).collect())
                .unwrap_or_default();
            let mut want = def.real_columns();
            want.push(fl.need("--time")?.to_string());
            want.push(fl.need("--group")?.to_string());
            want.extend(keep.iter().cloned());
            want.sort();
            want.dedup();
            let table = if crate::store::is_store_path(data) {
                let fs = crate::store::FileStore::open(data).map_err(|e| (3, e))?;
                let proj = crate::store::Projection::resolve_owned(&fs, &want).map_err(|e| (3, e))?;
                fs.to_table_projected(&proj).map_err(|e| (3, e))?
            } else {
                data::read_table(data).map_err(|e| (3, e))?
            };
            let seed: u32 = fl.opt("--seed").map_or(Ok(0), |s| s.parse().map_err(|_| (2, "--seed: invalid number".to_string())))?;
            let resolved = crate::lockbox::resolve(fl.opt("--lockbox"), data).map_err(|e| (e.exit, e.message))?;
            let range = resolved.sidecar_range();
            let (csv, segments) = prepare_csv_segments(&PrepareInput {
                table: &table,
                def: &def,
                time: fl.need("--time")?,
                group: fl.need("--group")?,
                keep: &keep,
                seed,
                locked: range,
            })
            .map_err(|e| (3, e))?;
            let out = fl.need("--out")?;
            std::fs::write(out, csv).map_err(|e| io_err(out, e))?;
            match &range {
                Some(lb) => {
                    let seg = format!(",\"segments\":{{\"before\":{},\"locked\":{},\"after\":{}}}", segments[0], segments[1], segments[2]);
                    let extra = format!("{}{}", crate::lockbox::also_json(resolved.sidecar_also().as_ref(), "rcsr ablation prepare"), seg);
                    crate::lockbox::write_sidecar_with(out, lb, &format!("rcsr ablation prepare ({})", resolved.source.as_str()), &extra).map_err(|e| (3, e))?;
                }
                None => {
                    let stale = crate::lockbox::sidecar_path(out);
                    if Path::new(&stale).exists() {
                        std::fs::remove_file(&stale).map_err(|e| io_err(&stale, e))?;
                    }
                }
            }
            crate::lockbox::finish_output(out, range.as_ref(), &["lockbox.json"]).map_err(|e| (e.exit, e.message))?;
            println!("wrote {}", out);
            Ok(())
        }
        "score" => {
            let fl = parse_flags(rest, &["--allow-in-sample"])?;
            let from_run = fl.opt("--from-run");
            if from_run.is_some() && fl.opt("--run").is_some() {
                return Err((2, "--run and --from-run are alternatives; --from-run also takes the cost, units and fixed threshold from the run".to_string()));
            }
            let run_path = match from_run {
                Some(p) => p,
                None => fl.need("--run")?,
            };
            let run_text = std::fs::read_to_string(run_path).map_err(|e| io_err(run_path, e))?;
            let facts = parse_run(&run_text).map_err(|e| (3, format!("{}: {}", run_path, e)))?;
            let book = match from_run {
                Some(_) => {
                    for flag in ["--thresholds", "--headline-threshold", "--target-spec"] {
                        if fl.opt(flag).is_some() {
                            return Err((2, format!("{} cannot be combined with --from-run: the run fixes the threshold rule and the target spec", flag)));
                        }
                    }
                    Some(crate::run_book::RunBook::parse(&run_text).map_err(|e| (3, format!("{}: {}", run_path, e)))?)
                }
                None => None,
            };
            let mut opts = Options::default();
            let num_flag = |name: &str| -> Result<Option<f64>, (i32, String)> {
                fl.opt(name).map(|c| c.parse().map_err(|_| (2, format!("{}: invalid number", name)))).transpose()
            };
            let (cost_flag, cost_override) = (num_flag("--cost-bps")?, num_flag("--cost-override")?);
            if let Some(c) = fl.opt("--purge-seconds") {
                opts.purge_seconds = c.parse().map_err(|_| (2, "--purge-seconds: invalid number".to_string()))?;
            }
            let thresholds: Vec<f64> = fl
                .opt("--thresholds")
                .unwrap_or("0.6,0.7,0.8")
                .split(',')
                .map(|s| s.trim().parse::<f64>().map_err(|_| (2, format!("--thresholds: cannot parse '{}'", s))))
                .collect::<Result<_, _>>()?;
            let headline_q: f64 = fl.opt("--headline-threshold").map_or(Ok(0.7), |s| s.parse().map_err(|_| (2, "--headline-threshold: invalid number".to_string())))?;
            let headline = thresholds
                .iter()
                .position(|q| (*q - headline_q).abs() < 1e-12)
                .ok_or_else(|| (2, "--headline-threshold must be one of --thresholds".to_string()))?;
            if let Some(b) = &book {
                opts.spec = Some(b.spec.clone());
            } else if let Some(path) = fl.opt("--target-spec") {
                let text = std::fs::read_to_string(path).map_err(|e| io_err(path, e))?;
                let specs = crate::target::parse_file(&text).map_err(|e| (3, format!("{}: {}", path, e)))?;
                let target = fl.need("--target")?;
                opts.spec = crate::target::entry_for(&specs, target).cloned();
            }
            let (cost, source) = match &book {
                Some(b) => match (cost_flag, cost_override) {
                    (Some(_), Some(_)) => return Err((2, "--cost-bps and --cost-override are alternatives; pass one of them".to_string())),
                    (None, Some(o)) => (o, crate::target::CostSource::Flag),
                    (Some(f), None) if (f - b.per_side_bps).abs() > 1e-12 => {
                        return Err((2, format!("--cost-bps {} differs from the run ({} per side); use --cost-override to price a different cost", f, b.per_side_bps)))
                    }
                    _ => (b.per_side_bps, b.source),
                },
                None => crate::target::resolve_cost(opts.spec.as_ref(), cost_flag, cost_override, ladder::DEFAULT_COST_BPS).map_err(|e| (2, e))?,
            };
            opts.cost_bps = cost;
            if opts.spec.is_some() || source == crate::target::CostSource::Flag {
                opts.cost = Some(crate::target::CostDecl::new(opts.spec.as_ref(), cost, source));
            }
            let target = match (&book, fl.opt("--target")) {
                (Some(b), None) => b.spec.return_column.clone().unwrap_or_else(|| b.spec.column.clone()),
                _ => fl.need("--target")?.to_string(),
            };
            let inp = Input {
                data: fl.need("--data")?.to_string(),
                target,
                ret: fl.opt("--return-col").map(str::to_string),
                time: fl.need("--time")?.to_string(),
                group: fl.opt("--group").map(str::to_string),
                features: facts.variables.clone(),
            };
            let windows_path = fl.need("--windows")?;
            let wtext = std::fs::read_to_string(windows_path).map_err(|e| io_err(windows_path, e))?;
            let rows = protocol::parse_windows_csv(&wtext, Some(fl.opt("--scheme").unwrap_or("expanding"))).map_err(|e| (3, e))?;
            let run_file = crate::run_json::RunFile::parse(&run_text).map_err(|e| (3, format!("{}: {}", run_path, e)))?;
            let labelled: Vec<(String, f64)> = rows.iter().map(|r| (r.window.label.clone(), r.window.test_start)).collect();
            let oof = run_file
                .oof_flags(&labelled, fl.has("--allow-in-sample"))
                .map_err(|e| (3, format!("{}: {}", run_path, e)))?;
            let panel = ladder::load_panel(&inp, &opts).map_err(|e| (3, e))?;
            let resolved = crate::lockbox::resolve(fl.opt("--lockbox"), fl.need("--data")?).map_err(|e| (e.exit, e.message))?;
            let declared = resolved.declared();
            let guard = declared.as_ref().map(|d| (d.lockbox.start as f64, d.lockbox.end as f64));
            let folds = ladder::make_folds(&panel, &rows, opts.purge_seconds, guard).map_err(|e| (3, e))?;
            if let Some(extra) = resolved.also {
                ladder::make_folds(&panel, &rows, opts.purge_seconds, Some((extra.start as f64, extra.end as f64))).map_err(|e| (3, e))?;
            }
            let all_labels: Vec<String> = folds.iter().map(|f| f.label.clone()).collect();
            let folds: Vec<ladder::Fold> = folds.into_iter().zip(&oof).filter(|(_, ok)| **ok).map(|(f, _)| f).collect();
            if folds.is_empty() {
                return Err((3, "no out-of-fold folds: every fold overlaps the rows the run trained or selected on".to_string()));
            }
            let res = match &book {
                Some(b) => score_run_fixed(&facts, &panel, &folds, b.threshold, &opts),
                None => score_run(&facts, &panel, &folds, &thresholds, headline, &opts),
            }
            .map_err(|e| (3, e))?;
            let cost_decl = opts.cost.clone().unwrap_or(crate::target::CostDecl {
                per_side_bps: cost,
                source,
                return_units: "simple".to_string(),
                spec_per_side_bps: None,
            });
            let rule = match &book {
                Some(b) => format!("{{\"kind\":\"fixed\",\"threshold\":{}}}", jnum(b.threshold)),
                None => "{\"kind\":\"quantile\"}".to_string(),
            };
            let json = res.json();
            let fold_oof = format!(
                "[{}]",
                all_labels
                    .iter()
                    .zip(&oof)
                    .map(|(l, ok)| format!("{{\"fold\":{},\"oof\":{}}}", jstr(l), ok))
                    .collect::<Vec<_>>()
                    .join(",")
            );
            let mut json = format!(
                "{},\"cost\":{},\"threshold_rule\":{},\"fold_oof\":{}}}",
                &json[..json.len() - 1],
                cost_decl.json(),
                rule,
                fold_oof
            );
            json = resolved.splice_report(&json, 0, false, true);
            let out = fl.need("--out")?;
            std::fs::write(out, json + "\n").map_err(|e| io_err(out, e))?;
            println!("wrote {}", out);
            Ok(())
        }
        "verdict" => {
            let fl = parse_flags(rest, &[])?;
            let def = load_definition(fl.need("--definition")?)?;
            let dir = fl.need("--results")?;
            let mut results = Vec::new();
            let mut names: Vec<_> = std::fs::read_dir(dir)
                .map_err(|e| io_err(dir, e))?
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.to_string_lossy().ends_with(".result.json"))
                .collect();
            names.sort();
            for p in names {
                let text = std::fs::read_to_string(&p).map_err(|e| io_err(&p.display().to_string(), e))?;
                results.push(SeedResult::parse(&text).map_err(|e| (3, format!("{}: {}", p.display(), e)))?);
            }
            if results.is_empty() {
                return Err((3, format!("{}: no *.result.json files", dir)));
            }
            let mut expected = BTreeMap::new();
            if let Some(plan) = fl.opt("--plan") {
                for arm in &def.arms {
                    let p = Path::new(plan).join(format!("{}.arm.json", arm.name));
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        if let Some(h) = crate::manifest::string_field(&text, "feature_list_sha256") {
                            expected.insert(arm.name.clone(), h);
                        }
                    }
                }
            }
            let v = verdict(&def, &results, &expected).map_err(|e| (3, e))?;
            if let Some(out) = fl.opt("--out") {
                std::fs::write(out, v.json() + "\n").map_err(|e| io_err(out, e))?;
            }
            if let Some(md) = fl.opt("--markdown") {
                if md == "-" {
                    print!("{}", v.markdown());
                } else {
                    std::fs::write(md, v.markdown()).map_err(|e| io_err(md, e))?;
                }
            }
            if fl.opt("--out").is_none() && fl.opt("--markdown").is_none() {
                println!("{}", v.json());
            }
            Ok(())
        }
        other => Err((2, format!("unknown subcommand '{}'", other))),
    }
}

/// Every real column of the definition must exist in the data header.
fn check_columns(def: &Definition, data: &str) -> Result<(), (i32, String)> {
    let names: Vec<String> = if crate::store::is_store_path(data) {
        crate::store::FileStore::open(data).map_err(|e| (3, e))?.header().names()
    } else {
        let f = std::fs::File::open(data).map_err(|e| io_err(data, e))?;
        let mut line = String::new();
        std::io::BufRead::read_line(&mut std::io::BufReader::new(f), &mut line).map_err(|e| io_err(data, e))?;
        line.trim_end().split(',').map(|s| s.trim().trim_matches('"').to_string()).collect()
    };
    let missing: Vec<String> = def.real_columns().into_iter().filter(|c| !names.contains(c)).collect();
    if missing.is_empty() {
        Ok(())
    } else {
        Err((3, format!("{}: missing columns {}", data, missing.join(","))))
    }
}

/// Helper for tests: parse a table and build the `Input` the scorer needs.
pub fn panel_for(table: &Table, time: &str, group: Option<&str>, target: &str, features: &[String]) -> Result<Panel, String> {
    ladder::panel_from_table(
        table,
        &Input {
            data: String::new(),
            target: target.to_string(),
            ret: None,
            time: time.to_string(),
            group: group.map(str::to_string),
            features: features.to_vec(),
        },
        target,
        &[],
    )
}
