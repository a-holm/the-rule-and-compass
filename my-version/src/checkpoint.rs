//! Versioned, deterministic checkpoints for a search (issue #17).
//!
//! A checkpoint carries **everything that influences the trajectory** of a
//! search: the search configuration, the completed generation, the per-island
//! populations and RNG states, the per-island stagnation counters, the patience
//! bookkeeping, the Pareto archive and a fingerprint of the training and test
//! data. Resuming from it therefore continues the exact same stream of random
//! numbers and the exact same populations, not an approximation of them.
//!
//! The format is plain UTF-8 text (no third-party parser), versioned by its
//! first line:
//!
//! ```text
//! rcsr-checkpoint v1
//! ```
//!
//! Everything after that line is a sequence of records, one per line:
//!
//! | record | meaning |
//! |---|---|
//! | `config <key> <value>` | one search setting (see [`Config`]) |
//! | `generations <n>` | generations completed |
//! | `stopped_early <bool>` | the saved run had already stopped |
//! | `best_prev <f64>` / `no_improve <n>` | patience bookkeeping |
//! | `data train|test <rows> <hash>` | data fingerprint |
//! | `archive <members> <cap>` | archive header |
//! | `sol <complexity> <cost> <error> <tree>` | one archive member |
//! | `error_kind f32_proxy|f64` | #256: what the `sol` and `ind` errors are |
//! | `archive_hash <hex>` | #256: `Archive::hash` of the `sol` members |
//! | `final <members> <cap> <hash>` | #256: the post-refit f64 archive header |
//! | `fsol <complexity> <cost> <error> <tree>` | #256: one post-refit member |
//! | `build <field> <value>` | #256: version, commit, binary_sha256, device, driver, threads |
//! | `island <i> <stagnant>` | per-island counter |
//! | `rng <i|main> <idx> <624 words>` | exact MT19937 state |
//! | `ind <i> <error> <complexity> <tree>` | one population member |
//!
//! Trees are encoded recursively as `op_index val arity child...`, where
//! `op_index` is [`Op::index`] and `val` is the exact floating-point value
//! (`{:?}`, or `inf`/`-inf`/`NaN`). `expr::to_string` is deliberately not
//! used: it rounds constants, which would change the search.
//!
//! The loader is strict: an unknown version, a truncated file, a bad operator
//! index, an arity mismatch, a duplicate record, a missing field or a data
//! fingerprint mismatch is a clear error, never a panic.

use crate::complexity::ComplexityModel;
use crate::data::Sample;
use crate::expr::{AggKind, Node, Op};
use crate::metrics::Metric;
use crate::pareto::{Archive, Sol};
use crate::provenance::BuildIdentity;
use crate::rng::Mt;
use crate::search::Config;
// #25: units are part of the saved search settings, so a resumed run keeps the
// same dimensional filter.
use crate::units::{Unit, Units};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::path::Path;

/// The exact first line of the format this build reads and writes.
pub const FORMAT_LINE: &str = "rcsr-checkpoint v1";

/// MT19937 state words carried by every `rng` record.
const STATE_WORDS: usize = 624;

/// One population member, as stored in a checkpoint.
#[derive(Clone, Debug)]
pub struct Individual {
    pub tree: Node,
    pub error: f64,
    /// Configured complexity under the saved [`Config::complexity_model`].
    pub complexity: f64,
}

/// One island: its RNG stream and its population, in order.
pub struct IslandState {
    pub rng: Mt,
    pub population: Vec<Individual>,
}

/// A fingerprint of a sample: its row count and a hash of every stored value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fingerprint {
    pub rows: usize,
    pub hash: u64,
}

/// The evaluator that produced a saved search state (issue #140).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    Gpu,
    /// A checkpoint written before the backend was recorded. Such a state is
    /// still resumable: there is no saved identity to check.
    Unknown,
}

impl Backend {
    pub fn as_str(self) -> &'static str {
        match self {
            Backend::Cpu => "cpu",
            Backend::Gpu => "gpu",
            Backend::Unknown => "unknown",
        }
    }

    pub fn parse(s: &str) -> Result<Backend, String> {
        match s {
            "cpu" => Ok(Backend::Cpu),
            "gpu" => Ok(Backend::Gpu),
            "unknown" => Ok(Backend::Unknown),
            other => Err(format!(
                "backend must be 'cpu', 'gpu' or 'unknown', got '{}'",
                other
            )),
        }
    }
}

/// #256: what the stored `sol` and `ind` errors are.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind {
    /// Scored by the f32 device kernel, optimistic by up to `3e-7` relative.
    F32Proxy,
    /// Exact CPU f64.
    F64,
    /// A checkpoint written before the kind was recorded.
    Unknown,
}

impl ErrorKind {
    pub fn as_str(self) -> &'static str {
        match self {
            ErrorKind::F32Proxy => "f32_proxy",
            ErrorKind::F64 => "f64",
            ErrorKind::Unknown => "unknown",
        }
    }

    pub fn parse(s: &str) -> Result<ErrorKind, String> {
        match s {
            "f32_proxy" => Ok(ErrorKind::F32Proxy),
            "f64" => Ok(ErrorKind::F64),
            "unknown" => Ok(ErrorKind::Unknown),
            other => Err(format!(
                "error_kind must be 'f32_proxy', 'f64' or 'unknown', got '{}'",
                other
            )),
        }
    }

    /// The kind a search on `backend` stores in its archive and populations.
    pub fn for_backend(backend: Backend) -> ErrorKind {
        match backend {
            Backend::Gpu => ErrorKind::F32Proxy,
            Backend::Cpu => ErrorKind::F64,
            Backend::Unknown => ErrorKind::Unknown,
        }
    }
}

/// A complete, resumable search state.
pub struct Checkpoint {
    /// The search settings that produced the saved state (including the
    /// generation budget).
    pub config: Config,
    /// The evaluator that produced this state: `cpu` or `gpu` (issue #140).
    pub backend: Backend,
    /// FNV-1a hash of the canonical, trajectory-relevant configuration
    /// (issue #140). The generation budget is excluded because resuming may
    /// raise it.
    pub config_hash: u64,
    /// Generations completed, i.e. the next generation to run.
    pub generations: usize,
    /// True when the saved run declared itself finished (solved or patience).
    /// Resuming such a checkpoint does not run more generations. A time-budget
    /// stop is never terminal (#256): the budget is a property of one process.
    pub stopped_early: bool,
    /// Best error seen so far, for the patience comparison.
    pub best_prev: f64,
    /// Consecutive generations without improvement.
    pub no_improve: usize,
    pub train: Fingerprint,
    pub test: Fingerprint,
    pub archive: Archive,
    pub islands: Vec<IslandState>,
    /// Per-island stagnation counter (`stagnant[i]`).
    pub stagnant: Vec<usize>,
    /// The RNG used to build the initial populations. Kept for completeness;
    /// a resumed search never re-runs initialisation.
    pub main_rng: Mt,
    /// #256: whether `archive` and the populations hold f32 proxy errors.
    pub error_kind: ErrorKind,
    /// #256: the post-refit archive in exact f64, exactly as the run reported
    /// it. Absent in a mid-run checkpoint. A resume ignores it: the search
    /// continues from `archive`.
    pub final_archive: Option<Archive>,
    /// #256: the process that wrote the checkpoint. `None` for a file written
    /// before the identity was recorded.
    pub identity: Option<BuildIdentity>,
    /// #345: sha256 of the declared target spec the run used, or `None` for a
    /// run without one. A resume must present the same spec.
    pub target_spec: Option<String>,
}

/// The operators in [`Op::index`] order. The array index is exactly
/// `op.index()`.
const OP_BY_INDEX: [Op; Op::COUNT] = [
    Op::Const,
    Op::Input,
    Op::Neg,
    Op::Exp,
    Op::Ln,
    Op::Abs,
    Op::Sqrt,
    Op::Sin,
    Op::Cos,
    Op::Tan,
    Op::Sinh,
    Op::Cosh,
    Op::Tanh,
    Op::Logistic,
    Op::Step,
    Op::Gauss,
    Op::Square,
    Op::Cube,
    Op::Floor,
    Op::Ceil,
    Op::Round,
    Op::Sign,
    Op::Erf,
    Op::Erfc,
    Op::Asinh,
    Op::Acosh,
    Op::Atanh,
    Op::Add,
    Op::Sub,
    Op::Mul,
    Op::Div,
    Op::Pow,
    Op::Mod,
    Op::Min,
    Op::Max,
    Op::Atan2,
    Op::If,
    Op::Equal,
    Op::Less,
    Op::LessOrEqual,
    Op::Greater,
    Op::GreaterOrEqual,
    Op::And,
    Op::Or,
    Op::Xor,
    Op::Not,
    Op::Require,
    Op::Between,
    Op::Constrain,
    Op::Complexity,
    Op::Variables,
    Op::Constants,
    Op::Contains,
    Op::ContainsOp,
    Op::Agg(AggKind::Mean),
    Op::Agg(AggKind::Std),
    Op::Agg(AggKind::Median),
    Op::Agg(AggKind::Iqr),
    Op::Agg(AggKind::Sum),
    Op::Agg(AggKind::Min),
    Op::Agg(AggKind::Max),
    Op::Agg(AggKind::Count),
    Op::Custom0,
    Op::Custom1,
    Op::Custom2,
    Op::Custom3,
];

fn op_by_index(i: usize) -> Option<Op> {
    OP_BY_INDEX.get(i).copied()
}

/// Encode one tree exactly: `op_index val arity child...` (prefix order).
///
/// This is the inverse of [`decode_tree`] and preserves every bit of every
/// constant and input index. It is public so the round-trip contract can be
/// tested directly.
pub fn encode_tree(n: &Node) -> String {
    let mut out = String::new();
    write_tree(n, &mut out);
    out
}

fn write_tree(n: &Node, out: &mut String) {
    if !out.is_empty() {
        out.push(' ');
    }
    let _ = write!(
        out,
        "{} {} {}",
        n.op.index(),
        encode_f64(n.val),
        n.kids.len()
    );
    for k in &n.kids {
        write_tree(k, out);
    }
}

/// Decode a tree written by [`encode_tree`], checking every operator index and
/// every arity. Trailing tokens are an error, so a corrupt line cannot decode
/// into a plausible-looking tree.
///
/// The tree is subject to the same [`Limits`](crate::parse::Limits) as parsed
/// text: at most 4096 nodes and a root-to-leaf depth of 256 (issue #153).
pub fn decode_tree(text: &str) -> Result<Node, String> {
    decode_tree_with_limits(text, crate::parse::Limits::default())
}

/// [`decode_tree`] with explicit node and depth limits.
pub fn decode_tree_with_limits(
    text: &str,
    limits: crate::parse::Limits,
) -> Result<Node, String> {
    let tokens: Vec<&str> = text.split_whitespace().collect();
    let mut pos = 0usize;
    let mut nodes = 0usize;
    let tree = parse_tree(&tokens, &mut pos, 1, &mut nodes, limits)?;
    if pos != tokens.len() {
        return Err(format!(
            "tree has {} trailing token(s) after the root",
            tokens.len() - pos
        ));
    }
    Ok(tree)
}

fn parse_tree(
    tokens: &[&str],
    pos: &mut usize,
    depth: usize,
    nodes: &mut usize,
    limits: crate::parse::Limits,
) -> Result<Node, String> {
    if depth > limits.depth {
        return Err(format!(
            "checkpoint tree depth {} exceeds the limit of {}",
            depth, limits.depth
        ));
    }
    *nodes += 1;
    if *nodes > limits.nodes {
        return Err(format!(
            "checkpoint tree has more than {} nodes",
            limits.nodes
        ));
    }
    let op_tok = next_token(tokens, pos, "operator index")?;
    let index: usize = op_tok
        .parse()
        .map_err(|_| format!("bad operator index '{}'", op_tok))?;
    let op = op_by_index(index).ok_or_else(|| {
        format!(
            "bad operator index {} (this build has {} operators, 0..={})",
            index,
            Op::COUNT,
            Op::COUNT - 1
        )
    })?;
    let val_tok = next_token(tokens, pos, "value")?;
    let val = decode_f64(val_tok)?;
    let arity_tok = next_token(tokens, pos, "arity")?;
    let arity: usize = arity_tok
        .parse()
        .map_err(|_| format!("bad arity '{}' for operator {}", arity_tok, op.symbol()))?;
    if arity != op.arity() {
        return Err(format!(
            "operator {} ({}) has arity {}, checkpoint says {}",
            index,
            op.symbol(),
            op.arity(),
            arity
        ));
    }
    let mut kids = Vec::with_capacity(arity);
    for _ in 0..arity {
        kids.push(parse_tree(tokens, pos, depth + 1, nodes, limits)?);
    }
    Ok(Node {
        op,
        val,
        kids,
        block: None,
    })
}

fn next_token<'a>(
    tokens: &[&'a str],
    pos: &mut usize,
    what: &str,
) -> Result<&'a str, String> {
    let token = tokens
        .get(*pos)
        .copied()
        .ok_or_else(|| format!("truncated tree: expected {}", what))?;
    *pos += 1;
    Ok(token)
}

fn encode_f64(v: f64) -> String {
    if v.is_nan() {
        "NaN".to_string()
    } else if v == f64::INFINITY {
        "inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        // Rust's `Debug` for f64 is the shortest representation that parses
        // back to the identical bit pattern, including `-0.0`.
        format!("{:?}", v)
    }
}

fn decode_f64(s: &str) -> Result<f64, String> {
    match s {
        "NaN" => Ok(f64::NAN),
        "inf" => Ok(f64::INFINITY),
        "-inf" => Ok(f64::NEG_INFINITY),
        _ => s
            .parse::<f64>()
            .map_err(|_| format!("bad floating-point value '{}'", s)),
    }
}

/// Hash a sample with FNV-1a over its dimensions and every value's bit pattern.
///
/// Two samples have the same fingerprint exactly when they have the same shape
/// and the same stored `x`, `y` and weight bits. Row order matters, as it does
/// for the search.
pub fn fingerprint(s: &Sample) -> Fingerprint {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    fn mix(h: &mut u64, v: u64) {
        *h ^= v;
        *h = h.wrapping_mul(PRIME);
    }
    let mut h = OFFSET;
    mix(&mut h, s.x.len() as u64);
    mix(&mut h, s.y.len() as u64);
    mix(&mut h, s.w.len() as u64);
    for row in &s.x {
        mix(&mut h, row.len() as u64);
        for v in row {
            mix(&mut h, v.to_bits());
        }
    }
    for v in &s.y {
        mix(&mut h, v.to_bits());
    }
    // #15: extra targets are part of the data a checkpoint was fitted on.
    // Guarded so a single-output fingerprint stays byte-for-byte compatible.
    if !s.ys.is_empty() {
        mix(&mut h, s.ys.len() as u64);
        for col in &s.ys {
            mix(&mut h, col.len() as u64);
            for v in col {
                mix(&mut h, v.to_bits());
            }
        }
    }
    for v in &s.w {
        mix(&mut h, v.to_bits());
    }
    Fingerprint {
        rows: s.x.len(),
        hash: h,
    }
}

/// The canonical configuration text written into a checkpoint (issue #140).
///
/// This is the single source of truth for [`config_hash`]: both the writer and
/// the reader derive the hash from these exact bytes, so a field cannot be
/// saved without being hashed.
pub fn config_canonical(cfg: &Config) -> String {
    let mut out = String::new();
    write_config(cfg, &mut out);
    out
}

/// FNV-1a hash of the trajectory-relevant configuration.
///
/// `generations` is a budget, not a property of the trajectory (resuming may
/// raise it), so it is the one canonical line excluded from the hash.
pub fn config_hash(cfg: &Config) -> u64 {
    hash_config_text(&config_canonical(cfg))
}

/// Hash a canonical configuration block, excluding the `generations` budget
/// line (see [`config_hash`]).
pub fn hash_config_text(text: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let mut h = OFFSET;
    for line in text.lines() {
        if line.starts_with("config generations ") {
            continue;
        }
        for &b in line.as_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(PRIME);
        }
        h ^= b'\n' as u64;
        h = h.wrapping_mul(PRIME);
    }
    h
}

impl Checkpoint {
    /// Serialise the checkpoint to the versioned text format.
    pub fn to_text(&self) -> String {
        self.to_text_for(self.target_spec.as_deref())
    }

    /// [`Checkpoint::to_text`] recording `target_spec` (the sha256 of the
    /// declared target spec of the writing run) instead of the stored one.
    pub fn to_text_for(&self, target_spec: Option<&str>) -> String {
        let mut out = String::new();
        writeln!(out, "{}", FORMAT_LINE).unwrap();
        write_config(&self.config, &mut out);
        // #140: backend identity and the canonical config hash. Both are
        // optional on read so checkpoints written before this feature still
        // load; new checkpoints always carry them.
        writeln!(out, "backend {}", self.backend.as_str()).unwrap();
        writeln!(out, "config_hash {:016x}", self.config_hash).unwrap();

        writeln!(out, "generations {}", self.generations).unwrap();
        writeln!(out, "stopped_early {}", self.stopped_early).unwrap();
        writeln!(out, "best_prev {}", encode_f64(self.best_prev)).unwrap();
        writeln!(out, "no_improve {}", self.no_improve).unwrap();
        writeln!(out, "data train {} {}", self.train.rows, self.train.hash).unwrap();
        writeln!(out, "data test {} {}", self.test.rows, self.test.hash).unwrap();
        writeln!(out, "error_kind {}", self.error_kind.as_str()).unwrap();
        if let Some(id) = &self.identity {
            writeln!(out, "build version {}", id.version).unwrap();
            writeln!(out, "build commit {}", id.commit).unwrap();
            writeln!(out, "build binary_sha256 {}", id.binary_sha256).unwrap();
            writeln!(out, "build device {}", id.device).unwrap();
            writeln!(out, "build driver {}", id.driver).unwrap();
            writeln!(out, "build threads {}", id.threads).unwrap();
        }
        if let Some(sha) = target_spec {
            writeln!(out, "target_spec {}", sha).unwrap();
        }
        writeln!(out, "archive_hash {:016x}", self.archive.hash()).unwrap();
        writeln!(out, "archive {} {}", self.archive.items.len(), self.archive.cap).unwrap();
        for sol in &self.archive.items {
            writeln!(
                out,
                "sol {} {} {} {}",
                sol.complexity,
                encode_f64(sol.cost),
                encode_f64(sol.error),
                encode_tree(&sol.tree)
            )
            .unwrap();
        }
        if let Some(fin) = &self.final_archive {
            writeln!(out, "final {} {} {:016x}", fin.items.len(), fin.cap, fin.hash()).unwrap();
            for sol in &fin.items {
                writeln!(
                    out,
                    "fsol {} {} {} {}",
                    sol.complexity,
                    encode_f64(sol.cost),
                    encode_f64(sol.error),
                    encode_tree(&sol.tree)
                )
                .unwrap();
            }
        }

        let (words, idx) = self.main_rng.state();
        let mut line = format!("rng main {} ", idx);
        for (i, w) in words.iter().enumerate() {
            if i > 0 {
                line.push(' ');
            }
            let _ = write!(line, "{}", w);
        }
        writeln!(out, "{}", line).unwrap();

        for (i, isl) in self.islands.iter().enumerate() {
            writeln!(out, "island {} {}", i, self.stagnant.get(i).copied().unwrap_or(0)).unwrap();
            let (words, idx) = isl.rng.state();
            let mut line = format!("rng {} {} ", i, idx);
            for (j, w) in words.iter().enumerate() {
                if j > 0 {
                    line.push(' ');
                }
                let _ = write!(line, "{}", w);
            }
            writeln!(out, "{}", line).unwrap();
            for ind in &isl.population {
                writeln!(
                    out,
                    "ind {} {} {} {}",
                    i,
                    encode_f64(ind.error),
                    encode_f64(ind.complexity),
                    encode_tree(&ind.tree)
                )
                .unwrap();
            }
        }
        out
    }

    /// Parse a checkpoint. Every failure mode is a descriptive `Err`.
    pub fn from_text(text: &str) -> Result<Checkpoint, String> {
        let mut lines = text.lines();
        let first = lines
            .next()
            .ok_or_else(|| "checkpoint is empty (expected 'rcsr-checkpoint v1')".to_string())?;
        if first != FORMAT_LINE {
            if first.starts_with("rcsr-checkpoint v") {
                return Err(format!(
                    "unsupported checkpoint version '{}'; this build reads '{}'",
                    first, FORMAT_LINE
                ));
            }
            return Err(format!(
                "not an rcsr checkpoint: expected first line '{}', got '{}'",
                FORMAT_LINE, first
            ));
        }

        let mut config_fields: BTreeMap<String, String> = BTreeMap::new();
        let mut backend = Backend::Unknown;
        let mut backend_seen = false;
        let mut stored_config_hash: Option<u64> = None;
        let mut generations: Option<usize> = None;
        let mut stopped_early: Option<bool> = None;
        let mut best_prev: Option<f64> = None;
        let mut no_improve: Option<usize> = None;
        let mut train: Option<Fingerprint> = None;
        let mut test: Option<Fingerprint> = None;
        let mut archive_header: Option<(usize, usize)> = None;
        let mut sols: Vec<Sol> = Vec::new();
        let mut island_stagnant: Vec<Option<usize>> = Vec::new();
        let mut island_rngs: Vec<Option<Mt>> = Vec::new();
        let mut island_pops: Vec<Option<Vec<Individual>>> = Vec::new();
        let mut main_rng: Option<Mt> = None;
        let mut error_kind: Option<ErrorKind> = None;
        let mut stored_archive_hash: Option<u64> = None;
        let mut build_fields: BTreeMap<String, String> = BTreeMap::new();
        let mut target_spec: Option<String> = None;
        let mut final_header: Option<(usize, usize, u64)> = None;
        let mut final_sols: Vec<Sol> = Vec::new();

        for (lineno, line) in lines.enumerate() {
            let lineno = lineno + 2; // first line is line 1
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let parts: Vec<&str> = line.split_whitespace().collect();
            match parts[0] {
                "config" => {
                    let rest = line
                        .strip_prefix("config ")
                        .ok_or_else(|| format!("line {}: malformed config record", lineno))?;
                    let (key, value) = rest.split_once(' ').ok_or_else(|| {
                        format!("line {}: config record '{}' has no value", lineno, key_hint(rest))
                    })?;
                    if config_fields
                        .insert(key.to_string(), value.trim().to_string())
                        .is_some()
                    {
                        return Err(format!("line {}: duplicate config field '{}'", lineno, key));
                    }
                }
                "backend" => {
                    if backend_seen {
                        return Err(format!("line {}: duplicate 'backend' record", lineno));
                    }
                    backend_seen = true;
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: backend has no value", lineno))?;
                    backend = Backend::parse(v)
                        .map_err(|e| format!("line {}: {}", lineno, e))?;
                }
                "config_hash" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: config_hash has no value", lineno))?;
                    let h = u64::from_str_radix(v, 16)
                        .map_err(|_| format!("line {}: bad config_hash '{}'", lineno, v))?;
                    set_once(&mut stored_config_hash, h, "config_hash", lineno)?;
                }
                "generations" => {
                    set_once(
                        &mut generations,
                        parse_num::<usize>(&parts, 1, lineno, "generations")?,
                        "generations",
                        lineno,
                    )?;
                }
                "stopped_early" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: stopped_early has no value", lineno))?;
                    let b = match *v {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(format!(
                                "line {}: stopped_early must be true or false, got '{}'",
                                lineno, v
                            ))
                        }
                    };
                    set_once(&mut stopped_early, b, "stopped_early", lineno)?;
                }
                "best_prev" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: best_prev has no value", lineno))?;
                    set_once(&mut best_prev, decode_f64(v)?, "best_prev", lineno)?;
                }
                "no_improve" => {
                    set_once(
                        &mut no_improve,
                        parse_num::<usize>(&parts, 1, lineno, "no_improve")?,
                        "no_improve",
                        lineno,
                    )?;
                }
                "data" => {
                    let kind = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: data record needs train|test", lineno))?;
                    let rows = parse_num::<usize>(&parts, 2, lineno, "data rows")?;
                    let hash = parse_num::<u64>(&parts, 3, lineno, "data hash")?;
                    let fp = Fingerprint { rows, hash };
                    match *kind {
                        "train" => set_once(&mut train, fp, "data train", lineno)?,
                        "test" => set_once(&mut test, fp, "data test", lineno)?,
                        other => {
                            return Err(format!(
                                "line {}: data record must be train or test, got '{}'",
                                lineno, other
                            ))
                        }
                    }
                }
                "archive" => {
                    let members = parse_num::<usize>(&parts, 1, lineno, "archive members")?;
                    let cap = parse_num::<usize>(&parts, 2, lineno, "archive cap")?;
                    if cap == 0 {
                        return Err(format!("line {}: archive cap must be at least 1", lineno));
                    }
                    set_once(&mut archive_header, (members, cap), "archive", lineno)?;
                }
                "error_kind" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: error_kind has no value", lineno))?;
                    let kind = ErrorKind::parse(v).map_err(|e| format!("line {}: {}", lineno, e))?;
                    set_once(&mut error_kind, kind, "error_kind", lineno)?;
                }
                "archive_hash" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: archive_hash has no value", lineno))?;
                    let h = u64::from_str_radix(v, 16)
                        .map_err(|_| format!("line {}: bad archive_hash '{}'", lineno, v))?;
                    set_once(&mut stored_archive_hash, h, "archive_hash", lineno)?;
                }
                "build" => {
                    let rest = line
                        .strip_prefix("build ")
                        .ok_or_else(|| format!("line {}: malformed build record", lineno))?;
                    let (key, value) = rest
                        .split_once(' ')
                        .ok_or_else(|| format!("line {}: build record '{}' has no value", lineno, key_hint(rest)))?;
                    if build_fields
                        .insert(key.to_string(), value.trim().to_string())
                        .is_some()
                    {
                        return Err(format!("line {}: duplicate build field '{}'", lineno, key));
                    }
                }
                "target_spec" => {
                    let v = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: target_spec has no value", lineno))?;
                    set_once(&mut target_spec, v.to_string(), "target_spec", lineno)?;
                }
                "final" => {
                    let members = parse_num::<usize>(&parts, 1, lineno, "final members")?;
                    let cap = parse_num::<usize>(&parts, 2, lineno, "final cap")?;
                    let hash = parts
                        .get(3)
                        .and_then(|v| u64::from_str_radix(v, 16).ok())
                        .ok_or_else(|| format!("line {}: bad final archive hash", lineno))?;
                    if cap == 0 {
                        return Err(format!("line {}: final cap must be at least 1", lineno));
                    }
                    set_once(&mut final_header, (members, cap, hash), "final", lineno)?;
                }
                "fsol" => {
                    if final_header.is_none() {
                        return Err(format!(
                            "line {}: fsol record before the final header",
                            lineno
                        ));
                    }
                    let complexity = parse_num::<usize>(&parts, 1, lineno, "fsol complexity")?;
                    let cost = parse_f64_at(&parts, 2, lineno, "fsol cost")?;
                    let error = parse_f64_at(&parts, 3, lineno, "fsol error")?;
                    let tree_text = parts.get(4..).filter(|t| !t.is_empty()).ok_or_else(|| {
                        format!("line {}: fsol record has no tree", lineno)
                    })?;
                    let tree = decode_tree(&tree_text.join(" "))
                        .map_err(|e| format!("line {}: {}", lineno, e))?;
                    final_sols.push(Sol { complexity, cost, error, tree });
                }
                "sol" => {
                    if archive_header.is_none() {
                        return Err(format!(
                            "line {}: sol record before the archive header",
                            lineno
                        ));
                    }
                    let complexity = parse_num::<usize>(&parts, 1, lineno, "sol complexity")?;
                    let cost = parse_f64_at(&parts, 2, lineno, "sol cost")?;
                    let error = parse_f64_at(&parts, 3, lineno, "sol error")?;
                    let tree_text = parts.get(4..).ok_or_else(|| {
                        format!("line {}: sol record has no tree", lineno)
                    })?;
                    if tree_text.is_empty() {
                        return Err(format!("line {}: sol record has no tree", lineno));
                    }
                    let tree = decode_tree(&tree_text.join(" "))
                        .map_err(|e| format!("line {}: {}", lineno, e))?;
                    sols.push(Sol {
                        complexity,
                        cost,
                        error,
                        tree,
                    });
                }
                "island" => {
                    let i = parse_num::<usize>(&parts, 1, lineno, "island index")?;
                    let stagnant = parse_num::<usize>(&parts, 2, lineno, "island stagnant")?;
                    set_slot(
                        &mut island_stagnant,
                        i,
                        stagnant,
                        "island",
                        lineno,
                    )?;
                }
                "rng" => {
                    let target = parts
                        .get(1)
                        .ok_or_else(|| format!("line {}: rng record has no target", lineno))?;
                    let idx = parse_num::<usize>(&parts, 2, lineno, "rng index")?;
                    if idx > STATE_WORDS {
                        return Err(format!(
                            "line {}: rng index {} is outside 0..={}",
                            lineno, idx, STATE_WORDS
                        ));
                    }
                    let values = &parts[3.min(parts.len())..];
                    if values.len() != STATE_WORDS {
                        return Err(format!(
                            "line {}: rng record has {} state words, expected {}",
                            lineno,
                            values.len(),
                            STATE_WORDS
                        ));
                    }
                    let mut words = [0u32; STATE_WORDS];
                    for (k, v) in values.iter().enumerate() {
                        words[k] = v
                            .parse()
                            .map_err(|_| format!("line {}: bad rng word '{}'", lineno, v))?;
                    }
                    let rng = Mt::from_state(words, idx);
                    if *target == "main" {
                        if main_rng.is_some() {
                            return Err(format!("line {}: duplicate main rng record", lineno));
                        }
                        main_rng = Some(rng);
                    } else {
                        let i: usize = target.parse().map_err(|_| {
                            format!("line {}: rng target must be 'main' or an island index, got '{}'", lineno, target)
                        })?;
                        set_slot(&mut island_rngs, i, rng, "rng", lineno)?;
                    }
                }
                "ind" => {
                    let i = parse_num::<usize>(&parts, 1, lineno, "ind island")?;
                    let error = parse_f64_at(&parts, 2, lineno, "ind error")?;
                    let complexity = parse_f64_at(&parts, 3, lineno, "ind complexity")?;
                    let tree_text = parts
                        .get(4..)
                        .ok_or_else(|| format!("line {}: ind record has no tree", lineno))?;
                    if tree_text.is_empty() {
                        return Err(format!("line {}: ind record has no tree", lineno));
                    }
                    let tree = decode_tree(&tree_text.join(" "))
                        .map_err(|e| format!("line {}: {}", lineno, e))?;
                    if island_pops.len() <= i {
                        island_pops.resize_with(i + 1, || None);
                    }
                    island_pops[i]
                        .get_or_insert_with(Vec::new)
                        .push(Individual {
                            tree,
                            error,
                            complexity,
                        });
                }
                other => {
                    return Err(format!(
                        "line {}: unknown checkpoint record '{}'",
                        lineno, other
                    ));
                }
            }
        }

        let config = build_config(&config_fields)?;
        let computed_hash = config_hash(&config);
        if let Some(stored) = stored_config_hash {
            if stored != computed_hash {
                return Err(format!(
                    "checkpoint config_hash mismatch: file says {:016x}, the saved \
                     configuration hashes to {:016x}; the checkpoint was modified or \
                     was written by a build whose configuration encoding differs",
                    stored, computed_hash
                ));
            }
        }
        let generations = generations
            .ok_or_else(|| "checkpoint is incomplete: missing 'generations'".to_string())?;
        let stopped_early = stopped_early
            .ok_or_else(|| "checkpoint is incomplete: missing 'stopped_early'".to_string())?;
        let best_prev = best_prev
            .ok_or_else(|| "checkpoint is incomplete: missing 'best_prev'".to_string())?;
        let no_improve = no_improve
            .ok_or_else(|| "checkpoint is incomplete: missing 'no_improve'".to_string())?;
        let train = train.ok_or_else(|| "checkpoint is incomplete: missing 'data train'".to_string())?;
        let test = test.ok_or_else(|| "checkpoint is incomplete: missing 'data test'".to_string())?;
        let (members, cap) = archive_header
            .ok_or_else(|| "checkpoint is incomplete: missing 'archive' header".to_string())?;
        if sols.len() != members {
            return Err(format!(
                "checkpoint is truncated or inconsistent: archive header says {} member(s), found {}",
                members,
                sols.len()
            ));
        }
        if generations > config.generations {
            return Err(format!(
                "checkpoint is inconsistent: {} generations completed but the saved budget is {}",
                generations, config.generations
            ));
        }

        let expected_islands = config.islands.max(1);
        let mut islands = Vec::with_capacity(expected_islands);
        let mut stagnant = Vec::with_capacity(expected_islands);
        for i in 0..expected_islands {
            let rng = island_rngs
                .get_mut(i)
                .and_then(|s| s.take())
                .ok_or_else(|| format!("checkpoint is incomplete: island {} has no rng state", i))?;
            let population = island_pops
                .get_mut(i)
                .and_then(|s| s.take())
                .ok_or_else(|| format!("checkpoint is incomplete: island {} has no population", i))?;
            if population.len() != config.pop {
                return Err(format!(
                    "checkpoint is inconsistent: island {} has {} individual(s), saved pop is {}",
                    i,
                    population.len(),
                    config.pop
                ));
            }
            let st = island_stagnant
                .get_mut(i)
                .and_then(|s| s.take())
                .ok_or_else(|| format!("checkpoint is incomplete: island {} has no counter", i))?;
            islands.push(IslandState { rng, population });
            stagnant.push(st);
        }
        if island_rngs.iter().skip(expected_islands).any(|s| s.is_some())
            || island_pops.iter().skip(expected_islands).any(|s| s.is_some())
            || island_stagnant.iter().skip(expected_islands).any(|s| s.is_some())
        {
            return Err(format!(
                "checkpoint has records for more than the saved {} island(s)",
                expected_islands
            ));
        }
        let main_rng =
            main_rng.ok_or_else(|| "checkpoint is incomplete: missing main rng state".to_string())?;

        let mut archive = Archive::new(cap);
        archive.items = sols;
        if let Some(stored) = stored_archive_hash {
            if stored != archive.hash() {
                return Err(format!(
                    "checkpoint is corrupt: archive_hash {:016x} but the stored members hash to {:016x}",
                    stored,
                    archive.hash()
                ));
            }
        }
        let final_archive = match final_header {
            Some((members, fcap, hash)) => {
                if final_sols.len() != members {
                    return Err(format!(
                        "checkpoint is truncated or inconsistent: final header says {} member(s), found {}",
                        members,
                        final_sols.len()
                    ));
                }
                let mut fin = Archive::new(fcap);
                fin.items = final_sols;
                if fin.hash() != hash {
                    return Err(format!(
                        "checkpoint is corrupt: final archive hash {:016x} but the stored members hash to {:016x}",
                        hash,
                        fin.hash()
                    ));
                }
                Some(fin)
            }
            None => None,
        };
        let identity = if build_fields.is_empty() {
            None
        } else {
            let field = |k: &str| -> Result<String, String> {
                build_fields
                    .get(k)
                    .cloned()
                    .ok_or_else(|| format!("checkpoint is incomplete: missing 'build {}'", k))
            };
            Some(BuildIdentity {
                version: field("version")?,
                commit: field("commit")?,
                binary_sha256: field("binary_sha256")?,
                device: field("device")?,
                driver: field("driver")?,
                threads: field("threads")?
                    .parse()
                    .map_err(|_| "checkpoint has a bad 'build threads' value".to_string())?,
            })
        };
        Ok(Checkpoint {
            target_spec,
            config,
            backend,
            config_hash: computed_hash,
            generations,
            stopped_early,
            best_prev,
            no_improve,
            train,
            test,
            archive,
            islands,
            stagnant,
            main_rng,
            error_kind: error_kind.unwrap_or(ErrorKind::Unknown),
            final_archive,
            identity,
        })
    }

    /// Read and parse a checkpoint file.
    pub fn load(path: impl AsRef<Path>) -> Result<Checkpoint, String> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path).map_err(|e| {
            format!(
                "cannot read checkpoint '{}': {}",
                path.display(),
                e
            )
        })?;
        Checkpoint::from_text(&text)
            .map_err(|e| format!("invalid checkpoint '{}': {}", path.display(), e))
    }

    /// Write the checkpoint atomically (temp file + fsync + rename).
    ///
    /// The previous good generation is retained as `<path>.prev`.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), String> {
        self.save_with_fault(path, crate::atomic::Fault::None)
    }

    /// [`Checkpoint::save`] recording the target spec sha256 of the running
    /// process (#345).
    pub fn save_for(
        &self,
        path: impl AsRef<Path>,
        target_spec: Option<&str>,
    ) -> Result<(), String> {
        let path = path.as_ref();
        let text = self.to_text_for(target_spec);
        crate::atomic::write_atomic(path, text.as_bytes(), crate::atomic::Fault::None)
            .map_err(|e| format!("cannot write checkpoint '{}': {}", path.display(), e))
    }

    /// [`Checkpoint::save`] with a deliberate fault, for the crash-safety test.
    pub fn save_with_fault(
        &self,
        path: impl AsRef<Path>,
        fault: crate::atomic::Fault,
    ) -> Result<(), String> {
        let path = path.as_ref();
        let text = self.to_text();
        crate::atomic::write_atomic(path, text.as_bytes(), fault)
            .map_err(|e| format!("cannot write checkpoint '{}': {}", path.display(), e))
    }

    /// Refuse a resume onto a different backend unless the caller explicitly
    /// allows the change (issue #140).
    pub fn check_backend(&self, current: Backend, allow_change: bool) -> Result<(), String> {
        if self.backend == Backend::Unknown {
            // A checkpoint from before #140 cannot be checked; resuming is not
            // silently reinterpreted, because the backend is simply unknown.
            return Ok(());
        }
        if self.backend != current && !allow_change {
            return Err(format!(
                "checkpoint backend mismatch: saved run used '{}', this run uses '{}'; \
                 pass --allow-backend-change to resume across backends (the trajectory \
                 and reported errors will differ)",
                self.backend.as_str(),
                current.as_str()
            ));
        }
        Ok(())
    }

    /// Refuse a resume under a different binary, commit, version, device,
    /// driver or thread count unless the caller explicitly allows it (#256).
    /// A checkpoint written before the identity was recorded cannot be checked.
    pub fn check_identity(
        &self,
        current: &BuildIdentity,
        allow_change: bool,
    ) -> Result<(), String> {
        let Some(saved) = &self.identity else {
            return Ok(());
        };
        let diff = saved.differences(current);
        if diff.is_empty() || allow_change {
            return Ok(());
        }
        Err(format!(
            "checkpoint build identity mismatch ({}); pass --allow-build-change to \
             resume under a different binary, device or thread count (the trajectory \
             may differ)",
            diff.join("; ")
        ))
    }

    /// #345: refuse a resume under a different declared target spec. `current`
    /// is the sha256 of this run's spec, or `None` without one. Both hashes
    /// are named; an absent spec reads as `none`.
    pub fn check_target_spec(&self, current: Option<&str>) -> Result<(), String> {
        if self.target_spec.as_deref() == current {
            return Ok(());
        }
        Err(format!(
            "checkpoint target spec mismatch: saved sha256 {}, this run {}; refusing to \
             resume a search under a different target spec",
            self.target_spec.as_deref().unwrap_or("none"),
            current.unwrap_or("none")
        ))
    }

    /// Refuse a resume onto data other than the data that produced this state.
    pub fn check_data(&self, train: &Sample, test: &Sample) -> Result<(), String> {
        check_fingerprint("training", &self.train, train)?;
        check_fingerprint("test", &self.test, test)
    }
}

fn check_fingerprint(kind: &str, saved: &Fingerprint, sample: &Sample) -> Result<(), String> {
    let now = fingerprint(sample);
    if now != *saved {
        return Err(format!(
            "checkpoint {} data mismatch: saved {} row(s) with fingerprint {:016x}, \
             current data has {} row(s) with fingerprint {:016x}; refusing to resume \
             a search on different data",
            kind, saved.rows, saved.hash, now.rows, now.hash
        ));
    }
    Ok(())
}

fn key_hint(rest: &str) -> &str {
    rest.split_whitespace().next().unwrap_or(rest)
}

fn set_once<T>(
    slot: &mut Option<T>,
    value: T,
    what: &str,
    lineno: usize,
) -> Result<(), String> {
    if slot.is_some() {
        return Err(format!("line {}: duplicate '{}' record", lineno, what));
    }
    *slot = Some(value);
    Ok(())
}

fn set_slot<T>(
    slots: &mut Vec<Option<T>>,
    index: usize,
    value: T,
    what: &str,
    lineno: usize,
) -> Result<(), String> {
    if slots.len() <= index {
        slots.resize_with(index + 1, || None);
    }
    if slots[index].is_some() {
        return Err(format!(
            "line {}: duplicate {} record for index {}",
            lineno, what, index
        ));
    }
    slots[index] = Some(value);
    Ok(())
}

fn parse_num<T: std::str::FromStr>(
    parts: &[&str],
    at: usize,
    lineno: usize,
    what: &str,
) -> Result<T, String> {
    let raw = parts
        .get(at)
        .ok_or_else(|| format!("line {}: {} is missing", lineno, what))?;
    raw.parse()
        .map_err(|_| format!("line {}: bad {} '{}'", lineno, what, raw))
}

fn parse_f64_at(
    parts: &[&str],
    at: usize,
    lineno: usize,
    what: &str,
) -> Result<f64, String> {
    let raw = parts
        .get(at)
        .ok_or_else(|| format!("line {}: {} is missing", lineno, what))?;
    decode_f64(raw).map_err(|e| format!("line {}: {}: {}", lineno, what, e))
}

fn required<'a>(fields: &'a BTreeMap<String, String>, key: &str) -> Result<&'a str, String> {
    fields.get(key).map(|s| s.as_str()).ok_or_else(|| {
        format!(
            "checkpoint is incomplete: missing config field '{}'",
            key
        )
    })
}

fn build_config(fields: &BTreeMap<String, String>) -> Result<Config, String> {
    const KEYS: &[&str] = &[
        "generations",
        "islands",
        "pop",
        "max_complexity",
        "complexity.op_costs",
        "complexity.var_weight",
        "complexity.const_weight",
        "complexity.depth_weight",
        "metric",
        "seed",
        "ops",
        "migration_interval",
        "restart_interval",
        "init_depth",
        "fit_iters",
        "dedup",
        "time_budget",
        "patience",
        "predictor_samples",
    ];
    // #25: optional so a checkpoint written before units existed still loads,
    // and so a units-free run writes exactly the old checkpoint bytes.
    // #85: `fit_samples` is optional for the same reason: missing means 0 (off),
    // which is byte-identical to a checkpoint written before screening existed.
    const OPTIONAL_KEYS: &[&str] = &[
        "units",
        "units.target",
        "require_dimensions",
        "fit_samples",
        "front_parents", // #76
        "fit_strategy",  // #116: missing means `exact`
        "fit_admission", // #223: missing means `screened`
        "fit_rows",      // #116: missing means 0 (off)
        "fit_fallback_max", // #255: missing means the default bound
        "qd_bins",       // #120
        "qd_cell_cap",   // #120
        "qd_cvt",        // #120
        "qd_policy",     // #120
        "qd_control",    // #120
        "variation",     // #122: missing means auto
    ];
    for key in KEYS {
        required(fields, key)?;
    }
    let extras: Vec<&str> = fields
        .keys()
        .map(|s| s.as_str())
        .filter(|k| !KEYS.contains(k) && !OPTIONAL_KEYS.contains(k))
        .collect();
    if !extras.is_empty() {
        return Err(format!(
            "checkpoint has unknown config field(s): {}",
            extras.join(", ")
        ));
    }

    let num_usize = |key: &str| -> Result<usize, String> {
        required(fields, key)?
            .parse()
            .map_err(|_| format!("bad config value for '{}': '{}'", key, required(fields, key).unwrap()))
    };
    let num_u32 = |key: &str| -> Result<u32, String> {
        required(fields, key)?
            .parse()
            .map_err(|_| format!("bad config value for '{}': '{}'", key, required(fields, key).unwrap()))
    };
    let num_f64 = |key: &str| -> Result<f64, String> {
        decode_f64(required(fields, key)?)
    };
    let bool_val = |key: &str| -> Result<bool, String> {
        match required(fields, key)? {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(format!("config '{}' must be true or false, got '{}'", key, other)),
        }
    };

    let metric = Metric::parse_opt(required(fields, "metric")?)
        .ok_or_else(|| format!("unknown metric '{}' in checkpoint", required(fields, "metric").unwrap()))?;

    let mut model = ComplexityModel::default();
    let costs: Vec<&str> = required(fields, "complexity.op_costs")?
        .split_whitespace()
        .collect();
    if costs.len() != Op::COUNT {
        return Err(format!(
            "config 'complexity.op_costs' has {} value(s), expected {}",
            costs.len(),
            Op::COUNT
        ));
    }
    for (i, raw) in costs.iter().enumerate() {
        let cost = decode_f64(raw)
            .map_err(|e| format!("bad op cost at index {}: {}", i, e))?;
        if !cost.is_finite() || cost <= 0.0 {
            return Err(format!(
                "op cost at index {} must be positive and finite, got '{}'",
                i, raw
            ));
        }
        model.set_op_cost(OP_BY_INDEX[i], cost);
    }
    let var = num_f64("complexity.var_weight")?;
    let con = num_f64("complexity.const_weight")?;
    let depth = num_f64("complexity.depth_weight")?;
    let spec = format!(
        "var={},const={},depth={}",
        encode_f64(var),
        encode_f64(con),
        encode_f64(depth)
    );
    model
        .apply_weights(&spec)
        .map_err(|e| format!("bad complexity weights in checkpoint: {}", e))?;

    let mut ops = Vec::new();
    let ops_text = required(fields, "ops")?;
    if !ops_text.is_empty() {
        for raw in ops_text.split(',') {
            let index: usize = raw
                .parse()
                .map_err(|_| format!("bad operator index '{}' in config 'ops'", raw))?;
            let op = op_by_index(index).ok_or_else(|| {
                format!(
                    "bad operator index {} in config 'ops' (0..={})",
                    index,
                    Op::COUNT - 1
                )
            })?;
            ops.push(op);
        }
    }
    if ops.is_empty() {
        return Err("config 'ops' is empty; the search needs at least one operator".to_string());
    }

    let time_budget = match required(fields, "time_budget")? {
        "none" => None,
        raw => Some(
            decode_f64(raw).map_err(|e| format!("bad config value for 'time_budget': {}", e))?,
        ),
    };
    let predictor_samples = match required(fields, "predictor_samples")? {
        "none" => None,
        raw => Some(
            raw.parse()
                .map_err(|_| format!("bad config value for 'predictor_samples': '{}'", raw))?,
        ),
    };
    if matches!(predictor_samples, Some(n) if n < 2) {
        return Err(format!(
            "config 'predictor_samples' must be none or at least 2, got {:?}",
            predictor_samples
        ));
    }
    // #85: missing means off. A present value must be 0 or at least 2, matching
    // the CLI; 1 cannot define a screening subset.
    let fit_samples = match fields.get("fit_samples").map(|s| s.as_str()) {
        None => 0,
        Some(raw) => {
            let n: usize = raw
                .parse()
                .map_err(|_| format!("bad config value for 'fit_samples': '{}'", raw))?;
            if n == 1 {
                return Err(
                    "config 'fit_samples' must be 0 (off) or at least 2, got 1".to_string(),
                );
            }
            n
        }
    };

    // #116: fit-only row screening; missing means off.
    let fit_rows = match fields.get("fit_rows").map(|s| s.as_str()) {
        None => 0,
        Some(raw) => {
            let n: usize = raw
                .parse()
                .map_err(|_| format!("bad config value for 'fit_rows': '{}'", raw))?;
            if n == 1 {
                return Err("config 'fit_rows' must be 0 (off) or at least 2, got 1".to_string());
            }
            n
        }
    };

    // #25: optional fields. Missing means "no units, no filter", which is
    // bit-identical to a checkpoint written before this feature existed.
    if fields.contains_key("units.target") && !fields.contains_key("units") {
        return Err(
            "checkpoint has 'units.target' without 'units'; both or neither are expected"
                .to_string(),
        );
    }
    let units = match fields.get("units").map(|s| s.as_str()) {
        None | Some("none") => None,
        Some(raw) => {
            let vars = Units::decode_vars(raw)
                .map_err(|e| format!("bad config value for 'units': {}", e))?;
            let target = Unit::parse(
                fields
                    .get("units.target")
                    .ok_or_else(|| "checkpoint has 'units' without 'units.target'".to_string())?,
            )
            .map_err(|e| format!("bad config value for 'units.target': {}", e))?;
            Some(Units::new(vars, target))
        }
    };
    let require_dimensions = match fields.get("require_dimensions").map(|s| s.as_str()) {
        None | Some("false") => false,
        Some("true") => true,
        Some(other) => {
            return Err(format!(
                "config 'require_dimensions' must be true or false, got '{}'",
                other
            ))
        }
    };
    // #76: missing means on, the default since the feature was introduced.
    let front_parents = match fields.get("front_parents").map(|s| s.as_str()) {
        None | Some("true") => true,
        Some("false") => false,
        Some(other) => {
            return Err(format!(
                "config 'front_parents' must be true or false, got '{}'",
                other
            ))
        }
    };
    // #116: missing means `exact`, byte-identical to a checkpoint written
    // before the solver portfolio existed.
    let fit_strategy = match fields.get("fit_strategy").map(|s| s.as_str()) {
        None => crate::fit::FitStrategy::Exact,
        Some(raw) => crate::fit::FitStrategy::parse(raw).ok_or_else(|| {
            format!(
                "config 'fit_strategy' must be exact, auto, linear, varpro, stream or gpu, got '{}'",
                raw
            )
        })?,
    };
    let fit_admission = match fields.get("fit_admission").map(|s| s.as_str()) {
        None => crate::fit::FitAdmission::Screened,
        Some(raw) => crate::fit::FitAdmission::parse(raw).ok_or_else(|| {
            format!("config 'fit_admission' must be screened or all, got '{}'", raw)
        })?,
    };
    if require_dimensions && units.is_none() {
        return Err(
            "config 'require_dimensions' is true but 'units' is none; \
             the filter needs declared units"
                .to_string(),
        );
    }
    // #120: absent means the QD layer was off.
    let qd_bins = match fields.get("qd_bins") {
        None => 0,
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| format!("bad config value for 'qd_bins': '{}'", raw))?,
    };
    let qd_cell_cap = match fields.get("qd_cell_cap") {
        None => 4,
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| format!("bad config value for 'qd_cell_cap': '{}'", raw))?,
    };
    if qd_cell_cap == 0 {
        return Err("config 'qd_cell_cap' must be at least 1".to_string());
    }
    let qd_cvt = match fields.get("qd_cvt").map(|s| s.as_str()) {
        None => false,
        Some("true") => true,
        Some("false") => false,
        Some(raw) => return Err(format!("config 'qd_cvt' must be true or false, got '{}'", raw)),
    };
    let qd_policy = match fields.get("qd_policy").map(|s| s.as_str()) {
        None => false,
        Some("true") => true,
        Some("false") => false,
        Some(raw) => {
            return Err(format!(
                "config 'qd_policy' must be true or false, got '{}'",
                raw
            ))
        }
    };
    let qd_control = match fields.get("qd_control") {
        None => 0,
        Some(raw) => raw
            .parse::<u8>()
            .map_err(|_| format!("bad config value for 'qd_control': '{}'", raw))?,
    };
    if qd_control > 2 {
        return Err(format!(
            "config 'qd_control' must be 0, 1 or 2, got {}",
            qd_control
        ));
    }
    // #255: absent means the default bound, matching checkpoints written
    // before the fallback batch was bounded and parallel.
    let fit_fallback_max = match fields.get("fit_fallback_max") {
        None => crate::search::DEFAULT_FIT_FALLBACK_MAX,
        Some(raw) => raw
            .parse::<usize>()
            .map_err(|_| format!("bad config value for 'fit_fallback_max': '{}'", raw))?,
    };
    // #122: absent means auto, matching checkpoints written before the flat
    // postfix engine existed.
    let variation = match fields.get("variation").map(|s| s.as_str()) {
        None => crate::search::VariationMode::Auto,
        Some(raw) => crate::search::VariationMode::parse(raw).map_err(|e| {
            format!("config 'variation' is invalid: {}", e)
        })?,
    };

    Ok(Config {
        generations: num_usize("generations")?,
        islands: num_usize("islands")?,
        pop: num_usize("pop")?,
        max_complexity: num_usize("max_complexity")?,
        complexity_model: model,
        metric,
        seed: num_u32("seed")?,
        ops,
        migration_interval: num_usize("migration_interval")?,
        restart_interval: num_usize("restart_interval")?,
        init_depth: num_usize("init_depth")?,
        fit_iters: num_usize("fit_iters")?,
        fit_strategy, // #116
        fit_admission, // #223
        fit_report: false,
        dedup: bool_val("dedup")?,
        time_budget,
        patience: num_usize("patience")?,
        predictor_samples,
        fit_samples, // #85
        fit_rows,     // #116
        fit_fallback_max, // #255
        // #116: a disk row source is a path choice, not saved search state.
        fit_stream_source: None,
        front_parents, // #76
        units,
        require_dimensions,
        qd_bins,
        qd_cell_cap,
        qd_cvt,
        qd_policy,
        qd_control,
        progress: None,
        dev_log: None, // #211: a reporting hook, never part of the saved state
        front_observer: None, // #462
        // #27/#26: evaluator choices are not saved search state; a resumed run
        // picks the backend afresh from the current build/flags.
        simd: true,
        gpu: false,
        gpu_notes: true,
        gpu_memory: None,
        dispatch: crate::dispatch::DispatchMode::Auto,
        // #119: scheduling only, not saved search state.
        pipeline_depth: 2,
        variation,
    })
}

fn write_config(cfg: &Config, out: &mut String) {
    writeln!(out, "config generations {}", cfg.generations).unwrap();
    writeln!(out, "config islands {}", cfg.islands).unwrap();
    writeln!(out, "config pop {}", cfg.pop).unwrap();
    writeln!(out, "config max_complexity {}", cfg.max_complexity).unwrap();
    let costs: Vec<String> = (0..Op::COUNT)
        .map(|i| encode_f64(cfg.complexity_model.op_cost(OP_BY_INDEX[i])))
        .collect();
    writeln!(out, "config complexity.op_costs {}", costs.join(" ")).unwrap();
    writeln!(
        out,
        "config complexity.var_weight {}",
        encode_f64(cfg.complexity_model.var_weight())
    )
    .unwrap();
    writeln!(
        out,
        "config complexity.const_weight {}",
        encode_f64(cfg.complexity_model.const_weight())
    )
    .unwrap();
    writeln!(
        out,
        "config complexity.depth_weight {}",
        encode_f64(cfg.complexity_model.depth_weight())
    )
    .unwrap();
    writeln!(out, "config metric {}", cfg.metric.name()).unwrap();
    writeln!(out, "config seed {}", cfg.seed).unwrap();
    let ops: Vec<String> = cfg.ops.iter().map(|op| op.index().to_string()).collect();
    writeln!(out, "config ops {}", ops.join(",")).unwrap();
    writeln!(out, "config migration_interval {}", cfg.migration_interval).unwrap();
    writeln!(out, "config restart_interval {}", cfg.restart_interval).unwrap();
    writeln!(out, "config init_depth {}", cfg.init_depth).unwrap();
    writeln!(out, "config fit_iters {}", cfg.fit_iters).unwrap();
    writeln!(out, "config fit_strategy {}", cfg.fit_strategy.name()).unwrap(); // #116
    // #223: written only when it differs from the default, so every other
    // checkpoint and config hash keeps its bytes.
    if cfg.fit_admission != crate::fit::FitAdmission::Screened {
        writeln!(out, "config fit_admission {}", cfg.fit_admission.name()).unwrap();
    }
    writeln!(out, "config fit_rows {}", cfg.fit_rows).unwrap(); // #116
    writeln!(out, "config fit_fallback_max {}", cfg.fit_fallback_max).unwrap(); // #255
    // #122: absent means auto (the historical path for every old checkpoint).
    writeln!(out, "config variation {}", cfg.variation.as_str()).unwrap();
    writeln!(out, "config dedup {}", cfg.dedup).unwrap();
    writeln!(
        out,
        "config time_budget {}",
        match cfg.time_budget {
            Some(v) => encode_f64(v),
            None => "none".to_string(),
        }
    )
    .unwrap();
    writeln!(out, "config patience {}", cfg.patience).unwrap();
    writeln!(
        out,
        "config predictor_samples {}",
        match cfg.predictor_samples {
            Some(n) => n.to_string(),
            None => "none".to_string(),
        }
    )
    .unwrap();
    // #85: only write the field when screening is on, so an off run's
    // checkpoint stays byte-identical to one written before the flag existed.
    if cfg.fit_samples > 0 {
        writeln!(out, "config fit_samples {}", cfg.fit_samples).unwrap();
    }
    // #76: only write the field when front feedback is off, so an on run's
    // checkpoint stays byte-identical to one written before the flag existed.
    if !cfg.front_parents {
        writeln!(out, "config front_parents false").unwrap();
    }
    // #25: only write these when units are active, so a units-free checkpoint
    // is byte-identical to one written before this feature existed.
    if let Some(u) = &cfg.units {
        writeln!(out, "config units {}", u.encode_vars()).unwrap();
        writeln!(out, "config units.target {}", u.target).unwrap();
        writeln!(out, "config require_dimensions {}", cfg.require_dimensions).unwrap();
    }
    // #120: off is the historical default, so only write when enabled.
    if cfg.qd_bins > 0 {
        writeln!(out, "config qd_bins {}", cfg.qd_bins).unwrap();
        writeln!(out, "config qd_cell_cap {}", cfg.qd_cell_cap).unwrap();
        writeln!(out, "config qd_cvt {}", cfg.qd_cvt).unwrap();
        writeln!(out, "config qd_policy {}", cfg.qd_policy).unwrap();
        writeln!(out, "config qd_control {}", cfg.qd_control).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn op_index_table_matches_the_enum_discriminants() {
        // The text format stores `op.index()`; the decoding table must be in
        // exactly that order. A new or reordered variant fails here rather
        // than silently decoding into a different expression.
        for (i, op) in OP_BY_INDEX.iter().enumerate() {
            assert_eq!(op.index(), i, "OP_BY_INDEX is out of order at {}", i);
        }
        assert_eq!(OP_BY_INDEX.len(), Op::COUNT);
    }
}
