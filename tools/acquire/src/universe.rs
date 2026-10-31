//! Deterministic universe filter: archive USDT symbols in, genuine crypto pairs out.
//!
//! Every input is a file. The module does no network access.

use crate::json::{self, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

const DEFAULTS: &str = include_str!("../universe-defaults.toml");

pub const KEPT_FILE: &str = "universe_all_usdt_pit.txt";
pub const EXCLUSIONS_FILE: &str = "symbol_exclusions.csv";

/// Exclusion rules. Lists match the base asset, the symbol without its `USDT` suffix.
#[derive(Debug, Clone)]
pub struct Rules {
    pub stablecoins: BTreeSet<String>,
    pub fiat: BTreeSet<String>,
    pub commodities: BTreeSet<String>,
    pub leveraged_suffixes: Vec<String>,
    pub leveraged_exceptions: BTreeSet<String>,
    pub stablecoin_legacy: BTreeSet<String>,
    pub crypto_b_bases: BTreeSet<String>,
    pub single_letter_equities: BTreeSet<String>,
    pub tag_tokenised_stock: BTreeSet<String>,
    pub tag_stablecoin: BTreeSet<String>,
    pub tag_commodity: BTreeSet<String>,
    pub top_min_base_len: usize,
    pub top_exclude: BTreeSet<String>,
}

enum Item {
    List(Vec<String>),
    Int(usize),
}

impl Rules {
    /// The rules that ship with the tool.
    pub fn defaults() -> Rules {
        let mut r = Rules {
            stablecoins: BTreeSet::new(),
            fiat: BTreeSet::new(),
            commodities: BTreeSet::new(),
            leveraged_suffixes: Vec::new(),
            leveraged_exceptions: BTreeSet::new(),
            stablecoin_legacy: BTreeSet::new(),
            crypto_b_bases: BTreeSet::new(),
            single_letter_equities: BTreeSet::new(),
            tag_tokenised_stock: BTreeSet::new(),
            tag_stablecoin: BTreeSet::new(),
            tag_commodity: BTreeSet::new(),
            top_min_base_len: 0,
            top_exclude: BTreeSet::new(),
        };
        r.apply(DEFAULTS).expect("built-in defaults parse");
        r
    }

    /// Overlays the keys of a TOML-like config text: `key = ["A", "B"]` or `key = 2`.
    pub fn apply(&mut self, text: &str) -> Result<(), String> {
        let items = parse_config(text)?;
        for (key, item) in items {
            match (key.as_str(), item) {
                ("leveraged_suffixes", Item::List(v)) => self.leveraged_suffixes = v,
                ("stablecoins", Item::List(v)) => self.stablecoins = v.into_iter().collect(),
                ("fiat", Item::List(v)) => self.fiat = v.into_iter().collect(),
                ("commodities", Item::List(v)) => self.commodities = v.into_iter().collect(),
                ("leveraged_exceptions", Item::List(v)) => self.leveraged_exceptions = v.into_iter().collect(),
                ("stablecoin_legacy", Item::List(v)) => self.stablecoin_legacy = v.into_iter().collect(),
                ("crypto_b_bases", Item::List(v)) => self.crypto_b_bases = v.into_iter().collect(),
                ("single_letter_equities", Item::List(v)) => self.single_letter_equities = v.into_iter().collect(),
                ("tag_tokenised_stock", Item::List(v)) => self.tag_tokenised_stock = v.into_iter().collect(),
                ("tag_stablecoin", Item::List(v)) => self.tag_stablecoin = v.into_iter().collect(),
                ("tag_commodity", Item::List(v)) => self.tag_commodity = v.into_iter().collect(),
                ("top_exclude", Item::List(v)) => self.top_exclude = v.into_iter().collect(),
                ("top_min_base_len", Item::Int(n)) => self.top_min_base_len = n,
                (k, _) => return Err(format!("config: unknown key or wrong value type for '{k}'")),
            }
        }
        Ok(())
    }
}

fn parse_config(text: &str) -> Result<Vec<(String, Item)>, String> {
    let mut clean = String::new();
    for line in text.lines() {
        let mut in_str = false;
        for c in line.chars() {
            match c {
                '"' => in_str = !in_str,
                '#' if !in_str => break,
                _ => {}
            }
            clean.push(c);
        }
        clean.push('\n');
    }
    let b: Vec<char> = clean.chars().collect();
    let mut i = 0;
    let mut out: Vec<(String, Item)> = Vec::new();
    let skip_ws = |i: &mut usize| {
        while *i < b.len() && b[*i].is_whitespace() {
            *i += 1;
        }
    };
    loop {
        skip_ws(&mut i);
        if i >= b.len() {
            return Ok(out);
        }
        let start = i;
        while i < b.len() && (b[i].is_ascii_alphanumeric() || b[i] == '_') {
            i += 1;
        }
        let key: String = b[start..i].iter().collect();
        if key.is_empty() {
            return Err(format!("config: unexpected '{}'", b[i]));
        }
        if out.iter().any(|(k, _)| *k == key) {
            return Err(format!("config: '{key}' given twice"));
        }
        skip_ws(&mut i);
        if b.get(i) != Some(&'=') {
            return Err(format!("config: '{key}' needs '='"));
        }
        i += 1;
        skip_ws(&mut i);
        match b.get(i) {
            Some('[') => {
                i += 1;
                let mut list = Vec::new();
                loop {
                    skip_ws(&mut i);
                    match b.get(i) {
                        Some(']') => {
                            i += 1;
                            break;
                        }
                        Some('"') => {
                            i += 1;
                            let s = i;
                            while i < b.len() && b[i] != '"' {
                                i += 1;
                            }
                            if i >= b.len() {
                                return Err(format!("config: unterminated string in '{key}'"));
                            }
                            list.push(b[s..i].iter().collect());
                            i += 1;
                            skip_ws(&mut i);
                            match b.get(i) {
                                Some(',') => i += 1,
                                Some(']') => {}
                                _ => return Err(format!("config: expected ',' or ']' in '{key}'")),
                            }
                        }
                        _ => return Err(format!("config: expected a quoted string or ']' in '{key}'")),
                    }
                }
                out.push((key, Item::List(list)));
            }
            Some(c) if c.is_ascii_digit() => {
                let s = i;
                while i < b.len() && b[i].is_ascii_digit() {
                    i += 1;
                }
                let n = b[s..i].iter().collect::<String>().parse().map_err(|_| format!("config: bad number for '{key}'"))?;
                out.push((key, Item::Int(n)));
            }
            _ => return Err(format!("config: '{key}' needs a list or an integer")),
        }
    }
}

/// One excluded symbol and the rule that excluded it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Exclusion {
    pub symbol: String,
    pub rule: &'static str,
    pub detail: String,
}

#[derive(Debug, Default)]
pub struct Outcome {
    pub usdt: usize,
    pub kept: Vec<String>,
    pub exclusions: Vec<Exclusion>,
}

/// Product tags per symbol, from the exchange product list. A symbol present here, even
/// with no tags, is not checked by the equity name heuristic.
pub type Products = BTreeMap<String, BTreeSet<String>>;

/// Splits the archive text into the sorted, unique USDT symbols.
pub fn usdt_symbols(archive: &str) -> Vec<String> {
    let set: BTreeSet<&str> = archive.lines().map(str::trim).filter(|s| s.ends_with("USDT")).collect();
    set.into_iter().map(str::to_string).collect()
}

/// Parses `{"data":[{"s":"BTCUSDT","tags":["pow"]},..]}`.
pub fn parse_products(text: &str) -> Result<Products, String> {
    let v = json::parse(text)?;
    let data = v.get("data").and_then(Value::as_array).ok_or("products: expected an object with a \"data\" array")?;
    let mut out = Products::new();
    for x in data {
        let s = x.get("s").and_then(Value::as_str).ok_or("products: entry without a string \"s\"")?;
        let tags = match x.get("tags") {
            None | Some(Value::Null) => BTreeSet::new(),
            Some(t) => t
                .as_array()
                .ok_or_else(|| format!("products: \"tags\" of {s} is not an array"))?
                .iter()
                .map(|t| t.as_str().map(str::to_string).ok_or_else(|| format!("products: non-string tag for {s}")))
                .collect::<Result<_, _>>()?,
        };
        out.entry(s.to_string()).or_default().extend(tags);
    }
    Ok(out)
}

/// Reads exchange-listed ticker files in the pipe-separated NASDAQ format: a header line,
/// the ticker in the first field, and a `File Creation Time` trailer.
pub fn parse_equities(text: &str, into: &mut BTreeSet<String>) {
    for line in text.lines().skip(1) {
        let first = line.split('|').next().unwrap_or("").trim();
        if !first.is_empty() && first != "File Creation Time" && !first.starts_with("File Creation Time:") {
            into.insert(first.to_uppercase());
        }
    }
}

/// Applies the rules to every USDT symbol. `products` is `None` when no product list was given.
pub fn filter(rules: &Rules, symbols: &[String], products: Option<&Products>, equities: &BTreeSet<String>) -> Outcome {
    let mut out = Outcome { usdt: symbols.len(), ..Outcome::default() };
    for s in symbols {
        let b = &s[..s.len() - 4];
        let mut hits: Vec<(&'static str, String)> = Vec::new();
        if !b.is_ascii() {
            hits.push(("non-ascii", "base has non-ASCII characters".into()));
        }
        let entry = products.and_then(|p| p.get(s));
        if let Some(tags) = entry {
            for (set, rule) in [
                (&rules.tag_tokenised_stock, "tokenised-stock"),
                (&rules.tag_stablecoin, "stablecoin"),
                (&rules.tag_commodity, "commodity-token"),
            ] {
                if let Some(t) = tags.iter().find(|t| set.contains(*t)) {
                    hits.push((rule, format!("product tag {t}")));
                }
            }
        }
        if rules.stablecoins.contains(b) {
            hits.push(("stablecoin", "base in stablecoins".into()));
        }
        if rules.fiat.contains(b) {
            hits.push(("fiat", "base in fiat".into()));
        }
        if rules.commodities.contains(b) {
            hits.push(("commodity-token", "base in commodities".into()));
        }
        if entry.is_none() {
            if b.ends_with('B') && equities.contains(&b[..b.len() - 1]) && !rules.crypto_b_bases.contains(b) {
                hits.push(("tokenised-stock", "base without trailing B is a listed equity ticker".into()));
            } else if rules.single_letter_equities.contains(b) {
                hits.push(("tokenised-stock", "single-letter equity base".into()));
            }
        }
        if let Some(suf) = rules.leveraged_suffixes.iter().find(|x| b.ends_with(x.as_str())) {
            if !rules.leveraged_exceptions.contains(b) {
                hits.push(("leveraged-token", format!("base ends with {suf}")));
            }
        }
        if rules.stablecoin_legacy.contains(b) {
            hits.push(("stablecoin-legacy", "base in stablecoin_legacy".into()));
        }
        if hits.is_empty() {
            out.kept.push(s.clone());
        }
        let mut seen: Vec<&str> = Vec::new();
        for (rule, detail) in hits {
            if !seen.contains(&rule) {
                seen.push(rule);
                out.exclusions.push(Exclusion { symbol: s.clone(), rule, detail });
            }
        }
    }
    out
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

pub fn exclusions_csv(rows: &[Exclusion]) -> String {
    let mut s = String::from("symbol,rule,detail\n");
    for r in rows {
        s.push_str(&format!("{},{},{}\n", csv_field(&r.symbol), r.rule, csv_field(&r.detail)));
    }
    s
}

/// The `n` kept symbols with the highest quote volume, ties broken by symbol. `tickers` is
/// a JSON array of objects with `symbol` and `quoteVolume`.
pub fn top_by_quote_volume(rules: &Rules, kept: &[String], tickers: &str, n: usize) -> Result<Vec<String>, String> {
    let v = json::parse(tickers)?;
    let arr = v.as_array().ok_or("tickers: expected a JSON array")?;
    let kept: BTreeSet<&str> = kept.iter().map(String::as_str).collect();
    let mut rows: Vec<(f64, &str)> = Vec::new();
    for x in arr {
        let Some(sym) = x.get("symbol").and_then(Value::as_str) else { continue };
        if !kept.contains(sym) {
            continue;
        }
        let base = &sym[..sym.len() - 4];
        if base.chars().count() < rules.top_min_base_len || rules.top_exclude.contains(base) {
            continue;
        }
        let vol = match x.get("quoteVolume") {
            Some(Value::Str(t)) | Some(Value::Number(t)) => t.trim().parse::<f64>().ok().filter(|f| f.is_finite()),
            _ => None,
        }
        .ok_or_else(|| format!("tickers: bad quoteVolume for {sym}"))?;
        rows.push((vol, sym));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("finite").then(a.1.cmp(b.1)));
    rows.dedup_by(|a, b| a.1 == b.1);
    Ok(rows.into_iter().take(n).map(|(_, s)| s.to_string()).collect())
}

/// Inputs and outputs of one `universe` run.
pub struct UniverseConfig {
    pub archive: PathBuf,
    pub products: Option<PathBuf>,
    pub equities: Vec<PathBuf>,
    pub config: Option<PathBuf>,
    pub tickers: Option<PathBuf>,
    pub top: Option<usize>,
    pub out: PathBuf,
}

#[derive(Debug)]
pub struct Summary {
    pub usdt: usize,
    pub kept: usize,
    pub by_rule: BTreeMap<&'static str, usize>,
    pub top_file: Option<PathBuf>,
    pub top_count: usize,
}

fn read_text(p: &Path) -> Result<String, String> {
    fs::read(p).map(|b| String::from_utf8_lossy(&b).into_owned()).map_err(|e| format!("cannot read {}: {e}", p.display()))
}

pub fn run(cfg: &UniverseConfig) -> Result<Summary, String> {
    if cfg.top.is_some() != cfg.tickers.is_some() {
        return Err("--top-by-quote-volume and --tickers go together".into());
    }
    let mut rules = Rules::defaults();
    if let Some(p) = &cfg.config {
        rules.apply(&read_text(p)?)?;
    }
    let symbols = usdt_symbols(&read_text(&cfg.archive)?);
    let products = cfg.products.as_deref().map(|p| read_text(p).and_then(|t| parse_products(&t))).transpose()?;
    let mut equities = BTreeSet::new();
    for p in &cfg.equities {
        parse_equities(&read_text(p)?, &mut equities);
    }
    let top = match (&cfg.tickers, cfg.top) {
        (Some(t), Some(n)) => Some((read_text(t)?, n)),
        _ => None,
    };
    let outcome = filter(&rules, &symbols, products.as_ref(), &equities);
    let top_list = top.map(|(t, n)| top_by_quote_volume(&rules, &outcome.kept, &t, n).map(|l| (n, l))).transpose()?;

    fs::create_dir_all(&cfg.out).map_err(|e| format!("cannot create {}: {e}", cfg.out.display()))?;
    let write = |name: &str, text: String| {
        let p = cfg.out.join(name);
        fs::write(&p, text).map_err(|e| format!("cannot write {}: {e}", p.display()))
    };
    write(KEPT_FILE, outcome.kept.iter().map(|s| format!("{s}\n")).collect())?;
    write(EXCLUSIONS_FILE, exclusions_csv(&outcome.exclusions))?;
    let (mut top_file, mut top_count) = (None, 0);
    if let Some((n, list)) = top_list {
        let name = format!("universe_top_{n}_by_quote_volume.txt");
        top_count = list.len();
        write(&name, list.iter().map(|s| format!("{s}\n")).collect())?;
        top_file = Some(cfg.out.join(name));
    }
    let mut by_rule = BTreeMap::new();
    for e in &outcome.exclusions {
        *by_rule.entry(e.rule).or_insert(0) += 1;
    }
    Ok(Summary { usdt: outcome.usdt, kept: outcome.kept.len(), by_rule, top_file, top_count })
}
