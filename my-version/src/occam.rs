//! #1184: description length of a formula under a pre-registered prefix code,
//! the Occam-adjusted p-value and `--select-rule occam`.
//!
//! The code is fixed by `docs/occam-code-v1.json`. A change of that file is a
//! new version and a new trial. Kraft's inequality holds for every part, so
//! `P(some formula with no alpha has p_f <= delta 2^(-L(f))) <= delta`.

use crate::devlog::{num, Json};
use crate::expr::{Node, Op};

/// The versioned spec file, embedded so the hash cannot drift from the code.
pub const SPEC_V1: &str = include_str!("../../docs/occam-code-v1.json");

/// Fewest blocks of block means the t statistic may rest on.
pub const MIN_BLOCKS: usize = 20;

/// The code of one run: the spec file plus the run's alphabet and variables.
#[derive(Clone, Debug)]
pub struct CodeSpec {
    pub version: u32,
    pub file_sha256: String,
    /// Distinct operators of the run (their `Debug` names, unique where symbols collide), sorted.
    pub alphabet: Vec<String>,
    pub variables: usize,
    tier_count: usize,
    grid_tiers: usize,
    grid_limit: f64,
    grid_tol: f64,
    digits: i32,
    exp_min: i32,
    exp_max: i32,
    type_len: [usize; 4],
    c3_form_bits: usize,
    menu_size: usize,
    cash_bits: usize,
    flag_bits: usize,
    dir_bits: usize,
    cut_levels: usize,
}

/// Description length split into structure and constants.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bits {
    pub structure: f64,
    pub constants: f64,
    pub total: f64,
    pub nodes: usize,
    pub variables: usize,
    pub constant_count: usize,
}

impl CodeSpec {
    /// Version 1 of the code for a run with these operators and `variables` search variables.
    pub fn v1(ops: &[Op], variables: usize) -> CodeSpec {
        CodeSpec::from_text(SPEC_V1, ops, variables).expect("the embedded spec parses")
    }

    pub fn from_text(text: &str, ops: &[Op], variables: usize) -> Result<CodeSpec, String> {
        let j = Json::parse(text.trim()).ok_or("occam code spec is not JSON")?;
        let get = |path: &[&str]| -> Result<&Json, String> {
            let mut cur = &j;
            for k in path {
                cur = cur.get(k).ok_or_else(|| format!("occam code spec: missing {}", path.join(".")))?;
            }
            Ok(cur)
        };
        let uint = |path: &[&str]| -> Result<usize, String> { get(path)?.as_usize().ok_or_else(|| format!("occam code spec: {} is not a number", path.join("."))) };
        let float = |path: &[&str]| -> Result<f64, String> { get(path)?.as_f64().ok_or_else(|| format!("occam code spec: {} is not a number", path.join("."))) };
        let mut type_len = [0usize; 4];
        for (k, name) in ["C0", "C1", "C2", "C3"].iter().enumerate() {
            type_len[k] = get(&["composite", "type_codewords", name])?.as_str().ok_or("occam code spec: type codeword")?.len();
        }
        let mut alphabet: Vec<String> = ops.iter().map(|o| format!("{:?}", o)).collect();
        alphabet.sort();
        alphabet.dedup();
        Ok(CodeSpec {
            version: uint(&["version"])? as u32,
            file_sha256: crate::manifest::sha256_hex(text.as_bytes()),
            alphabet,
            variables,
            tier_count: uint(&["constants", "tier_count"])?,
            grid_tiers: uint(&["constants", "grid_tiers"])?,
            grid_limit: float(&["constants", "grid_limit"])?,
            grid_tol: float(&["constants", "grid_tolerance"])?,
            digits: uint(&["constants", "mantissa_digits"])? as i32,
            exp_min: float(&["constants", "exponent_min"])? as i32,
            exp_max: float(&["constants", "exponent_max"])? as i32,
            type_len,
            c3_form_bits: uint(&["composite", "c3_form_bits"])?,
            menu_size: uint(&["composite", "menu_size"])?,
            cash_bits: uint(&["composite", "cash_side_bits"])?,
            flag_bits: uint(&["composite", "branch_flag_bits"])?,
            dir_bits: uint(&["composite", "grid_direction_bits"])?,
            cut_levels: uint(&["composite", "grid_cut_levels"])?,
        })
    }

    /// `A`: the operators plus the two leaf kinds.
    pub fn alphabet_size(&self) -> usize {
        self.alphabet.len() + 2
    }

    fn node_bits(&self) -> f64 {
        (self.alphabet_size() as f64).log2()
    }

    fn variable_bits(&self) -> f64 {
        (self.variables.max(1) as f64).log2()
    }

    fn grid_values(&self, tier: usize) -> f64 {
        2.0 * self.grid_limit * 10f64.powi(tier as i32) + 1.0
    }

    fn snap(&self, v: f64, tier: usize) -> f64 {
        let p = 10f64.powi(tier as i32);
        ((v * p).round() / p).clamp(-self.grid_limit, self.grid_limit)
    }

    fn on_grid(&self, v: f64, tier: usize) -> bool {
        let p = 10f64.powi(tier as i32);
        v.is_finite() && v.abs() <= self.grid_limit + self.grid_tol / p && (v * p - (v * p).round()).abs() <= self.grid_tol
    }

    fn sig_round(&self, v: f64) -> Option<f64> {
        let r: f64 = format!("{:.*e}", (self.digits - 1) as usize, v).parse().ok()?;
        let e = if r == 0.0 { 0 } else { r.abs().log10().floor() as i32 };
        (r.is_finite() && r != 0.0 && e >= self.exp_min && e <= self.exp_max).then_some(r)
    }

    /// Smallest tier that holds `v`, with the bits of its codeword. Tiers
    /// `0..grid_tiers` are grids; the next tier is a sign, `digits`
    /// significant digits and a decimal exponent.
    pub fn constant_bits(&self, v: f64) -> Option<(usize, f64)> {
        let select = (self.tier_count as f64).log2();
        for t in 0..self.grid_tiers {
            if self.on_grid(v, t) {
                return Some((t, select + self.grid_values(t).log2()));
            }
        }
        let r = self.sig_round(v)?;
        if r != v {
            return None;
        }
        let mantissas = 9.0 * 10f64.powi(self.digits - 1);
        let exps = (self.exp_max - self.exp_min + 1) as f64;
        Some((self.grid_tiers, select + 1.0 + mantissas.log2() + exps.log2()))
    }

    /// Hash of the code of this run: the spec file, the alphabet and the variable count.
    pub fn sha256(&self) -> String {
        crate::manifest::sha256_hex(format!("{}|version={}|ops={}|variables={}", self.file_sha256, self.version, self.alphabet.join(","), self.variables).as_bytes())
    }

    /// The `code` object of `objective.selection_rule`.
    pub fn json(&self) -> String {
        format!(
            "{{\"version\":{},\"spec_sha256\":\"{}\",\"alphabet_size\":{},\"alphabet\":[{}],\"variables\":{},\"sha256\":\"{}\"}}",
            self.version,
            self.file_sha256,
            self.alphabet_size(),
            self.alphabet.iter().map(|s| format!("\"{}\"", s)).collect::<Vec<_>>().join(","),
            self.variables,
            self.sha256()
        )
    }

    /// Bits of one tree. Every constant sits at the smallest tier that holds it
    /// exactly; a constant no tier holds, an operator outside the alphabet and a
    /// variable index outside `0..variables` have no codeword (`Err`). The
    /// affine maps `denormalize` adds are functions of the inputs and cost
    /// nothing. A custom block costs its operator symbol per call and its
    /// constants once per slot; the body structure is declared before the run.
    pub fn description_bits(&self, tree: &Node) -> Result<Bits, String> {
        let mut acc = Bits { structure: 0.0, constants: 0.0, total: 0.0, nodes: 0, variables: 0, constant_count: 0 };
        let mut seen = [false; 4];
        self.walk(tree, &mut acc, &mut seen)?;
        acc.structure = acc.nodes as f64 * self.node_bits() + acc.variables as f64 * self.variable_bits();
        acc.total = acc.structure + acc.constants;
        Ok(acc)
    }

    fn walk(&self, n: &Node, acc: &mut Bits, seen: &mut [bool; 4]) -> Result<(), String> {
        acc.nodes += 1;
        match n.op {
            Op::Const => self.add_constant(n.val, acc)?,
            Op::Input => {
                if !(n.val >= 0.0) || n.val as usize >= self.variables {
                    return Err(format!("variable index {} is outside the {} search variables", n.val, self.variables));
                }
                acc.variables += 1;
            }
            op => {
                if !self.alphabet.contains(&format!("{:?}", op)) {
                    return Err(format!("operator '{:?}' is not in the alphabet of the run", op));
                }
                for k in &n.kids {
                    self.walk(k, acc, seen)?;
                }
                if let (Some(slot), Some(block)) = (op.custom_slot(), &n.block) {
                    if !seen[slot] {
                        seen[slot] = true;
                        self.body_constants(&block.body, acc)?;
                    }
                }
            }
        }
        Ok(())
    }

    fn body_constants(&self, n: &Node, acc: &mut Bits) -> Result<(), String> {
        if n.op == Op::Const {
            self.add_constant(n.val, acc)?;
        }
        for k in &n.kids {
            self.body_constants(k, acc)?;
        }
        Ok(())
    }

    fn add_constant(&self, v: f64, acc: &mut Bits) -> Result<(), String> {
        let (_, bits) = self.constant_bits(v).ok_or_else(|| format!("constant {} has no codeword", v))?;
        acc.constants += bits;
        acc.constant_count += 1;
        Ok(())
    }

    /// The code words scored for a member: the tree rounded to tier `t` for
    /// `t = 0..grid_tiers` (tier label `t`), then the tree with each constant at
    /// its smallest tier (label `grid_tiers`, "exact"). Identical trees appear
    /// once. Each is its own code word with its own book, so taking the lowest
    /// `p_occam` is covered by the bound.
    pub fn candidates(&self, tree: &Node) -> Vec<Candidate> {
        let mut out: Vec<Candidate> = Vec::new();
        let rounders: Vec<(usize, Box<dyn Fn(f64) -> Option<f64> + '_>)> = (0..self.grid_tiers)
            .map(|t| (t, Box::new(move |v: f64| v.is_finite().then(|| self.snap(v, t))) as Box<dyn Fn(f64) -> Option<f64>>))
            .chain(std::iter::once((
                self.grid_tiers,
                Box::new(move |v: f64| match self.constant_bits(v) {
                    Some((t, _)) if t < self.grid_tiers => Some(self.snap(v, t)),
                    Some(_) => Some(v),
                    None => self.sig_round(v),
                }) as Box<dyn Fn(f64) -> Option<f64>>,
            )))
            .collect();
        for (tier, f) in rounders {
            let Some(t) = map_constants(tree, &*f) else { continue };
            if out.iter().any(|c| c.tree == t) {
                continue;
            }
            if let Ok(bits) = self.description_bits(&t) {
                out.push(Candidate { tier, tree: t, bits });
            }
        }
        out
    }

    // ---- composite layer (pre-registered gate menu) ----

    fn branch_bits(&self, b: &Branch) -> f64 {
        self.flag_bits as f64
            + match b {
                Branch::Grid => self.variable_bits() + self.dir_bits as f64 + (self.cut_levels as f64).log2(),
                Branch::Tree(bits) => *bits,
            }
    }

    fn gate_bits(&self, g: &Gate) -> Result<f64, String> {
        match g {
            Gate::Menu(i) if *i < self.menu_size => Ok((self.menu_size as f64).log2()),
            Gate::Menu(i) => Err(format!("menu gate {} is outside the {} pre-registered gates", i, self.menu_size)),
            Gate::Searched(b) => Ok(self.branch_bits(b) + if matches!(b, Branch::Tree(_)) { (self.cut_levels as f64).log2() } else { 0.0 }),
        }
    }

    /// Bits of a composite formula: type selector, gate, cash side and branch codes.
    pub fn composite_bits(&self, c: &Composite) -> Result<f64, String> {
        let (gate, branches, cash): (Option<&Gate>, Vec<&Branch>, bool) = match c {
            Composite::C0(b) => (None, vec![b], false),
            Composite::C1 { gate, branch } => (Some(gate), vec![branch], true),
            Composite::C2 { gate, on, off } => (Some(gate), vec![on, off], false),
        };
        let kind = match (c, gate) {
            (Composite::C0(_), _) => 0,
            (Composite::C1 { .. }, Some(Gate::Menu(_))) => 1,
            (Composite::C2 { .. }, Some(Gate::Menu(_))) => 2,
            _ => 3,
        };
        let mut bits = self.type_len[kind] as f64 + if kind == 3 { self.c3_form_bits as f64 } else { 0.0 };
        if let Some(g) = gate {
            bits += self.gate_bits(g)?;
        }
        if cash {
            bits += self.cash_bits as f64;
        }
        for b in branches {
            bits += self.branch_bits(b);
        }
        Ok(bits)
    }
}

/// Branch code of a composite: a grid threshold rule or a tree of `bits` bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Branch {
    Grid,
    Tree(f64),
}

/// A menu index (`0..8`) or a searched gate written in the branch code.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Gate {
    Menu(usize),
    Searched(Branch),
}

/// C0 an all-weather formula, C1 one formula and cash, C2 two branches. A
/// searched gate makes a C1 or C2 a C3.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Composite {
    C0(Branch),
    C1 { gate: Gate, branch: Branch },
    C2 { gate: Gate, on: Branch, off: Branch },
}

#[derive(Clone, Debug)]
pub struct Candidate {
    /// `0..grid_tiers` for a rounded tree, `grid_tiers` for the exact tree.
    pub tier: usize,
    pub tree: Node,
    pub bits: Bits,
}

fn map_constants(n: &Node, f: &dyn Fn(f64) -> Option<f64>) -> Option<Node> {
    let mut out = n.clone();
    if n.op == Op::Const {
        out.val = f(n.val)?;
    }
    out.kids = n.kids.iter().map(|k| map_constants(k, f)).collect::<Option<Vec<_>>>()?;
    if let Some(b) = &n.block {
        let mut nb = (**b).clone();
        nb.body = map_constants(&b.body, f)?;
        out.block = Some(Box::new(nb));
    }
    Some(out)
}

/// `min(1, p 2^bits)`, computed in the log domain.
pub fn p_occam(p: f64, bits: f64) -> f64 {
    if p <= 0.0 {
        return 0.0;
    }
    ((p.ln() + bits * std::f64::consts::LN_2).exp()).min(1.0)
}

// ---------------------------------------------------------------------------
// Evidence and the rule
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OccamParams {
    pub delta: f64,
    pub min_entries: usize,
}

impl Default for OccamParams {
    fn default() -> Self {
        OccamParams { delta: 0.05, min_entries: 30 }
    }
}

impl OccamParams {
    pub fn validate(&self) -> Result<(), String> {
        if !(self.delta > 0.0 && self.delta < 1.0) {
            return Err(format!("--select-delta must be inside (0, 1), got {}", self.delta));
        }
        Ok(())
    }
}

/// Block t statistic of one alpha series.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Evidence {
    pub mean_alpha: f64,
    pub blocks: usize,
    pub block_len: usize,
    /// `None` when the block means have zero variance.
    pub z: Option<f64>,
    pub p: Option<f64>,
}

/// Non-overlapping blocks of `max(ceil(cbrt(T)), hold)` periods, capped so that
/// at least [`MIN_BLOCKS`] remain; a trailing partial block is dropped. `z` is
/// the one-sample t statistic of the block means, `p` the upper tail of Student's
/// t with `blocks - 1` degrees of freedom.
pub fn evidence(series: &[f64], hold: usize) -> Result<Evidence, String> {
    let t = series.len();
    if t < MIN_BLOCKS {
        return Err(format!("--select-rule occam needs at least {} blocks of development periods, found {} periods", MIN_BLOCKS, t));
    }
    let block_len = ((t as f64).cbrt().ceil() as usize).max(hold).max(1).min(t / MIN_BLOCKS);
    let blocks = t / block_len;
    let means: Vec<f64> = (0..blocks).map(|b| series[b * block_len..(b + 1) * block_len].iter().sum::<f64>() / block_len as f64).collect();
    let m = means.iter().sum::<f64>() / blocks as f64;
    let var = means.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (blocks - 1) as f64;
    let mean_alpha = series.iter().sum::<f64>() / t as f64;
    let scale = series.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let (z, p) = if var > (scale * 1e-12) * (scale * 1e-12) && var.is_finite() {
        let z = m / (var / blocks as f64).sqrt();
        (Some(z), Some(crate::inference::student_t_sf(z, (blocks - 1) as f64)))
    } else {
        (None, None)
    };
    Ok(Evidence { mean_alpha, blocks, block_len, z, p })
}

/// What the caller measured for one code word of a member.
#[derive(Clone, Debug)]
pub struct Scored {
    pub tier: usize,
    pub expression_exact: String,
    pub bits: Bits,
    /// Per-period alpha on the development rows.
    pub series: Vec<f64>,
    pub entries: usize,
    /// Mean long hold in periods (at least 1).
    pub hold: usize,
}

#[derive(Clone, Debug)]
pub struct OccamBest {
    pub tier: usize,
    pub expression_exact: String,
    pub bits: Bits,
    pub entries: usize,
    pub ev: Evidence,
    pub p_occam: f64,
}

#[derive(Clone, Debug)]
pub struct OccamMember {
    pub archive_rank: usize,
    pub eligible: bool,
    pub best: Option<OccamBest>,
}

#[derive(Clone, Debug)]
pub struct OccamSelection {
    pub params: OccamParams,
    pub code: CodeSpec,
    pub members: Vec<OccamMember>,
    /// Index into `members`; `None` is the `no_signal` verdict.
    pub chosen: Option<usize>,
}

fn better(a: &OccamBest, b: &OccamBest) -> bool {
    a.p_occam.total_cmp(&b.p_occam).then(a.bits.total.total_cmp(&b.bits.total)).then(a.tier.cmp(&b.tier)).is_lt()
}

/// The Occam rule. `scored(i)` returns the code words of member `i` (empty for
/// an unusable member). A member is eligible with a positive mean alpha, at
/// least `min_entries` entries and a defined `z`; the finalist is the eligible
/// member of lowest `p_occam` (ties: fewer bits, lower archive rank) and the
/// verdict is `certified` only when that value is at most `delta`.
pub fn select_occam(
    metas: &[crate::selection::RobustMeta],
    params: OccamParams,
    code: &CodeSpec,
    scored: &mut dyn FnMut(usize) -> Result<Vec<Scored>, String>,
) -> Result<OccamSelection, String> {
    params.validate()?;
    let mut members = Vec::with_capacity(metas.len());
    for (i, m) in metas.iter().enumerate() {
        let words = if m.usable { scored(i)? } else { Vec::new() };
        let mut best: Option<OccamBest> = None;
        let mut eligible = false;
        for w in words {
            let ev = evidence(&w.series, w.hold)?;
            let po = ev.p.map_or(f64::INFINITY, |p| p_occam(p, w.bits.total));
            let ok = ev.z.is_some() && ev.mean_alpha > 0.0 && w.entries >= params.min_entries;
            let cand = OccamBest { tier: w.tier, expression_exact: w.expression_exact, bits: w.bits, entries: w.entries, ev, p_occam: po };
            // Eligible code words outrank ineligible ones for the report.
            let rank_better = match &best {
                None => true,
                Some(_) if ok != eligible => ok,
                Some(b) => better(&cand, b),
            };
            if rank_better {
                eligible = ok;
                best = Some(cand);
            }
        }
        members.push(OccamMember { archive_rank: m.archive_rank, eligible, best });
    }
    let chosen = members
        .iter()
        .enumerate()
        .filter(|(_, m)| m.eligible)
        .min_by(|(_, a), (_, b)| {
            let (a_b, b_b) = (a.best.as_ref().unwrap(), b.best.as_ref().unwrap());
            a_b.p_occam.total_cmp(&b_b.p_occam).then(a_b.bits.total.total_cmp(&b_b.bits.total)).then(a.archive_rank.cmp(&b.archive_rank))
        })
        .filter(|(_, m)| m.best.as_ref().unwrap().p_occam <= params.delta)
        .map(|(i, _)| i);
    Ok(OccamSelection { params, code: code.clone(), members, chosen })
}

impl OccamSelection {
    /// The members of `objective.selection_rule`, without braces.
    pub fn fields_json(&self, edge: Option<bool>) -> String {
        let verdict = if edge == Some(true) {
            "no_edge"
        } else if self.chosen.is_some() {
            "certified"
        } else {
            "no_signal"
        };
        format!(
            "\"rule\":\"occam\",\"delta\":{},\"min_entries\":{},\"code\":{},\"verdict\":\"{}\",\"chosen_rank\":{}{}",
            num(self.params.delta),
            self.params.min_entries,
            self.code.json(),
            verdict,
            self.chosen.map_or_else(|| "null".to_string(), |i| self.members[i].archive_rank.to_string()),
            if edge.is_some() { ",\"require_edge\":true" } else { "" }
        )
    }

    /// The `pareto[].occam` member of member `i` (leading comma).
    pub fn member_json(&self, i: usize) -> String {
        let Some(b) = &self.members[i].best else {
            return ",\"occam\":null".to_string();
        };
        let opt = |v: Option<f64>| v.map_or_else(|| "null".to_string(), num);
        let budget = b.ev.p.map(|p| (self.params.delta / p).log2());
        format!(
            ",\"occam\":{{\"bits\":{},\"bits_structure\":{},\"bits_constants\":{},\"tier\":{},\"expression_exact\":{},\"entries\":{},\"blocks\":{},\"block_len\":{},\"z\":{},\"p\":{},\"p_occam\":{},\"bit_budget\":{},\"eligible\":{}}}",
            num(b.bits.total),
            num(b.bits.structure),
            num(b.bits.constants),
            b.tier,
            jstr(&b.expression_exact),
            b.entries,
            b.ev.blocks,
            b.ev.block_len,
            opt(b.ev.z),
            opt(b.ev.p),
            if b.p_occam.is_finite() { num(b.p_occam) } else { "null".to_string() },
            opt(budget),
            self.members[i].eligible
        )
    }
}

fn jstr(s: &str) -> String {
    let mut out = String::from("\"");
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}
