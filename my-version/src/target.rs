//! Declared target specification (#254).
//!
//! A `TargetSpec` states how a label column was built from a forward return:
//! the window it looks into the future, the offset and scale of the
//! normalisation, the clip and the cost assumption. The engine cannot infer
//! this from data, so the spec is a JSON contract (`rcsr-target-spec/1`) loaded
//! from `--target-spec` or a `<data>.targets.json` sidecar.
//!
//! For `kind = "xs_rank"` the label is the per-timestamp rank of
//! `return_column` over the panel members; the book is priced on `return_column`.
//!
//! For `kind = "forward_return"` the label is
//! `T = clip((r - c) / (m * g), lo, hi)` with return `r`, offset `c`, scale
//! multiplier `m` and scale value `g` (a column or a constant). The null
//! prediction is the label at `r = 0`. It depends on a column exactly when the
//! scale is a column and `c != 0` or the clip is asymmetric: that is the
//! structural term the engine refuses by default.

use crate::devlog::{num, text, Json};
use crate::manifest::sha256_hex;

pub const SCHEMA: &str = "rcsr-target-spec/1";

/// The supported label kinds.
pub const KINDS: &[&str] = &["forward_return", "xs_rank"];

/// The label is the per-timestamp rank of `return_column` over the panel
/// members, mapped to `[-1, 1]` by `rcsr panel rank`.
pub const KIND_XS_RANK: &str = "xs_rank";

/// Where an evaluator's per-side cost came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CostSource {
    TargetSpec,
    Flag,
    Default,
}

impl CostSource {
    pub fn as_str(&self) -> &'static str {
        match self {
            CostSource::TargetSpec => "target_spec",
            CostSource::Flag => "flag",
            CostSource::Default => "default",
        }
    }
}

/// The one cost rule of every evaluator. A target spec that declares a cost
/// wins; a `--cost-bps` flag that disagrees is refused; `--cost-override`
/// prices a different cost on purpose and is recorded as `flag`.
pub fn resolve_cost(
    spec: Option<&TargetSpec>,
    flag: Option<f64>,
    cost_override: Option<f64>,
    default: f64,
) -> Result<(f64, CostSource), String> {
    if flag.is_some() && cost_override.is_some() {
        return Err("--cost-bps and --cost-override are alternatives; pass one of them".to_string());
    }
    if let Some(o) = cost_override {
        return Ok((o, CostSource::Flag));
    }
    match (spec.and_then(TargetSpec::per_side_cost_bps), flag) {
        (Some(s), Some(f)) if (s - f).abs() > 1e-12 => Err(format!(
            "--cost-bps {} differs from the target spec ({} per side); \
             use --cost-override to price a different cost",
            num(f),
            num(s)
        )),
        (Some(s), _) => Ok((s, CostSource::TargetSpec)),
        (None, Some(f)) => Ok((f, CostSource::Flag)),
        (None, None) => Ok((default, CostSource::Default)),
    }
}

/// The `cost` block of every report.
#[derive(Clone, Debug, PartialEq)]
pub struct CostDecl {
    pub per_side_bps: f64,
    pub source: CostSource,
    /// `log` or `simple`: the units of the return column as declared.
    pub return_units: String,
    /// The spec's own per-side cost when it differs from the one charged.
    pub spec_per_side_bps: Option<f64>,
}

impl CostDecl {
    pub fn new(spec: Option<&TargetSpec>, per_side_bps: f64, source: CostSource) -> CostDecl {
        let spec_cost = spec.and_then(TargetSpec::per_side_cost_bps);
        CostDecl {
            per_side_bps,
            source,
            return_units: spec.map_or("simple", |s| s.return_units.as_str()).to_string(),
            spec_per_side_bps: spec_cost.filter(|s| (s - per_side_bps).abs() > 1e-12),
        }
    }

    pub fn log_returns(&self) -> bool {
        self.return_units == "log"
    }

    pub fn json(&self) -> String {
        let spec_cost = self
            .spec_per_side_bps
            .map(|s| format!(",\"spec_per_side_bps\":{}", num(s)))
            .unwrap_or_default();
        format!(
            "{{\"per_side_bps\":{},\"source\":{},\"return_units\":{}{}}}",
            num(self.per_side_bps),
            text(self.source.as_str()),
            text(&self.return_units),
            spec_cost
        )
    }
}

/// The spec entry whose label column, name or return column is `target`.
pub fn entry_for<'a>(specs: &'a [TargetSpec], target: &str) -> Option<&'a TargetSpec> {
    specs
        .iter()
        .find(|s| s.column == target || s.name == target || s.return_column.as_deref() == Some(target))
}

#[derive(Clone, Debug, PartialEq)]
pub enum Scale {
    /// The label is divided by a constant (1 for a raw return).
    Constant { value: f64 },
    /// The label is divided by `multiplier * column` (a volatility estimate).
    Column {
        column: String,
        multiplier: f64,
        /// Length of the estimation window, seconds.
        window_seconds: f64,
        /// End of the estimation window relative to the row stamp, seconds.
        ends_at_offset_seconds: f64,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub struct TargetSpec {
    pub column: String,
    pub name: String,
    pub kind: String,
    pub deprecated: bool,
    pub return_column: Option<String>,
    /// `log` or `simple`.
    pub return_units: String,
    pub horizon_seconds: f64,
    pub entry_offset_seconds: f64,
    pub delay_seconds: f64,
    pub entry_price: String,
    pub exit_price: String,
    /// `c`, in return units.
    pub offset: f64,
    pub scale: Scale,
    pub clip: Option<(f64, f64)>,
    pub units: String,
    /// Round-trip cost assumption, basis points.
    pub cost_bps: Option<f64>,
    pub cost_in_label: bool,
    pub construction_columns: Vec<String>,
    pub note: Option<String>,
    /// Always true: the engine has no short label or barrier.
    pub long_only: bool,
}

impl TargetSpec {
    /// Label window relative to the row stamp: `(entry + delay, entry + delay + horizon)`.
    pub fn window_seconds(&self) -> (f64, f64) {
        let a = self.entry_offset_seconds + self.delay_seconds;
        (a, a + self.horizon_seconds)
    }

    /// Per-side cost in basis points: half the round-trip `cost_bps`. `None`
    /// when the spec declares no cost or the cost is already inside the label
    /// (an evaluator then charges nothing on top).
    pub fn per_side_cost_bps(&self) -> Option<f64> {
        if self.cost_in_label {
            return None;
        }
        self.cost_bps.map(|c| c / 2.0)
    }

    /// The spec members as written to a spec file: the hashed members plus the
    /// derived `cost_per_side_bps`.
    pub fn file_json(&self) -> String {
        match self.cost_bps {
            Some(c) => format!("{},\"cost_per_side_bps\":{}", self.fields_json(), num(c / 2.0)),
            None => self.fields_json(),
        }
    }

    /// Minimum stamp distance between a training row and a test row so that
    /// the training label ends before the test row's decision time.
    pub fn required_purge_seconds(&self) -> f64 {
        self.window_seconds().1
    }

    /// The scale is a column and the null prediction is not a constant.
    pub fn structural_term(&self) -> bool {
        let column = matches!(self.scale, Scale::Column { .. });
        let asymmetric = self.clip.is_some_and(|(lo, hi)| lo != -hi);
        column && (self.offset != 0.0 || asymmetric)
    }

    /// The scale column, when the scale is a column.
    pub fn scale_column(&self) -> Option<&str> {
        match &self.scale {
            Scale::Column { column, .. } => Some(column.as_str()),
            Scale::Constant { .. } => None,
        }
    }

    fn divisor(&self, scale_value: f64) -> f64 {
        match &self.scale {
            Scale::Constant { value } => *value,
            Scale::Column { multiplier, .. } => multiplier * scale_value,
        }
    }

    fn clipped(&self, t: f64) -> f64 {
        match self.clip {
            Some((lo, hi)) => t.max(lo).min(hi),
            None => t,
        }
    }

    /// The label at a zero return. NaN when the scale column is not positive.
    pub fn null_prediction(&self, scale_value: f64) -> f64 {
        if matches!(self.scale, Scale::Column { .. }) && !(scale_value > 0.0) {
            return f64::NAN;
        }
        let d = self.divisor(scale_value);
        // `0.0 - offset` keeps +0 for a zero offset.
        self.clipped((0.0 - self.offset) / d)
    }

    /// Target space to return space: `r = c + m * g * t`. This is the exact
    /// inverse of the label on the unclipped interval; a label at a clip bound
    /// only bounds the return. NaN when the scale column is not positive.
    pub fn return_from_target(&self, t: f64, scale_value: f64) -> f64 {
        if matches!(self.scale, Scale::Column { .. }) && !(scale_value > 0.0) {
            return f64::NAN;
        }
        self.offset + self.divisor(scale_value) * t
    }

    /// Return space to target space: the label itself.
    pub fn target_from_return(&self, r: f64, scale_value: f64) -> f64 {
        if matches!(self.scale, Scale::Column { .. }) && !(scale_value > 0.0) {
            return f64::NAN;
        }
        self.clipped((r - self.offset) / self.divisor(scale_value))
    }

    /// The null prediction as a formula over the scale column, e.g.
    /// `clip(-0.002/(6*vol96),-1,1)`; `0` when the offset is zero and the clip
    /// is symmetric or absent, and the constant otherwise.
    pub fn null_formula(&self) -> String {
        let inner = match &self.scale {
            Scale::Column { column, multiplier, .. } => {
                if self.offset == 0.0 {
                    return "0".to_string();
                }
                format!("-{}/({}*{})", num(self.offset), num(*multiplier), column)
            }
            Scale::Constant { value } => {
                return num(self.clipped((0.0 - self.offset) / value));
            }
        };
        match self.clip {
            Some((lo, hi)) => format!("clip({},{},{})", inner, num(lo), num(hi)),
            None => inner,
        }
    }

    /// Fixed field order, `devlog::num` numbers: the hashed form.
    pub fn canonical_json(&self) -> String {
        format!("{{{}}}", self.fields_json())
    }

    /// The spec fields as JSON members, without the braces.
    pub fn fields_json(&self) -> String {
        let opt_s = |v: &Option<String>| v.as_deref().map(text).unwrap_or_else(|| "null".into());
        let opt_n = |v: Option<f64>| v.map(num).unwrap_or_else(|| "null".into());
        let scale = match &self.scale {
            Scale::Constant { value } => {
                format!("{{\"kind\":\"constant\",\"value\":{}}}", num(*value))
            }
            Scale::Column { column, multiplier, window_seconds, ends_at_offset_seconds } => format!(
                "{{\"kind\":\"column\",\"column\":{},\"multiplier\":{},\"window_seconds\":{},\"ends_at_offset_seconds\":{}}}",
                text(column),
                num(*multiplier),
                num(*window_seconds),
                num(*ends_at_offset_seconds)
            ),
        };
        let clip = match self.clip {
            Some((lo, hi)) => format!("[{},{}]", num(lo), num(hi)),
            None => "null".into(),
        };
        let cols: Vec<String> = self.construction_columns.iter().map(|c| text(c)).collect();
        format!(
            "\"column\":{},\"name\":{},\"kind\":{},\"deprecated\":{},\"return_column\":{},\"return_units\":{},\"horizon_seconds\":{},\"entry_offset_seconds\":{},\"delay_seconds\":{},\"entry_price\":{},\"exit_price\":{},\"offset\":{},\"scale\":{},\"clip\":{},\"units\":{},\"cost_bps\":{},\"cost_in_label\":{},\"construction_columns\":[{}],\"note\":{},\"long_only\":{}",
            text(&self.column),
            text(&self.name),
            text(&self.kind),
            self.deprecated,
            opt_s(&self.return_column),
            text(&self.return_units),
            num(self.horizon_seconds),
            num(self.entry_offset_seconds),
            num(self.delay_seconds),
            text(&self.entry_price),
            text(&self.exit_price),
            num(self.offset),
            scale,
            clip,
            text(&self.units),
            opt_n(self.cost_bps),
            self.cost_in_label,
            cols.join(","),
            opt_s(&self.note),
            self.long_only
        )
    }

    pub fn sha256(&self) -> String {
        sha256_hex(self.canonical_json().as_bytes())
    }

    /// The derived members shared by every report.
    pub fn derived_json(&self) -> String {
        let (w0, w1) = self.window_seconds();
        format!(
            "\"window_seconds\":[{},{}],\"required_purge_seconds\":{},\"structural_term\":{},\"null_formula\":{}",
            num(w0),
            num(w1),
            num(self.required_purge_seconds()),
            self.structural_term(),
            text(&self.null_formula())
        )
    }

    /// The `# target:` header line.
    pub fn header_line(&self, sha: &str) -> String {
        let (w0, w1) = self.window_seconds();
        let cost = match (self.cost_bps, self.cost_in_label) {
            (Some(c), true) => format!("cost {}bp in label", num(c)),
            (Some(c), false) => format!("cost {}bp outside label", num(c)),
            (None, _) => "cost n/a".to_string(),
        };
        let dep = if self.deprecated { " (deprecated)" } else { "" };
        format!(
            "# target: {} is {}{} | window [t+{}s, t+{}s] | horizon {}s | units {} | {} | spec {}",
            self.column,
            self.name,
            dep,
            num(w0),
            num(w1),
            num(self.horizon_seconds),
            self.units,
            cost,
            &sha[..12.min(sha.len())]
        )
    }
}

fn valid_name(s: &str) -> bool {
    let mut it = s.chars();
    matches!(it.next(), Some(c) if c.is_ascii_lowercase())
        && it.all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn field<'a>(o: &'a Json, key: &str, what: &str) -> Result<&'a Json, String> {
    o.get(key)
        .ok_or_else(|| format!("{}: missing field '{}'", what, key))
}

fn f64_field(o: &Json, key: &str, what: &str) -> Result<f64, String> {
    match field(o, key, what)? {
        Json::Num(v) if v.is_finite() => Ok(*v),
        _ => Err(format!("{}: field '{}' must be a finite number", what, key)),
    }
}

fn str_field(o: &Json, key: &str, what: &str) -> Result<String, String> {
    match field(o, key, what)? {
        Json::Str(s) => Ok(s.clone()),
        _ => Err(format!("{}: field '{}' must be a string", what, key)),
    }
}

fn opt_str(o: &Json, key: &str, what: &str) -> Result<Option<String>, String> {
    match o.get(key) {
        None | Some(Json::Null) => Ok(None),
        Some(Json::Str(s)) => Ok(Some(s.clone())),
        _ => Err(format!("{}: field '{}' must be a string or null", what, key)),
    }
}

fn opt_bool(o: &Json, key: &str, what: &str, default: bool) -> Result<bool, String> {
    match o.get(key) {
        None | Some(Json::Null) => Ok(default),
        Some(Json::Bool(b)) => Ok(*b),
        _ => Err(format!("{}: field '{}' must be true or false", what, key)),
    }
}

fn parse_scale(o: &Json, what: &str) -> Result<Scale, String> {
    let kind = str_field(o, "kind", &format!("{} scale", what))?;
    match kind.as_str() {
        "constant" => Ok(Scale::Constant {
            value: f64_field(o, "value", &format!("{} scale", what))?,
        }),
        "column" => {
            let w = format!("{} scale", what);
            Ok(Scale::Column {
                column: str_field(o, "column", &w)?,
                multiplier: f64_field(o, "multiplier", &w)?,
                window_seconds: f64_field(o, "window_seconds", &w)?,
                ends_at_offset_seconds: f64_field(o, "ends_at_offset_seconds", &w)?,
            })
        }
        other => Err(format!(
            "{}: unknown scale kind '{}' (expected constant or column)",
            what, other
        )),
    }
}

/// Parse one spec object, such as the `target_spec` object of a `run.json`.
pub fn parse_object(o: &Json) -> Result<TargetSpec, String> {
    parse_one(o, 0)
}

fn parse_one(o: &Json, index: usize) -> Result<TargetSpec, String> {
    let what = match o.get("name") {
        Some(Json::Str(s)) => format!("target '{}'", s),
        _ => format!("target #{}", index + 1),
    };
    if o.as_obj().is_none() {
        return Err(format!("{}: must be an object", what));
    }
    let clip = match o.get("clip") {
        None | Some(Json::Null) => None,
        Some(Json::Arr(a)) if a.len() == 2 => match (a[0].as_f64(), a[1].as_f64()) {
            (Some(lo), Some(hi)) if lo.is_finite() && hi.is_finite() => Some((lo, hi)),
            _ => return Err(format!("{}: field 'clip' must hold two finite numbers", what)),
        },
        _ => return Err(format!("{}: field 'clip' must be null or [lo, hi]", what)),
    };
    let cost_bps = match o.get("cost_bps") {
        None | Some(Json::Null) => None,
        Some(Json::Num(v)) if v.is_finite() => Some(*v),
        _ => return Err(format!("{}: field 'cost_bps' must be a number or null", what)),
    };
    match o.get("cost_per_side_bps") {
        None | Some(Json::Null) => {}
        Some(Json::Num(v)) => match cost_bps {
            Some(c) if (*v - c / 2.0).abs() <= 1e-12 => {}
            _ => {
                return Err(format!(
                    "{}: cost_per_side_bps {} must equal cost_bps / 2 (cost_bps {:?})",
                    what,
                    num(*v),
                    cost_bps
                ))
            }
        },
        _ => return Err(format!("{}: field 'cost_per_side_bps' must be a number or null", what)),
    }
    let construction_columns = match field(o, "construction_columns", &what)? {
        Json::Arr(a) => a
            .iter()
            .map(|v| {
                v.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| format!("{}: 'construction_columns' must hold strings", what))
            })
            .collect::<Result<Vec<_>, _>>()?,
        _ => return Err(format!("{}: field 'construction_columns' must be an array", what)),
    };
    let spec = TargetSpec {
        column: str_field(o, "column", &what)?,
        name: str_field(o, "name", &what)?,
        kind: str_field(o, "kind", &what)?,
        deprecated: opt_bool(o, "deprecated", &what, false)?,
        return_column: opt_str(o, "return_column", &what)?,
        return_units: str_field(o, "return_units", &what)?,
        horizon_seconds: f64_field(o, "horizon_seconds", &what)?,
        entry_offset_seconds: f64_field(o, "entry_offset_seconds", &what)?,
        delay_seconds: f64_field(o, "delay_seconds", &what)?,
        entry_price: opt_str(o, "entry_price", &what)?.unwrap_or_default(),
        exit_price: opt_str(o, "exit_price", &what)?.unwrap_or_default(),
        offset: f64_field(o, "offset", &what)?,
        scale: parse_scale(field(o, "scale", &what)?, &what)?,
        clip,
        units: str_field(o, "units", &what)?,
        cost_bps,
        cost_in_label: opt_bool(o, "cost_in_label", &what, false)?,
        construction_columns,
        note: opt_str(o, "note", &what)?,
        long_only: opt_bool(o, "long_only", &what, true)?,
    };
    validate(&spec, &what)?;
    Ok(spec)
}

fn validate(s: &TargetSpec, what: &str) -> Result<(), String> {
    if !KINDS.contains(&s.kind.as_str()) {
        return Err(format!(
            "{}: unknown kind '{}' (expected {})",
            what,
            s.kind,
            KINDS.join(" or ")
        ));
    }
    if !s.long_only {
        return Err(format!(
            "{}: long_only must be true; the engine has no short label or barrier",
            what
        ));
    }
    if s.kind == KIND_XS_RANK {
        let unit_scale = matches!(&s.scale, Scale::Constant { value } if *value == 1.0);
        if !unit_scale || s.clip.is_some() || s.offset != 0.0 || s.cost_in_label {
            return Err(format!(
                "{}: an xs_rank target has scale constant 1, no clip, offset 0 and cost_in_label false",
                what
            ));
        }
        match &s.return_column {
            Some(r) if s.construction_columns.contains(r) => {}
            Some(r) => {
                return Err(format!(
                    "{}: an xs_rank target lists its return_column '{}' in construction_columns",
                    what, r
                ))
            }
            None => return Err(format!("{}: an xs_rank target needs a return_column", what)),
        }
    }
    if !valid_name(&s.name) {
        return Err(format!("{}: name must match ^[a-z][a-z0-9_]*$", what));
    }
    if s.column.is_empty() {
        return Err(format!("{}: column must not be empty", what));
    }
    if s.return_units != "log" && s.return_units != "simple" {
        return Err(format!("{}: return_units must be 'log' or 'simple'", what));
    }
    if !(s.horizon_seconds > 0.0) {
        return Err(format!("{}: horizon_seconds must be positive", what));
    }
    for (n, v) in [
        ("entry_offset_seconds", s.entry_offset_seconds),
        ("delay_seconds", s.delay_seconds),
    ] {
        if !(v >= 0.0) {
            return Err(format!("{}: {} must be non-negative", what, n));
        }
    }
    match &s.scale {
        Scale::Constant { value } => {
            if !(*value > 0.0) {
                return Err(format!("{}: constant scale must be positive", what));
            }
        }
        Scale::Column { column, multiplier, window_seconds, ends_at_offset_seconds } => {
            if !(*multiplier > 0.0) {
                return Err(format!("{}: scale multiplier must be positive", what));
            }
            if !(*window_seconds >= 0.0) || !(*ends_at_offset_seconds >= 0.0) {
                return Err(format!(
                    "{}: scale window_seconds and ends_at_offset_seconds must be non-negative",
                    what
                ));
            }
            if !s.construction_columns.contains(column) {
                return Err(format!(
                    "{}: scale column '{}' must be listed in construction_columns",
                    what, column
                ));
            }
        }
    }
    if let Some((lo, hi)) = s.clip {
        if !(lo < hi) {
            return Err(format!("{}: clip must satisfy lo < hi", what));
        }
    }
    if s.cost_in_label {
        match s.cost_bps {
            Some(c) if (s.offset - c * 1e-4).abs() <= 1e-12 => {}
            Some(c) => {
                return Err(format!(
                    "{}: cost_in_label needs offset == cost_bps * 1e-4 (offset {}, cost_bps {})",
                    what,
                    num(s.offset),
                    num(c)
                ))
            }
            None => return Err(format!("{}: cost_in_label needs cost_bps", what)),
        }
    }
    if !(s.offset.is_finite()) {
        return Err(format!("{}: offset must be finite", what));
    }
    Ok(())
}

/// Parse a spec file: `{"schema":"rcsr-target-spec/1","targets":[...]}`.
pub fn parse_file(src: &str) -> Result<Vec<TargetSpec>, String> {
    let doc = Json::parse(src.trim()).ok_or_else(|| "not valid JSON".to_string())?;
    match doc.get("schema") {
        Some(Json::Str(s)) if s == SCHEMA => {}
        Some(Json::Str(s)) => {
            return Err(format!("unsupported schema '{}' (expected {})", s, SCHEMA))
        }
        _ => return Err(format!("missing field 'schema' (expected {})", SCHEMA)),
    }
    let items = match doc.get("targets") {
        Some(Json::Arr(a)) => a,
        _ => return Err("missing array 'targets'".to_string()),
    };
    let mut out: Vec<TargetSpec> = Vec::with_capacity(items.len());
    for (i, o) in items.iter().enumerate() {
        let spec = parse_one(o, i)?;
        if out.iter().any(|p| p.name == spec.name) {
            return Err(format!("duplicate target name '{}'", spec.name));
        }
        if out.iter().any(|p| p.column == spec.column) {
            return Err(format!("duplicate target column '{}'", spec.column));
        }
        out.push(spec);
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq)]
pub enum SpecSource {
    /// `--target-spec FILE`.
    File(String),
    /// `<data stem>.targets.json` beside the data.
    Sidecar(String),
}

impl SpecSource {
    pub fn name(&self) -> &'static str {
        match self {
            SpecSource::File(_) => "file",
            SpecSource::Sidecar(_) => "sidecar",
        }
    }

    pub fn path(&self) -> &str {
        match self {
            SpecSource::File(p) | SpecSource::Sidecar(p) => p,
        }
    }
}

#[derive(Debug)]
pub enum ResolveError {
    /// Exit 2: the spec is invalid or does not cover the target.
    Usage(String),
    /// Exit 3: the file cannot be read.
    Input(String),
}

/// The sidecar path for a data file: the data path with its extension replaced
/// by `targets.json`.
pub fn sidecar_path(data_path: &str) -> Option<String> {
    if data_path == "-" || data_path.is_empty() {
        return None;
    }
    let p = std::path::Path::new(data_path);
    Some(p.with_extension("targets.json").to_string_lossy().into_owned())
}

/// `<data>.labels.json`, written by `rcsr enrich` beside the store.
pub fn label_sidecar_path(data_path: &str) -> Option<String> {
    if data_path == "-" || data_path.is_empty() {
        return None;
    }
    let p = std::path::Path::new(data_path);
    Some(p.with_extension("labels.json").to_string_lossy().into_owned())
}

/// The label report object of target column `column` in the label sidecar of
/// `data_path`, or `None` without a sidecar or an entry.
pub fn label_report_for(data_path: &str, column: &str) -> Option<String> {
    let text = std::fs::read_to_string(label_sidecar_path(data_path)?).ok()?;
    let head = format!("{{\"target\":{},\"report\":", crate::devlog::text(column));
    text.lines().find_map(|l| {
        let l = l.trim_end().trim_end_matches(',');
        l.strip_prefix(&head)?.strip_suffix('}').map(str::to_string)
    })
}

/// A resolved spec: the entry, where it came from and the file hash.
pub struct Resolved {
    pub spec: TargetSpec,
    pub source: SpecSource,
    pub file_sha256: String,
}

/// Find the spec for `target`. `cli` is `--target-spec`; without it the
/// sidecar is used when it exists and holds an entry for the target. An
/// explicit file without an entry for the target is a usage error.
pub fn resolve(
    data_path: &str,
    cli: Option<&str>,
    target: &str,
) -> Result<Option<Resolved>, ResolveError> {
    let (path, explicit) = match cli {
        Some(p) => (p.to_string(), true),
        None => match sidecar_path(data_path) {
            Some(p) if std::path::Path::new(&p).is_file() => (p, false),
            _ => return Ok(None),
        },
    };
    let bytes = std::fs::read(&path)
        .map_err(|e| ResolveError::Input(format!("cannot read target spec '{}': {}", path, e)))?;
    let src = String::from_utf8(bytes)
        .map_err(|_| ResolveError::Usage(format!("target spec '{}' is not UTF-8", path)))?;
    let specs = parse_file(&src)
        .map_err(|e| ResolveError::Usage(format!("invalid target spec '{}': {}", path, e)))?;
    let file_sha256 = sha256_hex(src.as_bytes());
    match specs.into_iter().find(|s| s.column == target) {
        Some(spec) => Ok(Some(Resolved {
            spec,
            source: if explicit {
                SpecSource::File(path)
            } else {
                SpecSource::Sidecar(path)
            },
            file_sha256,
        })),
        None if explicit => Err(ResolveError::Usage(format!(
            "target spec '{}' has no entry for target column '{}'",
            path, target
        ))),
        None => Ok(None),
    }
}

/// Outcome of the D13 detectors for one run.
#[derive(Clone, Debug)]
pub struct Check {
    /// `ok`, or `overridden` when D13c was allowed by `--allow-target-structure`.
    pub status: &'static str,
    /// Construction columns that are model inputs (recorded, not refused).
    pub construction_in_vars: Vec<String>,
    /// Inputs ranked by absolute Spearman correlation with the scale column,
    /// filled under the override only.
    pub proxies: Option<Vec<(String, f64)>>,
}

/// Run the D13 detectors. `Err` carries the findings that stop the run.
pub fn check(
    spec: &TargetSpec,
    vars: &[String],
    allow_structure: bool,
) -> Result<Check, crate::leakage::Report> {
    let mut report = crate::leakage::target_structure(spec, vars);
    let structural = report.leaks.iter().any(|l| l.detector == "D13c");
    if allow_structure {
        report.leaks.retain(|l| l.detector != "D13c");
    }
    if !report.leaks.is_empty() {
        return Err(report);
    }
    Ok(Check {
        status: if structural && allow_structure { "overridden" } else { "ok" },
        construction_in_vars: spec
            .construction_columns
            .iter()
            .filter(|c| vars.contains(c))
            .cloned()
            .collect(),
        proxies: None,
    })
}

/// Average ranks (ties share the mean rank).
fn ranks(v: &[f64]) -> Vec<f64> {
    let mut order: Vec<usize> = (0..v.len()).collect();
    order.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap_or(std::cmp::Ordering::Equal));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < order.len() {
        let mut j = i;
        while j + 1 < order.len() && v[order[j + 1]] == v[order[i]] {
            j += 1;
        }
        let mid = (i + j) as f64 / 2.0 + 1.0;
        for &k in &order[i..=j] {
            r[k] = mid;
        }
        i = j + 1;
    }
    r
}

/// Spearman rank correlation; NaN when either side is constant.
pub fn spearman(a: &[f64], b: &[f64]) -> f64 {
    let (ra, rb) = (ranks(a), ranks(b));
    let n = ra.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let (ma, mb) = (ra.iter().sum::<f64>() / n, rb.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..ra.len() {
        let (da, db) = (ra[i] - ma, rb[i] - mb);
        sab += da * db;
        saa += da * da;
        sbb += db * db;
    }
    sab / (saa * sbb).sqrt()
}

/// Rows in the deterministic stride sample behind the proxy table.
pub const PROXY_SAMPLE_ROWS: usize = 65_536;

/// Spearman correlation of every input with the scale column on a
/// deterministic stride sample of the training rows, top `top` by absolute
/// value (ties by name). The scale column itself is skipped.
pub fn proxy_table(
    prepared: &crate::data::Prepared,
    vars: &[String],
    scale_column: &str,
    top: usize,
) -> Vec<(String, f64)> {
    let Some(ts) = &prepared.target_scale else {
        return Vec::new();
    };
    let n = prepared.train.x.len();
    if n == 0 {
        return Vec::new();
    }
    let stride = n.div_ceil(PROXY_SAMPLE_ROWS).max(1);
    let rows: Vec<usize> = (0..n)
        .step_by(stride)
        .filter(|&i| ts.train.get(i).is_some_and(|g| g.is_finite()))
        .collect();
    let g: Vec<f64> = rows.iter().map(|&i| ts.train[i]).collect();
    let mut out: Vec<(String, f64)> = Vec::new();
    for (j, name) in vars.iter().enumerate() {
        if name == scale_column {
            continue;
        }
        let x: Vec<f64> = rows.iter().map(|&i| prepared.train.x[i][j]).collect();
        if x.iter().any(|v| !v.is_finite()) {
            continue;
        }
        let rho = spearman(&x, &g);
        if rho.is_finite() {
            out.push((name.clone(), rho));
        }
    }
    out.sort_by(|a, b| {
        b.1.abs()
            .partial_cmp(&a.1.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    out.truncate(top);
    out
}

/// A spec in force for one run: where it came from, the detector outcome and
/// the zero-parameter rung. Shared by every report.
#[derive(Clone, Debug)]
pub struct Active {
    pub spec: TargetSpec,
    pub source: SpecSource,
    pub file_sha256: String,
    pub check: Check,
    pub rung: Option<crate::diagnostics::TargetStructureRung>,
    /// Realised-purge facts for the `# purge:` line.
    pub purge_line: String,
    /// Label production report (JSON object) of the target column, read from
    /// `<data>.labels.json` when `rcsr enrich` wrote one.
    pub label_report: Option<String>,
}

impl Active {
    /// The `target_spec` object of `run.json` and the `--eval-expr` result.
    pub fn json(&self) -> String {
        let proxies = match &self.check.proxies {
            Some(p) => format!(
                "[{}]",
                p.iter()
                    .map(|(n, r)| format!("{{\"column\":{},\"spearman\":{}}}", text(n), num(*r)))
                    .collect::<Vec<_>>()
                    .join(",")
            ),
            None => "null".to_string(),
        };
        format!(
            "{{\"schema\":{},\"source\":{},\"path\":{},\"sha256\":{},\"file_sha256\":{},{},{},\"check\":{{\"status\":{},\"construction_in_vars\":[{}],\"proxies\":{}}},\"label_report\":{}}}",
            text(SCHEMA),
            text(self.source.name()),
            text(self.source.path()),
            text(&self.spec.sha256()),
            text(&self.file_sha256),
            self.spec.fields_json(),
            self.spec.derived_json(),
            text(self.check.status),
            self.check
                .construction_in_vars
                .iter()
                .map(|c| text(c))
                .collect::<Vec<_>>()
                .join(","),
            proxies,
            self.label_report.as_deref().unwrap_or("null")
        )
    }

    pub fn rung_json(&self) -> String {
        self.rung
            .as_ref()
            .map(|r| r.json())
            .unwrap_or_else(|| "null".to_string())
    }

    /// `# target:`, `# purge:` and `# target structure` lines.
    pub fn text_lines(&self) -> Vec<String> {
        let mut out = vec![
            self.spec.header_line(&self.spec.sha256()),
            self.purge_line.clone(),
        ];
        if let Some(r) = &self.rung {
            out.push(format!(
                "# target structure (0 params): {} train={:.9} test={:.9}",
                r.formula, r.train_error, r.test_error
            ));
        }
        out
    }

    pub fn warning_line(&self) -> Option<String> {
        (self.check.status == "overridden").then(|| {
            format!(
                "# WARNING: structural target term {} allowed by --allow-target-structure; errors are not return forecasts",
                self.spec.null_formula()
            )
        })
    }
}

/// The `# purge:` line from a split report.
pub fn purge_line(spec: &TargetSpec, report: &crate::data::SplitReport) -> String {
    let secs = |v: Option<f64>| v.map(|x| format!("{}s", num(x))).unwrap_or_else(|| "n/a".into());
    format!(
        "# purge: required {}s (entry {}s + delay {}s + horizon {}s) | realised train->test {} | fit->validation {} | purged {}+{} row(s)",
        num(spec.required_purge_seconds()),
        num(spec.entry_offset_seconds),
        num(spec.delay_seconds),
        num(spec.horizon_seconds),
        secs(report.purge_seconds),
        secs(report.validation_purge_seconds),
        report.purged_test_boundary_rows,
        report.purged_validation_boundary_rows
    )
}

/// The purge members of the `split` object, with a leading comma.
pub fn split_purge_json(r: &crate::data::SplitReport) -> String {
    let n = |v: Option<f64>| v.map(num).unwrap_or_else(|| "null".into());
    format!(
        ",\"required_purge_seconds\":{},\"purge_seconds\":{},\"validation_purge_seconds\":{},\"purged_rows\":{{\"test_boundary\":{},\"validation_boundary\":{}}},\"purge_source\":{}{}",
        n(r.required_purge_seconds),
        n(r.purge_seconds),
        n(r.validation_purge_seconds),
        r.purged_test_boundary_rows,
        r.purged_validation_boundary_rows,
        if r.purge_source.is_empty() { "null".to_string() } else { text(r.purge_source) },
        if r.boundary_purge_source.is_empty() {
            String::new()
        } else {
            format!(
                ",\"boundary_purge_seconds\":{},\"boundary_purge_source\":{},\"lockbox_embargo_seconds\":{},\"lockbox_embargo_source\":{}",
                num(r.boundary_purge_seconds),
                text(r.boundary_purge_source),
                r.lockbox_embargo_seconds,
                text(r.lockbox_embargo_source)
            )
        }
    )
}

/// The `target_spec` section of `rcsr schema`.
pub fn schema_json() -> String {
    // (name, type, required, description)
    let fields: &[(&str, &str, bool, &str)] = &[
        ("column", "string", true, "data column holding the label; unique in the file"),
        ("name", "string", true, "canonical name, ^[a-z][a-z0-9_]*$; unique in the file"),
        ("kind", "enum", true, "forward_return | xs_rank (per-timestamp rank of return_column over the panel members in [-1, 1], written by rcsr panel rank; needs return_column, scale constant 1, no clip, offset 0)"),
        ("deprecated", "bool", false, "the label is kept for replays only"),
        ("return_column", "string|null", false, "column holding the raw future return; refused as an input (D13a)"),
        ("return_units", "enum", true, "log | simple"),
        ("horizon_seconds", "number", true, "length of the label window, > 0"),
        ("entry_offset_seconds", "number", true, "row stamp to entry price, >= 0"),
        ("delay_seconds", "number", true, "decision to entry delay, >= 0"),
        ("entry_price", "string", false, "free text"),
        ("exit_price", "string", false, "free text"),
        ("offset", "number", true, "c in T = clip((r - c) / (m * g), lo, hi), return units"),
        ("scale", "object", true, "{kind: constant, value} or {kind: column, column, multiplier, window_seconds, ends_at_offset_seconds}; the scale column must be listed in construction_columns"),
        ("clip", "[lo, hi]|null", false, "lo < hi"),
        ("units", "string", true, "label units, e.g. log_return/(6*vol96)"),
        ("cost_bps", "number|null", false, "round-trip cost assumption in basis points; every evaluator reads cost_bps / 2 as the per-side cost"),
        ("cost_per_side_bps", "number|null", false, "written by rcsr enrich; must equal cost_bps / 2 when present; not part of the spec hash"),
        ("cost_in_label", "bool", true, "true requires offset == cost_bps * 1e-4"),
        ("construction_columns", "string[]", true, "columns used to build the label"),
        ("note", "string|null", false, "free text"),
        ("long_only", "bool", false, "default true; false is refused: labels are long-only (action {0,+1}, long-side barriers)"),
    ];
    let items: Vec<String> = fields
        .iter()
        .map(|(n, t, r, d)| {
            format!(
                "{{\"name\":{},\"type\":{},\"required\":{},\"description\":{}}}",
                text(n),
                text(t),
                r,
                text(d)
            )
        })
        .collect();
    format!(
        concat!(
            "{{\"schema\":{},\"summary\":{},\"doc\":\"docs/time-series.md\",",
            "\"flags\":[\"--target-spec\",\"--allow-target-structure\"],",
            "\"sidecar\":\"<data stem>.targets.json beside the data, used when --target-spec is absent\",",
            "\"file\":{},",
            "\"fields\":[{}],",
            "\"derived\":{{\"window_seconds\":\"[entry_offset_seconds + delay_seconds, + horizon_seconds]\",",
            "\"required_purge_seconds\":\"entry_offset_seconds + delay_seconds + horizon_seconds\",",
            "\"structural_term\":\"scale is a column and (offset != 0 or clip is asymmetric)\",",
            "\"null_formula\":\"label at zero return, e.g. clip(-0.002/(6*vol96),-1,1); 0 for a raw return\",",
            "\"sha256\":\"SHA-256 of the canonical JSON of the entry\"}},",
            "\"detectors\":[",
            "{{\"id\":\"D13a\",\"condition\":\"return_column or the target column is in --vars\",\"exit\":5,\"override\":null}},",
            "{{\"id\":\"D13b\",\"condition\":\"scale.ends_at_offset_seconds > entry_offset_seconds + delay_seconds\",\"exit\":5,\"override\":null}},",
            "{{\"id\":\"D13c\",\"condition\":\"structural_term\",\"exit\":5,\"override\":\"--allow-target-structure\"}}],",
            "\"purge\":{{\"rule\":\"drop every training row within required_purge_seconds (strict) of a test stamp, then every fit row within it of a validation stamp, on the global time axis; --gap applies on top\",",
            "\"needs\":\"a time column holding epoch seconds (exit 2 otherwise)\",",
            "\"split_fields\":[\"required_purge_seconds\",\"purge_seconds\",\"validation_purge_seconds\",\"purged_rows\",\"purge_source\"]}},",
            "\"outputs\":{{\"run_json\":[\"target_spec\",\"split.required_purge_seconds\",\"baselines.target_structure\"],",
            "\"eval_expr_json\":[\"target_spec\",\"target_structure\"]}},",
            "\"exit_codes\":{{\"2\":\"invalid spec, no entry for the target, or no epoch-seconds time column\",",
            "\"3\":\"spec file unreadable\",\"5\":\"D13 finding\"}}}}"
        ),
        text(SCHEMA),
        text("declared forward-return target: structure refusal, spec-derived purge, zero-parameter rung"),
        text("{\"schema\":\"rcsr-target-spec/1\",\"targets\":[<entry>,...]}"),
        items.join(",")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPACT: &str = include_str!("../tests/fixtures/targets/compact_15m_100.targets.json");
    const HOURLY: &str = include_str!("../tests/fixtures/targets/hourly_all.targets.json");

    fn y60m() -> TargetSpec {
        parse_file(COMPACT).unwrap().remove(0)
    }

    #[test]
    fn null_prediction_is_the_label_at_zero_return() {
        let s = y60m();
        for (g, want) in [
            (0.001, -1.0 / 3.0),
            (0.002, -1.0 / 6.0),
            (0.004, -1.0 / 12.0),
            (0.005, -1.0 / 15.0),
            (0.0002, -1.0),
        ] {
            assert!((s.null_prediction(g) - want).abs() <= 1e-15, "g={g}");
            assert!((s.target_from_return(0.0, g) - want).abs() <= 1e-15);
        }
        assert!(s.null_prediction(0.0).is_nan());
        assert!(s.null_prediction(f64::NAN).is_nan());
    }

    #[test]
    fn rung_matches_a_hand_computed_mse() {
        let s = y60m();
        let gs = [0.001, 0.002, 0.004, 0.005];
        let mse: f64 = gs.iter().map(|&g| s.null_prediction(g).powi(2)).sum::<f64>() / 4.0;
        assert!((mse - 541.0 / 14400.0).abs() <= 1e-15, "{mse}");
        assert!((mse - 0.0375694444444).abs() <= 1e-12);
    }

    #[test]
    fn inverse_map_round_trips_on_the_unclipped_interval() {
        let s = y60m();
        let g = 0.004;
        for r in [-0.01, 0.0, 0.002, 0.0151] {
            let t = s.target_from_return(r, g);
            assert!(t > -1.0 && t < 1.0);
            assert!((s.return_from_target(t, g) - r).abs() <= 1e-15, "r={r}");
        }
        // Zero label maps to the offset; the clipped label only bounds r.
        assert!((s.return_from_target(0.0, g) - 0.002).abs() <= 1e-18);
        assert_eq!(s.target_from_return(1.0, g), 1.0);
        assert!(s.return_from_target(0.5, -1.0).is_nan());
        let raw = parse_file(COMPACT).unwrap().remove(1);
        assert_eq!(raw.return_from_target(0.003, f64::NAN), 0.003);
        assert_eq!(raw.target_from_return(0.003, f64::NAN), 0.003);
    }

    fn with(mutate: impl Fn(&mut TargetSpec)) -> TargetSpec {
        let mut s = y60m();
        mutate(&mut s);
        s
    }

    #[test]
    fn structural_term_cases() {
        // net + column scale
        assert!(y60m().structural_term());
        // gross + column scale + symmetric clip
        assert!(!with(|s| s.offset = 0.0).structural_term());
        // gross + column scale + clip [-1, 3]
        assert!(with(|s| {
            s.offset = 0.0;
            s.cost_bps = None;
            s.cost_in_label = false;
            s.clip = Some((-1.0, 3.0));
        })
        .structural_term());
        // net + constant scale
        assert!(!with(|s| {
            s.scale = Scale::Constant { value: 1.0 };
            s.construction_columns.clear();
        })
        .structural_term());
    }

    #[test]
    fn null_formula_text() {
        assert_eq!(y60m().null_formula(), "clip(-0.002/(6*vol96),-1,1)");
        assert_eq!(parse_file(COMPACT).unwrap()[1].null_formula(), "0");
        assert_eq!(with(|s| s.offset = 0.0).null_formula(), "0");
    }

    #[test]
    fn legacy_specs_have_distinct_names_and_windows() {
        let mut all = parse_file(COMPACT).unwrap();
        all.extend(parse_file(HOURLY).unwrap());
        let names: Vec<&str> = all.iter().map(|s| s.name.as_str()).collect();
        assert_eq!(names, ["y60m", "ret_60m", "y1h", "ret_1h"]);
        let win: Vec<(f64, f64)> = all.iter().map(|s| s.window_seconds()).collect();
        assert_eq!(
            win,
            [(900.0, 4500.0), (900.0, 4500.0), (3600.0, 7200.0), (3600.0, 7200.0)]
        );
        let req: Vec<f64> = all.iter().map(|s| s.required_purge_seconds()).collect();
        assert_eq!(req, [4500.0, 4500.0, 7200.0, 7200.0]);
        assert!(all[0].structural_term() && all[0].deprecated);
        assert!(all[2].structural_term() && all[2].deprecated);
        assert!(!all[1].structural_term() && !all[1].deprecated);
        assert!(!all[3].structural_term());
        let shas: Vec<String> = all.iter().map(|s| s.sha256()).collect();
        for i in 0..4 {
            for j in 0..i {
                assert_ne!(shas[i], shas[j]);
            }
        }
        assert_eq!(all[0].sha256(), y60m().sha256());
    }

    #[test]
    fn invalid_specs_are_rejected() {
        let ok = COMPACT;
        let bad = |from: &str, to: &str| -> String {
            assert!(ok.contains(from), "fixture lacks {from}");
            parse_file(&ok.replacen(from, to, 1)).unwrap_err()
        };
        assert!(bad("\"horizon_seconds\":3600,", "").contains("horizon_seconds"));
        assert!(bad("\"kind\":\"forward_return\"", "\"kind\":\"barrier\"").contains("unknown kind"));
        assert!(bad("\"cost_bps\":20.0", "\"cost_bps\":10.0").contains("cost_in_label"));
        assert!(bad("\"name\":\"ret_60m\"", "\"name\":\"y60m\"").contains("duplicate target name"));
        assert!(bad("\"column\":\"fwd60\",\"name\"", "\"column\":\"y60\",\"name\"")
            .contains("duplicate target column"));
        assert!(bad(
            "\"construction_columns\":[\"vol96\"]",
            "\"construction_columns\":[]"
        )
        .contains("construction_columns"));
        assert!(bad("rcsr-target-spec/1", "rcsr-target-spec/2").contains("unsupported schema"));
        assert!(bad("\"clip\":[-1.0,1.0]", "\"clip\":[1.0,-1.0]").contains("lo < hi"));
        assert!(parse_file("{").is_err());
    }

    #[test]
    fn resolve_prefers_the_flag_and_reads_the_sidecar() {
        let dir = std::env::temp_dir().join(format!("rcsr-target-unit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let data = dir.join("d.csv");
        std::fs::write(&data, "x,y60\n1,2\n").unwrap();
        let side = dir.join("d.targets.json");
        std::fs::write(&side, COMPACT).unwrap();
        let d = data.to_str().unwrap();
        assert_eq!(sidecar_path(d).as_deref(), side.to_str());
        let r = resolve(d, None, "y60").unwrap().unwrap();
        assert_eq!(r.source.name(), "sidecar");
        assert_eq!(r.file_sha256, sha256_hex(COMPACT.as_bytes()));
        assert!(resolve(d, None, "other").unwrap().is_none());
        let other = dir.join("h.json");
        std::fs::write(&other, HOURLY).unwrap();
        let r = resolve(d, other.to_str(), "y60").unwrap().unwrap();
        assert_eq!(r.source.name(), "file");
        assert_eq!(r.spec.name, "y1h");
        assert!(matches!(
            resolve(d, other.to_str(), "other"),
            Err(ResolveError::Usage(_))
        ));
        assert!(matches!(
            resolve(d, Some("/nonexistent/x.json"), "y60"),
            Err(ResolveError::Input(_))
        ));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn with_cost(round_trip: Option<f64>, in_label: bool) -> TargetSpec {
        let mut s = y60m();
        s.cost_bps = round_trip;
        s.cost_in_label = in_label;
        s
    }

    #[test]
    fn resolve_cost_spec_only() {
        let s = with_cost(Some(20.0), false);
        assert_eq!(resolve_cost(Some(&s), None, None, 12.0), Ok((10.0, CostSource::TargetSpec)));
    }

    #[test]
    fn resolve_cost_flag_only() {
        assert_eq!(resolve_cost(None, Some(7.0), None, 12.0), Ok((7.0, CostSource::Flag)));
        let no_cost = with_cost(None, false);
        assert_eq!(resolve_cost(Some(&no_cost), Some(7.0), None, 12.0), Ok((7.0, CostSource::Flag)));
    }

    #[test]
    fn resolve_cost_both_equal_and_zero() {
        let s = with_cost(Some(20.0), false);
        assert_eq!(resolve_cost(Some(&s), Some(10.0), None, 12.0), Ok((10.0, CostSource::TargetSpec)));
        let z = with_cost(Some(0.0), false);
        assert_eq!(resolve_cost(Some(&z), Some(0.0), None, 12.0), Ok((0.0, CostSource::TargetSpec)));
    }

    #[test]
    fn resolve_cost_both_different_is_refused() {
        let s = with_cost(Some(20.0), false);
        let e = resolve_cost(Some(&s), Some(12.0), None, 12.0).unwrap_err();
        assert_eq!(
            e,
            "--cost-bps 12 differs from the target spec (10 per side); use --cost-override to price a different cost"
        );
    }

    #[test]
    fn resolve_cost_override_is_recorded_as_flag() {
        let s = with_cost(Some(20.0), false);
        let (c, src) = resolve_cost(Some(&s), None, Some(30.0), 12.0).unwrap();
        assert_eq!((c, src), (30.0, CostSource::Flag));
        let d = CostDecl::new(Some(&s), c, src);
        assert_eq!(d.spec_per_side_bps, Some(10.0));
        assert!(d.json().contains("\"source\":\"flag\"") && d.json().contains("\"spec_per_side_bps\":10"));
        assert!(resolve_cost(Some(&s), Some(10.0), Some(30.0), 12.0).is_err());
    }

    #[test]
    fn resolve_cost_neither_uses_the_default() {
        assert_eq!(resolve_cost(None, None, None, 12.0), Ok((12.0, CostSource::Default)));
        let in_label = with_cost(Some(20.0), true);
        assert_eq!(resolve_cost(Some(&in_label), None, None, 12.0), Ok((12.0, CostSource::Default)));
    }

    #[test]
    fn cost_per_side_field_must_match_the_round_trip() {
        let s = with_cost(Some(20.0), false);
        let body = format!("{{\"schema\":{},\"targets\":[{{{}}}]}}", text(SCHEMA), s.file_json());
        assert_eq!(parse_file(&body).unwrap()[0], s);
        assert!(parse_file(&body.replace("\"cost_per_side_bps\":10", "\"cost_per_side_bps\":9"))
            .unwrap_err()
            .contains("cost_per_side_bps"));
        assert_eq!(s.sha256(), with_cost(Some(20.0), false).sha256());
    }
}
