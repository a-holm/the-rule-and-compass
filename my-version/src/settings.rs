//! #1083: the search setting table.
//!
//! One row per search setting: its flag, checkpoint key, `--help` block, schema
//! row, write and resume rules, and its `rcsr-bench` and GUI exposure. The
//! readers loop over the table: `rcsr` parse and the resume check (`main.rs`),
//! [`Config::search_setting_differences`], the checkpoint reader and writer,
//! `rcsr schema`, `--help`, `rcsr-bench` and the GUI.
//!
//! A row belongs to a [`Group`], the place where its lines appear in every
//! reader. A reader emits a group at one anchor, rows in table order, so each
//! reader keeps its order while settings move into the table one group at a
//! time.
//!
//! Adding a saved search setting: add the `Config` field and its value in
//! `Config::default()` (the table reads defaults from there), add one row to
//! [`TABLE`] in the group whose place fits, and read the field where the search
//! uses it. A key added to an existing format must use [`Write::UnlessDefault`],
//! so older checkpoints still load and default checkpoints keep their bytes.
//! `tests/settings_table.rs` checks every row in every reader.

use crate::search::{Config, FitAccept, FrontScope, MigrationPolicy, MutationMix, RestartPolicy, Selection, VariationMode};
use std::collections::BTreeMap;

/// Where a row's lines appear in each reader.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Group {
    /// Next to the other stopping rules.
    Stop,
    /// The migration and restart cadence.
    Schedule,
    /// Island selection and restarts.
    Island,
    /// Node-path variation.
    Variation,
    /// Filters that reject trees before the constant fit.
    Filter,
    /// Constant-fit options.
    Fit,
    /// The variation engine selection, after the checkpoint options.
    Engine,
}

impl Group {
    pub const ALL: [Group; 7] = [
        Group::Stop,
        Group::Schedule,
        Group::Island,
        Group::Variation,
        Group::Filter,
        Group::Fit,
        Group::Engine,
    ];

    pub const fn name(self) -> &'static str {
        match self {
            Group::Stop => "stop",
            Group::Schedule => "schedule",
            Group::Island => "island",
            Group::Variation => "variation",
            Group::Filter => "filter",
            Group::Fit => "fit",
            Group::Engine => "engine",
        }
    }
}

/// The name of an anchor row in `schema::OPTIONS`; never a flag.
pub const ANCHOR: &str = "@settings";

/// An anchor row for `schema::OPTIONS`: the place of `group`'s rows.
pub const fn anchor(group: Group) -> (&'static str, &'static str, &'static str, &'static str) {
    (ANCHOR, group.name(), "", "")
}

/// The group an `OPTIONS` row anchors, if it is an anchor.
pub fn anchored(row: &(&str, &str, &str, &str)) -> Option<Group> {
    if row.0 != ANCHOR {
        return None;
    }
    Group::ALL.into_iter().find(|g| g.name() == row.1)
}

/// The value type of a setting.
pub trait Value: Clone + PartialEq + Send + Sync + 'static {
    /// Checkpoint text.
    fn text(&self) -> String;
    /// Text in messages and in schema defaults.
    fn show(&self) -> String;
    /// `rcsr-bench` knobs JSON.
    fn json(&self) -> String;
    /// Checkpoint value; the error completes `config '<key>' ...`.
    fn decode(raw: &str) -> Result<Self, String>;
}

impl Value for usize {
    fn text(&self) -> String {
        self.to_string()
    }
    fn show(&self) -> String {
        self.to_string()
    }
    fn json(&self) -> String {
        self.to_string()
    }
    fn decode(raw: &str) -> Result<Self, String> {
        raw.parse().map_err(|_| format!("is invalid: '{}'", raw))
    }
}

impl Value for f64 {
    fn text(&self) -> String {
        crate::checkpoint::encode_f64(*self)
    }
    fn show(&self) -> String {
        format!("{}", self)
    }
    fn json(&self) -> String {
        if self.is_finite() {
            format!("{}", self)
        } else {
            "null".to_string()
        }
    }
    fn decode(raw: &str) -> Result<Self, String> {
        crate::checkpoint::decode_f64(raw).map_err(|e| format!("is invalid: {}", e))
    }
}

impl Value for bool {
    fn text(&self) -> String {
        self.to_string()
    }
    fn show(&self) -> String {
        self.to_string()
    }
    fn json(&self) -> String {
        self.to_string()
    }
    fn decode(raw: &str) -> Result<Self, String> {
        match raw {
            "true" => Ok(true),
            "false" => Ok(false),
            other => Err(format!("must be true or false, got '{}'", other)),
        }
    }
}

macro_rules! enum_value {
    ($($t:ty),*) => {$(
        impl Value for $t {
            fn text(&self) -> String {
                self.as_str().to_string()
            }
            fn show(&self) -> String {
                self.as_str().to_string()
            }
            fn json(&self) -> String {
                format!("\"{}\"", self.as_str())
            }
            fn decode(raw: &str) -> Result<Self, String> {
                <$t>::parse(raw).map_err(|e| format!("is invalid: {}", e))
            }
        }
    )*};
}

enum_value!(Selection, RestartPolicy, MutationMix, FrontScope, MigrationPolicy, VariationMode, FitAccept);

/// How the command line sets a row.
pub enum Cli<T> {
    /// A flag without a value; it sets this value.
    Flag(T),
    /// A flag with a value; the parser's error is the whole message after `error: `.
    Value(fn(&str) -> Result<T, String>),
}

/// When the checkpoint writer emits `config <key> <value>`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Write {
    /// Every checkpoint; a checkpoint without the key is refused.
    Required,
    /// Every checkpoint; a checkpoint without the key loads the default.
    Always,
    /// Only when the row is active and its value differs from `Config::default()`;
    /// a checkpoint without the key loads the default.
    UnlessDefault,
}

/// The form of a value row's resume conflict line. A flag row always reads
/// `<flag> (saved <key> is <saved>)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conflict {
    /// `<flag> <value> (saved <saved>)`.
    Saved,
    /// `<flag> <value> (saved <key> is <saved>)`.
    SavedKey,
}

/// A row's `rcsr-bench` knob.
pub struct Bench {
    /// The exact `rcsr-bench --help` block, without a trailing newline.
    pub usage: &'static str,
    /// Leave the key out of the knobs JSON when the knob is not given, instead
    /// of writing `null` (or `false` for a flag).
    pub omit_unset: bool,
}

/// One search setting.
pub struct Row<T: Value> {
    pub flag: &'static str,
    /// Checkpoint key, `search_setting_differences` name and knobs JSON key.
    pub key: &'static str,
    pub group: Group,
    pub schema_type: &'static str,
    pub schema_default: &'static str,
    pub schema: &'static str,
    /// The exact `--help` block, without a trailing newline.
    pub help: &'static str,
    pub bench: Option<Bench>,
    pub gui: bool,
    pub get: fn(&Config) -> &T,
    pub set: fn(&mut Config) -> &mut T,
    pub cli: Cli<T>,
    pub write: Write,
    /// When the row is read: written, compared and checked on resume only while
    /// this holds. `None` means always.
    pub active: Option<fn(&Config) -> bool>,
    pub conflict: Conflict,
}

/// A table row, independent of its value type.
pub trait Setting: Sync {
    fn flag(&self) -> &'static str;
    fn key(&self) -> &'static str;
    fn group(&self) -> Group;
    fn schema_row(&self) -> (&'static str, &'static str, &'static str, &'static str);
    fn help(&self) -> &'static str;
    fn bench(&self) -> Option<&Bench>;
    fn gui(&self) -> bool;
    fn required(&self) -> bool;
    /// The checkpoint writer emits the key even at the default.
    fn always(&self) -> bool;
    fn takes_value(&self) -> bool;
    /// Set the row from the command line; `value` is `None` for a flag.
    fn apply_cli(&self, cfg: &mut Config, value: Option<&str>) -> Result<(), String>;
    /// Copy the row's value from `from` into `to`.
    fn copy(&self, from: &Config, to: &mut Config);
    fn active(&self, cfg: &Config) -> bool;
    fn differs(&self, a: &Config, b: &Config) -> bool;
    fn show(&self, cfg: &Config) -> String;
    fn conflict_line(&self, cfg: &Config, saved: &Config) -> String;
    /// Append `config <key> <value>` when the write rule asks for it.
    fn write(&self, cfg: &Config, default: &Config, out: &mut String);
    /// Set the row from a checkpoint value; `None` when the key is absent.
    fn read(&self, cfg: &mut Config, raw: Option<&str>) -> Result<(), String>;
    fn json(&self, cfg: &Config) -> String;
}

impl<T: Value> Setting for Row<T> {
    fn flag(&self) -> &'static str {
        self.flag
    }
    fn key(&self) -> &'static str {
        self.key
    }
    fn group(&self) -> Group {
        self.group
    }
    fn schema_row(&self) -> (&'static str, &'static str, &'static str, &'static str) {
        (self.flag, self.schema_type, self.schema_default, self.schema)
    }
    fn help(&self) -> &'static str {
        self.help
    }
    fn bench(&self) -> Option<&Bench> {
        self.bench.as_ref()
    }
    fn gui(&self) -> bool {
        self.gui
    }
    fn required(&self) -> bool {
        self.write == Write::Required
    }
    fn always(&self) -> bool {
        self.write == Write::Always
    }
    fn takes_value(&self) -> bool {
        matches!(self.cli, Cli::Value(_))
    }
    fn apply_cli(&self, cfg: &mut Config, value: Option<&str>) -> Result<(), String> {
        let v = match (&self.cli, value) {
            (Cli::Flag(v), None) => v.clone(),
            (Cli::Value(parse), Some(text)) => parse(text)?,
            (Cli::Flag(_), Some(_)) => return Err(format!("{} takes no value", self.flag)),
            (Cli::Value(_), None) => return Err(format!("{} requires a value", self.flag)),
        };
        *(self.set)(cfg) = v;
        Ok(())
    }
    fn copy(&self, from: &Config, to: &mut Config) {
        *(self.set)(to) = (self.get)(from).clone();
    }
    fn active(&self, cfg: &Config) -> bool {
        self.active.map_or(true, |f| f(cfg))
    }
    fn differs(&self, a: &Config, b: &Config) -> bool {
        (self.get)(a) != (self.get)(b)
    }
    fn show(&self, cfg: &Config) -> String {
        (self.get)(cfg).show()
    }
    fn conflict_line(&self, cfg: &Config, saved: &Config) -> String {
        match (&self.cli, self.conflict) {
            (Cli::Flag(_), _) => format!("{} (saved {} is {})", self.flag, self.key, self.show(saved)),
            (Cli::Value(_), Conflict::Saved) => {
                format!("{} {} (saved {})", self.flag, self.show(cfg), self.show(saved))
            }
            (Cli::Value(_), Conflict::SavedKey) => format!(
                "{} {} (saved {} is {})",
                self.flag,
                self.show(cfg),
                self.key,
                self.show(saved)
            ),
        }
    }
    fn write(&self, cfg: &Config, default: &Config, out: &mut String) {
        let emit = match self.write {
            Write::Required | Write::Always => true,
            Write::UnlessDefault => self.active(cfg) && self.differs(cfg, default),
        };
        if emit {
            out.push_str(&format!("config {} {}\n", self.key, (self.get)(cfg).text()));
        }
    }
    fn read(&self, cfg: &mut Config, raw: Option<&str>) -> Result<(), String> {
        match raw {
            Some(raw) => {
                *(self.set)(cfg) = T::decode(raw).map_err(|e| format!("config '{}' {}", self.key, e))?;
            }
            None if self.required() => {
                return Err(format!("checkpoint is incomplete: missing config field '{}'", self.key))
            }
            None => self.copy(&Config::default(), cfg),
        }
        Ok(())
    }
    fn json(&self, cfg: &Config) -> String {
        (self.get)(cfg).json()
    }
}

/// The rows the command line names: the resume check refuses only those.
#[derive(Clone, Debug, Default)]
pub struct Explicit(Vec<&'static str>);

impl Explicit {
    pub fn mark(&mut self, row: &dyn Setting) {
        if !self.contains(row) {
            self.0.push(row.key());
        }
    }

    pub fn contains(&self, row: &dyn Setting) -> bool {
        self.0.contains(&row.key())
    }
}

/// `rcsr-bench` knobs given on the command line, with their values.
#[derive(Clone, Default)]
pub struct Overrides {
    values: Config,
    given: Vec<&'static str>,
}

impl std::fmt::Debug for Overrides {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_map()
            .entries(rows().filter(|r| self.given.contains(&r.key())).map(|r| (r.key(), r.show(&self.values))))
            .finish()
    }
}

impl Overrides {
    /// Set the knob of `row` from the command line.
    pub fn set(&mut self, row: &dyn Setting, value: Option<&str>) -> Result<(), String> {
        row.apply_cli(&mut self.values, value)?;
        if !self.given.contains(&row.key()) {
            self.given.push(row.key());
        }
        Ok(())
    }

    /// Copy every given knob into `cfg`.
    pub fn apply(&self, cfg: &mut Config) {
        for row in rows().filter(|r| self.given.contains(&r.key())) {
            row.copy(&self.values, cfg);
        }
    }

    /// Append `,"<key>":<value>` for every bench row of `group`.
    pub fn json(&self, group: Group, out: &mut String) {
        for row in in_group(group) {
            let Some(bench) = row.bench() else { continue };
            let value = if self.given.contains(&row.key()) {
                row.json(&self.values)
            } else if bench.omit_unset {
                continue;
            } else if row.takes_value() {
                "null".to_string()
            } else {
                row.json(&Config::default())
            };
            out.push_str(&format!(",\"{}\":{}", row.key(), value));
        }
    }
}

pub fn rows() -> impl Iterator<Item = &'static dyn Setting> {
    TABLE.iter().copied()
}

pub fn in_group(group: Group) -> impl Iterator<Item = &'static dyn Setting> {
    rows().filter(move |r| r.group() == group)
}

pub fn find(flag: &str) -> Option<&'static dyn Setting> {
    rows().find(|r| r.flag() == flag)
}

pub fn find_key(key: &str) -> Option<&'static dyn Setting> {
    rows().find(|r| r.key() == key)
}

/// The `--help` blocks of `group`, one per line group, without a trailing newline.
pub fn help(group: Group) -> String {
    in_group(group).map(|r| r.help()).collect::<Vec<_>>().join("\n")
}

/// The `rcsr-bench --help` blocks of `group`'s knobs, without a trailing newline.
pub fn bench_usage(group: Group) -> String {
    in_group(group).filter_map(|r| r.bench().map(|b| b.usage)).collect::<Vec<_>>().join("\n")
}

/// The `rcsr schema` rows of `group`.
pub fn schema_rows(group: Group) -> Vec<(&'static str, &'static str, &'static str, &'static str)> {
    in_group(group).map(|r| r.schema_row()).collect()
}

/// Append the checkpoint lines of `group`.
pub fn write(group: Group, cfg: &Config, out: &mut String) {
    let default = Config::default();
    for row in in_group(group) {
        row.write(cfg, &default, out);
    }
}

/// Set every row of `cfg` from the checkpoint `config` fields.
pub fn read(cfg: &mut Config, fields: &BTreeMap<String, String>) -> Result<(), String> {
    for row in rows() {
        row.read(cfg, fields.get(row.key()).map(|s| s.as_str()))?;
    }
    Ok(())
}

/// Push the key of every active row of `group` whose value differs.
pub fn differences(group: Group, a: &Config, b: &Config, out: &mut Vec<&'static str>) {
    for row in in_group(group) {
        if row.active(a) && row.differs(a, b) {
            out.push(row.key());
        }
    }
}

/// Push a resume conflict line for every row of `group` that the command line
/// names, that is active in the saved configuration and that differs from it.
pub fn conflicts(group: Group, cfg: &Config, saved: &Config, explicit: &Explicit, out: &mut Vec<String>) {
    for row in in_group(group) {
        if explicit.contains(row) && row.active(saved) && row.differs(cfg, saved) {
            out.push(row.conflict_line(cfg, saved));
        }
    }
}

fn uint(flag: &str, text: &str) -> Result<usize, String> {
    text.parse().map_err(|_| format!("{} needs a non-negative integer, got '{}'", flag, text))
}

fn island_policy(cfg: &Config) -> bool {
    cfg.restart_policy == RestartPolicy::Island
}

/// The table. Within a group, the order is the order of every reader.
pub static TABLE: &[&dyn Setting] = &[
    &Row::<usize> {
        flag: "--patience",
        key: "patience",
        group: Group::Stop,
        schema_type: "int",
        schema_default: "40",
        schema: "stop after N gens without improvement",
        help: "  --patience <N>             stop after N generations without improvement (default 40)",
        bench: Some(Bench {
            usage: "  --patience <N>            generations without improvement before stopping",
            omit_unset: false,
        }),
        gui: true,
        get: |c| &c.patience,
        set: |c| &mut c.patience,
        cli: Cli::Value(|v| uint("--patience", v)),
        write: Write::Required,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<usize> {
        flag: "--migration-interval",
        key: "migration_interval",
        group: Group::Schedule,
        schema_type: "int",
        schema_default: "25",
        schema: "#1116: generations between island migrations (0: no migration). No effect with --islands 1. Saved search setting: a resume refuses a different value",
        help: "  --migration-interval <N>   generations between island migrations,
                             default 25. 0 turns migration off. No effect
                             with --islands 1. Saved search setting (#1116)",
        bench: Some(Bench {
            usage: "  --migration-interval <N>  generations between island migrations",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.migration_interval,
        set: |c| &mut c.migration_interval,
        cli: Cli::Value(|v| uint("--migration-interval", v)),
        write: Write::Required,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<usize> {
        flag: "--restart-interval",
        key: "restart_interval",
        group: Group::Schedule,
        schema_type: "int",
        schema_default: "60",
        schema: "#1109: generations between island restart checks (0: no restart check, so no restart of any kind, explore islands included). The stall length of --restart-policy island and the lifetime unit of --explore-islands. Saved search setting: a resume refuses a different value",
        help: "  --restart-interval <N>     generations between island restart checks,
                             default 60. Also the stall length of
                             --restart-policy island and the lifetime unit of
                             --explore-islands. 0 turns every restart off,
                             explore restarts included. Saved search setting
                             (#1109)",
        bench: Some(Bench {
            usage: "  --restart-interval <N>    generations between island restarts",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.restart_interval,
        set: |c| &mut c.restart_interval,
        cli: Cli::Value(|v| uint("--restart-interval", v)),
        write: Write::Required,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<Selection> {
        flag: "--selection",
        key: "selection",
        group: Group::Island,
        schema_type: "enum",
        schema_default: "lexicographic",
        schema: "#465: survivor and parent selection inside an island: lexicographic (generational replacement, tournament on error then size) or afp (age-fitness Pareto: survivors of parents, offspring and random immigrants by non-dominated sorting on error, complexity and age with crowding distance; parents by Pareto rank and crowding). Saved search setting. afp runs on the node path; --variation postfix with afp exits 2, auto picks the node path",
        help: "  --selection lexicographic|afp
                             survivor and parent selection inside an island.
                             lexicographic (default) replaces the island with
                             its offspring and picks parents by error, then
                             size. afp keeps the Pareto-best of parents and
                             offspring on (error, complexity, age), adds a few
                             random immigrants per generation and picks
                             parents by Pareto rank and crowding distance.
                             Node path only; refuses --variation postfix",
        bench: Some(Bench {
            usage: "  --selection <mode>        lexicographic (default) or afp (age-fitness Pareto)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.selection,
        set: |c| &mut c.selection,
        cli: Cli::Value(Selection::parse),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<RestartPolicy> {
        flag: "--restart-policy",
        key: "restart_policy",
        group: Group::Island,
        schema_type: "enum",
        schema_default: "archive",
        schema: "#1022: archive | island. Which stall rule restarts an island at every restart interval (--restart-interval, default 60 generations). archive: an island restarts after two checks in which its best error is worse than the archive best; once ring migration has spread the archive best to every island no island is ever behind it, so no restart fires. island: an island restarts when its own best error has not improved (relative 1e-9, the patience rule) for one restart interval, except the keeper, the lowest island whose best error equals the archive best; the restarted island takes no front parent and no migrant for --restart-isolation generations. With --islands 1 the only island is the keeper and nothing restarts. Applies to the node path and the flat postfix path. An isolated island also takes no --qd-policy parent (neither its global-front share nor a QD-archive parent): it breeds from its own members only (#1067). Saved search setting: a resume refuses a different value; a checkpoint without it loads as archive; island adds island_stall records to the checkpoint. The development log carries restarts, restarts_total and isolated",
        help: "  --restart-policy archive|island
                             which stall rule restarts an island every
                             restart interval (--restart-interval, default
                             60 generations). archive (default) restarts an
                             island after two checks in which its best error
                             is worse than the archive best. island restarts
                             an island whose own best error has not improved
                             for one restart interval, except the lowest
                             island that holds the archive best (the keeper),
                             and isolates it from front parents, --qd-policy
                             parents and migrants for --restart-isolation
                             generations. With one island nothing restarts.
                             Saved search setting (#1022)",
        bench: Some(Bench {
            usage: "  --restart-policy <mode>   archive (default) or island: restart an island on its own
                            stall and isolate it afterwards (#1022)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.restart_policy,
        set: |c| &mut c.restart_policy,
        cli: Cli::Value(RestartPolicy::parse),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<usize> {
        flag: "--restart-isolation",
        key: "restart_isolation",
        group: Group::Island,
        schema_type: "int",
        schema_default: "25",
        schema: "#1022: generations a restarted island takes no front parent, no --qd-policy parent and no migrant (the migration interval by default; 0 turns the isolation off). Read only under --restart-policy island. Saved search setting under island: a resume refuses a different value",
        help: "  --restart-isolation <N>    generations a restarted island takes no front
                             parent, no --qd-policy parent and no migrant,
                             default 25 (the migration interval). Read only
                             with --restart-policy island; 0 turns the
                             isolation off. Saved search setting (#1022)",
        bench: Some(Bench {
            usage: "  --restart-isolation <N>   generations a restarted island takes no front parent and
                            no migrant under --restart-policy island (default 25)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.restart_isolation,
        set: |c| &mut c.restart_isolation,
        cli: Cli::Value(|v| uint("--restart-isolation", v)),
        write: Write::UnlessDefault,
        active: Some(island_policy),
        conflict: Conflict::SavedKey,
    },
    &Row::<usize> {
        flag: "--explore-islands",
        key: "explore_islands",
        group: Group::Island,
        schema_type: "int",
        schema_default: "0",
        schema: "#1100: the highest K islands form an explore tier (0: none; K must be below --islands). An explore island takes no front parent and no --qd-policy parent, sends and receives no migrant, and restarts at the end of each lifetime of a Luby schedule (1, 1, 2, 1, 1, 2, 4, ... units of --restart-interval, default 60 generations, list-scheduled over the explore islands: each next lifetime goes to the island whose lifetimes add up to the least, ties to the lowest), so the restart generations depend on the generation alone. The other islands (the exploit tier) keep --restart-policy and a directed ring among themselves; explore results reach them only through the archive and its front parents. Applies to the node path and the flat postfix path. The development log counts explore restarts in restarts and restarts_total; isolated counts exploit windows only. Saved search setting: written to the checkpoint only when above 0, a resume refuses a different value, a checkpoint without it loads as 0",
        help: "  --explore-islands <K>      the highest K islands form an explore tier,
                             default 0 (none). An explore island takes no
                             front parent, no --qd-policy parent and no
                             migrant, sends no migrant, and restarts at the
                             end of each lifetime of a Luby schedule in
                             units of --restart-interval (default 60
                             generations; 0 turns restarts off, explore
                             islands included). The other islands keep
                             --restart-policy and a ring of their own;
                             explore results reach them through the
                             archive only. K must be below --islands.
                             Saved search setting (#1100)",
        bench: Some(Bench {
            usage: "  --explore-islands <K>     the highest K islands form an explore tier with Luby
                            restarts, no front parents and no migrants (default 0)",
            omit_unset: true,
        }),
        gui: false,
        get: |c| &c.explore_islands,
        set: |c| &mut c.explore_islands,
        cli: Cli::Value(|v| uint("--explore-islands", v)),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<FrontScope> {
        flag: "--front-scope",
        key: "front_scope",
        group: Group::Island,
        schema_type: "enum",
        schema_default: "global",
        schema: "#1070: global | island. Where an island draws its front parents (the 15 % #76 draw). global: the Pareto front of every island's members, so a new best reaches every island within about one generation. island: each island keeps its own error/complexity front of the members it scored (same insert rule, cap and #848 filter as the archive); at a migration the destination island's front becomes the canonical front of its own and the source's members, an island that receives no migrant (isolated or explore) keeps its front, and a restart empties it. The reported archive, patience and the restart rules still read the global archive, which every member enters each generation. Applies to the node path and the flat postfix path. Not with --qd-policy (exit 2): the QD archive is one parent pool shared by every island. Saved search setting: written to the checkpoint only for island together with island_front records, a resume refuses a different value, a checkpoint without it loads as global",
        help: "  --front-scope global|island
                             where front parents come from. global
                             (default) draws them from the Pareto front of
                             every island. island gives each island its own
                             error/complexity front; a front moves only
                             with a migrant (the destination front becomes
                             the front of both) and a restart empties it.
                             The reported archive stays global. Not with
                             --qd-policy. Saved search setting (#1070)",
        bench: Some(Bench {
            usage: "  --front-scope <scope>     global (default) or island: each island draws front parents
                            from its own front, exchanged along the migration moves",
            omit_unset: true,
        }),
        gui: false,
        get: |c| &c.front_scope,
        set: |c| &mut c.front_scope,
        cli: Cli::Value(FrontScope::parse),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<MigrationPolicy> {
        flag: "--migration-policy",
        key: "migration_policy",
        group: Group::Island,
        schema_type: "enum",
        schema_default: "ring",
        schema: "#1029: ring | ring-tournament. Which member an island sends at a migration (every --migration-interval generations). ring: the best member of the next island replaces the worst member of this one, so the global best reaches every island within a few migrations. ring-tournament: the migrant is the winner of a tournament of 8 draws over the source island, drawn with the destination island's own generator (node path) or a positional Philox stream (flat postfix path), so the source island's trajectory does not depend on who migrates from it; a source without a finite member sends nothing, and a destination that already holds a member with the migrant's error bits skips the migration. The victim, the isolation and explore rules and the island fronts are unchanged. Applies to the node path and the flat postfix path. Saved search setting: written to the checkpoint only for ring-tournament, a resume refuses a different value, a checkpoint without it loads as ring. The development log carries migrants and migrants_skipped",
        help: "  --migration-policy ring|ring-tournament
                             which member an island sends at a migration.
                             ring (default) sends the best member of the next
                             island. ring-tournament sends the winner of a
                             tournament of 8 draws over it, drawn with the
                             destination's generator, and skips a destination
                             that holds a member with the same error. Saved
                             search setting (#1029)",
        bench: Some(Bench {
            usage: "  --migration-policy <mode> ring (default) or ring-tournament: a random good migrant,
                            skipped when the destination holds its error (#1029)",
            omit_unset: true,
        }),
        gui: false,
        get: |c| &c.migration_policy,
        set: |c| &mut c.migration_policy,
        cli: Cli::Value(MigrationPolicy::parse),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<MutationMix> {
        flag: "--mutation-mix",
        key: "mutation_mix",
        group: Group::Variation,
        schema_type: "enum",
        schema_default: "classic",
        schema: "#471: classic | extended. Edit weights of the node-path mutation. classic: point 0.25, terminal 0.35, subtree 0.40. extended: point 0.20, terminal 0.30, subtree 0.20, insert 0.10, delete 0.10, swap 0.05, rotate 0.05. Saved search setting; a resume refuses a change. Exit 2 with --variation postfix",
        help: "  --mutation-mix <M>         node-path edit weights: classic (point .25, terminal
                             .35, subtree .40; default) or extended (adds insert,
                             delete, swap and rotate). Not with --variation
                             postfix (#471)",
        bench: Some(Bench {
            usage: "  --mutation-mix <mix>      classic (default) or extended: insert, delete, swap, rotate (#471)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.mutation_mix,
        set: |c| &mut c.mutation_mix,
        cli: Cli::Value(|v| MutationMix::parse(v).map_err(|e| format!("--mutation-mix: {}", e))),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<f64> {
        flag: "--crossover-internal",
        key: "crossover_internal",
        group: Group::Variation,
        schema_type: "float",
        schema_default: "0",
        schema: "#471: probability in [0, 1] that each crossover cut point is an internal node when the tree has one (Koza 0.9). 0 keeps the uniform cut. Saved search setting. Exit 2 with --variation postfix",
        help: "  --crossover-internal <P>   probability that a crossover cut point is an
                             internal node (default 0 = uniform cut). Not with
                             --variation postfix (#471)",
        bench: Some(Bench {
            usage: "  --crossover-internal <P>  probability of an internal-node crossover cut (default 0) (#471)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.crossover_internal,
        set: |c| &mut c.crossover_internal,
        cli: Cli::Value(|v| match v.parse::<f64>() {
            Ok(p) if (0.0..=1.0).contains(&p) => Ok(p),
            _ => Err(format!("--crossover-internal needs a probability in [0, 1], got '{}'", v)),
        }),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<bool> {
        flag: "--adaptive-mix",
        key: "adaptive_mix",
        group: Group::Variation,
        schema_type: "flag",
        schema_default: "false",
        schema: "#471: each island learns edit weights from the last 10 generations: w_k = max(0.02, (s_k + 1) / (a_k + 2)), normalised, where a child counts as a success when its error is below its first parent's. Uses all seven edit kinds; the --mutation-mix weights stand until a generation has been observed. Learns only from CPU-scored generations (a deferred GPU batch or a screening subset keeps the base weights). The counts are saved in the checkpoint. Saved search setting. Exit 2 with --variation postfix",
        help: "  --adaptive-mix             per-island edit weights learned from the success
                             rate of the last 10 generations (uses all seven edit
                             kinds; learns from CPU-scored generations only).
                             Off by default; not with --variation postfix (#471)",
        bench: Some(Bench {
            usage: "  --adaptive-mix            per-island edit weights from success rates (#471)",
            omit_unset: false,
        }),
        gui: false,
        get: |c| &c.adaptive_mix,
        set: |c| &mut c.adaptive_mix,
        cli: Cli::Flag(true),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<bool> {
        flag: "--reject-constant-offspring",
        key: "reject_constant_offspring",
        group: Group::Filter,
        schema_type: "flag",
        schema_default: "false",
        schema: "#477: reject offspring and random fill or restart trees with no input outside an aggregate or meta operator before the constant fit. The first such constant tree of an island is kept, so the front keeps its constant baseline; later ones count as rejected attempts (development log variation.constant_rejected). Seeds and migrants are never filtered. A no-op without input variables. Saved search state: a resume refuses a different value; a checkpoint without it loads as off",
        help: "  --reject-constant-offspring
                             reject offspring and random trees with no input
                             outside an aggregate before the constant fit; the
                             first constant of an island is kept (#477). Saved
                             search setting, off by default
                             and restores the pre-#76 trajectory exactly",
        bench: Some(Bench {
            usage: "  --reject-constant-offspring  reject input-free offspring before the constant fit (#477)",
            omit_unset: false,
        }),
        gui: true,
        get: |c| &c.reject_constant_offspring,
        set: |c| &mut c.reject_constant_offspring,
        cli: Cli::Flag(true),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<bool> {
        flag: "--reject-fixed-position",
        key: "reject_fixed_position",
        group: Group::Filter,
        schema_type: "flag",
        schema_default: "false",
        schema: "#848: under --fitness pnl, reject offspring and random fill or restart trees whose long/flat position cannot depend on the inputs before the constant fit, and keep them out of the archive and the rebuilt front. Before the constant fit the test takes every constant as the whole real line, so only a tree no constant values could make selective is rejected; archive admission and the rebuilt front test the fitted values. The test is the exact output range of the tree over all real inputs: a range that does not cross --threshold (a boundary that touches it counts) fixes the position, for example gauss(c * x), exp(c * x) and logistic(c * x) at threshold 0 (the same trees cross a threshold of 1e-30). Such a tree prices buy and hold or takes flat rows from float underflow, never a signal; the device f32 program underflows earlier than f64, so the two backends can rank it differently. Operators without a range rule count as unbounded, so the test only rejects what it proves. Counted in the development log as variation.fixed_position_rejected. Seeds and migrants are never filtered. When every candidate is fixed the front keeps them and the run JSON search.fixed_position_front_fallback is true. The search effect needs --fitness pnl and input variables. With --objective (#849) the flag also makes a member with a fixed position unusable as finalist under any --fitness: its objective is null and objective.usable_candidates counts it out (so does a validation book that was long on every row or on none, whatever the threshold: fixed_position_realised); run.json reports validation_book.fixed_position (long, flat or null) for every member and objective.finalist.fixed_position without the flag. Saved search state: a resume refuses a different value; a checkpoint without it loads as off",
        help: "  --reject-fixed-position    under --fitness pnl, reject trees whose long/flat
                             position cannot depend on the inputs (the exact
                             output range does not cross --threshold, for
                             example gauss(c * x) at threshold 0) before the
                             constant fit, and keep them off the front. Before
                             the fit every constant counts as free, so c + abs(x)
                             goes to the fit (#848). With --objective it also
                             makes such a member unusable as finalist, under
                             --fitness metric too; so is a member whose
                             validation book was long on every row or on none
                             (#849). Saved search setting, off by default",
        bench: None,
        gui: true,
        get: |c| &c.reject_fixed_position,
        set: |c| &mut c.reject_fixed_position,
        cli: Cli::Flag(true),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<bool> {
        flag: "--fit-thresholds",
        key: "fit_thresholds",
        group: Group::Fit,
        schema_type: "flag",
        schema_default: "false",
        schema: "#1024: after the LM iterations, fit by an exact 1-D search every constant that is an operand of a comparison (greater, less, greater_or_equal, less_or_equal, equal) or sits in the condition of if (directly or as the constant of a sub or add against a constant), which the LM never moves because its adjoint is zero. The candidates are the midpoints between consecutive sorted operand values on the fit rows, scored in levels of at most 64 that narrow around the best until every midpoint of the range is scored (at most 192 evaluations per threshold for 65,535 midpoints); the loss is the one the constants are fitted on (weighted squared error for the squared family and for --fitness pnl, the configured metric otherwise); the start value is kept unless a candidate is strictly better; ties keep the start, then the smaller absolute threshold. At most 2 constants per tree; constants under an aggregate, a custom block or a meta operator are never fitted. After a move the other constants are refit once on the new partition. Applies to the search fit, the prune refit and the exact finalist refit. A tree with such a constant takes the exact CPU fit arm, never the device, so the ledger counts it under the fallback cause threshold_constant. No effect with --fit-iters 0. Reported as threshold_moves in the development log (fit). A saved search setting: a resume refuses a different value; a checkpoint without it loads as off",
        help: "  --fit-thresholds          after the LM, fit each constant that is an operand of a
                             comparison or sits in the condition of if by an exact
                             1-D search (default off). The LM never moves such a
                             constant because its gradient is zero. Candidates are
                             the midpoints between sorted operand values on the
                             fit rows (levels of at most 64, narrowing around the best),
                             scored on the fitted loss (squared error for the
                             squared family and --fitness pnl); at most 2
                             constants per tree. Trees with such a constant take
                             the CPU fit arm, never the device. A saved search
                             setting: a resume refuses a different value. No
                             effect with --fit-iters 0",
        bench: Some(Bench {
            usage: "  --fit-thresholds          fit comparison and if thresholds by an exact 1-D search (#1024)",
            omit_unset: true,
        }),
        gui: true,
        get: |c| &c.fit_thresholds,
        set: |c| &mut c.fit_thresholds,
        cli: Cli::Flag(true),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::Saved,
    },
    &Row::<VariationMode> {
        flag: "--variation",
        key: "variation",
        group: Group::Engine,
        schema_type: "enum",
        schema_default: "auto",
        schema: "variation engine: auto (flat postfix arena for --no-fit GPU runs with population >= 4096), node (historical owned-tree loop) or postfix (require the flat path; error when the run does not qualify)",
        help: "  --variation auto|node|postfix
                             variation engine. auto (default) uses the flat
                             postfix arena for --no-fit GPU runs with a
                             population >= 4096; node keeps the owned-tree loop;
                             postfix requires the flat path and refuses a run
                             that does not qualify",
        bench: None,
        gui: false,
        get: |c| &c.variation,
        set: |c| &mut c.variation,
        cli: Cli::Value(VariationMode::parse),
        write: Write::Always,
        active: None,
        conflict: Conflict::SavedKey,
    },
    &Row::<FitAccept> {
        flag: "--fit-accept",
        key: "fit_accept",
        group: Group::Fit,
        schema_type: "enum",
        schema_default: "always",
        schema: "#1078: always | fitness. When a candidate keeps its fitted constants. always: every fit is kept. fitness: a candidate keeps the fitted constants only when its search fitness (the error the selection reads, scored on the scoring rows) is not worse than with the constants it had before the fit; a strictly lower unfitted error keeps the unfitted tree, a tie keeps the fitted one, a finite unfitted error beats a non-finite fitted one. Under --fitness pnl the fit minimises squared error and can move a profitable long/flat boundary; the rule keeps that boundary. Applies to the offspring fit, the restart fill, the carried-member refit under predictor subsampling, the device fit (including the exact CPU tail and fallback fits), the screening-survivor refit, the finalist refit, the prune refit and the seed fit before generation 1 (a seed that loses keeps its pre-fit constants; seeds[].fit_kept reports it). Costs one extra score per fitted candidate. Under fitness the development log fit record carries accepted and rejected (this generation) and the run JSON search.fit_accept carries accepted, rejected and by_path [accepted, rejected] for offspring, restart, carried, device, lost_batch, survivor, finalist, prune and seed; a fit that moved nothing is not counted; nothing is written under always. With --fit-iters 0 nothing is fitted and the rule is idle. Under fitness the overlap pipeline does not fit (--pipeline-depth overlap falls back to the synchronous fit batches). Saved search setting: a resume refuses a different value; a checkpoint without it loads as always",
        help: "  --fit-accept always|fitness
                             when a candidate keeps its fitted constants.
                             always (default) keeps every fit. fitness keeps
                             them only when the search fitness is not worse
                             than with the constants before the fit (a strictly
                             better unfitted tree is kept; a tie keeps the fit).
                             Under --fitness pnl the squared-error fit can move
                             a profitable long/flat boundary. One extra score
                             per fitted candidate, also for the seed fit. Under
                             fitness the dev log and the --json search.fit_accept
                             count the kept and refused fits. Saved search
                             setting (#1078)",
        bench: Some(Bench {
            usage: "  --fit-accept <mode>       always (default) or fitness: keep fitted constants only when the fitness is not worse (#1078)",
            omit_unset: true,
        }),
        gui: false,
        get: |c| &c.fit_accept,
        set: |c| &mut c.fit_accept,
        cli: Cli::Value(FitAccept::parse),
        write: Write::UnlessDefault,
        active: None,
        conflict: Conflict::SavedKey,
    },
];
