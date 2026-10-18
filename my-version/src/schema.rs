//! The machine-readable CLI schema.
//!
//! One source of truth for `rcsr schema` and the GUI's `/api/schema`. The
//! option table, operator list, complexity formula and exit codes live here, so
//! adding a flag reaches both surfaces without a second edit (and without a
//! test that can only detect the drift after it happens).

use crate::expr::Op;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// The schema document, identical for the CLI and the server.
pub fn json() -> String {
    let opts = [
        ("--data", "string", "none", "table path, or '-' for stdin"),
        ("--target", "string", "none", "target column name"),
        ("--vars", "csv", "all-except-target", "input column names"),
        ("--dtype", "csv", "all-continuous", "per-variable datatype, col=continuous|boolean|categorical (categorical expands to one-hot boolean columns named col=value)"),
        ("--max-levels", "int", "16", "maximum distinct values in a categorical column"),
        ("--test-frac", "float", "0.25", "held-out fraction"),
        ("--weight", "string", "none", "per-row weight column (finite, >= 0, at least one positive; supported metrics: absolute, squared, correlation, signed, log_squashed, aic, aicc, bic)"),
        ("--fill", "enum", "none", "missing-value method: none | drop | mean | median | zero | previous | interpolate"),
        ("--outlier-iqr", "float", "none", "drop rows beyond k interquartile ranges from the training median"),
        ("--smooth", "int", "0", "centred moving-average window in rows (0 = off)"),
        ("--center", "enum", "mean", "subtract mean | median | min | max | none before the search"),
        ("--scale", "enum", "std", "divide by std | iqr | range | none before the search"),
        ("--filter", "string", "none", "keep rows where this expression is > 0"),
        ("--exclude", "string", "none", "drop rows where this expression is > 0"),
        ("--train-select", "string", "none", "rows where this expression is > 0 go to training (overrides --split)"),
        ("--test-select", "string", "none", "rows where this expression is > 0 go to test (overrides --split)"),
        ("--split", "enum", "random", "random | temporal | group | temporal-group | auto"),
        ("--time", "string", "none", "time/ordering column (implies temporal)"),
        ("--group-by", "string", "none", "group column (implies group)"),
        ("--gap", "int", "0", "embargo rows between train and test"),
        ("--folds", "int", "1", "contiguous test blocks to report"),
        ("--lags", "int", "0", "delayed copies per input (time-delay embedding)"),
        ("--history-frac", "float", "0", "fraction of each series kept as history"),
        ("--gap-break", "float", "0", "new series when dt exceeds f x median dt"),
        ("--null-runs", "int", "0", "y-randomisation control runs (shuffled targets)"),
        ("--bootstrap-error", "int", "0", "bootstrap resamples for the .632+ error estimate (0 = off)"),
        ("--cv", "string", "none", "cross-validation scheme: k-fold | rolling | purged"),
        ("--cv-k", "int", "5", "number of cross-validation folds"),
        ("--stability", "int", "0", "bootstrap resamples for stability selection (0 = off)"),
        ("--metric", "enum", "absolute", "absolute | squared | correlation | median | maximum | signed | log_squashed | rank_correlation | interquartile | slope | auc | log_loss | hinge | aic | aicc | bic"),
        ("--generations", "int", "200", "max generations"),
        ("--islands", "int", "1", "number of islands"),
        ("--pop", "int", "200", "population per island"),
        ("--max-complexity", "int", "30", "max configured complexity per expression"),
        ("--op-complexity", "csv", "all-1", "per-operator costs name=cost, e.g. pow=2,sin=1.5; unlisted operators cost 1"),
        ("--complexity-weights", "csv", "var=1,const=1", "terminal weights name=weight; keys var, const and depth (depth defaults to 0)"),
        ("--depth-weight", "float", "0", "extra cost per nesting level (edges below the root)"),
        ("--seed", "int", "0", "RNG seed"),
        ("--seed-expr", "string", "none", "seed expression in original units (repeatable); parsed, simplified, constants fitted, then inserted into every island and the archive; kept even above --max-complexity"),
        ("--seed-file", "string", "none", "file of seed expressions, one per line ('#' comments and blank lines skipped; repeatable)"),
        ("--ops", "csv", "builtin-set", "operator names: primitives, comparisons, logical operators, constraints and meta operators (see \"operators\")"),
        ("--eval-expr", "string", "none", "score this formula on the split instead of searching"),
        ("--function", "string", "none", "user-defined building block name(a,b)=expr or name=expr (repeatable, at most four; parameters inferred from free variables; constants in a body are shared by every call)"),
        ("--fit-iters", "int", "8", "LM constant-fitting iterations"),
        ("--no-fit", "flag", "false", "disable constant fitting"),
        ("--no-dedup", "flag", "false", "disable structural de-duplication"),
        ("--predictor-samples", "int", "none", "training rows per generation for fitness and constant fitting (>=2; all rows when >= train size)"),
        ("--patience", "int", "40", "stop after N gens without improvement"),
        ("--time-budget", "float", "none", "wall-clock seconds"),
        ("--save", "string", "none", "write a checkpoint on exit and every --save-every generations"),
        ("--save-every", "int", "25", "checkpoint interval in generations (requires --save)"),
        ("--resume", "string", "none", "resume a checkpoint; saved search settings win, only --generations may be raised, data must match"),
        ("--format", "enum", "text", "text | json | ndjson"),
        ("--top", "int", "0", "limit reported Pareto rows (0 = all)"),
        ("--out", "string", "none", "write Pareto front TSV to this path"),
        ("--report", "enum", "none", "Markdown report: summary | sensitivity | benchmark (single value; last wins)"),
        ("--report-out", "string", "none", "write the Markdown report to this path (otherwise text=stdout, json/ndjson=stderr)"),
        ("--report-repeats", "int", "5", "independent searches for --report benchmark"),
    ];
    let mut items = String::new();
    for (i, (name, ty, default, desc)) in opts.iter().enumerate() {
        if i > 0 {
            items.push(',');
        }
        items.push_str(&format!(
            "\n    {{\"name\":{},\"type\":{},\"default\":{},\"description\":{}}}",
            jstr(name),
            jstr(ty),
            jstr(default),
            jstr(desc)
        ));
    }
    let operators: Vec<String> = Op::names().iter().map(|n| jstr(n)).collect();
    format!(
        "{{\n  \"tool\":\"rcsr\",\n  \"version\":{},\n  \"summary\":{},\n  \
         \"options\":[{}\n  ],\n  \
         \"operators\":[{}],\n  \
         \"complexity\":{{\"formula\":\"sum(op_cost) + var_weight*n_vars + const_weight*n_consts + depth_weight*max_depth\",\"defaults\":{{\"op_cost\":1,\"var_weight\":1,\"const_weight\":1,\"depth_weight\":0}},\"operator_nodes_only\":true,\"depth\":\"edges from the root; a single node has depth 0\"}},\n  \
         \"exit_codes\":{{\"0\":\"success\",\"2\":\"usage error\",\"3\":\"input error\",\"4\":\"no finite solution\"}},\n  \
         \"stdout\":\"machine-readable result\",\n  \"stderr\":\"human-readable log and progress\"\n}}\n",
        jstr(VERSION),
        jstr("symbolic regression: find an interpretable f(x) ~ y"),
        items,
        operators.join(",")
    )
}

/// JSON string literal, escaping only what JSON requires.
fn jstr(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            '\r' => o.push_str("\\r"),
            '\t' => o.push_str("\\t"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

