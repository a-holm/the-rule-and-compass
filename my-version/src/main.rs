//! `rcsr` — command-line symbolic-regression engine.
//!
//! Designed to be driven by other programs (including AI agents):
//! * everything machine-readable goes to **stdout**, anything human-facing to
//!   **stderr**, so `rcsr ... --json | jq` and `rcsr ... > out.json` just work;
//! * `rcsr schema` prints a self-describing JSON document (options + exit codes)
//!   with no side effects;
//! * `--json` emits one JSON object; `--ndjson` streams one object per line
//!   (start, progress, result) for long runs;
//! * runs are deterministic given `--seed`, and the seed is echoed in the output;
//! * `--time-budget` bounds wall-clock time, so a caller never blocks forever;
//! * exit codes are stable and documented (see [`EXIT_USAGE`] etc.).

use rcsr::expr::{self, Op};
use rcsr::metrics::Metric;
use rcsr::rng::Mt;
use rcsr::{data, search};
use std::io::Read;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const EXIT_OK: i32 = 0;
const EXIT_USAGE: i32 = 2;
const EXIT_INPUT: i32 = 3;
const EXIT_NO_SOLUTION: i32 = 4;

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Text,
    Json,
    Ndjson,
}

fn usage() -> String {
    format!(
        "rcsr {VERSION} — symbolic regression (rule-and-compass)

USAGE:
  rcsr --data <file|-> --target <col> [options]
  rcsr schema                 print a machine-readable description of this tool
  rcsr --help | --version

INPUT:
  --data <file|->            CSV/TSV/whitespace table, header row required
                             ('-' reads the table from standard input)
  --target <col>             name of the target column
  --vars <a,b,..>            input columns (default: all numeric columns except
                             the target and any --time/--group-by column)
  --test-frac <f>            held-out fraction (default 0.25)

HOW ROWS ARE SPLIT (important for time series and panel data):
  --split random|temporal|group|temporal-group|auto   (default random)
                             random          shuffled i.i.d. holdout
                             temporal        train on the past, test on the future
                             group           hold out whole groups (unseen stocks,
                                             subjects, sensors), never both sides
                             temporal-group  a temporal split inside every group
                             auto            infer from the columns and report
  --time <col>               time/ordering column; implies temporal
  --group-by <col>           group column; implies group (with --time: temporal
                             inside each group). Text columns are allowed.
  --gap <rows>               embargo: rows dropped between train and test to stop
                             autocorrelation leaking the future (default 0)
  --folds <k>                split the test region into k contiguous blocks and
                             report per-block error as well as the mean (default 1)

SEARCH:
  --metric absolute|squared|correlation  (default absolute)
  --generations <N>          (default 200)
  --islands <I>              (default 1)
  --pop <P>                  population per island (default 200)
  --max-complexity <K>       max nodes per expression (default 30)
  --seed <S>                 RNG seed, echoed in output (default 0)
  --ops <list>               e.g. add,sub,mul,div,sin,exp
  --fit-iters <N>            Levenberg–Marquardt iterations for constants (default 8)
  --no-fit                   disable constant fitting
  --no-dedup                 disable structural de-duplication
  --patience <N>             stop after N generations without improvement (default 40)
  --time-budget <seconds>    wall-clock limit (default: none)

OUTPUT:
  --format text|json|ndjson  (default text)
  --json                     shorthand for --format json
  --ndjson                   shorthand for --format ndjson
  --progress                 emit per-generation progress (stderr, or stdout in ndjson)
  --top <N>                  report at most N Pareto rows (default: all)
  --quiet                    suppress the human-readable header
  --out <file>               also write the Pareto front as TSV
  -h, --help                 show this help

EXIT CODES:
  0 success   2 usage error   3 input/data error   4 no finite solution found"
    )
}

fn schema() -> String {
    let opts = [
        ("--data", "string", "none", "table path, or '-' for stdin"),
        ("--target", "string", "none", "target column name"),
        ("--vars", "csv", "all-except-target", "input column names"),
        ("--test-frac", "float", "0.25", "held-out fraction"),
        ("--split", "enum", "random", "random | temporal | group | temporal-group | auto"),
        ("--time", "string", "none", "time/ordering column (implies temporal)"),
        ("--group-by", "string", "none", "group column (implies group)"),
        ("--gap", "int", "0", "embargo rows between train and test"),
        ("--folds", "int", "1", "contiguous test blocks to report"),
        ("--metric", "enum", "absolute", "absolute | squared | correlation"),
        ("--generations", "int", "200", "max generations"),
        ("--islands", "int", "1", "number of islands"),
        ("--pop", "int", "200", "population per island"),
        ("--max-complexity", "int", "30", "max nodes per expression"),
        ("--seed", "int", "0", "RNG seed"),
        ("--ops", "csv", "builtin-set", "allowed operators"),
        ("--fit-iters", "int", "8", "LM constant-fitting iterations"),
        ("--no-fit", "flag", "false", "disable constant fitting"),
        ("--no-dedup", "flag", "false", "disable structural de-duplication"),
        ("--patience", "int", "40", "stop after N gens without improvement"),
        ("--time-budget", "float", "none", "wall-clock seconds"),
        ("--format", "enum", "text", "text | json | ndjson"),
        ("--top", "int", "0", "limit reported Pareto rows (0 = all)"),
        ("--out", "string", "none", "write Pareto front TSV to this path"),
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
    format!(
        "{{\n  \"tool\":\"rcsr\",\n  \"version\":{},\n  \"summary\":{},\n  \
         \"options\":[{}\n  ],\n  \
         \"exit_codes\":{{\"0\":\"success\",\"2\":\"usage error\",\"3\":\"input error\",\"4\":\"no finite solution\"}},\n  \
         \"stdout\":\"machine-readable result\",\n  \"stderr\":\"human-readable log and progress\"\n}}\n",
        jstr(VERSION),
        jstr("symbolic regression: find an interpretable f(x) ~ y"),
        items
    )
}

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

/// JSON number, with non-finite values rendered as `null` (valid JSON).
fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

struct Row {
    rank: usize,
    complexity: usize,
    train_error: f64,
    test_error: f64,
    /// Per-block test errors (`--folds`); one entry when folds == 1.
    fold_errors: Vec<f64>,
    expression: String,
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        std::process::exit(EXIT_OK);
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("rcsr {}", VERSION);
        std::process::exit(EXIT_OK);
    }
    if args.first().map(|s| s.as_str()) == Some("schema") {
        print!("{}", schema());
        std::process::exit(EXIT_OK);
    }

    let mut cfg = search::Config::default();
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut test_frac = 0.25;
    let mut out_path: Option<String> = None;
    let mut format = Format::Text;
    let mut progress = false;
    let mut quiet = false;
    let mut top = 0usize;
    let mut split_arg: Option<String> = None;
    let mut time_name: Option<String> = None;
    let mut group_name: Option<String> = None;
    let mut gap = 0usize;
    let mut folds = 1usize;

    let mut i = 0;
    macro_rules! val {
        () => {{
            let v = args.get(i + 1).cloned().unwrap_or_else(|| {
                eprintln!("error: {} requires a value", args[i]);
                std::process::exit(EXIT_USAGE);
            });
            i += 1;
            v
        }};
    }
    while i < args.len() {
        match args[i].as_str() {
            "--data" => data_path = Some(val!()),
            "--target" => target = Some(val!()),
            "--vars" => {
                vars = val!()
                    .split(',')
                    .map(|x| x.trim().to_string())
                    .filter(|x| !x.is_empty())
                    .collect();
            }
            "--metric" => cfg.metric = Metric::parse(&val!()),
            "--generations" => cfg.generations = val!().parse().unwrap_or(cfg.generations),
            "--islands" => cfg.islands = val!().parse().unwrap_or(cfg.islands),
            "--pop" => cfg.pop = val!().parse().unwrap_or(cfg.pop),
            "--max-complexity" => cfg.max_complexity = val!().parse().unwrap_or(cfg.max_complexity),
            "--seed" => cfg.seed = val!().parse().unwrap_or(cfg.seed),
            "--test-frac" => test_frac = val!().parse().unwrap_or(test_frac),
            "--split" => split_arg = Some(val!()),
            "--time" => time_name = Some(val!()),
            "--group-by" => group_name = Some(val!()),
            "--gap" => gap = val!().parse().unwrap_or(0),
            "--folds" => folds = val!().parse().unwrap_or(1).max(1),
            "--ops" => cfg.ops = Op::parse_list(&val!()),
            "--fit-iters" => cfg.fit_iters = val!().parse().unwrap_or(cfg.fit_iters),
            "--no-fit" => cfg.fit_iters = 0,
            "--no-dedup" => cfg.dedup = false,
            "--patience" => cfg.patience = val!().parse().unwrap_or(cfg.patience),
            "--time-budget" => cfg.time_budget = val!().parse().ok(),
            "--format" => {
                format = match val!().as_str() {
                    "json" => Format::Json,
                    "ndjson" => Format::Ndjson,
                    "text" => Format::Text,
                    other => {
                        eprintln!("error: unknown --format '{}'", other);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--json" => format = Format::Json,
            "--ndjson" => format = Format::Ndjson,
            "--progress" => progress = true,
            "--top" => top = val!().parse().unwrap_or(0),
            "--quiet" => quiet = true,
            "--out" => out_path = Some(val!()),
            other => {
                eprintln!("error: unknown argument '{}'\nrun 'rcsr --help'", other);
                std::process::exit(EXIT_USAGE);
            }
        }
        i += 1;
    }

    let data_path = match data_path {
        Some(p) => p,
        None => {
            eprintln!("error: --data is required\n\n{}", usage());
            std::process::exit(EXIT_USAGE);
        }
    };
    let target = match target {
        Some(t) => t,
        None => {
            eprintln!("error: --target is required\n\n{}", usage());
            std::process::exit(EXIT_USAGE);
        }
    };

    let table = if data_path == "-" {
        let mut buf = String::new();
        if let Err(e) = std::io::stdin().read_to_string(&mut buf) {
            eprintln!("error reading stdin: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        data::parse_table(&buf)
    } else {
        data::read_table(&data_path)
    };
    let table = match table {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };

    let target_idx = match table.index_of(&target) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };

    // Resolve the two split-only columns (never used as model inputs).
    let mut time_col = match &time_name {
        Some(n) => match table.index_of(n) {
            Ok(j) => Some(j),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        },
        None => None,
    };
    let mut group_col = match &group_name {
        Some(n) => match table.index_of(n) {
            Ok(j) => Some(j),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        },
        None => None,
    };

    // Default inputs: numeric columns that are neither the target nor a
    // split-only column.
    if vars.is_empty() {
        vars = (0..table.names.len())
            .filter(|&j| {
                table.numeric[j]
                    && table.names[j] != target
                    && Some(j) != time_col
                    && Some(j) != group_col
            })
            .map(|j| table.names[j].clone())
            .collect();
    }
    if vars.is_empty() {
        eprintln!("error: no input variables");
        std::process::exit(EXIT_INPUT);
    }

    let known_vars: Vec<usize> = vars.iter().filter_map(|v| table.index_of(v).ok()).collect();
    let exclusion = |a: Option<usize>, b: Option<usize>| -> Vec<usize> {
        let mut e = vec![target_idx];
        e.extend_from_slice(&known_vars);
        if let Some(j) = a {
            e.push(j);
        }
        if let Some(j) = b {
            e.push(j);
        }
        e
    };

    let mut notes: Vec<String> = Vec::new();

    // Pick the split mode: explicit flag > columns named on the command line >
    // inference from the data.
    let requested = match split_arg.as_deref() {
        None => None,
        Some("auto") => None,
        Some(s) => match data::SplitMode::parse(s) {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
        },
    };
    let mode = if let Some(m) = requested {
        m
    } else if split_arg.as_deref() == Some("auto") {
        let (m, tc, gc, nt) = data::suggest_split(&table, target_idx, &known_vars);
        if time_col.is_none() {
            time_col = tc;
        }
        if group_col.is_none() {
            group_col = gc;
        }
        notes.extend(nt);
        m
    } else {
        match (time_col.is_some(), group_col.is_some()) {
            (true, true) => data::SplitMode::TemporalGroup,
            (true, false) => data::SplitMode::Temporal,
            (false, true) => data::SplitMode::Group,
            (false, false) => {
                // Warn rather than silently changing the split: a random split
                // of a series lets the model see the future.
                let (guess, tc, gc, nt) = data::suggest_split(&table, target_idx, &known_vars);
                if guess != data::SplitMode::Random {
                    let hint = match guess {
                        data::SplitMode::Temporal | data::SplitMode::TemporalGroup => {
                            let mut h = match tc {
                                Some(j) => format!("--time {}", table.names[j]),
                                None => "--time <col>".to_string(),
                            };
                            if let Some(j) = gc {
                                h = format!("{} --group-by {}", h, table.names[j]);
                            }
                            h
                        }
                        data::SplitMode::Group => match gc {
                            Some(j) => format!("--group-by {}", table.names[j]),
                            None => "--group-by <col>".to_string(),
                        },
                        data::SplitMode::Random => String::new(),
                    };
                    eprintln!(
                        "warning: the rows do not look exchangeable: {}\n         \
                         a random split would let the model see the future (or the same\n         \
                         group on both sides). Consider `--split auto` (chosen: {}),\n         \
                         or pass explicitly:  {}",
                        nt.join(" "),
                        guess.name(),
                        hint
                    );
                }
                data::SplitMode::Random
            }
        }
    };

    // Fill in a column the mode requires but the user did not name.
    if matches!(mode, data::SplitMode::Temporal | data::SplitMode::TemporalGroup)
        && time_col.is_none()
    {
        match data::find_time_col(&table, &exclusion(None, group_col)) {
            Some(j) => {
                notes.push(format!("time column: '{}'", table.names[j]));
                time_col = Some(j);
            }
            None => {
                eprintln!(
                    "error: --split {} needs --time <col> (no time-like column found; \
                     if the data is exchangeable use --split random)",
                    mode.name()
                );
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    if matches!(mode, data::SplitMode::Group | data::SplitMode::TemporalGroup)
        && group_col.is_none()
    {
        match data::find_group_col(&table, &exclusion(time_col, None)) {
            Some(j) => {
                notes.push(format!("group column: '{}'", table.names[j]));
                group_col = Some(j);
            }
            None => {
                eprintln!(
                    "error: --split {} needs --group-by <col> (no group-like column found)",
                    mode.name()
                );
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    if gap > 0 && !matches!(mode, data::SplitMode::Temporal | data::SplitMode::TemporalGroup) {
        notes.push("--gap only applies to temporal splits; ignored".to_string());
    }

    let spec = data::SplitSpec {
        mode,
        time_col,
        group_col,
        test_frac,
        gap,
        folds,
    };
    let mut rng = Mt::new(cfg.seed);
    let prepared = match data::prepare_full(&table, &vars, &target, &spec, &mut rng) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    notes.extend(prepared.report.notes.iter().cloned());

    // Streaming progress: ndjson goes to stdout, other formats to stderr.
    if progress {
        let to_stdout = format == Format::Ndjson;
        cfg.progress = Some(std::sync::Arc::new(move |p: search::Progress| {
            let line = format!(
                "{{\"event\":\"progress\",\"generation\":{},\"total\":{},\"best_error\":{},\"pareto_members\":{}}}",
                p.generation,
                p.total_generations,
                jnum(p.best_error),
                p.pareto_members
            );
            if to_stdout {
                println!("{}", line);
            } else {
                eprintln!("{}", line);
            }
        }));
    }

    let start = std::time::Instant::now();
    let split_line = format!(
        "split={} | train rows: {} | test rows: {}{}{}{}",
        prepared.report.mode,
        prepared.train.len(),
        prepared.test.len(),
        if prepared.report.folds > 1 {
            format!(" | folds: {}", prepared.report.folds)
        } else {
            String::new()
        },
        if prepared.report.gap > 0 {
            format!(" | gap: {}", prepared.report.gap)
        } else {
            String::new()
        },
        if prepared.report.groups_total > 0 {
            format!(
                " | groups: {}/{} in train",
                prepared.report.groups_train, prepared.report.groups_total
            )
        } else {
            String::new()
        }
    );
    for n in &notes {
        eprintln!("# {}", n);
    }
    if format == Format::Ndjson {
        let folds_json: Vec<String> = prepared
            .fold_samples
            .iter()
            .map(|f| f.len().to_string())
            .collect();
        println!(
            "{{\"event\":\"start\",\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{{\"mode\":{},\"gap\":{},\"folds\":[{}]}}}}",
            jstr(VERSION),
            cfg.seed,
            jstr(cfg.metric.name()),
            jstr(&target),
            vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
            prepared.train.len(),
            prepared.test.len(),
            jstr(prepared.report.mode),
            prepared.report.gap,
            folds_json.join(",")
        );
    } else if format == Format::Text && !quiet {
        eprintln!(
            "# rule-and-compass | {} island(s) x {} pop | metric={} | seed={}",
            cfg.islands,
            cfg.pop,
            cfg.metric.name(),
            cfg.seed
        );
        eprintln!(
            "# target: {} = f({}) | {}",
            target,
            vars.join(","),
            split_line
        );
        if prepared.report.groups_total > 1 && mode == data::SplitMode::Group {
            eprintln!(
                "# groups held out whole: {}/{} (never on both sides)",
                prepared.report.groups_test, prepared.report.groups_total
            );
        }
    }

    let outcome = search::run(&cfg, &prepared.train, &prepared.test);
    let elapsed = start.elapsed().as_secs_f64();

    // Test error is reported per block, then averaged, so an unstable model
    // cannot hide behind a lucky pooled number.
    let fold_samples: Vec<data::Sample> = if prepared.fold_samples.is_empty() {
        vec![prepared.test.clone()]
    } else {
        prepared.fold_samples.clone()
    };
    let sorted = outcome.archive.sorted();
    let mut rows: Vec<Row> = sorted
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let den = expr::simplify(&expr::denormalize(
                &expr::simplify(&s.tree),
                &prepared.means,
                &prepared.stds,
            ));
            let fold_errors: Vec<f64> = fold_samples
                .iter()
                .map(|f| search::eval_error(&s.tree, f, cfg.metric))
                .collect();
            let test_error = if fold_errors.is_empty() {
                f64::INFINITY
            } else {
                fold_errors.iter().sum::<f64>() / fold_errors.len() as f64
            };
            Row {
                rank: k + 1,
                complexity: s.complexity,
                train_error: s.error,
                test_error,
                fold_errors,
                expression: expr::to_string(&den, &vars),
            }
        })
        .collect();
    if top > 0 && rows.len() > top {
        rows.truncate(top);
    }

    let finite = rows.iter().any(|r| r.train_error.is_finite());

    match format {
        Format::Text => {
            if !quiet {
                eprintln!(
                    "# elapsed: {:.2}s | generations: {} | pareto members: {}{}",
                    elapsed,
                    outcome.generations,
                    outcome.archive.items.len(),
                    if outcome.converged_early {
                        " (stopped early)"
                    } else {
                        ""
                    }
                );
            }
            for r in &rows {
                println!(
                    "{:>3}  complexity={:<3} train_err={:<13} test_err={:<13}  {} = {}",
                    r.rank,
                    r.complexity,
                    fmt_err(r.train_error),
                    fmt_err(r.test_error),
                    target,
                    r.expression
                );
            }
        }
        Format::Json => {
            let split_json = format!(
                "{{\"mode\":{},\"rows\":{},\"train_rows\":{},\"test_rows\":{},\"folds\":[{}],\"gap\":{},\"groups_total\":{},\"groups_train\":{},\"groups_test\":{},\"notes\":[{}]}}",
                jstr(prepared.report.mode),
                prepared.report.rows_total,
                prepared.train.len(),
                prepared.test.len(),
                prepared
                    .fold_samples
                    .iter()
                    .map(|f| f.len().to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                prepared.report.gap,
                prepared.report.groups_total,
                prepared.report.groups_train,
                prepared.report.groups_test,
                notes.iter().map(|n| jstr(n)).collect::<Vec<_>>().join(",")
            );
            let pareto: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"rank\":{},\"complexity\":{},\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}],\"expression\":{}}}",
                        r.rank,
                        r.complexity,
                        jnum(r.train_error),
                        jnum(r.test_error),
                        r.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                        jstr(&r.expression)
                    )
                })
                .collect();
            println!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{},\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"pareto\":[{}]}}",
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                jstr(&target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                prepared.train.len(),
                prepared.test.len(),
                split_json,
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                pareto.join(",")
            );
        }
        Format::Ndjson => {
            for r in &rows {
                println!(
                    "{{\"event\":\"solution\",\"rank\":{},\"complexity\":{},\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}],\"expression\":{}}}",
                    r.rank,
                    r.complexity,
                    jnum(r.train_error),
                    jnum(r.test_error),
                    r.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                    jstr(&r.expression)
                );
            }
            println!(
                "{{\"event\":\"done\",\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"pareto_members\":{}}}",
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                outcome.archive.items.len()
            );
        }
    }

    if let Some(path) = out_path {
        let mut buf = String::from("rank\tcomplexity\ttrain_error\ttest_error\texpression\n");
        for r in &rows {
            buf.push_str(&format!(
                "{}\t{}\t{:.8}\t{:.8}\t{}\n",
                r.rank, r.complexity, r.train_error, r.test_error, r.expression
            ));
        }
        if let Err(e) = std::fs::write(&path, buf) {
            eprintln!("error writing {}: {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    }

    std::process::exit(if finite { EXIT_OK } else { EXIT_NO_SOLUTION });
}

fn fmt_err(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "inf".to_string()
    }
}
