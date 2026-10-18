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
  --vars <a,b,..>            input columns (default: all except the target)
  --test-frac <f>            held-out fraction (default 0.25)

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
        ("--vars", "csv", "all-but-target", "input column names"),
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

    if vars.is_empty() {
        vars = table
            .names
            .iter()
            .filter(|n| **n != target)
            .cloned()
            .collect();
    }
    if vars.is_empty() {
        eprintln!("error: no input variables");
        std::process::exit(EXIT_INPUT);
    }

    let mut rng = Mt::new(cfg.seed);
    let (train, test, means, stds) =
        match data::prepare(&table, &vars, &target, test_frac, &mut rng) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };

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
    if format == Format::Ndjson {
        println!(
            "{{\"event\":\"start\",\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{}}}",
            jstr(VERSION),
            cfg.seed,
            jstr(cfg.metric.name()),
            jstr(&target),
            vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
            train.len(),
            test.len()
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
            "# target: {} = f({}) | train rows: {} | test rows: {}",
            target,
            vars.join(","),
            train.len(),
            test.len()
        );
    }

    let outcome = search::run(&cfg, &train, &test);
    let elapsed = start.elapsed().as_secs_f64();

    let sorted = outcome.archive.sorted();
    let mut rows: Vec<Row> = sorted
        .iter()
        .zip(outcome.test_errors.iter())
        .enumerate()
        .map(|(k, (s, &te))| {
            let den = expr::simplify(&expr::denormalize(&expr::simplify(&s.tree), &means, &stds));
            Row {
                rank: k + 1,
                complexity: s.complexity,
                train_error: s.error,
                test_error: te,
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
            let pareto: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"rank\":{},\"complexity\":{},\"train_error\":{},\"test_error\":{},\"expression\":{}}}",
                        r.rank,
                        r.complexity,
                        jnum(r.train_error),
                        jnum(r.test_error),
                        jstr(&r.expression)
                    )
                })
                .collect();
            println!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"pareto\":[{}]}}",
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                jstr(&target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                train.len(),
                test.len(),
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                pareto.join(",")
            );
        }
        Format::Ndjson => {
            for r in &rows {
                println!(
                    "{{\"event\":\"solution\",\"rank\":{},\"complexity\":{},\"train_error\":{},\"test_error\":{},\"expression\":{}}}",
                    r.rank,
                    r.complexity,
                    jnum(r.train_error),
                    jnum(r.test_error),
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
