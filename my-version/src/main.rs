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

use rcsr::complexity::ComplexityModel;
use rcsr::expr::{self, Op};
use rcsr::fit;
use rcsr::metrics::{self, Metric};
use rcsr::rng::Mt;
use rcsr::{data, diagnostics, search, stability};
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

DATA PREPARATION (all opt-in; defaults reproduce a plain mean/std run):
  --fill <method>            what to do with a missing cell (empty, NA, NaN or
                             null). Default none: leave it, as before.
                             drop        drop the row
                             mean|median training-row column statistic
                             zero        literal 0
                             previous    last value above it in the same side
                             interpolate linear between the neighbours
  --outlier-iqr <k>          drop rows more than k interquartile ranges from the
                             column median (bounds from training rows only)
  --smooth <k>               k-wide centred moving average per column (edges
                             shrink; windows never cross the train/test split)
  --center <how>             subtract mean (default), median, min, max or none
  --scale <how>              divide by std (default), iqr, range or none
  --filter <expr>            keep only rows where the expression is > 0; may
                             use any numeric column name, e.g. \"x > 0.5 and y < 3\"
  --exclude <expr>           drop rows where the expression is > 0, before any
                             split or filter
  --train-select <expr>      rows where true go to training (overrides --split)
  --test-select <expr>       rows where true go to test (overrides --split);
                             a row matching both goes to training and is
                             counted, one matching neither is reported unused

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

HISTORY AND DELAYED VARIABLES (lags, for forecasting):
  --lags <k>                 add x_lag1..x_lagk as inputs per variable, computed
                             inside each series (group + time order + gaps)
  --history-frac <f>         keep the first fraction of each series as history:
                             those rows supply the past but are not scored
  --gap-break <f>            start a new series when the time step exceeds f x the
                             median step (so lags cannot jump across a gap)

OVERFITTING DIAGNOSTICS (always on):
  AICc, BIC                   fit vs. number of fitted constants
  Durbin-Watson               residual autocorrelation in row order (~2 is clean)
  Ljung-Box                   residual autocorrelation over several lags, with a
                              chi-square p-value
  Breusch-Pagan               residual variance vs. the fitted value, with a
                              chi-square p-value; also skewness/excess kurtosis
  null baselines              train/test error of the constant mean, an OLS
                              straight line and a 5-nearest-neighbour learner,
                              in the same metric as the search: a model must
                              beat them to earn its complexity
  overfit index               test_error / train_error
  constant errors             standard error and relative uncertainty of every
                              fitted constant, from the Gauss-Newton Jacobian
                              (J^T J); flags constants the data cannot pin down
  --cv <mode>                cross-validation: rerun the search on k folds and
                             report the mean and spread of the held-out errors.
                             k-fold  shuffled partition (i.i.d. data)
                             rolling forward chaining (time-ordered data)
                             purged  k-fold with a --gap embargo either side of
                                     each test block
  --cv-k <K>                 number of folds (default 5); --cv costs ~K searches
  --null-runs <n>            rerun the identical search on n shuffled targets
                             (y-randomisation control) and report the null error
                             distribution and a p-value; inferential, costs n
                             extra searches
  --bootstrap-error <n>      refit each solution's constants on n bootstrap
                             resamples and report the apparent, out-of-bag and
                             .632+ error estimates (Efron & Tibshirani 1997);
                             costs n constant refits per reported solution
  --stability <B>            bootstrap the training rows B times, re-run the
                             identical search and rank the structural forms and
                             input variables by how often they are selected;
                             costs B extra searches

MODE:
  --eval-expr <expr>         do not search: parse this formula, score it on the
                             split, print the result and exit. Write variables
                             by name in original units, e.g. \"2.5*x + sin(y)\";
                             operators as in --ops. `x0,x1,..` also work.

SEARCH:
  --metric <name>            error to minimise, lower is better (default absolute)
                             absolute  squared  correlation  median  maximum
                             signed  log_squashed  rank_correlation  interquartile
                             slope  auc  log_loss  hinge  aic  aicc  bic
                             (auc/log_loss/hinge treat the target as two classes)
  --generations <N>          (default 200)
  --islands <I>              (default 1)
  --pop <P>                  population per island (default 200)
  --max-complexity <K>       max configured complexity per expression (default 30)
  --op-complexity <spec>     per-operator costs, e.g. \"pow=2,sin=1.5,if=3\";
                             operators not listed keep the default cost of 1
  --complexity-weights <s>   terminal weights, e.g. \"var=1,const=1\"; keys
                             var, const and depth (default: 1, 1 and 0)
  --depth-weight <w>         cost per nesting level, i.e. edge below the root
                             (default 0); a single node has depth 0
  --seed <S>                 RNG seed, echoed in output (default 0)
  --ops <list>               comma/space-separated operators, e.g.
                             add,sub,mul,div,sin,exp,floor,min,max
                             conditions: less,equal,greater,and,or,xor,not
                             constraints: between,require,constrain
                             (full set: `rcsr schema` -> operators list)
  --fit-iters <N>            Levenberg–Marquardt iterations for constants (default 8)
  --no-fit                   disable constant fitting
  --no-dedup                 disable structural de-duplication
  --predictor-samples <N>    score and fit candidates on N training rows per
                             generation (default: all rows; N >= 2). The archive
                             and every reported error are always full-data
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
        ("--ops", "csv", "builtin-set", "operator names: primitives, comparisons, logical operators and constraints (see \"operators\")"),
        ("--eval-expr", "string", "none", "score this formula on the split instead of searching"),
        ("--fit-iters", "int", "8", "LM constant-fitting iterations"),
        ("--no-fit", "flag", "false", "disable constant fitting"),
        ("--no-dedup", "flag", "false", "disable structural de-duplication"),
        ("--predictor-samples", "int", "none", "training rows per generation for fitness and constant fitting (>=2; all rows when >= train size)"),
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

/// Parse a boolean row expression (`--filter`, `--exclude`, `--train-select`,
/// `--test-select`) against every numeric column of the table.
///
/// The expressions are parsed before lag columns are added, so they see the
/// columns as they appear in the file. A syntax error is a usage error and
/// keeps the parser's 1-based column number.
fn parse_filter(flag: &str, src: Option<&str>, table: &data::Table) -> Option<data::Filter> {
    let src = src?;
    let filter_vars: Vec<String> = (0..table.names.len())
        .filter(|&j| table.numeric[j])
        .map(|j| table.names[j].clone())
        .collect();
    match expr::parse(src, &filter_vars) {
        Ok(tree) => Some(data::Filter {
            tree,
            vars: filter_vars,
            expression: src.to_string(),
        }),
        Err(e) => {
            eprintln!("error: cannot parse {}: {}", flag, e);
            eprintln!("  expression: {}", src);
            std::process::exit(EXIT_USAGE);
        }
    }
}

struct Row {
    rank: usize,
    complexity: f64,
    /// Fitted constants: the free parameters of the model.
    k_constants: usize,
    train_error: f64,
    test_error: f64,
    /// Small-sample-corrected AIC (fit vs. constants), lower is better.
    aicc: f64,
    /// Bayesian information criterion, lower is better.
    bic: f64,
    /// Durbin–Watson residual autocorrelation, in row order (≈2 is clean).
    dw: f64,
    /// Ljung–Box, Breusch–Pagan and residual shape for this solution.
    resid_tests: diagnostics::ResidualTests,
    /// `.632+` bootstrap error estimate, when `--bootstrap-error` is set.
    bootstrap: Option<diagnostics::BootstrapError>,
    /// test_error / train_error: how much worse the model gets off-sample.
    overfit: f64,
    /// Per-block test errors (`--folds`); one entry when folds == 1.
    fold_errors: Vec<f64>,
    /// Standard errors of the fitted constants (issue #22); `None` when they
    /// are not defined (no constants, or no residual degrees of freedom).
    constant_errors: Option<fit::ConstantErrors>,
    /// Structural form of the (normalised) tree, with constants collapsed; the
    /// key stability selection counts.
    shape: String,
    /// Fraction of `--stability` resamples that selected this same form.
    stability: Option<f64>,
    expression: String,
}

/// Cross-validation summary (issue #19).
struct CvReport {
    mode: &'static str,
    k: usize,
    folds: Vec<CvFoldRow>,
}

struct CvFoldRow {
    train_error: f64,
    test_error: f64,
    complexity: f64,
    expression: String,
}

impl CvReport {
    fn mean_train(&self) -> f64 {
        mean_of(self.folds.iter().map(|r| r.train_error))
    }
    fn mean_test(&self) -> f64 {
        mean_of(self.folds.iter().map(|r| r.test_error))
    }
    /// Population standard deviation of the fold test errors.
    fn std_test(&self) -> f64 {
        let n = self.folds.len();
        if n == 0 {
            return 0.0;
        }
        let m = self.mean_test();
        (self.folds.iter().map(|r| (r.test_error - m).powi(2)).sum::<f64>() / n as f64).sqrt()
    }
}

fn mean_of(it: impl Iterator<Item = f64>) -> f64 {
    let v: Vec<f64> = it.collect();
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
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
    let mut lags = 0usize;
    let mut history_frac = 0.0f64;
    let mut gap_break = 0.0f64;
    let mut null_runs = 0usize;
    let mut bootstrap_resamples = 0usize;
    let mut eval_expr: Option<String> = None;
    let mut stability_runs = 0usize;
    let mut cv_mode: Option<data::CvMode> = None;
    let mut cv_k = 5usize;
    let mut fill = data::FillMethod::None;
    let mut outlier_iqr: Option<f64> = None;
    let mut smooth = 0usize;
    let mut center = data::Center::Mean;
    let mut scale = data::Scale::Std;
    let mut filter_src: Option<String> = None;
    let mut exclude_src: Option<String> = None;
    let mut train_select_src: Option<String> = None;
    let mut test_select_src: Option<String> = None;

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
            "--metric" => {
                let v = val!();
                match Metric::parse_opt(&v) {
                    Some(m) => cfg.metric = m,
                    None => {
                        eprintln!("error: unknown --metric '{}'", v);
                        eprintln!("known metrics: {}", metric_names());
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--generations" => cfg.generations = val!().parse().unwrap_or(cfg.generations),
            "--islands" => cfg.islands = val!().parse().unwrap_or(cfg.islands),
            "--pop" => cfg.pop = val!().parse().unwrap_or(cfg.pop),
            "--max-complexity" => cfg.max_complexity = val!().parse().unwrap_or(cfg.max_complexity),
            "--op-complexity" => {
                let v = val!();
                if let Err(e) = cfg.complexity_model.apply_op_costs(&v) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
            }
            "--complexity-weights" => {
                let v = val!();
                if let Err(e) = cfg.complexity_model.apply_weights(&v) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
            }
            "--depth-weight" => {
                let v = val!();
                match v.parse::<f64>() {
                    Ok(w) => {
                        if let Err(e) = cfg.complexity_model.set_depth_weight(w) {
                            eprintln!("error: --depth-weight: {}", e);
                            std::process::exit(EXIT_USAGE);
                        }
                    }
                    Err(_) => {
                        eprintln!(
                            "error: --depth-weight needs a finite number >= 0, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--seed" => cfg.seed = val!().parse().unwrap_or(cfg.seed),
            "--test-frac" => test_frac = val!().parse().unwrap_or(test_frac),
            "--fill" => {
                let v = val!();
                match data::FillMethod::parse(&v) {
                    Ok(m) => fill = m,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--outlier-iqr" => match val!().parse::<f64>() {
                Ok(k) if k > 0.0 => outlier_iqr = Some(k),
                Ok(_) => outlier_iqr = None,
                Err(_) => {
                    eprintln!("error: --outlier-iqr needs a positive number");
                    std::process::exit(EXIT_USAGE);
                }
            },
            "--smooth" => match val!().parse::<usize>() {
                Ok(k) => smooth = k,
                Err(_) => {
                    eprintln!("error: --smooth needs a non-negative integer window");
                    std::process::exit(EXIT_USAGE);
                }
            },
            "--center" => {
                let v = val!();
                match data::Center::parse(&v) {
                    Ok(c) => center = c,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--scale" => {
                let v = val!();
                match data::Scale::parse(&v) {
                    Ok(s) => scale = s,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--filter" => filter_src = Some(val!()),
            "--exclude" => exclude_src = Some(val!()),
            "--train-select" => train_select_src = Some(val!()),
            "--test-select" => test_select_src = Some(val!()),
            "--split" => split_arg = Some(val!()),
            "--time" => time_name = Some(val!()),
            "--group-by" => group_name = Some(val!()),
            "--gap" => gap = val!().parse().unwrap_or(0),
            "--folds" => folds = val!().parse().unwrap_or(1).max(1),
            "--lags" => lags = val!().parse().unwrap_or(0),
            "--history-frac" => history_frac = val!().parse().unwrap_or(0.0),
            "--gap-break" => gap_break = val!().parse().unwrap_or(0.0),
            "--null-runs" => null_runs = val!().parse().unwrap_or(0),
            "--bootstrap-error" => bootstrap_resamples = val!().parse().unwrap_or(0),
            "--ops" => cfg.ops = Op::parse_list(&val!()),
            "--eval-expr" => eval_expr = Some(val!()),
            "--stability" => stability_runs = val!().parse().unwrap_or(0),
            "--cv" => {
                let v = val!();
                match data::CvMode::parse(&v) {
                    Ok(m) => cv_mode = Some(m),
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--cv-k" => cv_k = val!().parse().unwrap_or(5).max(2),
            "--fit-iters" => cfg.fit_iters = val!().parse().unwrap_or(cfg.fit_iters),
            "--no-fit" => cfg.fit_iters = 0,
            "--no-dedup" => cfg.dedup = false,
            "--predictor-samples" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 2 => cfg.predictor_samples = Some(n),
                    _ => {
                        eprintln!(
                            "error: --predictor-samples needs an integer >= 2, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
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

    // Parse the row filter against every numeric column. A syntax error is a
    // usage error (exit 2) and keeps the parser's column number.
    let filter = parse_filter("--filter", filter_src.as_deref(), &table);
    // Row selection (issue #3).
    let exclude = parse_filter("--exclude", exclude_src.as_deref(), &table);
    let train_select = parse_filter("--train-select", train_select_src.as_deref(), &table);
    let test_select = parse_filter("--test-select", test_select_src.as_deref(), &table);

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

    // Time-delay embedding. New columns are only appended, so the indices in
    // `time_col`/`group_col` stay valid. The leading rows of each series become
    // history: they supply the past for later rows but are not scored.
    let mut lag_report = data::LagReport::default();
    let table = if lags > 0 || history_frac > 0.0 {
        let lag_spec = data::LagSpec {
            lags,
            history_frac,
            gap_break,
        };
        match data::add_lags(&table, &vars, &lag_spec, time_col, group_col) {
            Ok((t2, v2, rep)) => {
                lag_report = rep;
                vars = v2;
                t2
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    } else {
        table
    };
    notes.extend(lag_report.notes.iter().cloned());

    let spec = data::SplitSpec {
        mode,
        time_col,
        group_col,
        test_frac,
        gap,
        folds,
        train_select,
        test_select,
        exclude,
    };
    let prep = data::PrepSpec {
        fill,
        outlier_iqr,
        smooth,
        center,
        scale,
        filter,
    };
    let mut rng = Mt::new(cfg.seed);
    let prepared =
        match data::prepare_full_with(&table, &vars, &target, &spec, &mut rng, &prep) {
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
    if format == Format::Text {
        for line in prepared.prep.text_lines(outlier_iqr, smooth) {
            eprintln!("{}", line);
        }
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
        if prepared.prep.active() {
            let obj = prepared.prep.json();
            let inner = obj.trim_matches(|c| c == '{' || c == '}');
            println!("{{\"event\":\"preparation\",{}}}", inner);
        }
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

    if let Some(src) = eval_expr.as_deref() {
        run_eval(
            src,
            &vars,
            &target,
            &prepared,
            cfg.metric,
            &cfg.complexity_model,
            format,
            quiet,
            start,
        );
        std::process::exit(EXIT_OK);
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
    // A dedicated RNG stream so bootstrap resampling never perturbs the search
    // or the permutation control.
    let mut boot_rng = Mt::new(cfg.seed ^ 0xB007_57AA);
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
            let pred: Vec<f64> = prepared
                .train
                .x
                .iter()
                .map(|row| expr::eval(&s.tree, row))
                .collect();
            let resid: Vec<f64> = pred
                .iter()
                .zip(&prepared.train.y)
                .map(|(p, t)| p - t)
                .collect();
            let rss: f64 = resid.iter().map(|e| e * e).sum();
            let kconst = expr::count_constants(&s.tree);
            let (_, aicc, bic) =
                diagnostics::information_criteria(prepared.train.len(), rss, kconst);
            let dw = diagnostics::durbin_watson(&resid);
            let resid_tests = diagnostics::residual_tests(&resid, &pred);
            // The reported training error is always the full-data error, even
            // when the search scored candidates on `--predictor-samples` rows.
            let train_error = search::eval_error(&s.tree, &prepared.train, cfg.metric);
            let overfit = if train_error > 0.0 {
                test_error / train_error
            } else if test_error > 0.0 {
                f64::INFINITY
            } else {
                1.0
            };
            // `.632+` bootstrap error: refit constants on resamples of the
            // training rows and score the out-of-bag rows (issue #21).
            let bootstrap = if bootstrap_resamples > 0 {
                Some(diagnostics::bootstrap_error(
                    &s.tree,
                    &prepared.train,
                    cfg.metric,
                    bootstrap_resamples,
                    cfg.fit_iters,
                    &mut boot_rng,
                ))
            } else {
                None
            };
            Row {
                rank: k + 1,
                complexity: s.cost,
                k_constants: kconst,
                train_error,
                test_error,
                aicc,
                bic,
                dw,
                resid_tests,
                bootstrap,
                overfit,
                fold_errors,
                // Parameter uncertainty at the fitted solution (issue #22):
                // reuse the fitter's Jacobian on the training rows.
                constant_errors: fit::constant_errors(&s.tree, &prepared.train.x, &prepared.train.y),
                shape: stability::shape(&expr::simplify(&s.tree)),
                stability: None,
                expression: expr::to_string(&den, &vars),
            }
        })
        .collect();
    // Residual diagnostics for the best (lowest training-error) solution: the
    // one a reader would quote.
    let best_resid = rows
        .iter()
        .min_by(|a, b| {
            a.train_error
                .partial_cmp(&b.train_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|r| r.resid_tests);

    // Null baselines: the trivial models a real fit must beat (issue #23).
    let baselines = diagnostics::Baselines::fit(&prepared.train, &fold_samples, cfg.metric);

    // Stability selection (issue #20): bootstrap the training rows, re-run the
    // identical search, and count how often each structural form and each input
    // variable is selected. A form that wins most resamples is what the data
    // supports; one that wins once is a coincidence.
    let stab: Option<stability::Stability> = if stability_runs > 0 {
        let st = stability::run(&cfg, &prepared.train, &prepared.test, stability_runs, &vars);
        for r in rows.iter_mut() {
            r.stability = Some(st.frequency(&r.shape));
        }
        Some(st)
    } else {
        None
    };

    // Cross-validation (issue #19): rebuild the split as `k` folds, re-run the
    // search on each fold's own training rows, and score the model it returns on
    // the held-out block. Each fold is normalised on its training rows alone.
    let cv: Option<CvReport> = cv_mode.map(|mode| {
        let folds = match data::cv_folds(&table, &vars, &target, mode, cv_k, gap, cfg.seed) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let mut fold_rows = Vec::new();
        for f in &folds {
            let out = search::run(&cfg, &f.train, &f.test);
            // Pick by the engine's own criterion (lowest training error), then
            // score on rows that fold's search never saw.
            let best = out
                .archive
                .sorted()
                .into_iter()
                .min_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(s) = best {
                let test_error = search::eval_error(&s.tree, &f.test, cfg.metric);
                let train_error = search::eval_error(&s.tree, &f.train, cfg.metric);
                let den = expr::simplify(&expr::denormalize(
                    &expr::simplify(&s.tree),
                    &f.means,
                    &f.stds,
                ));
                fold_rows.push(CvFoldRow {
                    train_error,
                    test_error,
                    complexity: s.cost,
                    expression: expr::to_string(&den, &vars),
                });
            }
        }
        CvReport {
            mode: mode.name(),
            k: cv_k,
            folds: fold_rows,
        }
    });

    if top > 0 && rows.len() > top {
        rows.truncate(top);
    }

    // Permutation (y-randomisation) control: run the *same* search on a shuffled
    // target. If it reaches a comparable error, the real fit is what the search
    // can manufacture from noise, not evidence of a law.
    let mut null_control: Option<diagnostics::NullControl> = None;
    if null_runs > 0 {
        let observed = outcome
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        let saved_progress = cfg.progress.take();
        let mut best_null = Vec::with_capacity(null_runs);
        for r in 0..null_runs {
            let mut rng = Mt::new(cfg.seed ^ (0x51ED_2701u32.wrapping_mul(r as u32 + 1)));
            let mut y = prepared.train.y.clone();
            for j in (1..y.len()).rev() {
                let t = rng.usize_below(j + 1);
                y.swap(j, t);
            }
            let shuffled = data::Sample {
                x: prepared.train.x.clone(),
                y,
            };
            let out = search::run(&cfg, &shuffled, &shuffled);
            let b = out
                .archive
                .items
                .iter()
                .map(|s| s.error)
                .fold(f64::INFINITY, f64::min);
            if !quiet {
                eprintln!("# null run {}/{}: best train error {:.6}", r + 1, null_runs, b);
            }
            best_null.push(b);
        }
        cfg.progress = saved_progress;
        null_control = Some(diagnostics::NullControl::summarise(observed, best_null));
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
                    "{:>3}  complexity={:<3} train_err={:<13} test_err={:<13} aicc={:<12}  {} = {}",
                    r.rank,
                    r.complexity,
                    fmt_err(r.train_error),
                    fmt_err(r.test_error),
                    fmt_err(r.aicc),
                    target,
                    r.expression
                );
            }
            if let Some(r) = rows.iter().min_by(|a, b| {
                a.train_error
                    .partial_cmp(&b.train_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                eprintln!(
                    "# diagnostics: n={} k={} constants | overfit index (test/train)={:.3} | Durbin-Watson={:.3}{}",
                    prepared.train.len(),
                    r.k_constants,
                    r.overfit,
                    r.dw,
                    if mode == data::SplitMode::Temporal
                        || mode == data::SplitMode::TemporalGroup
                    {
                        ""
                    } else {
                        " (row order is arbitrary here)"
                    }
                );
            }
            if let Some(c) = &null_control {
                eprintln!(
                    "# null control ({} shuffled-target runs): best={:.6} median={:.6} p={:.3} — {}",
                    c.runs,
                    c.null_min,
                    c.null_median,
                    c.p_value,
                    c.verdict()
                );
            }
            if let Some(rt) = &best_resid {
                eprintln!(
                    "# residual tests (best solution): Ljung-Box h={} Q={:.3} p={:.3}{} | \
                     Breusch-Pagan LM={:.3} p={:.3}{} | skewness={:.3} excess kurtosis={:.3}",
                    rt.ljung_box.lags,
                    rt.ljung_box.statistic,
                    rt.ljung_box.p_value,
                    if rt.ljung_box.p_value < 0.05 {
                        " (autocorrelation left)"
                    } else {
                        ""
                    },
                    rt.breusch_pagan.statistic,
                    rt.breusch_pagan.p_value,
                    if rt.breusch_pagan.p_value < 0.05 {
                        " (variance changes with fitted)"
                    } else {
                        ""
                    },
                    rt.skewness,
                    rt.kurtosis
                );
            }
            for r in &rows {
                if let Some(b) = &r.bootstrap {
                    eprintln!(
                        "# bootstrap error (rank {}): apparent={:.6} out-of-bag={:.6} .632+={:.6} \
                         ({} resamples, {} used, weight={:.3})",
                        r.rank,
                        b.apparent,
                        b.oob,
                        b.estimate,
                        b.resamples,
                        b.used,
                        b.weight
                    );
                }
            }
            for r in &rows {
                if let Some(ce) = &r.constant_errors {
                    let parts: Vec<String> = ce
                        .values
                        .iter()
                        .zip(&ce.std_errors)
                        .zip(&ce.relative)
                        .map(|((v, se), rel)| format!("{:.6}±{:.2e} (rel {:.2e})", v, se, rel))
                        .collect();
                    eprintln!(
                        "# constant errors (rank {}): {} | dof={}{}",
                        r.rank,
                        parts.join(", "),
                        ce.dof,
                        if ce.ill_conditioned {
                            " | ill-conditioned: JᵀJ near-singular, constants not identified by the data"
                        } else {
                            ""
                        }
                    );
                }
            }
            eprintln!(
                "# null baselines ({}): mean train={:.6} test={:.6} | linear train={:.6} test={:.6} | {}-NN train={:.6} test={:.6}",
                cfg.metric.name(),
                baselines.mean.train_error,
                baselines.mean.test_error,
                baselines.linear.train_error,
                baselines.linear.test_error,
                baselines.knn_k,
                baselines.knn.train_error,
                baselines.knn.test_error
            );
            if let Some(cv) = &cv {
                eprintln!(
                    "# cross-validation ({}, k={}): mean test={:.6} +/- {:.6} | mean train={:.6}",
                    cv.mode,
                    cv.k,
                    cv.mean_test(),
                    cv.std_test(),
                    cv.mean_train()
                );
                for (i, r) in cv.folds.iter().enumerate() {
                    eprintln!(
                        "#   fold {}/{}: train={:.6} test={:.6} cx={}  y = {}",
                        i + 1,
                        cv.folds.len(),
                        r.train_error,
                        r.test_error,
                        r.complexity,
                        r.expression
                    );
                }
            }
            if let Some(st) = &stab {
                let top = st.forms.first();
                eprintln!(
                    "# stability ({} resamples, {} found a solution): modal form {:.0}%{}",
                    st.runs,
                    st.found,
                    top.map(|f| f.frequency * 100.0).unwrap_or(0.0),
                    top.map(|f| format!("  y = {}", f.example)).unwrap_or_default()
                );
                let names: Vec<String> = st
                    .variables
                    .iter()
                    .map(|(i, c)| {
                        format!(
                            "{} {}/{} ({:.0}%)",
                            vars.get(*i).cloned().unwrap_or_else(|| format!("x{}", i)),
                            c,
                            st.runs,
                            if st.runs > 0 {
                                *c as f64 * 100.0 / st.runs as f64
                            } else {
                                0.0
                            }
                        )
                    })
                    .collect();
                eprintln!("#   variable selection: {}", names.join(" | "));
                let forms: Vec<String> = st
                    .forms
                    .iter()
                    .take(5)
                    .map(|f| format!("{}/{} ({:.0}%) {}", f.count, st.runs, f.frequency * 100.0, f.example))
                    .collect();
                eprintln!("#   forms: {}", forms.join(" | "));
            }
        }
        Format::Json => {
            let split_json = format!(
                "{{\"mode\":{},\"rows\":{},\"train_rows\":{},\"test_rows\":{},\"folds\":[{}],\"gap\":{},\"excluded\":{},\"overlap\":{},\"unassigned\":{},\"groups_total\":{},\"groups_train\":{},\"groups_test\":{},\"lags\":{},\"series\":{},\"history_rows\":{},\"notes\":[{}]}}",
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
                prepared.report.excluded_rows,
                prepared.report.overlap_rows,
                prepared.report.unassigned_rows,
                prepared.report.groups_total,
                prepared.report.groups_train,
                prepared.report.groups_test,
                lag_report.lags,
                lag_report.series,
                lag_report.history_rows,
                notes.iter().map(|n| jstr(n)).collect::<Vec<_>>().join(",")
            );
            let pareto: Vec<String> = rows
                .iter()
                .map(|r| {
                    format!(
                        "{{\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{},\"stability\":{},\"expression\":{}}}",
                        r.rank,
                        r.complexity,
                        r.k_constants,
                        jnum(r.train_error),
                        jnum(r.test_error),
                        jnum(r.aicc),
                        jnum(r.bic),
                        jnum(r.dw),
                        jnum(r.overfit),
                        r.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                        bootstrap_json(&r.bootstrap),
                        constant_errors_json(&r.constant_errors),
                        r.stability.map(jnum).unwrap_or_else(|| "null".to_string()),
                        jstr(&r.expression)
                    )
                })
                .collect();
            let null_json = match &null_control {
                Some(c) => format!(
                    "{{\"runs\":{},\"best\":[{}],\"min\":{},\"median\":{},\"p_value\":{},\"verdict\":{}}}",
                    c.runs,
                    c.best.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                    jnum(c.null_min),
                    jnum(c.null_median),
                    jnum(c.p_value),
                    jstr(c.verdict())
                ),
                None => "null".to_string(),
            };
            let baselines_json = format!(
                "{{\"mean\":{{\"train_error\":{},\"test_error\":{}}},\"linear\":{{\"train_error\":{},\"test_error\":{}}},\"knn\":{{\"k\":{},\"train_error\":{},\"test_error\":{}}}}}",
                jnum(baselines.mean.train_error),
                jnum(baselines.mean.test_error),
                jnum(baselines.linear.train_error),
                jnum(baselines.linear.test_error),
                baselines.knn_k,
                jnum(baselines.knn.train_error),
                jnum(baselines.knn.test_error)
            );
            let resid_json = match &best_resid {
                Some(rt) => format!(
                    "{{\"n\":{},\"lags\":{},\"durbin_watson\":{},\"ljung_box\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"breusch_pagan\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"skewness\":{},\"kurtosis\":{}}}",
                    rt.n,
                    rt.lags,
                    jnum(rt.durbin_watson),
                    jnum(rt.ljung_box.statistic),
                    rt.ljung_box.lags,
                    jnum(rt.ljung_box.p_value),
                    jnum(rt.breusch_pagan.statistic),
                    rt.breusch_pagan.df,
                    jnum(rt.breusch_pagan.p_value),
                    jnum(rt.skewness),
                    jnum(rt.kurtosis)
                ),
                None => "null".to_string(),
            };
            let cv_json = match &cv {
                Some(c) => {
                    let folds: Vec<String> = c
                        .folds
                        .iter()
                        .map(|r| {
                            format!(
                                "{{\"train_error\":{},\"test_error\":{},\"complexity\":{},\"expression\":{}}}",
                                jnum(r.train_error),
                                jnum(r.test_error),
                                r.complexity,
                                jstr(&r.expression)
                            )
                        })
                        .collect();
                    format!(
                        "{{\"mode\":{},\"k\":{},\"mean_train_error\":{},\"mean_test_error\":{},\"std_test_error\":{},\"folds\":[{}]}}",
                        jstr(c.mode),
                        c.k,
                        jnum(c.mean_train()),
                        jnum(c.mean_test()),
                        jnum(c.std_test()),
                        folds.join(",")
                    )
                }
                None => "null".to_string(),
            };
            let stability_json = match &stab {
                Some(st) => {
                    let forms: Vec<String> = st
                        .forms
                        .iter()
                        .map(|f| {
                            format!(
                                "{{\"shape\":{},\"count\":{},\"frequency\":{},\"complexity\":{},\"example\":{}}}",
                                jstr(&f.shape),
                                f.count,
                                jnum(f.frequency),
                                f.complexity,
                                jstr(&f.example)
                            )
                        })
                        .collect();
                    let variables: Vec<String> = st
                        .variables
                        .iter()
                        .map(|(i, c)| {
                            format!(
                                "{{\"index\":{},\"name\":{},\"count\":{},\"frequency\":{}}}",
                                i,
                                jstr(vars.get(*i).map(|s| s.as_str()).unwrap_or("")),
                                c,
                                jnum(if st.runs > 0 {
                                    *c as f64 / st.runs as f64
                                } else {
                                    0.0
                                })
                            )
                        })
                        .collect();
                    format!(
                        "{{\"runs\":{},\"found\":{},\"forms\":[{}],\"variables\":[{}]}}",
                        st.runs,
                        st.found,
                        forms.join(","),
                        variables.join(",")
                    )
                }
                None => "null".to_string(),
            };
            println!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{},\"preparation\":{},\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"null_control\":{},\"baselines\":{},\"residual_tests\":{},\"cross_validation\":{},\"stability\":{},\"pareto\":[{}]}}",
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                jstr(&target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                prepared.train.len(),
                prepared.test.len(),
                split_json,
                prepared.prep.json(),
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                null_json,
                baselines_json,
                resid_json,
                cv_json,
                stability_json,
                pareto.join(",")
            );
        }
        Format::Ndjson => {
            for r in &rows {
                println!(
                    "{{\"event\":\"solution\",\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{},\"stability\":{},\"expression\":{}}}",
                    r.rank,
                    r.complexity,
                    r.k_constants,
                    jnum(r.train_error),
                    jnum(r.test_error),
                    jnum(r.aicc),
                    jnum(r.bic),
                    jnum(r.dw),
                    jnum(r.overfit),
                    r.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                    bootstrap_json(&r.bootstrap),
                    constant_errors_json(&r.constant_errors),
                    r.stability.map(jnum).unwrap_or_else(|| "null".to_string()),
                    jstr(&r.expression)
                );
            }
            if let Some(c) = &null_control {
                println!(
                    "{{\"event\":\"null_control\",\"runs\":{},\"min\":{},\"median\":{},\"p_value\":{},\"verdict\":{}}}",
                    c.runs,
                    jnum(c.null_min),
                    jnum(c.null_median),
                    jnum(c.p_value),
                    jstr(c.verdict())
                );
            }
            println!(
                "{{\"event\":\"baselines\",\"metric\":{},\"mean\":{{\"train_error\":{},\"test_error\":{}}},\"linear\":{{\"train_error\":{},\"test_error\":{}}},\"knn\":{{\"k\":{},\"train_error\":{},\"test_error\":{}}}}}",
                jstr(cfg.metric.name()),
                jnum(baselines.mean.train_error),
                jnum(baselines.mean.test_error),
                jnum(baselines.linear.train_error),
                jnum(baselines.linear.test_error),
                baselines.knn_k,
                jnum(baselines.knn.train_error),
                jnum(baselines.knn.test_error)
            );
            if let Some(cv) = &cv {
                println!(
                    "{{\"event\":\"cross_validation\",\"mode\":{},\"k\":{},\"mean_train_error\":{},\"mean_test_error\":{},\"std_test_error\":{}}}",
                    jstr(cv.mode),
                    cv.k,
                    jnum(cv.mean_train()),
                    jnum(cv.mean_test()),
                    jnum(cv.std_test())
                );
            }
            if let Some(rt) = &best_resid {
                println!(
                    "{{\"event\":\"residual_tests\",\"n\":{},\"lags\":{},\"ljung_box\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"breusch_pagan\":{{\"statistic\":{},\"df\":{},\"p_value\":{}}},\"skewness\":{},\"kurtosis\":{}}}",
                    rt.n,
                    rt.lags,
                    jnum(rt.ljung_box.statistic),
                    rt.ljung_box.lags,
                    jnum(rt.ljung_box.p_value),
                    jnum(rt.breusch_pagan.statistic),
                    rt.breusch_pagan.df,
                    jnum(rt.breusch_pagan.p_value),
                    jnum(rt.skewness),
                    jnum(rt.kurtosis)
                );
            }
            if let Some(st) = &stab {
                let forms: Vec<String> = st
                    .forms
                    .iter()
                    .map(|f| {
                        format!(
                            "{{\"shape\":{},\"count\":{},\"frequency\":{},\"example\":{}}}",
                            jstr(&f.shape),
                            f.count,
                            jnum(f.frequency),
                            jstr(&f.example)
                        )
                    })
                    .collect();
                let variables: Vec<String> = st
                    .variables
                    .iter()
                    .map(|(i, c)| {
                        format!(
                            "{{\"index\":{},\"name\":{},\"count\":{},\"frequency\":{}}}",
                            i,
                            jstr(vars.get(*i).map(|s| s.as_str()).unwrap_or("")),
                            c,
                            jnum(if st.runs > 0 {
                                *c as f64 / st.runs as f64
                            } else {
                                0.0
                            })
                        )
                    })
                    .collect();
                println!(
                    "{{\"event\":\"stability\",\"runs\":{},\"found\":{},\"forms\":[{}],\"variables\":[{}]}}",
                    st.runs,
                    st.found,
                    forms.join(","),
                    variables.join(",")
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

/// `--eval-expr`: score a user-written formula instead of searching.
///
/// The formula is written in **original units** (the same form a search run
/// prints as `y = ...`). Inputs are un-normalised per row before evaluation,
/// and the target is compared un-normalised too, so re-scoring a printed
/// right-hand side reproduces the run's numbers.
fn run_eval(
    src: &str,
    vars: &[String],
    target: &str,
    prepared: &data::Prepared,
    metric: Metric,
    model: &ComplexityModel,
    format: Format,
    quiet: bool,
    started: std::time::Instant,
) {
    let tree = match expr::parse(src, vars) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot parse --eval-expr: {}", e);
            eprintln!("  expression: {}", src);
            std::process::exit(EXIT_USAGE);
        }
    };
    let k = expr::count_constants(&tree);
    let complexity = model.complexity(&tree);
    let shown = expr::to_string(&tree, vars);

    let raw_row = |row: &[f64]| -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, v)| {
                v * prepared.stds.get(j).copied().unwrap_or(1.0)
                    + prepared.means.get(j).copied().unwrap_or(0.0)
            })
            .collect()
    };
    let score = |s: &data::Sample| -> f64 {
        let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(&tree, &raw_row(r))).collect();
        metrics::score_with_constants(&pred, &s.y, metric, k)
    };
    let train_error = score(&prepared.train);
    let fold_errors: Vec<f64> = prepared.fold_samples.iter().map(score).collect();
    let test_error = if fold_errors.is_empty() {
        f64::INFINITY
    } else {
        fold_errors.iter().sum::<f64>() / fold_errors.len() as f64
    };

    match format {
        Format::Json => {
            let folds: Vec<String> = fold_errors.iter().map(|e| jnum(*e)).collect();
            println!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"eval-expr\",\"metric\":{},\"target\":{},\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"preparation\":{},\"eval\":{{\"expression\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}]}}}}",
                jstr(VERSION),
                jstr(metric.name()),
                jstr(target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                prepared.train.len(),
                prepared.test.len(),
                prepared.prep.json(),
                jstr(&shown),
                complexity,
                k,
                jnum(train_error),
                jnum(test_error),
                folds.join(",")
            );
        }
        Format::Ndjson => println!(
            "{{\"event\":\"eval\",\"metric\":{},\"expression\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{}}}",
            jstr(metric.name()),
            jstr(&shown),
            complexity,
            k,
            jnum(train_error),
            jnum(test_error)
        ),
        Format::Text => {
            if !quiet {
                eprintln!(
                    "# eval-expr | metric={} | train rows: {} | test rows: {} | {:.3}s",
                    metric.name(),
                    prepared.train.len(),
                    prepared.test.len(),
                    started.elapsed().as_secs_f64()
                );
            }
            println!(
                "  1  complexity={}   k_constants={}   train_err={}      test_err={}      y = {}",
                complexity,
                k,
                fmt_err(train_error),
                fmt_err(test_error),
                shown
            );
        }
    }
}

fn fmt_err(v: f64) -> String {
    if v.is_finite() {
        format!("{:.6}", v)
    } else {
        "inf".to_string()
    }
}

/// JSON for one `.632+` bootstrap estimate, or `null` when it is off.
fn bootstrap_json(b: &Option<diagnostics::BootstrapError>) -> String {
    match b {
        Some(b) => format!(
            "{{\"resamples\":{},\"used\":{},\"apparent\":{},\"oob\":{},\"no_information\":{},\"relative_overfitting\":{},\"weight\":{},\"estimate\":{}}}",
            b.resamples,
            b.used,
            jnum(b.apparent),
            jnum(b.oob),
            jnum(b.no_information),
            jnum(b.relative_overfitting),
            jnum(b.weight),
            jnum(b.estimate)
        ),
        None => "null".to_string(),
    }
}

/// JSON for one solution's fitted-constant standard errors, or `null` when
/// they are not defined (issue #22).
fn constant_errors_json(ce: &Option<fit::ConstantErrors>) -> String {
    match ce {
        Some(ce) => {
            let items: Vec<String> = ce
                .values
                .iter()
                .zip(&ce.std_errors)
                .zip(&ce.relative)
                .map(|((v, se), rel)| {
                    format!(
                        "{{\"value\":{},\"std_error\":{},\"relative\":{}}}",
                        jnum(*v),
                        jnum(*se),
                        jnum(*rel)
                    )
                })
                .collect();
            format!(
                "{{\"dof\":{},\"ill_conditioned\":{},\"constants\":[{}]}}",
                ce.dof,
                ce.ill_conditioned,
                items.join(",")
            )
        }
        None => "null".to_string(),
    }
}

/// Comma-separated list of every metric name, for error messages.
fn metric_names() -> String {
    Metric::all()
        .iter()
        .map(|m| m.name())
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `--help` and `schema` are hand-written; this keeps them in step with the
    /// metric enum so a new metric cannot ship undocumented.
    #[test]
    fn help_and_schema_list_every_metric() {
        let help = usage();
        let schema_text = schema();
        for m in Metric::all() {
            assert!(
                help.contains(m.name()),
                "usage() is missing metric '{}'",
                m.name()
            );
            assert!(
                schema_text.contains(m.name()),
                "schema() is missing metric '{}'",
                m.name()
            );
        }
    }

    /// The `.632+` bootstrap flag must stay documented in `--help` and `schema`.
    #[test]
    fn help_and_schema_document_bootstrap_error() {
        assert!(
            usage().contains("--bootstrap-error"),
            "usage() is missing --bootstrap-error"
        );
        assert!(
            schema().contains("--bootstrap-error"),
            "schema() is missing --bootstrap-error"
        );
    }

    /// Data preparation stayed opt-in, but every flag must be documented.
    #[test]
    fn help_and_schema_document_data_preparation() {
        for flag in [
            "--fill",
            "--outlier-iqr",
            "--smooth",
            "--center",
            "--scale",
            "--filter",
        ] {
            assert!(usage().contains(flag), "usage() is missing {flag}");
            assert!(schema().contains(flag), "schema() is missing {flag}");
        }
        // The default for missing values is the old behaviour: leave them.
        assert!(usage().contains("Default none"));
    }

    /// The conditional primitives must be discoverable from `schema`, and the
    /// `--ops` help must point at them.
    #[test]
    fn help_and_schema_list_conditional_operators() {
        let schema_text = schema();
        for name in [
            "equal",
            "less",
            "less_or_equal",
            "greater",
            "greater_or_equal",
            "and",
            "or",
            "xor",
            "not",
            "require",
            "between",
            "constrain",
        ] {
            assert!(
                schema_text.contains(name),
                "schema() is missing operator '{name}'"
            );
        }
        assert!(usage().contains("between"), "usage() is missing 'between'");
        assert!(usage().contains("constrain"), "usage() is missing 'constrain'");
    }
}
