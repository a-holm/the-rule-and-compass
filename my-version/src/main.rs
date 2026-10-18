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

use rcsr::checkpoint::Checkpoint;
use rcsr::complexity::ComplexityModel;
use rcsr::expr::{self, Op};
use rcsr::fit;
use rcsr::metrics::{self, Metric};
use rcsr::rng::Mt;
use rcsr::schema;
use rcsr::units::{Unit, Units};
use rcsr::{data, diagnostics, search, stability};
use std::collections::BTreeMap;
use std::fmt::Write as _;
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

/// The optional Markdown report (issue #29). A single report per run; a later
/// `--report` replaces an earlier one, as with the other single-valued flags.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReportKind {
    Summary,
    Sensitivity,
    Benchmark,
}

impl ReportKind {
    fn parse(s: &str) -> Option<Self> {
        match s {
            "summary" => Some(ReportKind::Summary),
            "sensitivity" => Some(ReportKind::Sensitivity),
            "benchmark" => Some(ReportKind::Benchmark),
            _ => None,
        }
    }
}

/// Parse one `--units` value: `name=unit[,name=unit...]` (issue #25).
///
/// The unit expression is parsed immediately; names are resolved against the
/// prepared input list once `--vars`, `--dtype` and `--lags` are known.
fn parse_unit_assignments(src: &str) -> Result<Vec<(String, Unit)>, String> {
    let mut out = Vec::new();
    for item in src.split(',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let (name, expr) = item
            .split_once('=')
            .ok_or_else(|| format!("expected name=unit, got '{}'", item))?;
        let name = name.trim();
        if name.is_empty() {
            return Err(format!("missing name in '{}'", item));
        }
        out.push((name.to_string(), Unit::parse(expr.trim())?));
    }
    if out.is_empty() {
        return Err("no units given".to_string());
    }
    Ok(out)
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
  --dtype <col>=<kind>,..    per-variable datatype: continuous (default),
                             boolean (0/1, never standardised) or categorical
                             (expanded to one-hot columns named col=value).
                             Example: --dtype flag=boolean,symbol=categorical
  --max-levels <N>           maximum distinct values in a categorical column
                             before one-hot expansion (default 16)
  --test-frac <f>            held-out fraction (default 0.25)

ROW WEIGHTS:
  --weight <col>             per-row weight column: finite, >= 0, at least one
                             positive. A weight of 2 means \"this row twice\":
                             absolute, squared, correlation, signed and
                             log_squashed become weighted means, and aic/aicc/bic
                             use the weighted RSS. Refused for metrics with no
                             weighted definition (median, maximum, interquartile,
                             slope, rank_correlation, auc, log_loss, hinge).

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
  constant dimensions         with --units, the apparent dimension each fitted
                              constant needs (a value:dimension plausibility check)
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
  --seed-expr <expr>         start from a known formula (repeatable); variables
                             are the --vars names, written in original units,
                             simplified, constants fitted, then inserted into
                             every island and the archive. A seed is an elite:
                             it is kept even above --max-complexity
  --seed-file <path>         one seed expression per line; '#' comments and
                             blank lines are skipped (combine with --seed-expr)
  --ops <list>               comma/space-separated operators, e.g.
                             add,sub,mul,div,sin,exp,floor,min,max
                             conditions: less,equal,greater,and,or,xor,not
                             constraints: between,require,constrain
                             meta: complexity,variables,constants,contains,
                             contains_op (describe the argument's structure)
                             (full set: `rcsr schema` -> operators list)
  --units <name=dim,..>      declare dimensions of inputs and the target, e.g.
                             \"x=m,t=s,y=m/s\"; m,s,kg,A,K,mol,cd are SI names,
                             any other identifier is its own base dimension
                             (price=USD). Model dimensions, not unit symbols
                             (volume=m^3, not litre). *, /, ^integer, (...);
                             repeatable; an undeclared name is dimensionless.
                             Alone it reports constant dimensions and never
                             rejects a candidate
  --require-dimensions       reject a candidate whose dimensions cannot be made
                             consistent with the target's (infinite error);
                             requires --units (usage error without it). Without
                             this flag, units only report and never reject
  --fit-iters <N>            Levenberg–Marquardt iterations for constants (default 8)
  --no-fit                   disable constant fitting
  --no-dedup                 disable structural de-duplication
  --predictor-samples <N>    score and fit candidates on N training rows per
                             generation (default: all rows; N >= 2). The archive
                             and every reported error are always full-data
  --patience <N>             stop after N generations without improvement (default 40)
  --time-budget <seconds>    wall-clock limit (default: none)

CHECKPOINTS:
  --save <file>              write a checkpoint on exit and every --save-every
                             generations; saving does not change the search
  --save-every <N>           generation interval for --save (default 25)
  --resume <file>            continue from a checkpoint. The saved search
                             settings win; --generations may only be raised.
                             Training and test data must match the saved
                             fingerprint; a mismatch is refused (exit 3).

OUTPUT:
  --format text|json|ndjson  (default text)
  --json                     shorthand for --format json
  --ndjson                   shorthand for --format ndjson
  --progress                 emit per-generation progress (stderr, or stdout in ndjson)
  --top <N>                  report at most N Pareto rows (default: all)
  --quiet                    suppress the human-readable header
  --out <file>               also write the Pareto front as TSV

REPORTS (Markdown, issue #29):
  --report <type>            emit a Markdown report after the run. One type per
                             run; a later --report replaces an earlier one:
                             summary      data, chosen law, constants + standard
                                          errors, diagnostics, baselines, the
                                          whole Pareto front and a plain-language
                                          trust verdict
                             sensitivity  analytic d f / d x for every input,
                                          ranked by mean |d f/d x| sigma_x/sigma_f
                             benchmark    repeat the search N times at the same
                                          budget and report per-complexity
                                          median/IQR error and wall-clock time
  --report-out <file>        write the report to this path. Without it, text
                             prints the report to stdout; --json/--ndjson send
                             it to stderr so the machine-readable stream stays
                             valid JSON/NDJSON. Use --report-out for a clean file.
  --report-repeats <N>       searches for --report benchmark (default 5); seeds
                             are derived from --seed. Each repeat honours
                             --time-budget, --generations and --pop.
  -h, --help                 show this help

EXIT CODES:
  0 success   2 usage error   3 input/data error   4 no finite solution found"
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

/// The `variables_detail` array shared by `--json`, `--ndjson` and
/// `--eval-expr`: one object per search input with its datatype. A one-hot
/// indicator also reports its source column.
fn variables_detail_json(types: &[data::VariableDType]) -> String {
    types
        .iter()
        .map(|v| {
            format!(
                "{{\"name\":{},\"dtype\":{},\"source\":{}}}",
                jstr(&v.name),
                jstr(v.dtype.name()),
                v.source
                    .as_ref()
                    .map(|s| jstr(s))
                    .unwrap_or_else(|| "null".to_string())
            )
        })
        .collect::<Vec<_>>()
        .join(",")
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

#[derive(Clone)]
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
    /// #25: constant values and their apparent dimensions, in tree order.
    /// Empty when `--units` is not active; `None` means undetermined.
    constant_values: Vec<f64>,
    constant_units: Vec<Option<Unit>>,
    /// #25: `None` when units are inactive. `Some(false)` means the archived
    /// candidate would be rejected if `--require-dimensions` were on.
    dimensionally_consistent: Option<bool>,
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
    let command = format!("rcsr {}", args.join(" "));
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!("{}", usage());
        std::process::exit(EXIT_OK);
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!("rcsr {}", VERSION);
        std::process::exit(EXIT_OK);
    }
    if args.first().map(|s| s.as_str()) == Some("schema") {
        print!("{}", schema::json());
        std::process::exit(EXIT_OK);
    }

    let mut cfg = search::Config::default();
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut dtype_src: Option<String> = None;
    let mut max_levels = 16usize;
    let mut test_frac = 0.25;
    let mut out_path: Option<String> = None;
    let mut format = Format::Text;
    let mut progress = false;
    let mut quiet = false;
    let mut top = 0usize;
    let mut report: Option<ReportKind> = None;
    let mut report_out: Option<String> = None;
    let mut report_repeats = 5usize;
    let mut report_repeats_set = false;
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
    let mut seed_exprs: Vec<String> = Vec::new();
    let mut seed_files: Vec<String> = Vec::new();
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
    let mut weight_name: Option<String> = None;
    let mut save_path: Option<String> = None;
    let mut save_every = 25usize;
    let mut save_every_set = false;
    let mut resume_path: Option<String> = None;

    // Which search flags were written on the command line. Used to refuse a
    // resume whose explicit override contradicts the saved configuration.
    let mut generations_set = false;
    let mut islands_set = false;
    let mut pop_set = false;
    let mut max_complexity_set = false;
    let mut metric_set = false;
    let mut seed_set = false;
    let mut ops_set = false;
    let mut fit_iters_set = false;
    let mut dedup_set = false;
    let mut patience_set = false;
    let mut time_budget_set = false;
    let mut predictor_samples_set = false;
    // #25: `--units` is resolved against the prepared input list later; the
    // expressions themselves are parsed here so a typo is a usage error.
    let mut unit_assignments: Vec<(String, Unit)> = Vec::new();
    let mut require_dimensions = false;
    let mut op_complexity_spec: Option<String> = None;
    let mut complexity_weights_spec: Option<String> = None;
    let mut depth_weight_spec: Option<f64> = None;

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
            "--dtype" => dtype_src = Some(val!()),
            "--max-levels" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => max_levels = n,
                    _ => {
                        eprintln!("error: --max-levels needs an integer >= 1, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--metric" => {
                let v = val!();
                match Metric::parse_opt(&v) {
                    Some(m) => {
                        cfg.metric = m;
                        metric_set = true;
                    }
                    None => {
                        eprintln!("error: unknown --metric '{}'", v);
                        eprintln!("known metrics: {}", metric_names());
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--generations" => {
                cfg.generations = val!().parse().unwrap_or(cfg.generations);
                generations_set = true;
            }
            "--islands" => {
                cfg.islands = val!().parse().unwrap_or(cfg.islands);
                islands_set = true;
            }
            "--pop" => {
                cfg.pop = val!().parse().unwrap_or(cfg.pop);
                pop_set = true;
            }
            "--max-complexity" => {
                cfg.max_complexity = val!().parse().unwrap_or(cfg.max_complexity);
                max_complexity_set = true;
            }
            "--op-complexity" => {
                let v = val!();
                if let Err(e) = cfg.complexity_model.apply_op_costs(&v) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
                op_complexity_spec = Some(v);
            }
            "--complexity-weights" => {
                let v = val!();
                if let Err(e) = cfg.complexity_model.apply_weights(&v) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
                complexity_weights_spec = Some(v);
            }
            "--depth-weight" => {
                let v = val!();
                match v.parse::<f64>() {
                    Ok(w) => {
                        if let Err(e) = cfg.complexity_model.set_depth_weight(w) {
                            eprintln!("error: --depth-weight: {}", e);
                            std::process::exit(EXIT_USAGE);
                        }
                        depth_weight_spec = Some(w);
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
            "--seed" => {
                cfg.seed = val!().parse().unwrap_or(cfg.seed);
                seed_set = true;
            }
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
            "--weight" => weight_name = Some(val!()),
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
            "--ops" => {
                match Op::parse_list_checked(&val!()) {
                    Ok(ops) => cfg.ops = ops,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                ops_set = true;
            }
            "--eval-expr" => eval_expr = Some(val!()),
            "--seed-expr" => seed_exprs.push(val!()),
            "--seed-file" => seed_files.push(val!()),
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
            "--fit-iters" => {
                cfg.fit_iters = val!().parse().unwrap_or(cfg.fit_iters);
                fit_iters_set = true;
            }
            "--no-fit" => {
                cfg.fit_iters = 0;
                fit_iters_set = true;
            }
            "--no-dedup" => {
                cfg.dedup = false;
                dedup_set = true;
            }
            "--predictor-samples" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 2 => {
                        cfg.predictor_samples = Some(n);
                        predictor_samples_set = true;
                    }
                    _ => {
                        eprintln!(
                            "error: --predictor-samples needs an integer >= 2, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--units" => {
                match parse_unit_assignments(&val!()) {
                    Ok(mut entries) => unit_assignments.append(&mut entries),
                    Err(e) => {
                        eprintln!("error: --units: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--require-dimensions" => require_dimensions = true,
            "--patience" => {
                cfg.patience = val!().parse().unwrap_or(cfg.patience);
                patience_set = true;
            }
            "--time-budget" => {
                cfg.time_budget = val!().parse().ok();
                time_budget_set = true;
            }
            "--save" => save_path = Some(val!()),
            "--save-every" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => {
                        save_every = n;
                        save_every_set = true;
                    }
                    _ => {
                        eprintln!("error: --save-every needs an integer >= 1, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--resume" => resume_path = Some(val!()),
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
            "--report" => {
                let v = val!();
                match ReportKind::parse(&v) {
                    Some(k) => report = Some(k),
                    None => {
                        eprintln!(
                            "error: unknown --report '{}' (expected summary, sensitivity or benchmark)",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--report-out" => report_out = Some(val!()),
            "--report-repeats" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => {
                        report_repeats = n;
                        report_repeats_set = true;
                    }
                    _ => {
                        eprintln!(
                            "error: --report-repeats needs an integer >= 1, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            other => {
                eprintln!("error: unknown argument '{}'\nrun 'rcsr --help'", other);
                std::process::exit(EXIT_USAGE);
            }
        }
        i += 1;
    }

    if report_out.is_some() && report.is_none() {
        eprintln!("error: --report-out requires --report summary|sensitivity|benchmark");
        std::process::exit(EXIT_USAGE);
    }
    if report_repeats_set && report != Some(ReportKind::Benchmark) {
        eprintln!("error: --report-repeats only applies to --report benchmark");
        std::process::exit(EXIT_USAGE);
    }
    if report.is_some() && eval_expr.is_some() {
        eprintln!("error: --report cannot be combined with --eval-expr");
        std::process::exit(EXIT_USAGE);
    }
    // #25: the filter needs a declaration, unless a checkpoint saved one.
    if require_dimensions && unit_assignments.is_empty() && resume_path.is_none() {
        eprintln!("error: --require-dimensions needs --units");
        std::process::exit(EXIT_USAGE);
    }
    if save_every_set && save_path.is_none() {
        eprintln!("error: --save-every requires --save <file>");
        std::process::exit(EXIT_USAGE);
    }
    if eval_expr.is_some() && (save_path.is_some() || resume_path.is_some()) {
        eprintln!("error: --save and --resume cannot be combined with --eval-expr");
        std::process::exit(EXIT_USAGE);
    }
    if eval_expr.is_some() && (!seed_exprs.is_empty() || !seed_files.is_empty()) {
        eprintln!("error: --seed-expr and --seed-file cannot be combined with --eval-expr");
        std::process::exit(EXIT_USAGE);
    }
    if resume_path.is_some() && (!seed_exprs.is_empty() || !seed_files.is_empty()) {
        eprintln!(
            "error: --seed-expr and --seed-file cannot be combined with --resume \
             (seeds are inserted at the start of a fresh run; the checkpoint already \
             contains them)"
        );
        std::process::exit(EXIT_USAGE);
    }

    // Seed files are plain text: one expression per line, with `#` comment
    // lines and blank lines skipped. An unreadable file is an input error, not
    // a usage error, because the command line itself is well formed.
    let mut seed_lines: Vec<(String, String)> = Vec::new();
    for path in &seed_files {
        let text = match std::fs::read_to_string(path) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: --seed-file '{}': {}", path, e);
                std::process::exit(EXIT_INPUT);
            }
        };
        for (line_no, line) in text.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            seed_lines.push((
                format!("--seed-file {} line {}", path, line_no + 1),
                trimmed.to_string(),
            ));
        }
    }

    // A resume loads the saved search settings *before* the data is prepared:
    // the saved seed decides the split, and the saved settings are the base
    // that explicit flags are checked against.
    let mut resume_checkpoint: Option<Checkpoint> = None;
    let mut resumed_generation: Option<usize> = None;
    if let Some(path) = &resume_path {
        let cp = match Checkpoint::load(path) {
            Ok(cp) => cp,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let saved = cp.config.clone();
        let mut conflicts: Vec<String> = Vec::new();
        if generations_set && cfg.generations < saved.generations {
            conflicts.push(format!(
                "--generations {} would shorten the saved budget {}",
                cfg.generations, saved.generations
            ));
        }
        if islands_set && cfg.islands != saved.islands {
            conflicts.push(format!("--islands {} (saved {})", cfg.islands, saved.islands));
        }
        if pop_set && cfg.pop != saved.pop {
            conflicts.push(format!("--pop {} (saved {})", cfg.pop, saved.pop));
        }
        if max_complexity_set && cfg.max_complexity != saved.max_complexity {
            conflicts.push(format!(
                "--max-complexity {} (saved {})",
                cfg.max_complexity, saved.max_complexity
            ));
        }
        if metric_set && cfg.metric != saved.metric {
            conflicts.push(format!(
                "--metric {} (saved {})",
                cfg.metric.name(),
                saved.metric.name()
            ));
        }
        if seed_set && cfg.seed != saved.seed {
            conflicts.push(format!("--seed {} (saved {})", cfg.seed, saved.seed));
        }
        if ops_set && cfg.ops != saved.ops {
            conflicts.push("--ops differs from the saved operator list".to_string());
        }
        if fit_iters_set && cfg.fit_iters != saved.fit_iters {
            conflicts.push(format!(
                "--fit-iters {} (saved {})",
                cfg.fit_iters, saved.fit_iters
            ));
        }
        if dedup_set && cfg.dedup != saved.dedup {
            conflicts.push(format!("--no-dedup (saved dedup is {})", saved.dedup));
        }
        if patience_set && cfg.patience != saved.patience {
            conflicts.push(format!(
                "--patience {} (saved {})",
                cfg.patience, saved.patience
            ));
        }
        if time_budget_set && cfg.time_budget != saved.time_budget {
            conflicts.push(format!(
                "--time-budget {:?} (saved {:?})",
                cfg.time_budget, saved.time_budget
            ));
        }
        if predictor_samples_set && cfg.predictor_samples != saved.predictor_samples {
            conflicts.push(format!(
                "--predictor-samples {:?} (saved {:?})",
                cfg.predictor_samples, saved.predictor_samples
            ));
        }
        if let Some(spec) = &op_complexity_spec {
            let mut trial = saved.clone();
            if trial.complexity_model.apply_op_costs(spec).is_ok() && trial.complexity_model != saved.complexity_model
            {
                conflicts.push(format!(
                    "--op-complexity '{}' differs from the saved complexity model",
                    spec
                ));
            }
        }
        if let Some(spec) = &complexity_weights_spec {
            let mut trial = saved.clone();
            if trial.complexity_model.apply_weights(spec).is_ok() && trial.complexity_model != saved.complexity_model
            {
                conflicts.push(format!(
                    "--complexity-weights '{}' differs from the saved weights",
                    spec
                ));
            }
        }
        if let Some(w) = depth_weight_spec {
            let mut trial = saved.clone();
            if trial.complexity_model.set_depth_weight(w).is_ok() && trial.complexity_model != saved.complexity_model
            {
                conflicts.push(format!(
                    "--depth-weight {} differs from the saved depth weight",
                    w
                ));
            }
        }

        if !conflicts.is_empty() {
            eprintln!(
                "error: --resume '{}' conflicts with the saved search settings:",
                path
            );
            for c in &conflicts {
                eprintln!("  - {}", c);
            }
            eprintln!("  the saved run wins; omit the flag or pass the saved value");
            std::process::exit(EXIT_USAGE);
        }

        // The generation budget is the one setting that may be *raised* on
        // resume: all earlier generations are unaffected and a longer run is
        // the same trajectory plus more generations.
        let mut effective = saved;
        if generations_set {
            effective.generations = cfg.generations;
        }
        cfg = effective;
        resumed_generation = Some(cp.generations);
        resume_checkpoint = Some(cp);
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

    // Weights are either applied or refused: never silently dropped. This runs
    // after the table is read so a piped input is consumed before a usage error.
    if weight_name.is_some() && !metrics::weights_supported(cfg.metric) {
        eprintln!(
            "error: --weight is not supported by the '{}' metric",
            cfg.metric.name()
        );
        eprintln!(
            "supported weighted metrics: {}",
            metrics::weighted_metric_names().join(", ")
        );
        std::process::exit(EXIT_USAGE);
    }
    // The .632+ bootstrap resamples and refits without weights; refusing the
    // combination is honest about that rather than reporting a mixed estimate.
    if weight_name.is_some() && bootstrap_resamples > 0 {
        eprintln!(
            "error: --weight cannot be combined with --bootstrap-error \
             (the .632+ bootstrap resamples rows without weights)"
        );
        std::process::exit(EXIT_USAGE);
    }

    let target_idx = match table.index_of(&target) {
        Ok(j) => j,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };

    // Resolve the weight column by name here so it can be excluded from the
    // default inputs; numeric/value validation happens in `data`.
    let weight_col = match &weight_name {
        Some(n) => match table.index_of(n) {
            Ok(j) => Some(j),
            Err(_) => {
                eprintln!(
                    "error: weight column '{}' not found (columns: {:?})",
                    n, table.names
                );
                std::process::exit(EXIT_INPUT);
            }
        },
        None => None,
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
    let vars_explicit = !vars.is_empty();
    if vars.is_empty() {
        vars = (0..table.names.len())
            .filter(|&j| {
                table.numeric[j]
                    && table.names[j] != target
                    && Some(j) != time_col
                    && Some(j) != group_col
                    && Some(j) != weight_col
            })
            .map(|j| table.names[j].clone())
            .collect();
    }

    // Per-variable datatypes (issue #5). A declared datatype is also a request
    // to use the column as an input: when --vars was omitted, a text
    // categorical column is added to the defaults. An explicit --vars wins, and
    // naming a column that is not in it is a usage error rather than a silent
    // addition.
    let dtype_spec = match dtype_src.as_deref() {
        Some(s) => match data::parse_dtype_spec(s) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
        },
        None => Vec::new(),
    };
    for (name, _) in &dtype_spec {
        if name == &target {
            eprintln!(
                "error: column '{}' is the target; --dtype applies to input variables",
                name
            );
            std::process::exit(EXIT_USAGE);
        }
        let j = match table.index_of(name) {
            Ok(j) => j,
            Err(_) => {
                eprintln!(
                    "error: unknown --dtype column '{}' (columns: {:?})",
                    name, table.names
                );
                std::process::exit(EXIT_USAGE);
            }
        };
        if Some(j) == time_col || Some(j) == group_col || Some(j) == weight_col {
            eprintln!(
                "error: column '{}' is used for --time/--group-by/--weight and cannot be an input",
                name
            );
            std::process::exit(EXIT_USAGE);
        }
        if !vars.iter().any(|v| v == name) {
            if vars_explicit {
                eprintln!(
                    "error: column '{}' is not an input variable; add it to --vars or drop it from --dtype",
                    name
                );
                std::process::exit(EXIT_USAGE);
            }
            vars.push(name.clone());
        }
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
        if let Some(j) = weight_col {
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
        notes.push(format!("using the {} split", m.name()));
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
                         group on both sides). Consider `--split auto` (it would\n         \
                         choose: {}), or pass explicitly:  {}",
                        nt.join("; "),
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

    // Per-variable datatypes (issue #5): validate boolean columns and expand
    // categorical columns into one-hot indicators. This runs before lags so a
    // delayed copy of an indicator is itself a 0/1 boolean.
    let typed = match data::apply_dtypes(&table, &vars, &dtype_spec, max_levels) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(if e.usage { EXIT_USAGE } else { EXIT_INPUT });
        }
    };
    let table = typed.table;
    vars = typed.vars;
    let dtypes = typed.dtypes;
    let variable_types = typed.variables;

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

    // Delayed copies of boolean inputs are boolean too, so keep the datatype
    // plan aligned with the expanded variable list.
    let mut dtypes = dtypes;
    let mut variable_types = variable_types;
    if dtypes.len() < vars.len() {
        let base = dtypes.len();
        for v in &vars[base..] {
            let source = v
                .rsplit_once("_lag")
                .and_then(|(stem, n)| {
                    if n.parse::<usize>().is_ok() {
                        variable_types
                            .iter()
                            .find(|t| t.name == stem)
                            .map(|t| (t.dtype, t.name.clone()))
                    } else {
                        None
                    }
                });
            let (kind, src) = match source {
                Some((k, s)) => (k, Some(s)),
                None => (data::DType::Continuous, None),
            };
            dtypes.push(kind);
            variable_types.push(data::VariableDType {
                name: v.clone(),
                dtype: kind,
                source: src,
            });
        }
    }

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
        dtypes: dtypes.clone(),
    };
    let mut rng = Mt::new(cfg.seed);
    let prepared = match data::prepare_full_weighted_with(
        &table,
        &vars,
        &target,
        weight_name.as_deref(),
        &spec,
        &mut rng,
        &prep,
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    notes.extend(prepared.report.notes.iter().cloned());

    // #25: resolve --units names against the prepared input list. A lagged
    // copy (`x_lag1`) inherits the unit declared for its source (`x`); every
    // other undeclared name, including the target, is dimensionless.
    if !unit_assignments.is_empty() {
        for (i, (name, _)) in unit_assignments.iter().enumerate() {
            if unit_assignments[..i].iter().any(|(n, _)| n == name) {
                eprintln!("error: --units declares '{}' more than once", name);
                std::process::exit(EXIT_USAGE);
            }
            let is_target = name == &target;
            let is_input = vars.iter().any(|v| v == name);
            let is_lag_source = vars.iter().any(|v| {
                v.rsplit_once("_lag")
                    .map(|(base, n)| base == name && n.parse::<usize>().is_ok())
                    .unwrap_or(false)
            });
            if !is_target && !is_input && !is_lag_source {
                eprintln!(
                    "error: --units names '{}', which is neither the target nor an input \
                     (inputs: {})",
                    name,
                    vars.join(", ")
                );
                std::process::exit(EXIT_USAGE);
            }
        }
        let declared = |name: &str| {
            unit_assignments
                .iter()
                .find(|(n, _)| n == name)
                .map(|(_, u)| u.clone())
        };
        let target_unit = declared(&target).unwrap_or(Unit::DIMENSIONLESS);
        let mut var_units = Vec::with_capacity(vars.len());
        for v in &vars {
            let unit = declared(v).or_else(|| {
                v.rsplit_once("_lag")
                    .filter(|(_, n)| n.parse::<usize>().is_ok())
                    .and_then(|(base, _)| declared(base))
            });
            var_units.push(unit.unwrap_or(Unit::DIMENSIONLESS));
        }
        cfg.units = Some(Units::new(var_units, target_unit.clone()));
        notes.push(format!(
            "units: target '{}' is {}{}",
            target,
            target_unit,
            if require_dimensions || cfg.require_dimensions {
                " — dimensional filter on"
            } else {
                " — not filtering; pass --require-dimensions to reject mismatches"
            }
        ));
    }
    if require_dimensions {
        cfg.require_dimensions = true;
    }
    if cfg.require_dimensions && cfg.units.is_none() {
        eprintln!("error: --require-dimensions needs --units (or a checkpoint that saved them)");
        std::process::exit(EXIT_USAGE);
    }

    // A resumed search must run on exactly the rows that produced the saved
    // state, otherwise every saved error and population would be meaningless.
    if let Some(cp) = &resume_checkpoint {
        if let Err(e) = cp.check_data(&prepared.train, &prepared.test) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    }

    // Seed expressions are written in original units, like `--eval-expr`, and
    // rewritten into the search's normalised-input space before they enter the
    // population. A syntax error is a usage error and keeps the parser's
    // 1-based column.
    let mut seeds: Vec<expr::Node> = Vec::new();
    // Layer 3 only: check the seed in original units, before normalisation
    // inserts constants that could otherwise absorb a mismatch.
    let seed_dimensions_ok = |tree: &expr::Node| -> bool {
        match (&cfg.units, cfg.require_dimensions) {
            (Some(units), true) => units.accepts(tree),
            _ => true,
        }
    };
    for src in &seed_exprs {
        match expr::parse(src, &vars) {
            Ok(tree) => {
                if !seed_dimensions_ok(&tree) {
                    eprintln!(
                        "# seed rejected: --seed-expr '{}' is not dimensionally consistent \
                         with the declared target (--require-dimensions)",
                        src
                    );
                    // Keep it so the search records the infinite error; it can
                    // never enter the archive.
                    seeds.push(tree);
                    continue;
                }
                seeds.push(normalise_inputs(&tree, &prepared.means, &prepared.stds));
            }
            Err(e) => {
                eprintln!("error: --seed-expr '{}': {}", src, e);
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    for (origin, src) in &seed_lines {
        match expr::parse(src, &vars) {
            Ok(tree) => {
                if !seed_dimensions_ok(&tree) {
                    eprintln!(
                        "# seed rejected: {}: '{}' is not dimensionally consistent \
                         with the declared target (--require-dimensions)",
                        origin, src
                    );
                    seeds.push(tree);
                    continue;
                }
                seeds.push(normalise_inputs(&tree, &prepared.means, &prepared.stds));
            }
            Err(e) => {
                eprintln!("error: {}: '{}': {}", origin, src, e);
                std::process::exit(EXIT_USAGE);
            }
        }
    }

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
        if let Some(name) = &weight_name {
            let zero = prepared
                .train
                .w
                .iter()
                .filter(|&&v| v == 0.0)
                .count();
            eprintln!(
                "# weights: column {}, sum {} over {} rows ({} zero-weight)",
                name,
                prepared.train.weight_sum(),
                prepared.train.len(),
                zero
            );
            eprintln!("# weights: null baselines and residual tests are unweighted");
        }
    }
    if format == Format::Ndjson {
        let folds_json: Vec<String> = prepared
            .fold_samples
            .iter()
            .map(|f| f.len().to_string())
            .collect();
        println!(
            "{{\"event\":\"start\",\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"train_rows\":{},\"test_rows\":{},\"split\":{{\"mode\":{},\"gap\":{},\"folds\":[{}]}}}}",
            jstr(VERSION),
            cfg.seed,
            jstr(cfg.metric.name()),
            jstr(&target),
            vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
            variables_detail_json(&variable_types),
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
        let type_line: Vec<String> = variable_types
            .iter()
            .map(|v| format!("{} ({})", v.name, v.dtype.name()))
            .collect();
        eprintln!("# variables: {}", type_line.join(", "));
        if prepared.report.groups_total > 1 && mode == data::SplitMode::Group {
            eprintln!(
                "# groups held out whole: {}/{} (never on both sides)",
                prepared.report.groups_test, prepared.report.groups_total
            );
        }
    }

    if let Some(gen) = resumed_generation {
        let path = resume_path.as_deref().unwrap_or("?");
        if format == Format::Ndjson {
            println!(
                "{{\"event\":\"checkpoint\",\"action\":\"resumed\",\"path\":{},\"generation\":{}}}",
                jstr(path),
                gen
            );
        } else {
            eprintln!("# checkpoint: resumed from {} at generation {}", path, gen);
        }
    }

    if let Some(src) = eval_expr.as_deref() {
        run_eval(
            src,
            &vars,
            &target,
            &prepared,
            &variable_types,
            cfg.metric,
            &cfg.complexity_model,
            format,
            quiet,
            start,
        );
        std::process::exit(EXIT_OK);
    }

    let mut last_saved: Option<(usize, String)> = None;
    let outcome_result = if let Some(path) = save_path.clone() {
        let resumed = resume_checkpoint.take();
        search::run_resumable_seeded(
            &cfg,
            &prepared.train,
            &prepared.test,
            resumed,
            save_every,
            &seeds,
            |cp| {
                if let Err(e) = cp.save(&path) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
                let generation = cp.generations;
                last_saved = Some((generation, path.clone()));
                if format == Format::Ndjson {
                    println!(
                        "{{\"event\":\"checkpoint\",\"action\":\"saved\",\"path\":{},\"generation\":{}}}",
                        jstr(&path),
                        generation
                    );
                } else {
                    eprintln!(
                        "# checkpoint: saved to {} at generation {}",
                        path, generation
                    );
                }
            },
        )
    } else if resume_checkpoint.is_none() {
        // A fresh run with no save does not need any snapshot at all: keep the
        // historical path (and avoid hashing the data on exit).
        Ok(search::run_seeded(
            &cfg,
            &prepared.train,
            &prepared.test,
            &seeds,
        ))
    } else {
        let resumed = resume_checkpoint.take();
        search::run_resumable_seeded(
            &cfg,
            &prepared.train,
            &prepared.test,
            resumed,
            0,
            &seeds,
            |_| {},
        )
    };
    let outcome = match outcome_result {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    let elapsed = start.elapsed().as_secs_f64();

    // Seeded solutions, reported once each. The search holds them in
    // normalised-input space; show the original-units form, exactly like a
    // Pareto row. A seed above --max-complexity is still kept, and saying so
    // here means the rule is never a surprise.
    let seed_rows: Vec<(String, f64, f64)> = outcome
        .seeds
        .iter()
        .map(|s| {
            let tree = expr::simplify(&expr::denormalize(
                &expr::simplify(&s.tree),
                &prepared.means,
                &prepared.stds,
            ));
            (expr::to_string(&tree, &vars), s.complexity, s.error)
        })
        .collect();
    for (expression, complexity, _) in &seed_rows {
        if *complexity > cfg.max_complexity as f64 {
            eprintln!(
                "# note: seeded expression '{}' has complexity {} above --max-complexity {}; \
                 kept as an elite",
                expression, complexity, cfg.max_complexity
            );
        }
    }
    if format == Format::Text {
        for (expression, complexity, error) in &seed_rows {
            eprintln!(
                "# seeded: complexity={} train_error={}  y = {}",
                complexity,
                fmt_err(*error),
                expression
            );
        }
    }

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
            // #25: dimensions of the fitted constants in the search tree. This
            // is the same tree `constant_errors` uses, so values and units
            // line up; the printed law is the denormalised form of it.
            let (constant_values, constant_units) = match &cfg.units {
                Some(units) => (
                    rcsr::units::constant_values(&s.tree),
                    rcsr::units::constant_dimensions(&s.tree, &units.vars, &units.target),
                ),
                None => (Vec::new(), Vec::new()),
            };
            let dimensionally_consistent =
                cfg.units.as_ref().map(|units| units.accepts(&s.tree));
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
                constant_errors: if prepared.train.w.is_empty() {
                    fit::constant_errors(&s.tree, &prepared.train.x, &prepared.train.y)
                } else {
                    fit::constant_errors_weighted(
                        &s.tree,
                        &prepared.train.x,
                        &prepared.train.y,
                        &prepared.train.w,
                    )
                },
                constant_values,
                constant_units,
                dimensionally_consistent,
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
        let folds = match data::cv_folds_typed(
            &table,
            &vars,
            &target,
            weight_name.as_deref(),
            mode,
            cv_k,
            gap,
            cfg.seed,
            &dtypes,
        ) {
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
                w: prepared.train.w.clone(),
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
    let full_rows = rows.clone();

    // Optional Markdown report (issue #29), built from the full Pareto front so
    // `--top` only trims the ordinary table/TSV output below.
    let report_md = report.map(|kind| match kind {
        ReportKind::Summary => summary_markdown(
            &command,
            &prepared,
            &target,
            &vars,
            &variable_types,
            &cfg,
            &outcome,
            &full_rows,
            &baselines,
            &best_resid,
            &null_control,
            &cv,
            &stab,
            &lag_report,
            elapsed,
            outlier_iqr,
            smooth,
            &notes,
        ),
        ReportKind::Sensitivity => {
            let selected = full_rows.iter().min_by(|a, b| {
                a.train_error
                    .partial_cmp(&b.train_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let sorted = outcome.archive.sorted();
            let tree = selected
                .and_then(|r| r.rank.checked_sub(1).and_then(|i| sorted.get(i)))
                .map(|s| &s.tree);
            sensitivity_markdown(selected, tree, &prepared, &vars, &target)
        }
        ReportKind::Benchmark => benchmark_markdown(
            &cfg,
            &prepared,
            &vars,
            &target,
            &outcome,
            elapsed,
            report_repeats,
            quiet,
        ),
    });

    if top > 0 && rows.len() > top {
        rows.truncate(top);
    }

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
            // #25: a cheap plausibility check for the constants in the law.
            for r in &rows {
                if !r.constant_units.is_empty() {
                    eprintln!(
                        "# constant dimensions (rank {}): {} (value:dimension, tree order)",
                        r.rank,
                        constant_units_text(&r.constant_values, &r.constant_units)
                    );
                }
            }
            // Layer 2: units alone never filter, but say how many archived
            // solutions the Layer 3 flag would reject.
            if cfg.units.is_some() && !cfg.require_dimensions {
                let inconsistent = rows
                    .iter()
                    .filter(|r| r.dimensionally_consistent == Some(false))
                    .count();
                if inconsistent > 0 {
                    eprintln!(
                        "# dimensional note: {} of {} archived solution(s) would be rejected by \
                         --require-dimensions (units are not filtering)",
                        inconsistent,
                        rows.len()
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
            let (
                weight_column_json,
                weight_sum_json,
                weight_rows_json,
                weight_positive_rows_json,
            ) = match &weight_name {
                Some(name) => (
                    jstr(name),
                    jnum(prepared.train.weight_sum()),
                    prepared.train.len().to_string(),
                    prepared.train.positive_weight_rows().to_string(),
                ),
                None => (
                    "null".to_string(),
                    "null".to_string(),
                    "null".to_string(),
                    "null".to_string(),
                ),
            };
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
            let seeds_json: Vec<String> = seed_rows
                .iter()
                .map(|(expression, complexity, error)| {
                    format!(
                        "{{\"expression\":{},\"complexity\":{},\"train_error\":{}}}",
                        jstr(expression),
                        jnum(*complexity),
                        jnum(*error)
                    )
                })
                .collect();
            let pareto: Vec<String> = rows
                .iter()
                .map(|r| {
                    // Layer 1: no `--units` means no new output field at all.
                    let dimensions = if cfg.units.is_some() {
                        format!(
                            ",\"constant_dimensions\":{}",
                            constant_units_json(&r.constant_values, &r.constant_units)
                        )
                    } else {
                        String::new()
                    };
                    format!(
                        "{{\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{}{},\"stability\":{},\"expression\":{}}}",
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
                        dimensions,
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
            let checkpoint_json = format!(
                "{{\"resumed_from\":{},\"saved_to\":{},\"generation\":{}}}",
                resume_path
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                last_saved
                    .as_ref()
                    .map(|(_, p)| jstr(p))
                    .unwrap_or_else(|| "null".to_string()),
                outcome.generations
            );
            println!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"weight_column\":{},\"weight_sum\":{},\"weight_rows\":{},\"weight_positive_rows\":{},\"train_rows\":{},\"test_rows\":{},\"split\":{},\"preparation\":{},\"generations_run\":{},\"converged_early\":{},\"elapsed_seconds\":{},\"checkpoint\":{},\"null_control\":{},\"baselines\":{},\"residual_tests\":{},\"cross_validation\":{},\"stability\":{},\"seeds\":[{}],\"pareto\":[{}]}}",
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                jstr(&target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                variables_detail_json(&variable_types),
                weight_column_json,
                weight_sum_json,
                weight_rows_json,
                weight_positive_rows_json,
                prepared.train.len(),
                prepared.test.len(),
                split_json,
                prepared.prep.json(),
                outcome.generations,
                outcome.converged_early,
                jnum(elapsed),
                checkpoint_json,
                null_json,
                baselines_json,
                resid_json,
                cv_json,
                stability_json,
                seeds_json.join(","),
                pareto.join(",")
            );
        }
        Format::Ndjson => {
            for (expression, complexity, error) in &seed_rows {
                println!(
                    "{{\"event\":\"seed\",\"expression\":{},\"complexity\":{},\"train_error\":{}}}",
                    jstr(expression),
                    jnum(*complexity),
                    jnum(*error)
                );
            }
            for r in &rows {
                // Layer 1: no `--units` means no new output field at all.
                let dimensions = if cfg.units.is_some() {
                    format!(
                        ",\"constant_dimensions\":{}",
                        constant_units_json(&r.constant_values, &r.constant_units)
                    )
                } else {
                    String::new()
                };
                println!(
                    "{{\"event\":\"solution\",\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{}{},\"stability\":{},\"expression\":{}}}",
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
                    dimensions,
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

    // The Markdown report never shares stdout with a machine-readable stream:
    // `--json`/`--ndjson` put it on stderr unless `--report-out` names a file.
    if let Some(md) = report_md {
        match &report_out {
            Some(path) => {
                if let Err(e) = std::fs::write(path, &md) {
                    eprintln!("error writing {}: {}", path, e);
                    std::process::exit(EXIT_INPUT);
                }
                if !quiet {
                    eprintln!("# report written to {}", path);
                }
            }
            None if format == Format::Text => {
                println!("\n{}", md);
            }
            None => {
                eprintln!("{}", md);
            }
        }
    }

    std::process::exit(if finite { EXIT_OK } else { EXIT_NO_SOLUTION });
}

/// Rewrite a formula written in **original units** into the search's
/// normalised-input space: every `Input(i)` standing for the raw variable
/// `x_i` becomes `x_i * std_i + mean_i`, so the search's normalised input
/// evaluates it correctly.
///
/// This is the reverse of [`rcsr::expr::denormalize`], used for `--seed-expr`
/// and `--seed-file`. A variable whose affine transform is the identity (a
/// boolean column, or `--center none --scale none`) is left as a bare input.
fn normalise_inputs(n: &expr::Node, means: &[f64], stds: &[f64]) -> expr::Node {
    match n.op {
        Op::Input => {
            let i = n.val as usize;
            let m = means.get(i).copied().unwrap_or(0.0);
            let s = stds.get(i).copied().unwrap_or(1.0);
            if m == 0.0 && s == 1.0 {
                expr::input(i)
            } else {
                expr::node(
                    Op::Add,
                    vec![
                        expr::node(Op::Mul, vec![expr::input(i), expr::constant(s)]),
                        expr::constant(m),
                    ],
                )
            }
        }
        _ => expr::Node {
            op: n.op,
            val: n.val,
            kids: n
                .kids
                .iter()
                .map(|k| normalise_inputs(k, means, stds))
                .collect(),
        },
    }
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
    variable_types: &[data::VariableDType],
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
        metrics::score_weighted(&pred, &s.y, &s.w, metric, k)
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
                "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"eval-expr\",\"metric\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"train_rows\":{},\"test_rows\":{},\"preparation\":{},\"eval\":{{\"expression\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}]}}}}",
                jstr(VERSION),
                jstr(metric.name()),
                jstr(target),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                variables_detail_json(variable_types),
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

// ---------------------------------------------------------------------------
// Markdown reports (issue #29)
// ---------------------------------------------------------------------------

fn md_num(v: f64) -> String {
    if v.is_nan() {
        "n/a".to_string()
    } else if v == f64::INFINITY {
        "inf".to_string()
    } else if v == f64::NEG_INFINITY {
        "-inf".to_string()
    } else {
        format!("{:.6}", v)
    }
}

fn md_pct(v: f64) -> String {
    if v.is_finite() {
        format!("{:.1}%", 100.0 * v)
    } else {
        "n/a".to_string()
    }
}

/// Uncertainty ratios are often tiny or huge; four decimals plus scientific
/// notation for small values keeps them readable without losing the signal.
fn md_rel(v: f64) -> String {
    if !v.is_finite() {
        md_num(v)
    } else if v != 0.0 && v.abs() < 1e-3 {
        format!("{:.2e}", v)
    } else {
        format!("{:.4}", v)
    }
}

fn md_esc(s: &str) -> String {
    s.replace('|', "\\|").replace('\n', " ")
}

/// The "hand this to someone else" report: provenance, data, the law, every
/// diagnostic already computed, the whole Pareto front and a trust verdict that
/// cites the numbers rather than asserting confidence.
#[allow(clippy::too_many_arguments)]
fn summary_markdown(
    command: &str,
    prepared: &data::Prepared,
    target: &str,
    vars: &[String],
    variable_types: &[data::VariableDType],
    cfg: &search::Config,
    outcome: &search::Outcome,
    rows: &[Row],
    baselines: &diagnostics::Baselines,
    best_resid: &Option<diagnostics::ResidualTests>,
    null_control: &Option<diagnostics::NullControl>,
    cv: &Option<CvReport>,
    stab: &Option<stability::Stability>,
    lag_report: &data::LagReport,
    elapsed: f64,
    outlier_iqr: Option<f64>,
    smooth: usize,
    notes: &[String],
) -> String {
    let mut s = String::new();
    let chosen = rows.iter().min_by(|a, b| {
        a.train_error
            .partial_cmp(&b.train_error)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let _ = writeln!(s, "# rcsr model summary\n");
    let _ = writeln!(s, "## What produced this\n");
    let _ = writeln!(s, "```sh\n{}\n```\n", command);
    let _ = writeln!(
        s,
        "Generated by `rcsr {}` in {:.2} s. Metric `{}`, seed `{}`, {} generation(s){}.\n",
        VERSION,
        elapsed,
        cfg.metric.name(),
        cfg.seed,
        outcome.generations,
        if outcome.converged_early {
            " (stopped early)"
        } else {
            ""
        }
    );

    let _ = writeln!(s, "## Data\n");
    let _ = writeln!(s, "| item | value |");
    let _ = writeln!(s, "|---|---|");
    let _ = writeln!(s, "| rows read | {} |", prepared.report.rows_total);
    let _ = writeln!(s, "| training rows | {} |", prepared.train.len());
    let _ = writeln!(s, "| test rows | {} |", prepared.test.len());
    let _ = writeln!(
        s,
        "| split | `{}` (gap {}, {} fold(s)) |",
        prepared.report.mode, prepared.report.gap, prepared.report.folds
    );
    if prepared.report.groups_total > 0 {
        let _ = writeln!(
            s,
            "| groups | {}/{} in train, {}/{} in test |",
            prepared.report.groups_train,
            prepared.report.groups_total,
            prepared.report.groups_test,
            prepared.report.groups_total
        );
    }
    let _ = writeln!(s, "| target | `{}` |", md_esc(target));
    let _ = writeln!(
        s,
        "| inputs | {} |",
        vars.iter()
            .map(|v| format!("`{}`", md_esc(v)))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        s,
        "| variable types | {} |",
        variable_types
            .iter()
            .map(|v| format!("`{}` ({})", md_esc(&v.name), v.dtype.name()))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let _ = writeln!(
        s,
        "| normalisation | centre=`{}`, scale=`{}` |",
        prepared.prep.center, prepared.prep.scale
    );
    if lag_report.lags > 0 || lag_report.history_rows > 0 {
        let _ = writeln!(
            s,
            "| lags / history | {} lag(s), {} series, {} history row(s) |",
            lag_report.lags, lag_report.series, lag_report.history_rows
        );
    }
    let _ = writeln!(s);

    let prep_lines = prepared.prep.text_lines(outlier_iqr, smooth);
    let _ = writeln!(s, "### Preparation\n");
    if prep_lines.is_empty() {
        let _ = writeln!(
            s,
            "No opt-in preparation ran (missing values left alone, mean/std scaling).\n"
        );
    } else {
        for line in prep_lines {
            let _ = writeln!(s, "- {}", line.trim_start_matches("# "));
        }
        let _ = writeln!(s);
    }
    if !notes.is_empty() {
        let _ = writeln!(s, "### Run notes\n");
        for n in notes {
            let _ = writeln!(s, "- {}", n);
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "## Chosen law\n");
    match chosen {
        Some(r) => {
            let _ = writeln!(
                s,
                "Selected by lowest full-data training error; the Pareto table below\n\
                 lists every non-dominated trade-off.\n"
            );
            let _ = writeln!(s, "```text\n{} = {}\n```\n", target, r.expression);
            let _ = writeln!(s, "| quantity | value |");
            let _ = writeln!(s, "|---|---|");
            let _ = writeln!(s, "| complexity (nodes) | {} |", r.complexity);
            let _ = writeln!(s, "| fitted constants (k) | {} |", r.k_constants);
            let _ = writeln!(s, "| train error ({}) | {} |", cfg.metric.name(), md_num(r.train_error));
            let _ = writeln!(s, "| test error ({}) | {} |", cfg.metric.name(), md_num(r.test_error));
            let _ = writeln!(s, "| overfit index (test/train) | {} |", md_num(r.overfit));
            let _ = writeln!(s, "| AICc | {} |", md_num(r.aicc));
            let _ = writeln!(s, "| BIC | {} |", md_num(r.bic));
            let _ = writeln!(s, "| Durbin–Watson | {} |", md_num(r.dw));
            let _ = writeln!(s);

            let _ = writeln!(s, "### Fitted constants and standard errors\n");
            match &r.constant_errors {
                Some(ce) => {
                    let _ = writeln!(
                        s,
                        "Constants are listed in left-to-right order of the fitted\n\
                         (normalised-input) model; the printed law above is the same\n\
                         model denormalised to original units.\n"
                    );
                    let _ = writeln!(s, "| # | value | std. error | relative |");
                    let _ = writeln!(s, "|---:|---:|---:|---:|");
                    for (i, ((v, se), rel)) in ce
                        .values
                        .iter()
                        .zip(&ce.std_errors)
                        .zip(&ce.relative)
                        .enumerate()
                    {
                        let _ = writeln!(
                            s,
                            "| {} | {} | {} | {} |",
                            i + 1,
                            md_num(*v),
                            md_num(*se),
                            md_rel(*rel)
                        );
                    }
                    let _ = writeln!(s, "\nResidual degrees of freedom: {}.", ce.dof);
                    if ce.ill_conditioned {
                        let _ = writeln!(
                            s,
                            "**Warning:** JᵀJ is near-singular; the constants are not\n\
                             separately identified by the data."
                        );
                    }
                    let _ = writeln!(s);
                }
                None if r.k_constants == 0 => {
                    let _ = writeln!(s, "This law has no fitted constants.\n");
                }
                None => {
                    let _ = writeln!(
                        s,
                        "Standard errors are undefined (too few residual degrees of freedom).\n"
                    );
                }
            }
            // #25: the units are inferred from the printed law, not from the
            // fitted values, so they are a plausibility check rather than an
            // uncertainty estimate.
            if !r.constant_units.is_empty() {
                let _ = writeln!(
                    s,
                    "Apparent constant dimensions (value:dimension, tree order): {}.\n",
                    constant_units_text(&r.constant_values, &r.constant_units)
                );
            }
        }
        None => {
            let _ = writeln!(
                s,
                "No finite solution was found within the budget. Lower `--max-complexity`,\n\
                 raise `--generations`/`--time-budget`, or check the data.\n"
            );
        }
    }

    if let Some(r) = chosen {
        let _ = writeln!(s, "## Residual tests\n");
        match best_resid {
            Some(rt) => {
                let lb = if rt.ljung_box.p_value < 0.05 {
                    "autocorrelation left"
                } else {
                    "no evidence of autocorrelation"
                };
                let bp = if rt.breusch_pagan.p_value < 0.05 {
                    "variance changes with fitted value"
                } else {
                    "no evidence of changing variance"
                };
                let _ = writeln!(s, "| test | statistic | df | p-value | reading |");
                let _ = writeln!(s, "|---|---:|---:|---:|---|");
                let _ = writeln!(
                    s,
                    "| Ljung–Box | {} | {} | {} | {} |",
                    md_num(rt.ljung_box.statistic),
                    rt.ljung_box.lags,
                    md_num(rt.ljung_box.p_value),
                    lb
                );
                let _ = writeln!(
                    s,
                    "| Breusch–Pagan | {} | {} | {} | {} |",
                    md_num(rt.breusch_pagan.statistic),
                    rt.breusch_pagan.df,
                    md_num(rt.breusch_pagan.p_value),
                    bp
                );
                let _ = writeln!(s, "| residual skewness | {} | – | – | 0 is symmetric |", md_num(rt.skewness));
                let _ = writeln!(
                    s,
                    "| excess kurtosis | {} | – | – | 0 is Gaussian |",
                    md_num(rt.kurtosis)
                );
                let _ = writeln!(s);
            }
            None => {
                let _ = writeln!(s, "Residual tests are not available for this run.\n");
            }
        }

        let _ = writeln!(s, "## Baselines\n");
        let baseline_items = [
            ("constant mean", baselines.mean),
            ("least-squares line", baselines.linear),
            (
                "k-nearest-neighbour",
                diagnostics::BaselineError {
                    train_error: baselines.knn.train_error,
                    test_error: baselines.knn.test_error,
                },
            ),
        ];
        let _ = writeln!(s, "| baseline | train error | test error | chosen law beats it on test? |");
        let _ = writeln!(s, "|---|---:|---:|---|");
        for (name, b) in baseline_items {
            let label = if name == "k-nearest-neighbour" {
                format!("{name} (k={})", baselines.knn_k)
            } else {
                name.to_string()
            };
            let beats = if r.test_error < b.test_error { "yes" } else { "no" };
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} |",
                label,
                md_num(b.train_error),
                md_num(b.test_error),
                beats
            );
        }
        let _ = writeln!(s);

        let _ = writeln!(s, "## Null control\n");
        match null_control {
            Some(c) => {
                let _ = writeln!(s, "| shuffled runs | best null error | median null error | p-value | verdict |");
                let _ = writeln!(s, "|---:|---:|---:|---:|---|");
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | {} |\n",
                    c.runs,
                    md_num(c.null_min),
                    md_num(c.null_median),
                    md_num(c.p_value),
                    c.verdict()
                );
            }
            None => {
                let _ = writeln!(
                    s,
                    "_Not run. Add `--null-runs 19` (or more) to compare the real fit\n\
                     against the identical search on shuffled targets._\n"
                );
            }
        }

        let bootstrap_rows: Vec<&Row> = rows.iter().filter(|x| x.bootstrap.is_some()).collect();
        if !bootstrap_rows.is_empty() {
            let _ = writeln!(s, "## Bootstrap error (.632+)\n");
            let _ = writeln!(s, "| rank | apparent | out-of-bag | .632+ | resamples | weight |");
            let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|");
            for x in bootstrap_rows {
                if let Some(b) = &x.bootstrap {
                    let _ = writeln!(
                        s,
                        "| {} | {} | {} | {} | {} | {} |",
                        x.rank,
                        md_num(b.apparent),
                        md_num(b.oob),
                        md_num(b.estimate),
                        b.used,
                        md_num(b.weight)
                    );
                }
            }
            let _ = writeln!(s);
        }

        if let Some(c) = cv {
            let _ = writeln!(s, "## Cross-validation\n");
            let _ = writeln!(
                s,
                "| scheme | folds | mean test | std test | mean train |"
            );
            let _ = writeln!(s, "|---|---:|---:|---:|---:|");
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} |\n",
                c.mode,
                c.k,
                md_num(c.mean_test()),
                md_num(c.std_test()),
                md_num(c.mean_train())
            );
            let _ = writeln!(s, "| fold | train error | test error | complexity | law |");
            let _ = writeln!(s, "|---:|---:|---:|---:|---|");
            for (i, r) in c.folds.iter().enumerate() {
                let _ = writeln!(
                    s,
                    "| {} | {} | {} | {} | `{} = {}` |",
                    i + 1,
                    md_num(r.train_error),
                    md_num(r.test_error),
                    r.complexity,
                    md_esc(target),
                    md_esc(&r.expression)
                );
            }
            let _ = writeln!(s);
        }

        if let Some(st) = stab {
            let _ = writeln!(s, "## Stability selection\n");
            if let Some(top) = st.forms.first() {
                let _ = writeln!(
                    s,
                    "{} resamples, {} found a solution. Modal form selected in {:.1}% of runs:\n",
                    st.runs,
                    st.found,
                    100.0 * top.frequency
                );
                let _ = writeln!(s, "```text\n{} = {}\n```\n", target, top.example);
            }
            let _ = writeln!(s, "| input | selected in | frequency |");
            let _ = writeln!(s, "|---|---:|---:|");
            for (i, count) in &st.variables {
                let name = vars.get(*i).cloned().unwrap_or_else(|| format!("x{i}"));
                let freq = if st.runs > 0 {
                    *count as f64 / st.runs as f64
                } else {
                    0.0
                };
                let _ = writeln!(s, "| `{}` | {}/{} | {} |", name, count, st.runs, md_pct(freq));
            }
            let _ = writeln!(s);
        }
    }

    let _ = writeln!(s, "## Pareto front\n");
    let _ = writeln!(s, "| rank | complexity | constants | train | test | AICc | BIC | law |");
    let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|---:|---|");
    for r in rows {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | `{} = {}` |",
            r.rank,
            r.complexity,
            r.k_constants,
            md_num(r.train_error),
            md_num(r.test_error),
            md_num(r.aicc),
            md_num(r.bic),
            md_esc(target),
            md_esc(&r.expression)
        );
    }
    let _ = writeln!(s);

    let _ = writeln!(s, "## How much should you trust this?\n");
    let _ = writeln!(s, "{}\n", trust_paragraph(prepared, cfg, rows, chosen, baselines, best_resid, null_control, cv));
    s
}

/// Plain-language verdict assembled from the diagnostics. Each sentence cites
/// the statistic that drove it, so a reader can check the reasoning.
#[allow(clippy::too_many_arguments)]
fn trust_paragraph(
    prepared: &data::Prepared,
    cfg: &search::Config,
    rows: &[Row],
    chosen: Option<&Row>,
    baselines: &diagnostics::Baselines,
    best_resid: &Option<diagnostics::ResidualTests>,
    null_control: &Option<diagnostics::NullControl>,
    cv: &Option<CvReport>,
) -> String {
    let Some(r) = chosen else {
        return "There is no finite solution to judge. Treat this run as inconclusive \
                and adjust the search budget or the data before drawing a conclusion."
            .to_string();
    };

    let mut out = String::new();
    let mut strong = 0usize;
    let mut weak = 0usize;

    let baseline_items = [
        ("constant mean", baselines.mean),
        ("least-squares line", baselines.linear),
        ("k-nearest-neighbour", baselines.knn),
    ];
    let mut beats = 0usize;
    let mut best_name = "none";
    let mut best_test = f64::INFINITY;
    for (name, b) in baseline_items {
        if r.test_error < b.test_error {
            beats += 1;
        }
        if b.test_error < best_test {
            best_test = b.test_error;
            best_name = name;
        }
    }
    let beat_phrase = match beats {
        3 => "all three baselines".to_string(),
        0 => "none of the three baselines".to_string(),
        n => format!("{} of the three baselines", n),
    };
    let _ = write!(
        out,
        "The selected law's held-out {} error is {} against {} for the best trivial \
         baseline ({}); it beats {} on the test rows. ",
        cfg.metric.name(),
        md_num(r.test_error),
        md_num(best_test),
        best_name,
        beat_phrase
    );
    if beats == 3 {
        strong += 1;
    } else {
        weak += 1;
    }

    if r.overfit.is_finite() {
        let _ = write!(
            out,
            "The overfit index (test/train) is {}, so ",
            md_num(r.overfit)
        );
        if r.overfit <= 1.5 {
            let _ = write!(out, "the fit is essentially as good off-sample as on it. ");
            strong += 1;
        } else if r.overfit > 2.0 {
            let _ = write!(
                out,
                "it degrades materially on rows the search never saw. "
            );
            weak += 1;
        } else {
            let _ = write!(out, "it degrades somewhat off-sample. ");
        }
    } else {
        let _ = write!(
            out,
            "The test error is infinite, so the model does not generalise to the held-out rows. "
        );
        weak += 1;
    }

    match null_control {
        Some(c) => {
            let _ = write!(
                out,
                "Across {} shuffled-target runs the search reached a best error of {} \
                 (median {}); the permutation p-value is {}, which means ",
                c.runs,
                md_num(c.null_min),
                md_num(c.null_median),
                md_num(c.p_value)
            );
            if c.p_value <= 0.05 {
                let _ = write!(out, "the real fit is unlikely to be chance. ");
                strong += 1;
            } else if c.p_value <= 0.2 {
                let _ = write!(
                    out,
                    "it is only weakly better than what the search manufactures from noise. "
                );
            } else {
                let _ = write!(
                    out,
                    "the search fits shuffled targets about as well, so this is likely overfitting. "
                );
                weak += 1;
            }
        }
        None => {
            let _ = write!(
                out,
                "No permutation control was run, so this report cannot rule out the \
                 search fitting noise; add `--null-runs 19` or more. "
            );
            weak += 1;
        }
    }

    match best_resid {
        Some(rt) => {
            let lb = rt.ljung_box.p_value;
            let bp = rt.breusch_pagan.p_value;
            let _ = write!(
                out,
                "The residual checks give Ljung–Box p={} and Breusch–Pagan p={}, i.e. ",
                md_num(lb),
                md_num(bp)
            );
            if lb >= 0.05 && bp >= 0.05 {
                let _ = write!(
                    out,
                    "no significant leftover autocorrelation or changing variance. "
                );
                strong += 1;
            } else if lb < 0.01 || bp < 0.01 {
                let _ = write!(
                    out,
                    "the residuals still carry structure the law does not explain. "
                );
                weak += 1;
            } else {
                let _ = write!(out, "the evidence is borderline. ");
            }
        }
        None => {
            let _ = write!(out, "Residual tests were not available. ");
        }
    }

    match &r.constant_errors {
        Some(ce) if ce.ill_conditioned => {
            let _ = write!(
                out,
                "JᵀJ is near-singular, so at least one constant is not separately identified. "
            );
            weak += 1;
        }
        Some(ce) => {
            let worst = ce
                .relative
                .iter()
                .copied()
                .filter(|v| v.is_finite())
                .fold(0.0_f64, f64::max);
            let _ = write!(
                out,
                "The worst relative standard error among the {} constant(s) is {}",
                ce.values.len(),
                md_rel(worst)
            );
            if worst <= 0.25 {
                let _ = write!(out, ", so the fitted values are pinned down well. ");
                strong += 1;
            } else if worst > 0.5 {
                let _ = write!(out, ", so the data do not pin every constant down well. ");
                weak += 1;
            } else {
                let _ = write!(out, ". ");
            }
        }
        None => {}
    }

    if let Some(c) = cv {
        let mean = c.mean_test();
        let sd = c.std_test();
        let _ = write!(
            out,
            "Cross-validation over {} folds gives a mean held-out error of {} with a \
             fold standard deviation of {}. ",
            c.k,
            md_num(mean),
            md_num(sd)
        );
        if mean > 0.0 && mean.is_finite() {
            let rel = sd / mean;
            if rel <= 0.5 {
                strong += 1;
            } else if rel > 1.0 {
                weak += 1;
            }
        }
    }

    let n_finite = rows.iter().filter(|r| r.train_error.is_finite()).count();
    let verdict = if n_finite == 0 || strong == 0 {
        "weak"
    } else if weak >= 2 {
        "weak"
    } else if weak == 0 && strong >= 3 {
        "strong"
    } else {
        "moderate"
    };
    let _ = write!(
        out,
        "Overall: **{} evidence** from {} training rows and {} test rows in a `{}` split. ",
        verdict,
        prepared.train.len(),
        prepared.test.len(),
        prepared.report.mode
    );
    match verdict {
        "strong" => {
            let _ = write!(
                out,
                "The model beats every trivial baseline, survives the permutation \
                 control and leaves clean residuals; quote the test error, not the \
                 training error."
            );
        }
        "moderate" => {
            let _ = write!(
                out,
                "Some checks pass and some do not; treat the law as a candidate, not \
                 a result, and run more controls (`--null-runs`, `--cv`, `--stability`) \
                 before relying on it."
            );
        }
        _ => {
            let _ = write!(
                out,
                "At least two independent checks failed, so the fit should not be \
                 presented as a discovered law without further evidence."
            );
        }
    }
    out
}

/// Analytic variable sensitivity, ranked so the input that actually moves the
/// prediction is at the top.
fn sensitivity_markdown(
    selected: Option<&Row>,
    tree: Option<&expr::Node>,
    prepared: &data::Prepared,
    vars: &[String],
    target: &str,
) -> String {
    let mut s = String::new();
    let _ = writeln!(s, "# Variable sensitivity\n");
    let (Some(row), Some(tree)) = (selected, tree) else {
        let _ = writeln!(
            s,
            "No finite solution was found, so there is no law to differentiate.\n"
        );
        return s;
    };

    let report = diagnostics::sensitivity(tree, &prepared.train, &prepared.stds);
    let _ = writeln!(
        s,
        "Selected solution (lowest training error):\n\n```text\n{} = {}\n```\n",
        target, row.expression
    );
    let _ = writeln!(
        s,
        "The derivative below is **analytic** — the same adjoint (reverse-mode)\n\
         sweep the fitter uses for constants, seeded at the model output and\n\
         accumulated at each input leaf. It is evaluated at all {} training rows;\n\
         no finite differences are involved. The elasticity\n\n\
         `mean(|∂f/∂x_i| · σ_x_i / σ_f)`\n\n\
         is scale-free, so inputs with different units are directly comparable.\n\
         Here `σ_f = {}` on the training rows.\n",
        report.n,
        md_num(report.output_std)
    );
    let _ = writeln!(
        s,
        "| rank | input | mean abs elasticity | min contribution | max contribution | positive rows | negative rows |"
    );
    let _ = writeln!(s, "|---:|---|---:|---:|---:|---:|---:|");
    for (i, r) in report.rows.iter().enumerate() {
        let name = vars.get(r.input).cloned().unwrap_or_else(|| format!("x{}", r.input));
        let _ = writeln!(
            s,
            "| {} | `{}` | {} | {} | {} | {} | {} |",
            i + 1,
            md_esc(&name),
            md_num(r.mean_abs_elasticity),
            md_num(r.min_contribution),
            md_num(r.max_contribution),
            md_pct(r.positive_fraction),
            md_pct(r.negative_fraction)
        );
    }
    let _ = writeln!(s);

    if let (Some(first), Some(second)) = (report.rows.first(), report.rows.get(1)) {
        let a = vars
            .get(first.input)
            .cloned()
            .unwrap_or_else(|| format!("x{}", first.input));
        let b = vars
            .get(second.input)
            .cloned()
            .unwrap_or_else(|| format!("x{}", second.input));
        if second.mean_abs_elasticity > 0.0 {
            let ratio = first.mean_abs_elasticity / second.mean_abs_elasticity;
            let _ = writeln!(
                s,
                "**`{}` matters most:** its elasticity is {:.1}× that of `{}` ({} vs {}).",
                a,
                ratio,
                b,
                md_num(first.mean_abs_elasticity),
                md_num(second.mean_abs_elasticity)
            );
        } else {
            let _ = writeln!(
                s,
                "**`{}` is the only input with a non-zero derivative** (elasticity {}); \
                 the remaining inputs never enter the fitted law.",
                a,
                md_num(first.mean_abs_elasticity)
            );
        }
    }
    let _ = writeln!(
        s,
        "\nA positive fraction means the input pushes the prediction up; a negative\n\
         fraction means it pushes it down. The sign can change across rows, which is\n\
         exactly what a single correlation coefficient cannot show."
    );
    s
}

struct BenchmarkRun {
    seed: u32,
    seconds: f64,
    best_train: f64,
    best_test: f64,
    expression: String,
    points: Vec<diagnostics::BenchmarkPoint>,
}

/// One repeat: its per-complexity Pareto errors and its overall best law.
fn benchmark_run(
    cfg: &search::Config,
    prepared: &data::Prepared,
    vars: &[String],
    seed: u32,
    seconds: f64,
    outcome: &search::Outcome,
) -> BenchmarkRun {
    let fold_samples: &[data::Sample] = if prepared.fold_samples.is_empty() {
        std::slice::from_ref(&prepared.test)
    } else {
        &prepared.fold_samples
    };
    let mut best_map: BTreeMap<usize, diagnostics::BenchmarkPoint> = BTreeMap::new();
    let mut best: Option<(f64, f64, String)> = None;
    for sol in outcome.archive.sorted() {
        let train_error = search::eval_error(&sol.tree, &prepared.train, cfg.metric);
        if !train_error.is_finite() {
            continue;
        }
        let test_error = if fold_samples.is_empty() {
            f64::INFINITY
        } else {
            fold_samples
                .iter()
                .map(|f| search::eval_error(&sol.tree, f, cfg.metric))
                .sum::<f64>()
                / fold_samples.len() as f64
        };
        match best_map.get(&sol.complexity) {
            Some(old) if old.train_error <= train_error => {}
            _ => {
                best_map.insert(
                    sol.complexity,
                    diagnostics::BenchmarkPoint {
                        complexity: sol.complexity,
                        train_error,
                        test_error,
                    },
                );
            }
        }
        if best
            .as_ref()
            .map(|(e, _, _)| train_error < *e)
            .unwrap_or(true)
        {
            let den = expr::simplify(&expr::denormalize(
                &expr::simplify(&sol.tree),
                &prepared.means,
                &prepared.stds,
            ));
            best = Some((train_error, test_error, expr::to_string(&den, vars)));
        }
    }
    let (best_train, best_test, expression) =
        best.unwrap_or((f64::INFINITY, f64::INFINITY, "(no finite solution)".to_string()));
    BenchmarkRun {
        seed,
        seconds,
        best_train,
        best_test,
        expression,
        points: best_map.into_values().collect(),
    }
}

/// Repeated-search benchmark: per-complexity median/IQR error, wall-clock
/// median/IQR and the best law from each repeat.
fn benchmark_markdown(
    cfg: &search::Config,
    prepared: &data::Prepared,
    vars: &[String],
    target: &str,
    first_outcome: &search::Outcome,
    first_elapsed: f64,
    repeats: usize,
    quiet: bool,
) -> String {
    let repeats = repeats.max(1);
    let mut runs = Vec::with_capacity(repeats);
    runs.push(benchmark_run(
        cfg,
        prepared,
        vars,
        cfg.seed,
        first_elapsed,
        first_outcome,
    ));

    let mut extra = cfg.clone();
    extra.progress = None;
    for r in 1..repeats {
        extra.seed = cfg
            .seed
            .wrapping_add(0x9E37_79B9u32.wrapping_mul(r as u32));
        let start = std::time::Instant::now();
        let outcome = search::run(&extra, &prepared.train, &prepared.test);
        let seconds = start.elapsed().as_secs_f64();
        if !quiet {
            eprintln!(
                "# benchmark repeat {}/{}: seed={} ({:.2}s)",
                r + 1,
                repeats,
                extra.seed,
                seconds
            );
        }
        runs.push(benchmark_run(
            &extra, prepared, vars, extra.seed, seconds, &outcome,
        ));
    }

    let points: Vec<Vec<diagnostics::BenchmarkPoint>> =
        runs.iter().map(|r| r.points.clone()).collect();
    let cells = diagnostics::benchmark_by_complexity(&points);
    let seconds: Vec<f64> = runs.iter().map(|r| r.seconds).collect();

    let mut s = String::new();
    let _ = writeln!(s, "# Search benchmark\n");
    let _ = writeln!(
        s,
        "{} independent searches at the same budget: generations={}, islands={}, \
         pop={}, max-complexity={}, metric=`{}`. Seeds derive from `--seed {}` \
         (repeat 1 uses the seed itself, repeat r uses `seed + 0x9E3779B9·(r−1)`). \
         Each repeat honours `--time-budget`.\n",
        repeats,
        cfg.generations,
        cfg.islands,
        cfg.pop,
        cfg.max_complexity,
        cfg.metric.name(),
        cfg.seed
    );

    let _ = writeln!(s, "## Per-complexity Pareto error\n");
    if cells.is_empty() {
        let _ = writeln!(s, "The repeats produced no finite solution.\n");
    } else {
        let _ = writeln!(s, "| complexity | runs | train median | train IQR | test median | test IQR |");
        let _ = writeln!(s, "|---:|---:|---:|---:|---:|---:|");
        for c in &cells {
            let _ = writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} |",
                c.complexity,
                c.runs,
                md_num(c.train_median),
                md_num(c.train_iqr),
                md_num(c.test_median),
                md_num(c.test_iqr)
            );
        }
        let _ = writeln!(s);
    }

    let _ = writeln!(s, "## Wall clock\n");
    let _ = writeln!(s, "| statistic | seconds |");
    let _ = writeln!(s, "|---|---:|");
    let _ = writeln!(s, "| median | {} |", md_num(diagnostics::median(&seconds)));
    let _ = writeln!(s, "| IQR | {} |", md_num(diagnostics::iqr(&seconds)));
    let _ = writeln!(
        s,
        "| min | {} |",
        md_num(seconds.iter().copied().fold(f64::INFINITY, f64::min))
    );
    let _ = writeln!(
        s,
        "| max | {} |\n",
        md_num(seconds.iter().copied().fold(f64::NEG_INFINITY, f64::max))
    );

    let _ = writeln!(s, "## Best result per repeat\n");
    let _ = writeln!(s, "| repeat | seed | train error | test error | seconds | law |");
    let _ = writeln!(s, "|---:|---:|---:|---:|---:|---|");
    for (i, r) in runs.iter().enumerate() {
        let _ = writeln!(
            s,
            "| {} | {} | {} | {} | {} | `{} = {}` |",
            i + 1,
            r.seed,
            md_num(r.best_train),
            md_num(r.best_test),
            md_num(r.seconds),
            md_esc(target),
            md_esc(&r.expression)
        );
    }
    let _ = writeln!(s);
    let _ = writeln!(
        s,
        "A per-complexity cell uses each repeat's best training error at that\n\
         complexity; `runs` is how many repeats produced a solution there. Median\n\
         and IQR are Hyndman–Fan type-7 quantiles (the same definition the data\n\
         preparation uses)."
    );
    s
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

/// #25: `value:dimension` pairs in tree order, with `?` for undetermined.
fn constant_units_text(values: &[f64], units: &[Option<Unit>]) -> String {
    units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let value = values
                .get(i)
                .map(|v| format!("{}", v))
                .unwrap_or_else(|| "?".to_string());
            let dimension = u
                .as_ref()
                .map(|u| u.to_string())
                .unwrap_or_else(|| "?".to_string());
            format!("{}:{}", value, dimension)
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// #25: JSON array of `{value, dimension}` objects in tree order. An empty
/// array means `--units` was not active; `null` dimensions are undetermined.
fn constant_units_json(values: &[f64], units: &[Option<Unit>]) -> String {
    let items: Vec<String> = units
        .iter()
        .enumerate()
        .map(|(i, u)| {
            let value = values
                .get(i)
                .map(|v| jnum(*v))
                .unwrap_or_else(|| "null".to_string());
            let dimension = match u {
                Some(u) => jstr(&u.to_string()),
                None => "null".to_string(),
            };
            format!("{{\"value\":{},\"dimension\":{}}}", value, dimension)
        })
        .collect();
    format!("[{}]", items.join(","))
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
        let schema_text = schema::json();
        for m in Metric::all() {
            assert!(
                help.contains(m.name()),
                "usage() is missing metric '{}'",
                m.name()
            );
            assert!(
                schema_text.contains(m.name()),
                "schema::json() is missing metric '{}'",
                m.name()
            );
        }
    }

    /// Every report flag and type must stay documented in `--help` and `schema`.
    #[test]
    fn help_and_schema_document_reports() {
        for flag in ["--report", "--report-out", "--report-repeats"] {
            assert!(usage().contains(flag), "usage() is missing {flag}");
            assert!(schema::json().contains(flag), "schema::json() is missing {flag}");
        }
        for kind in ["summary", "sensitivity", "benchmark"] {
            assert!(usage().contains(kind), "usage() is missing report '{kind}'");
            assert!(
                schema::json().contains(kind),
                "schema::json() is missing report '{kind}'"
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
            schema::json().contains("--bootstrap-error"),
            "schema::json() is missing --bootstrap-error"
        );
    }

    /// Row weights are a first-class input: the flag and its supported metrics
    /// must appear in both hand-written descriptions.
    #[test]
    fn help_and_schema_document_weights() {
        assert!(usage().contains("--weight"), "usage() is missing --weight");
        assert!(schema::json().contains("--weight"), "schema::json() is missing --weight");
        for m in metrics::weighted_metric_names() {
            assert!(
                usage().contains(m),
                "usage() is missing weighted metric '{}'",
                m
            );
            assert!(
                schema::json().contains(m),
                "schema::json() is missing weighted metric '{}'",
                m
            );
        }
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
            assert!(schema::json().contains(flag), "schema::json() is missing {flag}");
        }
        // The default for missing values is the old behaviour: leave them.
        assert!(usage().contains("Default none"));
    }

    /// The conditional primitives must be discoverable from `schema`, and the
    /// `--ops` help must point at them.
    #[test]
    fn help_and_schema_list_conditional_operators() {
        let schema_text = schema::json();
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
                "schema::json() is missing operator '{name}'"
            );
        }
        assert!(usage().contains("between"), "usage() is missing 'between'");
        assert!(usage().contains("constrain"), "usage() is missing 'constrain'");
    }
}
