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

use rcsr::checkpoint::{Backend, Checkpoint};
use rcsr::complexity::ComplexityModel;
use rcsr::manifest;
use rcsr::memory;
use rcsr::expr::{self, Op};
use rcsr::fit;
use rcsr::metrics::{self, Metric};
use rcsr::rng::Mt;
use rcsr::schema;
use rcsr::units::{Unit, Units};
use rcsr::{
    backend, data, diagnostics, leakage, multi, qd, registry, search, stability, store, target,
    walkforward,
};
use std::collections::BTreeMap;
use std::fmt::Write as _;

const VERSION: &str = env!("CARGO_PKG_VERSION");
const EXIT_OK: i32 = 0;
const EXIT_USAGE: i32 = 2;
const EXIT_INPUT: i32 = 3;
const EXIT_NO_SOLUTION: i32 = 4;
/// #129: a leakage audit failed. Stable so CI can distinguish a real leak from
/// a usage (2) or input (3) error.
const EXIT_LEAKAGE: i32 = leakage::AUDIT_EXIT_CODE;

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
  rcsr --data <file|-> --target <col>[,<col>...] [options]
  rcsr import <input.csv> [<store.rcsb>] [--group-rows N] [--time <col>]
  rcsr prep --input <file|dir> --out <store.rcsb> [options]
  rcsr enrich --data <store.rcsb|CSV> --out <store.rcsb> --features <list> [options]
  rcsr walkforward --data <store.rcsb> --target <col> --expr <formula> ...
  rcsr audit --data <file> --target <col> [--time <col>] [--horizon N] [--json]
             [--lockbox <START,END|none>]
             [--target-spec <file>]
  rcsr verify-manifest <manifest.json> [--data <file>] [--binary <file>]
  rcsr dev-report <log.ndjson>
                              summarise the development log rcsr writes by
                              default (phases, fit/eval share, fallback
                              causes, bottlenecks)
  rcsr schema                 print a machine-readable description of this tool
  rcsr --help | --version

INPUT:
  --data <file|->            CSV/TSV/whitespace table, or a `.rcsb` column
                             store made by `rcsr import`; header row required
                             for CSV ('-' reads the table from standard input)
  --max-memory <size>        refuse when the estimated resident footprint is
                             larger than this (bytes, or KiB/MiB/GiB/GB; e.g.
                             512MB, 2GiB). Exit 3 with the estimate and an
                             actionable message instead of SIGABRT/SIGKILL.
                             The CSV reader streams, so the file text is not
                             resident. A `.rcsb` store is estimated exactly
                             from its header: resident dtypes (values only for
                             numeric columns, the i64 copy, 4-byte codes),
                             dictionaries, row groups, the used-column
                             projection and GPU/fit copies, plus 25% slack.
                             The full rows load whenever this fits; otherwise
                             the store is refused. The estimate and its basis
                             are reported in the preparation block.
  --force                    override --max-memory and load the full input
                             resident anyway (the kernel may still kill it)
  --sample-rows <n>          explicit hash sample of a `.rcsb` store (opt-in;
                             without this flag a store is never sampled).
                             Recorded as full_rows=false in the preparation
                             block, manifest and dev log. Mutually exclusive
                             with --force; n >= the row count is a full load
                             and must still fit the budget
  --target <col>[,<col>..]   target column; a comma-separated list enables
                             multi-output regression (one shared shape, one
                             fitted constant vector per output). See
                             docs/multi-output.md
  --vars <a,b,..>            input columns (default: all numeric columns except
                             the targets and any --time/--group-by column)
  --dtype <col>=<kind>,..    per-variable datatype: continuous (default),
                             boolean (0/1, never standardised) or categorical
                             (expanded to one-hot columns named col=value).
                             Example: --dtype flag=boolean,symbol=categorical
  --max-levels <N>           maximum distinct values in a categorical column
                             before one-hot expansion (default 16)
  --skip-malformed           drop data rows whose field count differs from the
                             header and report how many were skipped. Default:
                             refuse the file with an input error (exit 3) naming
                             the 1-based line and both field counts
  --test-frac <f>            held-out fraction (default 0.25)
  --validation-frac <f>      hold out the last <f> of the chronological training
                             period as a validation slice for selection
                             (0 = off). The test block is never used to choose
                             a candidate.
  --select-by train|validation|aicc
                             display order of the candidates (default validation
                             when a slice exists, otherwise train). The selected
                             member comes from the validation gates, not from
                             this order. test_error is scoring-only, never a
                             criterion.
  --objective mse|pnl|sharpe|log_growth
                             what finalist selection minimises on the
                             validation slice (default mse: the fit metric).
                             pnl, sharpe and log_growth score each Pareto
                             member's exact CPU f64 predictions as a long/flat
                             book (positions 0 or 1, never short) with cost per
                             side on |delta p| including the final exit; the
                             fit and, unless --fitness pnl, the archive use --metric. Needs a
                             temporal split with --validation-frac, a
                             forward_return --target-spec, and a cost (--cost-bps
                             or the spec cost_bps).
                             Rows of one group closer than the label horizon overlap
                             and are refused (exit 2); use long:<h>,stride=<h>
                             The test block is read once, for the winner only
  --fitness metric|pnl      what ranks candidates in the search (default metric:
                             --metric on the label). pnl ranks every candidate
                             by the exact long/flat net regret (B - net) / B on
                             the training rows, B the hindsight bound at the
                             same cost: lower is better, all-flat is 1.
                             Constants are still fitted with --metric (--weight
                             weights that fit only). Needs a trading --objective
                             and its refusals. Exit 2 with --predictor-samples,
                             --fit-samples, --implicit, --min-holding above 1,
                             --max-position other than 1, multiple targets,
                             overlapping rows, or no profitable path. On the
                             GPU the device scores the regret of its own f32
                             predictions, fused with the interpreter
                             (search_fitness.scorer_backend cuda); tiled
                             residency keeps the CPU. Reported values stay
                             exact CPU f64. Resume needs the same cost,
                             threshold and rows
  --pnl-blocks K          1 to 8 time blocks for --fitness pnl (default 1 =
                             off, the plain regret). The training timestamps
                             are cut into K equal runs, each block is its own
                             long/flat book, and the fitness is the mean block
                             regret plus --pnl-block-lambda times their
                             standard deviation. Needs --fitness pnl
  --pnl-block-lambda L      weight of the block spread (default 1.0, finite,
                             >= 0; 0 is the plain mean). Needs --fitness pnl
  --position long_only|top_tercile
                             long_only (default): 1[pred > --threshold]. top_tercile: per
                             timestamp the top floor(n/3) finite predictions are long (p=1),
                             ties by row order; fewer than 3 finite predictions hold nothing;
                             no threshold. Needs a --split temporal (whole timestamps on one
                             side), not --fitness pnl, --min-holding 1, --max-position 1.
                             Use it with an xs_rank target (rcsr panel rank). short is refused
  --allow-short              refused: there is no short P&L, borrow or funding
  --threshold <f>            long when the prediction is strictly above <f>
                             (default 0)
  --cost-bps <f>             cost per side in basis points (required with a P&L
                             objective unless the target spec declares cost_bps, which
                             gives cost_bps / 2 per side; a different value is
                             refused, exit 2; 0 only when stated)
  --cost-override <f>        per-side cost that differs from the target spec cost on
                             purpose; the objective block says cost.source = flag
  --slippage-bps <f>         extra cost per side in basis points (default 0)
  --cash-return <f>          return per row earned while flat (default 0)
  --funding <col>            per-row funding rate paid while long
  --max-position <f>         position cap in (0, 1] (default 1)
  --min-trades <n>           fewer trades on validation make a candidate
                             unusable (default 1)
  --min-holding <n>          rows a long position is held before an exit
                             (default 1)
  --require-validation       refuse (exit 2) when there is no validation slice
  --equivalence-z <z>        z for the validation gates: a member is supported
                             when it improves on the last supported entry by
                             more than z standard errors of the paired loss
                             differential (default 1.0, must be > 0)
  --noise-lag <n|auto>       Bartlett lag, in timestamps, of the Driscoll-Kraay
                             standard error (default auto: the larger of the
                             label horizon in steps minus one and the
                             Newey-West rule of thumb). Random splits use lag 0.
                             The gates run on time-ordered runs and when a
                             --target-spec is in force; see docs/overfitting.md

ROW WEIGHTS:
  --weight <col>             per-row weight column: finite, >= 0, at least one
                             positive. A weight of 2 means \"this row twice\":
                             absolute, squared, correlation, correlation_signed,
                             signed, log_squashed, log_loss, brier and the three
                             hinge forms, implicit and implicit_residual become
                             weighted means, and aic/aicc/bic use the weighted
                             RSS. A zero-weight row is skipped before its
                             prediction or label is read. Refused for metrics
                             with no weighted definition (median, maximum,
                             interquartile, slope, rank_correlation, auc).
                             On the GPU backend the device scorer carries the
                             weights for every metric listed above that has a
                             native scorer; a weighted run is never ranked on
                             unweighted loss.

DATA PREPARATION (all opt-in; defaults reproduce a plain mean/std run):
  See docs/time-series.md for which preparation steps read ahead.
  --fill <method>            what to do with a missing cell (empty, NA, NaN or
                             null). Default none: leave it, as before.
                             drop        drop the row
                             mean|median training-row column statistic
                             zero        literal 0
                             previous    last value above it in the same side
                             interpolate linear between the neighbours
  --outlier-iqr <k>          drop rows more than k interquartile ranges from the
                             column median (bounds from training rows only)
  --smooth <k>               k-wide TRAILING moving average per column (a row
                             reads only its own and earlier rows; edges shrink)
  --smooth-centred <k>       the old centred window. Reads future bars and is
                             a look-ahead on time series; reported as centred
  --allow-interpolate        permit --fill interpolate on time-ordered data
  --allow-lockbox            permit train/validation/test rows in 2026H2 (declared final test only)
                             It never opens the declared locked test (--lockbox).
  --lockbox-embargo <SECONDS>
                             widen the lookback embargo after the end of every locked range
                             (default: lockbox_embargo declared with the data, else 10972800);
                             a smaller value is refused (exit 2). Rows in
                             [START - purge, END + embargo) are refused (exit 3)
  --lockbox <START,END|none>
                             locked test (UTC, half-open). Order: this flag, the range
                             declared with the data (<data>.manifest.json, .schema.json,
                             .enrich.json or <csv>.lockbox.json), else the protocol
                             default 2025-07-01,2026-07-01. On undeclared data the flag adds to the default. A flag that does not cover
                             the declared range is refused (exit 2), as are conflicting
                             declarations (exit 3). `none` opts out and is refused when
                             the data declares a range. Train, validation and test rows
                             inside the range, or whose label reaches it (within the target
                             spec's required purge before the range; default 219600 s), are
                             refused (exit 3); run.json `lockbox`
                             reports rows, date range, read count, declared, source
                             (flag, store, protocol_default, none), declared_in,
                             rows_in_scope, evaluated
                             With a declaration every split side is checked: without
                             --time the data must hold one integer epoch-seconds
                             column, else the run exits 3 asking for --time; a
                             time column that is not epoch seconds, or a row without
                             a time stamp, is exit 3 too
  --ledger <file>            candidate ledger whose locked reads run.json counts
  --arm <name>              ablation arm label; run.json gains an `ablation` object with the arm,
                             the SHA-256 of the ordered variable list, the generation budget and
                             the generations reached
  --target-spec <file>       declared target spec (rcsr-target-spec/1 JSON; default
                             <data stem>.targets.json when it has an entry for the
                             target). The engine then: refuses a label whose
                             prediction at zero return depends on a column
                             (D13c, exit 5), the future return as an input (D13a)
                             and a look-ahead scale (D13b); purges training rows
                             within the label end offset of every validation and
                             test row (needs an epoch-seconds --time); prints the
                             zero-parameter rung baselines.target_structure
  --allow-target-structure   run a D13c target anyway for a diagnostic replay: warns,
                             records check.status=overridden and a proxy table.
                             D13a and D13b have no override
  --require-clean-inputs   refuse (exit 5) a run whose inputs include a construction column of the
                             target spec (for example the vol96 scale of lbl_*_y); without the flag
                             the column is only recorded in check.construction_in_vars
                             (it reads the next bar; otherwise refused)
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
  See docs/time-series.md for the full purge/embargo/look-ahead checklist.
  --split random|temporal|group|temporal-group|auto   (default random)
                             random          shuffled i.i.d. holdout
                             temporal        train on the past, test on the future
                             group           hold out whole groups (unseen stocks,
                                             subjects, sensors), never both sides
                             temporal-group  a temporal split inside every group
                             auto            infer from the columns and report
  --time <col>               time/ordering column; implies temporal. Accepts
                             epoch numbers or ISO 8601 with a Z / +-HH:MM offset
  --date-only <policy>       date-only time values: error (default, refuses an
                             under-specified instant), utc-midnight or
                             session-close (21:00 UTC)
  --time-unit s|ms|us|ns     unit of a numeric time column (default: detect;
                             a mixed or out-of-range column is refused)
  --tz <offset>              fixed offset for timestamps without one (UTC or
                             +-HH:MM; named zones need a tz database and are
                             refused rather than guessed)
  --group-by <col>           group column; implies group (with --time: temporal
                             inside each group). Text columns are allowed.
  --project <n|auto>         choose the variables as a deterministic family-stratified
                             view of n registered feature columns (24-48, research
                             cell 96-128; auto = 36), exactly n columns, one per family per
                             round, rotated by --seed and --view; a store with fewer than n
                             candidate columns exits 2. Needs a .rcsb store; not with --vars.
                             The view id and columns go to --run-manifest
  --view <k>                 view index for --project (default 0)
  --no-project               load every store column. The default projects the
                             read to --vars, targets, --weight, --time and
                             --group-by when a .rcsb store is read with --vars
                             and --time; the columns read by
                             --filter, --exclude, --train-select and --test-select
                             are added. --no-project is the A/B escape hatch.
  --gap <rows|time>          embargo: rows dropped between train and test to stop
                             autocorrelation leaking the future (default 0).
                             A suffixed value (90m, 1.5h, 30s, 2d) is wall time,
                             converted with ceil(seconds / median_dt); it is
                             dropped with a warning under explicit row selection
  --folds <k>                split the test region into k contiguous blocks and
                             report per-block error as well as the mean (default 1)

HISTORY AND DELAYED VARIABLES (lags, for forecasting):
  --lags <k>                 add x_lag1..x_lagk as inputs per variable, computed
                             inside each series (group + time order + gaps)
  --lag-set <k,..|default>  declared lag menu in bars, e.g. 1,4,24,96 (`default`
                             is that menu): adds x_lag<k> only for the listed k. Instead of
                             --lags; the run reports nvars before and after
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
  --baseline-samples <N>      score at most N query rows for the k-NN baseline
                              (any metric; default: every row up to 20000
                              training rows, else a seeded sample of 512;
                              0 = every row). The reference set is always
                              all training rows; sampled results report their
                              standard error with the finite-population
                              correction (#69). The exact every-row query is
                              O(train^2*d), so 0 is opt-in on large panels
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
                             Folds are drawn from the rows left by --exclude and
                             --filter. With a declared locked test every such row
                             is checked, also with --allow-lockbox.
  --cv-k <K>                 number of folds (default 5); --cv costs ~K searches
  --null-runs <n>            rerun the identical search on n shuffled targets
                             (y-randomisation control) and report the null error
                             distribution and a p-value; inferential, costs n
                             extra searches
  --null-block <N>           permute contiguous blocks of N rows instead of
                             single rows (N >= 2, and at least the label
                             horizon). For autocorrelated series the i.i.d.
                             shuffle is anti-conservative; the report records
                             method, block length and number of blocks. With no
                             --null-runs it runs 19 permutations
  --bootstrap-error <n>      refit each solution's constants on n bootstrap
                             resamples and report the apparent, out-of-bag and
                             .632+ error estimates (Efron & Tibshirani 1997);
                             costs n constant refits per reported solution.
                             Above n^0.7 >= 1000 rows the draw is a bounded BLB
                             subsample of b = n^0.7 rows (Kleiner et al. 2014)
                             and the resample size is reported
  --stability <B>            bootstrap the training rows B times, re-run the
                             identical search and rank the structural forms and
                             input variables by how often they are selected;
                             costs B extra searches. The same BLB cap and
                             resample-size report apply

MODE:
  --eval-expr <expr>         do not search: parse this formula, score it on the
                             split, print the result and exit. Write variables
                             by name in original units, e.g. \"2.5*x + sin(y)\";
                             operators as in --ops. `x0,x1,..` also work.
  --function <def>           define a reusable building block (repeatable, up to
                             four): \"name(a,b)=sin(a)*cos(b)\" or
                             \"name=<expr>\" (parameters are the free variables
                             in order of first appearance). Blocks are searched
                             as operators of their own arity, named in output,
                             and cost 1 by default; set a cost with
                             --op-complexity. Constants inside a body are shared
                             by every call. See docs/building-blocks.md.

LIMITS:
  Every expression read from text (--eval-expr, --filter, --exclude,
  --train-select, --test-select, --seed-expr, --seed-file, --function and the
  HTTP body) and every tree decoded from a checkpoint is bounded to 4096 nodes
  and a root-to-leaf depth of 256 (a single node has depth 1). A deeper or
  larger tree is a normal usage/input error with a parser column or a
  checkpoint message; it is never a stack overflow or abort.

MULTI-OUTPUT (--target y1,y2):
  Candidates share one tree shape; each target gets its own fitted constants.
  The combined error is the mean of the per-target errors, and the Pareto front
  ranks on that mean and the shape's complexity. Text prints one line per
  target; --json carries a targets array with each expression, constants and
  error plus the combined error. k_constants is the total across outputs.
  --ndjson emits one closing done event with the timing ledger. Not available
  with --cv, --null-runs, --bootstrap-error, --stability, --report,
  --save/--resume, or seed expressions.

SEARCH:
  --metric <name>            error to minimise, lower is better (default absolute)
                             absolute  squared  correlation  correlation_signed
                             median  maximum  signed  log_squashed
                             rank_correlation  rank_correlation_signed
                             interquartile  slope  auc  log_loss  brier  hinge
                             hinge_ordinary  hinge_squared  aic  aicc  bic
                             implicit  implicit_residual  (implicit model)
                             implicit_residual is the absolute metric on the all-zero
                             implicit target: it is scored and fitted on the GPU
                             like absolute
                             correlation/rank_correlation score 1-|r| and accept an
                             inverted model; the _signed variants score 1-r and
                             require the right direction. auc/log_loss/brier and
                             the hinge forms treat the target as two classes and
                             reject targets with more than two distinct values.
                             hinge is the historical bounded sigmoid surrogate
                             (2*sigma(-t*p)); hinge_ordinary is max(0,1-t*p) and
                             hinge_squared is its square. signed scores the
                             absolute mean residual |mean(y-f)| (a weighted mean
                             with --weight); it has a documented degeneracy: its
                             optimum is a whole manifold at exactly 0.
                             slope scores the mean absolute error of first
                             differences over row pairs that are adjacent in the
                             data order and in the same group (--group-by, or
                             one group), whatever the split shuffles; fewer
                             than 32 such training pairs is a usage error that
                             points at --split temporal. The GPU scores and
                             fits slope only when those pairs are the
                             consecutive training rows (a single series under
                             --split temporal); otherwise the CPU scores it and
                             backend_reason names the cause.
  --implicit                 fit an implicit relation f(inputs, target) = 0: the
                             target becomes one extra input (normalised with its
                             training mean/std) and the fit target is 0. Plain
                             metrics then score |f|; --smooth first is recommended
  --implicit-derivative      shorthand for --implicit --metric implicit: score
                             the level-set distance |f| / ||grad f||, the
                             recommended implicit objective
  --generations <N>          (default 200)
  --islands <I>              (default 1)
  --pop <P>                  population per island (default 200)
  --max-complexity <K>       max configured complexity per expression (default 30)
  --op-complexity <spec>     per-operator costs, e.g. \"pow=2,sin=1.5,if=3\";
                             operators not listed keep the default cost of 1.
                             A block costs 1 unless set here, by its --function
                             name or its custom0..custom3 slot
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
                             aggregates: mean,std,median,iqr,sum,amin,amax,
                             count (training-column statistics; min(x1) and
                             max(x1) with one argument also work)
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
  --fit-iters <N>            Levenberg–Marquardt iterations for constants
                             (default 8; 0 = no fitting at all, for every
                             metric and both fit strategies). Constants are
                             fitted to the configured --metric: squared/aic/
                             aicc/bic use exact least squares; absolute and
                             log_squashed use IRLS; auc/log_loss/brier/hinge use
                             logistic IRLS (a warm start judged by the configured
                             metric); hinge_ordinary adds majorise-minimise
                             rounds after the logistic rounds and hinge_squared
                             uses Gauss-Newton on the active set; signed uses the
                             squared fit plus a
                             constrained least-squares projection onto the
                             zero-mean-residual manifold (exact to f64, two
                             passes, no optimiser); correlation/correlation_signed
                             use the exact analytic rho gradient around a frozen
                             affine gauge; rank/order losses use a deterministic
                             coordinate polish (median, interquartile and
                             maximum on the device with --fit-strategy gpu,
                             #226/#232/#227)
  --fit-strategy <name>      exact | auto | linear | varpro | stream | gpu (default
                             gpu where admissible, else exact). With no flag, or
                             `auto`, the engine resolves the arm once: `gpu`
                             admitting every candidate with constants for
                             squared/aic/aicc/bic with fitting on, the GPU
                             evaluator, device memory for the f64 fit rows and
                             a run-level cost model that predicts a win; else
                             `exact` (no flag) or the structural `auto`. Other
                             resolutions are labelled with their reason on
                             stderr, in --json (GPU runs), --run-manifest and
                             the development log. A resume keeps the saved arm.
                             `exact` is the historical CPU
                             Levenberg–Marquardt path, bit for bit. `auto`
                             dispatches by structure: closed-form QR for trees
                             affine in every constant, variable projection for
                             separable models, exact LM otherwise. `linear`,
                             `varpro` and `stream` force one arm and fall back
                             to exact LM when it cannot prove a safe result;
                             `stream` merges f64 sufficient statistics over row
                             blocks. `gpu` is the #169 batched kernel with a CPU
                             fallback; it fits trees with at most 20 constants
                             on the device (p<=16 uses a 256-thread block,
                             p=17..20 a 128-thread one), screens unfitted
                             offspring, admits the
                             best max(32,pop/4) per island below 1 048 576
                             training rows and every candidate with constants
                             at or above that size, then refits finalists
                             exactly. It minimises the configured objective:
                             squared/AIC least squares, log_squashed with
                             per-row IRLS weights (#229), absolute (mean |r|)
                             with per-row Lawson weights and true-MAE
                             acceptance (#222), the logistic log-loss on the
                             binary label with per-row Fisher weights q(1-q) and
                             a softplus trial objective (#234; the device
                             proposes constants and the exact CPU f64 log-loss
                             judges and refits; since #289 the search also
                             scores log_loss natively on the device), the three
                             hinge forms on the same label sign (hinge_squared
                             by Gauss-Newton on the active set, hinge_ordinary
                             by logistic then majorise-minimise steps, hinge by
                             the logistic step; every trial judged on the
                             configured hinge, finalists refit exactly, a
                             single-class sample labelled single_class), slope
                             by IRLS on the first differences of consecutive
                             rows from the squared warm start, keeping the
                             incoming constants unless the slope improves and
                             the additive level at the squared fit (#236), the exact correlation
                             objective 1-|rho| or 1-rho with a frozen affine
                             gauge (correlation_signed refines both gauge
                             orientations; a tree affine in every constant takes
                             the CPU least-squares closed form, labelled
                             affine_closed_form; every proposal passes the
                             exact f64 gate), or - for
                             rank_correlation / rank_correlation_signed — the
                             exact integer rank polish after a squared LM start
                             (#230), with rank_correlation_signed scored as
                             1-rho so a reflected model is rejected and nothing
                             is judged on SSE or |rho| (#231);
                             for median, the exact f64 median polish (segmented
                             sort of the residual bits, first-improver ladder,
                             #226; only for >= 4096 training rows, the measured
                             crossover) and for interquartile the same sort
                             averaging ranks [n/4, 3n/4) of |r| (#232; same row
                             floor); for maximum (Chebyshev), the exact
                             one-dimensional minimiser per constant by
                             bisection on the error level (at most 32 rounds,
                             no step ladder and no optimiser, #227; from
                             >= 512 training rows, the measured cost-model
                             floor). median, interquartile, rank_correlation(_signed)
                             and auc also score every candidate natively from
                             4096 rows (#288) and maximum through the #287
                             objective kernel, so the fitness is the
                             configured objective, never a proxy. Without a batch backend it uses the exact
                             CPU arms (never a no-op).
                             The device refuses any other metric instead of
                             substituting an objective: use
                             `--metric squared`, `--metric log_squashed`,
                             `--metric absolute`, `--metric log_loss`, a hinge
                             form, `--metric correlation`, `--metric
                             correlation_signed`, a rank
                             metric, `--metric median`, `--metric
                             interquartile`, `--metric maximum`, or a CPU
                             strategy. `--metric signed` fits on the device as
                             the squared warm start of the exact f64 bias
                             projection; every proposal passes the
                             exact f64 gate and its score is native on the
                             device
  --fit-report               print the constant-fitting arm counts (linear,
                             varpro, stream, exact, GPU fallback/refit, signed
                             projection labels) at the end of the run
                             (diagnostics only)
  --fit-rows <N>             fit constants on a deterministic N-row subsample
                             and score/archive on all training rows (default 0
                             = off; N >= train rows is an exact full fit). The
                             archive and every reported error are refit on all
                             rows. With the GPU evaluator and --fit-strategy gpu
                             the subsample is fixed for the run and lives on the
                             device; the device still scores every row
  --fit-fallback-max <N>     hard per-generation bound on exact CPU fallback
                             refits for --fit-strategy gpu (default 4096,
                             0 = unbounded). Fallbacks beyond the bound keep
                             their finite device constants and are counted
  --fit-polish <f64|f32|off>  arithmetic of the device fit kernel's polish, the
                             undamped Gauss-Newton steps after the damped LM
                             (default f64). f64 re-baselines and steps on the
                             exact f64 rows; f32 steps on the f32 rows of the LM
                             phase; off returns after the LM phase. The search
                             ranks on the f32 evaluator and the archive is
                             refit exactly on the CPU in every mode. Only the
                             device fit arm reads it; a resume refuses a
                             different saved value
  --refit-band <F>           after a device-fit search, refit exactly (CPU f64)
                             only the trees whose full-data error is within
                             this relative band of the lowest error at no
                             greater cost, then every reported front member
                             still unrefit (default 0.001). inf refits every
                             unique tree
  --no-fit                   disable constant fitting entirely (= --fit-iters 0)
  --no-dedup                 disable structural de-duplication
  --selection lexicographic|afp
                             survivor and parent selection inside an island.
                             lexicographic (default) replaces the island with
                             its offspring and picks parents by error, then
                             size. afp keeps the Pareto-best of parents and
                             offspring on (error, complexity, age), adds a few
                             random immigrants per generation and picks
                             parents by Pareto rank and crowding distance.
                             Node path only; refuses --variation postfix
  --linear-scaling           wrap every candidate shape f in a fitted affine output
                             map a + b*f(x), so a shape that is right up to scale
                             and offset scores as right at once. Needs a
                             least-squares metric (squared, aic, aicc, bic) and
                             constant fitting; single target; off by default (#470)
  --mutation-mix <M>         node-path edit weights: classic (point .25, terminal
                             .35, subtree .40; default) or extended (adds insert,
                             delete, swap and rotate). Not with --variation
                             postfix (#471)
  --tree-arity <A>           operator draw in random trees: uniform (default) or
                             balanced (binary .5, unary .4, ternary .1). Not with
                             --variation postfix (#471)
  --crossover-internal <P>   probability that a crossover cut point is an
                             internal node (default 0 = uniform cut). Not with
                             --variation postfix (#471)
  --adaptive-mix             per-island edit weights learned from the success
                             rate of the last 10 generations (uses all seven edit
                             kinds; learns from CPU-scored generations only).
                             Off by default; not with --variation postfix (#471)
  --no-front-parents         draw every parent from the island tournament;
                             turns off the 15% Pareto-front feedback (#76)
  --reject-constant-offspring
                             reject offspring and random trees with no input
                             outside an aggregate before the constant fit; the
                             first constant of an island is kept (#477). Saved
                             search setting, off by default
                             and restores the pre-#76 trajectory exactly
  --prune                    remove neutral code from the final front (off by
                             default): replace subtrees with a child or a
                             constant when the training error stays within
                             1e-6 (1 + |e|) of the original (#466)
  --no-prune                 keep the prune off (the default)
  --prune-rows <N>           rows the prune fits on (default 20000, >= 2); a
                             larger training sample is subsampled
  --prune-budget <N>         cap on candidate fits per run (default 20000,
                             0 = no prune)
  --no-simd                  force the scalar tree walk instead of the
                             SoA/AVX evaluator (same results, slower; an
                             escape hatch for debugging and A/B runs)
  --predictor-samples <N>    score and fit candidates on N training rows per
                             generation (default: all rows; N >= 2). A request
                             above 3/5 of the training size is widened to all
                             rows (rebuild and refit cost more than it saves);
                             N >= train rows is an exact no-op. Carried
                             individuals are refitted on the same subset before
                             they are scored. The archive and every reported
                             error are always full-data
  --fit-samples <N>          screen constants on N training rows per generation
                             (default: 0 = off; N >= 2). All offspring are
                             fitted and scored on the subset; the top 1/16 are
                             refit and scored on all training rows before
                             selection and archiving. Independent of
                             --predictor-samples
  --patience <N>             stop after N generations without improvement (default 40)
  --patience-tol <r>         count a generation as no improvement when the best
                             error fell by at most r relative to itself (scale
                             free). Default: the 1e-9 rule with a 1+|error|
                             denominator
  --stop-on front[:<W>]      stop after W generations (default: --patience) in
                             which no front member beat the previous front by
                             more than --patience-tol (default 1e-6) and the
                             front hypervolume did not rise by that much;
                             stop_reason front_stable. Off by default
  --time-budget <seconds>    wall-clock limit of this process (default: none).
                             The generation in progress finishes, its progress
                             event and checkpoint are written, and the result
                             reports stop_reason time_budget with
                             converged_early false. The checkpoint is not
                             terminal: --resume runs more generations and may
                             set a new --time-budget
  --gpu                      assert that the GPU must guide the search: an
                             error (exit 2) when the binary was not built with
                             --features cuda or no device is present. Without
                             --gpu/--cpu, a CUDA build with a usable device uses
                             the GPU automatically when the #287 score table
                             names a native scorer: squared, absolute,
                             log_squashed, signed, maximum, correlation,
                             correlation_signed, aic/aicc/bic, the
                             classification family and slope (when the training
                             pairs are its consecutive rows, #236) through the
                             objective kernel, and median, interquartile,
                             rank_correlation, rank_correlation_signed and auc
                             through the exact f64 order-statistic/rank batch
                             scorer (#288) from 4096 rows (a two-valued target
                             for AUC); those statistics are bit-identical to
                             the CPU for the same prediction vector (median and
                             interquartile use a radix select from 4096 and
                             65536 rows, rank and auc a sort). --fitness pnl
                             is scored by the long/flat regret kernel (#481),
                             bit-identical to the CPU book on the device's
                             predictions; --metric then only fits. Every other
                             metric is *scored* on the CPU with its named
                             refusal, never with a proxy loss; with
                             --fit-strategy gpu its *constants* may still be
                             fitted on the device (the rank, median,
                             interquartile, maximum and log_loss arms).
                             Reported errors stay CPU f64. Cross-backend runs
                             are not bit-identical because f32 fitness can
                             change the trajectory. JSON reports the measured
                             kernel/sync/alloc/copy wall, the evaluator-window
                             kernel duty (eval_kernel_duty, with eval_window_s),
                             device_seconds_incl_fit, device_fraction_of_run and
                             eval_fraction_of_run (divided by
                             timing.total_seconds, the one runtime of the
                             process), batches_per_s (evaluator batches per
                             evaluator second, not generations), and the #284
                             run ledger: exclusive
                             stage seconds (vary/eval/fit/select/archive/
                             features/pnl/io/host_other), device-busy as the
                             union of device intervals over run wall, fit kernel
                             and CPU-fallback seconds with causes, and
                             numerator/denominator/units/scope for every derived
                             ratio. SM occupancy is null unless measured
                             separately with ncu. The packed batch arena
                             allocates once, keeps a <=64 MiB pinned staging
                             ring and copies on separate streams; set
                             RCSR_GPU_SYNC=1 to force the labelled blocking
                             fallback (#285)
  --gpu-memory <size>        cap the device memory the run may plan on (bytes, or
                             KiB/MiB/GiB/GB; default min(10.5 GiB, free - 1 GiB)).
                             The plan runs before the first allocation. When
                             the rows do not fit, --gpu exits 2 naming the
                             stage, the bytes needed, the budget and the
                             smallest fix (--vars, rows, --fit-strategy exact,
                             --cpu); without --gpu the run keeps the CPU with
                             backend_reason starting device_memory. An explicit
                             --fit-strategy gpu whose f64 fit rows do not fit
                             exits 2. When only the f32 rows do not fit, the
                             rows stream from the host in tiles (scores are
                             unchanged); the default fit and the exact rank and
                             order scorers then run on the CPU and an explicit
                             --fit-strategy gpu exits 2. JSON reports
                             gpu.memory_plan
  --dispatch <auto|device>   how the per-stage cost model (#290) routes the movable
                             stages (default auto). device skips the timing
                             comparison only and sends every stage that is not
                             refused (launch, unsupported_plan, too_few_rows,
                             too_many_constants, ill_conditioned, non_finite,
                             metric_not_native, device_memory, no_device,
                             exactness) and not CPU by contract to the device;
                             the route ledger records reason forced. Needs
                             --gpu (exit 2 otherwise). Required for
                             compute-sanitizer runs: the sanitized device is
                             slow, so auto keeps every stage on the CPU and no
                             kernel launches. With auto, a CPU run whose stages
                             all lost the timing comparison reports
                             backend_reason `dispatch: cost model kept
                             every stage on the cpu`
  --no-gpu                   force CPU evaluation (alias --cpu); also the
                             reproducibility escape hatch for a byte-identical
                             CPU trajectory
  --qd                       record an opt-in quality-diversity archive over
                             (variable usage x prediction skewness), passive;
                             it never changes parents or the Pareto front.
                             Reports reachable-support coverage, occupancy
                             entropy and QD-score; every run also reports a
                             fixed-reference validation hypervolume
  --qd-cells <N>             descriptor cells (grid bins per axis or CVT K;
                             2..=512; enables --qd)
  --qd-cell-cap <N>          maximum local Pareto front per descriptor cell (default 4)
  --qd-cvt                   use CVT-MAP-Elites (K centroids, Lloyd updates)
                             instead of the rectangular grid
  --qd-policy                feed the archive back into parent selection (UCB1
                             60/25/10/5), stratified migration and least-
                             occupied restarts; implies --qd --qd-cvt
  --qd-uniform               control: the policy selects occupied cells
                             uniformly instead of the UCB1 mix
  --qd-random                control: hash descriptors to random cells, so the
                             partition carries no behaviour information
  --threads <N>              worker threads for constant fitting and scoring of
                             offspring, inside each island (default: the CPU
                             cores available to the process; 1 = serial). The
                             islands share the budget: N / islands per island.
                             The exact gate of gated device fits (log_loss,
                             correlation, signed) uses all N. Results, archive
                             and checkpoints are identical for every value
  --pipeline-depth <N>       overlap CPU generation with GPU scoring using a
                             bounded pending queue of N batches (default 2; 0 =
                             synchronous). Results and archive are identical;
                             only effective with --gpu
                             Programs with no device opcode (constraints,
                             aggregates, blocks the lowerer cannot rewrite) are
                             scored by a bounded CPU worker pool, never on the
                             scoring thread; RCSR_CPU_FALLBACK_THREADS sets its
                             size (default min(cores, 8)). --json reports each
                             generation's native/lowered/unsupported counts and
                             fallback seconds
  --variation auto|node|postfix
                             variation engine. auto (default) uses the flat
                             postfix arena for --no-fit GPU runs with a
                             population >= 4096; node keeps the owned-tree loop;
                             postfix requires the flat path and refuses a run
                             that does not qualify

CHECKPOINTS:
  --save <file>              write a checkpoint on exit and every --save-every
                             generations; saving does not change the search.
                             Writes are atomic and the previous good generation
                             is kept as <file>.prev
  --save-every <N>           generation interval for --save (default 25). A
                             checkpoint is also written when 15 minutes have
                             passed since the last one, so slow generations
                             are never 25 generations of lost work. The exit
                             checkpoint carries the exact f64 post-refit
                             archive (final), its archive_hash and an
                             error_kind label for the search state
  --resume <file>            continue from a checkpoint. The saved search
                             settings win; --generations may only be raised.
                             Training and test data must match the saved
                             fingerprint; a mismatch is refused (exit 3).
                             The target spec sha256 must match too (exit 2).
                             The saved backend (cpu/gpu) must also match; a
                             mismatch is refused unless overridden below
  --allow-backend-change     resume a checkpoint on a different backend
                             (cpu<->gpu) and say so on stderr and in the
                             run manifest
  --allow-build-change       resume a checkpoint written by a different
                             binary, commit, version, device, driver or
                             thread count (#256). Without it the mismatch
                             is refused (exit 3). A time-budget stop is not
                             terminal: --resume continues it and
                             --time-budget may be set again

RUN INTEGRITY (#140):
  --run-manifest <file>      atomically write a JSON manifest with the build
                             commit (recorded when the binary was built, not by
                             a runtime git call), binary SHA-256, data checksum
                             and fingerprint, resolved config, seed, backend,
                             device/driver/CUDA, UTC start/end and phase timings;
                             update it atomically at the end. start_utc is the
                             process start, end_utc the end of the process work
                             and phases_seconds.total equals
                             timing.total_seconds in the result JSON
  verify-manifest <file>     check a manifest against the current binary (or the
  [--data <file>]            file named by --binary, for a preserved binary),
  [--binary <file>]          data file and canonical config; exit 1 on
                             any mismatch (exit 0 when everything matches)

DISPATCH (#290):
  Dispatch is planned per stage, not per run. A short startup probe times the
  host; with --gpu one device probe per process times the eval rate, fit rate,
  H2D/D2H bandwidth and launch overhead, and without a device the corpus
  anchors stay (device_source anchor). --json carries dispatch.calibration
  labels and dispatch.route_ledger; --run-manifest adds the measured rates and
  their spreads. The route is device, cpu or
  split with a machine reason; hysteresis (h=0.3) prevents oscillation at the
  crossover. A structural/arithmetic refusal (launch, unsupported_plan,
  too_few_rows, too_many_constants, ill_conditioned, non_finite,
  metric_not_native, device_memory, exactness) is never chosen by a timing test. Reported
  numbers and finalists are still the exact CPU f64 rebuild.

DEVELOPMENT LOG (#211) — on by default:
  The engine writes rcsr.dev.ndjson next to --json-out (or --save), else in
  the working directory. It is the in-engine replacement for the old external
  run monitor; --no-dev-log turns it off. Independent of --json/--ndjson/
  --progress. Every line carries `schema: rcsr-devlog/1` and no raw rows.
  --dev-log-level <level>    summary (default) or full.
                             summary: run_start, one event every 10-25
                             generations (gen, elapsed, best/median error,
                             Pareto size, phase times, fit/GPU/fallback counts,
                             GPU duty, convergence), checkpoint events and
                             run_end
                             full: every generation with the complete record
                             (all error quantiles, archive hypervolume/hash,
                             dedup rate, admissions, fit fallback-by-cause,
                             kernel/wall/duty, QD coverage/entropy, per-island
                             best, RNG key)
  --dev-log <path>           write to this path instead of the default one
  --no-dev-log               disable the log entirely
  --dev-log-every <N>        write a generation event every Nth generation
                             (default: level-dependent — 1 for full; for
                             summary clamp(generations/10, 10, 25), and every
                             generation when the budget is <= 10)
  --dev-log-max-bytes <size> cap on generation and checkpoint events (default
                             64MiB); run_start and run_end are always written
  Writes are one atomic line each, so a killed run leaves a valid prefix, and
  logging never changes the archive, the results or the configuration hash.
  Budget: ~50 ms once per run plus 7-16 us per written event. See
  docs/dev-log.md and docs/run-integrity.md
  dev-report <log>           Rust analyzer for a dev log: phase table, fit vs
                             evaluation share, fallback causes, convergence,
                             archive growth and a ranked bottleneck list

OUTPUT:
  --format text|json|ndjson  (default text)
  --json                     shorthand for --format json
  --ndjson                   shorthand for --format ndjson
  --progress                 emit per-generation progress, including
                             convergence statistics (confidence, stability,
                             maturity, percent_converged); stderr, or stdout in
                             ndjson. See docs/convergence.md
  --top <N>                  report at most N Pareto rows (default: all)
  --quiet                    suppress the human-readable header, the GPU
                             fallback note and the one-line resource summary
  --out <file>               also write the Pareto front as TSV (atomic)
  --json-out <file>          write the final JSON result to this file atomically
                             instead of streaming it to stdout (implies --json).
                             Progress still goes to stderr, or NDJSON on stdout

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


fn expand_block_aliases(spec: &str, blocks: &[expr::Block]) -> String {
    spec.split(',')
        .map(|token| match token.split_once('=') {
            Some((name, cost)) => match blocks.iter().position(|b| b.name == name.trim()) {
                Some(slot) => format!("{}={}", Op::Custom(slot).symbol(), cost),
                None => token.to_string(),
            },
            None => token.to_string(),
        })
        .collect::<Vec<_>>()
        .join(",")
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

fn parse_trade_number(flag: &str, text: &str) -> f64 {
    match text.trim().parse::<f64>() {
        Ok(v) if v.is_finite() => v,
        _ => {
            eprintln!("error: {} needs a finite number, got '{}'", flag, text);
            std::process::exit(EXIT_USAGE);
        }
    }
}

/// #263: a usage error for the P&L objective, exit 2.
/// #493: the book of one prediction vector. The context rows (removed only by a
/// missing label) join the ranks of a top-tercile book; they earn nothing
/// without a return.
fn trading_book(
    tree: &expr::Node,
    x: &[Vec<f64>],
    slice: &data::TradingSlice,
    ctx: Option<&data::TradingContext>,
    tspec: &rcsr::trading::TradingSpec,
) -> Result<rcsr::trading::PathStats, String> {
    let mut pred: Vec<f64> = x.iter().map(|row| expr::eval(tree, row)).collect();
    let Some(c) = ctx else {
        return rcsr::trading::evaluate(&pred, &slice.ret, &slice.rows, tspec);
    };
    pred.extend(c.x.iter().map(|row| expr::eval(tree, row)));
    let mut ret = slice.ret.clone();
    ret.extend_from_slice(&c.slice.ret);
    let mut rows = slice.rows.clone();
    rows.time.extend_from_slice(&c.slice.rows.time);
    rows.group.extend_from_slice(&c.slice.rows.group);
    rows.funding.extend_from_slice(&c.slice.rows.funding);
    rcsr::trading::evaluate(&pred, &ret, &rows, tspec)
}

fn trading_usage(msg: &str) -> ! {
    eprintln!("error: {}", msg);
    std::process::exit(EXIT_USAGE);
}

/// #480: the `"fitness"` member of a Pareto entry, in exact CPU `f64` on the
/// training rows. `train_net` is the excess net over cash.
fn fitness_member_json(book: &rcsr::trading::FitnessBook, pred: &[f64]) -> String {
    if book.blocks() > 1 {
        // #484: `train_regret` is the search fitness (mean + lambda * sd of the
        // block regrets); net, flips and exits sum the blocks.
        let blocks = book.block_totals(pred);
        let mut net = 0.0;
        let (mut flips, mut exits, mut longs) = (0u64, 0u64, 0u64);
        for t in &blocks {
            net += book.net_of(t);
            flips += t.flips;
            exits += t.exits;
            longs += t.long_rows;
        }
        let regrets: Vec<String> = book.block_regrets(&blocks).into_iter().map(jnum).collect();
        return format!(
            ",\"fitness\":{{\"train_regret\":{},\"train_net\":{},\"train_flips\":{},\"train_exits\":{},\"train_exposure\":{},\"block_regret\":[{}]}}",
            jnum(book.fitness_of(&blocks)),
            jnum(net),
            flips,
            exits,
            jnum(longs as f64 / book.len().max(1) as f64),
            regrets.join(",")
        );
    }
    let t = book.totals(pred);
    format!(
        ",\"fitness\":{{\"train_regret\":{},\"train_net\":{},\"train_flips\":{},\"train_exits\":{},\"train_exposure\":{}}}",
        jnum(book.regret_of(&t)),
        jnum(book.net_of(&t)),
        t.flips,
        t.exits,
        jnum(t.long_rows as f64 / book.len().max(1) as f64)
    )
}

/// JSON number, with non-finite values rendered as `null` (valid JSON).
fn jnum(v: f64) -> String {
    if v.is_finite() {
        format!("{}", v)
    } else {
        "null".to_string()
    }
}

/// #482: `split.train_time` and `split.validation_time` as `[first, last]` raw
/// time keys, with a leading comma. Empty without a time column.
fn split_time_json(keys: Option<&data::RowKeys>) -> String {
    let range = |v: &[f64]| -> Option<(f64, f64)> {
        let finite = || v.iter().copied().filter(|t| t.is_finite());
        let lo = finite().fold(f64::INFINITY, f64::min);
        let hi = finite().fold(f64::NEG_INFINITY, f64::max);
        (lo <= hi).then_some((lo, hi))
    };
    let show = |r: Option<(f64, f64)>| match r {
        Some((a, b)) => format!("[{},{}]", jnum(a), jnum(b)),
        None => "null".to_string(),
    };
    match keys {
        Some(k) => format!(
            ",\"train_time\":{},\"validation_time\":{}",
            show(range(&k.train)),
            show(k.validation.as_deref().and_then(range))
        ),
        None => String::new(),
    }
}

/// #118: best reported error and constant-mean baseline error, the two inputs
/// to quality-per-GPU-hour. Both are on the same full-data f64 scale as the
/// archive. `inf`/`nan` yields `quality_per_gpu_hour = null` rather than a
/// fabricated number.
fn gpu_best_error(outcome: &rcsr::search::Outcome) -> f64 {
    outcome
        .archive
        .items
        .iter()
        .map(|s| s.error)
        .fold(f64::INFINITY, f64::min)
}

fn gpu_baseline_error(train: &rcsr::data::Sample) -> f64 {
    let n = train.len();
    if n == 0 {
        return f64::NAN;
    }
    let mean = train.y.iter().sum::<f64>() / n as f64;
    // #256/#284: `gpu_best_error` is a mean (`SSE / n` for squared); the
    // baseline must therefore be the mean too. Dividing an SSE by an MSE was
    // the 636.9x unit bug.
    train.y.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n as f64
}

/// #284: a derived value plus the four facts that make it auditable. `null`
/// when the ratio is undefined, never a fabricated zero.
fn ratio_json(r: Option<&rcsr::gpu_stats::Ratio>) -> String {
    match r {
        Some(r) if r.value.is_finite() => format!(
            "{{\"value\":{},\"numerator\":{},\"denominator\":{},\"units\":{},\"scope\":{}}}",
            jnum(r.value),
            jnum(r.numerator),
            jnum(r.denominator),
            jstr(r.units),
            jstr(r.scope)
        ),
        Some(r) => format!(
            "{{\"value\":null,\"numerator\":{},\"denominator\":{},\"units\":{},\"scope\":{}}}",
            jnum(r.numerator),
            jnum(r.denominator),
            jstr(r.units),
            jstr(r.scope)
        ),
        None => "null".to_string(),
    }
}

/// #284: the run-scoped ledger, with every derived field carrying its
/// numerator, denominator, units and scope.
fn ledger_json(
    l: &rcsr::gpu_stats::RunLedger,
    a: &rcsr::gpu_stats::GpuStats,
    best_error: f64,
    baseline_error: f64,
    metric_is_squared: bool,
) -> String {
    use rcsr::gpu_stats::Stage;
    let stages = Stage::ALL
        .iter()
        .map(|s| format!("{}:{}", jstr(&format!("{}_s", s.label())), jnum(l.stage_s(*s))))
        .collect::<Vec<_>>()
        .join(",");
    let causes = l
        .fallback_by_cause()
        .iter()
        .map(|(name, n)| format!("{}:{}", jstr(name), n))
        .collect::<Vec<_>>()
        .join(",");
    let improved_causes = l
        .fallback_improved_by_cause()
        .iter()
        .map(|(name, n)| format!("{}:{}", jstr(name), n))
        .collect::<Vec<_>>()
        .join(",");
    let structural_zero = Stage::ALL
        .iter()
        .filter(|s| s.structurally_zero_in_search())
        .map(|s| jstr(s.label()))
        .collect::<Vec<_>>()
        .join(",");
    let outside = [Stage::Io, Stage::Features]
        .iter()
        .map(|s| format!("{}:{}", jstr(&format!("{}_s", s.label())), jnum(l.outside_s(*s))))
        .collect::<Vec<_>>()
        .join(",");
    let (by_event, by_host) = l.interval_counts();
    let closure = l.closure_residual();
    let quality = if metric_is_squared {
        l.quality_per_gpu_hour(best_error, baseline_error)
    } else {
        rcsr::gpu_stats::Ratio::undefined(f64::NAN, f64::NAN, "decades_per_gpu_hour", "run")
    };
    format!(
        "{{\"scope\":\"run\",\"wall_s\":{},\"stages\":{{{}}},\"stage_sum_s\":{},\"unaccounted_s\":{},\"closure\":{},\"device_busy_s\":{},\"device_busy_sum_s\":{},\"device_overlap_s\":{},\"device_intervals\":{},\"device_interval_basis\":{{\"basis\":{},\"events\":{},\"host_envelope\":{}}},\"structural_zero_stages\":[{}],\"outside_run\":{{\"scope\":\"process_before_search\",\"units\":\"seconds\",{},\"json_write\":\"emitted after the report is serialised; not measurable inside its own report\"}},\"device_split\":{{\"eval_kernel_s\":{},\"eval_launch_s\":{},\"eval_copy_s\":{},\"eval_alloc_s\":{},\"fit_kernel_s\":{},\"units\":\"seconds\"}},\"duty_run\":{},\"eval_kernel_duty\":{},\"fit_kernel_duty\":{},\"fit\":{{\"host_s\":{},\"kernel_s\":{},\"kernel_lm_s\":{},\"kernel_polish_s\":{},\"fallback_s\":{},\"gate_s\":{},\"refit_s\":{},\"refits\":{},\"admitted\":{},\"device_fits\":{},\"fallback_fits\":{},\"fallback_by_cause\":{{{}}},\"fallback_improved\":{},\"fallback_improved_by_cause\":{{{}}},\"alloc\":{{\"scope\":\"run\",\"units\":\"allocator calls\",\"calls\":{},\"growths\":{},\"steady_alloc_calls\":{},\"batches\":{}}}}},\"eval_fallback\":{{\"programs\":{},\"seconds\":{}}},\"quality_per_gpu_hour\":{}}}",
        jnum(l.wall_s()),
        stages,
        jnum(l.stage_sum_s()),
        jnum(l.unaccounted_s()),
        ratio_json(Some(&closure)),
        jnum(l.device_busy_s()),
        jnum(l.device_busy_sum_s()),
        jnum(l.device_overlap_s()),
        l.intervals().len(),
        jstr(l.interval_basis()),
        by_event,
        by_host,
        structural_zero,
        outside,
        jnum(a.kernel_s),
        jnum(a.launch_s),
        jnum(a.transfer_s()),
        jnum(a.alloc_s),
        jnum(l.fit_kernel_s()),
        ratio_json(Some(&l.duty_run())),
        ratio_json(Some(&l.eval_kernel_duty())),
        ratio_json(Some(&l.fit_kernel_duty())),
        jnum(l.fit_host_s()),
        jnum(l.fit_kernel_s()),
        jnum(l.fit_kernel_lm_s()),
        jnum(l.fit_kernel_polish_s()),
        jnum(l.fit_fallback_s()),
        jnum(l.fit_gate_s()),
        jnum(l.refit_s()),
        l.refits(),
        l.admitted(),
        l.device_fits(),
        l.fallback_fits(),
        causes,
        l.fallback_improved(),
        improved_causes,
        l.fit_alloc().alloc_calls,
        l.fit_alloc().growths,
        l.fit_alloc().steady_alloc_calls,
        l.fit_alloc().batches,
        l.eval_fallback_programs(),
        jnum(l.eval_fallback_s()),
        ratio_json(Some(&quality))
    )
}

/// #118: the GPU block shared by `--json` and the final NDJSON event. It keeps
/// #131's per-generation opcode/fallback report and adds the measured full-path
/// accounting. `sm_occupancy` is explicitly `null`: it needs `ncu`, and
/// `nvidia-smi`'s sampled utilisation is neither occupancy nor duty.
fn gpu_json(
    g: &Option<rcsr::search::GpuRunStats>,
    best_error: f64,
    baseline_error: f64,
    fit_strategy: &str,
    // #284: quality/GPU-hour is only unit-consistent for the squared metric
    // (best MSE versus constant-mean MSE). Every other metric reports null
    // rather than a mixed-unit ratio.
    metric_is_squared: bool,
    // #256: `timing.total_seconds`, the one runtime of the process. The
    // `*_of_run` fractions divide by it.
    process_total_s: f64,
) -> String {
    let Some(g) = g else {
        return "null".to_string();
    };
    let a = &g.accounting;
    let quality = if metric_is_squared {
        a.quality_per_gpu_hour(best_error, baseline_error)
            .map(jnum)
            .unwrap_or_else(|| "null".to_string())
    } else {
        "null".to_string()
    };
    let ops_json = |ops: &std::collections::BTreeMap<String, usize>| -> String {
        ops.iter()
            .map(|(name, n)| format!("{}:{}", jstr(name), n))
            .collect::<Vec<_>>()
            .join(",")
    };
    let gens: Vec<String> = g
        .generations
        .iter()
        .map(|s| {
            format!(
                "{{\"generation\":{},\"programs\":{},\"native\":{},\"supported\":{},\"lowered\":{},\"fallbacks\":{},\"unsupported_programs\":{},\"fallback_seconds\":{},\"fallback_workers\":{},\"fit_screened\":{},\"fit_admitted_wide\":{},\"fit_attempts\":{},\"fit_gpu\":{},\"fit_fallback\":{},\"fit_fallback_by_cause\":{{{}}},\"fit_fallback_improved\":{},\"fit_fallback_improved_by_cause\":{{{}}},\"fit_fallback_seconds\":{},\"fit_fallback_bounded\":{},\"fit_fallback_workers\":{},\"nonfinite\":{},\"nonfinite_full\":{},\"nonfinite_cpu_finite\":{},\"nonfinite_seconds\":{},\"opcodes\":{{{}}}}}",
                s.generation,
                s.programs,
                s.native,
                s.native + s.lowered,
                s.lowered,
                s.fallbacks,
                s.unsupported_programs,
                jnum(s.fallback_seconds),
                s.fallback_workers,
                s.fit_screened,
                s.fit_admitted_wide,
                s.fit_attempts,
                s.fit_gpu,
                s.fit_fallback,
                ops_json(&s.fit_fallback_by_cause),
                s.fit_fallback_improved_by_cause.values().sum::<usize>(),
                ops_json(&s.fit_fallback_improved_by_cause),
                jnum(s.fit_fallback_seconds),
                s.fit_fallback_bounded,
                s.fit_fallback_workers,
                s.nonfinite,
                s.nonfinite_full,
                s.nonfinite_cpu_finite,
                jnum(s.nonfinite_seconds),
                ops_json(&s.opcodes)
            )
        })
        .collect();
    // #121: `gpu.totals` aggregates the per-generation counters over the run.
    let mut total_ops = std::collections::BTreeMap::new();
    let (mut t_programs, mut t_native, mut t_lowered, mut t_fallbacks) = (0usize, 0usize, 0usize, 0usize);
    let (mut t_unsupported, mut t_seconds, mut t_workers) = (0usize, 0.0f64, 0usize);
    for s in &g.generations {
        t_programs += s.programs;
        t_native += s.native;
        t_lowered += s.lowered;
        t_fallbacks += s.fallbacks;
        t_unsupported += s.unsupported_programs;
        t_seconds += s.fallback_seconds;
        t_workers = t_workers.max(s.fallback_workers);
        for (name, n) in &s.opcodes {
            *total_ops.entry(name.clone()).or_insert(0usize) += *n;
        }
    }
    let totals = format!(
        "{{\"programs\":{},\"native\":{},\"supported\":{},\"lowered\":{},\"fallbacks\":{},\"unsupported_programs\":{},\"fallback_seconds\":{},\"fallback_workers\":{},\"opcodes\":{{{}}}}}",
        t_programs,
        t_native,
        t_native + t_lowered,
        t_lowered,
        t_fallbacks,
        t_unsupported,
        jnum(t_seconds),
        t_workers,
        ops_json(&total_ops)
    );
    // #169: the run-level fit-arm ledger. Emitted for every GPU run, so the
    // strategy and its attempt/fallback/refit counts are always auditable.
    let fit = format!(
        "\"fit\":{{\"strategy\":{},\"admission\":{},\"screened\":{},\"admitted_wide\":{},\"attempts\":{},\"gpu\":{},\"fallback\":{},\"fallback_by_cause\":{{{}}},\"fallback_improved\":{},\"fallback_improved_by_cause\":{{{}}},\"status_by_label\":{{{}}},\"refits\":{},\"refit_candidates\":{},\"refit_s\":{},\"rescore_s\":{},\"seconds\":{},\"kernel_lm_s\":{},\"kernel_polish_s\":{},\"polish\":{},\"host_seconds\":{},\"fallback_seconds\":{},\"gate_s\":{},\"fallback_bounded\":{},\"fallback_workers\":{},\"fallback_max\":{},\"fit_rows\":{},\"score_rows\":{},\"fit_twin_bytes\":{},\"subsample_hash\":{}}}",
        jstr(fit_strategy),
        jstr(g.fit_admission.name()),
        g.fit_screened,
        g.fit_admitted_wide,
        g.fit_attempts,
        g.fit_gpu,
        g.fit_fallback,
        ops_json(&g.fit_fallback_by_cause),
        g.fit_fallback_improved_by_cause.values().sum::<usize>(),
        ops_json(&g.fit_fallback_improved_by_cause),
        ops_json(&g.fit_status_by_label),
        g.fit_refits,
        g.fit_refit_candidates,
        jnum(g.fit_refit_s),
        jnum(g.fit_rescore_s),
        jnum(g.fit_ms / 1000.0),
        jnum(g.ledger.fit_kernel_lm_s()),
        jnum(g.ledger.fit_kernel_polish_s()),
        jstr(g.fit_polish.name()),
        jnum(g.fit_host_ms / 1000.0),
        jnum(g.fit_fallback_ms / 1000.0),
        jnum(g.ledger.fit_gate_s()),
        g.fit_fallback_bounded,
        g.fit_fallback_workers,
        g.fit_fallback_max,
        g.fit_rows,
        g.score_rows,
        g.fit_twin_bytes,
        match g.fit_subsample_hash {
            Some(h) => format!("\"{:016x}\"", h),
            None => "null".to_string(),
        }
    );
    let run_wall = if g.ledger.wall_s() > 0.0 {
        g.ledger.wall_s()
    } else {
        a.wall_s
    };
    let run_duty = g.ledger.duty_run();
    // #256: the device works in the evaluator calls and in the fit kernels.
    let device_incl_fit = a.wall_s + g.ledger.fit_kernel_s();
    let device_fraction = rcsr::gpu_stats::Ratio::new(
        device_incl_fit,
        process_total_s,
        "fraction",
        "process",
    );
    let eval_fraction =
        rcsr::gpu_stats::Ratio::new(a.wall_s, process_total_s, "fraction", "process");
    let eval_duty = rcsr::gpu_stats::Ratio::new(a.kernel_s, a.wall_s, "fraction", "evaluator_window");
    let ledger = ledger_json(
        &g.ledger,
        &a,
        best_error,
        baseline_error,
        metric_is_squared,
    );
    format!(
        "{{\"kernel_s\":{},\"launch_s\":{},\"sync_s\":{},\"alloc_s\":{},\"h2d_s\":{},\"d2h_s\":{},\"transfer_s\":{},\"window_s\":{},\"host_s\":{},\"accounted_s\":{},\"wall_s\":{},\"idle_s\":{},\"residual_fraction\":{},\"h2d_bytes\":{},\"d2h_bytes\":{},\"batches\":{},\"failed_batches\":{},\"programs\":{},\"evals\":{},\"row_evals\":{},\"evals_per_s\":{},\"row_evals_per_s\":{},\"batches_per_s\":{},\"eval_kernel_duty\":{},\"eval_window_s\":{},\"device_seconds_incl_fit\":{},\"device_fraction_of_run\":{},\"eval_fraction_of_run\":{},\"process_total_s\":{},\"run_wall_s\":{},\"device_busy_s\":{},\"duty_run\":{},\"sm_occupancy\":null,\"occupancy_source\":\"ncu (not run)\",\"quality_per_gpu_hour\":{},\"quality_scope\":\"evaluator_window_like_unit_ratio\",\"quality_per_gpu_hour_run\":{},\"ledger\":{},\"memory_plan\":{},\"generations\":[{}],\"totals\":{}, {}}}",
        jnum(a.kernel_s),
        jnum(a.launch_s),
        jnum(a.sync_s),
        jnum(a.alloc_s),
        jnum(a.h2d_s),
        jnum(a.d2h_s),
        jnum(a.transfer_s()),
        jnum(a.window_s()),
        jnum(a.host_s),
        jnum(a.accounted_s()),
        jnum(a.wall_s),
        jnum(a.idle_s()),
        jnum(a.residual_fraction()),
        a.h2d_bytes,
        a.d2h_bytes,
        a.batches,
        a.failed_batches,
        a.programs,
        a.evals,
        a.row_evals,
        jnum(a.evals_per_s()),
        jnum(a.row_evals_per_s()),
        jnum(if a.wall_s > 0.0 {
            a.batches as f64 / a.wall_s
        } else {
            0.0
        }),
        ratio_json(Some(&eval_duty)),
        jnum(a.wall_s),
        jnum(device_incl_fit),
        ratio_json(Some(&device_fraction)),
        ratio_json(Some(&eval_fraction)),
        jnum(process_total_s),
        jnum(run_wall),
        jnum(g.ledger.device_busy_s()),
        ratio_json(Some(&run_duty)),
        quality,
        jnum(if metric_is_squared {
            g.ledger.quality_per_gpu_hour(best_error, baseline_error).value
        } else {
            f64::NAN
        }),
        ledger,
        g.memory_plan
            .as_ref()
            .map(|p| p.to_json_observed(g.observed.as_ref()))
            .unwrap_or_else(|| "null".to_string()),
        gens.join(","),
        totals,
        fit
    )
}

/// #120: fixed-reference validation hypervolume of one run.
///
/// The reference error is the variance of the held-out target, which is a
/// property of the data rather than of the arm, so two arms of the same split
/// are comparable. The reported quantity is computed from the archive the run
/// actually produced, on the same rows the report already uses.
#[derive(Clone, Copy, Debug)]
struct ValidationHv {
    sample: &'static str,
    reference_error: f64,
    hypervolume: f64,
    points: usize,
    /// Complexity span of the points the hypervolume integrates.
    complexity_min: Option<f64>,
    complexity_max: Option<f64>,
}

fn validation_hypervolume(rows: &[Row], prepared: &data::Prepared) -> ValidationHv {
    let (sample, target) = if let Some(v) = &prepared.validation {
        ("validation", &v.y)
    } else if !prepared.test.y.is_empty() {
        ("test", &prepared.test.y)
    } else {
        ("train", &prepared.train.y)
    };
    let pick: fn(&Row) -> f64 = match sample {
        "validation" => |r: &Row| r.validation_error,
        "test" => |r: &Row| r.test_error,
        _ => |r: &Row| r.train_error,
    };
    let n = target.len();
    let mean = if n == 0 {
        0.0
    } else {
        target.iter().sum::<f64>() / n as f64
    };
    let var = if n == 0 {
        0.0
    } else {
        target.iter().map(|y| (y - mean) * (y - mean)).sum::<f64>() / n as f64
    };
    let reference_error = if var.is_finite() && var > 0.0 {
        var
    } else {
        1.0
    };
    let points: Vec<(f64, f64)> = rows
        .iter()
        .map(|r| (r.complexity, pick(r)))
        .filter(|(c, e)| c.is_finite() && e.is_finite())
        .collect();
    ValidationHv {
        sample,
        reference_error,
        hypervolume: qd::fixed_reference_hypervolume(&points, reference_error),
        points: points.len(),
        complexity_min: points.iter().map(|p| p.0).reduce(f64::min),
        complexity_max: points.iter().map(|p| p.0).reduce(f64::max),
    }
}

fn validation_hv_json(v: &ValidationHv) -> String {
    format!(
        "{{\"sample\":\"{}\",\"reference_error\":{},\"hypervolume\":{},\"points\":{},\"complexity_min\":{},\"complexity_max\":{}}}",
        v.sample,
        jnum(v.reference_error),
        jnum(v.hypervolume),
        v.points,
        v.complexity_min.map(jnum).unwrap_or_else(|| "null".to_string()),
        v.complexity_max.map(jnum).unwrap_or_else(|| "null".to_string())
    )
}

/// The phase timings with the gate seconds added when the gates ran (#257).
fn timing_json_g(t: &PhaseTimings, gates: Option<f64>) -> String {
    let mut out = t.json();
    if let Some(g) = gates {
        out.pop();
        out.push_str(&format!(",\"gates_seconds\":{}}}", jnum(g)));
    }
    out
}

/// #256: the human summary of a finished search: elapsed, phase timings and
/// the GPU ledger lines. It is printed for the text format and, unless
/// `--quiet`, for `--json`/`--json-out` too, so a result file never hides the
/// line that reports GPU time.
fn eprint_run_summary(
    outcome: &rcsr::search::Outcome,
    cfg: &search::Config,
    train: &data::Sample,
    elapsed: f64,
    timing: &PhaseTimings,
) {
    eprintln!(
        "# elapsed: {:.2}s | generations: {} | pareto members: {}{}",
        elapsed,
        outcome.generations,
        outcome.archive.items.len(),
        if outcome.stopped_early() {
            format!(" (stopped early: {})", outcome.stop_reason)
        } else {
            String::new()
        }
    );
    eprintln!("# time: {}", timing.text());
    // #121: the fallback counters are not only in --json; a text
    // run says how much work the CPU pool carried.
    if let Some(g) = &outcome.gpu {
        let programs: usize =
            g.generations.iter().map(|s| s.unsupported_programs).sum();
        if programs > 0 {
            let seconds: f64 =
                g.generations.iter().map(|s| s.fallback_seconds).sum();
            eprintln!(
                "# gpu: cpu fallback {} programs in {:.3}s on {} worker(s)",
                programs,
                seconds,
                g.generations
                    .last()
                    .map(|s| s.fallback_workers)
                    .unwrap_or(0)
            );
        }
    }
    if let Some(g) = &outcome.gpu {
        let a = &g.accounting;
        let ledger = &g.ledger;
        eprintln!(
            "# gpu: kernel={:.3}s sync={:.3}s alloc={:.3}s h2d={:.3}s d2h={:.3}s launch={:.3}s host={:.3}s wall={:.3}s | eval duty={:.1}% (eval window) | run duty={:.1}% (device-busy union / run wall {:.3}s) | {:.0} evals/s, {:.3e} row-evals/s | {:.3} decades/GPU-h",
            a.kernel_s,
            a.sync_s,
            a.alloc_s,
            a.h2d_s,
            a.d2h_s,
            a.launch_s,
            a.host_s,
            a.wall_s,
            100.0 * a.duty_cycle(),
            100.0 * ledger.duty_run().value,
            ledger.wall_s(),
            a.evals_per_s(),
            a.row_evals_per_s(),
            if cfg.metric == rcsr::metrics::Metric::Squared {
                ledger
                    .quality_per_gpu_hour(
                        gpu_best_error(outcome),
                        gpu_baseline_error(train),
                    )
                    .value
            } else {
                f64::NAN
            }
        );
        eprintln!(
            "# gpu ledger: wall={:.3}s stages[vary={:.3} eval={:.3} fit={:.3} select={:.3} archive={:.3} io={:.3} host_other={:.3}] unaccounted={:.3}s ({:.2}%) | device_busy={:.3}s intervals={} overlap={:.3}s | fit kernel={:.3}s fallback={:.3}s refits={:.3}s | admitted={} device={} fallback={} [{}]",
            ledger.wall_s(),
            ledger.stage_s(rcsr::gpu_stats::Stage::Vary),
            ledger.stage_s(rcsr::gpu_stats::Stage::Eval),
            ledger.stage_s(rcsr::gpu_stats::Stage::Fit),
            ledger.stage_s(rcsr::gpu_stats::Stage::Select),
            ledger.stage_s(rcsr::gpu_stats::Stage::Archive),
            ledger.stage_s(rcsr::gpu_stats::Stage::Io),
            ledger.stage_s(rcsr::gpu_stats::Stage::HostOther),
            ledger.unaccounted_s(),
            100.0 * if ledger.wall_s() > 0.0 {
                ledger.unaccounted_s() / ledger.wall_s()
            } else {
                0.0
            },
            ledger.device_busy_s(),
            ledger.intervals().len(),
            ledger.device_overlap_s(),
            ledger.fit_kernel_s(),
            ledger.fit_fallback_s(),
            ledger.refit_s(),
            ledger.admitted(),
            ledger.device_fits(),
            ledger.fallback_fits(),
            ledger
                .fallback_by_cause()
                .iter()
                .map(|(k, v)| format!("{}={}", k, v))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
}

/// #120: QD coverage in the human-readable log.
fn qd_text(q: Option<&qd::QdReport>) {
    if let Some(q) = q {
        eprintln!(
            "# qd ({}): {}/{} reachable cells occupied ({:.1}%), {} entries, best error {}, \
             entropy {:.3}/{:.3} bits, qd-score {:.4}, descriptor fallbacks {}, hypervolume {:.4} (self-normalised diagnostic)",
            q.mode,
            q.occupied_reachable,
            q.reachable,
            100.0 * q.reachable_coverage,
            q.entries,
            jnum(q.best_error),
            q.occupancy_entropy,
            q.occupancy_entropy_normalized,
            q.qd_score,
            q.descriptor_fallbacks,
            q.hypervolume
        );
    }
}

/// #120: QD coverage in the JSON result (`null` when the layer is off).
fn qd_json(q: Option<&qd::QdReport>) -> String {
    match q {
        None => "null".to_string(),
        Some(q) => format!(
            "{{\"mode\":\"{}\",\"bins\":{},\"cells\":{},\"reachable\":{},\"occupied\":{},\"occupied_reachable\":{},\"coverage\":{},\"reachable_coverage\":{},\"entries\":{},\"best_error\":{},\"cell_cap\":{},\"occupancy_entropy\":{},\"occupancy_entropy_normalized\":{},\"qd_score\":{},\"descriptor_fallbacks\":{},\"hypervolume\":{},\"evaluations\":{}}}",
            q.mode,
            q.bins,
            q.cells,
            q.reachable,
            q.occupied,
            q.occupied_reachable,
            jnum(q.coverage),
            jnum(q.reachable_coverage),
            q.entries,
            jnum(q.best_error),
            q.cell_cap,
            jnum(q.occupancy_entropy),
            jnum(q.occupancy_entropy_normalized),
            jnum(q.qd_score),
            q.descriptor_fallbacks,
            jnum(q.hypervolume),
            q.evaluations
        ),
    }
}

/// #134: non-overlapping wall-clock phases for one search run. `prep` covers
/// argument parsing, loading and preparation; `search` is the evolutionary
/// run; the rest are the post-search work. `total` is measured independently at
/// the end, so the phases can be checked against the wall instead of assumed to
/// add up. `elapsed_seconds` is kept as the search phase for compatibility.
#[derive(Clone, Copy, Default)]
struct PhaseTimings {
    prep: f64,
    search: f64,
    baselines: f64,
    diagnostics: f64,
    report: f64,
    total: f64,
    /// #118: GPU full-path wall measured inside `search`. Informational: it is a
    /// subset of `search`, not another additive phase.
    gpu: f64,
}

impl PhaseTimings {
    fn json(&self) -> String {
        format!(
            "{{\"prep_seconds\":{},\"search_seconds\":{},\"baselines_seconds\":{},\"diagnostics_seconds\":{},\"report_seconds\":{},\"unaccounted_seconds\":{},\"total_seconds\":{},\"gpu_seconds\":{}}}",
            jnum(self.prep),
            jnum(self.search),
            jnum(self.baselines),
            jnum(self.diagnostics),
            jnum(self.report),
            jnum(self.unaccounted()),
            jnum(self.total),
            jnum(self.gpu)
        )
    }

    fn text(&self) -> String {
        format!(
            "prep={:.2}s search={:.2}s baselines={:.2}s diagnostics={:.2}s report={:.2}s total={:.2}s gpu={:.2}s",
            self.prep, self.search, self.baselines, self.diagnostics, self.report, self.total, self.gpu
        )
    }

    /// #256: `total` minus the measured phases. The phases and this remainder
    /// add up to `total_seconds` exactly; `total_seconds` is the one runtime of
    /// the process.
    fn unaccounted(&self) -> f64 {
        self.total - self.phases_sum()
    }

    fn phases_sum(&self) -> f64 {
        self.prep + self.search + self.baselines + self.diagnostics + self.report
    }
}

/// #211: the shared `--dev-log` sink.
///
/// One per process. `main` writes `run_start`, `run_end` and checkpoint events
/// through it, and the search's generation callback holds a clone, so the file
/// is a single ordered stream no matter which part of the engine produced the
/// event. The `Mutex` is never contended: the search calls the callback on its
/// main thread.
#[derive(Clone)]
struct DevSink {
    log: std::sync::Arc<std::sync::Mutex<rcsr::devlog::Log>>,
    /// `--dev-log-level`: the shape of a `generation` event.
    level: rcsr::devlog::Level,
    /// The resolved cadence: write a generation event only when
    /// `generation % every == 0`.
    every: usize,
}

impl DevSink {
    fn open(
        path: &str,
        max_bytes: u64,
        every: usize,
        level: rcsr::devlog::Level,
    ) -> Result<DevSink, String> {
        Ok(DevSink {
            log: std::sync::Arc::new(std::sync::Mutex::new(rcsr::devlog::Log::create(
                path, max_bytes,
            )?)),
            level,
            every: every.max(1),
        })
    }

    fn write(&self, line: &str, terminal: bool) {
        match self.log.lock() {
            Ok(mut log) => {
                log.append(line, terminal);
            }
            Err(_) => eprintln!("warning: --dev-log: the log lock is poisoned"),
        }
    }

    /// The search-side callback.
    fn callback(&self) -> rcsr::search::DevFn {
        let sink = self.clone();
        std::sync::Arc::new(move |rec: &rcsr::devlog::GenRecord| {
            if sink.every == 1 || rec.generation % sink.every == 0 {
                sink.write(&rec.json_at(sink.level), false);
            }
        })
    }

    fn checkpoint(&self, generation: usize, path: &str) {
        // The hash and size let a reader prove the file it has is the one the
        // log names. A failed read is not a search error: the event records 0.
        let (bytes, hash) = match std::fs::read(path) {
            Ok(body) => (body.len() as u64, rcsr::devlog::fnv1a(&body)),
            Err(_) => (0, 0),
        };
        self.write(
            &rcsr::devlog::checkpoint_json(generation, path, bytes, hash),
            false,
        );
    }

    /// Bookkeeping for `run_end`: (events, bytes, dropped) before this event.
    fn stats(&self) -> (u64, u64, u64) {
        match self.log.lock() {
            Ok(log) => (log.events(), log.bytes(), log.dropped()),
            Err(_) => (0, 0, 0),
        }
    }

    fn end(
        &self,
        outcome: &rcsr::search::Outcome,
        wall_s: f64,
        search_s: f64,
        json_path: Option<String>,
    ) {
        let best = outcome
            .archive
            .items
            .iter()
            .map(|s| s.error)
            .fold(f64::INFINITY, f64::min);
        let gpu_wall = outcome.gpu.as_ref().map(|g| g.accounting.wall_s);
        // #256: the device was busy in the evaluator calls and in the fit
        // kernels; the evaluator window alone undercounts a fit-heavy run.
        let device_s = outcome
            .gpu
            .as_ref()
            .map(|g| g.accounting.wall_s + g.ledger.fit_kernel_s());
        let (events, bytes, dropped) = self.stats();
        let end = rcsr::devlog::RunEnd {
            generations: outcome.generations,
            stop: outcome.stop_reason.to_string(),
            wall_s,
            search_s,
            pareto_size: outcome.archive.items.len(),
            best_error: best,
            archive_hash: outcome.archive.hash(),
            json_path,
            peak_rss_bytes: rcsr::devlog::peak_rss_bytes(),
            gpu_wall_s: gpu_wall,
            device_hours: device_s.map(|d| d / 3600.0),
            device_seconds_incl_fit: device_s,
            device_fraction_of_run: device_s.map(|d| if wall_s > 0.0 { d / wall_s } else { 0.0 }),
            run_wall_s: outcome.gpu.as_ref().map(|g| g.ledger.wall_s()),
            device_busy_s: outcome.gpu.as_ref().map(|g| g.ledger.device_busy_s()),
            duty_run: outcome.gpu.as_ref().map(|g| g.ledger.duty_run().value),
            unaccounted_s: outcome.gpu.as_ref().map(|g| g.ledger.unaccounted_s()),
            fit_kernel_s: outcome.gpu.as_ref().map(|g| g.ledger.fit_kernel_s()),
            fit_kernel_lm_s: outcome.gpu.as_ref().map(|g| g.ledger.fit_kernel_lm_s()),
            fit_kernel_polish_s: outcome.gpu.as_ref().map(|g| g.ledger.fit_kernel_polish_s()),
            fit_fallback_s: outcome.gpu.as_ref().map(|g| g.ledger.fit_fallback_s()),
            prune_fits: outcome.prune.fits,
            prune_accepted: outcome.prune.accepted,
            prune_s: outcome.prune.seconds,
            fit_refit_s: outcome.gpu.as_ref().map(|g| g.ledger.refit_s()),
            fit_refits: outcome.gpu.as_ref().map(|g| g.fit_refits),
            events,
            bytes,
            dropped,
            end_utc: manifest::utc_now(),
        };
        self.write(&rcsr::devlog::run_end_json(&end), true);
    }
}

/// #211: the `run_start` event, built from the same facts the run manifest
/// records (#140).
#[allow(clippy::too_many_arguments)]
fn dev_run_start(
    command: &str,
    data_path: &str,
    prepared: &data::Prepared,
    cfg: &search::Config,
    backend: &str,
    device: Option<String>,
    level: rcsr::devlog::Level,
    every: usize,
    max_bytes: u64,
    resumed_from: Option<String>,
    fit_resolution: Option<(String, String)>,
) -> rcsr::devlog::RunStart {
    let (commit, dirty) = manifest::build_info();
    let exe = std::env::current_exe().unwrap_or_default();
    let train = rcsr::checkpoint::fingerprint(&prepared.train);
    let test = rcsr::checkpoint::fingerprint(&prepared.test);
    let canonical = rcsr::checkpoint::config_canonical(cfg);
    let config: Vec<(String, String)> = canonical
        .lines()
        .filter_map(|l| l.strip_prefix("config "))
        .filter_map(|l| l.split_once(' '))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect();
    rcsr::devlog::RunStart {
        command: command.to_string(),
        version: VERSION.to_string(),
        commit,
        dirty,
        binary_sha256: manifest::sha256_file(&exe).unwrap_or_else(|_| "unknown".to_string()),
        data_path: data_path.to_string(),
        data_sha256: if data_path == "-" || data_path.is_empty() {
            None
        } else {
            manifest::sha256_file(std::path::Path::new(data_path)).ok()
        },
        train_rows: train.rows,
        train_fingerprint: format!("{:016x}", train.hash),
        test_rows: test.rows,
        test_fingerprint: format!("{:016x}", test.hash),
        full_rows: prepared.prep.full_rows,
        sampled_rows: prepared.prep.sampled_rows,
        estimate_bytes: prepared.prep.store_estimate.map(|m| m.total_bytes),
        config,
        config_hash: rcsr::checkpoint::config_hash(cfg),
        seed: cfg.seed,
        backend: backend.to_string(),
        device,
        threads: cfg.threads,
        host: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        start_utc: manifest::utc_now(),
        level,
        every,
        max_bytes,
        resumed_from,
        fit_resolution,
    }
}

/// #211: where the default log goes — next to the result artifacts when the run
/// names them, otherwise the working directory.
///
/// `--json-out` wins over `--save` because it is the artifact the person reads;
/// a `--run-manifest` is not used because it is provenance, not a result.
fn default_dev_log_path(json_out: Option<&str>, save: Option<&str>) -> String {
    let dir = json_out
        .or(save)
        .map(std::path::Path::new)
        .and_then(|p| p.parent())
        .filter(|d| !d.as_os_str().is_empty());
    match dir {
        Some(d) => d.join(rcsr::devlog::DEFAULT_FILE_NAME),
        None => std::path::PathBuf::from(rcsr::devlog::DEFAULT_FILE_NAME),
    }
    .to_string_lossy()
    .into_owned()
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

/// #129: resolve `--gap`. An integer is rows; `90m`, `1.5h`, `30s` or `2d` is
/// wall time converted with `ceil(seconds / median_dt)`. A wall-time gap needs
/// a time column; without one the request is a usage error, never a zero-row
/// embargo.
fn parse_gap_spec(
    src: &str,
    table: &data::Table,
    time_col: Option<usize>,
    unit: Option<data::TimeUnit>,
) -> Result<(usize, Option<String>), String> {
    let trimmed = src.trim();
    if let Ok(rows) = trimmed.parse::<usize>() {
        return Ok((rows, None));
    }
    let (num_src, multiplier) = if let Some(v) = trimmed.strip_suffix(&['s', 'S'][..]) {
        (v, 1.0)
    } else if let Some(v) = trimmed.strip_suffix(&['m', 'M'][..]) {
        (v, 60.0)
    } else if let Some(v) = trimmed.strip_suffix(&['h', 'H'][..]) {
        (v, 3600.0)
    } else if let Some(v) = trimmed.strip_suffix(&['d', 'D'][..]) {
        (v, 86_400.0)
    } else {
        return Err(format!(
            "--gap '{}' is neither a row count nor a wall time (use 90m, 1.5h, 30s or 2d)",
            src
        ));
    };
    let value: f64 = num_src
        .trim()
        .parse()
        .map_err(|_| format!("--gap '{}' has no numeric value", src))?;
    if !(value > 0.0) {
        return Err("--gap must be positive".to_string());
    }
    let Some(tc) = time_col else {
        return Err(format!(
            "--gap {} needs --time <col> to convert wall time to rows",
            src
        ));
    };
    let mut raw: Vec<f64> = data::time_keys(table, tc).to_f64();
    raw.sort_by(f64::total_cmp);
    let mut diffs: Vec<f64> = raw.windows(2).map(|w| w[1] - w[0]).filter(|d| *d > 0.0).collect();
    diffs.sort_by(f64::total_cmp);
    if diffs.is_empty() {
        return Err(format!(
            "--gap {} cannot be converted: the time column has no positive spacing",
            src
        ));
    }
    let median = diffs[diffs.len() / 2];
    let to_seconds = match unit.unwrap_or(data::TimeUnit::Seconds) {
        data::TimeUnit::Seconds => 1.0,
        data::TimeUnit::Millis => 1e-3,
        data::TimeUnit::Micros => 1e-6,
        data::TimeUnit::Nanos => 1e-9,
    };
    let median_seconds = median * to_seconds;
    let seconds = value * multiplier;
    let rows = leakage::wall_time_gap(seconds, median_seconds).ok_or_else(|| {
        format!(
            "--gap {} cannot be converted: median spacing is {} s",
            src, median_seconds
        )
    })?;
    Ok((
        rows,
        Some(format!(
            "--gap {} = {:.1} s -> {} rows (ceil({:.1}/{:.4}))",
            src, seconds, rows, seconds, median_seconds
        )),
    ))
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
    /// Position in the archive's train-error order, before selection reorders
    /// the printed rows (#111).
    archive_rank: usize,
    complexity: f64,
    /// Fitted constants: the free parameters of the model.
    k_constants: usize,
    train_error: f64,
    /// Error on the chronological validation slice, when one exists (#111).
    /// Selection reads this (or train/AICc), never `test_error`.
    validation_error: f64,
    test_error: f64,
    /// #263: the P&L objective on the validation slice, lower is better;
    /// `NaN` without a trading objective.
    objective_value: f64,
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
    /// #257: gate members of the pareto JSON (leading comma); empty when the
    /// gates did not run.
    gate_json: String,
    /// #480: the `"fitness"` member of the pareto JSON (leading comma); empty
    /// without `--fitness pnl`.
    fitness_json: String,
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
    // #71: `import` is a subcommand, handled before the search option parser.
    if args.first().map(|s| s.as_str()) == Some("import") {
        run_import(&args[1..]);
    }
    // #123: stream raw klines/CSV archives straight to a column store.
    if args.first().map(|s| s.as_str()) == Some("prep") {
        rcsr::prep::run(&args[1..]);
    }
    // #200: append pre-computed features to an existing OHLCV store.
    if args.first().map(|s| s.as_str()) == Some("enrich") {
        rcsr::enrich::run(&args[1..]);
    }
    // #145: point-in-time membership and the rebuilt panel.
    if args.first().map(|s| s.as_str()) == Some("panel") {
        rcsr::panel::run(&args[1..]);
    }
    // #264: windows table, candidate ledger, sealed list.
    if args.first().map(|s| s.as_str()) == Some("protocol") {
        rcsr::protocol::run(&args[1..]);
    }
    // #266: long-only baseline ladder and add-one-family ablation.
    if args.first().map(|s| s.as_str()) == Some("ladder") {
        rcsr::ladder::run(&args[1..]);
    }
    if args.first().map(|s| s.as_str()) == Some("ablation") {
        rcsr::ablation::run(&args[1..]);
    }
    // #124: a reusable Rust walk-forward evaluator over the column store.
    if matches!(
        args.first().map(|s| s.as_str()),
        Some("walkforward") | Some("eval")
    ) {
        run_walkforward(&args[1..]);
    }
    // #211: summarise a --dev-log file (phases, fallbacks, bottlenecks).
    if args.first().map(|s| s.as_str()) == Some("dev-report") {
        rcsr::devreport::run(&args[1..]);
    }
    // #140: verify a run manifest without starting a search.
    if args.first().map(|s| s.as_str()) == Some("verify-manifest") {
        run_verify_manifest(&args[1..]);
    }
    // #129: differential leakage audit. Exit 5 when any detector fires.
    if args.first().map(|s| s.as_str()) == Some("audit") {
        run_audit(&args[1..]);
    }
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
    // #141: whether the user asserted a backend explicitly. `None` means the
    // run follows the build/device default; `Some(true)` is `--gpu` and
    // `Some(false)` is `--no-gpu`/`--cpu`.
    let mut gpu_explicit: Option<bool> = None;
    // #293: `--gpu-memory`, a device memory cap for the plan.
    let mut gpu_memory: Option<u64> = None;
    // #463: `--threads`, worker threads for fitting and scoring offspring.
    let mut threads: usize = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    // #438: `--dispatch auto|device`.
    let mut dispatch_mode = rcsr::dispatch::DispatchMode::Auto;
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut dtype_src: Option<String> = None;
    let mut max_levels = 16usize;
    let mut test_frac = 0.25;
    // #111: chronological validation slice inside the training period, and the
    // criterion used to choose the reported candidate. Selection never reads
    // the test lockbox.
    let mut validation_frac = 0.0f64;
    let mut select_by_arg: Option<String> = None;
    // #263: exact long/flat P&L objective for finalist selection.
    let mut objective_arg: Option<String> = None;
    // #480: what ranks candidates in the search.
    let mut fitness_arg: Option<String> = None;
    // #484: the time-block consistency term of the P&L fitness.
    let mut pnl_blocks_arg: Option<String> = None;
    let mut pnl_lambda_arg: Option<String> = None;
    let mut trading_flags: Vec<&'static str> = Vec::new();
    let mut position_arg: Option<String> = None;
    let mut allow_short = false;
    let mut trade_threshold = 0.0f64;
    let mut trade_cost_bps: Option<f64> = None;
    let mut trade_cost_override: Option<f64> = None;
    let mut trade_slippage_bps = 0.0f64;
    let mut trade_cash_return = 0.0f64;
    let mut trade_funding: Option<String> = None;
    let mut trade_max_position = 1.0f64;
    let mut trade_min_trades = 1usize;
    let mut trade_min_holding = 1usize;
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
    // #129: a bare integer is rows; a suffixed value (90m, 1.5h) is wall
    // time and is converted after the time column is known.
    let mut gap_src: Option<String> = None;
    let mut folds = 1usize;
    let mut lags = 0usize;
    let mut lag_set_src: Option<String> = None;
    let mut project_src: Option<String> = None;
    let mut view_index = 0usize;
    let mut history_frac = 0.0f64;
    let mut gap_break = 0.0f64;
    let mut date_only = data::DateOnlyPolicy::Error;
    let mut time_unit: Option<data::TimeUnit> = None;
    let mut tz: Option<i64> = None;
    let mut null_runs = 0usize;
    // #114: block length for a block-permutation null; 0 selects the i.i.d.
    // row shuffle. `--null-block` without `--null-runs` runs 19 permutations.
    let mut null_block = 0usize;
    // #69: query-row cap for the k-NN null baseline; 0 = no caps at all.
    // #497: unset = exact up to `KNN_EXACT_MAX_ROWS` training rows, sampled above.
    let mut baseline_samples: Option<usize> = None;
    let mut bootstrap_resamples = 0usize;
    let mut eval_expr: Option<String> = None;
    let mut seed_exprs: Vec<String> = Vec::new();
    let mut seed_files: Vec<String> = Vec::new();
    let mut function_defs: Vec<String> = Vec::new();
    let mut stability_runs = 0usize;
    let mut cv_mode: Option<data::CvMode> = None;
    let mut cv_k = 5usize;
    let mut fill = data::FillMethod::None;
    let mut skip_malformed = false;
    // #71: memory guard. `--max-memory` is an explicit byte budget; `--force`
    // is the deliberate override.
    let mut max_memory: Option<u64> = None;
    let mut force = false;
    // #112: sampling is opt-in. The automatic path never samples a store; it
    // loads the full rows or refuses with the header-derived estimate.
    let mut sample_rows: Option<usize> = None;
    let mut outlier_iqr: Option<f64> = None;
    let mut smooth = 0usize;
    let mut smooth_centred = false;
    let mut allow_interpolate = false;
    // #145: a search may not train or score inside 2026H2 by default.
    let mut allow_lockbox = false;
    // #264: declared locked test and the ledger whose reads run.json counts.
    let mut lockbox_arg: Option<String> = None;
    let mut lockbox_embargo_arg: Option<String> = None;
    let mut ledger_arg: Option<String> = None;
    // #266: ablation arm label recorded in the run report.
    let mut arm_arg: Option<String> = None;
    // #254: declared target spec and the structural-term override.
    let mut target_spec_path: Option<String> = None;
    let mut allow_target_structure = false;
    let mut require_clean_inputs = false;
    let mut require_validation = false;
    let mut equivalence_z = rcsr::gate_run::DEFAULT_Z;
    let mut noise_lag_arg: Option<String> = None;
    let mut center = data::Center::Mean;
    let mut scale = data::Scale::Std;
    let mut filter_src: Option<String> = None;
    let mut exclude_src: Option<String> = None;
    let mut train_select_src: Option<String> = None;
    let mut test_select_src: Option<String> = None;
    let mut weight_name: Option<String> = None;
    let mut implicit_mode = false;
    let mut save_path: Option<String> = None;
    let mut save_every = 25usize;
    let mut save_every_set = false;
    let mut resume_path: Option<String> = None;
    // #140: run integrity flags.
    let mut allow_backend_change = false;
    let mut allow_build_change = false;
    let mut run_manifest_path: Option<String> = None;
    let mut no_project = false;
    let mut json_out: Option<String> = None;
    // #211: the engine-level development log. It is **on by default** at
    // `summary` level, which is what replaces the old external run monitor.
    let mut dev_log_path: Option<String> = None;
    let mut dev_log_level = rcsr::devlog::Level::Summary;
    let mut dev_log_level_set = false;
    let mut dev_log_disabled = false;
    let mut dev_log_every: Option<usize> = None;
    let mut dev_log_max_bytes = rcsr::devlog::DEFAULT_MAX_BYTES;
    let mut dev_log_max_bytes_set = false;
    let mut dev_log_requested = false;

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
    // #116: remember an explicit --fit-strategy so a resume can refuse a
    // contradiction with the saved configuration.
    let mut fit_strategy_set = false;
    let mut dedup_set = false;
    let mut patience_set = false;
    let mut patience_tol_set = false;
    let mut stop_front_set = false;
    let mut time_budget_set = false;
    let mut predictor_samples_set = false;
    // #85: remember an explicit --fit-samples so a resume can refuse a
    // contradiction with the saved configuration.
    let mut fit_samples_set = false;
    // #116: explicit --fit-rows for the resume settings check.
    let mut fit_rows_set = false;
    // #255: explicit --fit-fallback-max for the resume settings check.
    let mut fit_fallback_max_set = false;
    // #495: explicit --refit-band for the resume settings check.
    let mut refit_band_set = false;
    let mut fit_polish_set = false;
    // #76: remember an explicit --no-front-parents so a resume can refuse a
    // contradiction with the saved configuration.
    let mut front_parents_set = false;
    let mut reject_constant_set = false;
    // #465: explicit --selection for the resume settings check.
    let mut selection_set = false;
    let mut linear_scaling_set = false; // #470
    let mut mutation_mix_set = false; // #471
    let mut tree_arity_set = false;
    let mut crossover_internal_set = false;
    let mut adaptive_mix_set = false;
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
            // #71: refuse before the resident Table is built.
            "--max-memory" => {
                let v = val!();
                match parse_memory_bytes(&v) {
                    Ok(n) if n > 0 => max_memory = Some(n),
                    _ => {
                        eprintln!(
                            "error: --max-memory needs a positive size (bytes, KiB/MiB/GiB/GB), got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--force" => force = true,
            // #112: explicit hash sample. Without this flag a store that does
            // not fit is refused, never sampled silently.
            "--sample-rows" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => sample_rows = Some(n),
                    _ => {
                        eprintln!(
                            "error: --sample-rows needs an integer >= 1, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--skip-malformed" => skip_malformed = true,
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
            "--implicit" => implicit_mode = true,
            "--implicit-derivative" => {
                // #14: the recommended implicit objective, as one flag.
                implicit_mode = true;
                cfg.metric = Metric::ImplicitDerivative;
                metric_set = true;
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
                // Applied after --function definitions are known, so a user's
                // block name can be resolved to its slot.
                op_complexity_spec = Some(val!());
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
            "--validation-frac" => validation_frac = val!().parse().unwrap_or(validation_frac),
            "--select-by" => select_by_arg = Some(val!()),
            "--objective" => objective_arg = Some(val!()),
            "--fitness" => fitness_arg = Some(val!()),
            "--pnl-blocks" => pnl_blocks_arg = Some(val!()),
            "--pnl-block-lambda" => pnl_lambda_arg = Some(val!()),
            "--position" => {
                trading_flags.push("--position");
                position_arg = Some(val!());
            }
            "--allow-short" => {
                trading_flags.push("--allow-short");
                allow_short = true;
            }
            "--threshold" => {
                trading_flags.push("--threshold");
                trade_threshold = parse_trade_number("--threshold", &val!());
            }
            "--cost-bps" => {
                trading_flags.push("--cost-bps");
                trade_cost_bps = Some(parse_trade_number("--cost-bps", &val!()));
            }
            "--cost-override" => {
                trading_flags.push("--cost-override");
                trade_cost_override = Some(parse_trade_number("--cost-override", &val!()));
            }
            "--slippage-bps" => {
                trading_flags.push("--slippage-bps");
                trade_slippage_bps = parse_trade_number("--slippage-bps", &val!());
            }
            "--cash-return" => {
                trading_flags.push("--cash-return");
                trade_cash_return = parse_trade_number("--cash-return", &val!());
            }
            "--funding" => {
                trading_flags.push("--funding");
                trade_funding = Some(val!());
            }
            "--max-position" => {
                trading_flags.push("--max-position");
                trade_max_position = parse_trade_number("--max-position", &val!());
            }
            "--min-trades" => {
                trading_flags.push("--min-trades");
                trade_min_trades = parse_trade_number("--min-trades", &val!()) as usize;
            }
            "--min-holding" => {
                trading_flags.push("--min-holding");
                trade_min_holding = parse_trade_number("--min-holding", &val!()) as usize;
            }
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
            // #110: centred smoothing reads future bars; opt in explicitly.
            "--smooth-centred" => match val!().parse::<usize>() {
                Ok(k) => {
                    smooth = k;
                    smooth_centred = true;
                }
                Err(_) => {
                    eprintln!("error: --smooth-centred needs a non-negative integer window");
                    std::process::exit(EXIT_USAGE);
                }
            },
            "--allow-interpolate" => allow_interpolate = true,
            "--lockbox" => lockbox_arg = Some(val!()),
            "--lockbox-embargo" => lockbox_embargo_arg = Some(val!()),
            "--ledger" => ledger_arg = Some(val!()),
            "--arm" => arm_arg = Some(val!()),
            "--allow-lockbox" => allow_lockbox = true,
            "--target-spec" => target_spec_path = Some(val!()),
            "--allow-target-structure" => allow_target_structure = true,
            "--require-clean-inputs" => require_clean_inputs = true,
            "--require-validation" => require_validation = true,
            "--equivalence-z" => {
                let v = val!();
                match v.parse::<f64>() {
                    Ok(z) if z.is_finite() && z > 0.0 => equivalence_z = z,
                    _ => {
                        eprintln!("error: --equivalence-z must be a number > 0, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--noise-lag" => {
                let v = val!();
                if v != "auto" && v.parse::<usize>().is_err() {
                    eprintln!("error: --noise-lag must be a non-negative integer or 'auto', got '{}'", v);
                    std::process::exit(EXIT_USAGE);
                }
                noise_lag_arg = Some(v);
            }
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
            "--gap" => gap_src = Some(val!()),
            "--folds" => folds = val!().parse().unwrap_or(1).max(1),
            "--lags" => lags = val!().parse().unwrap_or(0),
            "--lag-set" => lag_set_src = Some(val!()),
            "--project" => project_src = Some(val!()),
            "--view" => view_index = val!().parse().unwrap_or(0),
            "--history-frac" => history_frac = val!().parse().unwrap_or(0.0),
            "--gap-break" => gap_break = val!().parse().unwrap_or(0.0),
            // #130: strict time parsing. Every choice is reported.
            "--date-only" => {
                let v = val!();
                match data::DateOnlyPolicy::parse(&v) {
                    Ok(p) => date_only = p,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--time-unit" => {
                let v = val!();
                match data::TimeUnit::parse(&v) {
                    Ok(u) => time_unit = Some(u),
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--tz" => {
                let v = val!();
                match data::TimeSpec::parse_tz(&v) {
                    Ok(o) => tz = Some(o),
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--null-runs" => null_runs = val!().parse().unwrap_or(0),
            "--null-block" => null_block = val!().parse().unwrap_or(0),
            // #69: 0 = no caps; a positive N samples only the query rows.
            "--baseline-samples" => {
                baseline_samples = val!().parse().ok().or(baseline_samples);
            }
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
            "--function" => function_defs.push(val!()),
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
            // #116: reporting-only arm counts.
            "--fit-report" => cfg.fit_report = true,
            // #116: fit on a deterministic subsample, score and archive on all rows.
            "--fit-rows" => {
                let n: usize = val!().parse().unwrap_or(0);
                if n == 1 {
                    eprintln!("error: --fit-rows needs 0 (off) or an integer >= 2");
                    std::process::exit(EXIT_USAGE);
                }
                cfg.fit_rows = if n >= 2 { n } else { 0 };
                fit_rows_set = true;
            }
            // #255: hard bound on one generation's exact CPU fallback refits.
            "--fit-fallback-max" => {
                cfg.fit_fallback_max = val!().parse().unwrap_or(cfg.fit_fallback_max);
                fit_fallback_max_set = true;
            }
            // #466: front prune; a post-loop pass, not saved search state.
            "--prune" => cfg.prune = true,
            "--no-prune" => cfg.prune = false,
            "--prune-rows" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 2 => cfg.prune_rows = n,
                    _ => {
                        eprintln!("error: --prune-rows needs an integer >= 2, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--prune-budget" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) => cfg.prune_budget = n,
                    _ => {
                        eprintln!("error: --prune-budget needs a non-negative integer, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            // #495: relative band of the exact finalist refit.
            "--refit-band" => {
                let v = val!();
                cfg.refit_band = match v.as_str() {
                    "inf" => f64::INFINITY,
                    _ => match v.parse::<f64>() {
                        Ok(b) if b.is_finite() && b >= 0.0 => b,
                        _ => {
                            eprintln!("error: --refit-band needs a number >= 0 or inf");
                            std::process::exit(EXIT_USAGE);
                        }
                    },
                };
                refit_band_set = true;
            }
            // #496: arithmetic of the device fit polish.
            "--fit-polish" => {
                let v = val!();
                match fit::FitPolish::parse(&v) {
                    Some(s) => {
                        cfg.fit_polish = s;
                        fit_polish_set = true;
                    }
                    None => {
                        eprintln!("error: --fit-polish needs f64, f32 or off, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            // #116: exact constant-fitting arm.
            "--fit-strategy" => {
                let v = val!();
                match fit::FitStrategy::parse(&v) {
                    Some(s) => {
                        cfg.fit_strategy = s;
                        fit_strategy_set = true;
                    }
                    None => {
                        eprintln!(
                            "error: --fit-strategy needs exact, auto, linear, varpro, \
                             stream or gpu (cpu is an alias for exact), got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--no-dedup" => {
                cfg.dedup = false;
                dedup_set = true;
            }
            // #120: opt-in quality-diversity coverage archive.
            "--qd" => {
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = if cfg.qd_cvt { 512 } else { 16 };
                }
            }
            "--qd-cells" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if (2..=512).contains(&n) => cfg.qd_bins = n,
                    _ => {
                        eprintln!(
                            "error: --qd-cells needs an integer in 2..=512, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--qd-cell-cap" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if (1..=64).contains(&n) => cfg.qd_cell_cap = n,
                    _ => {
                        eprintln!(
                            "error: --qd-cell-cap needs an integer in 1..=64, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--qd-cvt" => {
                cfg.qd_cvt = true;
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = 512;
                }
            }
            "--qd-policy" => {
                cfg.qd_policy = true;
                cfg.qd_cvt = true;
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = 512;
                }
            }
            "--qd-uniform" => {
                cfg.qd_policy = true;
                cfg.qd_control = 1;
                cfg.qd_cvt = true;
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = 512;
                }
            }
            "--qd-random" => {
                cfg.qd_policy = true;
                cfg.qd_control = 2;
                if cfg.qd_bins == 0 {
                    cfg.qd_bins = 512;
                }
            }
            // #470: opt-in affine output map around every candidate.
            "--linear-scaling" => {
                cfg.linear_scaling = true;
                linear_scaling_set = true;
            }
            // #471: node-path variation settings, each off by default.
            "--mutation-mix" => {
                let v = val!();
                match rcsr::search::MutationMix::parse(&v) {
                    Ok(m) => cfg.mutation_mix = m,
                    Err(e) => {
                        eprintln!("error: --mutation-mix: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                mutation_mix_set = true;
            }
            "--tree-arity" => {
                let v = val!();
                match rcsr::search::TreeArity::parse(&v) {
                    Ok(m) => cfg.tree_arity = m,
                    Err(e) => {
                        eprintln!("error: --tree-arity: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                tree_arity_set = true;
            }
            "--crossover-internal" => {
                let v = val!();
                match v.parse::<f64>() {
                    Ok(p) if (0.0..=1.0).contains(&p) => cfg.crossover_internal = p,
                    _ => {
                        eprintln!("error: --crossover-internal needs a probability in [0, 1], got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                crossover_internal_set = true;
            }
            "--adaptive-mix" => {
                cfg.adaptive_mix = true;
                adaptive_mix_set = true;
            }
            // #76: exactness escape hatch; the default is on.
            // #465: survivor and parent selection inside an island.
            "--selection" => {
                let v = val!();
                match crate::search::Selection::parse(&v) {
                    Ok(s) => {
                        cfg.selection = s;
                        selection_set = true;
                    }
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--no-front-parents" => {
                cfg.front_parents = false;
                front_parents_set = true;
            }
            "--reject-constant-offspring" => {
                cfg.reject_constant_offspring = true;
                reject_constant_set = true;
            }
            // #27: evaluator path only; deliberately not a saved search
            // setting, because both paths produce identical errors.
            "--no-simd" => cfg.simd = false,
            "--gpu" => {
                cfg.gpu = true;
                gpu_explicit = Some(true);
            }
            // #463: scheduling only; results are identical for every value.
            "--threads" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if (1..=1024).contains(&n) => threads = n,
                    _ => {
                        eprintln!("error: --threads needs an integer from 1 to 1024, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            // #293: caps the device memory the plan may use.
            "--gpu-memory" => {
                let v = val!();
                match parse_memory_bytes(&v) {
                    Ok(n) if n > 0 => gpu_memory = Some(n),
                    _ => {
                        eprintln!(
                            "error: --gpu-memory needs a positive size (bytes, KiB/MiB/GiB/GB), got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            // #438: `device` skips the timing comparison; refusals still win.
            "--dispatch" => {
                let v = val!();
                match rcsr::dispatch::DispatchMode::parse(&v) {
                    Some(mode) => dispatch_mode = mode,
                    None => {
                        eprintln!("error: --dispatch needs auto or device, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--no-gpu" | "--cpu" => {
                cfg.gpu = false;
                gpu_explicit = Some(false);
            }
            // #119: bounded CPU/GPU overlap depth in batches (0 = off).
            "--pipeline-depth" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n <= 64 => cfg.pipeline_depth = n,
                    _ => {
                        eprintln!(
                            "error: --pipeline-depth needs an integer from 0 to 64, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            // #122: variation engine selection.
            "--variation" => {
                let v = val!();
                match crate::search::VariationMode::parse(&v) {
                    Ok(mode) => cfg.variation = mode,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
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
            // #85: opt-in constant-fitting screen. 0 means off (the historical
            // path); 1 cannot define a subset.
            "--fit-samples" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(0) => {
                        cfg.fit_samples = 0;
                        fit_samples_set = true;
                    }
                    Ok(n) if n >= 2 => {
                        cfg.fit_samples = n;
                        fit_samples_set = true;
                    }
                    _ => {
                        eprintln!(
                            "error: --fit-samples needs 0 (off) or an integer >= 2, got '{}'",
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
            "--patience-tol" => {
                let v = val!();
                match v.parse::<f64>() {
                    Ok(r) if r.is_finite() && r >= 0.0 => cfg.patience_tol = Some(r),
                    _ => {
                        eprintln!("error: --patience-tol needs a number >= 0, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                patience_tol_set = true;
            }
            "--stop-on" => {
                let v = val!();
                let window = match v.strip_prefix("front") {
                    Some("") => Some(0),
                    Some(rest) => rest
                        .strip_prefix(':')
                        .and_then(|w| w.parse::<usize>().ok())
                        .filter(|w| *w >= 1),
                    None => None,
                };
                match window {
                    Some(w) => cfg.stop_front = Some(w),
                    None => {
                        eprintln!("error: --stop-on needs front or front:<W> with W >= 1, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                stop_front_set = true;
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
            "--allow-backend-change" => allow_backend_change = true,
            "--allow-build-change" => allow_build_change = true,
            "--run-manifest" => run_manifest_path = Some(val!()),
            "--no-project" => no_project = true,
            // #211: the development log. It is on by default; `--no-dev-log`
            // turns it off, `--dev-log` picks the path, `--dev-log-level`
            // picks the detail, `--dev-log-every` the cadence and
            // `--dev-log-max-bytes` the bound.
            "--dev-log" => {
                dev_log_path = Some(val!());
                dev_log_requested = true;
            }
            "--no-dev-log" => {
                dev_log_disabled = true;
                dev_log_requested = true;
            }
            "--dev-log-level" => {
                let v = val!();
                match rcsr::devlog::Level::parse(&v) {
                    Some(level) => {
                        dev_log_level = level;
                        dev_log_level_set = true;
                        dev_log_requested = true;
                    }
                    None => {
                        eprintln!(
                            "error: --dev-log-level must be summary or full, got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--dev-log-every" => {
                let v = val!();
                match v.parse::<usize>() {
                    Ok(n) if n >= 1 => dev_log_every = Some(n),
                    _ => {
                        eprintln!("error: --dev-log-every needs an integer >= 1, got '{}'", v);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--dev-log-max-bytes" => {
                let v = val!();
                match parse_memory_bytes(&v) {
                    Ok(n) if n > 0 => {
                        dev_log_max_bytes = n;
                        dev_log_max_bytes_set = true;
                    }
                    _ => {
                        eprintln!(
                            "error: --dev-log-max-bytes needs a positive size (bytes, KiB/MiB/GiB/GB), got '{}'",
                            v
                        );
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--json-out" => json_out = Some(val!()),
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

    if !function_defs.is_empty() && (save_path.is_some() || resume_path.is_some()) {
        eprintln!(
            "error: --function cannot be combined with --save or --resume \
             (a checkpoint does not store block definitions)"
        );
        std::process::exit(EXIT_USAGE);
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
    // #211: the development log describes one search. It is on by default, so
    // only an *explicit* request is refused where it cannot be honoured;
    // `--report benchmark` simply runs its repeats without a log.
    if dev_log_disabled
        && (dev_log_path.is_some()
            || dev_log_level_set
            || dev_log_every.is_some()
            || dev_log_max_bytes_set)
    {
        eprintln!(
            "error: --no-dev-log conflicts with --dev-log, --dev-log-level, --dev-log-every \
             and --dev-log-max-bytes"
        );
        std::process::exit(EXIT_USAGE);
    }
    if dev_log_path.is_some() && eval_expr.is_some() {
        eprintln!("error: --dev-log cannot be combined with --eval-expr (it logs a search)");
        std::process::exit(EXIT_USAGE);
    }
    if dev_log_requested && report == Some(ReportKind::Benchmark) {
        eprintln!(
            "error: --dev-log/--dev-log-level cannot be combined with --report benchmark \
             (one log describes one search;\n  run the repeats separately or drop --report benchmark)"
        );
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
    if cfg.selection == crate::search::Selection::Afp
        && cfg.variation == crate::search::VariationMode::Postfix
    {
        eprintln!(
            "error: --selection afp runs on the node path only; \
             use --variation auto or node"
        );
        std::process::exit(EXIT_USAGE);
    }
    if allow_backend_change && resume_path.is_none() {
        eprintln!("error: --allow-backend-change only applies to --resume");
        std::process::exit(EXIT_USAGE);
    }
    if allow_build_change && resume_path.is_none() {
        eprintln!("error: --allow-build-change only applies to --resume");
        std::process::exit(EXIT_USAGE);
    }
    if json_out.is_some() {
        // #140: a file result is a JSON document, published atomically. It
        // replaces the stdout stream for the final object.
        format = Format::Json;
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
    // #14: the implicit metrics need the target as an input. Refuse the metric
    // on its own rather than silently fitting a different model.
    if cfg.metric.is_implicit() && !implicit_mode {
        eprintln!(
            "error: --metric {} requires --implicit (or use --implicit-derivative)",
            cfg.metric.name()
        );
        eprintln!(
            "  --implicit appends the target as an input and searches f(inputs, target) = 0"
        );
        std::process::exit(EXIT_USAGE);
    }
    // #223: the engine resolves the fit arm only when the person did not name
    // one, or named `auto`. A resume keeps the saved arm.
    let mut fit_request: Option<fit::FitRequest> = if !fit_strategy_set {
        Some(fit::FitRequest::Unset)
    } else if cfg.fit_strategy == fit::FitStrategy::Auto {
        Some(fit::FitRequest::Auto)
    } else {
        None
    };
    // #223 + #229 + #222: an explicit `--fit-strategy gpu` asks for the batched
    // GPU LM, which minimises squared error, log-squashed loss or mean absolute
    // error. Refuse a metric with no native device objective instead of
    // silently fitting its constants on the CPU with a different objective.
    // The check runs before any data is read, so it is a usage error.
    if fit_strategy_set && cfg.fit_strategy == fit::FitStrategy::Gpu {
        if let Some(why) = fit::gpu_fit_metric_error(cfg.metric) {
            eprintln!("error: {}", why);
            std::process::exit(EXIT_USAGE);
        }
    }
    // A y-randomisation control shuffles the target. In implicit mode the target
    // is an input and the scored target is all zeros, so every null run would be
    // the real run: refuse rather than report a meaningless p-value.
    if null_block > 0 && null_runs == 0 {
        null_runs = 19;
    }
    if null_block == 1 {
        eprintln!("error: --null-block must be at least 2 rows (use --null-runs for the i.i.d. shuffle)");
        std::process::exit(EXIT_USAGE);
    }
    if implicit_mode && null_runs > 0 {
        eprintln!("error: --implicit cannot be combined with --null-runs/--null-block");
        eprintln!("  the null control shuffles the target, which is an input here");
        std::process::exit(EXIT_USAGE);
    }
    if implicit_mode && cv_mode.is_some() {
        eprintln!("error: --implicit cannot be combined with --cv (not implemented yet)");
        eprintln!("  cross-validation rebuilds each fold from the raw table, without the implicit input");
        std::process::exit(EXIT_USAGE);
    }
    if cfg.metric == Metric::Slope {
        for (present, flag) in [
            (bootstrap_resamples > 0, "--bootstrap-error"),
            (stability_runs > 0, "--stability"),
            (cv_mode.is_some(), "--cv"),
        ] {
            if present {
                eprintln!("error: --metric slope cannot be combined with {}", flag);
                eprintln!("  slope differences rows that are adjacent in the data order; resampled or refolded rows have no such order");
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    if implicit_mode && bootstrap_resamples > 0 {
        eprintln!("error: --implicit cannot be combined with --bootstrap-error (not implemented yet)");
        eprintln!("  the bootstrap refits constants on |f|, not on the geometric residual");
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
        // #116: the fitting arm is saved search state.
        if fit_strategy_set && cfg.fit_strategy != saved.fit_strategy {
            conflicts.push(format!(
                "--fit-strategy {} (saved {})",
                cfg.fit_strategy.name(),
                saved.fit_strategy.name()
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
        if patience_tol_set && cfg.patience_tol != saved.patience_tol {
            conflicts.push(format!(
                "--patience-tol {:?} (saved {:?})",
                cfg.patience_tol, saved.patience_tol
            ));
        }
        if stop_front_set && cfg.stop_front != saved.stop_front {
            conflicts.push(format!(
                "--stop-on front {:?} (saved {:?})",
                cfg.stop_front, saved.stop_front
            ));
        }
        if predictor_samples_set && cfg.predictor_samples != saved.predictor_samples {
            conflicts.push(format!(
                "--predictor-samples {:?} (saved {:?})",
                cfg.predictor_samples, saved.predictor_samples
            ));
        }
        // #85: screening is saved search state, so an explicit contradiction is
        // refused just like --predictor-samples.
        if fit_samples_set && cfg.fit_samples != saved.fit_samples {
            conflicts.push(format!(
                "--fit-samples {} (saved {})",
                cfg.fit_samples, saved.fit_samples
            ));
        }
        // #116: fit-only row screening is saved search state too.
        if fit_rows_set && cfg.fit_rows != saved.fit_rows {
            conflicts.push(format!(
                "--fit-rows {} (saved {})",
                cfg.fit_rows, saved.fit_rows
            ));
        }
        // #255: the fallback bound is saved because reaching it changes which
        // candidates keep device constants.
        if fit_polish_set && cfg.fit_polish != saved.fit_polish {
            conflicts.push(format!(
                "--fit-polish {} (saved {})",
                cfg.fit_polish.name(),
                saved.fit_polish.name()
            ));
        }
        if fit_fallback_max_set && cfg.fit_fallback_max != saved.fit_fallback_max {
            conflicts.push(format!(
                "--fit-fallback-max {} (saved {})",
                cfg.fit_fallback_max, saved.fit_fallback_max
            ));
        }
        // #465: the selection scheme is saved search state.
        if selection_set && cfg.selection != saved.selection {
            conflicts.push(format!(
                "--selection {} (saved {})",
                cfg.selection.as_str(),
                saved.selection.as_str()
            ));
        }
        // #495: the band decides which finalists get the exact refit.
        if refit_band_set && cfg.refit_band != saved.refit_band {
            conflicts.push(format!(
                "--refit-band {} (saved {})",
                cfg.refit_band, saved.refit_band
            ));
        }
        // #470: linear scaling is saved search state.
        if linear_scaling_set && !saved.linear_scaling {
            conflicts.push("--linear-scaling (saved linear_scaling is false)".to_string());
        }
        // #471: the node-path variation settings are saved search state.
        {
            if mutation_mix_set && cfg.mutation_mix != saved.mutation_mix {
                conflicts.push(format!(
                    "--mutation-mix {} (saved {})",
                    cfg.mutation_mix.as_str(),
                    saved.mutation_mix.as_str()
                ));
            }
            if tree_arity_set && cfg.tree_arity != saved.tree_arity {
                conflicts.push(format!(
                    "--tree-arity {} (saved {})",
                    cfg.tree_arity.as_str(),
                    saved.tree_arity.as_str()
                ));
            }
            if crossover_internal_set && cfg.crossover_internal != saved.crossover_internal {
                conflicts.push(format!(
                    "--crossover-internal {} (saved {})",
                    cfg.crossover_internal, saved.crossover_internal
                ));
            }
            if adaptive_mix_set && cfg.adaptive_mix != saved.adaptive_mix {
                conflicts.push(format!("--adaptive-mix (saved adaptive_mix is {})", saved.adaptive_mix));
            }
        }
        // #76: front feedback is saved search state.
        if front_parents_set && cfg.front_parents != saved.front_parents {
            conflicts.push(format!(
                "--no-front-parents (saved front_parents is {})",
                saved.front_parents
            ));
        }
        if reject_constant_set && cfg.reject_constant_offspring != saved.reject_constant_offspring {
            conflicts.push(format!(
                "--reject-constant-offspring (saved reject_constant_offspring is {})",
                saved.reject_constant_offspring
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
        //
        // #140: evaluator choices are *not* saved search settings, but they
        // must survive the resume. Previously `cfg = saved` silently reset
        // `--gpu` to the default CPU, which is exactly how a GPU run was
        // continued on the CPU.
        let requested_gpu = cfg.gpu;
        let requested_simd = cfg.simd;
        let requested_dispatch = cfg.dispatch;
        let mut effective = saved;
        if generations_set {
            effective.generations = cfg.generations;
        }
        // #256: the time budget limits one process, not the trajectory, so a
        // resumed run takes the budget it is given (a stopped run is resumed
        // exactly to get more time).
        if time_budget_set {
            effective.time_budget = cfg.time_budget;
        }
        effective.gpu = requested_gpu;
        effective.simd = requested_simd;
        effective.dispatch = requested_dispatch;
        fit_request = None;
        cfg = effective;
        resumed_generation = Some(cp.generations);
        resume_checkpoint = Some(cp);
    }

    // #141: probe the device once, before the dataset is loaded. The probe
    // allocates nothing; the search opens its own resident context later. This
    // is the single runtime decision behind the startup summary and JSON.
    if dispatch_mode == rcsr::dispatch::DispatchMode::Device && gpu_explicit != Some(true) {
        eprintln!("error: --dispatch device needs --gpu (and a build with --features cuda)");
        std::process::exit(EXIT_USAGE);
    }
    cfg.dispatch = dispatch_mode;

    let gpu_probe = backend::probe();
    match gpu_explicit {
        Some(true) => {
            if !gpu_probe.available() {
                let why = gpu_probe
                    .reason
                    .as_deref()
                    .unwrap_or("no usable CUDA device");
                eprintln!(
                    "error: --gpu was requested but no GPU is available ({})",
                    why
                );
                eprintln!("       omit --gpu (or pass --cpu) to run on the CPU");
                std::process::exit(EXIT_USAGE);
            }
            cfg.gpu = true;
        }
        Some(false) => cfg.gpu = false,
        None => {
            cfg.gpu = gpu_probe.available();
            // A non-CUDA build is CPU by design; only a CUDA-capable binary
            // that failed to find a device needs a fallback warning.
            if !cfg.gpu && !quiet && cfg!(feature = "cuda") {
                eprintln!(
                    "# gpu: unavailable ({}); using the CPU",
                    gpu_probe.reason.as_deref().unwrap_or("no usable CUDA device")
                );
            }
        }
    }

    // User-defined building blocks (issue #10). The definitions are parsed
    // before the data so a syntax error is reported as a usage error; the
    // thread-local registry makes them visible to the parser, the evaluator,
    // the fitter and every search island.
    let blocks = match expr::parse_blocks(&function_defs) {
        Ok(blocks) => blocks,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_USAGE);
        }
    };
    if blocks.is_empty() && cfg.ops.iter().any(|op| op.is_custom()) {
        eprintln!(
            "error: --ops names a custom slot but no --function was supplied"
        );
        std::process::exit(EXIT_USAGE);
    }
    if !blocks.is_empty() {
        expr::set_blocks(blocks.clone());
        for slot in 0..blocks.len() {
            let op = Op::Custom(slot);
            if !cfg.ops.contains(&op) {
                cfg.ops.push(op);
            }
        }
    }
    if let Some(spec) = &op_complexity_spec {
        let spec = expand_block_aliases(spec, &blocks);
        if let Err(e) = cfg.complexity_model.apply_op_costs(&spec) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_USAGE);
        }
    }

    // #140: phase timings for the run manifest start here, before the table is
    // read, so the manifest covers the whole run rather than just the search.
    let run_start = std::time::Instant::now();
    // #256: the manifest's `start_utc` is the process start, not the moment the
    // first data hash finished.
    let run_start_utc = manifest::utc_now();
    let data_path = match data_path {
        Some(p) => p,
        None => {
            eprintln!("error: --data is required\n\n{}", usage());
            std::process::exit(EXIT_USAGE);
        }
    };
    // #112: sampling is a store-only, explicit experiment. Refusing the flag
    // on a CSV is honest; silently ignoring it would hide the difference.
    if sample_rows.is_some()
        && (data_path == "-" || !store::is_store_path(&data_path))
    {
        eprintln!(
            "error: --sample-rows applies only to a .rcsb store; a CSV is loaded in full \
             or refused"
        );
        std::process::exit(EXIT_USAGE);
    }
    // #112: `--force` means the full resident load and `--sample-rows` means a
    // stratified sample; honouring one silently would hide the other.
    if force && sample_rows.is_some() {
        eprintln!(
            "error: --force and --sample-rows contradict each other (--force loads the full \
             input, --sample-rows asks for a sample); pass one of them"
        );
        std::process::exit(EXIT_USAGE);
    }
    let target = match target {
        Some(t) => t,
        None => {
            eprintln!("error: --target is required\n\n{}", usage());
            std::process::exit(EXIT_USAGE);
        }
    };

    // #15: `--target` accepts a comma-separated list. One name is the historical
    // single-output path, byte for byte; two or more enable the shared-shape
    // multi-output mode.
    let target_names: Vec<String> = target
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    if target_names.is_empty() {
        eprintln!("error: --target needs at least one column name");
        std::process::exit(EXIT_USAGE);
    }
    for (i, name) in target_names.iter().enumerate() {
        if target_names[..i].iter().any(|n| n == name) {
            eprintln!("error: --target names column '{}' more than once", name);
            std::process::exit(EXIT_USAGE);
        }
    }
    let target_refs: Vec<&str> = target_names.iter().map(|s| s.as_str()).collect();
    let multi_output = target_names.len() > 1;
    if multi_output {
        // Honest limits: the single-target diagnostics and report builders
        // assume one target column. Refusing is better than a half-filled report.
        let unsupported = [
            (eval_expr.is_some(), "--eval-expr"),
            (!seed_exprs.is_empty() || !seed_files.is_empty(), "--seed-expr/--seed-file"),
            (save_path.is_some(), "--save"),
            (resume_path.is_some(), "--resume"),
            (cv_mode.is_some(), "--cv"),
            (null_runs > 0, "--null-runs"),
            (validation_frac > 0.0 || select_by_arg.is_some(), "--validation-frac/--select-by"),
            (require_validation, "--require-validation"),
            (null_block > 0, "--null-block"),
            (bootstrap_resamples > 0, "--bootstrap-error"),
            (cfg.metric == Metric::Slope, "--metric slope (its row pairs follow one target column)"),
            (cfg.linear_scaling, "--linear-scaling"),
            (stability_runs > 0, "--stability"),
            (report.is_some(), "--report"),
            (run_manifest_path.is_some(), "--run-manifest"),
        ];
        for (present, flag) in unsupported {
            if present {
                eprintln!(
                    "error: {} is not supported with multiple targets (--target y1,y2); \
                     see docs/multi-output.md",
                    flag
                );
                std::process::exit(EXIT_USAGE);
            }
        }
    }

    if cfg.linear_scaling {
        let refusal = if !cfg.metric.is_least_squares() {
            Some("linear scaling needs a least-squares metric (squared, aic, aicc, bic)")
        } else if cfg.fit_iters == 0 {
            Some("linear scaling needs constant fitting (--fit-iters > 0, no --no-fit)")
        } else if cfg.variation == rcsr::search::VariationMode::Postfix {
            Some("linear scaling needs constant fitting, which --variation postfix does not do")
        } else {
            None
        };
        if let Some(reason) = refusal {
            eprintln!("error: --linear-scaling: {}", reason);
            std::process::exit(EXIT_USAGE);
        }
    }

    if let Some(flag) = cfg.node_variation_flag() {
        if cfg.variation == rcsr::search::VariationMode::Postfix {
            eprintln!(
                "error: {}: the flat postfix engine has its own edits; use --variation node or auto",
                flag
            );
            std::process::exit(EXIT_USAGE);
        }
    }

    // #254: the declared target spec, resolved before any row is read. D13 runs
    // here when the inputs are named; otherwise right after they are resolved.
    let mut target_resolved: Option<target::Resolved> = if multi_output {
        if target_spec_path.is_some() {
            eprintln!("error: --target-spec is not supported with multiple targets (--target y1,y2)");
            std::process::exit(EXIT_USAGE);
        }
        None
    } else {
        match target::resolve(&data_path, target_spec_path.as_deref(), &target_names[0]) {
            Ok(r) => r,
            Err(target::ResolveError::Usage(e)) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
            Err(target::ResolveError::Input(e)) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    };
    if allow_target_structure && target_resolved.is_none() {
        eprintln!("error: --allow-target-structure needs a target spec (--target-spec or <data>.targets.json)");
        std::process::exit(EXIT_USAGE);
    }
    if require_clean_inputs && target_resolved.is_none() {
        eprintln!("error: --require-clean-inputs needs a target spec (--target-spec or <data>.targets.json)");
        std::process::exit(EXIT_USAGE);
    }
    let mut target_check: Option<target::Check> = None;
    if let (Some(r), false) = (&target_resolved, vars.is_empty()) {
        target_check = Some(enforce_target_check(&r.spec, &vars, allow_target_structure, require_clean_inputs));
    }

    // #71/#112: the guard and the backing choice happen before any resident
    // Table or search sample is built. A CSV estimate uses the table's shape
    // (rows x columns) once the text has been read, never `33 x file size`.
    let detected_budget = store::detected_memory_budget();
    let budget = max_memory.or(detected_budget);
    // A CSV refusal uses only a hard wall (`--max-memory`, `ulimit -v` or a
    // cgroup limit), never the soft MemAvailable figure.
    let guard_budget = max_memory.or_else(store::detected_hard_budget);
    // The search touches every explicitly named variable plus every target and
    // the weight column. With no `--vars` the projection is not known yet, so
    // the estimate assumes the whole header.
    let projected_cols = if vars.is_empty() {
        0
    } else {
        vars.len() + target_names.len() + usize::from(weight_name.is_some())
    };
    let mut out_of_core_note: Option<String> = None;
    let mut csv_shape: Option<data::CsvShape> = None;
    let mut store_resident = false;
    // #112: the store estimator's terms and whether the full rows were used.
    // They are reported in the preparation block and carried into the run
    // manifest/dev log, so a sampled run is never silent.
    let mut store_memory_estimate: Option<memory::MemoryEstimate> = None;
    let mut store_full_rows: Option<bool> = None;
    let mut store_sample_rows: Option<usize> = None;
    // #294: the resolved read projection used by this run, recorded in the
    // manifest. `None` means the historical full-column load.
    let mut store_projection: Option<store::Projection> = None;
    let mut projection_note: Option<String> = None;
    let mut view_record: Option<registry::View> = None;
    let lag_menu: Vec<usize> = match &lag_set_src {
        Some(s) => {
            if lags > 0 {
                eprintln!("error: --lags and --lag-set are alternatives; give one");
                std::process::exit(EXIT_USAGE);
            }
            match registry::parse_lag_menu(s) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
            }
        }
        None => Vec::new(),
    };
    if project_src.is_some() && !store::is_store_path(&data_path) {
        eprintln!("error: --project needs a .rcsb store with registered feature columns");
        std::process::exit(EXIT_USAGE);
    }
    let table = if data_path == "-" {
        // #126: stdin cannot be reopened for the streaming parser's two passes,
        // so spool it to a temporary file. A whole-file `String` is never built.
        let spooled = match spool_stdin() {
            Ok(t) => t,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let shape = match data::csv_shape_path(spooled.path_str()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        csv_shape = Some(shape);
        if let Some(limit) = guard_budget {
            // The parser streams from the spool, so the text is not resident.
            let estimate = data::estimate_csv_stream_bytes(shape, projected_cols);
            if estimate > limit && !force {
                let _ = std::fs::remove_file(&spooled.path);
                refuse_for_memory("<stdin>", shape, estimate, limit, Some(projected_cols));
            }
        }
        data::read_table_with(spooled.path_str(), skip_malformed)
    } else if store::is_store_path(&data_path) {
        match store::FileStore::open(&data_path) {
            Ok(fs) => {
                // #260: a --project view fixes --vars from the registry; a store
                // with a schema sidecar refuses a name it does not list.
                if let Some(src) = &project_src {
                    if !vars.is_empty() {
                        eprintln!("error: --project selects the variables; do not combine it with --vars");
                        std::process::exit(EXIT_USAGE);
                    }
                    let size = if src == "auto" {
                        registry::DEFAULT_VIEW_SIZE
                    } else {
                        src.parse().unwrap_or_else(|_| {
                            eprintln!("error: --project takes a view size or auto, got '{}'", src);
                            std::process::exit(EXIT_USAGE);
                        })
                    };
                    let excluded: Vec<&str> = target_names
                        .iter()
                        .map(|s| s.as_str())
                        .chain(time_name.as_deref())
                        .chain(group_name.as_deref())
                        .chain(weight_name.as_deref())
                        .collect();
                    let candidates: Vec<String> = fs
                        .header()
                        .columns()
                        .iter()
                        .filter(|c| c.numeric && !excluded.contains(&c.name.as_str()))
                        .map(|c| c.name.clone())
                        .collect();
                    let construction: Vec<(String, String)> = target_resolved
                        .as_ref()
                        .map(|r| {
                            r.spec
                                .construction_columns
                                .iter()
                                .map(|c| (c.clone(), format!("construction column of {}", r.spec.column)))
                                .collect()
                        })
                        .unwrap_or_default();
                    match registry::select_view_excluding(&candidates, size, cfg.seed as u64, view_index, &construction) {
                        Ok(v) => {
                            eprintln!(
                                "view {}: {} columns from {} families (seed {}, view {})",
                                v.id,
                                v.columns.len(),
                                v.families.len(),
                                v.seed,
                                v.view
                            );
                            vars = v.columns.clone();
                            view_record = Some(v);
                        }
                        Err(e) => {
                            eprintln!("error: {}", e);
                            let shortfall = matches!(e, registry::ViewError::Shortfall { .. });
                            std::process::exit(if shortfall { EXIT_USAGE } else { EXIT_INPUT });
                        }
                    }
                }
                if let Err(e) = registry::check_vars(&data_path, &vars) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
                // #112: project from the header before reading a group. The
                // search copies only variables, targets and the weight; time
                // and group are scanned during preparation.
                let target_set: Vec<&str> = target_names.iter().map(|s| s.as_str()).collect();
                let time_name = time_name.as_deref();
                let group_name = group_name.as_deref();
                let weight = weight_name.as_deref();
                let mut used_cols = if vars.is_empty() {
                    fs.header()
                        .columns()
                        .iter()
                        .filter(|c| {
                            c.numeric
                                && !target_set.iter().any(|t| c.name.as_str() == *t)
                                && Some(c.name.as_str()) != time_name
                                && Some(c.name.as_str()) != group_name
                                && Some(c.name.as_str()) != weight
                        })
                        .count()
                } else {
                    vars.len()
                };
                used_cols = used_cols
                    .saturating_add(target_names.len())
                    .saturating_add(usize::from(weight_name.is_some()));
                let scan_cols = usize::from(time_name.is_some())
                    .saturating_add(usize::from(group_name.is_some()));
                let mutable_prep = matches!(
                    fill,
                    data::FillMethod::Mean
                        | data::FillMethod::Median
                        | data::FillMethod::Zero
                        | data::FillMethod::Previous
                        | data::FillMethod::Interpolate
                ) || outlier_iqr.is_some()
                    || smooth > 0
                    || smooth_centred;
                // #294: project the read when the caller named the split
                // columns and the table shape is fully determined by names.
                // Filters, dtype expansion and sampling still need the whole
                // header, so they keep the full-column path.
                let group_like_present = fs.header().names().iter().any(|n| {
                    matches!(
                        n.to_ascii_lowercase().as_str(),
                        "symbol" | "ticker" | "asset" | "pair" | "group" | "code"
                    )
                });
                let projection_off_reason: Option<&str> = if no_project {
                    Some("--no-project")
                } else if vars.is_empty() {
                    Some("no --vars")
                } else if time_name.is_none() {
                    Some("no --time")
                } else if group_name.is_none() && group_like_present {
                    Some("no --group-by")
                } else if dtype_src.is_some() {
                    Some("--dtype")
                } else if sample_rows.is_some() {
                    Some("--sample-rows")
                } else {
                    None
                };
                let may_project = projection_off_reason.is_none();
                projection_note = Some(projection_off_reason.unwrap_or("on").to_string());
                if may_project {
                    let mut pnames: Vec<String> = vars.clone();
                    pnames.extend(target_names.iter().cloned());
                    if let Some(w) = weight_name.as_deref() {
                        pnames.push(w.to_string());
                    }
                    if let Some(t) = time_name.as_deref() {
                        pnames.push(t.to_string());
                    }
                    if let Some(g) = group_name.as_deref() {
                        pnames.push(g.to_string());
                    }
                    if let Some(c) = target_resolved.as_ref().and_then(|r| r.spec.scale_column()) {
                        pnames.push(c.to_string());
                    }
                    let numeric_names: Vec<String> = fs
                        .header()
                        .columns()
                        .iter()
                        .filter(|c| c.numeric)
                        .map(|c| c.name.clone())
                        .collect();
                    for (flag, src) in [
                        ("--filter", filter_src.as_deref()),
                        ("--exclude", exclude_src.as_deref()),
                        ("--train-select", train_select_src.as_deref()),
                        ("--test-select", test_select_src.as_deref()),
                    ] {
                        let Some(src) = src else { continue };
                        match data::row_expr_columns(src, &numeric_names) {
                            Ok(cols) => pnames.extend(cols),
                            Err(e) => {
                                eprintln!("error: cannot parse {}: {}", flag, e);
                                eprintln!("  expression: {}", src);
                                std::process::exit(EXIT_USAGE);
                            }
                        }
                    }
                    let trading_requested = objective_arg
                        .as_deref()
                        .is_some_and(|o| rcsr::trading::Objective::parse(o).is_ok_and(|o| o.is_trading()));
                    if trading_requested {
                        if let Some(c) = target_resolved.as_ref().and_then(|r| r.spec.return_column.as_deref()) {
                            if !fs.header().names().iter().any(|n| n == c) {
                                eprintln!(
                                    "error: target spec '{}' declares return_column '{}', which is not in {}",
                                    target_names[0], c, data_path
                                );
                                std::process::exit(EXIT_INPUT);
                            }
                            pnames.push(c.to_string());
                        }
                        if let Some(f) = trade_funding.as_deref() {
                            pnames.push(f.to_string());
                        }
                    }
                    pnames.sort();
                    pnames.dedup();
                    if let Ok(p) = store::Projection::resolve_owned(&fs, &pnames) {
                        store_projection = Some(p);
                    } else {
                        projection_note = Some("projection unresolved".to_string());
                    }
                }
                let mem_opts = memory::EstimateOptions {
                    used_cols,
                    scan_cols,
                    projection: store_projection.as_ref().map(|p| p.indices()),
                    gpu: cfg.gpu,
                    gpu_fit: cfg.gpu
                        && cfg.fit_iters > 0
                        && (cfg.fit_strategy.defers_to_gpu()
                            || (fit_request.is_some() && cfg.metric.is_least_squares())),
                    mutable_prep,
                    simd: cfg.simd,
                };
                let estimate = fs.estimate_resident(&mem_opts);
                let rows = fs.n_rows();
                let limit = budget.unwrap_or(estimate.total_bytes);
                // The default is the full load whenever it fits. `--force`
                // promotes an over-budget store to resident; `--sample-rows`
                // is the only way to sample, even when the full store fits.
                let full = force || (sample_rows.is_none() && budget.map_or(true, |b| estimate.total_bytes <= b));
                if full {
                    store_resident = true;
                    store_memory_estimate = Some(estimate);
                    store_full_rows = Some(true);
                    match &store_projection {
                        Some(p) => fs.to_table_projected(p),
                        None => fs.to_table(),
                    }
                } else if let Some(requested) = sample_rows {
                    // Sampling is explicit. Never let `--sample-rows` be a
                    // quieter way to exceed the budget: a request that covers
                    // every row is a full load, so it must pass the same guard
                    // as the default path.
                    let n = requested.min(rows);
                    if n >= rows {
                        if estimate.total_bytes > limit {
                            refuse_store_full_with_sample_request(
                                &data_path,
                                &fs,
                                &estimate,
                                requested,
                                limit,
                            );
                        }
                        store_resident = true;
                        store_memory_estimate = Some(estimate);
                        store_full_rows = Some(true);
                        match &store_projection {
                            Some(p) => fs.to_table_projected(p),
                            None => fs.to_table(),
                        }
                    } else {
                        let sampled = estimate.with_sampled_rows(n, &fs.memory_shape(), &mem_opts);
                        if sampled.total_bytes > limit && !force {
                            refuse_store_sample_for_memory(
                                &data_path, &fs, &sampled, requested, limit,
                            );
                        }
                        store_memory_estimate = Some(sampled);
                        store_full_rows = Some(false);
                        store_sample_rows = Some(n);
                        match fs.sample_table(n, cfg.seed as u64) {
                            Ok(t) => {
                                out_of_core_note = Some(format!(
                                    "sampling (explicit): '{}' has {} rows x {} cols; \
                                     --sample-rows {} selected {} rows (seed {}), full_rows=false; \
                                     estimated sampled footprint {}",
                                    data_path,
                                    rows,
                                    fs.n_cols(),
                                    requested,
                                    t.len(),
                                    cfg.seed,
                                    human_bytes(sampled.total_bytes)
                                ));
                                Ok(t)
                            }
                            Err(e) => Err(e),
                        }
                    }
                } else {
                    // No automatic sample: a store that does not fit is
                    // refused with its exact terms, and the user chooses
                    // --force or --sample-rows explicitly.
                    refuse_store_for_memory(&data_path, &fs, &estimate, limit);
                }
            }
            Err(e) => Err(e),
        }
    } else {
        // #126: measure and parse by streaming the file; never `read_to_string`.
        let shape = match data::csv_shape_path(&data_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        csv_shape = Some(shape);
        if let Some(limit) = guard_budget {
            let estimate = data::estimate_csv_stream_bytes(shape, projected_cols);
            if estimate > limit && !force {
                refuse_for_memory(&data_path, shape, estimate, limit, Some(projected_cols));
            }
        }
        data::read_table_with(&data_path, skip_malformed)
    };
    let table = match table {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    let table_loaded_s = run_start.elapsed().as_secs_f64();

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

    // #15: resolve every target up front. The single-target branch keeps the
    // historical error text exactly; the multi branch names the bad target.
    let target_indices: Vec<usize> = if multi_output {
        target_names
            .iter()
            .map(|name| match table.index_of(name) {
                Ok(j) => j,
                Err(_) => {
                    eprintln!(
                        "error: target column '{}' not found (columns: {:?})",
                        name, table.names
                    );
                    std::process::exit(EXIT_INPUT);
                }
            })
            .collect()
    } else {
        match table.index_of(&target) {
            Ok(j) => vec![j],
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    };
    let target_idx = target_indices[0];

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

    // Default inputs: numeric columns that are neither a target nor a
    // split-only column.
    let vars_explicit = !vars.is_empty();
    if vars.is_empty() {
        vars = (0..table.names.len())
            .filter(|&j| {
                table.numeric[j]
                    && !target_names.iter().any(|t| t == &table.names[j]) // #15
                    && Some(j) != time_col
                    && Some(j) != group_col
                    && Some(j) != weight_col
                    && !registry::lookup(&table.names[j]).is_some_and(|r| r.family == "label")
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
        if target_names.iter().any(|t| t == name) {
            eprintln!(
                "error: column '{}' is {}target; --dtype applies to input variables",
                name,
                if multi_output { "a " } else { "the " }
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
    if let Err(e) = registry::check_no_labels(&vars) {
        eprintln!("error: {}", e);
        std::process::exit(EXIT_LEAKAGE);
    }
    if let (Some(r), None) = (&target_resolved, &target_check) {
        target_check = Some(enforce_target_check(&r.spec, &vars, allow_target_structure, require_clean_inputs));
    }

    let known_vars: Vec<usize> = vars.iter().filter_map(|v| table.index_of(v).ok()).collect();
    let exclusion = |a: Option<usize>, b: Option<usize>| -> Vec<usize> {
        // #15: every target is a split-only column, never a model input.
        let mut e = target_indices.clone();
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
    // #71: the out-of-core note belongs to the run report. It is deliberately
    // explicit that a sampled run is not a full-data run.
    if let Some(note) = out_of_core_note {
        notes.push(note);
    }

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
    // #129: resolve --gap. Rows pass through; wall time becomes
    // ceil(seconds / median_dt) and is never silently rounded down to zero.
    let mut gap = 0usize;
    if let Some(src) = gap_src.as_deref() {
        match parse_gap_spec(src, &table, time_col, time_unit) {
            Ok((rows, note)) => {
                gap = rows;
                if let Some(n) = note {
                    notes.push(n);
                }
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    if gap > 0 && (train_select.is_some() || test_select.is_some()) {
        notes.push(
            "--gap is dropped because explicit --train-select/--test-select replace \
             positional splitting; add the embargo to the selection expressions instead"
                .to_string(),
        );
        gap = 0;
    }
    if gap > 0 && !matches!(mode, data::SplitMode::Temporal | data::SplitMode::TemporalGroup) {
        notes.push("--gap only applies to temporal splits; ignored".to_string());
        gap = 0;
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
    let table = if lags > 0 || !lag_menu.is_empty() || history_frac > 0.0 {
        let lag_spec = data::LagSpec {
            lags,
            history_frac,
            gap_break,
            time: data::TimeSpec {
                date_only,
                unit: time_unit,
                tz,
            },
        };
        let lagged = if lag_menu.is_empty() {
            data::add_lags(&table, &vars, &lag_spec, time_col, group_col)
        } else {
            data::add_lags_menu(&table, &vars, &lag_spec, &lag_menu, time_col, group_col)
        };
        match lagged {
            Ok((t2, v2, rep)) => {
                eprintln!("nvars {} -> {}", rep.nvars_before, rep.nvars_after);
                notes.push(format!("nvars {} -> {}", rep.nvars_before, rep.nvars_after));
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

    // #263: the exact long/flat P&L objective. Every refusal is a usage error
    // raised before any data is read.
    let objective = match rcsr::trading::Objective::parse(objective_arg.as_deref().unwrap_or("mse")) {
        Ok(o) => o,
        Err(e) => trading_usage(&e),
    };
    if !objective.is_trading() {
        if let Some(flag) = trading_flags.first() {
            trading_usage(&format!("{} needs --objective pnl, sharpe or log_growth", flag));
        }
    }
    let fitness_pnl = match fitness_arg.as_deref() {
        None | Some("metric") => false,
        Some("pnl") => true,
        Some(other) => trading_usage(&format!("--fitness must be metric or pnl, got '{}'", other)),
    };
    let pnl_blocks: usize = match pnl_blocks_arg.as_deref() {
        None => 1,
        Some(t) => match t.trim().parse::<usize>() {
            Ok(k) if (1..=rcsr::trading::FITNESS_MAX_BLOCKS).contains(&k) => k,
            _ => trading_usage(&format!(
                "--pnl-blocks must be an integer from 1 to {}, got '{}'",
                rcsr::trading::FITNESS_MAX_BLOCKS, t
            )),
        },
    };
    let pnl_lambda: f64 = match pnl_lambda_arg.as_deref() {
        None => 1.0,
        Some(t) => match t.trim().parse::<f64>() {
            Ok(l) if l.is_finite() && l >= 0.0 => l,
            _ => trading_usage(&format!("--pnl-block-lambda must be a finite number >= 0, got '{}'", t)),
        },
    };
    if !fitness_pnl {
        for (flag, given) in [("--pnl-blocks", pnl_blocks_arg.is_some()), ("--pnl-block-lambda", pnl_lambda_arg.is_some())] {
            if given {
                trading_usage(&format!("{} needs --fitness pnl", flag));
            }
        }
    }
    let mut trading_spec: Option<rcsr::trading::TradingSpec> = None;
    let mut trading_cols: Option<data::TradingColumns> = None;
    let mut trading_cost: Option<target::CostDecl> = None;
    if objective.is_trading() {
        let name = objective.name();
        if multi_output {
            trading_usage("--objective is not supported with multiple targets (--target y1,y2)");
        }
        let position_map = match position_arg.as_deref() {
            None => rcsr::trading::PositionMap::Threshold,
            Some("short") => trading_usage(
                "--position short and --allow-short are refused: the book is long-only, \
                 there is no short P&L, borrow or funding",
            ),
            Some(p) => match rcsr::trading::PositionMap::parse(p) {
                Ok(m) => m,
                Err(e) => trading_usage(&e),
            },
        };
        if allow_short {
            trading_usage(
                "--position short and --allow-short are refused: the book is long-only, \
                 there is no short P&L, borrow or funding",
            );
        }
        let top_tercile = position_map == rcsr::trading::PositionMap::TopTercile;
        if top_tercile {
            if !matches!(mode, data::SplitMode::Temporal) {
                trading_usage(
                    "--position top_tercile needs --split temporal: another split can put part of a \
                     timestamp's cross-section on the other side of the split",
                );
            }
            if trading_flags.contains(&"--threshold") {
                trading_usage("--position top_tercile has no threshold: remove --threshold");
            }
        }
        if !matches!(mode, data::SplitMode::Temporal | data::SplitMode::TemporalGroup)
            || time_col.is_none()
        {
            trading_usage(&format!(
                "--objective {} needs a temporal split: add --time <col> and --split temporal",
                name
            ));
        }
        let ts = match &target_resolved {
            Some(r) => &r.spec,
            None => trading_usage(&format!(
                "--objective {} needs a forward-return target: declare it with --target-spec",
                name
            )),
        };
        if ts.kind != "forward_return" && ts.kind != target::KIND_XS_RANK {
            trading_usage(&format!(
                "--objective {} needs a forward_return or xs_rank target, the spec declares '{}'",
                name, ts.kind
            ));
        }
        if ts.kind == target::KIND_XS_RANK && !top_tercile {
            trading_usage(&format!(
                "--objective {} on an xs_rank target prices the top-tercile book: add --position top_tercile",
                name
            ));
        }
        let spec_declares_cost = ts.per_side_cost_bps().is_some();
        if trade_cost_bps.is_none() && trade_cost_override.is_none() && !spec_declares_cost {
            trading_usage(&format!(
                "--objective {} needs a cost assumption: a target spec with cost_bps, or --cost-bps <f> per side (0 only when stated)",
                name
            ));
        }
        let (cost, cost_source) =
            match target::resolve_cost(Some(ts), trade_cost_bps, trade_cost_override, 0.0) {
                Ok(v) => v,
                Err(e) => trading_usage(&e),
            };
        let mut decl = target::CostDecl::new(Some(ts), cost, cost_source);
        decl.per_side_bps = cost + trade_slippage_bps;
        trading_cost = Some(decl);
        if cost < 0.0 || trade_slippage_bps < 0.0 {
            trading_usage("--cost-bps and --slippage-bps must be non-negative");
        }
        if ts.cost_in_label && cost + trade_slippage_bps != 0.0 {
            trading_usage(
                "cost_already_in_label: the target spec says the label is net of cost, \
                 so a non-zero --cost-bps or --slippage-bps would charge it twice",
            );
        }
        let unit_scale = matches!(&ts.scale, target::Scale::Constant { value } if *value == 1.0);
        if ts.return_column.is_none() && (ts.offset != 0.0 || ts.clip.is_some() || !unit_scale) {
            trading_usage(
                "the target is scaled, clipped or offset: declare a return_column in the \
                 target spec so the P&L uses realised returns",
            );
        }
        trading_cols = Some(data::TradingColumns {
            return_column: ts.return_column.clone(),
            log_returns: ts.return_units == "log",
            funding_column: trade_funding.clone(),
            train: fitness_pnl, // #480
            cross_section: top_tercile,
        });
        let tspec = rcsr::trading::TradingSpec {
            cost_bps: cost,
            slippage_bps: trade_slippage_bps,
            funding_col: trade_funding.clone(),
            cash_return: trade_cash_return,
            min_trades: trade_min_trades,
            min_holding: trade_min_holding,
            threshold: trade_threshold,
            max_position: trade_max_position,
            position: position_map,
            ..Default::default()
        };
        if let Err(e) = tspec.validate() {
            trading_usage(&e);
        }
        trading_spec = Some(tspec);
    }
    if fitness_pnl {
        // #480: every refusal names its flag; scoring needs every training row, in order.
        if !objective.is_trading() {
            trading_usage("--fitness pnl needs --objective pnl, sharpe or log_growth");
        }
        if multi_output {
            trading_usage("--fitness pnl is not supported with multiple targets (--target y1,y2)");
        }
        if cfg.predictor_samples.is_some() {
            trading_usage("--fitness pnl is not supported with --predictor-samples: the long/flat book needs every training row in order");
        }
        if cfg.fit_samples > 0 {
            trading_usage("--fitness pnl is not supported with --fit-samples: the long/flat book scores every training row (use --fit-rows to fit on fewer rows)");
        }
        if implicit_mode || cfg.metric.is_implicit() {
            trading_usage("--fitness pnl is not supported with --implicit or an implicit metric");
        }
        if trade_min_holding > 1 {
            trading_usage("--fitness pnl is not supported with --min-holding > 1: the fitness is the plain threshold rule");
        }
        if trading_spec.as_ref().is_some_and(|s| s.position != rcsr::trading::PositionMap::Threshold) {
            trading_usage("--fitness pnl is not supported with --position top_tercile: the fitness is the plain threshold rule");
        }
        if trade_max_position != 1.0 {
            trading_usage("--fitness pnl needs --max-position 1: the fitness is the unit long/flat book");
        }
    }

    let resolved_lockbox = rcsr::lockbox::resolve_with(lockbox_arg.as_deref(), lockbox_embargo_arg.as_deref(), &data_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e.message);
        std::process::exit(e.exit);
    });
    let declared_lockbox = resolved_lockbox.declared();
    // Several targets resolve no single spec: the boundary purge is the largest
    // required purge of the targets that have a spec entry.
    let multi_boundary_purge: Option<f64> = if multi_output {
        let mut largest: Option<f64> = None;
        for name in &target_names {
            match target::resolve(&data_path, None, name) {
                Ok(Some(r)) => largest = Some(largest.map_or(r.spec.required_purge_seconds(), |l| l.max(r.spec.required_purge_seconds()))),
                Ok(None) => {}
                Err(target::ResolveError::Usage(e)) => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_USAGE);
                }
                Err(target::ResolveError::Input(e)) => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            }
        }
        largest
    } else {
        None
    };
    let spec = data::SplitSpec {
        mode,
        time_col,
        group_col,
        test_frac,
        gap,
        validation_frac,
        folds,
        train_select,
        test_select,
        exclude,
        time: data::TimeSpec {
            date_only,
            unit: time_unit,
            tz,
        },
        allow_lockbox,
        declared_lockbox,
        purge_seconds: target_resolved.as_ref().map(|r| r.spec.required_purge_seconds()),
        boundary_purge_seconds: multi_boundary_purge,
        trading: trading_cols.clone(),
        canonical_train_order: fitness_pnl, // #480
        scale_column: target_resolved
            .as_ref()
            .and_then(|r| r.spec.scale_column())
            .map(str::to_string),
    };
    let prep = data::PrepSpec {
        fill,
        outlier_iqr,
        smooth,
        smooth_centred,
        allow_interpolate,
        center,
        scale,
        filter,
        dtypes: dtypes.clone(),
    };
    // #116: `--fit-strategy stream` can read the training rows back from a
    // resident `.rcsb` store instead of keeping a second resident copy. This is
    // exact only when the store's raw rows *are* the prepared population and
    // the preparation is affine (centre/scale) or absent; smoothing, fill,
    // filters, exclusions and lags rewrite values or row ids and fall back to
    // the resident sample. The row ids cost one `usize` per training row and
    // are requested only for this arm.
    let stream_store_candidate = cfg.fit_strategy == fit::FitStrategy::Stream
        && data_path != "-"
        && store::is_store_path(&data_path)
        && store_resident
        && !multi_output
        && !implicit_mode
        && lags == 0
        && lag_menu.is_empty()
        && history_frac == 0.0
        && prep.fill == data::FillMethod::None
        && prep.outlier_iqr.is_none()
        && prep.smooth == 0
        && prep.filter.is_none()
        && spec.exclude.is_none();
    let mut rng = Mt::new(cfg.seed);
    let mut prepared = match if stream_store_candidate {
        data::prepare_full_targets_weighted_with_indices(
            &table,
            &vars,
            &target_refs,
            weight_name.as_deref(),
            &spec,
            &mut rng,
            &prep,
        )
    } else {
        data::prepare_full_targets_weighted_with(
            &table,
            &vars,
            &target_refs,
            weight_name.as_deref(),
            &spec,
            &mut rng,
            &prep,
        )
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(if e.starts_with("target spec") {
                EXIT_USAGE
            } else {
                EXIT_INPUT
            });
        }
    };
    notes.extend(prepared.report.notes.iter().cloned());
    let prepared_s = run_start.elapsed().as_secs_f64();
    // #145, #518: cross-validation draws its own folds, so the split guard
    // cannot see them. Every row of the population (after --exclude and
    // --filter) is a training or test row of some fold; refuse before the
    // search. The declared locked test is checked even with --allow-lockbox.
    let cv_keep: Option<Vec<bool>> = if cv_mode.is_some() {
        let keep = match data::population_mask(&table, &spec, &prep) {
            Ok(k) => k,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        if let Err(e) = data::check_cv_lockbox(&table, &spec, &keep, allow_lockbox) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        Some(keep)
    } else {
        None
    };

    if cfg.metric == Metric::Slope {
        match prepared.require_slope_pairs() {
            Ok(pairs) => notes.push(format!(
                "slope: differencing {} adjacent-row pair(s) among {} training row(s)",
                pairs,
                prepared.train.len()
            )),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
        }
    }

    // #468: the long/flat book earns the full per-row return, so rows of one
    // group must not be closer than the label horizon.
    let target_horizon = target_resolved.as_ref().map_or(0.0, |r| r.spec.horizon_seconds);
    let mut min_group_step: Option<f64> = None;
    if let (Some(_), Some(split), Some(r)) = (&trading_spec, prepared.trading.as_ref(), target_resolved.as_ref()) {
        let horizon = r.spec.horizon_seconds;
        for slice in split.validation.iter().chain(std::iter::once(&split.test)) {
            match rcsr::trading::check_return_spacing(&slice.rows, horizon) {
                Ok(step) => {
                    min_group_step = match (min_group_step, step) {
                        (Some(a), Some(b)) => Some(a.min(b)),
                        (a, b) => a.or(b),
                    }
                }
                Err(e) => trading_usage(&e),
            }
        }
    }
    // #480: the exact long/flat fitness book on the canonical training rows.
    let fitness_book: Option<std::sync::Arc<rcsr::trading::FitnessBook>> = if fitness_pnl {
        let slice = prepared.trading.as_ref().and_then(|t| t.train.as_ref());
        let (Some(tspec), Some(slice)) = (trading_spec.as_ref(), slice) else {
            trading_usage("--fitness pnl found no training trading rows");
        };
        match rcsr::trading::FitnessBook::with_blocks(slice, tspec, target_horizon, pnl_blocks, pnl_lambda) {
            Ok(book) if book.len() == prepared.train.len() => Some(std::sync::Arc::new(book)),
            Ok(book) => trading_usage(&format!(
                "--fitness pnl: {} training trading rows for {} training rows",
                book.len(),
                prepared.train.len()
            )),
            Err(e) => trading_usage(&format!("--fitness pnl: {}", e)),
        }
    } else {
        None
    };
    if let Some(book) = &fitness_book {
        cfg.fitness = search::Fitness::LongPnl(book.clone());
        if weight_name.is_some() && !quiet {
            eprintln!("# note: --weight weights the constant fit (--metric) only; the long/flat regret counts every row once");
        }
    } else if !cfg.fitness.is_metric() {
        trading_usage(
            "the checkpoint was written with --fitness pnl: pass --fitness pnl and the same objective, cost and split flags",
        );
    }
    if let Some(cp) = &resume_checkpoint {
        if cp.config.fitness.key() != cfg.fitness.key() {
            trading_usage(&format!(
                "--resume conflicts with the saved search settings: fitness {} (saved {}); the book depends on the cost, threshold, min-trades and training rows",
                cfg.fitness.key(),
                cp.config.fitness.key()
            ));
        }
    }
    // #254: the zero-parameter rung and, under the override, the proxy table.
    let target_active: Option<target::Active> = match (target_resolved.take(), target_check.take()) {
        (Some(r), Some(mut check)) => {
            if check.status == "overridden" {
                if let Some(col) = r.spec.scale_column() {
                    check.proxies = Some(target::proxy_table(&prepared, &vars, col, 5));
                }
            }
            let rung = diagnostics::target_structure_rung(&r.spec, &prepared, cfg.metric);
            let purge_line = target::purge_line(&r.spec, &prepared.report);
            let label_report = target::label_report_for(&data_path, &r.spec.column);
            Some(target::Active {
                spec: r.spec,
                source: r.source,
                file_sha256: r.file_sha256,
                check,
                rung,
                purge_line,
                label_report,
            })
        }
        _ => None,
    };
    let target_sha: Option<String> = target_active.as_ref().map(|a| a.spec.sha256());
    let target_spec_json = target_active
        .as_ref()
        .map(|a| a.json())
        .unwrap_or_else(|| "null".to_string());
    let target_rung_json = target_active
        .as_ref()
        .map(|a| a.rung_json())
        .unwrap_or_else(|| "null".to_string());

    // #257: the validation gates run on time-ordered runs, with a target spec,
    // or when a validation slice is required.
    if require_validation && prepared.validation.is_none() {
        eprintln!(
            "error: --require-validation: no validation slice; add --validation-frac <f> \
             (the gates select a member only from a validation slice)"
        );
        std::process::exit(EXIT_USAGE);
    }
    let time_ordered = matches!(mode, data::SplitMode::Temporal | data::SplitMode::TemporalGroup)
        || time_col.is_some();
    let gates_active = time_ordered || target_active.is_some() || require_validation;
    let gate_opts = rcsr::gate_run::Options {
        metric: cfg.metric,
        z: equivalence_z,
        alpha: rcsr::gate_run::DEFAULT_ALPHA,
        noise_lag: match noise_lag_arg.as_deref() {
            Some("auto") | None => None,
            Some(n) => n.parse::<usize>().ok(),
        },
        time_ordered,
    };
    if gates_active && time_ordered && prepared.validation.is_none() {
        eprintln!(
            "warning: time-ordered run without a validation slice: no member is selected \
             (verdict unvalidated); add --validation-frac <f>"
        );
    }

    if stream_store_candidate {
        let resolved = (|| -> Result<fit::StoreSpec, String> {
            let fs = std::sync::Arc::new(store::FileStore::open(&data_path)?);
            let columns = vars
                .iter()
                .map(|v| table.index_of(v))
                .collect::<Result<Vec<_>, _>>()?;
            let mut stream_spec = fit::StoreSpec::new(fs, columns, table.index_of(target_refs[0])?)?;
            if let Some(name) = weight_name.as_deref() {
                stream_spec = stream_spec.with_weight(table.index_of(name)?)?;
            }
            stream_spec = stream_spec
                .with_standardisation(prepared.means.clone(), prepared.stds.clone())?
                .with_rows(prepared.train_indices.clone())?;
            Ok(stream_spec)
        })();
        match resolved {
            Ok(stream_spec) => {
                cfg.fit_stream_source = Some(std::sync::Arc::new(stream_spec));
                notes.push(format!(
                    "stream: fitting from '{}' group by group ({} training rows, O(p²) resident state)",
                    data_path,
                    prepared.train_indices.len()
                ));
            }
            Err(e) => notes.push(format!(
                "stream: disk row source unavailable ({}); using the resident training sample",
                e
            )),
        }
    }

    // #112: report the guard's estimate and its rows x columns basis in the
    // normal preparation block, not only when the guard refuses the file. Use
    // the resolved projection (after --dtype/lag expansion) here.
    if let Some(shape) = csv_shape {
        let used = vars.len()
            + target_refs.len()
            + usize::from(weight_col.is_some());
        prepared.prep.memory_estimate = Some(data::csv_estimate(shape, used));
    }
    if let Some(estimate) = store_memory_estimate {
        // #112: the store estimate used at the guard time is reported even on
        // a successful full load, so the decision has an observable basis.
        prepared.prep.store_estimate = Some(estimate);
        prepared.prep.store_projection = projection_note.clone();
        prepared.prep.full_rows = store_full_rows;
        prepared.prep.sampled_rows = store_sample_rows;
    }

    // #113: a binary metric needs a binary target. Reject a multi-valued target
    // here, before any search, instead of silently merging its classes at the
    // midpoint and reporting a number that means something else.
    if cfg.metric.is_classification() {
        for j in 0..prepared.train.target_count() {
            let name = target_names.get(j).map(|s| s.as_str()).unwrap_or(target.as_str());
            // Check the union of both sides: a distinct class that happens to
            // sit only in the held-out rows still makes the target multi-valued.
            // #289: a zero-weight row is invisible to the metric, so it must
            // not be able to make an otherwise binary target look
            // multi-valued. Filter each side by its own weights before the
            // distinct-value check.
            let weighted_values = |sample: &data::Sample| -> Vec<f64> {
                let target = sample.target(j);
                if sample.w.is_empty() {
                    target.to_vec()
                } else {
                    target
                        .iter()
                        .zip(&sample.w)
                        .filter(|(_, w)| **w > 0.0)
                        .map(|(v, _)| *v)
                        .collect()
                }
            };
            let mut values = weighted_values(&prepared.train);
            values.extend(weighted_values(&prepared.test));
            if let Some(why) = metrics::classification_target_error(&values) {
                eprintln!(
                    "error: --metric {} treats '{}' as a binary target, but {}",
                    cfg.metric.name(),
                    name,
                    why
                );
                eprintln!(
                    "  collapse the target to two classes yourself, or pick a regression metric \
                     (absolute, squared, log_squashed, ...)"
                );
                std::process::exit(EXIT_INPUT);
            }
        }
    }
    // #111: selection criterion. Default to the validation slice when one is
    // present, otherwise train (the historical behaviour). Asking for
    // validation without a slice is a usage error rather than a silent
    // fallback to train.
    let mut select_by: Option<rcsr::selection::SelectBy> = match select_by_arg.as_deref() {
        Some(src) => match rcsr::selection::SelectBy::parse(src) {
            Ok(by) => Some(by),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
        },
        // No explicit criterion and no slice: keep the historical archive
        // order (complexity, then train error) rather than re-ranking.
        None => None,
    };
    if select_by.is_none() && prepared.validation.is_some() {
        select_by = Some(rcsr::selection::SelectBy::Validation);
    }
    if objective.is_trading() {
        if select_by_arg
            .as_deref()
            .is_some_and(|s| !matches!(s.trim().to_ascii_lowercase().as_str(), "validation" | "val"))
        {
            trading_usage("--objective selects on the validation slice; use --select-by validation or drop it");
        }
        if prepared.validation.is_none() {
            trading_usage(&format!(
                "--objective {} needs a validation slice: add --validation-frac <f>",
                objective.name()
            ));
        }
        select_by = Some(rcsr::selection::SelectBy::ValidationObjective);
    }
    if select_by == Some(rcsr::selection::SelectBy::Validation) && prepared.validation.is_none() {
        eprintln!(
            "error: --select-by validation needs a validation slice;              add --validation-frac <f>"
        );
        std::process::exit(EXIT_USAGE);
    }

    // #14: implicit relations. The target becomes one more input (normalised
    // with its training mean/std, so denormalisation prints original units) and
    // the target itself becomes the constant 0. From here on every search,
    // diagnostic and report sees an ordinary sample with one extra input; the
    // tree machinery is unchanged.
    if implicit_mode {
        let (mean, std) = data::make_implicit(&mut prepared);
        vars.push(target.clone());
        notes.push(format!(
            "implicit: target '{}' is input {} (mean {:.6}, std {:.6}); searching \
             f({}) = 0 with metric {}",
            target,
            vars.len(),
            mean,
            std,
            vars.join(", "),
            cfg.metric.name()
        ));
        if smooth < 2 {
            notes.push(
                "warning: --implicit on unsmoothed data: the level-set distance divides by a \
                 gradient estimated from row-to-row noise. Smooth the series first (for \
                 example --smooth 5); see docs/implicit.md"
                    .to_string(),
            );
        }
    }
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
    let mut backend_change: Option<(String, String)> = None;
    if let Some(cp) = &resume_checkpoint {
        if let Err(e) = cp.check_target_spec(target_sha.as_deref()) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_USAGE);
        }
        if let Err(e) = cp.check_data(&prepared.train, &prepared.test) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        // #140: a GPU checkpoint must not be continued on the CPU (or vice
        // versa) without an explicit override. The effective backend mirrors
        // the search's own capability checks.
        let current_backend = search::effective_backend(&cfg, &prepared.train);
        if let Err(e) = cp.check_backend(current_backend, allow_backend_change) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        // #256: the binary, commit, device, driver and thread count are part
        // of what produced the saved state.
        let current_identity =
            rcsr::provenance::BuildIdentity::current(current_backend == Backend::Gpu);
        if let Err(e) = cp.check_identity(&current_identity, allow_build_change) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        if let Some(saved) = &cp.identity {
            let diff = saved.differences(&current_identity);
            if !diff.is_empty() {
                eprintln!(
                    "# checkpoint: build identity changed ({}) (allowed by --allow-build-change)",
                    diff.join("; ")
                );
            }
        }
        if cp.backend != Backend::Unknown && cp.backend != current_backend {
            backend_change = Some((
                cp.backend.as_str().to_string(),
                current_backend.as_str().to_string(),
            ));
            eprintln!(
                "# checkpoint: backend changed {} -> {} (allowed by --allow-backend-change)",
                cp.backend.as_str(),
                current_backend.as_str()
            );
        }
    }
    let effective_backend = search::effective_backend(&cfg, &prepared.train);

    // #223: resolve the fit arm once, before the memory plan and the run
    // records. A CPU-only run keeps today's arm and prints nothing new.
    let fit_resolution: Option<fit::FitResolution> = fit_request
        .map(|request| search::resolve_fit_strategy(&cfg, &prepared.train, request));
    if let Some(r) = &fit_resolution {
        cfg.fit_strategy = r.strategy;
        cfg.fit_admission = r.admission;
        if !quiet && effective_backend == Backend::Gpu {
            if r.strategy == fit::FitStrategy::Gpu {
                eprintln!(
                    "# gpu-fit: {} -> gpu ({}, admit {})",
                    r.source,
                    cfg.metric.name(),
                    r.admission.name()
                );
            } else {
                eprintln!(
                    "# gpu-fit: {} -> {} ({})",
                    r.source,
                    r.strategy.name(),
                    r.reason
                );
            }
        }
    }
    let fit_provenance: Option<(String, String)> = fit_resolution
        .as_ref()
        .map(|r| (r.source.to_string(), r.reason.clone()));

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
        let pnl_kind = if fitness_book.is_some() { ",\"best_error_kind\":\"long_pnl_regret\"" } else { "" };
        cfg.progress = Some(std::sync::Arc::new(move |p: search::Progress| {
            let line = format!(
                "{{\"event\":\"progress\",\"generation\":{},\"total\":{},\"best_error\":{}{},\"pareto_members\":{},\"convergence\":{}}}",
                p.generation,
                p.total_generations,
                jnum(p.best_error),
                pnl_kind,
                p.pareto_members,
                // #18: confidence, stability, maturity, percent_converged.
                p.convergence.json()
            );
            if to_stdout {
                println!("{}", line);
            } else {
                eprintln!("{}", line);
            }
        }));
    }

    let search_start = std::time::Instant::now();
    // #140: publish the run manifest before the search so a killed process
    // still leaves evidence of what it was doing. It is updated atomically at
    // the end with the end time and phase timings.
    let data_prepare_secs = run_start.elapsed().as_secs_f64();
    let mut manifest_state: Option<manifest::Manifest> = None;
    if let Some(path) = &run_manifest_path {
        let m = manifest::collect(
            VERSION,
            &data_path,
            rcsr::checkpoint::fingerprint(&prepared.train),
            rcsr::checkpoint::fingerprint(&prepared.test),
            &cfg,
            effective_backend.as_str(),
            resume_path.clone(),
            backend_change.clone(),
            store_full_rows,
            store_sample_rows,
            store_memory_estimate.map(|m| m.total_bytes),
        );
        let mut m = m;
        m.start_utc = run_start_utc.clone();
        m.fit_resolution = fit_provenance.clone();
        m.fit_subsample = search::fit_subsample_record(&cfg, &prepared.train);
        if let Some(p) = &store_projection {
            m.projection = Some(manifest::ProjectionRecord {
                hash: p.hash_hex(),
                columns: p.names(),
                indices: p.indices(),
            });
        }
        if let Some(v) = &view_record {
            m.view = Some(v.json());
        }
        if let Some(a) = &target_active {
            m.target_spec = Some((
                a.source.name().to_string(),
                a.source.path().to_string(),
                a.spec.sha256(),
            ));
        }
        if let Err(e) = m.write(path, None, &[("data_prepare", data_prepare_secs)]) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
        manifest_state = Some(m);
    }

    let start = search_start;
    let split_line = format!(
        "split={} | selector: {} | train rows: {} | test rows: {}{}{}{}",
        prepared.report.mode,
        prepared.report.selector,
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
    if let Some(a) = &target_active {
        if format == Format::Text && !quiet {
            for line in a.text_lines() {
                eprintln!("{}", line);
            }
        }
        if let Some(w) = a.warning_line() {
            eprintln!("{}", w);
        }
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
        if prepared.prep.active()
            || !prepared.prep.time_unit.is_empty()
            || prepared.prep.memory_estimate.is_some()
        {
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
        if multi_output {
            // #15: one shared shape, one fitted constant vector per output.
            eprintln!(
                "# targets: {} = f({}) | {}",
                target_names.join(","),
                vars.join(","),
                split_line
            );
        } else {
            eprintln!(
                "# target: {} = f({}) | {}",
                target,
                vars.join(","),
                split_line
            );
        }
        let type_line: Vec<String> = variable_types
            .iter()
            .map(|v| format!("{} ({})", v.name, v.dtype.name()))
            .collect();
        eprintln!("# variables: {}", type_line.join(", "));
        // #85: say how many rows the fitter screened and that the reported
        // archive was refit on every training row.
        if cfg.fit_iters > 0 && cfg.fit_samples >= 2 && cfg.fit_samples < prepared.train.len() {
            let n = cfg.fit_samples;
            eprintln!(
                "# fit: constants screened on {} of {} training rows per generation; \
                 survivors refit on all {} rows",
                n,
                prepared.train.len(),
                prepared.train.len()
            );
        }
        // #86: state the effective screening size, and say so when a near-full
        // request was widened to the full sample instead of paying the
        // per-generation rebuild for a negligible fit saving.
        if let Some(requested) = cfg.predictor_samples {
            let rows = prepared.train.len();
            match search::effective_predictor_samples(&cfg, &prepared.train) {
                Some(n) => eprintln!(
                    "# predictor: {} of {} training rows per generation (requested {})",
                    n, rows, requested
                ),
                None if requested >= rows => eprintln!(
                    "# predictor: requested {} >= {} training rows; using all rows",
                    requested, rows
                ),
                None => eprintln!(
                    "# predictor: requested {} of {} rows is above the 60% break-even; using all {} rows",
                    requested, rows, rows
                ),
            }
        }
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

    // #26/#141: the backend note belongs to the CLI, not to the library.
    // `--quiet` promises an empty stderr for machine-readable output, so it
    // suppresses the note and the one-line resource summary.
    cfg.gpu_notes = !quiet;
    cfg.gpu_memory = gpu_memory;
    cfg.threads = threads; // #463
    // #293: the device memory plan runs before the search allocates anything.
    // A refused fit stage, or a refused eval stage under an explicit `--gpu`,
    // is a usage error naming the bytes and the smallest fix.
    #[cfg(feature = "cuda")]
    if cfg.gpu {
        if let Some(plan) = search::device_plan_for(&cfg, &prepared.train) {
            if let (Some(refusal), Some(text)) = (&plan.refusal, plan.refusal_text()) {
                if refusal.stage == "fit" || gpu_explicit == Some(true) {
                    eprintln!("error: {}", text);
                    std::process::exit(EXIT_USAGE);
                }
            }
        }
    }

    // #141: one line stating the resources this run will use. `rows`/`cols`
    // are the training table (the search's hot loop); test rows are in the
    // split object. The backend is the effective one (#140's own check), so a
    // configuration the device cannot score is not reported as a GPU run
    // (#236); the search still prints its own `# gpu: off (...)` note.
    if !quiet {
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cols = prepared.train.x.first().map(|r| r.len()).unwrap_or(0);
        let blocked = search::backend_blocker_reason(&cfg, &prepared.train);
        let (kind, device, reason) = if effective_backend == Backend::Gpu && gpu_probe.available() {
            ("gpu", gpu_probe.device.as_deref().unwrap_or("unknown"), None)
        } else {
            (
                "cpu",
                "-",
                gpu_explicit
                    .filter(|explicit| !*explicit)
                    .map(|_| "--cpu/--no-gpu")
                    .or(gpu_probe.reason.as_deref())
                    .or(blocked),
            )
        };
        match reason {
            Some(why) => eprintln!(
                "# resource: backend={} device={} threads={} rows={} cols={} ({})",
                kind, device, threads, prepared.train.len(), cols, why
            ),
            None => eprintln!(
                "# resource: backend={} device={} threads={} rows={} cols={}",
                kind, device, threads, prepared.train.len(), cols
            ),
        }
    }

    // #211: open the development log and write its first event before the
    // search, so a killed process still records what it was doing.
    //
    // The log is **on by default** at `summary` level: that is the in-engine
    // replacement for the external run monitor. `--no-dev-log` turns it off, and
    // with it off `cfg.dev_log` stays `None` so the search skips every
    // measurement the log would need. `--report benchmark` repeats one search
    // several times, so it runs without a log rather than interleaving runs.
    let explicit_benchmark = report == Some(ReportKind::Benchmark);
    let dev_target: Option<(String, bool)> = if dev_log_disabled || explicit_benchmark {
        None
    } else {
        match &dev_log_path {
            Some(p) => Some((p.clone(), true)),
            None => Some((
                default_dev_log_path(json_out.as_deref(), save_path.as_deref()),
                false,
            )),
        }
    };
    let dev_every = dev_log_every.unwrap_or_else(|| dev_log_level.default_every(cfg.generations));
    let dev_sink: Option<DevSink> = match dev_target {
        Some((path, explicit)) => {
            let opened = DevSink::open(&path, dev_log_max_bytes, dev_every, dev_log_level);
            match opened {
                Ok(sink) => {
                    // #236: the log's backend must be the effective one, not
                    // the startup probe's; a metric the kernel cannot score
                    // runs on the CPU and the evidence has to say so.
                    let (backend_name, device) = if effective_backend == Backend::Gpu {
                        ("gpu", gpu_probe.device.clone())
                    } else {
                        ("cpu", None)
                    };
                    let info = dev_run_start(
                        &command,
                        &data_path,
                        &prepared,
                        &cfg,
                        backend_name,
                        device,
                        dev_log_level,
                        dev_every,
                        dev_log_max_bytes,
                        resume_path.clone(),
                        fit_provenance.clone(),
                    );
                    sink.write(&rcsr::devlog::run_start_json(&info), true);
                    if !quiet {
                        eprintln!(
                            "# dev-log: {} ({} level, every {} generation(s), cap {})",
                            path,
                            dev_log_level.as_str(),
                            dev_every,
                            human_bytes(dev_log_max_bytes)
                        );
                    }
                    cfg.dev_log = Some(sink.callback());
                    Some(sink)
                }
                // An explicit path is a promise: refuse to run without it. The
                // default path is a convenience, so a read-only or missing
                // directory must not stop the search.
                Err(e) if explicit => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
                Err(e) => {
                    if !quiet {
                        eprintln!(
                            "warning: {}; continuing without a development log (use --dev-log \
                             <path> or --no-dev-log to choose)",
                            e
                        );
                    }
                    None
                }
            }
        }
        None => None,
    };

    // #15: multi-output reporting has its own shape (one entry per target);
    // the single-output path below is untouched.
    if multi_output {
        run_multi_output(
            &cfg,
            &prepared,
            &target_names,
            &vars,
            format,
            quiet,
            top,
            out_path,
            json_out.clone(),
            run_start,
            start,
            dev_sink.as_ref(),
        );
        std::process::exit(EXIT_OK);
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
            implicit_mode,
            format,
            quiet,
            search_start,
            json_out.as_deref(),
            target_active.as_ref(),
        );
        std::process::exit(EXIT_OK);
    }

    // #116: arm counts are process-wide diagnostics; take a delta around this
    // run so repeated runs in one process (`--report benchmark`) stay readable.
    let fit_before = fit::fit_arm_counts_full();
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
                if let Err(e) = cp.save_for(&path, target_sha.as_deref()) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
                let generation = cp.generations;
                last_saved = Some((generation, path.clone()));
                // #211: the checkpoint file is saved; record its size and hash.
                if let Some(sink) = &dev_sink {
                    sink.checkpoint(generation, &path);
                }
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
    let mut outcome = match outcome_result {
        Ok(outcome) => outcome,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    if let Some(g) = outcome.gpu.as_mut() {
        g.ledger.record_outside(rcsr::gpu_stats::Stage::Io, table_loaded_s);
        g.ledger
            .record_outside(rcsr::gpu_stats::Stage::Features, prepared_s - table_loaded_s);
    }
    let search_seconds = search_start.elapsed().as_secs_f64();
    // #211: close the development log now that the search has returned: the
    // final archive, the stop reason, the wall and the process peak RSS. It is
    // written before diagnostics so a later failure still leaves a complete
    // log; the report phase is therefore not part of `wall` here (it is in the
    // run manifest's phase timings).
    if let Some(sink) = &dev_sink {
        sink.end(
            &outcome,
            run_start.elapsed().as_secs_f64(),
            search_seconds,
            json_out.clone(),
        );
    }
    // Compatibility: `elapsed_seconds` has always meant the search phase.
    let elapsed = search_seconds;
    let diagnostics_start = std::time::Instant::now();

    // #116: opt-in arm counts. This line is diagnostics only and never feeds
    // back into the search.
    if cfg.fit_report {
        let counts = fit::fit_arm_counts_full().since(fit_before);
        let total = counts.total();
        let share = if total == 0 {
            0.0
        } else {
            100.0 * counts.linear as f64 / total as f64
        };
        eprintln!(
            "# fit: strategy={} linear={} varpro={} stream={} exact={} ({:.1}% linear, {} calls; gpu={} fallbacks={} refits={})",
            cfg.fit_strategy.name(),
            counts.linear,
            counts.varpro,
            counts.stream,
            counts.exact,
            share,
            total,
            counts.gpu,
            counts.gpu_fallbacks,
            counts.gpu_refits
        );
        eprintln!("# fit: {}", fit::backend_description());
        // #228: the signed-bias projection is the only non-squared metric with
        // an exact closed-form arm. Its labels say why a candidate left the arm.
        let signed: u64 = fit::signed_fit_counts().iter().map(|(_, n)| n).sum();
        if signed > 0 {
            let labels = fit::signed_fit_counts()
                .iter()
                .map(|(label, n)| format!("{}={}", label, n))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!("# fit: signed projection [{}]", labels);
        }
        if let Some(g) = &outcome.gpu {
            let causes = search::FitFallbackCause::ALL
                .iter()
                .map(|c| format!("{}={}", c.label(), g.fit_fallback_by_cause.get(c.label()).copied().unwrap_or(0)))
                .collect::<Vec<_>>()
                .join(" ");
            let labels = g
                .fit_status_by_label
                .iter()
                .map(|(label, n)| format!("{}={}", label, n))
                .collect::<Vec<_>>()
                .join(" ");
            eprintln!(
                "# fit: gpu-fit screened={} admitted_wide={} admitted={} gpu={} fallback={} [{}] labels=[{}] refit_candidates={} refits={} refit_s={:.3} rescore_s={:.3} fallback_s={:.3} bounded={}/{} workers={}",
                g.fit_screened,
                g.fit_admitted_wide,
                g.fit_attempts,
                g.fit_gpu,
                g.fit_fallback,
                causes,
                labels,
                g.fit_refit_candidates,
                g.fit_refits,
                g.fit_refit_s,
                g.fit_rescore_s,
                g.fit_fallback_ms / 1000.0,
                g.fit_fallback_bounded,
                g.fit_fallback_max,
                g.fit_fallback_workers
            );
        }
    }

    // Seeded solutions, reported once each. The search holds them in
    // normalised-input space; show the original-units form, exactly like a
    // Pareto row. A seed above --max-complexity is still kept, and saying so
    // here means the rule is never a surprise.
    let seed_rows: Vec<(String, f64, f64)> = outcome
        .seeds
        .iter()
        .map(|s| {
            let tree = expr::simplify_display(&expr::denormalize(
                &expr::simplify_display(&s.tree),
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
        // #14: an implicit model is printed as `0 = f(...)`.
        let lhs = if implicit_mode { "0" } else { target.as_str() };
        for (expression, complexity, error) in &seed_rows {
            eprintln!(
                "# seeded: complexity={} train_error={}  {} = {}",
                complexity,
                fmt_err(*error),
                lhs,
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
    // #257: Driscoll-Kraay constant errors on fit-train keys and lag.
    let gate_train_keys: Option<(Vec<f64>, usize)> = if gates_active {
        let keys: Vec<f64> = match (time_ordered, prepared.keys.as_ref()) {
            (true, Some(k)) => k.train.clone(),
            _ => (0..prepared.train.len()).map(|i| i as f64).collect(),
        };
        let (lag, _) = rcsr::gate_run::lag_for(
            &keys,
            &gate_opts,
            target_active.as_ref().map(|a| &a.spec),
        );
        Some((keys, lag))
    } else {
        None
    };
    let mut rows: Vec<Row> = sorted
        .iter()
        .enumerate()
        .map(|(k, s)| {
            let den = expr::simplify_display(&expr::denormalize(
                &expr::simplify_display(&s.tree),
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
            let validation_error = match &prepared.validation {
                Some(v) => search::eval_error(&s.tree, v, cfg.metric),
                None => f64::NAN,
            };
            let objective_value = match (&trading_spec, &prepared.validation, prepared.trading.as_ref().and_then(|t| t.validation.as_ref())) {
                (Some(tspec), Some(v), Some(slice)) => {
                    let ctx = prepared.trading.as_ref().and_then(|t| t.validation_context.as_ref());
                    match trading_book(&s.tree, &v.x, slice, ctx, tspec) {
                        Ok(stats) => objective.score(&stats, tspec.min_trades),
                        Err(_) => f64::INFINITY,
                    }
                }
                _ => f64::NAN,
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
                archive_rank: k + 1,
                complexity: s.cost,
                k_constants: kconst,
                train_error,
                validation_error,
                test_error,
                objective_value,
                aicc,
                bic,
                dw,
                resid_tests,
                bootstrap,
                overfit,
                fold_errors,
                // Parameter uncertainty at the fitted solution (issue #22):
                // reuse the fitter's Jacobian on the training rows.
                constant_errors: if let Some((keys, lag)) = &gate_train_keys {
                    fit::constant_errors_robust(
                        &s.tree,
                        &prepared.train.x,
                        &prepared.train.y,
                        &prepared.train.w,
                        keys,
                        *lag,
                    )
                } else if prepared.train.w.is_empty() {
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
                gate_json: String::new(),
                fitness_json: fitness_book
                    .as_ref()
                    .map(|b| fitness_member_json(b, &pred))
                    .unwrap_or_default(),
            }
        })
        .collect();

    // #113: sign/step have a zero adjoint, so a constant behind one cannot be
    // fitted. Say so once, rather than letting the printed threshold look
    // optimised.
    if cfg.fit_iters > 0 {
        let mut zero_ops: Vec<&'static str> = Vec::new();
        for s in &sorted {
            if expr::count_constants(&s.tree) > 0 {
                for op in fit::zero_adjoint_ops(&s.tree) {
                    if !zero_ops.contains(&op) {
                        zero_ops.push(op);
                    }
                }
            }
        }
        if !zero_ops.is_empty() {
            eprintln!(
                "# warning: the Pareto front contains fitted constants behind {}: the adjoint \
                 is zero there, so their position is not optimised. Fit a smooth surrogate \
                 (e.g. logistic or tanh) and threshold only at inference, or pin the threshold \
                 as a fixed constant. See docs/optimisation.md.",
                zero_ops.join(" and ")
            );
        }
    }
    // #111: selection. Reorder the front by the chosen criterion and renumber
    // the display rank; the archive rank is kept for reports that need the
    // underlying search order. `test_error` is never a criterion. With no
    // criterion the historical archive order is untouched.
    if let Some(by) = select_by {
        let candidates: Vec<rcsr::selection::Candidate> = rows
            .iter()
            .map(|r| rcsr::selection::Candidate {
                train: r.train_error,
                validation: if objective.is_trading() {
                    Some(r.objective_value)
                } else if r.validation_error.is_finite() {
                    Some(r.validation_error)
                } else {
                    None
                },
                test: Some(r.test_error),
                aicc: r.aicc,
            })
            .collect();
        let order = rcsr::selection::order(&candidates, by);
        let reordered: Vec<Row> = order.into_iter().map(|i| rows[i].clone()).collect();
        rows = reordered;
        for (k, r) in rows.iter_mut().enumerate() {
            r.rank = k + 1;
        }
        if !quiet {
            eprintln!("# selection: {} (test errors are scoring-only)", by.name());
        }
    }

    // #263: the trading objective's finalist. Selection above read only the
    // validation slice; the test book is scored here, for the winner alone.
    let objective_report: Option<rcsr::trading::FinalistReport> = match (&trading_spec, rows.first()) {
        (Some(tspec), Some(top)) => {
            let tree = &sorted[top.archive_rank - 1].tree;
            let book = |sample: &data::Sample, slice: Option<&data::TradingSlice>, ctx: Option<&data::TradingContext>| match slice {
                Some(slice) => trading_book(tree, &sample.x, slice, ctx, tspec),
                None => Err("no trading rows".to_string()),
            };
            let split = prepared.trading.as_ref();
            let usable = rows.iter().filter(|r| r.objective_value.is_finite()).count();
            if usable == 0 && !quiet {
                eprintln!("# warning: no candidate has a usable P&L book on the validation slice");
            }
            Some(rcsr::trading::FinalistReport {
                objective,
                spec: tspec.clone(),
                min_group_step,
                horizon_seconds: target_horizon,
                cost: trading_cost.clone().unwrap_or_else(|| target::CostDecl::new(None, tspec.per_side_bps(), target::CostSource::Flag)),
                constant: rcsr::trading::is_constant(tree),
                candidates: rows.len(),
                usable,
                expression: top.expression.clone(),
                archive_rank: top.archive_rank,
                fit_metric: cfg.metric.name().to_string(),
                fit_error: top.train_error,
                validation_error: top.validation_error,
                test_error: top.test_error,
                validation_value: top.objective_value,
                search_fitness: fitness_book.as_ref().map(|_| "long_pnl_regret"),
                validation: match &prepared.validation {
                    Some(v) => book(v, split.and_then(|t| t.validation.as_ref()), split.and_then(|t| t.validation_context.as_ref())),
                    None => Err("no validation slice".to_string()),
                },
                test: book(&prepared.test, split.map(|t| &t.test), split.and_then(|t| t.test_context.as_ref())),
            })
        }
        _ => None,
    };
    if let Some(rep) = &objective_report {
        if format == Format::Text && !quiet {
            eprintln!("{}", rep.text());
        }
    }

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

    let mut diagnostics_seconds = diagnostics_start.elapsed().as_secs_f64();

    // Null baselines: the trivial models a real fit must beat (issue #23).
    // #69: the k-NN reference set is always all training rows; only the query
    // rows are optionally sampled by `--baseline-samples`.
    let baselines_start = std::time::Instant::now();
    let baseline_samples =
        diagnostics::resolve_baseline_samples(baseline_samples, prepared.train.len());
    let (baselines, linear_coef) = diagnostics::Baselines::fit_sampled_coef(
        &prepared.train,
        &fold_samples,
        cfg.metric,
        baseline_samples,
        cfg.seed,
    );
    let baselines_seconds = baselines_start.elapsed().as_secs_f64();

    // #257: validation gates on the archive members (archive order).
    let gate_out: Option<rcsr::gate_run::Output> = if gates_active {
        let out = {
            let mut by_archive: Vec<&Row> = rows.iter().collect();
            by_archive.sort_by_key(|r| r.archive_rank);
            let facts: Vec<rcsr::gate_run::MemberFacts> = sorted
                .iter()
                .zip(&by_archive)
                .enumerate()
                .map(|(k, (s, r))| rcsr::gate_run::MemberFacts {
                    archive_rank: k + 1,
                    cost: s.cost,
                    tree: &s.tree,
                    expression: &r.expression,
                    identified: r.constant_errors.as_ref().map_or(true, |c| c.identified()),
                    practically_identified: r
                        .constant_errors
                        .as_ref()
                        .and_then(|c| c.practically_identified()),
                })
                .collect();
            rcsr::gate_run::run(
                &prepared,
                target_active.as_ref().map(|a| &a.spec),
                &facts,
                &gate_opts,
                linear_coef.as_deref(),
            )
        };
        for (k, j) in out.member_json.iter().enumerate() {
            if let Some(r) = rows.iter_mut().find(|r| r.archive_rank == k + 1) {
                r.gate_json = j.clone();
            }
        }
        if format == Format::Text && !quiet {
            for line in &out.text {
                eprintln!("{}", line);
            }
        }
        Some(out)
    } else {
        None
    };
    let gate_seconds = gate_out.as_ref().map(|g| g.seconds);
    let gate_selected_archive_rank = gate_out.as_ref().and_then(|g| g.report.selected_rank);
    let diagnostics_start = std::time::Instant::now();

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
            cv_keep.as_deref(),
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
                let den = expr::simplify_display(&expr::denormalize(
                    &expr::simplify_display(&s.tree),
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
        // #211: a null control is a separate search; it must not write into the
        // real run's log.
        let saved_dev_log = cfg.dev_log.take();
        let mut best_null = Vec::with_capacity(null_runs);
        for r in 0..null_runs {
            let mut rng = Mt::new(cfg.seed ^ (0x51ED_2701u32.wrapping_mul(r as u32 + 1)));
            // #114: a block null is seeded from the same stream, so it is as
            // reproducible as the i.i.d. shuffle.
            let y = if null_block > 1 {
                diagnostics::block_permute(&prepared.train.y, null_block, &mut rng)
            } else {
                let mut y = prepared.train.y.clone();
                for j in (1..y.len()).rev() {
                    let t = rng.usize_below(j + 1);
                    y.swap(j, t);
                }
                y
            };
            let shuffled = data::Sample {
                x: prepared.train.x.clone(),
                y,
                w: prepared.train.w.clone(),
                ys: Vec::new(),
                pairs: None,
            };
            let out = search::run(&cfg, &shuffled, &shuffled);
            let b = out
                .archive
                .items
                .iter()
                .map(|s| s.error)
                .fold(f64::INFINITY, f64::min);
            if !quiet {
                eprintln!(
                    "# null run {}/{} ({}): best train error {:.6}",
                    r + 1,
                    null_runs,
                    if null_block > 1 {
                        format!("block {}", null_block)
                    } else {
                        "iid".to_string()
                    },
                    b
                );
            }
            best_null.push(b);
        }
        cfg.progress = saved_progress;
        cfg.dev_log = saved_dev_log; // #211
        null_control = Some(if null_block > 1 {
            diagnostics::NullControl::summarise_block(
                observed,
                best_null,
                null_block,
                diagnostics::block_count(prepared.train.len(), null_block),
            )
        } else {
            diagnostics::NullControl::summarise(observed, best_null)
        });
    }

    diagnostics_seconds += diagnostics_start.elapsed().as_secs_f64();

    let finite = rows.iter().any(|r| r.train_error.is_finite());
    let full_rows = rows.clone();

    // Optional Markdown report (issue #29), built from the full Pareto front so
    // `--top` only trims the ordinary table/TSV output below.
    let report_start = std::time::Instant::now();
    let mut report_md = report.map(|kind| match kind {
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
            select_by,
        ),
        ReportKind::Sensitivity => {
            let selected = if let Some(ar) = gate_selected_archive_rank {
                full_rows.iter().find(|r| r.archive_rank == ar)
            } else if select_by.is_some() {
                full_rows.first()
            } else {
                full_rows.iter().min_by(|a, b| {
                    a.train_error
                        .partial_cmp(&b.train_error)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            };
            let sorted = outcome.archive.sorted();
            let tree = selected
                .and_then(|r| r.archive_rank.checked_sub(1).and_then(|i| sorted.get(i)))
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
    let report_seconds = report_start.elapsed().as_secs_f64();
    // #134: measured independently, after all phases. Emitting the JSON/text is
    // not part of `total`; the phases cover everything up to this point.
    let timing = PhaseTimings {
        prep: (search_start - run_start).as_secs_f64(),
        search: search_seconds,
        baselines: baselines_seconds,
        diagnostics: diagnostics_seconds,
        report: report_seconds,
        total: run_start.elapsed().as_secs_f64(),
        // #118: the GPU full-path wall is a subset of `search`; it is not added
        // to `phases_sum`.
        gpu: outcome
            .gpu
            .as_ref()
            .map(|g| g.accounting.wall_s)
            .unwrap_or(0.0),
    };
    if let Some(md) = report_md.as_mut() {
        md.push_str(&time_markdown(&timing));
    }

    // #140: atomically finish the manifest with an end timestamp and timings.
    // #256: written after the diagnostics, so `end_utc` is the end of the
    // process work and `phases_seconds.total` is `timing.total_seconds`.
    if let (Some(path), Some(m)) = (&run_manifest_path, &mut manifest_state) {
        // #169: record the fit arm's ledger when the GPU fit ran.
        if outcome.gpu.is_some() {
            let (gpu, fallback, refits) = fit::gpu_fit_counts();
            m.fit = Some((gpu, fallback, refits));
        }
        // #290: the calibrated model and the route ledger are provenance, not
        // a result: record them whenever a run manifest was requested.
        m.dispatch = Some(outcome.dispatch.clone());
        let phases = [
            ("data_prepare", data_prepare_secs),
            ("search", timing.search),
            ("baselines", timing.baselines),
            ("diagnostics", timing.diagnostics),
            ("report", timing.report),
            ("total", timing.total),
        ];
        if let Err(e) = m.write(path, Some(&manifest::utc_now()), &phases) {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    }

    // #120: fixed-reference validation HV is computed before any display
    // truncation or selection reordering, from the full archive this run
    // produced.
    let validation_hv = validation_hypervolume(&rows, &prepared);
    let gate_mean_validation = gate_out
        .as_ref()
        .map(|g| g.mean_validation_json.clone())
        .unwrap_or_default();
    let gate_linear_validation = gate_out
        .as_ref()
        .map(|g| g.linear_validation_json.clone())
        .unwrap_or_default();
    let gate_rung_json = match gate_out.as_ref().and_then(|g| g.structure_json.as_ref()) {
        Some(extra) if target_rung_json.ends_with('}') => {
            let mut o = target_rung_json.clone();
            o.pop();
            o.push_str(extra);
            o.push('}');
            o
        }
        _ => target_rung_json.clone(),
    };
    let gate_top_json = gate_out.as_ref().map(|g| g.top_json.clone()).unwrap_or_default();

    if top > 0 && rows.len() > top {
        rows.truncate(top);
    }

    match format {
        Format::Text => {
            if !quiet {
                eprint_run_summary(&outcome, &cfg, &prepared.train, elapsed, &timing);
                qd_text(outcome.qd.as_ref());
                eprintln!(
                    "# validation hv ({}): {:.6} (ref error {}, {} front points; fixed reference; complexity span {}..{}, compare only runs with the same span)",
                    validation_hv.sample,
                    validation_hv.hypervolume,
                    jnum(validation_hv.reference_error),
                    validation_hv.points,
                    validation_hv.complexity_min.map(jnum).unwrap_or_else(|| "null".to_string()),
                    validation_hv.complexity_max.map(jnum).unwrap_or_else(|| "null".to_string())
                );
            }
            for r in &rows {
                // #14: an implicit model is printed as 0 = f(...).
                let lhs = if implicit_mode { "0" } else { target.as_str() };
                if prepared.validation.is_some() {
                    println!(
                        "{:>3}  complexity={:<3} train_err={:<13} val_err={:<13} test_err={:<13} aicc={:<12}  {} = {}",
                        r.rank,
                        r.complexity,
                        fmt_err(r.train_error),
                        fmt_err(r.validation_error),
                        fmt_err(r.test_error),
                        fmt_err(r.aicc),
                        lhs,
                        r.expression
                    );
                } else {
                    println!(
                        "{:>3}  complexity={:<3} train_err={:<13} test_err={:<13} aicc={:<12}  {} = {}",
                        r.rank,
                        r.complexity,
                        fmt_err(r.train_error),
                        fmt_err(r.test_error),
                        fmt_err(r.aicc),
                        lhs,
                        r.expression
                    );
                }
            }
            let selected_row = if select_by.is_some() {
                rows.first()
            } else {
                rows.iter().min_by(|a, b| {
                    a.train_error
                        .partial_cmp(&b.train_error)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
            };
            if let Some(r) = selected_row {
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
                    "# null control ({}, observed={:.6}): best={:.6} median={:.6} p={:.3} — {}",
                    if c.method == "block" {
                        format!(
                            "{} block-permutation runs, {} blocks of up to {} rows",
                            c.runs, c.blocks, c.block_len
                        )
                    } else {
                        format!("{} i.i.d. shuffled-target runs", c.runs)
                    },
                    c.observed,
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
                         ({} resamples, {} used, weight={:.3}, rows={}{})",
                        r.rank,
                        b.apparent,
                        b.oob,
                        b.estimate,
                        b.resamples,
                        b.used,
                        b.weight,
                        b.resample_rows,
                        // #92: make a capped estimator explicit.
                        if b.capped {
                            format!(" of {} (BLB cap b=n^0.7)", b.source_rows)
                        } else {
                            String::new()
                        }
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
            // #69: state whether queries were sampled and the train/test SEMs.
            if baselines.knn_capped {
                eprintln!(
                    "# knn baseline: reference_rows={} train_query_rows={} train_sem={:.3e} test_query_rows={} test_sem={:.3e} sampled=true sample_size={} sample_seed={}",
                    baselines.knn_reference_rows,
                    baselines.knn_train_query_rows,
                    baselines.knn_train_query_sem,
                    baselines.knn_test_query_rows,
                    baselines.knn_test_query_sem,
                    baselines.knn_sample_size,
                    baselines.knn_seed
                );
            } else {
                eprintln!(
                    "# knn baseline: reference_rows={} train_query_rows={} test_query_rows={} sampled=false",
                    baselines.knn_reference_rows,
                    baselines.knn_train_query_rows,
                    baselines.knn_test_query_rows
                );
            }
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
                    "# stability ({} resamples, {} found a solution, rows={}{}): modal form {:.0}%{}",
                    st.runs,
                    st.found,
                    st.resample_rows,
                    // #92: report the bounded BLB resample size.
                    if st.capped {
                        format!(" of {} (BLB cap b=n^0.7)", st.source_rows)
                    } else {
                        String::new()
                    },
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
            // #256: `--json-out` must not hide the line that reports GPU time.
            if !quiet {
                eprint_run_summary(&outcome, &cfg, &prepared.train, elapsed, &timing);
            }
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
            let validation_split_json = if prepared.report.validation_rows > 0 {
                format!(",\"validation_rows\":{}", prepared.report.validation_rows)
            } else {
                String::new()
            };
            let split_json = format!(
                "{{\"mode\":{},\"rows\":{},\"train_rows\":{}{},\"test_rows\":{},\"folds\":[{}],\"gap\":{},\"excluded\":{},\"overlap\":{},\"unassigned\":{},\"groups_total\":{},\"groups_train\":{},\"groups_test\":{},\"lags\":{},\"lag_menu\":[{}],\"nvars_before\":{},\"nvars_after\":{},\"series\":{},\"history_rows\":{},\"notes\":[{}]{}{}}}",
                jstr(prepared.report.mode),
                prepared.report.rows_total,
                prepared.train.len(),
                validation_split_json,
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
                lag_report.menu.iter().map(|k| k.to_string()).collect::<Vec<_>>().join(","),
                lag_report.nvars_before,
                lag_report.nvars_after,
                lag_report.series,
                lag_report.history_rows,
                notes.iter().map(|n| jstr(n)).collect::<Vec<_>>().join(","),
                target::split_purge_json(&prepared.report),
                split_time_json(prepared.keys.as_ref())
            );
            // #264: lockbox facts of the loaded table (rows, date range, read count).
            let lockbox_json = {
                let lockbox = match &spec.declared_lockbox {
                    Some(d) => d.lockbox,
                    None => rcsr::lockbox::Lockbox::crypto_2026h2(),
                };
                let times: Vec<i64> = match data::lockbox_stamps(&table, &spec) {
                    Ok(Some((_, keys))) => keys
                        .iter()
                        .filter(|v| v.is_finite())
                        .map(|v| v.floor() as i64)
                        .collect(),
                    _ => Vec::new(),
                };
                let overlap = lockbox.overlap(times.iter().copied());
                let reads = ledger_arg
                    .as_ref()
                    .map(|p| rcsr::protocol::Ledger::new(p).read_count().unwrap_or(0))
                    .unwrap_or(0);
                let evaluated = spec.declared_lockbox.is_none() && allow_lockbox && overlap.rows > 0;
                let mut j = lockbox.json(&overlap, allow_lockbox, false, evaluated);
                j.pop();
                j.push_str(&format!(",\"read_count\":{}", reads));
                j.push(',');
                j.push_str(&resolved_lockbox.report_fields(0, false, prepared.report.lockbox_checked));
                j.push('}');
                j
            };
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
                    let dimensions = format!("{}{}{}", dimensions, r.gate_json, r.fitness_json);
                    // Only a run that actually has a selection criterion gains
                    // the new fields; an ordinary run stays byte-identical.
                    let selection_json = if select_by.is_some() {
                        format!(
                            "\"archive_rank\":{},\"validation_error\":{},",
                            r.archive_rank,
                            jnum(r.validation_error)
                        )
                    } else {
                        String::new()
                    };
                    format!(
                        "{{\"rank\":{},{}\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{}{},\"stability\":{},\"expression\":{}}}",
                        r.rank,
                        selection_json,
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
                        constant_errors_json_g(&r.constant_errors, gates_active),
                        dimensions,
                        r.stability.map(jnum).unwrap_or_else(|| "null".to_string()),
                        jstr(&r.expression)
                    )
                })
                .collect();
            let null_json = match &null_control {
                Some(c) => format!(
                    "{{\"runs\":{},\"method\":{},\"block_len\":{},\"blocks\":{},\"observed\":{},\"best\":[{}],\"min\":{},\"median\":{},\"p_value\":{},\"verdict\":{}}}",
                    c.runs,
                    jstr(c.method),
                    c.block_len,
                    c.blocks,
                    jnum(c.observed),
                    c.best.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                    jnum(c.null_min),
                    jnum(c.null_median),
                    jnum(c.p_value),
                    jstr(c.verdict())
                ),
                None => "null".to_string(),
            };
            let baselines_json = format!(
                "{{\"mean\":{{\"train_error\":{},\"test_error\":{}{}}},\"linear\":{{\"train_error\":{},\"test_error\":{}{}}},\"knn\":{{\"k\":{},\"train_error\":{},\"test_error\":{},{}}},\"target_structure\":{}}}",
                jnum(baselines.mean.train_error),
                jnum(baselines.mean.test_error),
                gate_mean_validation,
                jnum(baselines.linear.train_error),
                jnum(baselines.linear.test_error),
                gate_linear_validation,
                baselines.knn_k,
                jnum(baselines.knn.train_error),
                jnum(baselines.knn.test_error),
                baselines.knn_json_extra(), // #69
                gate_rung_json
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
                        "{{\"runs\":{},\"found\":{},\"source_rows\":{},\"resample_rows\":{},\"capped\":{},\"forms\":[{}],\"variables\":[{}]}}",
                        st.runs,
                        st.found,
                        // #92: the resample size and whether the BLB cap ran.
                        st.source_rows,
                        st.resample_rows,
                        st.capped,
                        forms.join(","),
                        variables.join(",")
                    )
                }
                None => "null".to_string(),
            };
            // #86: emitted only when the flag is set, so an off run's JSON is
            // byte-identical to the historical output. `effective` is the row
            // count actually used; `widened` marks a near-full request that
            // resolved to the full sample (or an exact `n >= rows` no-op).
            let predictor_json = match cfg.predictor_samples {
                Some(requested) => {
                    let effective = if outcome.predictor_samples == 0 {
                        prepared.train.len()
                    } else {
                        outcome.predictor_samples
                    };
                    format!(
                        "\"predictor_samples\":{{\"requested\":{},\"effective\":{},\"widened\":{}}},",
                        requested,
                        effective,
                        effective == prepared.train.len()
                    )
                }
                None => String::new(),
            };
            // #85: only emitted when screening is on, so an off run's JSON is
            // byte-identical to the historical output.
            let fit_json = if outcome.fit_samples > 0 {
                format!(
                    "\"fit\":{{\"samples\":{},\"train_rows\":{},\"refit\":{}}},",
                    outcome.fit_samples,
                    prepared.train.len(),
                    outcome.archive_refit
                )
            } else {
                String::new()
            };
            // #466: omitted unless --prune is given, which keeps the historical JSON.
            let prune_json = if cfg.prune && cfg.prune_budget > 0 {
                format!(
                    "\"prune\":{{\"fits\":{},\"accepted\":{},\"seconds\":{}}},",
                    outcome.prune.fits,
                    outcome.prune.accepted,
                    jnum(outcome.prune.seconds)
                )
            } else {
                String::new()
            };
            let search_meta_json = format!("{}{}{}", predictor_json, fit_json, prune_json);
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
            let best_fit_loss = rows
                .iter()
                .map(|r| r.train_error)
                .fold(f64::INFINITY, f64::min);
            // #223: the resolution is reported when the device ran the search,
            // so a CPU-backend run keeps the historical object.
            let fit_provenance_json = match (&fit_provenance, outcome.backend.name()) {
                (Some((source, reason)), "gpu") => format!(
                    ",\"strategy_source\":{},\"strategy_reason\":{}",
                    jstr(source),
                    jstr(reason)
                ),
                _ => String::new(),
            };
            let fitting_json = if cfg.fit_strategy == fit::FitStrategy::Exact
                && outcome.fit_rows == 0
                && fit_provenance_json.is_empty()
            {
                // #116: the historical default JSON stays byte-identical; the
                // portfolio fields appear when a non-default arm or fit-row
                // screen is in use.
                format!(
                    "{{\"loss\":{},\"value\":{}}}",
                    jstr(cfg.metric.name()),
                    jnum(best_fit_loss)
                )
            } else {
                format!(
                    "{{\"loss\":{},\"value\":{},\"strategy\":{},\"fit_rows\":{},\"gpu_batches\":{}{}}}",
                    jstr(cfg.metric.name()),
                    jnum(best_fit_loss),
                    jstr(cfg.fit_strategy.name()),
                    outcome.fit_rows,
                    fit::fit_arm_counts_full().since(fit_before).gpu,
                    fit_provenance_json
                )
            };
            // #141: report the backend that actually guided the search, with
            // the device name and a reason when the CPU ran. This is derived
            // from the probe/outcome, never from the flags alone.
            let backend_reason = if gpu_explicit == Some(false) {
                Some("--cpu/--no-gpu".to_string())
            } else {
                outcome
                    .backend_reason
                    .clone()
                    .or_else(|| gpu_probe.reason.clone())
            };
            let backend_json = format!(
                "\"backend\":\"{}\",\"backend_reason\":{},\"device\":{},\"threads\":{},",
                outcome.backend.name(),
                backend_reason.as_deref().map(jstr).unwrap_or_else(|| "null".to_string()),
                outcome
                    .device
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                cfg.threads, // #463
            );
            let backend_json = format!(
                "{}{}",
                backend_json,
                match &fitness_book {
                    Some(b) => format!(
                        "\"search_fitness\":{{\"kind\":\"long_pnl_regret\",\"rows\":{},\"groups\":{},\"cost_per_side\":{},\"threshold\":{},\"bound\":{},\"min_trades\":{},\"backend\":\"cpu_f64\",\"book_hash\":\"{:016x}\",{}{}}},",
                        b.len(),
                        b.groups(),
                        jnum(b.side()),
                        jnum(b.threshold()),
                        jnum(b.bound()),
                        b.min_trades(),
                        b.hash(),
                        // #484: omitted for one block, so the default report is unchanged.
                        if b.blocks() > 1 {
                            format!(
                                "\"blocks\":{},\"lambda\":{},\"block_bounds\":[{}],",
                                b.blocks(),
                                jnum(b.lambda()),
                                b.block_bounds().iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(",")
                            )
                        } else {
                            String::new()
                        },
                        // #481: what ranked the generations; `backend` above is
                        // the reported values, always exact CPU f64.
                        match outcome.backend {
                            rcsr::backend::Kind::Gpu => "\"scorer\":\"long_pnl_regret\",\"scorer_backend\":\"cuda\",\"scorer_values\":\"screening_f32_predictions\"",
                            rcsr::backend::Kind::Cpu => "\"scorer\":\"long_pnl_regret\",\"scorer_backend\":\"cpu\",\"scorer_values\":\"exact_f64\"",
                        }
                    ),
                    None => "\"search_fitness\":{\"kind\":\"metric\"},".to_string(),
                }
            );
            let backend_json = match &objective_report {
                Some(rep) => format!(
                    "{}\"objective\":{},",
                    backend_json,
                    rep.json(outcome.backend.name(), backend_reason.as_deref())
                ),
                None => backend_json,
            };
            let backend_json = match &arm_arg {
                Some(arm) => format!(
                    "{}\"ablation\":{{\"arm\":{},\"feature_list_sha256\":{},\"generation_budget\":{},\"generations_run\":{}}},",
                    backend_json,
                    jstr(arm),
                    jstr(&rcsr::ablation::feature_list_sha256(&vars)),
                    cfg.generations,
                    outcome.generations
                ),
                None => backend_json,
            };
            let result_json = format!(
                "{{{}{}\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"fitting\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"weight_column\":{},\"weight_sum\":{},\"weight_rows\":{},\"weight_positive_rows\":{},\"train_rows\":{},\"test_rows\":{},\"split\":{},\"lockbox\":{},\"target_spec\":{},\"preparation\":{},\"generations_run\":{},\"search\":{{\"candidates_evaluated\":{}}},\"converged_early\":{},\"stop_reason\":{},\"convergence\":{},\"elapsed_seconds\":{},\"timing\":{},\"gpu\":{},\"dispatch\":{},\"checkpoint\":{},\"null_control\":{},\"baselines\":{},\"residual_tests\":{},\"cross_validation\":{},\"stability\":{},\"validation_hv\":{}{},\"qd\":{},\"seeds\":[{}],\"pareto\":[{}]}}",
                search_meta_json, // #85 #86
                backend_json,     // #141
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                fitting_json,
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
                lockbox_json,
                target_spec_json,
                prepared.prep.json(),
                outcome.generations,
                outcome.candidates_evaluated,
                outcome.converged_early,
                jstr(outcome.stop_reason),
                // #18: the four convergence numbers.
                outcome.convergence.json(),
                jnum(elapsed),
                timing_json_g(&timing, gate_seconds),
                gpu_json(
                    &outcome.gpu,
                    gpu_best_error(&outcome),
                    gpu_baseline_error(&prepared.train),
                    cfg.fit_strategy.name(),
                    cfg.metric == rcsr::metrics::Metric::Squared,
                    timing.total,
                ),
                outcome.dispatch.json(),
                checkpoint_json,
                null_json,
                baselines_json,
                resid_json,
                cv_json,
                stability_json,
                validation_hv_json(&validation_hv),
                gate_top_json,
                qd_json(outcome.qd.as_ref()),
                seeds_json.join(","),
                pareto.join(",")
            );
            // #140: a requested result file is published atomically; stdout
            // never carries a stream that a killed process can truncate.
            if let Some(path) = &json_out {
                if let Err(e) = rcsr::atomic::write_text_atomic(path, &result_json) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            } else {
                println!("{}", result_json);
            }
        }
        Format::Ndjson => {
            // #141: state the effective backend before the stream of results.
            let nd_backend_reason = if gpu_explicit == Some(false) {
                Some("--cpu/--no-gpu".to_string())
            } else {
                outcome
                    .backend_reason
                    .clone()
                    .or_else(|| gpu_probe.reason.clone())
            };
            println!(
                "{{\"event\":\"backend\",\"backend\":\"{}\",\"backend_reason\":{},\"device\":{}}}",
                outcome.backend.name(),
                nd_backend_reason
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                outcome
                    .device
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string())
            );
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
                let dimensions = format!("{}{}{}", dimensions, r.gate_json, r.fitness_json);
                let selection_json = if select_by.is_some() {
                    format!(
                        "\"archive_rank\":{},\"validation_error\":{},",
                        r.archive_rank,
                        jnum(r.validation_error)
                    )
                } else {
                    String::new()
                };
                println!(
                    "{{\"event\":\"solution\",\"rank\":{},{}\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"aicc\":{},\"bic\":{},\"durbin_watson\":{},\"overfit_index\":{},\"fold_errors\":[{}],\"bootstrap_error\":{},\"constant_errors\":{}{},\"stability\":{},\"expression\":{}}}",
                    r.rank,
                    selection_json,
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
                    constant_errors_json_g(&r.constant_errors, gates_active),
                    dimensions,
                    r.stability.map(jnum).unwrap_or_else(|| "null".to_string()),
                    jstr(&r.expression)
                );
            }
            if let Some(c) = &null_control {
                println!(
                    "{{\"event\":\"null_control\",\"runs\":{},\"method\":{},\"block_len\":{},\"blocks\":{},\"observed\":{},\"min\":{},\"median\":{},\"p_value\":{},\"verdict\":{}}}",
                    c.runs,
                    jstr(c.method),
                    c.block_len,
                    c.blocks,
                    jnum(c.observed),
                    jnum(c.null_min),
                    jnum(c.null_median),
                    jnum(c.p_value),
                    jstr(c.verdict())
                );
            }
            println!(
                "{{\"event\":\"baselines\",\"metric\":{},\"mean\":{{\"train_error\":{},\"test_error\":{}{}}},\"linear\":{{\"train_error\":{},\"test_error\":{}{}}},\"knn\":{{\"k\":{},\"train_error\":{},\"test_error\":{},{}}},\"target_structure\":{}}}",
                jstr(cfg.metric.name()),
                jnum(baselines.mean.train_error),
                jnum(baselines.mean.test_error),
                gate_mean_validation,
                jnum(baselines.linear.train_error),
                jnum(baselines.linear.test_error),
                gate_linear_validation,
                baselines.knn_k,
                jnum(baselines.knn.train_error),
                jnum(baselines.knn.test_error),
                baselines.knn_json_extra(), // #69
                gate_rung_json
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
                    "{{\"event\":\"stability\",\"runs\":{},\"found\":{},\"source_rows\":{},\"resample_rows\":{},\"capped\":{},\"forms\":[{}],\"variables\":[{}]}}",
                    st.runs,
                    st.found,
                    // #92: the resample size and whether the BLB cap ran.
                    st.source_rows,
                    st.resample_rows,
                    st.capped,
                    forms.join(","),
                    variables.join(",")
                );
            }
            // #85: the closing event carries the screening note only when the
            // flag is on, so the off path keeps its exact bytes.
            let fit_ndjson = if outcome.fit_samples > 0 {
                format!(
                    ",\"fit\":{{\"samples\":{},\"train_rows\":{},\"refit\":{}}}",
                    outcome.fit_samples,
                    prepared.train.len(),
                    outcome.archive_refit
                )
            } else {
                String::new()
            };
            // #86: same rule for the effective predictor-sample size.
            let predictor_ndjson = match cfg.predictor_samples {
                Some(requested) => {
                    let effective = if outcome.predictor_samples == 0 {
                        prepared.train.len()
                    } else {
                        outcome.predictor_samples
                    };
                    format!(
                        ",\"predictor_samples\":{{\"requested\":{},\"effective\":{},\"widened\":{}}}",
                        requested,
                        effective,
                        effective == prepared.train.len()
                    )
                }
                None => String::new(),
            };
            let search_meta_ndjson = format!("{}{}", predictor_ndjson, fit_ndjson);
            println!(
                "{{\"event\":\"done\",\"generations_run\":{},\"converged_early\":{},\"stop_reason\":{},\"convergence\":{},\"elapsed_seconds\":{},\"timing\":{},\"gpu\":{},\"dispatch\":{},\"pareto_members\":{}{}}}",
                outcome.generations,
                outcome.converged_early,
                jstr(outcome.stop_reason),
                // #18: the final reading also closes the NDJSON stream.
                outcome.convergence.json(),
                jnum(elapsed),
                timing_json_g(&timing, gate_seconds),
                gpu_json(
                    &outcome.gpu,
                    gpu_best_error(&outcome),
                    gpu_baseline_error(&prepared.train),
                    cfg.fit_strategy.name(),
                    cfg.metric == rcsr::metrics::Metric::Squared,
                    timing.total,
                ),
                outcome.dispatch.json(),
                outcome.archive.items.len(),
                search_meta_ndjson
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
        if let Err(e) = rcsr::atomic::write_text_atomic(&path, &buf) {
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

// ---------------------------------------------------------------------------
// Multi-output reporting (issue #15)
// ---------------------------------------------------------------------------

struct MultiTargetRow {
    name: String,
    expression: String,
    constants: Vec<f64>,
    train_error: f64,
    test_error: f64,
    fold_errors: Vec<f64>,
    constant_errors: Option<fit::ConstantErrors>,
}

struct MultiParetoRow {
    rank: usize,
    complexity: f64,
    k_constants: usize,
    train_error: f64,
    test_error: f64,
    targets: Vec<MultiTargetRow>,
}

/// Run the shared-shape search and report one expression per target.
///
/// `k_constants` is the **total** across outputs (`shape constants x targets`).
/// `constant_errors`, the printed formula and each target's error are per
/// output; `train_error`/`test_error` on the row are the combined (mean) error
/// the archive was ranked on. Shape complexity is counted once.
#[allow(clippy::too_many_arguments)]
#[allow(clippy::too_many_arguments)]
fn run_multi_output(
    cfg: &search::Config,
    prepared: &data::Prepared,
    target_names: &[String],
    vars: &[String],
    format: Format,
    quiet: bool,
    top: usize,
    out_path: Option<String>,
    json_out: Option<String>,
    run_start: std::time::Instant,
    start: std::time::Instant,
    // #211: the development log, if one is open.
    dev: Option<&DevSink>,
) {
    let outcome = search::run(cfg, &prepared.train, &prepared.test);
    let search_seconds = start.elapsed().as_secs_f64();
    let elapsed = search_seconds;
    let report_start = std::time::Instant::now();
    let sorted = outcome.archive.sorted();
    let fold_samples: Vec<data::Sample> = if prepared.fold_samples.is_empty() {
        vec![prepared.test.clone()]
    } else {
        prepared.fold_samples.clone()
    };
    let n_targets = target_names.len();

    let mut rows: Vec<MultiParetoRow> = sorted
        .iter()
        .enumerate()
        .map(|(k, s)| {
            // Fit one constant vector per output on the training rows, then
            // score every output on the same fitted trees (no test leakage).
            let trees = multi::fitted_outputs_metric(
                &s.tree,
                &prepared.train,
                cfg.metric,
                cfg.fit_iters,
            );
            let train_errors = multi::output_errors(&trees, &prepared.train, cfg.metric);
            let test_errors = multi::output_errors(&trees, &prepared.test, cfg.metric);
            let targets = target_names
                .iter()
                .enumerate()
                .map(|(j, name)| {
                    let den = expr::simplify_display(&expr::denormalize(
                        &expr::simplify_display(&trees[j]),
                        &prepared.means,
                        &prepared.stds,
                    ));
                    let ce = if prepared.train.w.is_empty() {
                        fit::constant_errors(&trees[j], &prepared.train.x, prepared.train.target(j))
                    } else {
                        fit::constant_errors_weighted(
                            &trees[j],
                            &prepared.train.x,
                            prepared.train.target(j),
                            &prepared.train.w,
                        )
                    };
                    MultiTargetRow {
                        name: name.clone(),
                        expression: expr::to_string(&den, vars),
                        constants: multi::constants(&trees[j]),
                        train_error: train_errors[j],
                        test_error: test_errors[j],
                        fold_errors: fold_samples
                            .iter()
                            .map(|f| multi::eval_target(&trees[j], f, cfg.metric, j))
                            .collect(),
                        constant_errors: ce,
                    }
                })
                .collect::<Vec<_>>();
            MultiParetoRow {
                rank: k + 1,
                complexity: s.cost,
                // Every output carries its own constant vector.
                k_constants: expr::count_constants(&s.tree) * n_targets,
                train_error: multi::mean_error(&train_errors),
                test_error: multi::mean_error(&test_errors),
                targets,
            }
        })
        .collect();
    let finite = rows.iter().any(|r| r.train_error.is_finite());
    if top > 0 && rows.len() > top {
        rows.truncate(top);
    }

    // #134: the multi-output path gets the same non-overlapping ledger as a
    // single-target run. It does not run the single-target baseline or
    // diagnostics phases, so those two windows are explicitly zero rather than
    // silently excluded. `total` is measured independently at the end of the
    // report phase, exactly like the single-output path.
    let timing = PhaseTimings {
        prep: (start - run_start).as_secs_f64(),
        search: search_seconds,
        baselines: 0.0,
        diagnostics: 0.0,
        report: report_start.elapsed().as_secs_f64(),
        total: run_start.elapsed().as_secs_f64(),
        gpu: 0.0,
    };

    match format {
        Format::Text => {
            if !quiet {
                eprintln!(
                    "# elapsed: {:.2}s | generations: {} | pareto members: {}{} | targets: {}",
                    elapsed,
                    outcome.generations,
                    outcome.archive.items.len(),
                    if outcome.stopped_early() {
                        " (stopped early)"
                    } else {
                        ""
                    },
                    target_names.join(",")
                );
                qd_text(outcome.qd.as_ref());
                eprintln!("# time: {}", timing.text());
                eprintln!(
                    "# multi-output: one shared shape, one fitted constant vector per output; \
                     train_err/test_err are the mean over {} target(s)",
                    n_targets
                );
            }
            for r in &rows {
                println!(
                    "{:>3}  complexity={:<3} combined_train_err={:<13} combined_test_err={:<13}",
                    r.rank,
                    r.complexity,
                    fmt_err(r.train_error),
                    fmt_err(r.test_error)
                );
                for t in &r.targets {
                    println!(
                        "     {} = {}  (train_err={} test_err={})",
                        t.name,
                        t.expression,
                        fmt_err(t.train_error),
                        fmt_err(t.test_error)
                    );
                }
                // Constant uncertainties are per output, like the expressions.
                for t in &r.targets {
                    if let Some(ce) = &t.constant_errors {
                        let parts: Vec<String> = ce
                            .values
                            .iter()
                            .zip(&ce.std_errors)
                            .zip(&ce.relative)
                            .map(|((v, se), rel)| {
                                format!("{:.6}±{:.2e} (rel {:.2e})", v, se, rel)
                            })
                            .collect();
                        eprintln!(
                            "# constant errors (rank {}, {}): {} | dof={}{}",
                            r.rank,
                            t.name,
                            parts.join(", "),
                            ce.dof,
                            if ce.ill_conditioned {
                                " | ill-conditioned"
                            } else {
                                ""
                            }
                        );
                    }
                }
            }
            if let Some(r) = rows.iter().min_by(|a, b| {
                a.train_error
                    .partial_cmp(&b.train_error)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }) {
                eprintln!("# diagnostics: n={} k={} constants (total across outputs)", prepared.train.len(), r.k_constants);
            }
        }
        Format::Json => {
            let pareto: Vec<String> = rows
                .iter()
                .map(|r| {
                    let targets: Vec<String> = r
                        .targets
                        .iter()
                        .map(|t| {
                            format!(
                                "{{\"name\":{},\"expression\":{},\"constants\":[{}],\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}],\"constant_errors\":{}}}",
                                jstr(&t.name),
                                jstr(&t.expression),
                                t.constants.iter().map(|v| jnum(*v)).collect::<Vec<_>>().join(","),
                                jnum(t.train_error),
                                jnum(t.test_error),
                                t.fold_errors.iter().map(|e| jnum(*e)).collect::<Vec<_>>().join(","),
                                constant_errors_json(&t.constant_errors)
                            )
                        })
                        .collect();
                    format!(
                        "{{\"rank\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"targets\":[{}]}}",
                        r.rank,
                        jnum(r.complexity),
                        r.k_constants,
                        jnum(r.train_error),
                        jnum(r.test_error),
                        targets.join(",")
                    )
                })
                .collect();
            let best_fit_loss = rows
                .iter()
                .map(|r| r.train_error)
                .fold(f64::INFINITY, f64::min);
            let fitting_json = if cfg.fit_strategy == fit::FitStrategy::Exact && outcome.fit_rows == 0 {
                format!(
                    "{{\"loss\":{},\"value\":{}}}",
                    jstr(cfg.metric.name()),
                    jnum(best_fit_loss)
                )
            } else {
                format!(
                    "{{\"loss\":{},\"value\":{},\"strategy\":{},\"fit_rows\":{},\"gpu_batches\":{}}}",
                    jstr(cfg.metric.name()),
                    jnum(best_fit_loss),
                    jstr(cfg.fit_strategy.name()),
                    outcome.fit_rows,
                    fit::fit_arm_counts_full().gpu
                )
            };
            let result_json = format!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"seed\":{},\"metric\":{},\"fitting\":{},\"multi_output\":true,\"target\":{},\"targets\":[{}],\"variables\":[{}],\"train_rows\":{},\"test_rows\":{},\"generations_run\":{},\"converged_early\":{},\"stop_reason\":{},\"elapsed_seconds\":{},\"timing\":{},\"backend\":\"{}\",\"backend_reason\":{},\"device\":{},\"qd\":{},\"pareto\":[{}]}}",
                jstr(VERSION),
                cfg.seed,
                jstr(cfg.metric.name()),
                fitting_json,
                jstr(&target_names.join(",")),
                target_names.iter().map(|t| jstr(t)).collect::<Vec<_>>().join(","),
                vars.iter().map(|v| jstr(v)).collect::<Vec<_>>().join(","),
                prepared.train.len(),
                prepared.test.len(),
                outcome.generations,
                outcome.converged_early,
                jstr(outcome.stop_reason),
                jnum(elapsed),
                timing.json(),
                // #141: the effective backend, never inferred from flags.
                outcome.backend.name(),
                outcome
                    .backend_reason
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                outcome
                    .device
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                qd_json(outcome.qd.as_ref()),
                pareto.join(",")
            );
            if let Some(path) = &json_out {
                if let Err(e) = rcsr::atomic::write_text_atomic(path, &result_json) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            } else {
                println!("{}", result_json);
            }
        }
        Format::Ndjson => {
            // #134: the closing event carries the same timing ledger as the
            // single-output stream and the multi-output JSON object.
            println!(
                "{{\"event\":\"done\",\"multi_output\":true,\"generations_run\":{},\"converged_early\":{},\"stop_reason\":{},\"elapsed_seconds\":{},\"timing\":{},\"pareto_members\":{},\"targets\":[{}],\"backend\":\"{}\",\"backend_reason\":{},\"device\":{}}}",
                outcome.generations,
                outcome.converged_early,
                jstr(outcome.stop_reason),
                jnum(elapsed),
                timing.json(),
                outcome.archive.items.len(),
                target_names.iter().map(|t| jstr(t)).collect::<Vec<_>>().join(","),
                outcome.backend.name(),
                outcome
                    .backend_reason
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string()),
                outcome
                    .device
                    .as_deref()
                    .map(jstr)
                    .unwrap_or_else(|| "null".to_string())
            );
        }
    }

    if let Some(path) = out_path {
        let mut buf = String::from(
            "rank\tcomplexity\tcombined_train_error\tcombined_test_error\ttarget\ttrain_error\ttest_error\texpression\n",
        );
        for r in &rows {
            for t in &r.targets {
                buf.push_str(&format!(
                    "{}\t{}\t{:.8}\t{:.8}\t{}\t{:.8}\t{:.8}\t{}\n",
                    r.rank,
                    r.complexity,
                    r.train_error,
                    r.test_error,
                    t.name,
                    t.train_error,
                    t.test_error,
                    t.expression
                ));
            }
        }
        if let Err(e) = rcsr::atomic::write_text_atomic(&path, &buf) {
            eprintln!("error writing {}: {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    }

    // #211: close the development log for a multi-output run too.
    if let Some(sink) = dev {
        sink.end(
            &outcome,
            run_start.elapsed().as_secs_f64(),
            search_seconds,
            json_out.clone(),
        );
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
    // A meta operator describes the searched tree, and an aggregate names a
    // dataset column: rewriting an input inside either would change its
    // meaning. They are carried over verbatim (the aggregate is bound to the
    // training sample later, in `search`).
    if n.op.is_meta() || n.op.is_agg() {
        return n.clone();
    }
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
            // A block body is in parameter space and must not be rewritten.
            block: n.block.clone(),
        },
    }
}

/// #254: run the D13 detectors on a declared target; exit 5 on a finding that
/// is not overridden.
fn enforce_target_check(
    spec: &target::TargetSpec,
    vars: &[String],
    allow_structure: bool,
    require_clean: bool,
) -> target::Check {
    match target::check(spec, vars, allow_structure) {
        Ok(check) if require_clean && !check.construction_in_vars.is_empty() => {
            eprintln!(
                "error: --require-clean-inputs: construction column(s) of target '{}' are inputs: {}",
                spec.column,
                check.construction_in_vars.join(", ")
            );
            std::process::exit(EXIT_LEAKAGE);
        }
        Ok(check) => check,
        Err(report) => {
            if let Err(failure) = report.result() {
                eprintln!("error: {}", failure);
            }
            std::process::exit(EXIT_LEAKAGE);
        }
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
    implicit: bool,
    format: Format,
    quiet: bool,
    started: std::time::Instant,
    json_out: Option<&str>,
    target_active: Option<&target::Active>,
) {
    let mut tree = match expr::parse(src, vars) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: cannot parse --eval-expr: {}", e);
            eprintln!("  expression: {}", src);
            std::process::exit(EXIT_USAGE);
        }
    };
    // `--eval-expr` evaluates in original units, so an aggregate is the
    // statistic of the raw training column, not the normalised one.
    let raw_row = |row: &[f64]| -> Vec<f64> {
        row.iter()
            .enumerate()
            .map(|(j, v)| {
                v * prepared.stds.get(j).copied().unwrap_or(1.0)
                    + prepared.means.get(j).copied().unwrap_or(0.0)
            })
            .collect()
    };
    let raw_train: Vec<Vec<f64>> = prepared.train.x.iter().map(|r| raw_row(r)).collect();
    let aggs = expr::AggTable::from_rows(&raw_train);
    if let Err(e) = expr::bind_aggregates(&mut tree, &aggs) {
        eprintln!("error: cannot evaluate --eval-expr: {}", e);
        eprintln!("  expression: {}", src);
        std::process::exit(EXIT_USAGE);
    }
    let k = expr::count_constants(&tree);
    let complexity = model.complexity(&tree);
    let shown = expr::to_string(&tree, vars);

    let score = |s: &data::Sample| -> f64 {
        let pred: Vec<f64> = s.x.iter().map(|r| expr::eval(&tree, &raw_row(r))).collect();
        metrics::score_sample(&pred, s, metric, k)
    };
    let train_error = score(&prepared.train);
    // One test block leaves `fold_samples` empty (#133): score the test block.
    let fold_errors: Vec<f64> = if prepared.fold_samples.is_empty() {
        vec![score(&prepared.test)]
    } else {
        prepared.fold_samples.iter().map(score).collect()
    };
    let test_error = fold_errors.iter().sum::<f64>() / fold_errors.len() as f64;

    match format {
        Format::Json => {
            let folds: Vec<String> = fold_errors.iter().map(|e| jnum(*e)).collect();
            let result_json = format!(
                "{{\"tool\":\"rcsr\",\"version\":{},\"mode\":\"eval-expr\",\"metric\":{},\"target\":{},\"variables\":[{}],\"variables_detail\":[{}],\"train_rows\":{},\"test_rows\":{},\"preparation\":{},\"eval\":{{\"expression\":{},\"complexity\":{},\"k_constants\":{},\"train_error\":{},\"test_error\":{},\"fold_errors\":[{}]}},\"target_spec\":{},\"target_structure\":{}}}",
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
                folds.join(","),
                target_active
                    .map(|a| a.json())
                    .unwrap_or_else(|| "null".to_string()),
                target_active
                    .map(|a| a.rung_json())
                    .unwrap_or_else(|| "null".to_string())
            );
            if let Some(path) = json_out {
                if let Err(e) = rcsr::atomic::write_text_atomic(path, &result_json) {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            } else {
                println!("{}", result_json);
            }
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
                "  1  complexity={}   k_constants={}   train_err={}      test_err={}      {} = {}",
                complexity,
                k,
                fmt_err(train_error),
                fmt_err(test_error),
                if implicit { "0" } else { target },
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
/// #134: the phase breakdown in text reports.
fn time_markdown(t: &PhaseTimings) -> String {
    format!(
        "\n## Time\n\n\
         | phase | seconds |\n|---|---:|\n\
         | preparation | {:.3} |\n\
         | search | {:.3} |\n\
         | baselines | {:.3} |\n\
         | diagnostics | {:.3} |\n\
         | report | {:.3} |\n\
         | **total wall** | **{:.3}** |\n\n\
         `elapsed_seconds` in JSON is the search phase, kept for compatibility;\n\
         `total_seconds` is the one runtime of the process. The phases and\n\
         `unaccounted_seconds` add up to it.\n",
        t.prep, t.search, t.baselines, t.diagnostics, t.report, t.total
    )
}

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
    select_by: Option<rcsr::selection::SelectBy>,
) -> String {
    let mut s = String::new();
    let chosen = if select_by.is_some() {
        rows.first()
    } else {
        rows.iter().min_by(|a, b| {
            a.train_error
                .partial_cmp(&b.train_error)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    };

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
        if outcome.stopped_early() {
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
    let _ = writeln!(s, "| selector | {} |", prepared.report.selector);
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
                let _ = writeln!(
                    s,
                    "_Null: {}; observed error {}._\n",
                    if c.method == "block" {
                        format!("block permutation, {} blocks of up to {} rows", c.blocks, c.block_len)
                    } else {
                        "i.i.d. target shuffle".to_string()
                    },
                    md_num(c.observed)
                );
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
                "Across {} {} the search reached a best error of {} \
                 (median {}); the permutation p-value is {}, which means ",
                c.runs,
                if c.method == "block" {
                    format!("block-permutation runs ({} blocks of up to {} rows)", c.blocks, c.block_len)
                } else {
                    "shuffled-target runs".to_string()
                },
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
            let den = expr::simplify_display(&expr::denormalize(
                &expr::simplify_display(&sol.tree),
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
    extra.dev_log = None; // #211: a repeat is not the logged run
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
            "{{\"resamples\":{},\"used\":{},\"source_rows\":{},\"resample_rows\":{},\"capped\":{},\"apparent\":{},\"oob\":{},\"no_information\":{},\"relative_overfitting\":{},\"weight\":{},\"estimate\":{}}}",
            b.resamples,
            b.used,
            // #92: the resample size and whether the BLB cap ran.
            b.source_rows,
            b.resample_rows,
            b.capped,
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

/// As [`constant_errors_json`], plus the #257 identification fields when the
/// gates ran.
fn constant_errors_json_g(ce: &Option<fit::ConstantErrors>, gates: bool) -> String {
    let base = constant_errors_json(ce);
    match (gates, ce) {
        (true, Some(c)) => {
            let list = |v: &[f64]| v.iter().map(|x| jnum(*x)).collect::<Vec<_>>().join(",");
            let robust = |v: &Option<Vec<f64>>| match v {
                Some(v) => format!("[{}]", list(v)),
                None => "null".to_string(),
            };
            let mut out = base;
            out.pop();
            out.push_str(&format!(
                ",\"rank\":{},\"condition_index\":{},\"involved\":[{}],\"std_errors_robust\":{},\"relative_robust\":{}}}",
                c.rank,
                jnum(c.condition_index),
                c.involved.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","),
                robust(&c.std_errors_robust),
                robust(&c.relative_robust)
            ));
            out
        }
        _ => base,
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

// ---------------------------------------------------------------------------
// #71: memory guard and `import`
// ---------------------------------------------------------------------------

/// Parse `bytes`, `KB`/`KiB`, `MB`/`MiB`, `GB`/`GiB` (case-insensitive).
/// Bare numbers are bytes. Decimal suffixes use 1000, binary suffixes 1024.
fn parse_memory_bytes(src: &str) -> Result<u64, String> {
    let s = src.trim();
    if s.is_empty() {
        return Err("empty size".to_string());
    }
    let split = s
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(s.len());
    let (num, unit) = s.split_at(split);
    let value: f64 = num.parse().map_err(|_| format!("'{}' is not a number", num))?;
    if !value.is_finite() || value <= 0.0 {
        return Err("size must be positive".to_string());
    }
    let unit = unit.trim().to_ascii_lowercase();
    let factor: f64 = match unit.as_str() {
        "" | "b" | "byte" | "bytes" => 1.0,
        "k" | "kb" => 1_000.0,
        "m" | "mb" => 1_000_000.0,
        "g" | "gb" => 1_000_000_000.0,
        "t" | "tb" => 1_000_000_000_000.0,
        "ki" | "kib" => 1_024.0,
        "mi" | "mib" => 1_048_576.0,
        "gi" | "gib" => 1_073_741_824.0,
        "ti" | "tib" => 1_099_511_627_776.0,
        other => return Err(format!("unknown size unit '{}'", other)),
    };
    Ok((value * factor) as u64)
}

fn human_bytes(bytes: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = bytes as f64;
    let mut i = 0;
    while v >= 1024.0 && i + 1 < UNITS.len() {
        v /= 1024.0;
        i += 1;
    }
    if i == 0 {
        format!("{} {}", bytes, UNITS[0])
    } else {
        format!("{:.2} {}", v, UNITS[i])
    }
}

/// A temporary file that removes itself when the CSV load finishes (issue
/// #126: stdin is spooled so the streaming reader can reopen it).
struct TempCsv {
    path: std::path::PathBuf,
}

impl TempCsv {
    fn path_str(&self) -> &str {
        self.path.to_str().unwrap_or("")
    }
}

impl Drop for TempCsv {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

/// Copy standard input into a temporary CSV file. This keeps `--data -`
/// streaming: the two parser passes reopen the file instead of a whole-file
/// `String` staying resident.
fn spool_stdin() -> Result<TempCsv, String> {
    let dir = std::env::temp_dir();
    let pid = std::process::id();
    for attempt in 0..64u32 {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let path = dir.join(format!("rcsr-stdin-{}-{}-{}.csv", pid, nanos, attempt));
        let mut file = match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => continue,
            Err(e) => {
                return Err(format!(
                    "cannot spool stdin to {}: {}",
                    path.display(),
                    e
                ))
            }
        };
        let mut input = std::io::stdin().lock();
        if let Err(e) = std::io::copy(&mut input, &mut file) {
            let _ = std::fs::remove_file(&path);
            return Err(format!("error reading stdin: {}", e));
        }
        return Ok(TempCsv { path });
    }
    Err("error reading stdin: could not create a temporary file".to_string())
}

/// #112: a store that does not fit is refused with its exact header-derived
/// terms. Sampling is never automatic; the user chooses `--force` (resident) or
/// `--sample-rows N` (explicit, recorded as `full_rows=false`).
fn refuse_store_for_memory(
    path: &str,
    fs: &store::FileStore,
    estimate: &memory::MemoryEstimate,
    limit: u64,
) -> ! {
    eprintln!("error: input would exceed the memory budget");
    let disk = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let dictionary_cols = fs
        .header()
        .columns()
        .iter()
        .filter(|c| c.has_codes)
        .count();
    eprintln!(
        "  {}: {} on disk, {} rows x {} columns ({}, {} dictionary column(s))",
        path,
        human_bytes(disk),
        estimate.rows,
        estimate.cols,
        match fs.header().version {
            1 => "store v1",
            _ => "store v2",
        },
        dictionary_cols
    );
    eprintln!("  estimate basis: store header dtypes + row groups + used columns");
    eprintln!(
        "    storage {} + directory/group {} + search {} + fit {} + ring {} + fixed {} + slack {} \
         = {}{}",
        human_bytes(estimate.storage_bytes),
        human_bytes(estimate.directory_bytes.saturating_add(estimate.group_buffer_bytes)),
        human_bytes(estimate.search_bytes),
        human_bytes(estimate.fit_copy_bytes),
        human_bytes(estimate.ring_bytes),
        human_bytes(estimate.fixed_bytes),
        human_bytes(estimate.slack_bytes),
        human_bytes(estimate.total_bytes),
        if estimate.saturated { " (at least)" } else { "" }
    );
    eprintln!(
        "    projection: {} search + {} scan used column(s) of {} ({} projected read)",
        estimate.used_cols, estimate.scan_cols, estimate.cols, estimate.projected_cols
    );
    eprintln!(
        "  memory budget is {} ({} bytes; set it with --max-memory)",
        human_bytes(limit),
        limit
    );
    eprintln!("  pass --force to load the full store resident anyway, or");
    eprintln!(
        "  pass --sample-rows <n> to hash-sample explicitly (the manifest/dev log then \
         records full_rows=false)"
    );
    std::process::exit(EXIT_INPUT);
}

/// #112: an explicit `--sample-rows` request that is still over budget is
/// refused instead of becoming a quieter out-of-budget run.
fn refuse_store_sample_for_memory(
    path: &str,
    fs: &store::FileStore,
    estimate: &memory::MemoryEstimate,
    requested: usize,
    limit: u64,
) -> ! {
    eprintln!("error: --sample-rows {} would still exceed the memory budget", requested);
    eprintln!(
        "  {}: {} requested rows of {} (full store is {} rows x {} columns)",
        path,
        requested,
        fs.n_rows(),
        fs.n_rows(),
        fs.n_cols()
    );
    eprintln!(
        "  sampled estimate: storage {} + search {} + fit {} + fixed {} + slack {} = {}",
        human_bytes(estimate.storage_bytes),
        human_bytes(estimate.search_bytes),
        human_bytes(estimate.fit_copy_bytes),
        human_bytes(estimate.fixed_bytes),
        human_bytes(estimate.slack_bytes),
        human_bytes(estimate.total_bytes)
    );
    eprintln!(
        "  memory budget is {} ({} bytes); pass --force to attempt it anyway",
        human_bytes(limit),
        limit
    );
    std::process::exit(EXIT_INPUT);
}

/// #112: `--sample-rows N` with `N >= rows` asks for every row, i.e. the full
/// load. It must not bypass the budget the default path enforces.
fn refuse_store_full_with_sample_request(
    path: &str,
    fs: &store::FileStore,
    estimate: &memory::MemoryEstimate,
    requested: usize,
    limit: u64,
) -> ! {
    eprintln!(
        "error: --sample-rows {} covers every row of '{}', so it is a full load, \
         and the full store does not fit the memory budget",
        requested, path
    );
    eprintln!(
        "  full store: {} rows x {} columns; estimate {} = storage {} + directory/group {} + \
         search {} + fit {} + fixed {} + slack {}{}",
        estimate.rows,
        estimate.cols,
        human_bytes(estimate.total_bytes),
        human_bytes(estimate.storage_bytes),
        human_bytes(estimate.directory_bytes.saturating_add(estimate.group_buffer_bytes)),
        human_bytes(estimate.search_bytes),
        human_bytes(estimate.fit_copy_bytes),
        human_bytes(estimate.fixed_bytes),
        human_bytes(estimate.slack_bytes),
        if estimate.saturated { " (at least)" } else { "" }
    );
    eprintln!(
        "  memory budget is {} ({} bytes; set it with --max-memory)",
        human_bytes(limit),
        limit
    );
    eprintln!(
        "  pass --sample-rows with fewer than {} rows to sample, or --force to load the full \
         store anyway",
        fs.n_rows()
    );
    std::process::exit(EXIT_INPUT);
}

fn refuse_for_memory(
    path: &str,
    shape: data::CsvShape,
    estimate: u64,
    limit: u64,
    prep_cols: Option<usize>,
) -> ! {
    eprintln!("error: input would exceed the memory budget");
    eprintln!(
        "  {}: {} on disk, {} rows x {} columns; estimated resident footprint {}",
        path,
        human_bytes(shape.bytes),
        shape.rows,
        shape.cols,
        human_bytes(estimate)
    );
    let breakdown = match prep_cols {
        Some(projected) => data::csv_estimate(shape, projected),
        None => data::csv_load_estimate(shape),
    };
    eprintln!(
        "  estimate basis: {} rows x {} columns ({} used); text {} + table {} \
         ({} B/cell) + preparation {} ({} B/used cell, {} B/row) + slack {}",
        breakdown.rows,
        breakdown.cols,
        breakdown.used_cols,
        human_bytes(breakdown.text_bytes),
        human_bytes(breakdown.table_bytes),
        data::CSV_TABLE_BYTES_PER_CELL,
        human_bytes(breakdown.prep_bytes),
        data::CSV_PREP_BYTES_PER_USED_CELL,
        data::CSV_PREP_FIXED_BYTES_PER_ROW,
        human_bytes(breakdown.slack_bytes)
    );
    eprintln!(
        "  memory budget is {} ({} bytes; set it with --max-memory)",
        human_bytes(limit),
        limit
    );
    eprintln!("  re-run with a larger budget or pass --force to attempt it anyway;");
    eprintln!("  a `.rcsb` store can also be sampled explicitly:");
    eprintln!("    rcsr import <input.csv> <store.rcsb>");
    eprintln!(
        "    rcsr --data <store.rcsb> --target <col> --max-memory {} [--sample-rows N]",
        human_bytes(limit)
    );
    std::process::exit(EXIT_INPUT);
}

fn walkforward_usage() -> String {
    format!(
        "rcsr {VERSION} — walk-forward evaluation over a .rcsb column store

USAGE:
  rcsr walkforward --data <store.rcsb|csv> --target <col> --expr <formula> \\
                   --time <col> --window <label,train_start,train_end,test_start,test_end> ...

  Evaluates one expression with the engine's own f64 evaluator (no Python, no
  code strings), applies a long/flat rule and charges costs on position changes.
  Windows are half-open by time; the purge gap test_start-train_end must be at
  least --horizon and --embargo. The threshold comes from training/validation
  rows only. See docs/walkforward.md.

OPTIONS:
  --data <path>              .rcsb column store, or a CSV table as a fallback
  --target <col>             forward-return column used for P&L (optional with
                             --target-spec or --from-run: the spec's return column)
  --expr <formula>           score expression, variables resolved by column name;
                             repeat to select among candidates on the validation slice
  --front <run.json>         candidates from the Pareto front of a run (one per
                             member, labelled front_rank_<k>), instead of --expr.
                             --vars, --target (the run's return column) and
                             --trials (search.candidates_evaluated) default to
                             the run. The run's position rule is carried: the
                             objective threshold as a fixed threshold and its
                             per-side cost; --threshold-fixed,
                             --threshold-quantile and --cost-bps override.
                             A window whose test_start is below the last train
                             or validation time of the run plus its required
                             purge exits 2
  --allow-in-sample-windows  with --front or --from-run: report such a window as
                             in_sample:true instead of refusing it
                             (diagnostics only)
  --select-band              #308: select among repeated --expr candidates with
                             the P&L band instead of validation MSE. The screen is
                             the fused long-only reduction on validation rows; the
                             admitted set is rescored on the full validation rows in
                             exact CPU f64 and ranked by
                             (net, turnover, complexity, index). Requires
                             --validation-frac > 0 or a --windows table with validation_start, and more than one --expr; the
                             JSON carries surrogate_rank, exact_rank, band,
                             epsilon, finalists_considered, recall_band and
                             scored_by=cpu_f64
  --vars <a,b,...>           feature columns (default: all numeric except target,
                             time and group)
  --time <col>               time/ordering column (numbers or YYYY-MM-DD)
  --group-by <col>           panel group column; positions and costs are per
                             group and rows sharing a timestamp are equal-weighted
  --window <spec>            LABEL,TRAIN_START,TRAIN_END,TEST_START,TEST_END
                             (repeatable; the four-bound form is also accepted)
  --windows <file>           CSV file of window rows with the same columns
  --horizon <seconds>        forward-label horizon; minimum purge/embargo. Test rows of one
                             group closer than the horizon overlap and are refused (exit 2)
                             (default 0)
  --embargo <seconds>        extra gap between train_end and test_start
                             (default: the horizon)
  --cost-bps <f>             cost per side in basis points (default {DEFAULT_COST_BPS}); a
                             --target-spec or --from-run cost wins and a different
                             value is refused (exit 2)
  --cost-override <f>        price a different per-side cost on purpose; the report
                             says cost.source = flag and keeps the spec cost
  --target-spec <file>       target spec (rcsr-target-spec/1): --horizon, --entry-offset,
                             --delay, the return column, return units (log is
                             converted with exp_m1) and the cost come from the entry
                             matching --target; a flag that disagrees is refused
  --from-run <run.json>      the book a search with --objective selected: expression,
                             variables, fixed threshold, per-side cost and target spec
                             of the run. Refuses --expr, --threshold-quantile,
                             --threshold-fixed and --target-spec, and a run without an
                             objective block
  --threshold-quantile <q>   exact type-7 training/validation score quantile
                             (default {DEFAULT_QUANTILE}); selected by order-preserving
                             radix count, so theta is bit-identical to numpy
  --threshold-fixed <theta>  pre-registered fixed threshold instead of a quantile;
                             reported with threshold_source = fixed
  --policy <p>               threshold | band | dp | top-tercile (default threshold). top-tercile
                             holds the top floor(n/3) finite scores of each timestamp long and
                             the rest flat (no threshold; --threshold-quantile, --threshold-fixed,
                             --band, --min-hold, --select-band, --device-pnl and --batch-pnl are
                             refused). band is a
                             long-only no-trade band; dp keeps the threshold book
                             and reports the exact hindsight DP as a bound, never
                             as the strategy's P&L
  --band <f>                 half-width beta of the no-trade band in score units
                             (default 0; with --policy band, beta=0 reproduces
                             the threshold rule bit for bit)
  --min-hold <n>             minimum consecutive long rows before a band exit
                             (default 1; requires --policy band)
  --gpu                      run the --policy band/cooldown scan and the --policy
                             dp hindsight scan on the CUDA device. The result is
                             bit-identical to the CPU scan; when the device
                             refuses (no device, non-CUDA build, allocation),
                             the CPU scan runs and band/hindsight_bound carry
                             backend=cpu and fallback_reason
  --validation-frac <f>      trailing fraction of the training time range used to
                             set the threshold instead of all training rows
                             (default 0: use all training rows)
  --pool-reset-flat          reset every group to flat at a window seam (the
                             historical pooled P&L); default carries the boundary
                             position, so total variation includes |p_end-q_start|
  --periods-per-year <f>     annualisation override (default: infer from the
                             median test time step; 365 for daily crypto bars)
  --device-pnl               recompute each test block through the fused
                             long-only reduction (f64 canonical segments; the
                             CUDA kernel when built with --features cuda) and
                             report it with cash and buy-and-hold benchmarks.
                             The scalar evaluator remains the reported authority
  --control-seed <n>         seed of the matched-exposure random-long control
                             (default 0). Every report lists the long-only
                             controls per window: cash, buy-and-hold, constant
                             long/flat (side fixed from the training mean) and
                             matched-exposure random long, each with exposure,
                             turnover, beta and excess over cash. With
                             --device-pnl they are cross-checked on the device
  --batch-pnl                score every candidate on every window in one batch
                             (CUDA kernel when built with --features cuda, the
                             identical host path otherwise), per window and
                             pooled, with the baseline ladder. Reported
                             finalist values stay exact CPU f64; batch values
                             are labelled gpu_proxy and checked against them
  --entry-offset <seconds>   seconds from the row stamp to the label entry
                             (default 0). Required purge = entry-offset +
                             horizon + delay; a window with a shorter gap is
                             refused
  --delay <seconds>          execution delay, part of the required purge
                             (default 0)
  --windows-scheme <name>    with a protocol/windows.csv table: keep the rows of
                             one scheme (expanding | rolling | locked). The
                             table's validation_start column replaces
                             --validation-frac
  --bootstrap <B>            circular block bootstrap with B draws (default 0 =
                             off; 10000 recommended). Reports CIs for the mean
                             and Sharpe, the design effect against an iid row
                             bootstrap, effective n, DSR at N_eff and 10 N_eff,
                             PBO (CSCV), SPA against the best long/flat baseline
                             and BH-FDR over the candidates
  --block-len <L>            block length in periods (default
                             max(h, hold, n^(1/3))); must be >= max(h, hold)
  --null-draws <B>           draws of the hour-block permutation null and the
                             matched-exposure random long/flat null (default 0
                             = off); p = (1+count)/(B+1)
  --trials <N>               total trials behind the candidates, the N of the
                             DSR (default: the candidate count)
  --protocol-seed <n>        seed of every bootstrap, null and SPA draw
  --pbo-blocks <S>           even CSCV block count (default 16)
  --fdr-q <q>                Benjamini-Hochberg rate (default 0.10)
  --ledger <file>            append-only JSONL candidate ledger. A development
                             run appends a candidate (test_read:false) and
                             refuses windows that reach the declared lockbox
  --lockbox-embargo <SECONDS>
                             widen the lookback embargo after the end of every locked range
                             (default: lockbox_embargo declared with the data, else 10972800);
                             a smaller value is refused (exit 2). Rows in
                             [START - purge, END + embargo) are refused (exit 3)
  --lockbox <START,END|none>
                             locked test, UTC half-open. Order: this flag, the range
                             declared with --data (store manifest, schema or sidecar),
                             else the protocol default 2025-07-01,2026-07-01. On undeclared
                             data a flag adds to the default. A flag
                             that does not cover the declared range is refused (exit
                             2), as are conflicting declarations (exit 3). `none` opts
                             out; it is refused when the data declares a range and
                             with --ledger or --locked-test. A window ending after START minus
                             the label purge (entry offset + horizon + delay) is refused
                             (exit 3) unless --locked-test reads it
  --locked-test              read the locked test once for a sealed candidate.
                             Needs --ledger and --sealed. A second read, an
                             unsealed candidate or a code/data hash that differs
                             from the sealed entry is refused. After `protocol seal
                             --final`: at most the sealed number of reads, a read
                             through another ledger of the same store is refused
                             (<data>.locked_reads.jsonl), --batch-pnl is refused
                             (exit 2) and the report holds the selected candidate
                             only (no per-candidate series, PBO or FDR)
  --sealed <file>            locked_candidates.json from `rcsr protocol seal`
  --verify-read              with --locked-test: recompute a recorded read and
                             compare its report hash; records nothing
  --allow-unbound-ledger     with --locked-test: read a candidate whose ledger
                             recorded no locked test range (old ledger); the read
                             is reported as unbound. Without it that read is
                             refused (exit 3), as is a range that differs from
                             the one the candidate was developed against
  --expect-report-sha256 <h> exit 1 unless the report JSON hashes to h
  --json                     print the full JSON report to stdout
                             (it ends with a ledger object: the phase ledger of the
                             command, io and pnl seconds on the run wall, outside the
                             report hash)
  --out <file>               write the JSON report to a file
  --markdown <file|->        write the Markdown report ('-' = stdout)
  --allow-lockbox            permit a test window overlapping 2026H2
                             (declared final test only; default refuses)
  --quiet                    reserved; text output stays on stdout
  -h, --help                 show this help
",
        DEFAULT_COST_BPS = walkforward::DEFAULT_COST_BPS,
        DEFAULT_QUANTILE = walkforward::DEFAULT_QUANTILE
    )
}

fn run_walkforward(args: &[String]) -> ! {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", walkforward_usage());
        std::process::exit(EXIT_OK);
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("rcsr {}", VERSION);
        std::process::exit(EXIT_OK);
    }
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut expressions: Vec<String> = Vec::new();
    let mut vars_src: Option<String> = None;
    let mut time_col: Option<String> = None;
    let mut group_col: Option<String> = None;
    let mut windows: Vec<walkforward::Window> = Vec::new();
    let mut windows_file: Option<String> = None;
    let mut horizon: Option<f64> = None;
    let mut embargo: Option<f64> = None;
    let mut cost_flag: Option<f64> = None;
    let mut cost_override: Option<f64> = None;
    let mut target_spec_path: Option<String> = None;
    let mut from_run_path: Option<String> = None;
    let mut entry_offset_flag: Option<f64> = None;
    let mut delay_flag: Option<f64> = None;
    let mut quantile = walkforward::DEFAULT_QUANTILE;
    let mut quantile_set = false;
    let mut threshold_fixed: Option<f64> = None;
    let mut policy = walkforward::Policy::Threshold;
    let mut band_width = 0.0f64;
    let mut min_hold = 1usize;
    let mut gpu = false;
    let mut validation_frac = 0.0f64;
    let mut periods_per_year: Option<f64> = None;
    let mut pool_reset_flat = false;
    let mut device_pnl = false;
    let mut control_seed = rcsr::baselines::DEFAULT_CONTROL_SEED;
    let mut band_select = false;
    let mut json = false;
    let mut out: Option<String> = None;
    let mut markdown: Option<String> = None;
    let mut quiet = false;
    let mut allow_lockbox = false;
    let mut popts = walkforward::ProtocolOptions::default();
    let mut windows_scheme: Option<String> = None;
    let mut ledger_path: Option<String> = None;
    let mut sealed_path: Option<String> = None;
    let mut locked_test = false;
    let mut verify_read = false;
    let mut allow_unbound = false;
    let mut lockbox_arg: Option<String> = None;
    let mut lockbox_embargo_arg: Option<String> = None;
    let mut expect_hash: Option<String> = None;
    let mut front_path: Option<String> = None;
    let mut allow_in_sample = false;

    let need = |args: &[String], i: usize, flag: &str| -> String {
        match args.get(i) {
            Some(v) => v.clone(),
            None => {
                eprintln!("error: {} requires a value", flag);
                std::process::exit(EXIT_USAGE);
            }
        }
    };
    let number = |src: String, flag: &str| -> f64 {
        match src.parse::<f64>() {
            Ok(v) if v.is_finite() => v,
            _ => {
                eprintln!("error: {} needs a finite number, got '{}'", flag, src);
                std::process::exit(EXIT_USAGE);
            }
        }
    };
    let mut i = 0usize;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                data_path = Some(need(args, i + 1, "--data"));
                i += 1;
            }
            "--target" => {
                target = Some(need(args, i + 1, "--target"));
                i += 1;
            }
            "--expr" | "--score-expr" => {
                expressions.push(need(args, i + 1, "--expr"));
                i += 1;
            }
            "--vars" => {
                vars_src = Some(need(args, i + 1, "--vars"));
                i += 1;
            }
            "--time" => {
                time_col = Some(need(args, i + 1, "--time"));
                i += 1;
            }
            "--group-by" => {
                group_col = Some(need(args, i + 1, "--group-by"));
                i += 1;
            }
            "--window" => {
                let src = need(args, i + 1, "--window");
                match walkforward::Window::parse(&src) {
                    Ok(w) => windows.push(w),
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
                i += 1;
            }
            "--windows" => {
                windows_file = Some(need(args, i + 1, "--windows"));
                i += 1;
            }
            "--horizon" => {
                horizon = Some(number(need(args, i + 1, "--horizon"), "--horizon"));
                i += 1;
            }
            "--embargo" => {
                embargo = Some(number(need(args, i + 1, "--embargo"), "--embargo"));
                i += 1;
            }
            "--cost-bps" => {
                cost_flag = Some(number(need(args, i + 1, "--cost-bps"), "--cost-bps"));
                i += 1;
            }
            "--cost-override" => {
                cost_override = Some(number(need(args, i + 1, "--cost-override"), "--cost-override"));
                i += 1;
            }
            "--target-spec" => {
                target_spec_path = Some(need(args, i + 1, "--target-spec"));
                i += 1;
            }
            "--from-run" => {
                from_run_path = Some(need(args, i + 1, "--from-run"));
                i += 1;
            }
            "--front" => {
                front_path = Some(need(args, i + 1, "--front"));
                i += 1;
            }
            "--allow-in-sample-windows" => allow_in_sample = true,
            "--threshold-quantile" => {
                quantile = number(
                    need(args, i + 1, "--threshold-quantile"),
                    "--threshold-quantile",
                );
                quantile_set = true;
                i += 1;
            }
            "--threshold-fixed" => {
                threshold_fixed = Some(number(
                    need(args, i + 1, "--threshold-fixed"),
                    "--threshold-fixed",
                ));
                i += 1;
            }
            "--policy" => {
                let src = need(args, i + 1, "--policy");
                policy = match walkforward::Policy::parse(&src) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                };
                i += 1;
            }
            "--band" | "--band-width" => {
                band_width = number(need(args, i + 1, "--band"), "--band");
                i += 1;
            }
            "--min-hold" => {
                let src = need(args, i + 1, "--min-hold");
                min_hold = match src.parse::<usize>() {
                    Ok(v) if v >= 1 => v,
                    _ => {
                        eprintln!("error: --min-hold needs an integer >= 1, got '{}'", src);
                        std::process::exit(EXIT_USAGE);
                    }
                };
                i += 1;
            }
            "--validation-frac" => {
                validation_frac = number(
                    need(args, i + 1, "--validation-frac"),
                    "--validation-frac",
                );
                i += 1;
            }
            "--periods-per-year" => {
                periods_per_year = Some(number(
                    need(args, i + 1, "--periods-per-year"),
                    "--periods-per-year",
                ));
                i += 1;
            }
            "--pool-reset-flat" => pool_reset_flat = true,
            "--device-pnl" => device_pnl = true,
            "--batch-pnl" => popts.batch_pnl = true,
            "--gpu" => gpu = true,
            "--control-seed" => {
                let src = need(args, i + 1, "--control-seed");
                control_seed = match src.parse::<u64>() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("error: --control-seed needs a non-negative integer, got '{}'", src);
                        std::process::exit(EXIT_USAGE);
                    }
                };
                i += 1;
            }
            "--select-band" => band_select = true,
            "--json" => json = true,
            "--out" => {
                out = Some(need(args, i + 1, "--out"));
                i += 1;
            }
            "--markdown" => {
                markdown = Some(need(args, i + 1, "--markdown"));
                i += 1;
            }
            "--quiet" => quiet = true,
            "--entry-offset" => {
                entry_offset_flag = Some(number(need(args, i + 1, "--entry-offset"), "--entry-offset"));
                i += 1;
            }
            "--delay" => {
                delay_flag = Some(number(need(args, i + 1, "--delay"), "--delay"));
                i += 1;
            }
            "--windows-scheme" => {
                windows_scheme = Some(need(args, i + 1, "--windows-scheme"));
                i += 1;
            }
            "--bootstrap" | "--block-len" | "--null-draws" | "--trials" | "--protocol-seed" | "--pbo-blocks" => {
                let flag = args[i].clone();
                let src = need(args, i + 1, &flag);
                let v = match src.parse::<u64>() {
                    Ok(v) => v,
                    Err(_) => {
                        eprintln!("error: {} needs a non-negative integer, got '{}'", flag, src);
                        std::process::exit(EXIT_USAGE);
                    }
                };
                match flag.as_str() {
                    "--bootstrap" => popts.bootstrap_draws = v as usize,
                    "--block-len" => popts.block_len = Some(v as usize),
                    "--null-draws" => popts.null_draws = v as usize,
                    "--trials" => popts.trials = Some(v as usize),
                    "--protocol-seed" => popts.seed = v as u32,
                    _ => popts.pbo_blocks = v as usize,
                }
                i += 1;
            }
            "--fdr-q" => {
                popts.fdr_q = number(need(args, i + 1, "--fdr-q"), "--fdr-q");
                i += 1;
            }
            "--ledger" => {
                ledger_path = Some(need(args, i + 1, "--ledger"));
                i += 1;
            }
            "--sealed" => {
                sealed_path = Some(need(args, i + 1, "--sealed"));
                i += 1;
            }
            "--lockbox" => {
                lockbox_arg = Some(need(args, i + 1, "--lockbox"));
                i += 1;
            }
            "--lockbox-embargo" => {
                lockbox_embargo_arg = Some(need(args, i + 1, "--lockbox-embargo"));
                i += 1;
            }
            "--expect-report-sha256" => {
                expect_hash = Some(need(args, i + 1, "--expect-report-sha256"));
                i += 1;
            }
            "--locked-test" => locked_test = true,
            "--verify-read" => verify_read = true,
            "--allow-unbound-ledger" => allow_unbound = true,
            "--allow-lockbox" => allow_lockbox = true,
            other if other.starts_with("--policy=") => {
                let src = &other["--policy=".len()..];
                policy = match walkforward::Policy::parse(src) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                };
            }
            other if other.starts_with("--band=") => {
                band_width = number(
                    other["--band=".len()..].to_string(),
                    "--band",
                );
            }
            other if other.starts_with("--band-width=") => {
                band_width = number(
                    other["--band-width=".len()..].to_string(),
                    "--band-width",
                );
            }
            other if other.starts_with("--min-hold=") => {
                let src = &other["--min-hold=".len()..];
                min_hold = match src.parse::<usize>() {
                    Ok(v) if v >= 1 => v,
                    _ => {
                        eprintln!("error: --min-hold needs an integer >= 1, got '{}'", src);
                        std::process::exit(EXIT_USAGE);
                    }
                };
            }
            other if other.starts_with('-') => {
                eprintln!(
                    "error: unknown walkforward option '{}'\n\n{}",
                    other,
                    walkforward_usage()
                );
                std::process::exit(EXIT_USAGE);
            }
            other => {
                eprintln!("error: unexpected argument '{}'", other);
                std::process::exit(EXIT_USAGE);
            }
        }
        i += 1;
    }

    let data_path = data_path.unwrap_or_else(|| {
        eprintln!("error: walkforward needs --data\n\n{}", walkforward_usage());
        std::process::exit(EXIT_USAGE);
    });
    let usage_fail = |msg: &str| -> ! {
        eprintln!("error: {}", msg);
        std::process::exit(EXIT_USAGE);
    };
    let input_fail = |msg: &str| -> ! {
        eprintln!("error: {}", msg);
        std::process::exit(EXIT_INPUT);
    };
    // #487: one declaration of the label window, the return units and the
    // cost, read from a target spec or from the run that selected the book.
    if from_run_path.is_some() && (target_spec_path.is_some() || front_path.is_some()) {
        usage_fail("--from-run, --front and --target-spec are alternatives: a run carries its own target spec");
    }
    if front_path.is_some() && target_spec_path.is_some() {
        usage_fail("--front and --target-spec are alternatives: the run carries its own target spec");
    }
    if policy == walkforward::Policy::TopTercile && (quantile_set || threshold_fixed.is_some()) {
        usage_fail("--policy top-tercile has no threshold: do not pass --threshold-quantile or --threshold-fixed");
    }
    let mut cost_spec: Option<rcsr::target::TargetSpec> = None;
    let mut run_cost: Option<(f64, rcsr::target::CostSource)> = None;
    let mut front_run: Option<rcsr::run_json::RunFile> = None;
    if let Some(path) = &from_run_path {
        if !expressions.is_empty() {
            usage_fail("--from-run takes the expression from the run; do not pass --expr");
        }
        if quantile_set || threshold_fixed.is_some() {
            usage_fail(
                "--from-run takes the threshold rule from the run; do not pass \
                 --threshold-quantile or --threshold-fixed",
            );
        }
        let run = rcsr::run_json::RunFile::load(path).unwrap_or_else(|e| input_fail(&e));
        let book = run.book().unwrap_or_else(|e| input_fail(&format!("{}: {}", path, e)));
        expressions.push(book.expression.clone());
        vars_src = Some(book.variables.join(","));
        if book.position == rcsr::trading::PositionMap::TopTercile {
            if policy != walkforward::Policy::Threshold && policy != walkforward::Policy::TopTercile {
                usage_fail("--policy conflicts with the run: it priced the top_tercile book");
            }
            policy = walkforward::Policy::TopTercile;
        } else {
            if policy == walkforward::Policy::TopTercile {
                usage_fail("--policy top-tercile conflicts with the run: it priced the threshold book");
            }
            threshold_fixed = Some(book.threshold);
        }
        run_cost = Some((book.per_side_bps, book.source));
        cost_spec = Some(book.spec);
        front_run = Some(run);
    } else if let Some(path) = &front_path {
        if !expressions.is_empty() {
            eprintln!(
                "error: --front and --expr are alternatives; the candidates come from one source\n\n{}",
                walkforward_usage()
            );
            std::process::exit(EXIT_USAGE);
        }
        let run = rcsr::run_json::RunFile::load(path).unwrap_or_else(|e| {
            eprintln!("error: --front {}", e);
            std::process::exit(EXIT_INPUT);
        });
        if run.front.is_empty() {
            eprintln!("error: --front {}: the Pareto front is empty", path);
            std::process::exit(EXIT_INPUT);
        }
        if run.train_time.is_none() {
            eprintln!(
                "error: --front {}: the run has no split.train_time (no time column); \
                 the walk-forward needs time",
                path
            );
            std::process::exit(EXIT_INPUT);
        }
        expressions = run.front.iter().map(|m| m.expression.clone()).collect();
        popts.front = Some(walkforward::FrontSource {
            run: path.clone(),
            run_sha256: run.sha256.clone(),
            labels: run.front.iter().map(|m| format!("front_rank_{}", m.rank)).collect(),
        });
        if vars_src.is_none() && !run.variables.is_empty() {
            vars_src = Some(run.variables.join(","));
        }
        if target.is_none() {
            target = run.return_column.clone();
        }
        if popts.trials.is_none() {
            popts.trials = Some(match run.candidates_evaluated {
                Some(n) if n > 0 => n as usize,
                _ => {
                    eprintln!(
                        "error: --front {}: the run has no search.candidates_evaluated; pass --trials",
                        path
                    );
                    std::process::exit(EXIT_USAGE);
                }
            });
        }
        // The front is priced as the books the search compared: the run's
        // fixed threshold, cost and target spec.
        if run.position == rcsr::trading::PositionMap::TopTercile {
            if policy != walkforward::Policy::Threshold && policy != walkforward::Policy::TopTercile {
                usage_fail("--policy conflicts with the run: it priced the top_tercile book");
            }
            policy = walkforward::Policy::TopTercile;
        } else if let Some(theta) = run.threshold {
            if policy == walkforward::Policy::TopTercile {
                usage_fail("--policy top-tercile conflicts with the run: it priced the threshold book");
            }
            if threshold_fixed.is_none() && !quantile_set {
                threshold_fixed = Some(theta);
            }
        }
        if let Some(bps) = run.per_side_bps {
            run_cost = Some((bps, run.cost_source()));
            cost_spec = run.target_spec();
        }
        front_run = Some(run);
    } else if let Some(path) = &target_spec_path {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| input_fail(&format!("{}: {}", path, e)));
        let specs = rcsr::target::parse_file(&text).unwrap_or_else(|e| input_fail(&format!("{}: {}", path, e)));
        let want = target.clone().unwrap_or_else(|| usage_fail("--target-spec needs --target to pick the spec entry"));
        match rcsr::target::entry_for(&specs, &want) {
            Some(s) => cost_spec = Some(s.clone()),
            None => input_fail(&format!("{}: no entry for target '{}'", path, want)),
        }
    }
    let mut cost_bps = walkforward::DEFAULT_COST_BPS;
    if let Some(s) = &cost_spec {
        let effective = s.return_column.clone().unwrap_or_else(|| s.column.clone());
        if let Some(t) = &target {
            if *t != effective && *t != s.column && *t != s.name {
                usage_fail(&format!(
                    "--target {} is not the spec's return column '{}' or label column '{}'",
                    t, effective, s.column
                ));
            }
        }
        let unit_scale = matches!(&s.scale, rcsr::target::Scale::Constant { value } if *value == 1.0);
        if s.return_column.is_none() && (s.offset != 0.0 || s.clip.is_some() || !unit_scale) {
            input_fail(
                "the target is scaled, clipped or offset: declare a return_column in the \
                 target spec so the P&L uses realised returns",
            );
        }
        target = Some(effective);
        let agree = |flag: &str, given: Option<f64>, declared: f64| -> f64 {
            match given {
                Some(g) if (g - declared).abs() > 1e-9 => usage_fail(&format!(
                    "{} {} differs from the target spec ({}); remove the flag or edit the spec",
                    flag, g, declared
                )),
                _ => declared,
            }
        };
        horizon = Some(agree("--horizon", horizon, s.horizon_seconds));
        popts.entry_offset = agree("--entry-offset", entry_offset_flag, s.entry_offset_seconds);
        popts.delay = agree("--delay", delay_flag, s.delay_seconds);
    } else {
        popts.entry_offset = entry_offset_flag.unwrap_or(0.0);
        popts.delay = delay_flag.unwrap_or(0.0);
    }
    let resolved = match run_cost {
        Some((per_side, source)) => match (cost_flag, cost_override) {
            (Some(_), Some(_)) => Err("--cost-bps and --cost-override are alternatives; pass one of them".to_string()),
            (None, Some(o)) => Ok((o, rcsr::target::CostSource::Flag)),
            (Some(f), None) if (f - per_side).abs() > 1e-12 => Err(format!(
                "--cost-bps {} differs from the run ({} per side); \
                 use --cost-override to price a different cost",
                f, per_side
            )),
            _ => Ok((per_side, source)),
        },
        None => rcsr::target::resolve_cost(cost_spec.as_ref(), cost_flag, cost_override, cost_bps),
    };
    let (resolved_cost, cost_source) = resolved.unwrap_or_else(|e| usage_fail(&e));
    cost_bps = resolved_cost;
    if cost_spec.is_some() {
        popts.cost = Some(rcsr::target::CostDecl::new(cost_spec.as_ref(), cost_bps, cost_source));
    } else if cost_source == rcsr::target::CostSource::Flag {
        popts.cost = Some(rcsr::target::CostDecl::new(None, cost_bps, cost_source));
    }
    let horizon = horizon.unwrap_or(0.0);
    let target = target.unwrap_or_else(|| {
        eprintln!("error: walkforward needs --target\n\n{}", walkforward_usage());
        std::process::exit(EXIT_USAGE);
    });
    if expressions.is_empty() {
        eprintln!("error: walkforward needs --expr or --front\n\n{}", walkforward_usage());
        std::process::exit(EXIT_USAGE);
    }
    if threshold_fixed.is_some() && quantile_set {
        eprintln!(
            "error: --threshold-fixed and --threshold-quantile are alternatives; \
             pass one of them\n\n{}",
            walkforward_usage()
        );
        std::process::exit(EXIT_USAGE);
    }
    if band_select && expressions.len() < 2 {
        eprintln!(
            "error: --select-band needs at least two --expr candidates; \
             one candidate has nothing to select\n\n{}",
            walkforward_usage()
        );
        std::process::exit(EXIT_USAGE);
    }
    let expression = expressions[0].clone();
    let candidates = if expressions.len() > 1 {
        expressions
    } else {
        Vec::new()
    };
    let time_col = time_col.unwrap_or_else(|| {
        eprintln!("error: walkforward needs --time\n\n{}", walkforward_usage());
        std::process::exit(EXIT_USAGE);
    });
    if let Some(file) = windows_file {
        match std::fs::read_to_string(&file) {
            Ok(text) if rcsr::protocol::is_windows_table(&text) => {
                match rcsr::protocol::parse_windows_csv(&text, windows_scheme.as_deref()) {
                    Ok(rows) => {
                        for r in rows {
                            popts
                                .validation_starts
                                .push((r.window.label.clone(), r.validation_start));
                            windows.push(r.window);
                        }
                    }
                    Err(e) => {
                        eprintln!("error: {}: {}", file, e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            Ok(text) => {
                let mut seen = 0usize;
                for (n, line) in text.lines().enumerate() {
                    let line = line.trim();
                    if line.is_empty() || line.starts_with('#') {
                        continue;
                    }
                    // Skip a header row: it names the columns rather than
                    // carrying four parseable timestamps.
                    if seen == 0
                        && (line.contains("train_start")
                            || line.contains("train_end")
                            || line.starts_with("label,"))
                    {
                        seen += 1;
                        continue;
                    }
                    seen += 1;
                    match walkforward::Window::parse(line) {
                        Ok(w) => windows.push(w),
                        Err(e) => {
                            eprintln!("error: {}:{}: {}", file, n + 1, e);
                            std::process::exit(EXIT_USAGE);
                        }
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {}: {}", file, e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }
    if band_select && validation_frac <= 0.0 && popts.validation_starts.is_empty() {
        eprintln!(
            "error: --select-band needs --validation-frac > 0 or a windows table with validation_start; \
             the test block may not choose the candidate\n\n{}",
            walkforward_usage()
        );
        std::process::exit(EXIT_USAGE);
    }
    if windows.is_empty() {
        eprintln!(
            "error: walkforward needs at least one --window or --windows file\n\n{}",
            walkforward_usage()
        );
        std::process::exit(EXIT_USAGE);
    }
    if let Some(run) = front_run.as_ref() {
        for w in &windows {
            if run.in_sample(w.test_start) {
                if allow_in_sample {
                    popts.in_sample_windows.push(w.label.clone());
                    continue;
                }
                match run.earliest_test_start() {
                    Some((_, seen, purge)) => eprintln!(
                        "error: window '{}' tests rows the run trained or selected on \
                         (test_start {} < last seen {} + purge {})",
                        w.label, w.test_start, seen, purge
                    ),
                    None => eprintln!(
                        "error: window '{}' tests rows the run trained or selected on \
                         (the run has no split.train_time)",
                        w.label
                    ),
                }
                std::process::exit(EXIT_USAGE);
            }
        }
    }
    let vars: Vec<String> = vars_src
        .map(|s| {
            s.split(',')
                .map(|v| v.trim().to_string())
                .filter(|v| !v.is_empty())
                .collect()
        })
        .unwrap_or_default();
    let spec = walkforward::EvalSpec {
        expression,
        vars,
        target,
        time: time_col,
        group: group_col,
        windows,
        horizon,
        embargo,
        cost_bps,
        threshold_quantile: quantile,
        threshold_fixed,
        validation_frac,
        periods_per_year,
        candidates,
        band_select,
        pool_reset_flat,
        allow_lockbox,
        device_pnl,
        control_seed,
        policy,
        band_width,
        min_hold,
        gpu,
    };
    let resolved = rcsr::lockbox::resolve_with(lockbox_arg.as_deref(), lockbox_embargo_arg.as_deref(), &data_path).unwrap_or_else(|e| {
        eprintln!("error: {}", e.message);
        std::process::exit(e.exit);
    });
    let declared_lock: Option<(f64, f64)> = resolved.range.map(|l| (l.start as f64, l.end as f64));
    // A sealed read opens the primary range only; every other declared range
    // stays sealed until its own sealed read.
    popts.locked_ranges = resolved.ranges().iter().skip(usize::from(locked_test)).map(|r| (r.start, r.end)).collect();
    popts.locked_embargo = resolved.embargo.seconds;
    if declared_lock.is_none() && (ledger_path.is_some() || locked_test) {
        eprintln!("error: --ledger and --locked-test need a locked test range: drop --lockbox none");
        std::process::exit(EXIT_USAGE);
    }
    if locked_test && (ledger_path.is_none() || sealed_path.is_none()) {
        eprintln!("error: --locked-test needs --ledger and --sealed");
        std::process::exit(EXIT_USAGE);
    }
    for (ri, rg) in resolved.ranges().into_iter().enumerate() {
        let sealed_read = locked_test && ri == 0;
        let (ls, le) = (rg.start as f64, rg.end as f64);
        let label_purge = popts.required_purge_seconds(spec.horizon).max(spec.embargo_seconds());
        let boundary = ls - label_purge;
        let reach_end = if sealed_read { le } else { le + resolved.embargo.seconds as f64 };
        for w in &spec.windows {
            let train_hits = w.train_start < reach_end && w.train_end > boundary;
            let test_hits = w.test_start < reach_end && w.test_end > boundary;
            let test_inside = w.test_start >= ls && w.test_end <= le;
            let problem = if sealed_read {
                if train_hits {
                    Some("trains within the label purge of the locked test".to_string())
                } else if !test_inside {
                    Some("tests outside the locked test".to_string())
                } else {
                    None
                }
            } else if train_hits || test_hits {
                Some(format!(
                    "reaches the declared locked test {}..{} (label end = stamp + {} s, lookback embargo {} s after the end); \
                     end it at or before {} or start it at or after {}, or read the locked test only with --locked-test and a sealed list",
                    rcsr::protocol::iso(ls),
                    rcsr::protocol::iso(le),
                    label_purge,
                    resolved.embargo.seconds,
                    rcsr::protocol::iso(boundary),
                    rcsr::protocol::iso(reach_end)
                ))
            } else {
                None
            };
            if let Some(p) = problem {
                eprintln!("error: window '{}' {}", w.label, p);
                std::process::exit(EXIT_INPUT);
            }
        }
    }
    let (data_sha, build_id) = if ledger_path.is_some() {
        let sha = match rcsr::manifest::sha256_file(&data_path) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let (commit, dirty) = rcsr::manifest::build_info();
        (sha, if dirty { format!("{}-dirty", commit) } else { commit })
    } else {
        (String::new(), String::new())
    };
    let spec_sha = rcsr::manifest::sha256_hex(rcsr::protocol::procedure_spec(&spec, &popts).as_bytes());
    let candidate_id = rcsr::protocol::Candidate::address(&spec_sha, &data_sha, &build_id);
    let ledger = ledger_path.as_ref().map(rcsr::protocol::Ledger::new);
    let current_range = resolved
        .range
        .map(|l| rcsr::protocol::LockedRange::new(&l, resolved.source.as_str()));
    let binding = current_range
        .as_ref()
        .map(|current| rcsr::protocol::RangeBinding { current, allow_unbound });
    let mut read_count = 0u64;
    if let (Some(l), true) = (&ledger, locked_test) {
        match l.final_seal() {
            Ok(Some((_, terms))) if !terms.report_candidates => {
                if popts.batch_pnl {
                    eprintln!(
                        "error: --batch-pnl scores every candidate on the locked test; the final seal reports \
                         the selected candidate only (seal --final --report-candidates keeps the full report)"
                    );
                    std::process::exit(EXIT_USAGE);
                }
                popts.final_read = true;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }
    if let Some(l) = &ledger {
        let r = (|| -> Result<u64, String> {
            l.verify_chain()?;
            if locked_test {
                let sealed = std::path::Path::new(sealed_path.as_deref().unwrap_or(""));
                let entry = if verify_read {
                    l.authorize_sealed(sealed, &candidate_id, &data_sha, &build_id, binding)?
                } else {
                    l.authorize_read(sealed, &candidate_id, &data_sha, &build_id, binding)?
                };
                if entry.locked_test.is_none() {
                    eprintln!("warning: candidate {} has no recorded locked test; the read is unbound", candidate_id);
                }
                l.check_read_marker(&data_path)?;
            } else {
                l.refuse_if_final()?;
            }
            l.read_count()
        })();
        match r {
            Ok(c) => read_count = c,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }
    let report = match walkforward::run_path_with(&data_path, &spec, &popts) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(if e.contains("overlapping returns") { EXIT_USAGE } else { EXIT_INPUT });
        }
    };
    if spec.threshold_fixed.is_none() {
        for c in &report.candidates {
            let constant = expr::parse(c, &report.variables)
                .map(|t| rcsr::trading::is_constant(&t))
                .unwrap_or(false);
            if constant {
                eprintln!(
                    "warning: score expression '{}' is constant: every quantile threshold makes \
                     the book flat; pass --from-run or --threshold-fixed to price the book the search selected",
                    c
                );
            }
        }
    }
    let report_sha = rcsr::manifest::sha256_hex(report.json().as_bytes());
    if let Some(want) = &expect_hash {
        if want != &report_sha {
            eprintln!("error: report sha256 {} differs from the expected {}", report_sha, want);
            std::process::exit(EXIT_INPUT);
        }
    }
    let protocol_json: Option<String>;
    if let Some((ls, le)) = declared_lock {
        let rows: usize = if locked_test { report.windows.iter().map(|w| w.test_rows).sum() } else { 0 };
        let first = report.windows.iter().map(|w| w.window.test_start).fold(f64::INFINITY, f64::min);
        let last = report.windows.iter().map(|w| w.window.test_end).fold(f64::NEG_INFINITY, f64::max);
        if let Some(l) = &ledger {
            let outcome = if locked_test && verify_read {
                match l.read_report_sha(&candidate_id) {
                    Some(h) if h == report_sha => Ok(read_count),
                    Some(h) => Err(format!("report sha256 {} differs from the recorded read {}", report_sha, h)),
                    None => Err("no recorded read for this candidate".to_string()),
                }
            } else if locked_test {
                l.write_read_marker(&data_path, "test_read", &report_sha)
                    .and_then(|_| l.record_read(&candidate_id, &report_sha, current_range.as_ref()))
            } else {
                l.add_candidate(&rcsr::protocol::Candidate {
                    id: candidate_id.clone(),
                    spec_sha256: spec_sha.clone(),
                    data_sha256: data_sha.clone(),
                    build_commit: build_id.clone(),
                    report_sha256: report_sha.clone(),
                    expression: report.candidates.join(" | "),
                    locked_test: current_range.clone(),
                })
                .map(|_| read_count)
            };
            match outcome {
                Ok(c) => read_count = c,
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            }
        }
        let facts = rcsr::protocol::LockboxFacts {
            start: ls,
            end: le,
            rows,
            first_ts: if rows > 0 { Some(first) } else { None },
            last_ts: if rows > 0 { Some(last) } else { None },
            read_count,
        };
        let mut lockbox_facts = facts.json();
        lockbox_facts.pop();
        lockbox_facts.push(',');
        lockbox_facts.push_str(&resolved.report_fields(rows, locked_test, true));
        if locked_test {
            lockbox_facts.push_str(&format!(
                ",\"data_sha256\":{},\"read_marker\":{}",
                jstr(&data_sha),
                jstr(&rcsr::protocol::marker_path(&data_path))
            ));
        }
        lockbox_facts.push('}');
        protocol_json = Some(format!(
            "{{\"lockbox\":{},\"candidate_id\":{},\"report_sha256\":{},\"test_read\":{},\
             \"ledger\":{},\"windows_scheme\":{}}}",
            lockbox_facts,
            if ledger.is_some() { format!("\"{}\"", candidate_id) } else { "null".to_string() },
            format!("\"{}\"", report_sha),
            locked_test,
            ledger_path.as_ref().map_or("null".to_string(), |p| format!("{:?}", p)),
            windows_scheme.as_ref().map_or("null".to_string(), |s| format!("{:?}", s)),
        ));
    } else {
        protocol_json = Some(format!(
            "{{\"lockbox\":{{{}}},\"candidate_id\":null,\"report_sha256\":\"{}\",\"test_read\":false,\
             \"ledger\":null,\"windows_scheme\":{}}}",
            resolved.report_fields(0, false, false),
            report_sha,
            windows_scheme.as_ref().map_or("null".to_string(), |s| format!("{:?}", s)),
        ));
    }
    let report_json = || report.json_with_ledger(protocol_json.as_deref());
    for w in &report.windows {
        let reasons = [
            w.band.as_ref().and_then(|b| b.fallback_reason.as_deref()),
            w.hindsight_bound.as_ref().and_then(|h| h.fallback_reason.as_deref()),
        ];
        for reason in reasons.into_iter().flatten() {
            eprintln!(
                "note: window '{}': --gpu scan ran on the CPU: {}",
                w.window.label, reason
            );
        }
    }
    let wrote_out = out.is_some();
    let wrote_markdown = markdown.is_some();
    if let Some(path) = out {
        if let Err(e) = std::fs::write(&path, report_json()) {
            eprintln!("error: {}: {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    }
    if let Some(path) = markdown {
        if path == "-" {
            print!("{}", report.markdown());
        } else if let Err(e) = std::fs::write(&path, report.markdown()) {
            eprintln!("error: {}: {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    }
    if json {
        println!("{}", report_json());
    } else if !quiet && !wrote_out && !wrote_markdown {
        print!("{}", report.text());
    }
    std::process::exit(EXIT_OK);
}

fn import_usage() -> String {
    format!(
        "rcsr {VERSION} — import a CSV into a versioned column store

USAGE:
  rcsr import <input.csv> [<store.rcsb>] [options]

  The output defaults to the input path with its extension replaced by `.rcsb`.
  The store is deterministic: the same input and --group-rows produce the same
  bytes. It is searched with `rcsr --data <store.rcsb> ...`; the guard reads
  the header and loads the full rows when the exact estimate fits. If it does
  not, the run refuses unless `--force` or an explicit `--sample-rows N` is
  given; it never samples silently.

OPTIONS:
  --data <input.csv>       input table (alternative to the positional path)
  --out <store.rcsb>       output store
  --group-rows <N>         rows per I/O group (default {})
  --time <column>          store this column as exact i64 timestamps
                           (default: the first integral, time-like column)
  --skip-malformed         drop ragged rows instead of refusing the input
  --max-memory <size>      refuse before parsing when the streamed and
                           calibrated rows x
                           columns estimate exceeds this (not 33x file size)
  --force                  override --max-memory and load the full input
  --quiet                  do not print the summary
  -h, --help               show this help
",
        store::DEFAULT_GROUP_ROWS
    )
}

/// `rcsr verify-manifest <path> [--data <file>] [--binary <file>]` (issues #140, #256).
///
/// Exit 0 when every check passes, 1 on any mismatch, 2 on a usage error.
fn run_verify_manifest(args: &[String]) -> ! {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        println!(
            "rcsr verify-manifest <manifest.json> [--data <file>] [--binary <file>]\n\n\
             Check a --run-manifest against the current binary, build commit, data\n\
             file and canonical configuration. --binary names the executable to\n\
             hash instead of the running one (the preserved binary of a finished\n\
             run). Exit 0 when all match, 1 otherwise."
        );
        std::process::exit(EXIT_OK);
    }
    let mut path: Option<String> = None;
    let mut data: Option<String> = None;
    let mut binary: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--binary" | "-b" => {
                i += 1;
                match args.get(i) {
                    Some(v) => binary = Some(v.clone()),
                    None => {
                        eprintln!("error: --binary requires a value");
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--data" | "-d" => {
                i += 1;
                match args.get(i) {
                    Some(v) => data = Some(v.clone()),
                    None => {
                        eprintln!("error: --data requires a value");
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            other if other.starts_with('-') => {
                eprintln!("error: unknown verify-manifest argument '{}'", other);
                std::process::exit(EXIT_USAGE);
            }
            other => {
                if path.is_some() {
                    eprintln!("error: verify-manifest takes exactly one manifest path");
                    std::process::exit(EXIT_USAGE);
                }
                path = Some(other.to_string());
            }
        }
        i += 1;
    }
    let Some(path) = path else {
        eprintln!("error: verify-manifest needs a manifest path");
        std::process::exit(EXIT_USAGE);
    };
    match manifest::verify(&path, data.as_deref(), binary.as_deref()) {
        Ok(lines) => {
            for line in lines {
                println!("{}", line);
            }
            println!("PASS manifest {}", path);
            std::process::exit(EXIT_OK);
        }
        Err(lines) => {
            for line in lines.lines() {
                eprintln!("{}", line);
            }
            eprintln!("FAIL manifest {}", path);
            std::process::exit(1);
        }
    }
}

/// #129: `rcsr audit` — the fail-closed leakage suite.
///
/// Exit codes: 0 clean, 2 usage, 3 input, 5 leakage. The JSON report always
/// includes the resolved spec hash (D11), every finding's detector/column/row,
/// and the effective sample size (D6).
fn run_audit(args: &[String]) -> ! {
    let mut data_path: Option<String> = None;
    let mut target: Option<String> = None;
    let mut time_name: Option<String> = None;
    let mut vars: Vec<String> = Vec::new();
    let mut horizon = 1usize;
    let mut unit = data::TimeUnit::Seconds;
    let mut json_out = false;
    let mut report_path: Option<String> = None;
    let mut join_on: Option<String> = None;
    let mut bars_path: Option<String> = None;
    let mut universe_path: Option<String> = None;
    let mut actions_path: Option<String> = None;
    let mut symbol_col: Option<String> = None;
    let mut price_col: Option<String> = None;
    let mut resample_small = 15usize;
    let mut resample_large = 60usize;
    let mut session_gap: Option<f64> = None;
    let mut lockbox_arg: Option<String> = None;
    let mut lockbox_embargo_arg: Option<String> = None;
    let mut target_spec: Option<String> = None;
    let mut i = 0usize;
    while i < args.len() {
        let a = args[i].as_str();
        let take = |i: &mut usize| -> String {
            *i += 1;
            args.get(*i).cloned().unwrap_or_default()
        };
        match a {
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(EXIT_OK);
            }
            "--json" => json_out = true,
            "--data" => data_path = Some(take(&mut i)),
            "--target" => target = Some(take(&mut i)),
            "--time" => time_name = Some(take(&mut i)),
            "--vars" => {
                vars = take(&mut i)
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            "--horizon" | "--label-horizon" => {
                horizon = take(&mut i).parse().unwrap_or(0);
            }
            "--time-unit" => {
                let v = take(&mut i);
                match data::TimeUnit::parse(&v) {
                    Ok(u) => unit = u,
                    Err(e) => {
                        eprintln!("error: {}", e);
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--report" => report_path = Some(take(&mut i)),
            "--lockbox" => lockbox_arg = Some(take(&mut i)),
            "--lockbox-embargo" => lockbox_embargo_arg = Some(take(&mut i)),
            "--target-spec" => target_spec = Some(take(&mut i)),
            "--join-on" => join_on = Some(take(&mut i)),
            "--bars" => bars_path = Some(take(&mut i)),
            "--universe" => universe_path = Some(take(&mut i)),
            "--corporate-actions" => actions_path = Some(take(&mut i)),
            "--symbol" => symbol_col = Some(take(&mut i)),
            "--price" => price_col = Some(take(&mut i)),
            "--resample-small" => {
                resample_small = take(&mut i).parse().unwrap_or(0);
            }
            "--resample-large" => {
                resample_large = take(&mut i).parse().unwrap_or(0);
            }
            "--session-gap" => {
                session_gap = take(&mut i).parse::<f64>().ok();
            }
            other => {
                eprintln!("error: unknown audit option '{}'", other);
                std::process::exit(EXIT_USAGE);
            }
        }
        i += 1;
    }
    let Some(path) = data_path else {
        eprintln!("error: audit needs --data <file>");
        std::process::exit(EXIT_USAGE);
    };
    let Some(target) = target else {
        eprintln!("error: audit needs --target <col>");
        std::process::exit(EXIT_USAGE);
    };
    if horizon == 0 {
        eprintln!("error: --horizon must be at least 1");
        std::process::exit(EXIT_USAGE);
    }
    let table = match data::read_table(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    let target_idx = match table.index_of(&target) {
        Ok(j) if table.numeric[j] => j,
        Ok(_) => {
            eprintln!("error: target '{}' is not numeric", target);
            std::process::exit(EXIT_INPUT);
        }
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };

    // Assemble the value columns. The time column is metadata, never a feature.
    let time_idx = match &time_name {
        Some(name) => match table.index_of(name) {
            Ok(j) => Some(j),
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        },
        None => None,
    };
    let mut value_idx: Vec<usize> = if vars.is_empty() {
        (0..table.names.len())
            .filter(|&j| table.numeric[j] && Some(j) != time_idx)
            .collect()
    } else {
        let mut v = Vec::new();
        for name in &vars {
            match table.index_of(name) {
                Ok(j) if table.numeric[j] => v.push(j),
                Ok(_) => {
                    eprintln!("error: input '{}' is not numeric", name);
                    std::process::exit(EXIT_INPUT);
                }
                Err(e) => {
                    eprintln!("error: {}", e);
                    std::process::exit(EXIT_INPUT);
                }
            }
        }
        v
    };
    if !value_idx.contains(&target_idx) {
        value_idx.push(target_idx);
    }
    value_idx.sort_unstable();
    value_idx.dedup();
    let names: Vec<&str> = value_idx.iter().map(|&j| table.names[j].as_str()).collect();

    let (times, time_audit, time_report) = match time_idx {
        Some(j) => {
            let raw = data::time_keys(&table, j).to_f64();
            let (seconds, audit, report) = leakage::time_normalizer(&raw, unit);
            (seconds, Some(audit), report)
        }
        None => {
            let raw: Vec<f64> = (0..table.len()).map(|i| i as f64).collect();
            let (seconds, audit, report) = leakage::time_normalizer(&raw, data::TimeUnit::Seconds);
            (seconds, Some(audit), report)
        }
    };
    let resolved = rcsr::lockbox::resolve_with(lockbox_arg.as_deref(), lockbox_embargo_arg.as_deref(), &path).unwrap_or_else(|e| {
        eprintln!("error: {}", e.message);
        std::process::exit(e.exit);
    });
    let finite_times = times.iter().copied().filter(|v| v.is_finite());
    let (t_lo, t_hi) = finite_times.fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), v| (a.min(v), b.max(v)));
    let ranges: Vec<rcsr::lockbox::Lockbox> = if rcsr::lockbox::plausible_epoch_seconds(t_lo, t_hi) {
        resolved.ranges()
    } else {
        Vec::new()
    };
    // Rows whose label window reaches a range (the target spec purge, else the
    // default) or whose features read it (the lookback embargo) are left out too.
    let audit_purge: i64 = match target::resolve(&path, target_spec.as_deref(), &target) {
        Ok(Some(r)) => r.spec.required_purge_seconds().ceil() as i64,
        _ => rcsr::protocol::DEFAULT_PURGE_SECONDS as i64,
    };
    let keep: Vec<usize> = (0..times.len())
        .filter(|&r| {
            !ranges.iter().any(|l| {
                times[r].is_finite() && {
                    let t = times[r].floor() as i64;
                    t >= l.start - audit_purge && t < l.end + resolved.embargo.seconds
                }
            })
        })
        .collect();
    let rows_left_out = times.len() - keep.len();
    let (times, mut time_audit, time_report) = if rows_left_out == 0 {
        (times, time_audit, time_report)
    } else {
        // The time audit and its D1 findings describe the rows that stay.
        let kept: Vec<f64> = keep.iter().map(|&r| times[r]).collect();
        let (seconds, audit, report) = leakage::time_normalizer(&kept, data::TimeUnit::Seconds);
        (seconds, Some(audit), report)
    };
    let panel = match leakage::Panel::new(
        times,
        names
            .iter()
            .map(|name| {
                let j = table.index_of(name).expect("column was resolved above");
                let col = table.col(j);
                (name.to_string(), keep.iter().map(|&r| col[r]).collect())
            })
            .collect(),
    ) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };

    // #145: the leakage audit is read-only; it reports lockbox coverage rather
    // than refusing, but uses the same central guard's overlap counter.
    let mut report = time_report;
    let lockbox_coverage = rcsr::lockbox::Guard::default().overlap(
        panel.time.iter().filter(|v| v.is_finite()).map(|v| *v as i64),
    );
    report.merge(leakage::audit_panel(&panel, &target, horizon));
    // D5 differential learner is already part of audit_panel; run it explicitly
    // as well so its count is visible even when no name canary exists.
    report.merge(leakage::differential_suite(&panel, &target, horizon));

    // #254: D13 on a declared target, against the probed inputs.
    if let Some(spec_path) = target_spec.as_deref() {
        match target::resolve(&path, Some(spec_path), &target) {
            Ok(Some(r)) => {
                let probed: Vec<String> = if vars.is_empty() {
                    names
                        .iter()
                        .filter(|n| **n != target)
                        .map(|n| n.to_string())
                        .collect()
                } else {
                    vars.clone()
                };
                report.merge(leakage::target_structure(&r.spec, &probed));
            }
            Ok(None) => {}
            Err(target::ResolveError::Usage(e)) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_USAGE);
            }
            Err(target::ResolveError::Input(e)) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }

    // D10: a repeated key on the probe side would multiply joined rows.
    if let Some(col) = &join_on {
        match table.index_of(col) {
            Ok(j) => {
                let keys: Vec<u64> = keep.iter().map(|&r| table.col(j)[r].to_bits()).collect();
                let (_matched, join_report) = leakage::audit_join_cardinality(&keys, &keys);
                report.merge(join_report);
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }

    // D2: reconcile a user-supplied bar table (1m->1h vs 1m->15m->1h).
    if let Some(path) = bars_path {
        match data::read_table(&path) {
            Ok(t) => {
                let need = |name: &str| -> Result<usize, String> {
                    t.index_of(name).map_err(|e| e.to_string())
                };
                let idx = (|| -> Result<[usize; 6], String> {
                    Ok([
                        need("time")?,
                        need("open")?,
                        need("high")?,
                        need("low")?,
                        need("close")?,
                        need("volume")?,
                    ])
                })();
                match idx {
                    Ok([it, io, ih, il, ic, iv]) => {
                        let vw = t.index_of("vwap").ok();
                        let bars: Vec<leakage::Ohlcv> = (0..t.len())
                            .map(|r| leakage::Ohlcv {
                                time: t.value(r, it),
                                open: t.value(r, io),
                                high: t.value(r, ih),
                                low: t.value(r, il),
                                close: t.value(r, ic),
                                volume: t.value(r, iv),
                                vwap: vw.map(|j| t.value(r, j)).unwrap_or(t.value(r, ic)),
                            })
                            .collect();
                        report.merge(leakage::reconcile_resample(
                            &bars,
                            resample_small,
                            resample_large,
                            session_gap,
                        ));
                    }
                    Err(e) => {
                        eprintln!("error: --bars needs columns time,open,high,low,close,volume[,vwap]: {}", e);
                        std::process::exit(EXIT_INPUT);
                    }
                }
            }
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        }
    }

    // D8: point-in-time universe from an event CSV (time,symbol,change[,to]).
    if let Some(path) = universe_path {
        let Ok(t) = data::read_table(&path) else {
            eprintln!("error: cannot read universe '{}'", path);
            std::process::exit(EXIT_INPUT);
        };
        let (it, isym, ich) = match (t.index_of("time"), t.index_of("symbol"), t.index_of("change")) {
            (Ok(a), Ok(b), Ok(c)) => (a, b, c),
            _ => {
                eprintln!("error: --universe needs columns time,symbol,change[,to]");
                std::process::exit(EXIT_INPUT);
            }
        };
        let ito = t.index_of("to").ok();
        let mut events = Vec::new();
        for r in 0..t.len() {
            let change = match t.cell_text(r, ich).to_ascii_lowercase().as_str() {
                "listed" | "list" | "add" => leakage::UniverseChange::Listed,
                "delisted" | "delist" | "remove" => leakage::UniverseChange::Delisted,
                "renamed" | "rename" => leakage::UniverseChange::Renamed,
                other => {
                    eprintln!("error: unknown universe change '{}'", other);
                    std::process::exit(EXIT_INPUT);
                }
            };
            events.push(leakage::UniverseEvent {
                time: t.value(r, it),
                symbol: t.cell_text(r, isym),
                change,
                to: ito.map(|j| t.cell_text(r, j)),
            });
        }
        let Some(sym_name) = symbol_col.as_deref() else {
            eprintln!("error: --universe needs --symbol <col> to identify panel rows");
            std::process::exit(EXIT_USAGE);
        };
        let sym_idx = match table.index_of(sym_name) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let rows: Vec<leakage::UniverseRow> = (0..panel.len())
            .map(|r| leakage::UniverseRow {
                time: panel.time[r],
                symbol: table.cell_text(keep[r], sym_idx),
            })
            .collect();
        report.merge(leakage::audit_universe(&rows, &events));
    }

    // D9: corporate actions from an event CSV (time,symbol,kind,value).
    if let Some(path) = actions_path {
        let Ok(t) = data::read_table(&path) else {
            eprintln!("error: cannot read corporate actions '{}'", path);
            std::process::exit(EXIT_INPUT);
        };
        let (it, isym, ikind, ival) = match (
            t.index_of("time"),
            t.index_of("symbol"),
            t.index_of("kind"),
            t.index_of("value"),
        ) {
            (Ok(a), Ok(b), Ok(c), Ok(d)) => (a, b, c, d),
            _ => {
                eprintln!("error: --corporate-actions needs columns time,symbol,kind,value");
                std::process::exit(EXIT_INPUT);
            }
        };
        let Some(price_name) = price_col.as_deref() else {
            eprintln!("error: --corporate-actions needs --price <col>");
            std::process::exit(EXIT_USAGE);
        };
        let pj = match table.index_of(price_name) {
            Ok(j) => j,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let mut actions = Vec::new();
        let mut symbols: Vec<String> = Vec::new();
        for r in 0..t.len() {
            let symbol = t.cell_text(r, isym);
            if !symbols.contains(&symbol) {
                symbols.push(symbol.clone());
            }
            let v = t.value(r, ival);
            let kind = match t.cell_text(r, ikind).to_ascii_lowercase().as_str() {
                "split" => leakage::CorporateActionKind::Split(v),
                "dividend" => leakage::CorporateActionKind::Dividend(v),
                other => {
                    eprintln!("error: unknown corporate action '{}'", other);
                    std::process::exit(EXIT_INPUT);
                }
            };
            actions.push(leakage::CorporateAction {
                time: t.value(r, it),
                symbol,
                kind,
            });
        }
        let prices: Vec<f64> = keep.iter().map(|&r| table.col(pj)[r]).collect();
        let cuts = leakage::cut_points(&panel, 10);
        for symbol in symbols {
            report.merge(leakage::corporate_action_replay(
                &panel.time,
                &prices,
                &actions,
                &symbol,
                &cuts,
            ));
        }
    }

    // D0/D11: resolved spec and its deterministic hash.
    let digest = match manifest::sha256_file(&path) {
        Ok(h) => h,
        Err(_) => "stdin".to_string(),
    };
    let horizon_s = horizon.to_string();
    let rows_s = table.len().to_string();
    let target_s = target.clone();
    let time_s = time_name.clone().unwrap_or_else(|| "<row-index>".to_string());
    let unit_s = unit.name().to_string();
    let spec: Vec<(&str, &str)> = vec![
        ("tool", "rcsr"),
        ("command", "audit"),
        ("data_sha256", &digest),
        ("rows", &rows_s),
        ("target", &target_s),
        ("time", &time_s),
        ("time_unit", &unit_s),
        ("horizon", &horizon_s),
        ("tzdb_version", leakage::TZDB_VERSION),
    ];
    let required = [("target", target_s.as_str()), ("tzdb_version", leakage::TZDB_VERSION)];
    let defaults = [("horizon", "1"), ("time_unit", "s")];
    report.merge(leakage::audit_spec(&spec, &required, &defaults));
    report.merge(leakage::audit_determinism(&spec, &spec));
    let hash = leakage::spec_hash(&spec);

    let clean = report.is_clean();
    let n_eff = leakage::effective_n(panel.len(), horizon);
    let leaks_json: Vec<String> = report
        .leaks
        .iter()
        .map(|l| {
            format!(
                "{{\"detector\":{},\"column\":{},\"row\":{},\"message\":{}}}",
                jstr(l.detector),
                l.column.as_deref().map(jstr).unwrap_or_else(|| "null".to_string()),
                l.row.map(|r| r.to_string()).unwrap_or_else(|| "null".to_string()),
                jstr(&l.message)
            )
        })
        .collect();
    let ta = time_audit.take().unwrap_or(leakage::TimeAudit {
        rows: panel.len(),
        median_dt: 0.0,
        min_dt: 0.0,
        max_dt: 0.0,
        duplicate_instants: 0,
        non_monotonic_rows: 0,
    });
    let out = format!(
        "{{\"tool\":\"rcsr\",\"version\":{},\"command\":\"audit\",\"clean\":{},\
         \"checks\":{},\"exit_code\":{},\"spec_hash\":\"{:016x}\",\
         \"tzdb_version\":{},\"rows\":{},\"target\":{},\"time\":{},\
         \"time_unit\":{},\"horizon\":{},\"overlap_rows\":{},\"effective_n\":{},\
         \"time_audit\":{{\"rows\":{},\"median_dt\":{},\"min_dt\":{},\"max_dt\":{},\
         \"duplicate_instants\":{},\"non_monotonic_rows\":{}}},\
         \"lockbox\":{},\
         \"locked_test\":{{{},\"rows_left_out\":{}}},\
         \"leaks\":[{}]}}",
        jstr(VERSION),
        clean,
        report.checks,
        if clean { EXIT_OK } else { EXIT_LEAKAGE },
        hash,
        jstr(leakage::TZDB_VERSION),
        panel.len(),
        jstr(&target),
        jstr(&time_s),
        jstr(&unit_s),
        horizon,
        horizon.saturating_sub(1),
        n_eff,
        ta.rows,
        ta.median_dt,
        ta.min_dt,
        ta.max_dt,
        ta.duplicate_instants,
        ta.non_monotonic_rows,
        rcsr::lockbox::Lockbox::crypto_2026h2().json(
            &lockbox_coverage,
            false,
            false,
            false
        ),
        resolved.report_fields(0, false, !ranges.is_empty()),
        rows_left_out,
        leaks_json.join(",")
    );
    let locked_line = match resolved.range {
        Some(_) => format!(
            "# audit: locked test {} ({}) rows left out={}",
            resolved.ranges().iter().map(rcsr::lockbox::span).collect::<Vec<_>>().join(" + "),
            resolved.source.as_str(),
            rows_left_out
        ),
        None => "# audit: locked test none (--lockbox none) rows left out=0".to_string(),
    };
    if let Some(path) = report_path {
        if let Err(e) = std::fs::write(&path, format!("{out}\n")) {
            eprintln!("error: cannot write audit report '{}': {}", path, e);
            std::process::exit(EXIT_INPUT);
        }
    }
    if json_out {
        println!("{}", out);
    } else if clean {
        println!("{}", locked_line);
        println!(
            "# audit: lockbox rows={} windows={} (read-only, not evaluated)",
            lockbox_coverage.rows,
            lockbox_coverage.windows.len()
        );
        println!(
            "audit clean: {} checks, {} rows, N_eff {:.1}, spec {:016x}",
            report.checks,
            panel.len(),
            n_eff,
            hash
        );
    } else {
        eprintln!("audit failed ({} finding(s)):", report.leaks.len());
        eprintln!("{}", locked_line);
        for leak in &report.leaks {
            eprintln!(
                "  [{}]{}{}: {}",
                leak.detector,
                leak.column.as_deref().map(|c| format!(" column '{}'", c)).unwrap_or_default(),
                leak.row.map(|r| format!(" row {}", r)).unwrap_or_default(),
                leak.message
            );
        }
        eprintln!("report: {}", out);
    }
    std::process::exit(if clean { EXIT_OK } else { EXIT_LEAKAGE });
}

fn run_import(args: &[String]) -> ! {
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", import_usage());
        std::process::exit(EXIT_OK);
    }
    if args.iter().any(|a| a == "-V" || a == "--version") {
        println!("rcsr {}", VERSION);
        std::process::exit(EXIT_OK);
    }
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut group_rows = store::DEFAULT_GROUP_ROWS;
    let mut time_name: Option<String> = None;
    let mut skip_malformed = false;
    let mut max_memory: Option<u64> = None;
    let mut force = false;
    let mut quiet = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--data" | "-d" => {
                i += 1;
                input = args.get(i).cloned();
                if input.is_none() {
                    eprintln!("error: {} requires a value", args[i - 1]);
                    std::process::exit(EXIT_USAGE);
                }
            }
            "--out" | "-o" => {
                i += 1;
                output = args.get(i).cloned();
                if output.is_none() {
                    eprintln!("error: {} requires a value", args[i - 1]);
                    std::process::exit(EXIT_USAGE);
                }
            }
            "--group-rows" => {
                i += 1;
                match args.get(i).and_then(|v| v.parse::<usize>().ok()) {
                    Some(n) if n >= 1 => group_rows = n,
                    _ => {
                        eprintln!("error: --group-rows needs an integer >= 1");
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--skip-malformed" => skip_malformed = true,
            "--time" => {
                i += 1;
                match args.get(i) {
                    Some(name) => time_name = Some(name.clone()),
                    None => {
                        eprintln!("error: --time requires a column name");
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--max-memory" => {
                i += 1;
                match args.get(i).map(|s| parse_memory_bytes(s)) {
                    Some(Ok(n)) if n > 0 => max_memory = Some(n),
                    _ => {
                        eprintln!("error: --max-memory needs a positive size");
                        std::process::exit(EXIT_USAGE);
                    }
                }
            }
            "--force" => force = true,
            "--quiet" => quiet = true,
            other if other.starts_with('-') => {
                eprintln!("error: unknown import option '{}'\n\n{}", other, import_usage());
                std::process::exit(EXIT_USAGE);
            }
            other => {
                if input.is_none() {
                    input = Some(other.to_string());
                } else if output.is_none() {
                    output = Some(other.to_string());
                } else {
                    eprintln!("error: unexpected argument '{}'", other);
                    std::process::exit(EXIT_USAGE);
                }
            }
        }
        i += 1;
    }
    let input = input.unwrap_or_else(|| {
        eprintln!("error: import needs an input CSV\n\n{}", import_usage());
        std::process::exit(EXIT_USAGE);
    });
    let output = output.unwrap_or_else(|| {
        std::path::Path::new(&input)
            .with_extension("rcsb")
            .to_string_lossy()
            .into_owned()
    });
    if store::is_store_path(&input) {
        eprintln!("error: '{}' is already a column store", input);
        std::process::exit(EXIT_USAGE);
    }
    // #126: shape and guard come from a streaming scan; the table is then read
    // by the streaming parser. The file text is never materialised.
    if let Some(limit) = max_memory.or_else(store::detected_hard_budget) {
        let shape = match data::csv_shape_path(&input) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: {}", e);
                std::process::exit(EXIT_INPUT);
            }
        };
        let estimate = data::estimate_csv_load_stream_bytes(shape);
        if estimate > limit && !force {
            refuse_for_memory(&input, shape, estimate, limit, None);
        }
    }
    let table = match data::read_table_with_time(&input, skip_malformed, time_name.as_deref()) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("error: {}", e);
            std::process::exit(EXIT_INPUT);
        }
    };
    let time_col = match &time_name {
        Some(name) => match table.index_of(name) {
            Ok(c) => Some(c),
            Err(e) => {
                eprintln!("error: --time {}", e);
                std::process::exit(EXIT_USAGE);
            }
        },
        None => None,
    };
    finish_import(&output, &table, group_rows, quiet, time_col);
}

fn finish_import(
    output: &str,
    table: &data::Table,
    group_rows: usize,
    quiet: bool,
    time_col: Option<usize>,
) -> ! {
    if let Err(e) = store::write_store_with_time(output, table, group_rows, time_col) {
        eprintln!("error: {}", e);
        std::process::exit(EXIT_INPUT);
    }
    if !quiet {
        let stored = time_col
            .or_else(|| (0..table.n_cols()).find(|&c| table.ints(c).is_some()))
            .map(|c| format!(", time='{}' as i64", table.names[c]))
            .unwrap_or_default();
        println!(
            "imported {} rows x {} columns to {} (group_rows={}{})",
            table.len(),
            table.n_cols(),
            output,
            group_rows,
            stored
        );
    }
    std::process::exit(EXIT_OK);
}

#[cfg(test)]
mod tests {
    /// #91: a default run must be CPU regardless of build features, so the
    /// same command yields the same archive in a CUDA and a non-CUDA build.
    #[test]
    fn the_gpu_default_follows_the_build_and_flags_are_documented() {
        // #141: a CUDA build prefers the GPU; every other build is CPU.
        assert_eq!(search::Config::default().gpu, cfg!(feature = "cuda"));
        assert!(usage().contains("--gpu"));
        assert!(schema::json().contains("\"--gpu\""));
        assert!(usage().contains("--no-gpu"));
        assert!(schema::json().contains("\"backend\""));
        assert!(usage().contains("not bit-identical"));
    }

    use super::*;

    /// #169: the GPU fit ledger in `--json` must expose admission and a label
    /// for every fallback cause, not just a single opaque counter.
    #[test]
    fn gpu_fit_json_labels_every_fallback_cause() {
        let mut g = search::GpuRunStats::default();
        g.fit_screened = 100;
        g.fit_attempts = 40;
        g.fit_gpu = 37;
        g.fit_fallback = 3;
        g.fit_fallback_by_cause
            .insert("ill_conditioned".to_string(), 2);
        g.fit_fallback_by_cause.insert("non_finite".to_string(), 1);
        g.fit_fallback_improved_by_cause
            .insert("ill_conditioned".to_string(), 1);
        g.fit_status_by_label
            .insert("rank_deficient".to_string(), 4);
        g.fit_refits = 5;
        g.fit_ms = 250.0;
        g.fit_host_ms = 500.0;
        // #255: the fallback time is a subset of host_seconds, and the bound
        // and worker count are reported beside it.
        g.fit_fallback_ms = 125.0;
        g.fit_fallback_bounded = 0;
        g.fit_fallback_workers = 8;
        g.fit_fallback_max = 4096;
        g.fit_polish = rcsr::fit::FitPolish::F32;
        g.ledger.record_fit(0.5, 0.75, 0.125);
        g.ledger.record_fit_polish_split(0.25);
        g.ledger.record_fit_gate(0.375);
        let json = gpu_json(&Some(g), 1.0, 2.0, "gpu", true, 10.0);
        assert!(
            json.contains("\"fit\":{\"strategy\":\"gpu\",\"admission\":\"screened\",\"screened\":100,\"admitted_wide\":0,\"attempts\":40,\"gpu\":37,\"fallback\":3,\"fallback_by_cause\":{\"ill_conditioned\":2,\"non_finite\":1},\"fallback_improved\":1,\"fallback_improved_by_cause\":{\"ill_conditioned\":1},\"status_by_label\":{\"rank_deficient\":4},\"refits\":5,\"refit_candidates\":0,\"refit_s\":0,\"rescore_s\":0,\"seconds\":0.25,\"kernel_lm_s\":0.5,\"kernel_polish_s\":0.25,\"polish\":\"f32\",\"host_seconds\":0.5,\"fallback_seconds\":0.125,\"gate_s\":0.375,\"fallback_bounded\":0,\"fallback_workers\":8,\"fallback_max\":4096,\"fit_rows\":0,\"score_rows\":0,\"fit_twin_bytes\":0,\"subsample_hash\":null}"),
            "{}",
            json
        );
    }

    /// #284: the JSON ledger carries stage windows, the device union (not the
    /// sum), fit leaves and a provenance-carrying duty. A derived field without
    /// its denominator/units/scope is a schema failure.
    #[test]
    fn gpu_json_reports_the_run_ledger_with_provenance() {
        use rcsr::gpu_stats::{RunLedger, Stage};
        let mut l = RunLedger::new();
        l.set_wall(10.0);
        l.record_stage(Stage::Vary, 1.0);
        l.record_stage(Stage::Eval, 3.0);
        l.record_stage(Stage::Fit, 2.0);
        l.record_stage(Stage::Select, 0.5);
        l.record_stage(Stage::Archive, 0.5);
        l.record_stage(Stage::HostOther, 3.0);
        l.record_fit(2.0, 1.5, 0.4);
        l.record_fit_counters(10, 8, 2, [("non_finite".to_string(), 2)].into_iter().collect(), [("non_finite".to_string(), 1)].into_iter().collect());
        l.record_eval_fallback(3, 0.2);
        l.record_interval(Stage::Eval, 0, 1.0, 4.0);
        l.record_interval(Stage::Fit, 1, 3.0, 5.0);
        let mut g = search::GpuRunStats::default();
        g.ledger = l;
        let json = gpu_json(&Some(g), 0.1, 0.2, "gpu", true, 10.0);
        for key in [
            "\"ledger\":",
            "\"stages\":",
            "\"device_busy_s\":4",
            "\"device_busy_sum_s\":5",
            "\"device_overlap_s\":1",
            "\"device_split\":{\"eval_kernel_s\":0,\"eval_launch_s\":0,\"eval_copy_s\":0,\"eval_alloc_s\":0,\"fit_kernel_s\":1.5,\"units\":\"seconds\"}",
            "\"duty_run\":{\"value\":0.4,\"numerator\":4,\"denominator\":10,\"units\":\"fraction\",\"scope\":\"run\"}",
            "\"fallback_by_cause\":{\"non_finite\":2}",
            "\"fallback_improved\":1,\"fallback_improved_by_cause\":{\"non_finite\":1}",
            "\"alloc\":{\"scope\":\"run\",\"units\":\"allocator calls\",\"calls\":0,\"growths\":0,\"steady_alloc_calls\":0,\"batches\":0}",
            "\"units\":\"fraction\"",
            "\"scope\":\"run\"",
            "\"quality_per_gpu_hour\":",
            "\"units\":\"decades_per_gpu_hour\"",
        ] {
            assert!(json.contains(key), "gpu JSON is missing {key}: {}", json);
        }
        assert!(json.contains("\"scope\":\"evaluator_window\""));
        assert!(!json.contains("\"duty_cycle\""));
        assert!(!json.contains("\"generations_per_s\""));
        for key in [
            "\"device_seconds_incl_fit\":1.5",
            "\"device_fraction_of_run\":{\"value\":0.15,\"numerator\":1.5,\"denominator\":10,\"units\":\"fraction\",\"scope\":\"process\"}",
            "\"eval_fraction_of_run\":{\"value\":0,",
            "\"process_total_s\":10",
        ] {
            assert!(json.contains(key), "gpu JSON is missing {key}: {}", json);
        }
        // The schema and help must name the run-scoped fields, not just the
        // evaluator-window ones.
        for needle in ["device_busy_s", "duty_run", "fallback_by_cause", "run_ledger"] {
            assert!(
                schema::json().contains(needle),
                "schema is missing the run ledger field {needle}"
            );
        }
        assert!(usage().contains("device-busy"));
        assert!(usage().contains("numerator/denominator/units/scope"));
    }

    /// #256/#284: the baseline is a mean (MSE) because the best error is a
    /// mean. The published 10 h cell must give 4.844205, never 3085.48.
    #[test]
    fn gpu_quality_baseline_is_the_same_unit_as_the_best_error() {
        let train = rcsr::data::Sample {
            x: vec![vec![0.0], vec![1.0]],
            y: vec![0.0, 2.0],
            w: Vec::new(),
            ys: Vec::new(),
            pairs: None,
        };
        // SSE = 2, n = 2 -> MSE = 1.
        assert!((gpu_baseline_error(&train) - 1.0).abs() < 1e-12);
        let mut st = rcsr::gpu_stats::RunLedger::new();
        st.set_wall(7.467_618_67);
        let q = st.quality_per_gpu_hour(0.103_464_003_567_670_26, 0.105_885_819_341_126_06);
        assert!((q.value - 4.844_205).abs() < 1e-5);
        assert!((q.value - 3085.48).abs() > 1.0);
    }

    /// #256: the published 10 h cell, end to end through the JSON writer. The
    /// evaluator window gives `4.844205` decades per hour (best MSE
    /// 0.10346400356767026 against the constant-mean MSE 0.10588581934112606
    /// over 7.46761867 s), never `3085.48`.
    #[test]
    fn gpu_json_reports_the_published_cell_with_like_units() {
        let mut g = search::GpuRunStats::default();
        g.accounting.wall_s = 7.467_618_67;
        g.accounting.kernel_s = 7.0;
        g.accounting.batches = 62;
        let json = gpu_json(
            &Some(g),
            0.103_464_003_567_670_26,
            0.105_885_819_341_126_06,
            "exact",
            true,
            38_114.998,
        );
        assert!(!json.contains("3085.48"), "{}", json);
        assert!(json.contains("\"quality_per_gpu_hour\":4.8442"), "{}", json);
        // Batches per evaluator second, under its true name.
        assert!(json.contains("\"batches_per_s\":8.302"), "{}", json);
        assert!(!json.contains("\"generations_per_s\""), "{}", json);
    }

    /// #256: the run-level device fractions, from the 10 h run's own numbers:
    /// 596.599 s of fit kernels plus the 7.468 s evaluator window is 604.067 s
    /// of device work, 1.585 % of the 38 114.998 s process.
    #[test]
    fn gpu_json_device_fractions_use_the_process_runtime() {
        use rcsr::gpu_stats::{RunLedger, Stage};
        let mut l = RunLedger::new();
        l.set_wall(37_870.245);
        l.record_stage(Stage::Fit, 36_959.0);
        l.record_fit(36_959.0, 596.599, 100.0);
        let mut g = search::GpuRunStats::default();
        g.accounting.wall_s = 7.468;
        g.ledger = l;
        let json = gpu_json(&Some(g), 0.1, 0.2, "gpu", true, 38_114.998);
        let key = "\"device_fraction_of_run\":{\"value\":";
        let at = json.find(key).expect("device_fraction_of_run") + key.len();
        let value: f64 = json[at..]
            .split(',')
            .next()
            .and_then(|v| v.parse().ok())
            .expect("a number");
        assert!((value - 604.067 / 38_114.998).abs() < 1e-9, "{}", value);
        assert!((value - 0.01585).abs() < 1e-5, "{}", value);
        assert!(json.contains("\"device_seconds_incl_fit\":604.06"), "{}", json);
        assert!(json.contains("\"process_total_s\":38114.998"), "{}", json);
    }

    /// #256: the phases and the unaccounted remainder add up to the runtime.
    #[test]
    fn timing_phases_and_remainder_add_up_to_the_runtime() {
        let t = PhaseTimings {
            prep: 10.099,
            search: 37_870.245,
            baselines: 213.711,
            diagnostics: 20.943,
            report: 0.0,
            total: 38_114.998,
            gpu: 7.468,
        };
        let sum = t.phases_sum() + t.unaccounted();
        assert!((sum - t.total).abs() < 1e-9);
        assert!(t.json().contains("\"unaccounted_seconds\":"));
    }

    /// #71: the guard and the import command must be discoverable.
    #[test]
    fn help_and_schema_document_the_memory_guard_and_import() {
        for flag in ["--max-memory", "--force"] {
            assert!(usage().contains(flag), "usage() is missing {flag}");
            assert!(
                schema::json().contains(&format!("\"{}\"", flag)),
                "schema::json() is missing {flag}"
            );
        }
        assert!(usage().contains("rcsr import"));
        assert!(import_usage().contains("--group-rows"));
        assert_eq!(parse_memory_bytes("512MB").unwrap(), 512_000_000);
        assert_eq!(parse_memory_bytes("2GiB").unwrap(), 2 * 1_073_741_824);
        assert_eq!(parse_memory_bytes("4096").unwrap(), 4096);
        assert!(parse_memory_bytes("12 quatloos").is_err());
    }

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

    /// #211: the development log is a documented surface: every flag, the
    /// schema id, the four event names and the analyzer are in `--help` and in
    /// `rcsr schema`.
    #[test]
    fn help_and_schema_document_the_development_log() {
        let help = usage();
        let schema = schema::json();
        for flag in [
            "--dev-log",
            "--dev-log-level",
            "--dev-log-every",
            "--dev-log-max-bytes",
            "--no-dev-log",
        ] {
            assert!(help.contains(flag), "usage() is missing {flag}");
            assert!(schema.contains(flag), "schema::json() is missing {flag}");
        }
        for line in [
            "rcsr dev-report <log.ndjson>",
            "DEVELOPMENT LOG (#211)",
            "on by default",
            "rcsr.dev.ndjson",
            "summary",
            "full",
            "rcsr-devlog/1",
            "docs/dev-log.md",
        ] {
            assert!(help.contains(line), "usage() is missing '{line}'");
        }
        for field in [
            "rcsr-devlog/1",
            "run_start",
            "generation",
            "checkpoint",
            "run_end",
            "dev-report",
            "on (summary level)",
            "--no-dev-log",
        ] {
            assert!(schema.contains(field), "schema::json() is missing '{field}'");
        }
        // The analyzer's own usage names both jobs it does.
        let report_usage = rcsr::devreport::usage();
        assert!(report_usage.contains("bottleneck"));
        assert!(report_usage.contains("<log>"));
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

    /// #115: the time-series checklist must be reachable from the CLI contract,
    /// and the smoothing flags must describe the causal default correctly.
    #[test]
    fn help_and_schema_link_the_time_series_checklist() {
        for surface in [usage(), schema::json()] {
            assert!(
                surface.contains("docs/time-series.md"),
                "the time-series checklist is not linked from the CLI contract"
            );
        }
        assert!(usage().contains("--smooth-centred"));
        assert!(schema::json().contains("--smooth-centred"));
        assert!(
            schema::json().contains("trailing moving-average"),
            "schema must describe --smooth as trailing"
        );
        assert!(
            !schema::json().contains("centred moving average per column"),
            "the stale README wording must not reappear in schema"
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
            "--skip-malformed",
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

    /// Dataset aggregates must be discoverable from `schema`, named in
    /// `--ops`, and listed in the help text.
    #[test]
    fn help_and_schema_list_dataset_aggregates() {
        let schema_text = schema::json();
        for name in [
            "mean", "std", "median", "iqr", "sum", "amin", "amax", "count",
        ] {
            assert!(
                schema_text.contains(&format!("\"{name}\"")),
                "schema::json() is missing operator '{name}'"
            );
            assert!(usage().contains(name), "usage() is missing '{name}'");
        }
        assert!(usage().contains("aggregates:"));
        assert!(Op::parse_list_checked("mean,count").is_ok());
    }

    /// #134: the phase ledger exposes non-overlapping windows and a separate
    /// measured total, so a report can never silently quote search time as wall.
    #[test]
    fn phase_timings_sum_to_the_measured_total() {
        let t = PhaseTimings {
            prep: 0.10,
            search: 0.20,
            baselines: 0.30,
            diagnostics: 0.05,
            report: 0.02,
            total: 0.67,
            gpu: 0.12,
        };
        assert!((t.phases_sum() - t.total).abs() <= 0.01 * t.total);
        let json = t.json();
        for key in [
            "prep_seconds",
            "search_seconds",
            "baselines_seconds",
            "diagnostics_seconds",
            "report_seconds",
            "total_seconds",
            "gpu_seconds",
        ] {
            assert!(json.contains(key), "timing JSON is missing {key}");
        }
        assert!(t.text().contains("total=0.67s"));
        assert!(t.text().contains("gpu=0.12s"), "GPU wall is a subset, not a phase");
    }
}
