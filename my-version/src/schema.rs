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
        ("--data", "string", "none", "table path, a .rcsb column store made by `rcsr import`, or '-' for stdin"),
        ("--max-memory", "size", "auto", "refuse when the estimated resident footprint exceeds this (bytes, or KiB/MiB/GiB/GB). A CSV is estimated from rows x columns (text + table + preparation) once the header and row count are known, and exits 3 before parsing. A .rcsb store that does not fit is searched from a bounded hash sample"),
        ("--force", "flag", "false", "override --max-memory and attempt the resident path anyway"),
        ("--target", "csv", "none", "target column name; a comma-separated list (y1,y2) enables multi-output regression with one shared shape and per-output constants"),
        ("--vars", "csv", "all-except-targets", "input column names"),
        ("--dtype", "csv", "all-continuous", "per-variable datatype, col=continuous|boolean|categorical (categorical expands to one-hot boolean columns named col=value)"),
        ("--max-levels", "int", "16", "maximum distinct values in a categorical column"),
        ("--skip-malformed", "flag", "false", "drop data rows whose field count differs from the header and report the count; default refuses the file with an input error naming the 1-based line"),
        ("--test-frac", "float", "0.25", "held-out fraction"),
        ("--validation-frac", "float", "0", "fraction of the chronological training period held out as a validation slice for selection (0 = off); the test block is scoring-only"),
        ("--select-by", "enum", "validation-if-present", "candidate ranking criterion: train | validation | aicc; test_error is never a criterion"),
        ("--weight", "string", "none", "per-row weight column (finite, >= 0, at least one positive; supported metrics: absolute, squared, correlation, signed, log_squashed, implicit, implicit_residual, aic, aicc, bic)"),
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
        ("--time", "string", "none", "time/ordering column (implies temporal); epoch numbers or ISO 8601 with Z/+-HH:MM offsets"),
        ("--date-only", "enum", "error", "date-only values: error | utc-midnight | session-close (21:00 UTC)"),
        ("--time-unit", "enum", "auto", "numeric time unit: s | ms | us | ns (default auto-detect; mixed/out-of-range refuses)"),
        ("--tz", "string", "UTC", "fixed UTC offset for offset-less timestamps: UTC or +-HH:MM; named zones are refused"),
        ("--group-by", "string", "none", "group column (implies group)"),
        ("--gap", "int", "0", "embargo rows between train and test"),
        ("--folds", "int", "1", "contiguous test blocks to report"),
        ("--lags", "int", "0", "delayed copies per input (time-delay embedding)"),
        ("--history-frac", "float", "0", "fraction of each series kept as history"),
        ("--gap-break", "float", "0", "new series when dt exceeds f x median dt"),
        ("--null-runs", "int", "0", "y-randomisation control runs (shuffled targets)"),
        ("--null-block", "int", "0", "block-permutation null: contiguous block length in rows (>= label horizon; 0 = i.i.d. shuffle). Implies 19 runs when --null-runs is absent"),
        // #69: reference set always exact; only query rows are sampled, and
        // only for mean-loss metrics.
        ("--baseline-samples", "int", "0", "k-NN null baseline query rows to score (0 = every row; absolute/squared/log_squashed only). The reference set is always all training rows; sampled results report train/test SEM with the finite-population correction"),
        ("--bootstrap-error", "int", "0", "bootstrap resamples for the .632+ error estimate (0 = off); above n^0.7 >= 1000 rows the draw is a BLB subsample of b = n^0.7 rows and the resample size is reported"),
        ("--cv", "string", "none", "cross-validation scheme: k-fold | rolling | purged"),
        ("--cv-k", "int", "5", "number of cross-validation folds"),
        ("--stability", "int", "0", "bootstrap resamples for stability selection (0 = off); above n^0.7 >= 1000 rows each resample searches a BLB subsample of b = n^0.7 rows and the resample size is reported"),
        ("--metric", "enum", "absolute", "absolute | squared | correlation | correlation_signed | median | maximum | signed | log_squashed | rank_correlation | rank_correlation_signed | interquartile | slope | implicit | implicit_residual | auc | log_loss | hinge | aic | aicc | bic. correlation/rank_correlation minimise 1-|r| and accept an inverted model; the _signed variants minimise 1-r and require the right direction. auc/log_loss/hinge read the target as two classes and reject more than two distinct values"),
        ("--implicit", "flag", "false", "fit f(inputs, target) = 0: the target becomes an extra (normalised) input and the fit target is 0"),
        ("--implicit-derivative", "flag", "false", "shorthand for --implicit --metric implicit: score the level-set distance |f| / ||grad f||"),
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
        ("--units", "csv", "none", "declared dimensions name=dim (repeatable or comma-separated), e.g. x=m,t=s,y=m/s; SI names m,s,kg,A,K,mol,cd plus any other identifier as its own base dimension (price=USD); no unit table or scale factors; *,/,^integer,(); undeclared names are dimensionless; alone it reports constant dimensions and never rejects"),
        ("--require-dimensions", "flag", "false", "reject candidates whose dimensions cannot be made consistent with the target's (infinite error; requires --units, a usage error on its own; without this flag units only report and never reject)"),
        ("--eval-expr", "string", "none", "score this formula on the split instead of searching"),
        ("--function", "string", "none", "user-defined building block name(a,b)=expr or name=expr (repeatable, at most four; parameters inferred from free variables; constants in a body are shared by every call)"),
        ("--fit-iters", "int", "8", "LM constant-fitting iterations; 0 disables fitting entirely and leaves constants as generated"),
        ("--no-fit", "flag", "false", "disable constant fitting entirely (same as --fit-iters 0)"),
        ("--no-dedup", "flag", "false", "disable structural de-duplication"),
        ("--no-front-parents", "flag", "false", "draw every parent from the island tournament; disables the 15% Pareto-front feedback and restores the pre-#76 trajectory exactly"),
        ("--no-simd", "flag", "false", "force the scalar tree walk instead of the SoA/AVX evaluator (same results; for debugging and A/B runs)"),
        ("--predictor-samples", "int", "none", "training rows per generation for fitness and constant fitting (>=2; all rows when >= train size)"),
        // #85: fit-only screening; survivors are refit on the full training sample.
        ("--fit-samples", "int", "0", "training rows per generation used to screen constants (0 = off, >=2); all offspring are fitted/scored on the subset, the top 1/16 are refit and scored on all training rows before selection and archiving"),
        ("--patience", "int", "40", "stop after N gens without improvement"),
        ("--time-budget", "float", "none", "wall-clock seconds"),
        ("--gpu", "flag", "false", "assert that the GPU must guide the search (exit 2 when built without --features cuda or no device is present); without --gpu/--cpu a CUDA build with a usable device selects the GPU automatically. Reported errors stay CPU f64, but f32 fitness can change the trajectory, so cross-backend runs are not bit-identical"),
        ("--no-gpu", "flag", "false", "force CPU evaluation (alias --cpu); the reproducibility escape hatch. A non-CUDA build is CPU by default"),
        ("--save", "string", "none", "write a checkpoint on exit and every --save-every generations; the write is atomic and the previous generation is kept as <file>.prev"),
        ("--save-every", "int", "25", "checkpoint interval in generations (requires --save)"),
        ("--resume", "string", "none", "resume a checkpoint; saved search settings win, only --generations may be raised, data and backend must match"),
        ("--allow-backend-change", "flag", "false", "permit --resume across cpu<->gpu (recorded on stderr and in --run-manifest)"),
        // #140: run integrity.
        ("--run-manifest", "string", "none", "atomically write a JSON manifest (commit, dirty, binary SHA-256, data checksum/fingerprint, resolved config, seed, backend, device/driver/CUDA, UTC start/end, phase timings); updated atomically at the end"),
        ("--json-out", "string", "none", "write the final JSON result to this file atomically instead of stdout (implies --json); progress stays on stderr/NDJSON"),
        ("--format", "enum", "text", "text | json | ndjson"),
        ("--progress", "flag", "false", "emit per-generation progress with convergence statistics (confidence, stability, maturity, percent_converged); see docs/convergence.md"),
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
         \"convergence\":{{\"window\":20,\"confidence\":\"exp(-max(0,(e[g-w]-e[g])/(1+|e[g-w]|))/1e-3)\",\"stability\":\"mean Jaccard overlap of the front's structural forms over the last w comparisons\",\"maturity\":\"(g - first_generation_of_oldest_surviving_form) / max(g,1)\",\"percent_converged\":\"0.5*confidence + 0.3*stability + 0.2*maturity\",\"converged_threshold\":0.75,\"doc\":\"docs/convergence.md\"}},\n  \
         \"backend\":{{\"default\":\"gpu when built with --features cuda and a usable device is probed at startup; cpu otherwise\",\"device_probe\":\"once at startup\",\"override\":\"--no-gpu/--cpu forces cpu; --gpu asserts a device and exits 2 when unavailable\",\"reporting\":\"backend, backend_reason and device are recorded from the probe/outcome, never inferred from flags\",\"exactness\":\"cross-backend runs are not bit-identical; reported metrics and constants are rebuilt from exact CPU f64 and --cpu is the reproducible escape hatch\",\"doc\":\"docs/gpu-backend.md\"}},,\n  \
         \"walkforward\":{{\"command\":\"rcsr walkforward\",\"summary\":\"walk-forward evaluation/backtest of one expression over a .rcsb column store or CSV\",\"doc\":\"docs/walkforward.md\",\"options\":[\
{{\"name\":\"--data\",\"type\":\"string\",\"default\":\"none\",\"description\":\"column store (.rcsb) or CSV table\"}},\
{{\"name\":\"--target\",\"type\":\"string\",\"default\":\"none\",\"description\":\"forward-return column used for P&L\"}},\
{{\"name\":\"--expr\",\"type\":\"string\",\"default\":\"none\",\"description\":\"score expression; variables resolve by column name\"}},\
{{\"name\":\"--vars\",\"type\":\"csv\",\"default\":\"all numeric except target/time/group\",\"description\":\"feature columns\"}},\
{{\"name\":\"--time\",\"type\":\"string\",\"default\":\"none\",\"description\":\"time/ordering column\"}},\
{{\"name\":\"--group-by\",\"type\":\"string\",\"default\":\"none\",\"description\":\"panel group column; costs and positions are per group\"}},\
{{\"name\":\"--window\",\"type\":\"string\",\"default\":\"none\",\"description\":\"LABEL,TRAIN_START,TRAIN_END,TEST_START,TEST_END (repeatable)\"}},\
{{\"name\":\"--windows\",\"type\":\"string\",\"default\":\"none\",\"description\":\"CSV file of window rows\"}},\
{{\"name\":\"--horizon\",\"type\":\"float\",\"default\":\"0\",\"description\":\"forward-label horizon in seconds; minimum purge/embargo\"}},\
{{\"name\":\"--embargo\",\"type\":\"float\",\"default\":\"horizon\",\"description\":\"extra gap between train_end and test_start\"}},\
{{\"name\":\"--cost-bps\",\"type\":\"float\",\"default\":\"10\",\"description\":\"cost per side in basis points\"}},\
{{\"name\":\"--threshold-quantile\",\"type\":\"float\",\"default\":\"0.7\",\"description\":\"training/validation score quantile\"}},\
{{\"name\":\"--validation-frac\",\"type\":\"float\",\"default\":\"0\",\"description\":\"trailing training fraction used to set the threshold\"}},\
{{\"name\":\"--periods-per-year\",\"type\":\"float\",\"default\":\"inferred\",\"description\":\"annualisation override\"}},\
{{\"name\":\"--json\",\"type\":\"flag\",\"default\":\"false\",\"description\":\"print the JSON report\"}},\
{{\"name\":\"--out\",\"type\":\"string\",\"default\":\"none\",\"description\":\"write the JSON report to a path\"}},\
{{\"name\":\"--markdown\",\"type\":\"string\",\"default\":\"none\",\"description\":\"write the Markdown report ('-' = stdout)\"}}]}},\n  \
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

