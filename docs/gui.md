# Designing a GUI for `rcsr`

*A first-principles design for a graphical front end to the `rcsr`
symbolic-regression engine.*

`rcsr` is a command-line, JSON-first tool. Everything the engine knows how to do
is reachable from `rcsr --help`, discoverable from `rcsr schema`, and reported
in a machine-readable shape from `--json` (one object per run) or `--ndjson`
(one object per line while a run streams). This document designs a GUI **on top
of that contract**, not around it. The command line is the source of truth; the
GUI is a second, friendlier view of the same interface.

The document is deliberately concrete: each pane names the engine output that
feeds it, each interaction names the flag it sets, and the architecture states
how a new flag reaches the GUI without new GUI code.

---

## As built — MVP from issue #38

This section records what the first implementation actually ships, so the design
above stays the target while the present state is unambiguous.

### How to run it

```sh
cd my-version
cargo build --release
./target/release/rcsr_serve --port 8080 --open
# open the printed URL (default http://127.0.0.1:8080)
```

`rcsr_serve` accepts `--host` (default `127.0.0.1`), `--port` (default `8080`;
`0` asks the OS for a free port) and `--open` (prints the URL; it never launches
a browser). The binary has no dependencies beyond the Rust standard library, and
the server refuses anything but the address it was bound to.

### What exists

* **Data pane** — open a local CSV/TSV/text file or paste a table; the page
  parses it locally only to render a preview. Each column gets a role selector:
  input, target, time, group, weight, ignore. Roles build `--vars`,
  `--target`, `--time`, `--group-by` and `--weight`; the target, time, group and
  weight roles are unique.
* **Search pane** — the form is generated from `GET /api/schema`. Metric,
  generations, population, islands, maximum complexity and seed are pinned at
  the top; every other option the schema declares is rendered under **More
  options**. The widget is chosen from the declared `type` (`enum` → select,
  `int`/`float` → number, `flag` → checkbox, `string`/`csv` → text). There is
  no hand-written list for rendering; execution support is declared separately
  and tested for every flag.
* **Run pane** — `POST /api/run` streams `application/x-ndjson`; the page
  parses one event per line and updates the progress line and a canvas Pareto
  chart (x = complexity, y = train and test error on a log axis). Solution
  events append points as they arrive; baseline events draw reference rules.
* **Solutions pane** — the front table (rank, complexity, `k_constants`, train
  and test error, AICc, expression) and a detail panel for the selected row:
  formula, constants ± standard errors, overfit index, AICc/BIC,
  Durbin–Watson, Ljung–Box and Breusch–Pagan p-values, skewness/kurtosis,
  baselines, `.632+` bootstrap (when requested), cross-validation (when
  requested), stability frequency and the null-control verdict.
* **Copy as CLI** — emits the exact `rcsr` command for the current roles and
  form values. When the table was pasted, the command carries it as a
  here-document so it reproduces the run headlessly.

### Endpoints and stream contract

| endpoint | response |
|---|---|
| `GET /` | the embedded single-page app (HTML, relative `app.js`/`style.css`) |
| `GET /api/schema` | the same document as `rcsr schema` |
| `GET /api/example` | a 20-row bundled CSV |
| `POST /api/run` | `application/x-ndjson` |

`POST /api/run` takes `{"args":["--data","-","--target","y",…],"table":"x,y\n…"}`.
`args` is a CLI argv without the program name; `--data -` reads the request's
`table` string, while `--data <path>` reads that local file. Errors in the
request or the arguments return HTTP 400 with a JSON `{"error": …}` object;
failures after the stream starts arrive as `{"event":"error","code":3,"message":…}`.

### Locked test

The server resolves the locked test with the same resolver and the same split and cross-validation checks as the CLI: `--lockbox START,END|none`, else the range declared next to a `--data <path>` file (`.lockbox.json`, schema or manifest), else the protocol default 2025-07-01,2026-07-01 (always the case for an uploaded table, where a flag adds to the default). A train, validation, test or cross-validation row inside the range is refused with HTTP 400 and the CLI message. With a declared range and no `--time`, the one epoch-seconds integer column is used. The `start` event carries the `lockbox` object (declared range, source, declared_in, rows_in_scope, evaluated, checked).

The stream uses the `--ndjson` event names and payloads: `start`, `progress`,
`solution`, `note`, `preparation`, `null_control`, `baselines`,
`cross_validation`, `residual_tests`, `stability`, `done` (and `eval` for
`--eval-expr`). The `solution` event additionally carries `bic`, which the CLI's
`--json` object reports but its NDJSON event did not; this is additive and does
not change existing field meaning. HTTP status replaces the CLI exit-code
channel: a completed run is HTTP 200 and ends with `done`, including a front
with no finite member.

### Deliberate deviation from the design above

The design's §6.6 and the original issue text describe driving `rcsr` as a child
process. The implementation instead runs the **same library pipeline
in-process** (`data::parse_table` → `data::prepare_full_weighted_with` →
`search::run`), because this build is a single local HTTP response and adding a
process/pipe layer would only duplicate the streaming code. It still consumes
the JSON/NDJSON contract and never parses human-readable text. The CLI remains
the canonical interface: the server is a second front end over the same
functions, and the page can hand back the equivalent command.

### Deferred (not in this MVP)

* Project save/load, multiple datasets and multi-search front overlays.
* A dedicated Prepare pane, split stripe, series preview and per-fold strip;
  the split, lags, history and gap flags are still settable through the
  schema-generated form.
* The observed-vs-predicted chart, the four-panel residual gallery and the
  overfitting-wall chart. The numbers behind them are present; only the Pareto
  chart is drawn.
* Front/JSON export, the Markdown report and TSV output. The server rejects
  `--out`, `--save*`, `--resume` and `--report*` with a clear message instead of
  writing files on the server.
* Keyboard shortcuts beyond native tab order, a project tree, and a
  paste-a-command parser.
* Checkpoint streaming events and resume. The CLI supports them; the web MVP
  does not.
* Multi-run comparison, seed-stability views and rolling-origin charts.

### Headless test strategy

`my-version/tests/gui.rs` starts the real server on port 0, speaks HTTP/1.1 over
a socket and exercises the real endpoints:

1. `/api/schema` returns 200 and valid JSON with `options`, `operators` and
   `exit_codes`.
2. `/api/schema` is compared byte-for-byte with `rcsr::schema::json()`, and a
   behavioural test feeds every documented root option to the GUI parser. Each
   one must be either implemented or appear in the explicit unsupported list —
   never "unknown argument".
3. `POST /api/run` over a 20-row table returns lines that all parse as JSON and
   end with `done`.
4. Malformed HTTP and malformed JSON both return 400 without hanging.
5. `GET /` contains the app title and the page's JS reads the schema.
6. The three embedded assets contain no `http://` or `https://` string.
7. Dropping the server handle stops the listener.
8. A sidecar-declared range, a `--lockbox` flag range and the protocol default refuse an overlapping split or cross-validation request with 400; `--exclude` clearing the range lets cross-validation run; the `start` event carries the `lockbox` object.

There is **no screenshot test**: no browser or display was used in this build.
The visual layer is documented and checked by these headless contract tests, not
by an image comparison.

---

## 1. Principles

### 1.1 The formula is the hero

A symbolic-regression tool exists to produce an equation a human can read, check
and reuse. Every other pixel is in service of that. The equation is rendered in
one place (the solution detail), it is copyable as plain text, it is what the
"export" and "copy" actions emit, and it is never truncated behind a tooltip
when it matters. Charts and tables are scaffolding around it.

Concretely: the solution list, the detail panel, the report, the clipboard and
the exported file all present the **same** `expression` string that
`rcsr --json` emits for that Pareto row — never a re-rendered approximation.

### 1.2 Never show a fit without its diagnostics

A small error is not evidence; the engine's whole point is that a symbolic
search can fit noise. So the GUI refuses to present a number in isolation:

* a solution's error is always shown next to its **complexity**, its
  **`k_constants`** and its **`overfit_index`**;
* the **test error** is shown next to the **train error**, never alone;
* the front is shown against the **null baselines** (`mean`, `linear`, `knn`)
  so a fit that does not beat a straight line is visibly pointless;
* the detail panel always carries the residual checks
  (`durbin_watson`, `ljung_box`, `breusch_pagan`, `skewness`, `kurtosis`) and,
  when requested, the `null_control` verdict and the `.632+` bootstrap
  (`apparent`, `oob`, `estimate`).

There is no "headline number" view. Diagnostics are not an optional tab; they
arrive with the result because the engine already computed them.

### 1.3 Every action has a CLI equivalent

The GUI is not a separate program with its own semantics. Every button, menu
item and drag writes a concrete `rcsr` invocation, and every one can be turned
back into a command:

* **Copy as CLI** on any search yields the exact argv that reproduces it
  (deterministic given `--seed`).
* Any pane can be **paste-a-command** driven: type or paste a `rcsr ...` line
  and the GUI parses it (via `rcsr schema`) into the form.
* The GUI never invents an option the CLI does not have, and never hides an
  option the CLI does. If `rcsr schema` lists it, the GUI can set it.

### 1.4 Charts earn their space

A chart is added only when it answers a question a number cannot. Four answer
real questions; the rest do not belong in v1:

1. **Pareto front** — where is the knee of error versus complexity?
2. **Observed vs predicted** — where does the model miss?
3. **Residuals** (vs fitted, vs row order, histogram, QQ) — is anything left in
   the error?
4. **Overfitting wall** (train vs held-out error against complexity) — at what
   complexity does the fit stop generalising?

Everything else (donuts, gauges, three-dimensional scatter) is refused.

### 1.5 Honest defaults

`--split` defaults to `random`, which is right only for exchangeable rows. The
GUI must make the choice visible and warn like the CLI does. When the engine
emits the non-exchangeable warning (the `notes` array or the stderr warning),
the Search pane surfaces it as a banner, and the Data pane offers a one-click
`--split auto`. The GUI must never quietly "help" by changing a split the user
chose.

---

## 2. Information architecture

### 2.1 The project

A **project** is the unit of saved work and is exactly the engine's mental
model:

```
project
├── dataset            (one table: a path or inline rows)
│   ├── column roles   (input / target / time / group / weight / ignore)
│   └── searches
│       ├── search     (the problem: --target/--vars/--metric/--ops/…/--seed)
│       ├── runs       (each `rcsr` invocation: seed, argv, exit code, timing)
│       └── results    (the parsed --json object: split, pareto, diagnostics)
└── dataset …
```

* A dataset maps to `--data` plus a role per column. Roles map to flags:
  `input → --vars`, `target → --target`, `time → --time`, `group → --group-by`,
  `weight → --weight` (once #1 lands), `ignore → ` (excluded from `--vars`).
* A search maps one-to-one to a set of flag values, i.e. to an argv.
* A run maps to one process execution and its captured `--json`/`--ndjson`.
* A result maps to the JSON object in §6.1.

The same dataset can carry many searches (one per hypothesis, metric or split),
which is the point of a project: a notebook of competing laws over one table.

### 2.2 What is persisted

A project file is **JSON** (human-readable, diffable) containing the datasets'
roles, inline-or-referenced data, each search's argv template, and the captured
results. Nothing is persisted that the CLI cannot reconstruct: on load, the GUI
can always re-derive the run by replaying the argv. A project is therefore
portable and reviewable, and a corrupt result can be recreated rather than
guessed.

### 2.3 Windows and modals

* **One main window**, with a docked project tree on the left and a tabbed
  workspace on the right. No MDI, no floating panels by default.
* **Dialogs** only for genuinely modal decisions: import options, "this run will
  discard unsaved results", settings. Everything else lives in a pane.

---

## 3. Panes

The workspace tabs, in the order a user visits them, are **Data · Prepare ·
Search · Run · Solutions · Diagnostics**. The first four are inputs, the last
two are outputs.

### 3.1 Data

**Job:** enter or import a table and say what each column is for.

* A spreadsheet grid (CSV/TSV/whitespace, matching `data::parse_table`). Import
  via a file picker or by pasting a block; `--data -` reads stdin, and the pane
  supports "Paste table" for the same effect.
* Each column header carries a **role selector**: *input*, *target*, *time*,
  *group*, *weight* (when available), *ignore*. The default is the CLI default:
  every numeric column is an input except the target and any time/group column.
  The role selection is written straight into the argv:
  `--target <col>`, `--vars <comma list>`, `--time <col>`, `--group-by <col>`.
* **Split stripe.** A thin gutter down the row numbers colours each row by the
  role the *current split* assigns it: training, held-out, embargo (`--gap`),
  history (`--history-frac`), or dropped. The stripe is computed by re-running
  the splitter (the same code path as `--split`) on the in-memory table, so the
  user sees — before searching — exactly which rows the model will see. This is
  the single most important addition over a plain grid.
* Row/column operations (insert, delete, clear) edit the in-memory table only;
  "Export data" writes it back out. Editing marks the dataset dirty.
* Validation is live and matches the engine's exit code 3 conditions: a
  non-numeric target column, no inputs, an empty training or held-out side are
  reported in place, not at run time.

**Feeds:** `--data`, `--target`, `--vars`, `--time`, `--group-by`.
**Consumes:** the `split` object of a previous run, to draw the stripe once a
run exists.

### 3.2 Prepare

**Job:** per-column preparation and the time-series embedding.

The engine today handles missing values and non-finite cells by rejecting the
table (`data.rs`), normalises using **training rows only**, and offers
`--lags`, `--history-frac`, `--gap-break`. The pane exposes exactly that, and
labels the rest as planned:

* **Per-column**: a read-only summary (type, count, distinct, min/max, mean/σ,
  missing count) and, when #2 lands, the preparation controls (missing-value
  method, outlier removal, smoothing, normalise). Until then the pane tells the
  user in plain words that missing values must be fixed *before* `rcsr` can
  read the table, and offers "drop rows with missing values" / "drop column".
* **Time-series block** (shown when a *time* role exists):
  * `--lags K` — number of delayed copies appended per input.
  * `--history-frac f` — share of each series reserved as history (kept, not
    scored).
  * `--gap-break f` — series break when the time step exceeds `f ×` the median.
  * A **series preview**: for the selected group, plot the target against
    `--time` with the resulting lag columns ghosted, so the effect of `K` and
    `f` is visible.
* **Normalisation is stated, not offered**: the engine always standardises using
  training rows only, and the pane says so, because that is a correctness
  property, not a preference.

**Feeds:** `--lags`, `--history-frac`, `--gap-break`, and (planned) the #2
preparation flags. **Consumes:** the `split` object (`lags`, `series`,
`history_rows`) to show what the last run actually did.

### 3.3 Search

**Job:** the whole problem statement, generated from `rcsr schema`.

The form is built by walking the `options` array of `rcsr schema`. The mapping
from the declared `type` to a widget is mechanical:

| `type` | widget |
|---|---|
| `enum` | combo box; the choices are parsed from the `description` (`a | b | c`) |
| `csv` | token/chip field; for `--ops` the chips come from the `operators` array |
| `int` | spin box |
| `float` | spin box (or slider where a sane range exists, e.g. `--test-frac`) |
| `flag` | check box (present = set) |
| `string` | text field, with a file/path picker for `--data` and `--out` |

The schema carries no grouping metadata, so the GUI keeps a small, **optional**
layout table that places known options into labelled groups (Data, Objective,
Operators, Complexity, Splitting, History, Controls, Diagnostics, Output) and
renders **every unplaced option in an "Other" group automatically**. The
consequence, which is the point: *adding a flag to the CLI makes it appear in
the GUI, in some group, with no GUI code.* A GUI release may lag the binary; it
will still expose the new flag.

Groups:

* **Data / roles** — mirrors the Data pane (`--data`, `--target`, `--vars`,
  `--time`, `--group-by`).
* **Splitting** — `--split`, `--test-frac`, `--gap`, `--folds`; a live sentence:
  "75% of 400 rows train (300 rows), 5-row embargo, 95 test rows in 1 block."
* **History** — `--lags`, `--history-frac`, `--gap-break`.
* **Objective** — `--metric` (16 choices), each with its one-line meaning.
* **Operators** — `--ops`, a chip field seeded from the schema `operators`
  array; presets ("polynomial", "trig", "smooth", "all") are just named subsets.
* **Complexity** — `--max-complexity`, and notice of the planned per-operator
  cost table (#13).
* **Search controls** — `--generations`, `--islands`, `--pop`, `--seed`,
  `--patience`, `--time-budget`, `--fit-iters`, `--no-fit`, `--no-dedup`,
  `--eval-expr`.
* **Evaluation** — `--null-runs`, `--bootstrap-error`, and `--folds` again
  (this is where a user decides how hard to test a result).
* **Output** — `--top`, `--out`, format.

A persistent **CLI bar** along the bottom of the pane shows the argv the current
form values produce, live, with a **Copy** and a **Run** button. The form and
the command line can never disagree because the argv is the form's serialisation.

**Feeds:** every search flag. **Consumes:** `rcsr schema`.

### 3.4 Run (live progress)

**Job:** start, watch, pause-by-stopping a bounded run, and keep the log.

The engine has no pause primitive; it has `--time-budget`, `--generations`,
`--patience` and `--seed`. The GUI therefore models a run as **bounded and
deterministic**:

* **Run** starts the library pipeline with the same `--ndjson` event contract and parses one
  JSON object per line. Nothing is scraped from prose.
* A small control set: **Run**, **Stop**, and for the *next* run a
  `--time-budget` / `--generations` cap. "Pause" is offered honestly as "Stop
  after this generation budget", because the engine's determinism means a
  stopped-then-rerun search with the same `--seed` is reproducible; true
  mid-run pause/resume needs #17 (save and resume a run) and is deferred.
* **Progress plot**: `best_error` (and `pareto_members`) against `generation`
  from the `progress` events, with `total` marking the horizon. This is the
  original engine's "Progress over time", driven by real streamed points.
* A **statistics strip** below it: elapsed time, generation / total,
  `pareto_members`, and — once the engine emits them (#18) — confidence,
  stability, maturity and percent converged. Each is shown only if the stream
  provides it, so the GUI tracks the engine rather than guessing.
* The **log** is the child's stderr, verbatim (the `# …` lines the CLI already
  prints: the split line, `notes`, elapsed, diagnostics). It is copyable and is
  what the GUI shows when something goes wrong.

**Feeds:** the `start`, `progress`, `solution`, `null_control`, `baselines`,
`residual_tests` and `done` events. **Consumes:** the argv from §3.3 and the
process exit code (`0`, `2`, `3`, `4`).

### 3.5 Solutions

**Job:** the Pareto front as a table, and the selected law as text.

* **Front table** — one row per `pareto` entry, columns from the JSON:
  `rank`, `complexity`, `k_constants`, `train_error`, `test_error`,
  `overfit_index`, `aicc`, `bic`, `durbin_watson`, `fold_errors`, and the
  `expression`. The default sort is the engine's order (simple → complex), so
  the knee is easy to find.
* **Selection** drives the Diagnostics pane and the charts. Selecting a row
  never re-runs anything: the row already carries the diagnostics.
* **Null-baseline strip** — the `baselines` object rendered as three reference
  rows (`mean`, `linear`, `knn:5`) in the *same metric*, so any front row that
  is worse than `knn` is visually flagged. The CLI's rule — "a model must beat
  them to earn its complexity" — becomes a visible threshold line.
* **Equation view** — the selected `expression`, in a proportional-font box with
  a copy button, complexity and constant count beside it.
* **Bootstrap column** — when the run used `--bootstrap-error`, each row shows
  `estimate` (the `.632+` error) and `weight`; the tooltip carries `apparent`
  and `oob`.

**Feeds:** the `solution` events / `pareto` array, and `baselines`.

### 3.6 Diagnostics

**Job:** everything needed to decide whether to trust the selected law. This is
the pane that makes principle 1.2 real.

* **Split report** — from the `split` object: mode, train/test rows, folds,
  gap, groups, lags, series, history rows, and the `notes` list verbatim.
* **Error table** — train vs test, per-fold `fold_errors`, `overfit_index`.
* **Information criteria** — `aicc`, `bic`, with the reminder that lower is
  better and that they compare models on the same data.
* **Residual tests** — `durbin_watson`, `ljung_box` (statistic, df, p-value),
  `breusch_pagan` (statistic, df, p-value), `skewness`, `kurtosis`. A p-value
  below 0.05 is coloured, with the plain-language reading beside it.
* **Null control** — when `--null-runs` was used: the null error distribution,
  `min`, `median`, `p_value` and `verdict`.
* **Bootstrap** — the `.632+` `apparent` / `oob` / `estimate` triple.
* **Chart gallery** — the four charts of §5 for the selected solution.

**Feeds:** the `residual_tests`, `null_control`, `baselines` and `done` events,
and the selected `pareto` row.

---

## 4. Workflows

### 4.1 The happy path

```
import → prepare → define → run → inspect → compare → export
```

1. **Import.** Open or paste a table. The grid fills; roles default sensibly.
2. **Prepare.** Set the target and inputs, choose the split, and (for series)
   the lags and history. Watch the split stripe.
3. **Define.** Fill the Search form (metric, operators, complexity, budgets).
   The CLI bar shows the argv.
4. **Run.** Launch in `--ndjson`; watch progress; read the log if it exits
   non-zero.
5. **Inspect.** Pick a front row; read the formula and its diagnostics.
6. **Compare.** Keep several searches in the project and diff their fronts
   (below).
7. **Export.** Write the front to CSV/JSON/TSV, copy the equation, copy the
   reproducing command.

### 4.2 Compare

Selecting two or more searches in the project overlays their Pareto fronts on
one chart (colour per search) and lists, for each, the simplest solution that
beats the `knn` baseline. A "seed stability" view (planned, #20) re-runs the
same search across several `--seed` values and shows how often the same
*variables* recur — the engine's own advice ("prefer the answer the search
agrees on") made visible.

### 4.3 The time-series path

Time series is a first-class path, not a special case, because the engine's
split logic is where most mistakes happen.

1. In the Data pane, set one column to **time** and (for panels) one to
   **group**. The split stripe re-draws by `--time` order.
2. Choose the split: `temporal` for one series, `temporal-group` for panel data.
   Add `--gap` so near-in-time rows cannot leak across the boundary; the stripe
   shows the embargo rows too.
3. In Prepare, set `--lags K` (the "look-back"), `--history-frac f` (the leading
   rows that seed the lags but are not scored) and `--gap-break f` (so a lag
   never jumps a hole). The series preview shows the resulting columns.
4. Run and inspect. The Diagnostics pane flags that the row order is real, so
   `durbin_watson` is meaningful (the CLI suppresses that remark for random
   splits; the GUI mirrors it).
5. The **rolling-origin** view (planned, bound to the cross-validation mode of
   #19/#20) plots the forward-chaining runs: train on the past, test on the next
   block, roll forward, one point per origin. Until that lands, `--folds K`
   already reports contiguous test blocks and the GUI plots the per-fold errors
   as a bar strip, which exposes an unstable model.

### 4.4 Evaluate a hand-written formula

`--eval-expr` scores a formula without searching. The Solutions pane offers an
"Evaluate expression" box: type `2*x + 1`, and the GUI runs the same command
with `--format json`, showing `train_error`, `test_error`, `complexity`,
`k_constants` and the canonical `expression`. This is the round trip from a
printed right-hand side back into the tool.

### 4.5 Export

* **Front** — write the `--out` TSV (columns as the CLI writes them) or the full
  JSON.
* **Equation** — copy the selected `expression` as text, or the whole selected
  solution as a small JSON object.
* **Command** — Copy as CLI: the exact argv, including `--seed`, so anyone can
  reproduce the run.
* **Report** (planned, #29) — a self-contained HTML summary per project.

---

## 5. Charts

Five charts, each bound to a specific engine output. No chart is drawn from
prose or from a re-computation the engine did not do.

### 5.1 Pareto front (error vs complexity)

* x: `complexity`; y: error (train and test as two series from each row's
  `train_error` / `test_error`).
* One point per `pareto` row; the knee (largest error reduction per unit
  complexity) is highlighted.
* Reference lines at the `baselines` (`linear`, `knn`) so "does it beat a line?"
  is answered visually.
* Series toggle for train vs test: the gap between them *is* the overfitting.
* This is the live chart too: as `solution` events arrive during a run, points
  are appended.

### 5.2 Observed vs predicted

* Scatter of observed target vs fitted value, per row, for the selected
  solution. Points come from the run's own evaluation; the diagonal is `y = x`.
* Coloured by split side (train vs test) so a model that only fits the training
  rows is visible at a glance.

### 5.3 Residual gallery

Four panels for the selected solution, all standard, all answering a question
the numbers only hint at:

* **Residual vs fitted** — is there curvature or changing spread left?
  (Breusch–Pagan in pictures.)
* **Residual vs row order** — is there autocorrelation? (Durbin–Watson and
  Ljung–Box in pictures; only meaningful when the order is real.)
* **Residual histogram** — is the error symmetric and roughly Gaussian?
  (relates to `skewness` and `kurtosis`.)
* **Normal QQ** — the same question as the histogram, read at the tails, where
  big misses live.

### 5.4 The overfitting wall

* x: `complexity`; y: error; two lines: train and held-out (test), plus the
  per-fold error band (`fold_errors`).
* The **wall** is the complexity at which the held-out line turns up while the
  train line keeps falling. To the right of the wall, more complexity buys
  fitting, not generalisation.
* The `knn` and `linear` baselines are horizontal reference lines. The reading
  rule is stated on the chart: "pick the simplest model left of the wall that
  beats `knn`".

### 5.5 Cross-validation / stability (staged)

* When the cross-validation mode (#19/#20) lands, a **fold spread** chart:
  one point or interval per fold/origin, so the variance of the estimate is
  visible next to its mean.
* Today, `--folds K` already gives `fold_errors`; the GUI draws them as a strip
  under the front so a lucky pooled average cannot hide a bad block.

---

## 6. Architecture

### 6.1 The contract the GUI is built on

**`rcsr schema`** — one JSON document: `tool`, `version`, `options[]`
(each `{name,type,default,description}`), `operators[]`, `exit_codes`, and the
stdout/stderr convention. This is the whole interface definition and it is read
at startup. Nothing about the flag list is hard-coded in the GUI.

**`rcsr --json`** — one object per run:

```
tool version seed metric target variables train_rows test_rows
split { mode rows train_rows test_rows folds[] gap groups_total
        groups_train groups_test lags series history_rows notes[] }
generations_run converged_early stop_reason elapsed_seconds
timing { prep_seconds search_seconds baselines_seconds diagnostics_seconds
         report_seconds unaccounted_seconds total_seconds gpu_seconds }
gpu { kernel_s launch_s sync_s alloc_s h2d_s d2h_s transfer_s window_s
      host_s accounted_s wall_s idle_s residual_fraction h2d_bytes d2h_bytes batches
      failed_batches programs evals row_evals evals_per_s row_evals_per_s
      batches_per_s eval_kernel_duty eval_window_s device_seconds_incl_fit
      device_fraction_of_run eval_fraction_of_run process_total_s
      sm_occupancy occupancy_source
      quality_per_gpu_hour generations[ { generation programs fallbacks
      opcodes{} } ] } | null
null_control { runs best[] min median p_value verdict } | null
baselines { mean{train_error,test_error} linear{…} knn{k,train_error,test_error} }
residual_tests { n lags durbin_watson
                 ljung_box{statistic,df,p_value}
                 breusch_pagan{statistic,df,p_value} skewness kurtosis } | null
pareto [ { rank complexity k_constants train_error test_error aicc bic
           durbin_watson overfit_index fold_errors[] bootstrap_error expression } ]
```

**`rcsr --ndjson`** — one object per line while a run streams, in order:
`start`, `progress`…, `solution`…, then `null_control`, `baselines`,
`residual_tests`, and `done`. (`--eval-expr --ndjson` emits a single `eval`.)
This is what the Run pane consumes; it is also what makes the Pareto chart grow
live.

**Exit codes** — `0` success, `2` usage error, `3` input/data error,
`4` no finite solution. The GUI maps each to a message and, for `2` and `3`,
highlights the offending form field where it can and otherwise shows the log.

### 6.2 Driving the stream

The as-built server runs the same library pipeline in-process (see
"As-built screens and deviations" above) and emits the NDJSON event objects
directly:

```
prepare-full-weighted-with → search::run → start / progress / solution … / done
```

It still dispatches on the `event` field and never parses human-readable text.
There is no child process or pipe to manage; the CLI's `--ndjson` stream and the
server's stream carry the same event names and payloads (the server adds `bic`
to `solution`). Cancelling the HTTP request stops the run through the server's
shutdown flag.

### 6.3 Schema-driven forms (how a new flag reaches the GUI)

1. A developer adds `--weight` to `schema::OPTIONS` in `src/schema.rs`.
2. The GUI, on next launch, reads the new option from `rcsr schema`.
3. The generic renderer maps its `type` to a widget and, because it is not in
   the GUI's layout table, places it in the **"Other"** group automatically.
4. It appears in the form immediately. Whether the server can execute it is
   decided by `parse_args` (or the explicit `UNSUPPORTED` table), and the
   contract test fails if the answer is neither "supported" nor a named,
   reasoned rejection.

The renderer needs no GUI change, but the execution parser is still a separate
match in `gui.rs`: a new flag must be added to that match or to the explicit
`UNSUPPORTED` table, with the reason shown to the user. `tests/gui.rs` enforces
this for every documented root option, so "rendered but rejected as unknown"
cannot ship. The layout table only *improves* placement; it is never required
for the form to render a flag.

### 6.4 Version skew

The GUI records the `version` from `rcsr schema` at startup and compares it to
the version of the results it loads (each result carries `tool` and `version`).

* **Same version** — full fidelity.
* **Newer binary than GUI** — unknown options render generically (§6.3);
  unknown result fields are preserved untouched and shown in a "raw" disclosure.
* **Older binary than result** — fields the binary no longer emits are marked
  "not produced by this version" rather than silently dropped.
* **Schema missing or unparsable** — the GUI still works in a reduced mode:
  file I/O, table view, and running a pasted command line, with the form
  disabled and a clear banner. It never blocks on a schema it cannot read.

### 6.5 Headless mode

For CI and screenshot tests, the GUI accepts a project file plus a run and
renders panes to PNG/SVG **without a display** and without a window:

```
rcsr-gui --project p.json --render front,diagnostics --out shot.png
```

This keeps the visual layer testable and makes "does the GUI still build the
right chart from this JSON?" a checkable question.

### 6.6 Language and process

The engine is Rust; the GUI is too, sharing the crate's data types to avoid
drift. The as-built server deliberately **links the library and runs the search
in-process** instead of spawning `rcsr`: one local HTTP response, no
process/pipe layer, and the same functions the CLI uses. This is the documented
deviation above. It means the server is not a way to drive a *newer* binary,
but it does share one schema document and one search implementation. The CLI
remains the canonical interface, and every action still has a Copy-as-CLI
equivalent.

---

## 7. Accessibility and keyboard-first operation

The engine is used by people who live in a terminal; the GUI should not slow
them down.

* **Full keyboard operation.** Every action has a shortcut. Suggested core set:

  | key | action |
  |---|---|
  | `Ctrl+O` | open data / project |
  | `Ctrl+S` | save project |
  | `Ctrl+N` | new dataset |
  | `Ctrl+Shift+N` | new search |
  | `Ctrl+R` / `F5` | run search |
  | `Esc` | stop run |
  | `Ctrl+L` | focus the CLI bar |
  | `Ctrl+Shift+C` | copy as CLI |
  | `Ctrl+E` | copy the selected equation |
  | `Tab` / arrow keys | move within the grid and tables |
  | `,` / `.` or `[` / `]` | previous / next solution |
  | `Ctrl+1..6` | jump to workspace tab |

* **Focus is always visible**, and every pane announces itself to a screen
  reader with a role and a name. The grid, tables and charts expose row/column
  headers so a reader can navigate a result without seeing it.
* **The formula panel is a plain text buffer**, so it can be selected, copied
  and read by assistive technology; charts carry a text alternative (the
  underlying numbers) via a "show data" toggle.
* **Colour is never the only signal.** The split stripe uses pattern as well as
  colour; the overfitting wall uses line style and labels; significant p-values
  carry an icon and a word ("autocorrelation left"), not just red.
* **Respecting the OS**: system font scaling, a light/dark theme, and a
  minimum hit target. The default theme is calm and high-contrast.
* **No modal on the happy path.** Dialogs are reserved for destructive or
  irreversible choices (discard unsaved results).

### 7.1 Agent parity

The parity statement, in one line:

> **Every GUI action is a CLI flag, and every CLI output is a GUI view.**

Operationally:

* The **CLI bar** is always visible in the Search pane; any GUI edit updates it
  and any edit to it updates the form. Copying it reproduces the run exactly
  (`--seed` included).
* Any run can be **imported as a command line** and is indistinguishable in the
  GUI from a run started by clicking.
* The GUI adds no hidden options and removes none; it sets exactly the flags the
  schema declares and writes them in a deterministic, sorted order, so the same
  form always yields byte-identical argv.
* A machine can drive the GUI headlessly (§6.5) or drive the binary directly;
  the two paths produce the same result object, which is what makes the GUI
  optional and the CLI canonical.

---

## 8. Non-goals for v1, and the MVP plan

### 8.1 Non-goals (v1)

* **No second search engine.** The GUI does not reimplement the search; it
  calls the same library entry point as the CLI. That is the as-built
  in-process design, not a fork of the algorithm.
* **No mid-run pause/resume.** The engine has no pause primitive; until #17,
  "pause" is honestly "stop"; runs are bounded and deterministic instead.
* **No cloud or remote execution.** Renting compute, accounts, credentials and
  billing are out of scope. The engine is local and `std`-only; the GUI stays
  local.
* **No licensing, accounts or telemetry.** There is no product key, no trial
  countdown, no usage reporting. (If the project ever needs any of that it gets
  its own design, not a corner of this one.)
* **No editing the dataset in a way the CLI cannot read.** The grid edits only
  what a CSV/TSV round-trips.
* **No new metrics, operators or search algorithms.** Those live in the engine
  and reach the GUI through the schema. The GUI does not duplicate them.
* **No 3-D, animated or decorative charts.** Only the four charts of §5.
* **No report designer.** A single self-contained HTML summary (when #29 lands),
  not a template language.
* **No auto-changing the split.** The GUI warns and offers, it does not decide.
* **No cross-platform packaging promises in v1** beyond "builds and runs on the
  developer's Linux with a Rust toolchain".

### 8.2 Staged MVP plan (maps onto #38)

**Stage 0 — contract plumbing (no UI).**
Read `rcsr schema`; stand up the child-process NDJSON reader; parse a `--json`
object into typed structs; round-trip to the CLI bar. Prove it with a headless
run of the bundled `examples/line.csv`. *(#38 "It launches `rcsr` as a
subprocess … and streams".)*

**Stage 1 — Data and Search (the form).**
Spreadsheet pane with roles; schema-generated Search form; CLI bar with copy.
This alone makes `rcsr` usable without a terminal. *(#38 items 1–2.)*

**Stage 2 — Run and live Pareto.**
Run as `--ndjson --progress`; progress plot; Pareto plot that grows as
solutions arrive; solution list; log. *(#38 items 3.)*

**Stage 3 — Inspect.**
Solution detail: formula, complexity, `k_constants`, train/test error,
`-aicc`/`bic`, `durbin_watson`, residual tests, baselines, `bootstrap_error`,
`fold_errors`. Diagnostics pane with the residual gallery. *(#38 items 4.)*

**Stage 4 — Export and compare.**
Export front (CSV/JSON), copy equation, Copy as CLI, project save/load, and the
multi-search front overlay. *(#38 item 5.)*

**Stage 5 — Time series and polish.**
The time-series path (§4.3): split stripe, series preview, per-fold strip;
accessibility pass; headless render mode for CI.

**Deferred, tracked by their engine issues.** Data preparation controls (#2),
row weights (#1), selection-expression splits (#3), CV/rolling-origin
(#19/#20), stability (#20), progress statistics (#18), save/resume (#17),
reports (#29). Each is enabled in the GUI automatically when its flag appears in
`rcsr schema` (§6.3) — the GUI does not wait for a redesign.

---

## 9. Main-window sketch

```
┌──────────────────────────────────────────────────────────────────────────────────────┐
│  rule-and-compass   File   Edit   Project   Search   View   Help          rcsr 0.1.0   │
├──────────────────────────────────────────────────────────────────────────────────────┤
│  ＋Dataset  ＋Search │  ▶ Run   ■ Stop │  ⧉ Copy as CLI │  ⌕ Find solution…            │
├───────────────────────────────┬──────────────────────────────────────────────────────┤
│ PROJECT                       │ [ Data ] [ Prepare ] [ Search ] [ Run ]               │
│                               │ [ Solutions ] [ Diagnostics ]                         │
│ ▾  panel.csv                  │ ┌──────────────────────────────────────────────────┐  │
│      d  Search 1   done ▸     │ │ Pareto front — error vs complexity               │  │
│      d  Search 2   done       │ │  err ┤                                           │  │
│ ▾  sales.csv                  │ │      │·            train ●━━●                       │  │
│      d  Search 1   running    │ │      │ ·        test  ○╌╌╌○                       │  │
│                               │ │      │  ·  ·  ·   ○──○────○  ← knee               │  │
│                               │ │      │ ── knn baseline ────────────────────────  │  │
│                               │ │      └──────────────────────────────────▶         │  │
│                               │ │                     complexity                    │  │
│                               │ └──────────────────────────────────────────────────┘  │
│                               │ ┌──────────────────────────────────────────────────┐  │
│                               │ │ #  cmplx  k   train_err  test_err  overfit  aicc  │  │
│                               │ │ 1    1    1   0.41200    0.43800   1.06    -12.1  │  │
│                               │ │ 2    3    2   0.00910    0.01100   1.21    -88.4  │  │
│                               │ │ 3    5    2   0.00001    0.00001   1.00   -402.7  │  │
│                               │ └──────────────────────────────────────────────────┘  │
│                               │ ┌──────────────────────────────────────────────────┐  │
│                               │ │ y = 1 + 2·x            [ Copy equation ]          │  │
│                               │ │ complexity 5 · k_constants 2 · Durbin–Watson 2.01 │  │
│                               │ │ beats knn? yes · Ljung–Box p=0.49 · BP p=0.29     │  │
│                               │ └──────────────────────────────────────────────────┘  │
├───────────────────────────────┴──────────────────────────────────────────────────────┤
│ rcsr --data panel.csv --target y --vars x --metric absolute --generations 200        │
│ --islands 4 --pop 200 --max-complexity 30 --seed 0 --split random --folds 1           │
│                                         [ Copy ]  [ Run ]                             │
├──────────────────────────────────────────────────────────────────────────────────────┤
│ gen 200/200 · 3 solutions · 1.9 s · exit 0                        │  ● ready           │
└──────────────────────────────────────────────────────────────────────────────────────┘
```

* The left dock is the project notebook (§2.1); `d` is a dataset, `▸` a search
  with its run state.
* The centre is one tab at a time; the sketch shows Solutions with the Pareto
  chart, the front table and the selected equation's one-line diagnostics.
* The **CLI bar** sits above the status bar and is always visible, closing the
  loop between clicking and typing (§1.3, §7.1).
* The status bar carries the engine's own progress and the exit code.

---

## 10. Summary

The GUI is a **view over the CLI's contract**. It reads `rcsr schema` to build
its forms, drives `rcsr --ndjson` to stream a run, and renders the `--json`
result as a front, a set of diagnostics and four charts. It makes the split
visible, puts the diagnostics beside every fit, and keeps the formula in front.
It adds nothing the engine cannot reproduce, and it can always hand the user the
exact command that produced what they are looking at.
