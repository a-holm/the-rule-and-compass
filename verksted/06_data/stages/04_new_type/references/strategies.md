---
type: reference
description: How to classify an unknown dataset and choose its time, split, lock, target and view rules.
---

# Strategies for a dataset no standard type covers

Use this file when [the register](../../../references/data-types.md) has no `ready` row for the dataset. Answer from `profile.md` and the card only. Never open a data row to answer a question. A guess is allowed only when it is written into `deviations`.

## The six structure questions

1. Time: is there a time column, in which unit, and is it the event time or the publication time?
2. Spacing: is the spacing regular, and what is the median step?
3. Entity: is there an entity key (symbol, id), and do entities enter and leave?
4. Columns: which columns are prices, which are amounts, which are flags, which are identifiers?
5. Revision: can a value change after its first release?
6. Overlap: may a later row depend on an earlier target, so that labels overlap?

## Classification

| Answers | Type | Action |
|---|---|---|
| Regular spacing and open, high, low, close columns | `04_ohlcv` | Reclassify: set `type_stage` in `profile.md`, stop here. |
| Trades or quotes with a time column | `04_ohlcv` after a fold | Fold to bars with a hashed step (record input and output hashes), reclassify, and keep the raw files for a later tick stage. |
| Publication time differs from event time | this stage | Join by publication time plus the stated lag, never by event time (a point-in-time join). |
| No time column | this stage | Not a time-series dataset. Random splits are allowed only here. The lock is a held-out file declared by its SHA-256. |
| Anything else | this stage | Use the rule table below. |

A fold is a conversion, not a feature step: it never adds a column the source does not hold.

## Rules by structure

| Structure | Time and split | Notes |
|---|---|---|
| Time-ordered, single entity | chronological split, purge and embargo | One lock at the newest end. |
| Time-ordered panel with entry and exit | chronological split on the shared clock | Point-in-time membership, per-group reporting, no one-hot identity input (`docs/time-series.md` section 6). Keep exited entities. |
| Irregular events | bucket to a fixed step or use event-count bars first | State the choice and the step in `deviations`. A median step is not a spacing. |
| Revised series | first-release values only, or both with the release time | A revised value with no release time stops the run. |
| Cross-section per period | split by period, never inside a period | Rank targets, per-period books. |
| No time axis | random or grouped split | Group rows that share an entity so one entity never sits on both sides. |

## Choosing the lock

- Take the newest part of the data. It must cover the label horizon and the regimes the question names.
- Declare it before the first look, with the data. `rcsr import` takes no lock flag, so write the sidecar `<store>.lockbox.json` (and the same file as `<file.csv>.lockbox.json` for the audit). Builders such as `rcsr prep` and `rcsr enrich` take `--locked-test START,END`. The rules of [time-rules](../../../references/time-rules.md) hold.
- Development ends one purge before the lock start. Record it as `dev_end`.
- Data with no epoch time: hold out a file, record its SHA-256 as `locked_test` in `adapt.md`, and never open it before the final read. The engine reads no range for such data, so the hold-out is kept by process.

Sidecar shape (epoch seconds, half-open): `{"locked_test":{"start":S,"end":E,"start_date":"YYYY-MM-DD","end_date":"YYYY-MM-DD","source":"sidecar"}}`.

## Choosing the target

1. Decision time first: which row is the last one the decision may use?
2. Entry: the next row after the decision row.
3. Horizon: how many rows the position is held.
4. Cost: per side, inside the label.
5. The label must not encode a column of the view (`docs/time-series.md` section 2b). Long only.

## Choosing purge, embargo, stride and periods per year

Use the formulas of `time-rules.md`: purge is `entry_offset + horizon + delay`, the embargo is the longest feature lookback, one period is `max(horizon_seconds, stride_seconds)`, and `periods_per_year = 31536000 / period_seconds`.

Worked example: 15-minute rows (900 s), horizon 8 rows, entry offset 1 row, delay 0, longest lookback 96 rows. Purge is 9 rows (8100 s). Embargo is 96 rows (86400 s). With stride 8 rows the period is 7200 s and `periods_per_year` is 4380. A Sharpe at this period is not a daily Sharpe.

## Choosing the first view

- Start with a few geometry columns and one or two lags (`docs/time-series.md` section 9). Add more only when the audit stays clean.
- Every derived column is causal on a prefix: its value at row `i` is the same on the full data and on data cut at `i`.
- Identifier and label columns are never inputs. Run with `--require-clean-inputs`.
- A column that needs a raw field the source lacks goes into `view_excludes` with the reason.
- Write what the engine cannot use into `view_excludes`: text, free-form categories with many levels, columns known only after the decision time.

## Mandatory checks before the store leaves the stage

1. Prefix causality of every derived column: recompute on a cut copy and compare.
2. `rcsr audit --data <file.csv> --target <col> --time <col> --horizon <N>` exits 0 (it reads a CSV, not a store; exit 5 is a leak). It reads the range from `<file.csv>.lockbox.json`, or from `--lockbox START,END`.
3. Lock refusal drill: `rcsr --data <store> --target <col> --time <col> --split temporal --test-frac 0.3 --generations 2 --pop 20` exits 3 and names the declared locked test; `rcsr lock check --record <record.json> <dir>` exits 0 on the sidecars.
4. The copy-paste audit of `docs/time-series.md`.

## Stop rules

Stop and ask when:
- the licence is unclear;
- the time unit cannot be decided from the profile;
- values look revised and no release time exists;
- the lock cannot be declared before a development result exists;
- a file name or header suggests rows at or after `dev_end`.

## Record

- Every guess goes into `deviations`.
- A dataset shape that is likely to recur gets a proposal: a new row in the register (type, stage `04_<name>`, status `planned`, what a source holds, what decides the variant) and a placeholder issue that names the proposal file. The new stage is written from the lessons of the runs that used this stage.
- Lessons go through [09_lessons](../../09_lessons/CONTEXT.md): a rule true for every dataset of the shape goes to a tracked reference, a rule for this dataset or machine goes to the local area.

## Sources

- Purged k-fold and embargo: Marcos Lopez de Prado, Advances in Financial Machine Learning, Wiley 2018, chapter 7 (Cross-Validation in Finance).
- Time-series cross-validation with training sets that precede each test point: Hyndman and Athanasopoulos, Forecasting: Principles and Practice, section Time series cross-validation, https://otexts.com/fpp3/tscv.html
- Cross-validation of time-series predictors: Bergmeir and Benitez, On the use of cross-validation for time series predictor evaluation, Information Sciences 191, 2012.
- Chronological splits with a gap: scikit-learn, `TimeSeriesSplit`, https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split
- Point-in-time data keyed on the release date, not the period date: S&P Capital IQ, Quantamental research, point-in-time versus lagged fundamentals, https://spglobal.com/content/dam/spglobal/mi/en/documents/general/sp-capitaliq-quantamental-point-in-time-vs-lagged-fundamentals.pdf
- Vintage (as-of) data: Federal Reserve Bank of St. Louis, ALFRED, https://alfred.stlouisfed.org/
- Repo: `docs/time-series.md`, `docs/leakage.md`, `docs/data-format.md`, `docs/data-preparation.md`.
