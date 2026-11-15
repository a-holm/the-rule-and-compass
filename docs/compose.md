# Regime-switching composition (`rcsr compose`, #1181)

`rcsr compose` builds formulas that switch between a bull and a bear candidate. It reads the fronts of finished pool searches, builds every composite that a pre-registered gate menu allows, chooses one on the training and validation rows, and writes a run.json whose finalist is one exact expression.

```sh
rcsr compose --data dev.rcsb --target lbl_h24a_y --target-spec dev.targets.json --vars "$VIEW" --ops "$OPS" \
  --time ts --group-by symbol --split-table cpcv61/splits.csv --split-id s07 \
  --objective sharpe --cost-bps 10 \
  --gates menu.json --bull bull_s07.run.json --bear bear_s07.run.json --all-weather f1_s07.run.json \
  --top 5 --json --out compose_s07.run.json
rcsr walkforward --data dev.rcsb --target lbl_h24a_y --from-run compose_s07.run.json --windows cpcv61/windows_s07.csv --json
```

The flags before `--gates` are the flags of the pool searches. `--fitness pnl`, `--select-rule robust` and `--cpu` are implied. The command builds the rows of that run (training, validation and test) and stops before any search. `--vars` and `--ops` define the variables and the operator alphabet of the branch code (below); pass the ones of the pool searches.

## Pools

A pool run is a run.json with an objective block. Each `--bull`, `--bear` and `--all-weather` flag takes one run and is repeatable. A run gives its top `--top K` front members by archive rank (default 5). Bull and bear runs are usually searched with the regime-masked fitness of `--pnl-mask` (#1180); a plain run is accepted as well.

Every input run must share with this invocation:

| field | source in the run |
|---|---|
| target | `target` |
| target spec | `target_spec` (by its hash) |
| cost | `objective.spec.per_side_bps` |
| split | `split.mode`, `train_rows`, `validation_rows`, `test_rows`, `train_time`, `validation_time`, `required_purge_seconds`, `purge_seconds`, `split_id`, `intervals`, `interval_purge_seconds`, `embargo_seconds` |

A difference exits 2 and names the field. A run.json records no data hash, so the row counts, the time ranges and the intervals stand in for it. A run that priced a band, a minimum hold, the top-tercile book, a position size or a funding column exits 2, as does a front member without `expression_exact`.

`--embargo auto` takes the lookback of the gate and branch columns into account, so a gate with a long lookback gives a longer embargo. The pool searches of such a configuration use the same embargo; otherwise their row counts differ and compose exits 2.

## The gate menu

```json
{"schema":"rcsr-gate-menu/1","switch_budget":8,"gates":[
  {"name":"btc28","codeword":"000","expr":"btc_r672","band":0.02,"dwell":168},
  {"name":"breadth","codeword":"010","expr":"mkt_breadth - 0.5","band":0.05,"dwell":168},
  {"name":"btc7","codeword":"111","expr":"btc_r168","band":0.0,"dwell":1}]}
```

The menu is pre-registered: each gate fixes its expression, band, dwell and codeword, so the menu index is the only free choice. The composite code (`docs/occam-code-v1.json`) registers 8 menu gates, so a menu holds 1 to 8 gates and each codeword is 3 binary digits, distinct; the codeword read as a binary number is the gate's menu index in the code. The menu's sha256 is in `compose.menu`.

The state of a row follows the `--pnl-mask` rule of #1180 (`regime_gate::hysteresis_states`): on above `+band`, off below `-band`, else the previous state; a change waits until the current state has held `dwell` rows of the group; a non-finite value is off. The state starts at each group's first row of the slice, and on the training rows at each training segment. A gate with band 0 and dwell 1 is the expression's own `greater(gate, 0)`.

Gate columns pass the `--pnl-mask` column checks: a label-family column, the target, its return column or a construction column exits 5, a missing column exits 3, an aggregate or meta operator exits 2. The branch columns pass the same checks.

## Composites

| type | expression | built from |
|---|---|---|
| C0 | `f` | every all-weather member |
| C1 | `if(greater(gate, 0), f, -1.0)` | every bull member, per gate |
| C1 | `if(greater(gate, 0), -1.0, f)` | every bear member, per gate |
| C2 | `if(greater(gate, 0), f_bull, f_bear)` | every bull and bear pair, per gate |

A branch is its member with its run's threshold subtracted, `(f) - (t)` (just `f` for threshold 0), so the composite is long where its value is above 0 and the composite threshold is 0; `-1.0` is cash. For finite values `f - t > 0` holds exactly when `f > t`, so each branch holds the book of its run. The composite count is

`gates x (bull x bear + bull + bear) + all_weather`

(285 for 8 gates, 5 bull, 5 bear and 5 all-weather members). Every composite is counted in `search.candidates_evaluated` and listed in `compose.composites` with its id, rank, type, gate, codeword, branches, expression, complexity, bits and validation figures.

### Description length (report-only)

`bits` is the length of the composite under the composite code of #1184: the type selector (`0` C0, `10` C1, `110` C2), 3 bits for the menu gate, 1 bit for the cash side of a C1, and per branch the flag bit plus the #1184 tree code of the member under the alphabet of `--ops` and the `--vars` count. A constant that no tier of the code holds is written at the code's precision; `members[].exact_codeword` is false then. A member that reads a column outside `--vars` or an operator outside `--ops` exits 2. `compose.code` holds the code and its sha256.

### Pruning

With `switch_budget k` in the menu, the k-switch bound of #1185 is computed for every C1 and C2 pair on the training rows. A pair whose bound gain over its better branch is not positive is pruned for every gate: it is counted and listed (`pruned`, `bound_gain`) but not scored. The bound is a necessary condition only.

## The choice

Every composite that is not pruned is priced on the validation rows by the long/flat book code. The choice is the robust rule of `--select-rule robust` (`--select-blocks`, `--select-se`, `--select-block-len`, `--select-blocks intervals`) with one more condition for a C1 or C2: the alpha over exposure-matched buy-and-hold is at least 0 in the gate-on rows and in the gate-off rows of the validation slice. The finalist is the eligible composite of lowest complexity within `--select-se` standard errors of the best pooled alpha. No eligible composite gives verdict `no_signal` and the best all-weather member by the validation objective, or the best composite when there is no all-weather pool.

The state view of a book splits it by gate state. The book of state s holds the composite's score on the rows of state s and a flat score elsewhere; a row with no finite score stays one in every state book, so all of them follow the same missing-score rule as the composite (#1206). Each is priced by the book code. Its alpha is its net minus its exposure (long rows over priced rows of the state) times the net of the buy-and-hold book of the same rows. The composite net is the sum of the two state nets and the seam, `2 side J` of the seam identity (#1185). A period is in the state that most of its priced rows hold.

Only the training and validation rows are read before the finalist is fixed. The test rows are priced afterwards, for the finalist, its branches and the best all-weather member.

## Output

The run.json has the fields of a search run that the evaluators read: `target`, `variables` (the gate and branch columns), `split` (of this invocation), `target_spec`, `search.candidates_evaluated`, `objective` (`spec` with threshold 0, `finalist.expression_exact`, `validation`, `test`, `selection_rule` of the robust rule over the composites) and `pareto`: the finalist first, then the other composites whose gate is a row expression, in rank order. `compose` (`rcsr-compose/1`) adds:

* `menu`, `code`, `runs`, `members`, `count` and `candidates_evaluated`;
* `pruned`, `scored`, `usable` and `rows` (training, validation and test rows);
* `selection`: rule, blocks, regime condition, number eligible, verdict and the chosen id;
* `gate`: the finalist's gate, `carried` when it has a band or a dwell;
* `chosen`: the validation and the test book of the finalist with its `regime` block (per state rows, share, net, exposure, `buy_and_hold_net` and alpha; the seam; switches, switches per year and the mean state length in periods; `switch_cost`, the cost of the position changes in the periods where the state changes, against `total_cost`), the seam identity on the training rows (`train_seam`), each branch alone and the best all-weather member on the same rows;
* `composites`: every composite.

`walkforward --from-run`, `walkforward --front` and `ladder --sr-run` price the composite unchanged. Out of sample the run is one trial; a walk-forward window on rows the composition selected on takes `search.candidates_evaluated` as its trial count (#1027).

A gate with a band or a dwell has state, so its composite is not a row expression. Such a finalist is carried in `compose.gate` (`carried: true`), and every reader of a run.json (walk-forward, ladder, the seal, `--reselect-run` and compose itself) refuses the run: the finalist expression alone would price another book. `objective.test` holds the composite's own test book. The other composites with a stateful gate stay in `compose.composites` and are left out of `pareto`.

## Refusals

| case | exit |
|---|---|
| no `--gates`, no `--bull` and no `--bear`, `--top 0`, a bad menu | 2 |
| input runs that differ in target, target spec, cost or split, from each other or from this invocation | 2 |
| `--threshold`, `--position`, `--book-target`, `--band`, `--band-units`, `--min-holding`, `--max-position`, `--cash-return`, `--funding`, `--pnl-mask`, `--gpu`, a `--select-rule` other than robust, a `--fitness` other than pnl | 2 |
| `--lags`, `--lag-set`, `--fill`, `--outlier-iqr`, `--smooth` | 2 |
| no validation rows, or no composite with a usable validation book | 2 |
| a gate or branch column of the label family, the target, its return column or a construction column | 5 |
| an unreadable menu or run, a missing column | 3 |
