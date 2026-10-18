# Complexity: beyond the node count

`rcsr` trades error against **complexity**, and the Pareto front it prints is
ranked on that complexity. By default complexity is the number of nodes in the
expression, which is the right first approximation for interpretability. Real
parsimony often wants more: a `pow` or an `if` is harder to read than an `add`,
a fitted constant is cheaper than a new variable, and deeply nested expressions
are harder to follow than flat ones.

The complexity model makes those preferences explicit. Every candidate is
scored with

```
complexity(f) = sum(op_cost)               # operator nodes only
              + var_weight   * n_vars
              + const_weight * n_consts
              + depth_weight * max_depth
```

* `sum(op_cost)` runs over every node that is **not** a terminal (`Const` or
  `Input`).
* `n_vars` and `n_consts` count `Input` and `Const` leaves.
* `max_depth` is the number of edges on the longest root-to-node path, so a
  single terminal has depth `0`.

The defaults are `op_cost = 1` for every operator, `var_weight = 1`,
`const_weight = 1` and `depth_weight = 0`. That is **exactly the node count**,
so an untouched run behaves as before.

## Flags

| flag | meaning | default |
|---|---|---|
| `--op-complexity "pow=2,sin=1.5,if=3"` | per-operator costs; unlisted operators cost `1` | `1` each |
| `--complexity-weights "var=1,const=1"` | terminal weights; keys `var`/`variable`/`input` and `const`/`constant`, plus `depth` | `1` and `1` |
| `--depth-weight 0.5` | cost per nesting level (same as `depth=0.5` in the weights spec) | `0` |
| `--max-complexity K` | reject candidates whose **configured** complexity exceeds `K` | `30` |

Operator names use the same vocabulary as `--ops` (`rcsr schema` lists them):
`pow`, `sin`, `add`, `if`, `logistic`, and so on. Weights must be positive and
finite; the depth weight may also be zero. A malformed pair, an unknown operator
or key, or a non-positive weight is a usage error (exit code `2`) and the
message names the token that was wrong.

The configured value is what the search prunes on, what the Pareto front is
sorted and dominated on, and what every solution reports as `complexity=`
(text) or `"complexity"` (JSON/NDJSON). `--eval-expr` uses the same model.

## Worked examples

Default, node count:

```sh
rcsr --data examples/line.csv --target y --vars x
#  5  complexity=5   ...  y = ((2 * x) + 1)
```

Make powers expensive, so a product is preferred to a power of the same size:

```sh
rcsr --data examples/sine.csv --target y --vars x \
     --op-complexity "pow=6,sin=1.5" --max-complexity 20
```

Make variables cheaper than fitted constants (constants are still fitted; they
just cost more on the front):

```sh
rcsr --data data.csv --target y --vars x1,x2 \
     --complexity-weights "var=1,const=3"
```

Prefer flat expressions: two trees with the same node count now score
differently when one nests more deeply.

```sh
rcsr --data data.csv --target y --vars x1,x2 --depth-weight 0.5
```

For example, `(x*x) * (x*x)` and `x * (x * (x*x))` both have seven nodes and
cost `7` by default. At `--depth-weight 0.5` the balanced tree has depth `2`
and costs `8`, while the chain has depth `3` and costs `8.5`.

## Cost

Complexity is computed for every candidate in every generation, so the model is
a single allocation-free recursive walk; no strings are formatted in the hot
path. The per-operator table is a fixed array indexed by the operator
discriminant, and terminal weights are two `f64`s.

## Tests

`my-version/tests/complexity.rs` covers the contract:

* the default model equals `expr::complexity` (node count) bit for bit on a
  table of trees including `if`, `pow`, `neg`, inputs and constants;
* `--op-complexity "pow=10"` makes a `pow` tree cost more than a same-node
  expression and changes the front order;
* `var`/`const` weights price terminals as specified;
* two seven-node trees with different depths score differently under
  `--depth-weight`;
* malformed specs exit `2` and name the bad token;
* a search with custom costs and weights still recovers `y = 2x + 1`.
