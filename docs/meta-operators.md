# Meta operators: expressions that describe themselves

A **meta operator** inspects the *shape* of the expression below it and returns
a number describing that shape. It is the one family of primitives whose value
is not a function of the input row: two rows of data always see the same value
from the same tree.

They are **opt-in**. They are accepted by `--ops` and listed by `rcsr schema`
under `"operators"`, but they are not in the default operator set.

## Semantics

| op | returns |
|---|---|
| `complexity(e)` | the node count of `e` |
| `variables(e)` | the number of **distinct** input variables appearing in `e` |
| `constants(e)` | the number of constant leaves in `e` |
| `contains(e)` | `1` if `e` contains at least one input variable, else `0` |
| `contains_op(e)` | `1` if `e` contains at least one operator node (i.e. is not a bare leaf), else `0` |

All five are unary. `complexity` counts every node, including a nested meta
node and its argument; `variables(x1 * x1)` is `1`; `constants(3)` is `1` and
`constants` never counts its own node. `contains_op(x1)` is `0` because `x1` is
a bare leaf, while `contains_op(sin(x1))` is `1`.

The operators compose with the rest of the language, so they can appear in
arithmetic and conditionals:

```text
complexity(x1 * x2)                 -> 3
variables(x1 * x2 + x3)             -> 3
constants(2 * x1 + 1)               -> 1
if(contains(x1), complexity(x1), 0)
```

The printed form parses back to the same tree (`expr::to_string` and
`expr::parse` share the naming table).

## Why they are structural

`expr::eval` is the innermost loop of the engine. Walking a meta argument on
every row would make a candidate cost depend on a subtree it does not use, and
would produce different values on different rows for what is really a property
of the model.

Instead the value is computed **once**, when the node is constructed:

* `expr::meta(kind, child)` computes the value and stores it in `Node::val`;
* `expr::node(op, kids)` does the same for any meta operator, so the parser,
  `random_tree`, mutation and crossover cannot build an inconsistent node;
* `expr::eval` returns the cached `val` without looking at the child (a debug
  assertion checks that the cache is finite);
* `simplify` recomputes the cache after simplifying the argument, because that
  can change its shape;
* `mutate` and `crossover` refresh the cache bottom-up after a structural edit;
* `denormalize` leaves a meta argument untouched, so the printed value still
  describes the searched tree and agrees with the fitted model.

The cache cannot go stale when the constant fitter moves a constant: the value
depends on the argument's structure (and, for `variables`, its input indices),
not on a constant's numeric value. `count_constants` therefore counts only real
`Const` leaves, never the meta node.

The Levenberg–Marquardt fitter gives a meta node **no constant slot and a zero
gradient**, exactly like `floor`/`step`: the operator cannot be moved, and no
gradient flows into its argument. A constant inside the argument still exists,
but it is not identifiable from the model output.

## Caveat: they describe the model, not the data

Because a meta operator returns a single number per tree, it is a **constant**
with respect to the data. A model such as `complexity(x1 * x2)` predicts `3`
for every row. That can be useful for constraints, reporting and
self-describing models — for example `if(complexity(x1 * x2) > 5, ..., ...)`
selects a branch based on the model's own form — but it cannot supply
row-level signal that a plain constant could not.

A target built directly from a meta value is therefore a constant column, and
the parsimony pressure will normally prefer a one-node constant to the larger
meta expression. The recovery test in `my-version/tests/meta.rs` says this
plainly: it verifies the wiring end to end, not a predictive advantage.

## Enabling them

```sh
rcsr --data data.csv --target y --ops add,mul,sin,complexity,variables
```

`schema` lists the canonical names (`complexity`, `variables`, `constants`,
`contains`, `contains_op`); an unknown name in `--ops` is a usage error
(exit 2).
