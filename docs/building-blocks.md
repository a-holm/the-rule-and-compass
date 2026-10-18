# User-defined building blocks (`--function`)

`--function` supplies a sub-formula the search may reuse as an operator of its
own. It is useful when a domain already gives you a trusted piece — a seasonal
term, a known response curve, a physical law — and you want the engine to fit
the *rest* of the model around it instead of rediscovering the piece from
scratch.

```sh
rcsr --data data.csv --target y --vars x1,x2 \
  --function "wave(a,b)=sin(a)*cos(b)" \
  --generations 200 --metric squared
```

The search can then produce, and the report can print, a solution such as

```text
y = wave(x1, x2)
```

## Syntax

Up to **four** `--function` definitions may be given, in command-line order.
Each is one of:

```text
name(a, b) = <expression>      explicit parameter list
name       = <expression>      parameters inferred from free variables
```

* `name` must be a valid identifier and must not shadow a primitive operator
  (`sin`, `add`, …) or a canonical slot name (`custom0` … `custom3`).
* Parameters are 0 to 4 names. For `name = <expr>` they are the free variables
  in **order of first appearance**; a name the parser already understands
  (`x0`, `pi`, a primitive function, an earlier block) is not a parameter.
* The body is parsed by the same expression parser as `--eval-expr` and
  `--seed-expr`, against the parameter names. A body may call any block defined
  **before** it; forward references and self-recursion are rejected.
* Definitions are numbered by position: the first is slot `custom0`, then
  `custom1`, and so on. Both the user's name and the canonical `customN` name
  are accepted wherever an expression is parsed (`--eval-expr`, seeds, and the
  printed form itself).

### Examples

```text
--function "f(a,b)=sin(a)*cos(b)"    # two parameters, slot custom0
--function "g=2*x1 + x2"             # parameters inferred: x1, x2
--function "k=2.5"                   # a named constant (arity 0)
--function "f(a)=sin(a)" \
--function "g(a,b)=f(a)*b"           # g may use the earlier f
```

A block definition is a **usage** error (exit 2) when it does not parse, has
more than four parameters, repeats a name, shadows a primitive, or refers to a
block that is not defined yet.

## Evaluation and printing

A call is an expression node of the block's arity. Its arguments are ordinary
expressions in the search's variable space; the body is kept in *parameter*
space. Evaluating `f(expr1, expr2)` evaluates the two arguments and then the
body with `a = expr1`, `b = expr2`.

Printed output names the block, so a solution reads as the model you supplied:

```text
y = wave(x1, x2)
y = 2.5 * wave(x1, x2) + 0.3
```

A printed expression parses back when the same `--function` flags are supplied.
Without them, the block name is an unknown function (and `customN` is reported
as a slot with no definition).

## Constants inside a block body are shared

This is the point of a building block: if its body contains a numeric constant,
that constant is a **single parameter shared by every use of the block in a
candidate**, not a fresh parameter per call.

For example, with `f(a)=2*a` and a candidate `f(x) + f(x)`, the data
`y = 2 * (3.25 x)` is fitted by moving the one shared `2` to `3.25`; both calls
then evaluate with `3.25`. The engine does this deliberately:

1. For the constant fit, each call is inlined in parameter space and the body's
   constants are mapped to **one** Levenberg–Marquardt slot per
   `(block slot, constant ordinal)`.
2. After the fit, the shared value is written back into every copy of that
   block body in the candidate, so later evaluation and printing agree.

As a consequence `count_constants` — the parameter count used by AIC, AICc and
BIC — counts a body's constants **once per distinct block**, no matter how many
calls appear. Argument constants are counted normally because they are part of
the call site, not the definition.

Blocks used inside other blocks follow the same rule per slot.

## Complexity

A call costs the block's operator cost plus the cost of its arguments. The
body's internal nodes are **not** charged again: the configured cost is the
price of treating the whole building block as one primitive. The default is `1`.

```sh
--function "wave(a,b)=sin(a)*cos(b)" --op-complexity "wave=2"
# or by canonical slot:
--function "wave(a,b)=sin(a)*cos(b)" --op-complexity "custom0=2"
```

The Pareto front, `--max-complexity` and the reported `complexity=` value all
use this configured cost. See [`complexity.md`](complexity.md) for the rest of
the formula.

## Search behaviour

Supplying a block adds its slot to the operator set for that run, even if
`--ops` was given explicitly. Mutation can therefore introduce a call, remove
one, or swap a call for a primitive of the same arity. A run without
`--function` is unchanged: the block registry is empty, no custom operator is
added, and results are bit-for-bit the same as before the feature existed.

Variables inside a body are parameters only. Data columns are written through
the call's arguments in the usual `--vars` space, so the normal denormalising of
inputs happens to the arguments; the body itself is never rewritten.

## Limits and interactions

* At most four blocks, each with at most four parameters.
* A block body may use constants, inputs/parameters, primitives and earlier
  blocks. It may not recurse or refer forward.
* `--function` cannot be combined with `--save` or `--resume`: the checkpoint
  format does not store block definitions, and silently resuming with different
  definitions would change the meaning of saved expressions.
* Blocks are not listed in `rcsr schema` under `"operators"` because their arity
  is defined at run time; `--function` is in the option table and `--help`.
* `--eval-expr` and `--seed-expr`/`--seed-file` accept block calls whenever the
  same `--function` flags are present.

## Tests

`my-version/tests/blocks.rs` covers definition parsing and parameter inference,
evaluation, round-tripping, arity/recursion/duplicate rejection, per-block
complexity, shared-constant fitting (including `count_constants`), an
end-to-end recovery of `sin(x1)*cos(x2)` through a supplied block, and the
no-`--function` baseline.
