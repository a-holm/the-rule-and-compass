# Dimensional-consistency constraints

A dimensionally inconsistent equation is almost certainly wrong: `distance =
time` cannot be a law, however well it fits one noisy table. `--units` declares
the dimensions of the inputs and the target, and `--require-dimensions` rejects
a candidate whose dimensions cannot be reconciled with the target's. The
feature is **opt-in**: with no `--units`, the search is bit-for-bit the same as
before.

```sh
rcsr --data orbit.csv --target period --vars radius,mass \
     --units 'radius=m,mass=kg,period=s^2/m' --require-dimensions
```

## Dimensions, not units

The model is a vector of integer exponents over **named base dimensions**.
There is deliberately no unit table and no scale factor. Only dimensions
matter:

* a litre and a cubic metre are the same dimension (`m^3`); the fitted constant
  absorbs the factor `1e-3`;
* `degC` and `K` are the same dimension (`K`); the fitted intercept absorbs the
  affine offset, not the dimension pass;
* `USD`, `shares`, `people` and `widgets` are just as valid as SI names.

Declare **dimensions**, not unit symbols: write `volume=m^3`, not
`volume=litre`, and `temp=K`, not `temp=degC`. An identifier that is not one of
the seven conventional SI names is a new base dimension (see below), so `litre`
on its own would declare a *different* dimension from `m^3`; that is how you
would model a genuinely separate counting unit, not a scale conversion.

## The algebra

A dimension is a sorted map from dimension name to integer exponent. The seven
conventional SI names are `m`, `s`, `kg`, `A`, `K`, `mol` and `cd`; **any other
identifier in a unit expression becomes its own base dimension**:

```sh
--units 'price=USD,volume=shares'   # catches price + volume
```

The parser accepts identifiers, `*`, `/`, integer powers with `^`, parentheses
and the empty dimension `1`. Only syntax and non-integer exponents are errors;
an unknown name is not an error.

| operator | rule |
|---|---|
| `x_i` (input) | the declared dimension of input `i`; **undeclared means dimensionless** |
| `c` (constant) | a free dimension solved for by the filter (see below) |
| `neg`, `abs` | same dimension as the argument |
| `sqrt` | introduces the relation `child = result^2`; the solve needs an integer result dimension (so `sqrt(c*x)` can work by giving `c` an area dimension, while `sqrt(x)` alone cannot be a length) |
| `square`, `cube` | argument raised to 2 or 3 |
| `add`, `sub`, `min`, `max`, `mod` | both sides must have **equal** dimensions |
| `mul` | product (exponents added) |
| `div` | quotient (exponents subtracted) |
| `pow(a, b)` | `b` must be dimensionless; `a` is dimensionless or a constant integer power of a dimensioned base |
| `exp`, `ln`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `logistic`, `gauss`, `erf`, `erfc`, `asinh`, `acosh`, `atanh` | argument dimensionless; result dimensionless |
| `floor`, `ceil`, `round`, `sign`, `step` | argument dimensionless; result dimensionless |
| `if(c, a, b)` | `c` dimensionless; `a` and `b` equal; result has their dimension |
| `==`, `<`, `<=`, `>`, `>=` | operands equal; result dimensionless |
| `and`, `or`, `xor`, `not` | arguments dimensionless; result dimensionless |
| `require`, `constrain`, `between` | arguments dimensionless; result dimensionless |
| `atan2(y, x)` | `y` and `x` equal; result dimensionless |
| `complexity`, `variables`, `constants`, `contains`, `contains_op` | result dimensionless; the argument is structural |

## Fitted constants absorb units

A constant is not forced to be dimensionless. The filter treats each constant
leaf as an unknown integer dimension vector and solves the resulting linear
constraints. This is the only way a coefficient can absorb a unit ratio and an
intercept can absorb the target's dimension:

```
y = 2*x + 1        x in m, y in m
                     coefficient 2: dimensionless
                     intercept   1: m
```

A strict "every constant is dimensionless" rule would reject that perfectly
ordinary law. The bottom-up `infer` function still exposes the strict algebra
(constants dimensionless); `Units::accepts`, the search predicate, uses the
solve. When there is no integer solution — for example `c*x + d` where `x` is
`m` and the target is `m/s` with no term that can supply the missing `s` — the
candidate is rejected.

## Reporting constant dimensions

When `--units` is active, each reported solution gets a plausibility note:

```text
# constant dimensions (rank 1): 1:m, 2:1 (value:dimension, tree order)
```

`--json`/`--ndjson` carry the same information as a `constant_dimensions` array
of `{value, dimension}` objects. The order is pre-order tree order (the order
the constants are fitted), which can differ from the printed left-to-right
order because simplification may reorder commutative operands; the value is
shown to keep the pairing unambiguous. `?` means the solve did not determine an
integer dimension, which normally means the tree was rejected.

## `--require-dimensions`

Without it, `--units` only prints a summary note and the search is unchanged.
With it, a candidate with no consistent integer assignment gets an **infinite
error**:

* it is selected against (the error is not a soft penalty),
* it never enters the Pareto archive (the archive ignores non-finite errors),
* it cannot be resurrected by re-evaluation, and
* a seeded expression that is inconsistent is reported with infinite training
  error instead of being used as an elite.

We chose a **summary note** rather than a rejected-candidate counter when
`--units` is given without `--require-dimensions`: a counter would have to be
threaded through every island for information that does not change the result.
The note says that units are not filtering and how to turn the filter on.

The declaration and the flag are part of the checkpoint's search settings, so a
save/resume keeps the same filter.

## Worked example

`speed.csv`:

```csv
distance,time,speed
10,2,5
30,3,10
40,4,10
```

```sh
rcsr --data speed.csv --target speed --vars distance,time \
     --units 'distance=m,time=s,speed=m/s' --require-dimensions \
     --seed-expr 'distance/time' --generations 20
```

The seed `distance/time` has root dimension `m/s`; it is accepted and fits
exactly. A seed `distance` is numerically wrong here anyway, but even on data
where `distance == speed` it is rejected: the root is `m`, the target is `m/s`,
and no free constant can change an additive mismatch. A law with an intercept,
`a*distance/time + b`, is accepted, with `a` dimensionless and `b` in `m/s`.

At the library level:

```rust
use rcsr::units::{constant_dimensions, Unit, Units};
use rcsr::expr::{constant, input, node, Op};

let units = Units::new(
    vec![Unit::parse("m").unwrap(), Unit::parse("s").unwrap()],
    Unit::parse("m/s").unwrap(),
);
let law = node(Op::Div, vec![input(0), input(1)]);
assert!(units.accepts(&law));
assert!(!units.accepts(&input(0)));

let with_intercept = node(
    Op::Add,
    vec![law, constant(1.0)],
);
assert!(units.accepts(&with_intercept));
assert_eq!(
    constant_dimensions(&with_intercept, &units.vars, &units.target),
    vec![Some(Unit::parse("m/s").unwrap())]
);
```

## Radians, degrees and nonlinear units

The honest exception: **radians and degrees are both dimensionless**, so no
dimensional check can separate them. `sin(x)` expects radians, and
`sin(x_degrees)` is not fixed by a constant because the scale sits inside a
nonlinear function. Convert the column to radians before the run. Leave the
angle undeclared (or `1`); writing `angle=rad` and `angle=deg` would only create
two arbitrary base dimensions, not a physical distinction. Do not expect the
algebra to catch this.

The same applies to logarithmic or nonlinear units such as dB and pH. They are
not modelled. Either transform the column to its linear physical quantity first
(for example amplitude or concentration), or declare the reading as its own
base dimension (`level=dB`). A separate base dimension prevents accidental
addition to a linear quantity, but it does not perform the logarithmic
conversion.

## Undeclared names and compatibility

Only names listed in `--units` get a dimension. Every other input, and the
target unless it is named, has the empty dimension `1`. This is why the flag is
opt-in: a run without a complete declaration keeps the old behaviour. A delayed
copy (`x_lag1` from `--lags`) inherits the dimension declared for `x`; you may
also declare the lagged name directly.

Without `--units`, `Config::accepts_dimensions` always returns true and the
search is unchanged bit for bit (a regression test compares archive errors,
costs and structural hashes).

## Honest limitations

* **It is a hard filter.** A law that is correct in a system of dimensions you
  did not declare is rejected. There is no "unknown dimension" state:
  undeclared means dimensionless, not "anything". Declare the target and every
  dimensional input, or leave `--require-dimensions` off.
* **The search runs on normalised inputs.** The pass is structural: it uses the
  declared dimensions of the input leaves, not the centring/scaling constants
  applied by data preparation. The printed law's constants are what the report
  describes.
* **Integer exponents only.** `sqrt(m)` with no free constant is rejected
  because the exponent 1/2 is not representable; the solve can still accept
  `sqrt(c*x)` by choosing `c` so the argument is a square (for example an
  area).
* **No unit table.** Write `kg*m/s^2` rather than `N`, and `m^3` rather than
  `litre`; an unknown identifier is a new base dimension, not an alias.
* **The filter applies to the search**, not to `--eval-expr`, which still
  scores any formula you give it.
* **One-hot categorical names containing `=` cannot be declared** because the
  flag separates a name from its dimension with `=`.
