# Dimensional-consistency constraints

A dimensionally inconsistent equation is almost certainly wrong: `distance =
time` cannot be a law, however well it fits one noisy table. `--units` declares
the physical dimensions of the inputs and the target, and
`--require-dimensions` rejects a candidate whose inferred root dimension does
not match the target's. The feature is **opt-in**: with no `--units`, the search
is bit-for-bit the same as before.

```sh
rcsr --data orbit.csv --target period --vars radius,mass \
     --units 'radius=m,mass=kg,period=s' --require-dimensions
```

## The algebra

A unit is a vector of integer exponents over seven SI base dimensions:

```
m  s  kg  A  K  mol  cd
```

`*` adds the vectors, `/` subtracts them and `^n` multiplies every exponent by
the integer `n`. Parentheses group, and `1` is the empty (dimensionless) unit.

| operator | rule |
|---|---|
| `c` (constant) | dimensionless |
| `x_i` (input) | the declared unit of input `i`; **undeclared means dimensionless** |
| `neg`, `abs` | same dimension as the argument |
| `sqrt` | argument's exponents halved; every exponent must be even |
| `square`, `cube` | argument raised to 2 or 3 |
| `add`, `sub`, `min`, `max`, `mod` | both arguments must have **equal** dimensions; result has that dimension |
| `mul` | product (exponents added) |
| `div` | quotient (exponents subtracted) |
| `pow(a, b)` | `b` must be dimensionless; `a` is dimensionless, or has a **constant integer** exponent, in which case the result is `a^b` |
| `exp`, `ln`, `sin`, `cos`, `tan`, `sinh`, `cosh`, `tanh`, `logistic`, `gauss`, `erf`, `erfc`, `asinh`, `acosh`, `atanh` | argument must be dimensionless; result dimensionless |
| `floor`, `ceil`, `round`, `sign`, `step` | argument must be dimensionless; result dimensionless |
| `if(c, a, b)` | `c` must be dimensionless; `a` and `b` must have equal dimensions; result has theirs |
| `==`, `<`, `<=`, `>`, `>=` | operands must have equal dimensions; result is dimensionless (0/1) |
| `and`, `or`, `xor`, `not` | arguments dimensionless; result dimensionless |
| `require`, `constrain`, `between` | arguments dimensionless; result dimensionless |
| `atan2(y, x)` | `y` and `x` must have equal dimensions; result dimensionless |
| `complexity`, `variables`, `constants`, `contains`, `contains_op` | result dimensionless; the argument is structural, not dimension-checked |

The inference is one bottom-up pass over the tree. It returns the root unit or
the first inconsistency, naming the operator and the clashing units.

## Undeclared names are dimensionless

Only names listed in `--units` get a dimension. Every other input, and the
target unless it is named, has the empty unit `1`. This is why the flag is
opt-in: a run without a complete declaration simply keeps the old behaviour.

```sh
# x is a length, t is a time, and y is a velocity.
--units 'x=m,t=s,y=m/s'
```

If the target's unit is not declared, a candidate must be **dimensionless** to
pass `--require-dimensions`. Declare the target explicitly in that case. A
delayed copy (`x_lag1` from `--lags`) inherits the unit declared for `x`; you may
also declare the lagged name directly.

## `--require-dimensions`

Without it, `--units` only prints a summary note and the search is unchanged.
With it, a candidate whose root dimension differs from the target's gets an
**infinite error**:

* it is selected against (the error is not a soft penalty),
* it never enters the Pareto archive (the archive ignores non-finite errors),
* it cannot be resurrected by re-evaluation, and
* a seeded expression that is dimensionally wrong is reported with infinite
  training error instead of being used as an elite.

We chose a **summary note** rather than a rejected-candidate counter when
`--units` is given without `--require-dimensions`: the counter would have to be
threaded through every island for information that does not change the result.
The note says that units are not filtering and how to turn the filter on.

The declaration and the flag are part of the checkpoint's search settings, so a
save/resume keeps the same filter.

## Worked example

Data (`orbit.csv`):

```csv
radius,period
1,1
2,2.828427
3,5.196152
```

Kepler's third law says `period^2 ∝ radius^3`, so `period` has dimension
`m^(3/2)`, which is not one of the available integer forms unless we declare a
different target. A cleaner example is a speed:

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
exactly. A seed `distance` is numerically wrong here anyway, but even with data
where `distance == speed` (a rescaling of the units), it would be rejected: the
root is `m`, the target is `m/s`.

At the library level the same check is:

```rust
use rcsr::units::{Unit, Units};
use rcsr::expr::{input, node, Op};

let units = Units::new(
    vec![Unit::parse("m").unwrap(), Unit::parse("s").unwrap()],
    Unit::parse("m/s").unwrap(),
);
let speed = node(Op::Div, vec![input(0), input(1)]);
assert!(units.accepts(&speed));
assert!(!units.accepts(&input(0)));
```

## Honest limitations

* **It is a hard filter.** A law that is correct in a system of units you did
  not declare is rejected. There is no "unknown dimension" state: undeclared
  means dimensionless, not "anything". Declare the target and every dimensional
  input, or leave `--require-dimensions` off.
* **The search runs on normalised inputs.** The unit pass is structural: it
  uses the declared dimensions of the input leaves, not the centring/scaling
  constants that the data preparation applies. That is exactly what makes it
  useful (the printed law keeps those dimensions), but it does not check the
  numerical constants produced by fitting.
* **Integer exponents only.** `sqrt(m)` is rejected because the exponent 1/2 is
  not representable; only units whose exponents are all even can be rooted.
* **No derived-unit names.** Write `kg*m/s^2` instead of `N`; there is no
  dimensional equivalence beyond the seven base vectors.
* **The filter applies to the search**, not to `--eval-expr`, which still
  scores any formula you give it.
* **One-hot categorical names containing `=` cannot be declared** because the
  flag itself separates a name from its unit with `=`.
