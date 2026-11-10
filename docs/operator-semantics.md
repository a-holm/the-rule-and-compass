# Operator semantics

`expr::eval` (the exact f64 scorer) and the fit's flat evaluator (`fit.rs`) give every operator the same value at the same operands: the same finiteness, and the same bits where the value is finite. `ln` and `pow` are one shared function in `expr.rs` (`expr::ln`, `expr::pow`). Property test `evaluators_agree_on_finiteness_for_random_trees` and unit tests `pow_semantics_match_the_documented_table` and `ln_semantics_match_the_documented_table` hold both evaluators to this table.

A non-finite value anywhere in a tree makes the candidate infeasible. NaN and infinities propagate through the arithmetic operators by IEEE rules. Comparison, logical, `min`, `max` and `atan2` return NaN when an operand is NaN.

| Operator | Domain | Result outside the domain |
|---|---|---|
| `ln(a)` | `a > 0` | NaN for `a <= 0` (`0` and `-0` included) and for NaN. `ln(+inf)` is `+inf`. |
| `pow(a, b)` | `a >= 0`, or `a < 0` with `b` within `1e-9` of an integer | NaN for `a < 0` with `b` further than `1e-9` from an integer (`-inf` base included). Otherwise IEEE `powf`: `pow(x, 0)` and `pow(1, y)` are 1 even for NaN, `pow(0, 0)` is 1, `pow(0, y < 0)` is `+inf` (`-inf` for a `-0` base and odd integer `y`), overflow gives `+-inf`, never NaN. For a negative base the `1e-9` window passes a non-integer `b` to `powf`, which returns NaN for a finite base. |
| `exp(a)` | `a <= 700` | `+inf` for `a > 700`. |
| `a / b` | `abs(b) >= 1e-12` | NaN. |
| `a mod b` | `a`, `b` finite, `abs(b) >= 1e-12` | NaN. Floored remainder. |
| `sqrt(a)` | `a >= 0` | NaN. |
| `tan(a)` | finite result | NaN for a non-finite result. |
| `acosh(a)` | `a >= 1` | NaN. |
| `atanh(a)` | `abs(a) < 1` | NaN. |
| `sign(a)` | not NaN | NaN. |
| `tanh`, `sinh`, `cosh`, `sin`, `cos`, `asinh`, `erf`, `erfc`, `logistic`, `gauss`, `abs`, `neg`, `square`, `cube`, `floor`, `ceil`, `round` | all reals | IEEE: overflow gives `+-inf`, NaN in gives NaN out. |
| `step(a)`, `not(a)` | all reals | `step` is 1 for `a > 0`, else 0. `not` is 1 for `a <= 0`, 0 for `a > 0`, NaN for NaN. |
| `require(a)`, `constrain(a)` | `a > 0` | NaN. |
| `between(v, lo, hi)` | `lo <= v <= hi` | NaN. |
| `if(c, a, b)` | all | `a` for `c > 0`, else `b`; NaN in the untaken branch is ignored. |
| `min`, `max`, `atan2`, `equal`, `less`, `less_or_equal`, `greater`, `greater_or_equal`, `and`, `or`, `xor` | all | NaN when an operand is NaN. |

The device evaluator works in f32 and may differ in the last bits and in overflow range; its `ln` and `pow` use the same domain rules as this table. The status routing of the device fit (`docs/gpu-backend.md`) covers the f32 cases.
